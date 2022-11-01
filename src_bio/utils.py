import torch
import random
import numpy as np
from torch import nn


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    graphs = [f['graph'] for f in batch]
    sub2words = [torch.tensor(f['sub2word'], dtype=torch.float) for f in batch]
    paths = [f['path'] for f in batch]

    hts = [f["pos_hts"] + f['neg_hts'] for f in batch]
    labels = [f["pos_rels"] + f['neg_rels'] for f in batch]

    sampled_hts = []
    sampled_labels = []
    neg_ratio = 4
    for cur_b in batch:
        cur_sampled_hts = []
        cur_sampled_labels = []
        pos_relations = cur_b['pos_rels']
        neg_relations = cur_b['neg_rels']
        pos_hts = cur_b['pos_hts']
        neg_hts = cur_b['neg_hts']
        # print('-------->', len(pos_hts), len(neg_hts))
        for pos_rel, pos_ht in zip(pos_relations, pos_hts):
            cur_sampled_labels.append(pos_rel)
            cur_sampled_hts.append(pos_ht)

        neg_cnt = min(len(neg_hts), max(len(pos_hts) * neg_ratio, 4))
        neg_rels = random.sample(list(zip(neg_relations, neg_hts)), neg_cnt)
        for neg_rel, neg_ht in neg_rels:
            cur_sampled_labels.append(neg_rel)
            cur_sampled_hts.append(neg_ht)
        sampled_hts.append(cur_sampled_hts)
        sampled_labels.append(cur_sampled_labels)

    output = (
        input_ids, input_mask, labels, entity_pos, hts, graphs, sub2words,
        paths, sampled_hts, sampled_labels)
    return output


def filter_g(g, features):
    return norm_g(g.adj().to_dense().cuda())
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sim_A = cos(features.unsqueeze(2), features.t().unsqueeze(0))
    adj = g.adj().to_dense().cuda()
    unorder = ((adj + adj.t()) == 2).float()
    ordered = (adj - unorder) * (sim_A > 0.6)
    A = ordered + unorder
    return norm_g(A)


class AttnLayer(nn.Module):
    def __init__(self, in_feats, activation=None, dropout=0.0):
        super(AttnLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            in_feats, 8, dropout=dropout)
        self.activation = activation
        self.v_proj = nn.Linear(in_feats, in_feats)

    def forward(self, query, key, value):
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = self.v_proj(value)
        value = value.unsqueeze(1)
        out_fea = self.attn(query, key, value, need_weights=False)[0]
        out_fea = out_fea.squeeze(1)
        if self.activation:
            return self.activation(out_fea)
        return out_fea


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, activation=None, dropout=0.0):
        super(GCNLayer, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = activation
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, A, X):
        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        X = self.act(X) if self.act else X
        return X