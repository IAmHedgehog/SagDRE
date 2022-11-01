from html import entities
import torch
import torch.nn as nn
from collections import defaultdict
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
from utils import GCNLayer, AttnLayer, filter_g


class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss(config.num_class)

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.activation = nn.ReLU()
        self.start_dim = self.emb_size
        self.gcn_dim = config.gcn_dim
        self.start_gcn = GCNLayer(self.start_dim, self.gcn_dim)
        self.GCNs = nn.ModuleList([
            GCNLayer(
                self.gcn_dim, self.gcn_dim, activation=self.activation)
            for _ in range(config.gcn_layers)])

        self.Attns = nn.ModuleList([
            AttnLayer(
                self.gcn_dim, activation=self.activation)
            for _ in range(config.gcn_layers)
        ])
        self.bank_size = self.start_dim + self.gcn_dim * (self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(config.dropout)
        self.rnn = nn.LSTM(
            self.bank_size, self.bank_size, 2,
            bidirectional=False, batch_first=True)

        self.path_attn = nn.MultiheadAttention(self.bank_size, 4)
        self.final_fea_dim = self.bank_size * 5
        self.predict = nn.Sequential(
            nn.Linear(self.final_fea_dim, self.final_fea_dim),
            self.activation,
            self.dropout,
        )
        self.out_linear = nn.Linear(self.final_fea_dim, config.num_labels)

    def encode(self, input_ids, attention_mask):
        # output = self.model(
        #     input_ids=input_ids[:, :512],
        #     attention_mask=attention_mask[:, :512],
        #     output_attentions=True,
        # )
        # sequence_output = output[0]
        # attention = output[-1][-1]

        config = self.config
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)

        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, sub2words):

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            # i is the batch
            encoder_output = sequence_output[i]
            sub2word = sub2words[i]
            sub_word_num = min(sub2word.shape[1], encoder_output.shape[0])
            entity_embs = torch.mm(sub2word[:, :sub_word_num], encoder_output[:sub_word_num])
            entity_embs = entity_embs[-len(entity_pos[i]):]

            entity_atts = []
            for e_idx, e in enumerate(entity_pos[i]):
                # e is the entity in each batch i
                e_emb, e_att = [], []
                for start, end in e:
                    for sw_idx in range(start, end):
                        if sw_idx + offset < c:
                            e_emb.append(sequence_output[i, sw_idx + offset])
                            e_att.append(attention[i, :, sw_idx + offset])
                if len(e_emb) > 0:
                    e_emb = torch.stack(e_emb, dim=0).mean(0)
                    e_att = torch.stack(e_att, dim=0).mean(0)
                else:
                    e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(h, c).to(attention)
                entity_atts.append(e_att)

            # [n_e, h, seq_len]
            entity_atts = torch.stack(entity_atts, dim=0)

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward2(self,
                 input_ids=None,
                 attention_mask=None,
                 labels=None,
                 entity_pos=None,
                 hts=None,
                 instance_mask=None, **kw,
                 ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts, kw['sub2words'])

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                graphs=None, sub2words=None, paths=None,
                ):

        sequence_output, _ = self.encode(input_ids, attention_mask)
        features = self.get_graph_encoded_features(sequence_output, graphs, sub2words)
        head_entity_feas, tail_entity_feas, path_feas = self.get_entity_path_info(entity_pos, features, paths, hts)
        logits = self.get_logits(head_entity_feas, tail_entity_feas, path_feas)
        preds = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            return loss, preds
        return preds

    def get_graph_encoded_features(self, encoder_outputs, graphs, sub2words):
        features = []
        for i, graph in enumerate(graphs):
            encoder_output = encoder_outputs[i]
            sub2word = sub2words[i]
            sub_word_num = min(sub2word.shape[1], encoder_output.shape[0])
            x = torch.mm(sub2word[:, :sub_word_num], encoder_output[:sub_word_num])
            graph = filter_g(graph, x)
            xs = [x]
            x = self.start_gcn(graph, x)
            xs.append(x)
            for GCN, Attn in zip(self.GCNs, self.Attns):
                x1 = GCN(graph, x)
                x2 = Attn(x, x1, x1)
                x = x1 + x2
                xs.append(x)
            out_feas = torch.cat(xs, dim=-1)
            features.append(out_feas)
        return features

    def get_entity_path_info(self, entity_pos, features, paths, hts):
        path_len_dict = defaultdict(list)
        entity_bank = []
        head_entity_feas = []
        tail_entity_feas = []
        for batch_idx in range(len(entity_pos)):
            cur_head_ids = []
            cur_tail_ids = []
            cur_head_feas = []
            cur_tail_feas = []
            num_entity = len(entity_pos[batch_idx])
            entity_feas = features[batch_idx][-num_entity:]
            entity_bank.append(entity_feas)
            path_t = paths[batch_idx]
            for pair_idx, (h_ent, t_ent) in enumerate(hts[batch_idx]):
                cur_head_ids.append(h_ent)
                cur_tail_ids.append(t_ent)
                cur_head_feas.append(entity_feas[h_ent])
                cur_tail_feas.append(entity_feas[t_ent])
                cur_paths = path_t[(h_ent, t_ent)]
                for path in cur_paths:
                    path = torch.LongTensor(path).cuda()
                    cur_h = torch.index_select(features[batch_idx], 0, path)
                    path_len_dict[len(path)].append((batch_idx, pair_idx, cur_h))

            # entity_embs = features[batch_idx][-len(entity_pos[batch_idx]):]
            # ht_i = torch.LongTensor(hts[batch_idx]).cuda()
            # cur_head_feas = torch.index_select(entity_embs, 0, ht_i[:, 0])
            # cur_tail_feas = torch.index_select(entity_embs, 0, ht_i[:, 1])
            cur_head_feas = torch.stack(cur_head_feas)
            cur_tail_feas = torch.stack(cur_tail_feas)
            head_entity_feas.append(cur_head_feas)
            tail_entity_feas.append(cur_tail_feas)

        path_embedding = {}

        for items in path_len_dict.values():
            cur_hs = torch.stack([h for _, _, h in items], 0)
            cur_hs2, _ = self.rnn(cur_hs)
            cur_hs = cur_hs2.max(1)[0]
            for idx, (i, j, _) in enumerate(items):
                if (i, j) not in path_embedding:
                    path_embedding[(i, j)] = []
                path_embedding[(i, j)].append(cur_hs[idx])

        path_info = [[] for _ in range(len(entity_pos))]

        for (i, j), emb in path_embedding.items():
            if len(emb) > 1:
                query = head_entity_feas[i][j] - tail_entity_feas[i][j]
                keys = torch.stack(emb).unsqueeze(1)
                output, _ = self.path_attn(query.unsqueeze(0).unsqueeze(0), keys, keys)
                path_info[i].append((output.squeeze(0).squeeze(0), j))
            else:
                path_info[i].append((emb[0], j))

        path_feas = []
        for i in range(len(path_info)):
            path_info[i].sort(key=lambda x: x[1])
            path_feas.append([p_fea[0] for p_fea in path_info[i]])

        return head_entity_feas, tail_entity_feas, path_feas

    def get_logits(self, head_entity_feas, tail_entity_feas, path_feas):
        all_path_feas = []
        for p_feas in path_feas:
            all_path_feas.extend(p_feas)
        head_feas = torch.cat(head_entity_feas, dim=0)
        tail_feas = torch.cat(tail_entity_feas, dim=0)
        path_feas = torch.stack(all_path_feas, dim=0)

        out_feas = torch.cat([
            head_feas, tail_feas,
            torch.abs(head_feas - tail_feas),
            torch.mul(head_feas, tail_feas),
            path_feas,
        ], dim=-1)

        out_feas = self.predict(out_feas)
        m_preds = self.out_linear(out_feas)
        return m_preds