from collections import defaultdict
import torch
import torch.nn as nn
from utils import GCNLayer, AttnLayer, BiLSTM, filter_g


class GAIN_GLOVE(nn.Module):
    def __init__(self, config, data_config):
        super(GAIN_GLOVE, self).__init__()
        self.config = config
        self.activation = nn.ReLU()
        word_emb_size = config.word_emb_size
        encoder_input_size = word_emb_size + config.entity_id_size + \
            config.entity_type_size

        self.word_emb = nn.Embedding(
            data_config.data_word_vec.shape[0], word_emb_size,
            padding_idx=config.word_pad)
        self.word_emb.weight.data.copy_(
            torch.from_numpy(data_config.data_word_vec[:, :word_emb_size]))
        self.word_emb.weight.requires_grad = True

        self.encoder = BiLSTM(encoder_input_size, config)

        self.entity_type_emb = nn.Embedding(
            config.entity_type_num, config.entity_type_size,
            padding_idx=config.entity_type_pad)
        self.entity_id_emb = nn.Embedding(
            config.max_entity_num + 1, config.entity_id_size,
            padding_idx=config.entity_id_pad)

        self.start_dim = config.lstm_hidden_size * 2
        self.gcn_dim = config.gcn_dim

        self.start_GCN = GCNLayer(
            self.start_dim, self.gcn_dim, activation=self.activation)

        self.GCNs = nn.ModuleList([
            GCNLayer(
                self.gcn_dim, self.gcn_dim, activation=self.activation)
            for i in range(config.gcn_layers)])

        self.Attns = nn.ModuleList([
            AttnLayer(
                self.gcn_dim, activation=self.activation,
                dropout=config.dropout)
            for _ in range(config.gcn_layers)
        ])

        # self.bank_size = self.gcn_dim * (self.config.gcn_layers + 1)
        self.bank_size = self.start_dim + self.gcn_dim * (
            self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)

        self.path_info_mapping = nn.Linear(self.gcn_dim * 4, self.gcn_dim * 4)

        self.rnn = nn.LSTM(
            self.bank_size, self.bank_size, 2,
            bidirectional=False, batch_first=True)

        self.path_attn = nn.MultiheadAttention(self.bank_size, 4)

        self.predict2 = nn.Sequential(
            nn.Linear(self.bank_size*5, self.bank_size*5),
            self.activation,
            self.dropout,
        )

        self.out_linear = nn.Linear(self.bank_size * 5, config.relation_nums)
        self.out_linear_binary = nn.Linear(self.bank_size * 5, 2)

    def forward(self, **params):
        words = params['words'].cuda()
        mask = params['mask'].cuda()
        bsz = words.size(0)

        encoder_outputs = self.word_emb(words)
        encoder_outputs = torch.cat([
            encoder_outputs,
            self.entity_type_emb(params['entity_type']),
            self.entity_id_emb(params['entity_id'])], dim=-1)
        encoder_outputs, _ = self.encoder(
            encoder_outputs, params['src_lengths'])
        encoder_outputs[mask == 0] = 0

        graphs = params['graph2s']
        sub2words = params['sub2words']
        features = []

        for i, graph in enumerate(graphs):
            encoder_output = encoder_outputs[i]
            sub2word = sub2words[i]
            ent_x = torch.mm(sub2word, encoder_output)
            num_nodes = graph.number_of_nodes() - ent_x.size(0)
            x = torch.cat((encoder_output[:num_nodes], ent_x), dim=0)
            graph = filter_g(graph, x)
            xs = [x]
            x = self.start_GCN(graph, x)
            xs.append(x)
            for GCN, Attn in zip(self.GCNs, self.Attns):
                x1 = GCN(graph, x)
                x2 = Attn(x, x1, x1)
                x = x1 + x2
                xs.append(x)
            out_feas = torch.cat(xs, dim=-1)
            features.append(out_feas)

        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1
        h_t_limit = h_t_pairs.size(1)
        path_info = torch.zeros((bsz, h_t_limit, self.bank_size)).cuda()
        rel_mask = params['relation_mask']
        path_table = params['path2_table']

        path_len_dict = defaultdict(list)

        entity_num = torch.max(params['entity_id'])
        entity_bank = torch.Tensor(bsz, entity_num, self.bank_size).cuda()

        for i in range(len(graphs)):
            max_id = torch.max(params['entity_id'][i])
            entity_feas = features[i][-max_id:]
            entity_bank[i, :entity_feas.size(0)] = entity_feas
            path_t = path_table[i]
            for j in range(h_t_limit):
                h_ent = h_t_pairs[i, j, 0].item()
                t_ent = h_t_pairs[i, j, 1].item()

                if rel_mask is not None and rel_mask[i, j].item() == 0:
                    break

                if rel_mask is None and h_ent == 0 and t_ent == 0:
                    continue

                paths = path_t[(h_ent+1, t_ent+1)]
                for path in paths:
                    path = torch.LongTensor(path).cuda()
                    cur_h = torch.index_select(features[i], 0, path)
                    path_len_dict[len(path)].append((i, j, cur_h))

        h_ent_idx = h_t_pairs[:, :, 0].unsqueeze(-1).expand(
            -1, -1, self.bank_size)
        t_ent_idx = h_t_pairs[:, :, 1].unsqueeze(-1).expand(
            -1, -1, self.bank_size)
        h_ent_feas = torch.gather(input=entity_bank, dim=1, index=h_ent_idx)
        t_ent_feas = torch.gather(input=entity_bank, dim=1, index=t_ent_idx)

        path_embedding = {}

        for items in path_len_dict.values():
            cur_hs = torch.stack([h for _, _, h in items], 0)
            cur_hs, _ = self.rnn(cur_hs)
            cur_hs = cur_hs.max(1)[0]
            # cur_hs = cur_hs.mean(1)
            for idx, (i, j, _) in enumerate(items):
                if (i, j) not in path_embedding:
                    path_embedding[(i, j)] = []
                path_embedding[(i, j)].append(cur_hs[idx])

        querys = h_ent_feas - t_ent_feas

        for (i, j), emb in path_embedding.items():
            query = querys[i:i+1, j:j+1]
            keys = torch.stack(emb).unsqueeze(1)
            output, attn_weights = self.path_attn(query, keys, keys)
            path_info[i, j] = output.squeeze(0).squeeze(0)

        out_feas = torch.cat([
            h_ent_feas, t_ent_feas,
            torch.abs(h_ent_feas - t_ent_feas),
            torch.mul(h_ent_feas, t_ent_feas),
            path_info], dim=-1)
        out_feas = self.predict2(out_feas)
        m_preds = self.out_linear(out_feas)
        b_preds = self.out_linear_binary(out_feas)

        # b_preds_score = torch.softmax(b_preds, dim=-1)
        # m_preds_score = torch.sigmoid(m_preds)
        # m_preds_score2 = torch.zeros(m_preds_score.shape).cuda()
        # m_preds_score2[:, :, 1:] = m_preds_score[:, :, 1:] * b_preds_score[:, :, 1].unsqueeze(-1)
        # m_preds_score2[:, :, 0] = m_preds_score[:, :, 0] * b_preds_score[:, :, 0]

        return m_preds, b_preds, None
