from tqdm import tqdm
import ujson as json
import numpy as np
from graph_utils import build_g

cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
chr_rel2id = {'1:NR:2': 0, '1:React:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}
biored_rel2id = {
    '1:NR:2': 0, '1:Positive_Correlation:2': 1, '1:Association:2': 2, '1:Bind:2': 3,
    '1:Negative_Correlation:2': 4, '1:Cotreatment:2': 5, '1:Comparison:2': 6,
    '1:Conversion:2': 7, '1:Drug_Interaction:2': 8}

biored_valid_pairs = {
    ('ChemicalEntity', 'ChemicalEntity'),
    ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
    ('ChemicalEntity', 'GeneOrGeneProduct'),
    ('ChemicalEntity', 'SequenceVariant'),
    ('DiseaseOrPhenotypicFeature', 'ChemicalEntity'),
    ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
    ('DiseaseOrPhenotypicFeature', 'SequenceVariant'),
    ('GeneOrGeneProduct', 'ChemicalEntity'),
    ('GeneOrGeneProduct', 'DiseaseOrPhenotypicFeature'),
    ('GeneOrGeneProduct', 'GeneOrGeneProduct'),
    ('SequenceVariant', 'ChemicalEntity'),
    ('SequenceVariant', 'DiseaseOrPhenotypicFeature'),
    ('SequenceVariant', 'GeneOrGeneProduct'),
    ('SequenceVariant', 'SequenceVariant')}

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def read_docred(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        sub2words = []
        cur_pos = 0

        sent_diffs = [0]

        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                next_pos = cur_pos + len(tokens_wordpiece)
                # if (i_s, i_t) in entity_start:
                #     tokens_wordpiece = ["*"] + tokens_wordpiece
                #     next_pos += 1
                # if (i_s, i_t) in entity_end:
                #     tokens_wordpiece = tokens_wordpiece + ["*"]
                #     next_pos += 1
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                sub2words.append((cur_pos, next_pos))
                cur_pos = next_pos
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)
            sent_diffs.append(sent_diffs[-1] + len(sent))

        assert len(sents) == cur_pos
        # ######################
        offset = 1
        pos_idx = {}
        sub2word_tensor = np.zeros((len(sub2words) + len(entities), cur_pos + offset))
        for w_idx, (start, end) in enumerate(sub2words):
            if end > start:
                sub2word_tensor[w_idx, start + offset: end + offset] = 1 / (end - start)

        # TODO: double check if the entity id needs to start from 1
        entity_ranges = []
        for ent_idx, entity in enumerate(entities):
            pos_idx[ent_idx] = []
            cur_ranges = []
            cur_cnt = 0
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                sent_diff = sent_diffs[sent_id]
                word_start = sent_diff + pos[0]
                word_end = sent_diff + pos[1]
                start = sub2words[word_start][0]
                if word_end >= len(sub2words):
                    end = sub2words[-1][1]
                else:
                    end = sub2words[word_end][0]
                cur_ranges.append((start, end))
                cur_cnt += end - start
                pos_idx[ent_idx].extend(range(word_start, word_end))
            entity_ranges.append(cur_ranges)
            for start, end in cur_ranges:
                sub2word_tensor[len(sub2words) + ent_idx, start + offset: end + offset] = 1 / cur_cnt

        graph, path = build_g(sample['sents'], pos_idx, len(entities))

        # ######################

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        pos_relations, neg_relations, pos_hts, neg_hts = [], [], [], []

        # get positive relations
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            pos_relations.append(relation)
            pos_hts.append([h, t])
            pos_samples += 1

        # get negative relations
        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in pos_hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    neg_relations.append(relation)
                    neg_hts.append([h, t])
                    neg_samples += 1

        assert len(pos_relations) + len(neg_relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        # input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {
            'input_ids': input_ids, 'entity_pos': entity_pos,
            'pos_rels': pos_relations, 'neg_rels': neg_relations,
            'pos_hts': pos_hts, 'neg_hts': neg_hts,
            'title': sample['title'], 'graph': graph,
            'sub2word': sub2word_tensor, 'path': path,
        }
        features.append(feature)

    print(
        "# of documents {}.".format(i_line),
        "# of positive examples {}.".format(pos_samples),
        "# of negative examples {}.".format(neg_samples))
    return features


def read_cdr(file_in, tokenizer, max_seq_length=1024):
    return read_bio(file_in, tokenizer, cdr_rel2id, max_seq_length)


def read_chr(file_in, tokenizer, max_seq_length=1024):
    return read_bio(file_in, tokenizer, chr_rel2id, max_seq_length)


def read_biored(file_in, tokenizer, max_seq_length=1024):
    return read_bio(file_in, tokenizer, biored_rel2id, max_seq_length, biored_valid_pairs)


def read_bio(file_in, tokenizer, rel2id, max_seq_length=1024, valid_ents_pair=None):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                org_sents = sents
                new_sents = []
                sent_map = {}
                i_t = 0

                sub2words = []
                cur_pos = 0

                sent_diffs = [0]

                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        next_pos = cur_pos + len(tokens_wordpiece)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                                next_pos += 1
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                                next_pos += 1
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        sub2words.append((cur_pos, next_pos))
                        cur_pos = next_pos
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                    sent_diffs.append(sent_diffs[-1] + len(sent))
                sents = new_sents

                assert len(sents) == cur_pos

                entity_pos = []
                entities = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                        h_tpy, t_tpy = p[7], p[13]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                        t_tpy, h_tpy = p[7], p[13]
                    h_start_word = list(map(int, h_start.split(':')))
                    h_end_word = list(map(int, h_end.split(':')))
                    t_start_word = list(map(int, t_start.split(':')))
                    t_end_word = list(map(int, t_end.split(':')))
                    h_start = [sent_map[idx] for idx in h_start_word]
                    h_end = [sent_map[idx] for idx in h_end_word]
                    t_start = [sent_map[idx] for idx in t_start_word]
                    t_end = [sent_map[idx] for idx in t_end_word]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                        entities.append(list(zip(h_start_word, h_end_word)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                        entities.append(list(zip(t_start_word, t_end_word)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = rel2id[p[0]]
                    if valid_ents_pair is not None and (h_tpy, t_tpy) not in valid_ents_pair:
                        # print('YW =======> invalid pair: ', h_tpy, t_tpy)
                        continue
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})
                # ######################
                offset = 1
                pos_idx = {}
                sub2word_tensor = np.zeros((len(sub2words) + len(entities), cur_pos + offset))
                ner2word_tensor = np.zeros((cur_pos + offset, ))
                for w_idx, (start, end) in enumerate(sub2words):
                    if end > start:
                        sub2word_tensor[w_idx, start + offset: end + offset] = 1 / (end - start)

                # TODO: double check if the entity id needs to start from 1
                entity_ranges = []
                for ent_idx, entity in enumerate(entities):
                    pos_idx[ent_idx] = []
                    cur_ranges = []
                    cur_cnt = 0
                    for mention in entity:
                        word_start, word_end = mention
                        start = sub2words[word_start][0]
                        if word_end >= len(sub2words):
                            end = sub2words[-1][1]
                        else:
                            end = sub2words[word_end][0]
                        cur_ranges.append((start, end))
                        cur_cnt += end - start
                        pos_idx[ent_idx].extend(range(word_start, word_end))
                    entity_ranges.append(cur_ranges)
                    for start, end in cur_ranges:
                        sub2word_tensor[len(sub2words) + ent_idx, start + offset: end + offset] = 1 / cur_cnt

                graph, path = build_g(org_sents, pos_idx, len(entities))

                # ######################

                relations, hts = [], []
                pos_relations, neg_relations, pos_hts, neg_hts = [], [], [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    if relation[0] == 1:
                        neg_relations.append(relation)
                        neg_hts.append([h, t])
                    else:
                        pos_relations.append(relation)
                        pos_hts.append([h, t])
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                # feature = {
                #     'input_ids': input_ids, 'entity_pos': entity_pos, 'labels': relations,
                #     'hts': hts, 'sents': [t.split(' ') for t in text.split('|')],
                #     'title': pmid, 'graph': graph, 'sub2word': sub2word_tensor,
                #     'prs': prs, 'path': path,
                # }
                feature = {
                    'input_ids': input_ids, 'entity_pos': entity_pos,
                    'pos_rels': pos_relations, 'neg_rels': neg_relations,
                    'sents': [t.split(' ') for t in text.split('|')],
                    'pos_hts': pos_hts, 'neg_hts': neg_hts,
                    'title': pmid, 'graph': graph, 'prs': prs, 
                    'sub2word': sub2word_tensor, 'path': path,
                }
                # org_sents_all = []
                # for org_sent in org_sents:
                #     org_sents_all.extend(org_sent)
                # for ent_subs, ent_words in zip(entity_pos, entities):
                #     for ent_sub, ent_word in zip(ent_subs, ent_words):
                #         print(
                #             '------->', ent_sub, sents[ent_sub[0]: ent_sub[1]],
                #             ent_word, org_sents_all[ent_word[0]: ent_word[1]])
                # 3 / 0
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features
