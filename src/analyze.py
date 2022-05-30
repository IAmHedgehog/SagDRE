import json
import csv
import networkx as nx
import matplotlib.pyplot as plt
from graph_utils import nlp, parse_sent, add_same_words_links, add_entity_node, get_entity_paths


def load_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data


def build_g2(sentences, pos_idx, max_id):
    # sentences should be a list of word lists
    # [[sent_1], [sent_2], ..., [sent_m]]
    # senti = [w_0, w_1, ..., w_n]
    pre_roots = []
    g = nx.DiGraph()
    for tokens in nlp.pipe(sentences):
        g, pre_roots = parse_sent(tokens, g, pre_roots)
    g = add_same_words_links(g, remove_stopwords=True)
    g, start = add_entity_node(g, pos_idx, max_id)
    paths = get_entity_paths(g, max_id, start)
    return g, paths


def get_details(data_file, result_file, all_result_file, out_file, rel2id_file):
    rel2id = load_json(rel2id_file)
    data = load_json(data_file)
    results = load_json(result_file)
    all_results = load_json(all_result_file)
    # here will omit records that share the same index id
    results = [d for d in results if not d['correct']]
    result_inds = set([d['index'] for d in results])
    all_data = {(res[-4], res[-3], res[-2], res[-1]): res[1]
                for res in all_results if res[-4] in result_inds}
    del all_results
    data = {i: d for i, d in enumerate(data)}
    ori_headers = [
        'correct', 'index', 'h_idx', 't_idx', 'r_idx', 'score',
        'r']
    headers = ori_headers + ['title', 'h_ent', 't_ent', 'scores', 'doc']
    with open(out_file, 'w', newline='\n') as csvfile:
        writer = csv.writer(
            csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for result in results:
            cur_d = [result[c] for c in ori_headers]
            cur_d.append(result['title'].replace(',', ' '))
            cur_data = data[result['index']]
            h_ent = cur_data['vertexSet'][result['h_idx']][0]['name'].replace(',', ' ')
            t_ent = cur_data['vertexSet'][result['t_idx']][0]['name'].replace(',', ' ')
            doc = ' '.join([' '.join(words) for words in cur_data['sents']])
            doc = doc.replace(',', ' ')

            key = (result['h_idx'], result['t_idx'])
            labels = {}
            for d in cur_data['labels']:
                cur_key = (d['h'], d['t'])
                if cur_key not in labels:
                    labels[cur_key] = []
                labels[cur_key].append(rel2id[d['r']])
            if key not in labels:
                correct_rs = [0]
            else:
                correct_rs = labels[key]
            cr_scores = []
            for cr in correct_rs:
                cr_key = (result['index'], result['h_idx'], result['t_idx'], cr)
                if cr_key not in all_data:
                    score = 0.9102
                else:
                    score = all_data[cr_key]
                cr_scores.append(str(cr)+'='+str(score))
            cur_d += [h_ent, t_ent, ' '.join(cr_scores), doc]
            writer.writerow(cur_d)
    return data


def get_pos_idx(doc):
    entity_list = doc['vertexSet']
    sentences = doc['sents']
    Ls = [0]
    L = 0
    for x in sentences:
        L += len(x)
        Ls.append(L)
    for j in range(len(entity_list)):
        for k in range(len(entity_list[j])):
            sent_id = int(entity_list[j][k]['sent_id'])
            entity_list[j][k]['sent_id'] = sent_id

            dl = Ls[sent_id]
            pos0, pos1 = entity_list[j][k]['pos']
            entity_list[j][k]['global_pos'] = (pos0 + dl, pos1 + dl)

    pos_idx = {}
    for idx, vertex in enumerate(entity_list, 1):
        for v in vertex:
            pos0, pos1 = v['global_pos']
            if idx not in pos_idx:
                pos_idx[idx] = []
            pos_idx[idx].extend(range(pos0, pos1))
    return pos_idx


def build_graph(doc_id, data):
    doc = data[doc_id]
    sentences = doc['sents']
    pos_idx = get_pos_idx(doc)
    max_id = max(pos_idx.keys())
    g, path = build_g2(sentences, pos_idx, max_id)
    return g, path, max_id


def draw_graph(h_idx, t_idx, G, path, max_id):
    pos = nx.kamada_kawai_layout(G)
    node_colors = ['gray'] * len(G.nodes)
    edge_color = ['gray'] * len(G.edges)
    nodes = list(G.nodes)
    edges = list(G.edges)
    # words = dict(G.nodes.data("is_root"))

    ents_ids = []
    for i in range(len(nodes)):
        if i in path:
            node_colors[i] = 'purple'
        if i in [h_idx, t_idx]:
            node_colors[i] = 'green'
            ents_ids.append(i)

    total_path = []
    # import ipdb; ipdb.set_trace()
    for i in range(len(path)-1):
        total_path.append((path[i], path[i+1]))
    for i in range(len(edges)):
        if edges[i] in total_path or (edges[i][1], edges[i][0]) in total_path:
            edge_color[i] = 'red'

    node_labels = nx.get_node_attributes(G, 'text')
    nx.draw_networkx(
        G, pos, node_size=30, labels=node_labels, font_size=7,
        node_color=node_colors, font_color='purple', edge_color=edge_color)
    plt.show()


if __name__ == "__main__":
    data_file = '../data/dev.json'
    all_result_file = 'dev_all.json'
    result_file = 'dev_index.json'
    out_file = 'dev_result.csv'
    rel2id_file = '../data/rel2id.json'
    # data = get_details(
    #     data_file, result_file, all_result_file, out_file, rel2id_file)

    data = load_json(data_file)
    data = {i: d for i, d in enumerate(data)}

    doc_id = 2
    h_idx, t_idx = 0, 1
    g, path, max_id = build_graph(doc_id, data)
    draw_graph(h_idx+1, t_idx+1, g, path[(h_idx+1, t_idx+1)], max_id)
