import json
from tqdm import tqdm
import numpy as np
import torch
from config import get_opt
from data_bio import DGLREDataloader, BERTDGLREDataset, GloveDGLREDataset
from GAIN import GAIN_BERT
from GAIN_glove import GAIN_GLOVE
from utils import logging


def get_test_results(model, dataloader, relation_num, id2rel):

    test_results = []

    for d in tqdm(dataloader, unit='b'):
        labels = d['labels']
        L_vertex = d['L_vertex']
        with torch.no_grad():
            m_preds, _, _ = model(
                words=d['context_idxs'],
                src_lengths=d['context_word_length'],
                mask=d['context_word_mask'],
                entity_type=d['context_ner'],
                entity_id=d['context_pos'],
                h_t_pairs=d['h_t_pairs'],
                relation_mask=None,
                graph2s=d['graph2s'],
                path2_table=d['path2_table'],
                sub2words=d['sub2words'])

        m_preds = m_preds.data.cpu().numpy()

        for i in range(len(labels)):
            label = labels[i]
            L = L_vertex[i]
            j = 0
            for h_idx in range(L):
                for t_idx in range(L):
                    if h_idx != t_idx:
                        gold = int((h_idx, t_idx, 1) in label)
                        pred = 1 if m_preds[i, j, 1] > m_preds[i, j, 0] else 0
                        cur_result = (gold, pred)
                        test_results.append(cur_result)
                        j += 1
    return test_results


def test(model, dataloader, id2rel, input_theta=-1, output=False,
         test_prefix='dev', relation_num=97, test_theta=None, test_bound=None):

    test_results = get_test_results(
        model, dataloader, relation_num, id2rel)

    golds = np.array([result[0] for result in test_results])
    preds = np.array([result[1] for result in test_results])

    tp = ((preds == 1) & (golds == 1)).astype(np.float32).sum()
    fn = ((preds != 1) & (golds == 1)).astype(np.float32).sum()
    fp = ((preds == 1) & (golds != 1)).astype(np.float32).sum()

    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    print('--->', tp, fn, fp, precision, recall, f1)
    return f1, input_theta, test_theta


if __name__ == '__main__':
    opt, data_opt = get_opt()
    data_opt.data_word_vec = data_opt.word2vec
    if opt.use_model == 'bert':
        train_set = BERTDGLREDataset(
            opt.train_set, data_opt.ner2id, data_opt.rel2id, dataset='train',
            model_name=opt.model_name)
        dev_set = BERTDGLREDataset(
            opt.dev_set, data_opt.ner2id, data_opt.rel2id, dataset='dev',
            instance_in_train=train_set.instance_in_train,
            model_name=opt.model_name)
        test_set = BERTDGLREDataset(
            opt.test_set, data_opt.ner2id, data_opt.rel2id, dataset='test',
            instance_in_train=train_set.instance_in_train,
            model_name=opt.model_name)
        model = GAIN_BERT(opt)
    else:
        train_set = GloveDGLREDataset(
            opt.train_set, data_opt.word2id, data_opt.ner2id, data_opt.rel2id,
            dataset='train')
        dev_set = GloveDGLREDataset(
            opt.dev_set, data_opt.word2id, data_opt.ner2id, data_opt.rel2id,
            dataset='dev', instance_in_train=train_set.instance_in_train)
        test_set = GloveDGLREDataset(
            opt.test_set, data_opt.word2id, data_opt.ner2id, data_opt.rel2id,
            dataset='test', instance_in_train=train_set.instance_in_train)
        model = GAIN_GLOVE(opt, data_opt)

    # dev_loader = DGLREDataloader(
    #     dev_set, batch_size=opt.test_batch_size, dataset_type='test')
    test_loader = DGLREDataloader(
        test_set, batch_size=opt.test_batch_size, dataset_type='test')

    chkpt = torch.load(opt.pretrain_model, map_location=torch.device('cpu'))
    model.load_state_dict(chkpt['checkpoint'])
    logging('load checkpoint from {}'.format(opt.pretrain_model))

    model = model.cuda()
    model.eval()

    # f1, input_theta, test_theta = test(
    #     model, dev_loader, id2rel=data_opt.id2rel,
    #     input_theta=opt.input_theta, output=False, test_prefix='dev')
    # json.dump([input_theta, test_theta], open("dev_theta.json", "w"))
    input_theta, test_theta = json.load(open("dev_theta.json"))
    test(
        model, test_loader, id2rel=data_opt.id2rel, input_theta=input_theta,
        output=True, test_prefix='test', test_theta=test_theta,
        relation_num=opt.relation_nums,
        # test_bound=12626)
        test_bound=20000)
    print('finished')
