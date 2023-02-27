import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_gda, read_biored, read_chr, read_cdr
from tqdm import tqdm


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        total_steps = len(train_dataloader) * num_epoch
        warmup_steps = int(total_steps * args.warmup_ratio)
        # warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        best_test_score = 0
        for epoch in range(int(num_epoch)):
            model.zero_grad()
            for batch in tqdm(train_dataloader, desc='Epoch ' + str(epoch)):
                model.train()
                inputs = {
                    'input_ids': batch[0].cuda(), 'attention_mask': batch[1].cuda(),
                    'labels': batch[9], 'entity_pos': batch[3], 'hts': batch[8],
                    'graphs': batch[5], 'sub2words': [f.cuda() for f in batch[6]],
                    'paths': batch[7],
                }
                loss, _ = model(**inputs)
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            dev_score, dev_output = evaluate(epoch, args, model, dev_features, tag="dev")
            test_score, test_output = evaluate(epoch, args, model, test_features, tag="test")
            print(dev_output)
            print(test_output)
            if dev_score > best_score:
                best_score = dev_score
                if args.save_path != "":
                    torch.save(model.state_dict(), args.save_path)
            best_test_score = max(best_test_score, test_score)
        print('Best test score: ', best_test_score)

    bert_param_ids = list(map(id, model.model.parameters()))
    base_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids, model.parameters())
    optimizer = AdamW([
        {'params': model.model.parameters(), 'lr': 0.00005},
        {'params': base_params, 'weight_decay': 0.0005}
        # {'params': base_params}
    ], lr=0.0001)

    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs)


def evaluate(epoch, args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds_list, golds_list = [], []
    for batch in tqdm(dataloader, desc='Evaluate'):
        model.eval()
        inputs = {
            'input_ids': batch[0].cuda(), 'attention_mask': batch[1].cuda(),
            'entity_pos': batch[3], 'hts': batch[4], 'graphs': batch[5],
            'sub2words': [f.cuda() for f in batch[6]], 'paths': batch[7],
        }

        with torch.no_grad():
            pred = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds_list.append(pred)
            golds_list.append(np.concatenate([np.array(label, np.float32) for label in batch[2]], axis=0))

    preds = np.concatenate(preds_list, axis=0).astype(np.float32)
    golds = np.concatenate(golds_list, axis=0).astype(np.float32)

    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    tn = ((preds[:, 1] != 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    fn = ((preds[:, 1] != 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    # output = {
    #     "{}_f1".format(tag): f1 * 100,
    # }
    output = {
        "{}_f1".format(tag): f1 * 100, "{}_P".format(tag): precision * 100,
        "{}_R".format(tag): recall * 100,
    }
    # print('-------->', 'TP', tp, 'TN', fn, 'FP', fp, 'FN', tn)
    return f1, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/cdr", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="allenai/scibert_scivocab_cased", type=str)

    parser.add_argument("--train_file", default="train_filter.data", type=str)
    parser.add_argument("--dev_file", default="dev_filter.data", type=str)
    parser.add_argument("--test_file", default="test_filter.data", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=1, type=int,
                        help="Max number of labels in the prediction.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization.")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of relation types in dataset.")
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--gcn_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.4)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    if "cdr" in args.data_dir:
        read = read_cdr
    elif "chr" in args.data_dir:
        read = read_chr
    elif "biored" in args.data_dir:
        read = read_biored
    else:
        read = read_gda

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    config.gcn_dim = args.gcn_dim
    config.gcn_layers = args.gcn_layers
    config.dropout = args.dropout
    config.num_class = args.num_class
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    if args.load_path == "":
        train(args, model, train_features, dev_features, test_features)
    else:
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(dev_output)
        print(test_output)


if __name__ == "__main__":
    main()
