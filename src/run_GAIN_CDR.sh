#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=GAIN_GLOVE_base
lr=0.001
batch_size=16
test_batch_size=16
epoch=40
test_epoch=1
save_model_freq=1
negativa_alpha=4

python3 train_bio.py \
  --train_set ../data/CDR_json/CDR_train_data.json \
  --dev_set ../data/CDR_json/CDR_dev_data.json \
  --test_set ../data/CDR_json/CDR_test_data.json \
  --use_model glove \
  --model_name ${model_name} \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epoch ${epoch} \
  --test_epoch ${test_epoch} \
  --save_model_freq ${save_model_freq} \
  --negativa_alpha ${negativa_alpha} \
  --gcn_layers 2 \
  --bert_hid_size 768 \
  --gcn_dim 32 \
  --word_emb_size 100 \
  --lstm_hidden_size 256 \
  --dropout 0.0 \
  --activation relu \
  --relation_nums 2
