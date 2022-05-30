#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=GAIN_BERT_base
lr=0.001
batch_size=12
test_batch_size=16
epoch=40
test_epoch=5
save_model_freq=1
negativa_alpha=4

python3 train.py \
  --train_set ../data/train_annotated.json \
  --dev_set ../data/dev.json \
  --test_set ../data/test.json \
  --use_model bert \
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
  --gcn_dim 128 \
  --use_entity_type \
  --use_entity_id \
  --dropout 0.4 \
  --activation relu \
  --coslr
