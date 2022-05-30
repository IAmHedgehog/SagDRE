#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

input_theta=${1--1}
batch_size=6
test_batch_size=6
dataset=dev

model_name=GAIN_BERT_large

python3 test.py \
  --train_set ../data/train_annotated.json \
  --dev_set ../data/dev.json \
  --test_set ../data/test.json \
  --model_name ${model_name} \
  --use_model bert \
  --pretrain_model checkpoint/GAIN_ROBERTA_38.pt \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --gcn_layers 2 \
  --bert_hid_size 1024 \
  --gcn_dim 64 \
  --use_entity_type \
  --use_entity_id \
  --dropout 0.6 \
  --activation relu \
  --input_theta ${input_theta}
