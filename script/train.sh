#!/bin/bash

set -e


for model in "DPCNN" "BertCNN" "TextRNNAtt" "TextRNN" "Transfromer" "FastText" "BertLSTM" "Bert" "TextCNN"; do
  python -m src.run --model Bert --do_train True --data_dir assets/data/company --batch_size 512
  wait
done
