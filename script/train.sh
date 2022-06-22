#!/bin/bash

set -e


for model in "DPCNN" "BertCNN" "TextRNNAtt" "TextRNN" "Transfromer" "FastText" "BertLSTM" "Bert" "TextCNN"; do
  python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --batch_size 512
  wait
done