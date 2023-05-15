#!/bin/bash

set -e

for model in 'Bert' ; do
  for bert_type in "bert-base-chinese" "hfl/chinese-bert-wwm-ext" "hfl/chinese-roberta-wwm-ext" "hfl/chinese-roberta-wwm-ext-large" ""; do
    for data_dir in "assets/data/my_classify/" ; do
      echo "${data_dir}*******${model} **********start**********************************"
      python -m src.run --model ${model} --bert_type ${bert_type} --do_train true --data_dir ${data_dir} --gpu_ids 2 --num_epochs 20 --loss soft_bootstrapping_loss --batch_size 10
      wait
      echo "${data_dir}*******${model} ***********end**********************************"
    done
  done
done
