#!/bin/bash

set -e
#"DPCNN"

for model in 'FastText' 'TextRNN' 'Transformer' 'TextRCNN' 'TextCNN' 'DPCNN' 'TextRNNAtt' ; do
  echo "***************************************************************"
#  time=$(date "+%Y%m%d")
#  nohup python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --batch_size 512 --num_epochs 200 >> ./${model}_${time}.log 2>&1 &
  python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --batch_size 512 --num_epochs 200
  wait
done

#for model in 'BertCNN' 'BertRCNN' 'BertRNN' 'Bert' 'BertDPCNN'; do
#  for bert_type in "bert-base-chinese" "hfl/chinese-bert-wwm-ext" "hfl/chinese-roberta-wwm-ext" "albert-base-v2" "hfl/chinese-xlnet-base" "hfl/chinese-electra-180g-base-discriminator" "distilbert-base-uncased" "nghuyong/ernie-gram-zh" ; do
##    time=$(date "+%Y%m%d")
##    nohup python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --batch_size 512 --num_epochs 200 >> ./${model}_${time}.log 2>&1 &
#    echo "***************************************************************"
#    python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --lr 5e-5 --dropout 0.1 --batch_size 16 --num_epochs 200 --bert_type ${bert_type}
#    wait
#  done
#done
