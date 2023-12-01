#!/bin/bash

set -e
#"DPCNN"

#for model in 'DPCNN' 'TextRNNAtt' 'TextRNN' 'FastText' 'Transformer' 'TextRCNN' 'TextCNN'; do
#  echo "***************************************************************"
##  time=$(date "+%Y%m%d")
##  nohup python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --batch_size 512 --num_epochs 200 >> ./${model}_${time}.log 2>&1 &
#  python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --batch_size 512 --num_epochs 200
#  wait
#done

for model in 'BertRCNN'; do
  for data_dir in "1清洁能源" "2可持续建筑" "3能效管理" "4生态保护" "5绿色农业" "6水资源利用" "7污染防治" "8绿色交通" ; do
      #time=$(date "+%Y%m%d")
#    nohup python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --batch_size 512 --num_epochs 200 >> ./${model}_${time}.log 2>&1 &
    echo "***************************************************************"
    python -m src.run --model ${model} --do_train True --data_dir assets/data/${data_dir} --gpu_ids 0 --lr 5e-5 --dropout 0.1 --batch_size 16 --num_epochs 2000
    wait
  done
  for data_dir in "1清洁能源" "2可持续建筑" "3能效管理" "4生态保护" "5绿色农业" "6水资源利用" "7污染防治" "8绿色交通" ; do
      #time=$(date "+%Y%m%d")
#    nohup python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --batch_size 512 --num_epochs 200 >> ./${model}_${time}.log 2>&1 &
    echo "***************************************************************"
    python -m src.run --model ${model} --data_dir assets/data/${data_dir} --gpu_ids 0 --lr 5e-5 --dropout 0.1 --batch_size 16 --num_epochs 2000
    wait
  done
done
