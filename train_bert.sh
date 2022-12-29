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

for model in 'MYBertRNN' ; do
  for bert_layer_nums in '11' '10' '9' '8' '7' ; do
#    time=$(date "+%Y%m%d")
#    nohup python -m src.run --model ${model} --do_train True --data_dir assets/data/gr --gpu_ids 0 --batch_size 512 --num_epochs 200 >> ./${model}_${time}.log 2>&1 &
    echo "*******${model} *********${bert_layer_nums}*************start**********************************"
    python -m src.run --model ${model} --bert_type nbroad/ESG-BERT --do_train true --do_predict_news True --data_dir assets/data/topic_en_Customer_Privacy_Incidents/ --gpu_ids 1 --num_epochs 30 --loss soft_bootstrapping_loss --bert_layer_nums ${bert_layer_nums} --save_model_name nbroad/ESG-BERT_${bert_layer_nums}
    wait
    echo "*******${model} *********${bert_layer_nums}*************end**********************************"
  done
done
