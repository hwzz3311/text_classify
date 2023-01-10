#!/bin/bash

set -e

for model in 'MYBert' ; do
  for bert_layer_nums in '11' '10' '9' ; do
    for data_dir in "assets/data/topic_en_Customer_Privacy_Incidents/" "assets/data/topic_en_greenwashing/" "assets/data/topic_en_Safety_Accidents_v1/" ; do
      save_model_name="ESG-BERT_230106_${bert_layer_nums}"
      echo "${data_dir}*******${model} *********${bert_layer_nums}*************start**********************************"
      python -m src.run --model ${model} --bert_type nbroad/ESG-BERT --do_train true --do_predict_news True --data_dir ${data_dir} --gpu_ids 1 --num_epochs 20 --loss soft_bootstrapping_loss --bert_layer_nums ${bert_layer_nums} --save_model_name ${save_model_name}
      wait
      echo "${data_dir}*******${model} *********${bert_layer_nums}*************end**********************************"
      check_point_path="${data_dir}saved_dict/${model}/${save_model_name}.cpkt"
      for threshold in "0.5" "0.55" "0.6" "0.65" "0.7" "0.75" "0.8"; do
        echo "${data_dir}*******${model} *********${bert_layer_nums}*********${threshold}****from****${check_point_path}**start"
        python -m src.run --model ${model} --bert_type nbroad/ESG-BERT --do_predict_news True --data_dir ${data_dir} --gpu_ids 1 --bert_layer_nums ${bert_layer_nums} --threshold ${threshold} --check_point_path ${check_point_path}
        wait
        echo "${data_dir}*******${model} *********${bert_layer_nums}*********${threshold}****from****${check_point_path}**end"
      done
    done
  done
done
