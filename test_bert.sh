#!/bin/bash

set -e

for model in 'MYBert' ; do
  for bert_layer_nums in '10' '9' ; do
    for data_dir in "assets/data/topic_en_Customer_Privacy_Incidents/" ; do
      save_model_name="ESG-BERT_230206_${bert_layer_nums}"
      echo "${data_dir}*******${model} *********${bert_layer_nums}*************start**********************************"
      echo "${data_dir}*******${model} *********${bert_layer_nums}*************end**********************************"
      check_point_path="${data_dir}saved_dict/${model}/${save_model_name}.cpkt"
      for threshold in "0.55" "0.6" "0.65" "0.7" "0.75" "0.8" "0.85" "0.9"; do
        echo "${data_dir}*******${model} *********${bert_layer_nums}*********${threshold}****from****${check_point_path}**start"
        python -m src.run --model ${model} --bert_type nbroad/ESG-BERT --do_predict_news True --data_dir ${data_dir} --gpu_ids 1 --bert_layer_nums ${bert_layer_nums} --threshold ${threshold} --check_point_path ${check_point_path}
        wait
        echo "${data_dir}*******${model} *********${bert_layer_nums}*********${threshold}****from****${check_point_path}**end"
      done
    done
  done
done
