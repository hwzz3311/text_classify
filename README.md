## 介绍
本项目中包含了多个常用的模型方案，可作为文本分类的baseline，可用于快速进行模型选择，只需要一种数据格式可在不同的模型中测试。

模型主要包括：DPCNN、BertCNN、TextRNNAtt、TextRNN、BertRCNN、FastText、Transformer、BertRNN、Bert、BertDPCNN、TextRCNN、TextCNN；


## 目录结构
├── app  
│     └── app.py # API启动，Flask  
├── assets  
│     ├── data # 数据集存放目录  
│     │     ├── gr1_清洁能源  
│     │     │     ├── eval.json # 验证集  
│     │     │     ├── labels.txt # 标签  
│     │     │     ├── test.json # 测试集  
│     │     │     ├── train.json # 训练集  
├── script  
│     ├── train.sh # 训练的脚本  
│     └── train_sample.txt # 训练的示例  
├── src  # 源代码  
│     ├── models  模型存放  
│     │     ├── Bert.py  
│     │     ├── .....   
│     │     ├── config.py  # 将接收的参数进行封装成类   
│     │     ├── plugs.py # 插件  
│     ├── options.py # 配置接受的参数  
│     ├── processors.py # 数据处理类  
│     ├── run.py # 入口  
│     ├── train_eval.py # 训练和验证  
│     └── utils   
│         ├── model_utils.py # model相关的其他方法  
│         └── raw_data_utils.py # 处理原始的pdf数据  
├── tests # 其他的测试，可忽略  
│     ├── 0_basic.py # 其他的测试，可忽略  

### 模型训练步骤
1、准备数据，数据准备示例：{"label":"----","text":"------"}，将自定义的数据目录放在assets/data/
2、开始训练：

    示例：python -m src.run --model FastText --do_train true --data_dir assets/data/gr1_清洁能源 --gpu_ids 0 --batch_size 512
    
    查看更多参数：python -m src.run -h

    ps：--bert_type可以指定 bert类型或者目录clif

3、将训练的模型使用flask启动，进行接口测试

    python -m app.app --model FastText --data_dir assets/data/gr1_清洁能源 --gpu_ids 0 --check_point_path ****

    ps：check_point_path是可选选项，为空的时候会自动加载data_dir目录下的saved_dict/{model}.cpkt

### 待优化问题
1、虽然可以接收 --gpus，但是暂不支持多卡训练  
2、由于可以要兼容不同模型参数，部分超参需要去代码里修改