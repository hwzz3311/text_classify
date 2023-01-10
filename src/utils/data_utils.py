import ast
import hashlib
import os

import regex


def text_to_md5(input_text: str):
    m = hashlib.md5()
    m.update(input_text.encode("utf8"))
    return m.hexdigest()

def cut_sent(para):
    patterns = ['([。;；！？\?])([^”’])', '(\.{6})([^”’])', '(\…{2})([^”’])', '([。！？\?][”’])([^，。！？\?])']

    if all([regex.search(p, para) is None for p in patterns]):
        para = para + '.'
    para = regex.sub('([。;；！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = regex.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = regex.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = regex.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 破折号、英文双引号等忽略，需要的再做些简单调整即可。
    sents = para.split("\n")
    sents = [s.strip() for s in sents]
    sents = [s for s in sents if s]
    return sents


def cut_sentences(json_data, config):
    para = json_data.get("title", "") + "\n" + json_data.get("content", "")
    patterns = ['([。;；！？\?])([^”’])', '(\.{6})([^”’])', '(\…{2})([^”’])', '([。！？\?][”’])([^，。！？\?])']

    if all([regex.search(p, para) is None for p in patterns]):
        para = para + '.'
    para = regex.sub('([。;；！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = regex.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = regex.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = regex.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 破折号、英文双引号等忽略，需要的再做些简单调整即可。
    sentences = regex.split("([?？!！。.])", para)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    res = []
    for i in range(0, len(sentences)):
        text = "".join(sentences[i:i + config.cut_sen_len])  # 每steps 句拼接为一个text，进一次模型
        res.append(text)
    return res

def out_data_to_jsonl(datas, out_file_path, mode="w"):
    file_dir = os.path.dirname(out_file_path)
    os.makedirs(file_dir, exist_ok=True)
    with open(out_file_path, mode) as f:
        for e in datas:
            f.write(str(e) + "\n")
    print(f"write data len :{len(datas)} to file :{out_file_path}")


def load_jsonl_file(file_path, is_predict=False, config=None):
    res = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            json_data: dict = ast.literal_eval(line)
            res.append(json_data)
    if is_predict:
        assert config is not None, "config cannot be None under the prediction model"
        res_e = []
        for json_data in res:
            for text in cut_sentences(json_data, config):
                # 如果是 predict 模式，则将所有的label都mask掉，因为预测模式下用不到 label
                res_e.append({"text": text, "label": config.class_list[0], "news_id": json_data["news_id"]})
        return res_e
    return res
