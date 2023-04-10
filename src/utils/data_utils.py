import ast
import hashlib
import os
import re

import regex


def text_to_md5(input_text: str):
    m = hashlib.md5()
    m.update(input_text.encode("utf8"))
    return m.hexdigest()


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


def para_clear(para):
    patterns = ['([。;；！？\?])([^”’])', '(\.{6})([^”’])', '(\…{2})([^”’])', '([。！？\?][”’])([^，。！？\?])']

    if all([regex.search(p, para) is None for p in patterns]) and not str(para).endswith(".") and str(para).find(
            ".") == -1:
        para = para + '.'
    para = regex.sub('([。;；！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = regex.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = regex.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = regex.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para


def para_sentences(para, clear=True):
    if clear:
        para = para_clear(para)

    replace_dict = {}
    re_pattern_dict = {
        "[0-9]\d*\.[0-9]\d*": regex.IGNORECASE,
        "PTE\. LTD\.": regex.IGNORECASE,
        "PTE\. LTD": regex.IGNORECASE,
        "S\.A\.": regex.IGNORECASE,
        "[A-Z]\.": regex.FULLCASE,
        "\n[A-Z-a-z].[^ ]\.": regex.FULLCASE,
        "[ (.]Rs\.": regex.IGNORECASE,
        "[ (.]No\.": regex.IGNORECASE,
        " Dr\.": regex.IGNORECASE,
        " Co\.": regex.IGNORECASE,
        " Act\.": regex.IGNORECASE,
        " Ltd\.": regex.IGNORECASE,
        " Mr\.": regex.IGNORECASE,
        "www\.[^\s]+[\.\?]": regex.IGNORECASE,
        "United States\.": regex.IGNORECASE,
        " St\.": regex.IGNORECASE,
        " p\.m\.": regex.IGNORECASE,
        " i\. e\.": regex.IGNORECASE,
        " Inc\.": regex.IGNORECASE,
        " Dept\.": regex.IGNORECASE,
        "[. ][^\s]{1}\.": regex.IGNORECASE,
        " Corp\.": regex.IGNORECASE,
        '"\.': regex.IGNORECASE,
        '[A-Z]\.[A-Z]\.': regex.IGNORECASE,
    }
    for pattern, v in re_pattern_dict.items():
        regex_res_list = regex.findall(pattern, para, v)
        for regex_res in regex_res_list:
            new_text = regex_res
            for k in "?？!！。.":
                new_text = new_text.replace(k, "#&^%")
            replace_dict[regex_res] = new_text
    for k, v in replace_dict.items():
        para = para.replace(k, v)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 破折号、英文双引号等忽略，需要的再做些简单调整即可。
    sentences = regex.split("([?？!！。.])", para)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    new_sentences = []
    # new_mid_sentences = []
    for sen in sentences:
        for k, v in replace_dict.items():
            if v in sen:
                sen = sen.replace(v, k)
        if len(sen):
            new_sentences.append(sen)

    return new_sentences


def out_data_to_jsonl(datas, out_file_path, mode="w"):
    file_dir = os.path.dirname(out_file_path)
    os.makedirs(file_dir, exist_ok=True)
    with open(out_file_path, mode) as f:
        for e in datas:
            f.write(str(e) + "\n")
    print(f"write data len :{len(datas)} to file :{out_file_path}")


def pre_cut_by_sentences(news_id, sentences, config):
    assert config is not None, "config cannot be None under the prediction model"
    res_e = []
    for i in range(0, len(sentences), max(1, int(config.cut_sen_len / 2))):
        try:
            text = " ".join(sentences[i:i + config.cut_sen_len])  # 每steps 句拼接为一个text，进一次模型
        except:
            print("")
        text = text.replace("\n", " ")
        if len(text.strip()) <= 0:
            continue
        res_e.append({"text": text, "label": config.class_list[0], "news_id": news_id})
    return res_e


def pre_cut_sentences(datas, config):
    assert config is not None, "config cannot be None under the prediction model"
    res_e = []
    for json_data in datas:
        para = json_data.get("title", "") + "\n" + json_data.get("content", "")
        sentences = para_sentences(para)
        for i in range(0, len(sentences), max(1, int(config.cut_sen_len / 2))):
            text = " ".join(sentences[i:i + config.cut_sen_len])  # 每steps 句拼接为一个text，进一次模型
            text = text.replace("\n", " ")
            if len(text.strip()) <= 0:
                continue
            res_e.append({"text": text, "label": config.class_list[0], "news_id": json_data["news_id"]})
    return res_e

def load_jsonl_file(file_path, is_predict=False, config=None):
    res = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            json_data: dict = ast.literal_eval(line)
            res.append(json_data)
    if is_predict:
        return pre_cut_sentences(res, config)
    return res

def gen_pattern(keywords: list, expansion_num=2, close_brackets=False) -> str:
    keywords = keywords.copy()
    if expansion_num > 0:
        for index in range(len(keywords)):
            keyword = keywords[index]
            keyword = (".{0," + str(expansion_num) + "}").join(list(keyword))
            keywords[index] = keyword
    keyword_pattern = "|".join([f"({i})" for i in keywords if len(i) > 0])
    if close_brackets:
        keyword_pattern = "(" + keyword_pattern + ")"
    return keyword_pattern
