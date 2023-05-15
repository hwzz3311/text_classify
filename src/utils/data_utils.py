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
            # print(text)
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


if __name__ == "__main__":
    text = """Greenpeace’s scorecard on consumer goods companies zero-deforestation policies.  Image: Greenpeace Under fierce pressure from environmental groups, aff
ected communities, and members of the public to\xa0address the crisis, palm oil companies and consumer goods firms alike have in recent years made ‘zero deforestation’ promises, which prohibit palm oil cultivation on forest land, peat soil, a
nd ban the use of fire to clear land."""

    e = """{'news_id': '318d6c16bbf4d94b2207f24c913027ba', 'title': 'Consumer goods giants under fire for poor palm oil record', 'content': 'Italian confectionery maker Ferrero is leading the consumer goods industry on sustainable palm oil procurement w
hile household names Colgate-Palmolive, Johnson&Johnson, and PepsiCo are the biggest laggards, according to a new scorecard by Greenpeace\xa0International.\nDeforestation in Borneo to make way for palm oil plantations. Several palm oil grower
s and consumer goods companies have made no-deforestation pledges in recent years. Image: Rich Carey / Shutterstock.com\nBy\nVaidehi Shah\n4 minute read\nMarch 6, 2016\nGreenpeace International on Thursday released a new report slamming consu
mer goods giants such as Colgate-Palmolive, Johnson&Johnson, and PepsiCo for failing to cut deforestation out of their palm oil supply chain.\nThe scorecard ranked\xa0the\xa014 global companies\xa0which have made no-deforestation promises in
recent years\xa0on their performance in three key areas: responsible sourcing of palm oil; transparency about their supply chain; and their support for wider\xa0industry\xa0reform.\nThe 14 corporations were:\xa0Colgate-Palmolive, Danone, Ferr
ero, General Mills, Ikea, Johnson & Johnson, Kellogg, Mars, Mondelez,\xa0Nestle, Orkla, PepsiCo, P&G and Unilever.\nGreenpeace\xa0found that none of the firms could confirm that there is no deforestation in their supply chain despite promisin
g so, nor had a single company\xa0published a full list of its palm oil suppliers. Most\xa0of them\xa0also\xa0did\xa0not have any external verification to prove that their palm oil is sustainable.\nAnnisa Rahmawati, forests campaigner, Greenp
eace Indonesia, said in a statement that “palm oil is found in so many products, which is why brands have a responsibility to their customers to act”.\nThe ubiquitous commodity is used\xa0in goods ranging from food to soaps and lotions to cos
metics, but the industry’s expansion has been widely blamed for accelerating deforestation in countries like Indonesia and Malaysia.\nIn Indonesia, smallholder farmers often burn land to make way for oil palm plantations. This practice is che
aper than using bulldozers and other equipment to properly clear land, and\xa0results in an annual smog which plagues the archipelago nation and its Southeast Asian neighbours every year.\nGreenpeace’s scorecard on consumer goods companies ze
ro-deforestation policies. Image: Greenpeace\nUnder fierce pressure from environmental groups, affected communities, and members of the public to\xa0address the crisis, palm oil companies and consumer goods firms alike have in recent years ma
de ‘zero deforestation’ promises, which prohibit palm oil cultivation on forest land, peat soil, and ban the use of fire to clear land.\nThe Roundtable on Sustainable Palm Oil (RSPO), an industry association, has\xa0a certification scheme whi
ch certifies palm oil grown in an environmentally and socially responsible way. This oil, which is sold at a premium price, accounts for 21 per cent of the global palm oil supply.\n“Palm oil can be grown responsibly without destroying forests
, harming local communities or threatening orangutans,” said Rahmawati. “But our survey shows that brands are not doing enough to stop the palm oil industry ransacking Indonesia’s rainforests.”\nLess GreenPalm, more transparency\nItalian\xa0c
onfectionery\xa0maker Ferrero, which topped the scorecard, is the only firm whose palm oil is fully traceable back to the plantation where it was grown. This is an important requirement for confirming the commodity is deforestation free.\nFer
rero also scored well for its membership in the Palm Oil Innovation Group, which\xa0advocates\xa0greater transparency and higher sustainability standards in the sector than those set by RSPO.\nThe worst-ranked companies were personal care pro
duct giants Colgate-Palmolive and Johnson&Johnson, along with food and beverage behemoth PepsiCo. All three are American firms\xa0and announced zero-deforestation policies in 2014.\nGreenpeace noted that rather than buying actual certified su
stainable palm oil, Colgate-Palmolive and PepsiCo rely on GreenPalm for their sustainability credentials.\xa0GreenPalm is an independent programme operated by\xa0Book&Claim Limited, a United Kingdom-headquartered company.\nThis is the palm oi
l equivalent of purchasing carbon credits, where companies do not purchase actual sustainable palm oil, but\xa0pay for certificates which allow them to claim sustainability credentials. The money from certificate sales goes to support sustain
able palm oil cultivation.\nInstead of using GreenPalm, “companies need to start\xa0getting physical certified palm oil to a high standard, beyond RSPO”, said Rahmawati.\n“We understand this does not happen overnight, but there is a clear\xa0
distinction between companies that are moving in that direction and the\xa0ones that remain focused on using GreenPalm.”\nShe also pointed out that even though sustainable palm oil is only a fifth of the world’s supply, RSPO has said that “su
pply currently outstrips demand”, suggesting that this lack\xa0of demand is the real problem.\nAll three companies have also not revealed their full supplier lists - something no other company on the scorecard has done -\xa0and their ability
to trace palm oil back to the mill is lower than that of their competitors, said Greenpeace.\nWhen contacted for responses to the index,\xa0Johnson&Johnson told Eco-Business that “we are implementing programmes\xa0across the world to limit ou
r footprint and environmental impact”.\nSteps taken include establishing its Responsible Palm Oil Sourcing Criteria in 2014,\xa0which prohibits development on high carbon stock forests, peatlands, and burning land to clear it.\nJohnson&Johnso
n also works with non-profit consultancy\xa0The Forest Trust to monitor its supply chain for compliance, said the company, adding that it has removed one supplier for violating its standards, and will continue to monitor its supply chain.\nIn
 a separate statement, Colgate-Palmolive\xa0said\xa0that “we’re proud of our goals to fight deforestation and our progress towards them, including working with The Forest Trust and the Roundtable on Sustainable Palm Oil (RSPO)”.\nThe certifie
d oils and GreenPalm certificates the company has bought have contributed nearly US$8 million to support sustainable palm oil production since 2013, said the company.\nIt added that it aims to have a fully deforestation-free palm oil supply c
hain within four years, and will work with Greenpeace and others on tracing back its supply to the mill.\nWith another burning\xa0season looking likely this year, Greenpeace called on consumer companies and growers alike to speed up their eff
orts to improve the industry’s transparency and eliminate deforestation from the supply chain, as “the situation is critical for Indonesia’s forests”.\nRahmawati said: “This is the moment that the industry needs to work together to prevent th
e\xa0greatest environmental crime of the century from happening again.”\nDid you find this article useful? Join the EB Circle!\nYour support helps keep our journalism independent and our content free for everyone to read. Join our community h
ere.\nFind out more and join the EB Circle →\nFind out more and join the EB Circle →\nRelated to this storyTopicsFood & AgricultureRegionsGlobalIndonesiaTagsconsumer productsdeforestationforestshazepalm oilsupply chain', 'topics': ['Greenwash
ing'], 'mains': {'detail': {'Greenwashing': [{'probability': 0.9529739022254944, 'text': ' Greenpeace’s scorecard on consumer goods companies zero-deforestation policies.  Image: Greenpeace Under fierce pressure from environmental groups, aff
ected communities, and members of the public to\xa0address the crisis, palm oil companies and consumer goods firms alike have in recent years made ‘zero deforestation’ promises, which prohibit palm oil cultivation on forest land, peat soil, a
nd ban the use of fire to clear land.'}, {'probability': 0.8576178550720215, 'text': ' Image: Greenpeace Under fierce pressure from environmental groups, affected communities, and members of the public to\xa0address the crisis, palm oil compa
nies and consumer goods firms alike have in recent years made ‘zero deforestation’ promises, which prohibit palm oil cultivation on forest land, peat soil, and ban the use of fire to clear land.  The Roundtable on Sustainable Palm Oil (RSPO),
 an industry association, has\xa0a certification scheme which certifies palm oil grown in an environmentally and socially responsible way.'}]}, 'status': 0}}"""
    e = ast.literal_eval(e)
    res_e = []
    cut_sen_len = 4
    class_list = ["other"]
    for json_data in [e]:
        para = json_data.get("title", "") + "\n" + json_data.get("content", "")
        sentences = para_sentences(para)
        for i in range(0, len(sentences), max(1, int(cut_sen_len / 2))):
            text = " ".join(sentences[i:i + cut_sen_len])  # 每steps 句拼接为一个text，进一次模型
            text = text.replace("\n", " ")
            print(text)
            if len(text.strip()) <= 0:
                continue
            res_e.append({"text": text, "label": class_list[0], "news_id": json_data["news_id"]})
    # sentences = pre_cut_sentences([e], )
    # print(len(sentences))
    print(res_e)
