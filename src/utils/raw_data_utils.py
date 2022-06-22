import os
import re
from typing import List


def gen_identifiers():
    """
    生成所有的段落编号
    :return:
    """
    identifiers = [f"{i}、" for i in range(1, 10)]
    identifiers.extend([f"{i}." for i in range(1, 10)])
    identifiers.extend([f"({i})." for i in range(1, 10)])
    identifiers.extend([f"({i})、" for i in range(1, 10)])
    identifiers.extend([f"（{i}）、" for i in range(1, 10)])
    identifiers.extend([f"（{i}）." for i in range(1, 10)])
    identifiers.extend([f"({i})" for i in "一二三四五六七八九"])
    identifiers.extend([f"（{i}）" for i in "一二三四五六七八九"])
    identifiers.extend([f"({i})、" for i in "一二三四五六七八九"])
    identifiers.extend([f"({i})." for i in "一二三四五六七八九"])
    identifiers.extend([f"（{i}）、" for i in "一二三四五六七八九"])
    identifiers.extend([f"（{i}）." for i in "一二三四五六七八九"])
    identifiers.extend([f"{i}." for i in "一二三四五六七八九"])
    identifiers.extend([f"{i}、" for i in "一二三四五六七八九"])
    identifiers.extend([f"{i} " for i in "一二三四五六七八九"])
    return identifiers


def get_next_identifier(line: str):
    """
    获取当前编号的下一个编号（同级）
    :param line:
    :return:
    """
    current_identifier = get_current_identifier(line)
    if current_identifier is not None:
        identifiers = gen_identifiers()
        current_identifier_index = identifiers.index(current_identifier)
        if current_identifier_index < len(identifiers) - 1:
            return identifiers[current_identifier_index + 1]
    return None


def get_current_identifier(line: str):
    """
    获取当前句子上的编号
    :param line:
    :return:
    """
    identifiers = gen_identifiers()
    for identifier in identifiers:
        if line.lstrip().startswith(identifier):
            return identifier
    return None


def get_smooth_sentences(paragraph: List[str]) -> list:
    """
    获取优化后的段落句子，
    TODO 还有很大的优化空间，可以优化到更精细
    :param paragraph:
    :return:
    """
    identifiers = gen_identifiers()
    i = 0
    smooth_sentences = []
    last_sentences = []
    last_ident = ""
    while i < len(paragraph):
        sentence = paragraph[i]
        flag = False
        for iden_index, identifier in enumerate(identifiers):
            if sentence.lstrip().startswith(identifier):
                # TODO 看看是否可以将多级编号搞出来
                flag = True
                break
        if last_ident != identifier and flag:
            if len(last_sentences) > 0:
                smooth_sentences.append(last_sentences)
            last_ident = identifier
            last_sentences = []
        last_sentences.append(sentence)
        i += 1
    smooth_sentences.append(last_sentences)
    return smooth_sentences


def load_content(txt_path: str) -> list:
    """
    加载文件
    :param txt_path:
    :return:
    """
    assert os.path.exists(txt_path), "txt file not found!"
    with open(txt_path, "r") as f:
        content = f.readlines()
    return content


def get_rough_sentences_by_re(content: list, p_pattern: str, n_pattern: str, match_nums: int = 1) -> list:
    """
    根据正则表达式，获取一个粗糙的段落内容
    :param content: pdf 文章
    :param p_pattern: 符合的正则
    :param n_pattern: 符合p_pattern且需要额外排除的正则
    :param match_nums: 正则匹配成功的次数
    :return:
    """
    out_lines = []
    match_count = 0
    for i, line in enumerate(content):
        line = line.replace(" ", "")
        if re.search(p_pattern, line):
            # 如果设置n_pattern,且re n_pattern 不为空就跳过该条
            if len(n_pattern) > 0 and re.search(re.escape(n_pattern), line) is not None:
                continue
            match_count += 1
            next_ident = get_next_identifier(line)
            # 如果有编号, 通过re找到的句子所在的编号，找到写一个编号是什么，将当前编号到下一个编号之间的内容返回，最多20行
            j = 1
            if next_ident:
                while j < 20:
                    if content[i + j].strip().startswith(next_ident):
                        break
                    if len(content[i + j].strip()) > 0:
                        out_lines.append(content[i + j])
                    j += 1
            else:
                # 如果没有编号，就往下找，最多20行
                while j < 20:
                    #
                    if content[i + j].endswith("。\n") and content[i + j + 1].strip() != "":
                        break
                    # 如果当前的句子以及在out_lines 就 跳出
                    if content[i + j].strip() in out_lines:
                        break
                    if len(content[i + j].strip()) > 0:
                        out_lines.append(content[i + j].strip())
                    j += 1
            # 只返回第一个re找到的结果
            if match_count >= match_nums:
                break
    return out_lines


def get_main_business(content: list) -> list:
    """
    获取主营业务的句子
    :param content: pdf 文章
    :return:
    """
    p_pattern = "(公司(所)*从事的((主要业务)|(业务概要)|(业务情况)))|(公司主要从事)|(公司主要业务)|(公司从事(的)*主要业务)|(报告期内主要的业务情况)"
    # p_pattern = "公司(所)*从事的((主要业务)|(业务概要)|(业务情况))"
    n_pattern = "((具体详见)|(情况详见))"
    main_business_sentences = get_rough_sentences_by_re(content, p_pattern, n_pattern)
    main_business_sentences = re.split("([?？\n!！。])", "".join(main_business_sentences))
    main_business_sentences.append("")
    main_business_sentences = ["".join(i) for i in zip(main_business_sentences[0::2], main_business_sentences[1::2])]
    main_business_sentences = get_smooth_sentences(main_business_sentences)
    return main_business_sentences


if __name__ == "__main__":
    # contents = ['（一）公司主要业务\n',
    #             ' 报告期内，公司主要从事中成药、化学药和女性卫生用品的研制、生产和销售以及药品的批发和零售业务。公司现有片剂、胶囊剂、颗粒剂、丸剂、煎膏剂、散剂和溶液剂等 12 种制剂、22\n',
    #             '条自动化生产线、122 项药品注册批件和 328 项专利技术。公司及控股子公司共有 19 个药品被列\n',
    #             '入《国家基本药物目录（2018 年版）》、41 个药品入选了《国家基本医疗保险、工伤保险和生育保险药品目录（2017 年版）》。公司拥有良好的品牌形象，“千金”品牌在国内妇科用药领域居于领先地位。公司主导产品妇科千金片（胶囊）、补血益母丸（颗粒）是独家拥有的国家中药保护品种、国家基本药物目录品种、国家基本医疗保险目录甲类品种。\n',
    #             ' 目前，公司已初步构建了包括医药制造、医药流通和中药种植在内的医药全产业链业务架构。同时，公司秉承“跳出妇科，做女性健康系列；跳出本业，做中药衍生系列”的发展战略，以妇科中药为核心，逐步向女性大健康产业领域延伸。千金净雅妇科专用棉巾已经成为公司在女性大健康产业领域的明星产品。\n',
    #             ' （二）公司经营模式\n',
    #             ' 公司经营主要包含医药工业和医药商业，医药工业主要生产中成药和西药，医药商业主要包括医药批发及零售。\n',
    #             ' 报告期内，公司经营模式未发生重要变化。详情请参阅公司《2018 年年度报告》。\n',
    #             ' （三）行业情况分析\n',
    #             '  1.医药工业主营业务收入增长平稳\n',
    #             '  公司所处的行业为医药制造业。根据国家统计局数据，2018 年，我国医药制造业总体经济运行平稳。2018 年我国医药制造业主营业务收入 25840 亿元，同比增长 12.7%，增幅高于全国工业平均值 4.2 个百分点。各子行业中，增长最快的是化学制药工业和卫生材料及医药用品工业。\n',
    #             '  图表：2018 年医药工业及子行业营业收入增幅（%）\n',
    #             '行业                                    收入增幅%\n',
    #             '化学制药工业                            16.5\n']
    #
    # sentences = re.split("([?？\n!！。])", "".join(contents))
    # sentences.append("")
    # sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    # result = get_smooth_sentences(sentences)
    # print(result)
    content = load_content("../../tests/tmp_file/601003:柳钢股份2020年年度报告.txt")
    result = get_main_business(content)
    for paragraph in result:
        print("*"*20)
        # print("".join(paragraph).replace("\n",""))
        paragraph = re.split("([?？!！。])", "".join(paragraph))
        paragraph.append("")
        paragraph = ["".join(i) for i in zip(paragraph[0::2], paragraph[1::2])]
        paragraph = [i.strip().replace("\n","") for i in paragraph if len(str(i).strip()) > 0]
        # for i in zip(paragraph[0::1], paragraph[1::1]):
        #     print(i)
        sentences = ["".join(i) for i in zip(paragraph[0::1], paragraph[1::1])]
        print("\n".join(sentences))
        print("*" * 20)
    # print(result)
