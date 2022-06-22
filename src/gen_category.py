#encoding=utf-8
import re
import hanlp

import pandas as pd
import jieba
gr_file = pd.read_csv("/GR Taxonomy - GR Taxonomy.csv")
categorys = []
l1_temp_dict = {}
category_l1 = []
l2_temp_dict = {}
category_l2 = []
category_l3 = []

last_l1 = gr_file.Level_1[0]
last_l2 = gr_file.Level_2[0]
for index, (l1,l2,l3) in enumerate(zip(gr_file.Level_1,gr_file.Level_2,gr_file.Level_3)):
    if l1 != last_l1 or index == len(gr_file)-1:
        if len(l2_temp_dict) > 0:
            category_l1.append(l2_temp_dict)
        l2_temp_dict = {}
        category_l3 = []
        l1_temp_dict[last_l1] = category_l1
        categorys.append(l1_temp_dict)
        l1_temp_dict = {}
        category_l1 = []
        last_l1 = l1
        last_l2 = l2
    if l2 != last_l2:
        if len(l2_temp_dict) > 0:
            category_l1.append(l2_temp_dict)
        l2_temp_dict = {}
        category_l3 = []
        last_l2 = l2
    if not pd.isnull(l3):
        category_l3.append(l3)
        l2_temp_dict[l2] = category_l3

with open("category.json","w") as f:
    f.writelines(str(categorys))

# sens = "本公司主要业务活动集科学研究、设计、产品制造销售、工程承包、生产运营、煤炭生产销售、技术服务、金融工具支持为一体。为行业进步发展提供技术和服务，为煤炭行业客户解决安全高效绿色智能化开采与清洁高效低碳集约化利用技术问题。"
# sens = "本公司的传统业务为高性能金属基复合材料及制品的研究、开发、生产和销售。2018年3月，康泰威光电通过受让中国科学院上海光学精密机械研究所的热压 ZnS 红外陶瓷生产技术，新增了ZnS光学材料与制品业务。"
sens = "公司主要从事中高档瓦楞纸箱、纸板及缓冲包装材料的研发与设计、生产、销售及服务。公司的瓦楞纸箱以其优越的使用性能和良好的加工性能，逐渐取代了传统的木箱等运输包装容器，成为现代包装运输的主要载体。公司的产品依靠卓越的质量水平及先进的工业设计理念，不仅实现保护商品、便于仓储、装卸运输的功能，还起到美化商品、宣传商品的作用，同时能够减少损耗及包装空间，属于绿色环保产品。"
sens = "公司采用“标准化工厂”的生产模式，已建立一整套标准化的生产流程，包括厂房、生产线、机器设备、仓库的设计和布局以及员工的生产技能培训设计方案，使公司可以在最短的时间内实现布点、建设、投产和生产。“标准化工厂”除有利于快速复制外，也有利于人员培训、有利于总部对工厂管理指导。通过标准化管理，各项生产和管理指标在横向及纵向上进行对比，不断提升标准化水平。"
# TODO 可以通过l3的关键字判断当前的句子可能是属于那些l1分类的

from collections import Counter
all_l3_keys = []
_dict = []
for category in categorys:
    for sen in sens.split("。"):
        for l1, l1_list in category.items():
            # print(l1)
            l3_list = []
            for l2_dict in l1_list:
                for l2, _l3_list in l2_dict.items():
                    try:
                        l3_list.extend([w for w in jieba.lcut("".join(_l3_list)) if len(w) > 1])
                    except:
                        print("")
            # if list(set(l3_list)) not in all_l3_keys:
            all_l3_keys.extend(list(set(l3_list)))
            if len(l3_list) <= 0:
                continue
            pattern = "|".join([f"({k})" for k in list(set(l3_list)) if k not in ['设备', '装备', '高效', '利用', '气体', '设施', '运营', '制造', '建设', '持续', '改造', '绿色', '节能', '系统', '产品', '资源']])
            re_result = re.search(pattern, sen)
            if re_result:
                _dict.append(l1)
                print(l1, re_result.group(0))
            l3_list = []

print(Counter(_dict).items())
# TODO 此处的输出可以大致的判断出可能是那几个l1，从而再去对应的l1中去预测
print([k for k, v in Counter(_dict).items() if v > 1])

        # print(",".join(list(set(l3_list))))
        # print("*"*50)

print([k for k,v in Counter(all_l3_keys).items() if v > 3])

