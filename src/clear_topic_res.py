import ast
import os

from tqdm import tqdm


def load_datas(file_path):
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            lines.append(line)
    res = []
    for line in tqdm(lines, total=len(lines), desc=f"load success {len(lines)}, restore data ing "):
        res.append(ast.literal_eval(line))
    return res


def out_data_to_jsonl(datas, file_path):
    with open(file_path, 'w') as f:
        for e in datas:
            f.write(str(e) + "\n")
    print(f"write len :{len(datas)} success to {file_path}!")


def clean(_company):
    new_company = str(_company).replace("\\ u3000", "").replace("\\ u3000公司", "").replace(
        "\\ n", "").replace("n \\ u3000 \\ u3000", "").replace("u3000", "").replace("\\", "")
    if new_company != _company:
        print(new_company, "-----", _company)
    if len(new_company.strip()) > 0 and (
            len(new_company) == 1 and new_company.isalnum()) and new_company.strip() not in ["公司", "公", "2017", "2016",
                                                                                             "齐鲁石化公司淄博万昌集团有限公司山东齐隆化工股份山东宏信化工股份淄博齐翔石油化工集团山东清源集团有限公司山东联合化工股份齐旺达集团山东海力化工股份山东金诚石化集团淄博鲁华泓锦化工股份淄博德信联邦化学工业山东迅达化工集团山东东岳集团山东大成集团淄博德信联邦化学工业淄博中轩集团蓝帆化工集团山东东大化工集 齐鲁石化公司"]:
        return new_company


topic_res_dir = "/nas/miotech-science/leonzheng/data_backfill/2018_2019_emain_ner/"
topic_res_clean_out_dir = "/nas/miotech-science/leonzheng/data_backfill/2018_2019_emain_ner_clean/"
for res_file in tqdm(os.listdir(topic_res_dir)):
    topic_res = load_datas(os.path.join(topic_res_dir, res_file))
    clean_end_datas = []
    for e in topic_res:
        mains = e["mains"]
        for _topic, companys in mains.items():
            new_companys = []
            for k in companys:
                k = clean(k)
                if k is not None:
                    new_companys.append(k)
            mains[_topic] = new_companys
        hk_main_entity_raw = e["hk_main_entity_raw"]
        for _topic, companys_dict in hk_main_entity_raw["detail"].copy().items():
            new_company_dict = {}
            for k, v in companys_dict.items():
                k = clean(k)
                if k is not None:
                    new_company_dict[k] = v
            hk_main_entity_raw["detail"][_topic] = new_company_dict
        clean_end_datas.append(e)
    out_data_to_jsonl(clean_end_datas, os.path.join(topic_res_clean_out_dir, res_file))
