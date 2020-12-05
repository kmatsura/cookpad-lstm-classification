import re
import torch


def sub_number_and_symbol(sentence):
    # 半角全角英数字除去
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # 記号もろもろ除去
    sentence = re.sub(
        r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□☸♥°♬(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    return sentence


def get_category_index(df):
    category2index = {}
    for cat in df["category"]:
        if cat in category2index:
            continue
        else:
            category2index[cat] = len(category2index)
    return category2index


def category2tensor(cat, category_index):
    return torch.tensor([category_index[cat]], dtype=torch.long)
