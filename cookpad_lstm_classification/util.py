import torch


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