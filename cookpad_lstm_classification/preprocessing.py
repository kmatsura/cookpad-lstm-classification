import os
import re
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from ginza import *
import spacy


def sub_number_and_symbol(sentence):
    # 半角全角英数字除去
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # 記号もろもろ除去
    sentence = re.sub(
        r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□☸♥°♬(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    return sentence


def main():
    BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATADIR = os.path.join(BASEDIR, 'data/reciepe_category/')
    FILENAME = "recipe_category_datasets.csv"
    assert os.path.exists(os.path.join(DATADIR, FILENAME)
                          ), "input file not found"
    OUTPUTFILENAME = "recipe_category_datasets_preprocessed.pkl"
    datasets = pd.read_csv(os.path.join(DATADIR, FILENAME))
    nlp = spacy.load("ja_ginza")  # GiNZAモデルの読み込み
    word2index = {}  # 単語ID用の辞書
    datasets["title_divided"] = None  # init
    for i in range(len(datasets)):
        tmp = []
        sentence = datasets["title"][i]
        sentence = sub_number_and_symbol(sentence)
        doc = nlp(sentence)
        for sent in doc.sents:
            for token in sent:
                tmp.append(token.lemma_)
                if token.lemma_ in word2index:
                    continue
                else:
                    word2index[token.lemma_] = len(word2index)
        datasets["title_divided"][i] = tmp.copy()
    VOCAB_SIZE = len(word2index)
    EMBEDDING_DIM = 10
    datasets["title_embeded"] = None
    embeds = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    for i in range(len(datasets)):
        indexed = torch.tensor(
            [word2index[w] for w in datasets["title_divided"][i]], dtype=torch.long)
        sentence_matrix = embeds(indexed)
        datasets["title_embeded"][i] = sentence_matrix.view(
            len(sentence_matrix), 1, -1)
    with open(os.path.join(DATADIR, OUTPUTFILENAME), 'wb') as f:
        pickle.dump(datasets , f)

if __name__ == "__main__":
    main()
