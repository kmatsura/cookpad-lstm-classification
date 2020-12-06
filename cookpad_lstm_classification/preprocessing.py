import os
import re
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from ginza import *
import spacy
from util import sub_number_and_symbol
from dotenv import load_dotenv

def main():
    load_dotenv(verbose=True)
    BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATADIR = os.path.join(BASEDIR, 'data/reciepe_category/')
    FILENAME = "recipe_category_datasets.csv"
    assert os.path.exists(os.path.join(DATADIR, FILENAME)
                          ), "input file not found"
    OUTPUTFILENAME = "recipe_category_datasets_preprocessed_pad.pkl"
    datasets = pd.read_csv(os.path.join(DATADIR, FILENAME))
    nlp = spacy.load("ja_ginza")  # GiNZAモデルの読み込み
    word2index = {}  # 単語ID用の辞書
    word2index["<pad>"] = 0  # パディング文字列を追加
    max_len = 0  # パディングのために系列の最大の長さを計算
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
        if max_len < len(tmp):
            max_len = len(tmp)
    VOCAB_SIZE = len(word2index)
    EMBEDDING_DIM = os.environ.get("EMBEDDING_DIM")
    datasets["title_embeded"] = None
    embeds = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=0)
    for i in range(len(datasets)):
        indexed = torch.tensor(
            [word2index[w] for w in datasets["title_divided"][i]], dtype=torch.long)
        for i in range(max_len - len(indexed)):  # padding
            indexed.insert(0, 0)
        sentence_matrix = embeds(indexed)
        datasets["title_embeded"][i] = sentence_matrix.view(
            len(sentence_matrix), 1, -1)
    with open(os.path.join(DATADIR, OUTPUTFILENAME), 'wb') as f:
        pickle.dump(datasets , f)
    with open(os.path.join(DATADIR, "word_index.pkl"), 'wb') as f:
        pickle.dump(word2index , f)

if __name__ == "__main__":
    main()
