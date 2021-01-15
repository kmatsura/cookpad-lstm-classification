import os
import pickle
import torch
from dotenv import load_dotenv
from util import *
from ginza import *
import spacy

def main():
    title = input()
    load_dotenv(verbose=True)
    BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELPATH = os.environ.get("MODELPATH")
    assert os.path.exists(os.path.join(BASEDIR, MODELPATH), "modelpath not found"
    with open(os.path.join(BASEDIR, MODELPATH), 'rb') as f:
        losses, category_index, model = pickle.load(f)
    nlp = spacy.load("ja_ginza")  # GiNZAモデルの読み込み
    
    with torch.no_grad():
        out = model(title_embeded)
        _, predict = torch.max(out, 1)
        answer = category2tensor(cat, category_index)
    print(answer)


if __name__ == "__main__":
    main()