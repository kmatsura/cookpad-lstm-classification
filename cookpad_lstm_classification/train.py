import os
import pickle
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from model import LSTMClassifier
from util import get_category_index, category2tensor


def main():
    BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATADIR = os.path.join(BASEDIR, 'data/reciepe_category/')
    FILENAME = "recipe_category_datasets_preprocessed.pkl"
    assert os.path.exists(os.path.join(DATADIR, FILENAME)
                          ), "data file not found"
    with open(os.path.join(DATADIR, FILENAME), 'rb') as f:
        datasets = pickle.load(f)
    traindata, testdata = train_test_split(datasets, train_size=0.9)
    load_dotenv(verbose=True)
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM"))
    HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM"))
    category_index = get_category_index(datasets)
    tag_size = len(category_index)
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, tag_size)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(
        os.environ.get("LEARNING_RATE")))

    losses = []
    for epoch in range(int(os.environ.get("EPOCHS"))):
        all_loss = 0
        for title_embeded, cat in zip(traindata["title_embeded"], traindata["category"]):
            model.zero_grad()
            out = model(title_embeded)
            answer = category2tensor(cat, category_index)
            loss = loss_function(out, answer)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        losses.append(all_loss)
        print("epoch", epoch, "\t", "loss", all_loss)
    print("done.")
    RESULTDIR = os.path.join(BASEDIR, os.environ.get("RESULTDIR"))
    if not os.path.exists(RESULTDIR):
        os.mkdirs(RESULTDIR)
    now = str(datetime.datetime.now()).strip()
    OUTPUTFILENAME = "trainresult-{}.pkl".format(now)
    with open(os.path.join(RESULTDIR, OUTPUTFILENAME), 'wb') as f:
        pickle.dump((losses, model), f)

    test_num = len(testdata)
    a = 0
    with torch.no_grad():
        for title_embeded, cat in zip(testdata["title_embeded"], testdata["category"]):
            out = model(title_embeded)
            _, predict = torch.max(out, 1)
            answer = category2tensor(cat, category_index)
            if predict == answer:
                a += 1
    print("predict: ", a / test_num)

if __name__ == "__main__":
    main()
