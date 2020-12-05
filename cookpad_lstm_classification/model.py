import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, embeded_sentence):
        _, lstm_out = self.lstm(embeded_sentence)
        tag_space = self.hidden2tag(lstm_out[0].view(-1, self.hidden_dim))
        tag_scores = self.softmax(tag_space)
        return tag_scores
