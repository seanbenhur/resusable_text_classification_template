import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class Attention_Pooling_Model(nn.Module):
    def __init__(self, path, dropout, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.head = AttentionHead(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask)
        x = x["last_hidden_state"]
        x = self.head(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
