import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModel


class Transformer(nn.Module):
    def __init__(self, path, dropout, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        _, outputs = self.bert(input_ids, attention_mask, return_dict=False)
        outputs = self.dropout(outputs)
        outputs = self.linear(outputs)
        return outputs
