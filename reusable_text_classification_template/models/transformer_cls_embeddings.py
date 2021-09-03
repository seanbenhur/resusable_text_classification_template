import torch.nn as nn
from transformers import AutoConfig, AutoModel


class Transformer_CLS_Embeddings(nn.Module):
    def __init__(self, path, dropout, num_labels):
        super().__init__()
        self.model = AutoModel.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        cls_embeddings = last_hidden_state[:, 0]
        outputs = self.dropout(cls_embeddings)
        outputs = self.linear(outputs)
        return outputs
