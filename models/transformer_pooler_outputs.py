import torch.nn as nn
from transformers import AutoConfig, AutoModel


class Transformer_Pooler_Outputs(nn.Module):
    def __init__(self, path, dropout, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooler_output = outputs[1]
        outputs = self.dropout(pooler_output)
        outputs = self.linear(outputs)
        return outputs
