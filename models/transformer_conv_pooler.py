import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class Conv_Pooling_Model(nn.Module):
    def __init__(self, path, dropout, num_labels):
        super().__init__()
        self.model = AutoModel.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.dropout = nn.Dropout(dropout)
        self.cnn_1 = nn.Conv1d(self.config.hidden_size, 256, kernel_size=2, padding=1)
        self.cnn_2 = nn.Conv1d(256, num_labels, kernel_size=2, padding=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        cnn_embeddings = F.relu(self.cnn_1(last_hidden_state))
        cnn_embeddings = self.cnn_2(cnn_embeddings)
        logits, _ = torch.max(cnn_embeddings, 2)
        return logits
