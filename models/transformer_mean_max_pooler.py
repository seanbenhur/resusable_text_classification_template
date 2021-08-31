import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class Mean_Max_Pooling_Model(nn.Module):
    """First find mean and max-pooling embeddings, then
    concatenate this to have a final representation that is twice the hidden size"""

    def __init__(self, path, dropout, num_labels):
        super().__init__()
        self.model = AutoModel.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat(
            (mean_pooling_embeddings, max_pooling_embeddings), 1
        )
        outputs = self.dropout(mean_max_embeddings)
        outputs = self.linear(outputs)
        return outputs
