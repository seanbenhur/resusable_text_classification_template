import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class Max_Pooling_Model(nn.Module):

    """MaxPooling considers the last hidden state [batch, maxlen, hidden_state],
    then take max across maxlen dimensions to get max pooling embeddings."""

    def __init__(self, path, dropout, num_labels):
        super().__init__()
        self.model = AutoModel.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        last_hidden_state[
            input_mask_expanded == 0
        ] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(last_hidden_state, 1)[0]
        outputs = self.dropout(max_embeddings)
        outputs = self.linear(outputs)
        return outputs
