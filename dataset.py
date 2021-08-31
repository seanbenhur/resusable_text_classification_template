import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text, label, tokenizer, train=True, max_len=512):

        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.train = train
        self.max_len = max_len

    def __len__(self):
        """Returns the length of the text"""
        return len(self.text)

    def __getitem__(self, idx):

        batch_text = str(self.text[idx])
        inputs = self.tokenizer(
            batch_text, max_length=self.max_len, padding="max_length", truncation=True
        )

        if self.train:
            ids = inputs["input_ids"]
            mask = inputs["attention_mask"]
            label = self.label[idx]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.float),
            }

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }
