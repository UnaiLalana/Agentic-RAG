import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts.iloc[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long)
        }

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        dropout: float = 0.3,
        pooling: str = "cls",
        freeze_encoder: bool = False,
    ):
        super(TransformerClassifier, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

        hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.pooling == "cls":
            pooled_output = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled_output = summed / summed_mask
        else:
            raise ValueError("Pooling must be 'cls' or 'mean'")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits