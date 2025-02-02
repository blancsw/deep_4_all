# 03-classifier-embedding.ipynb

````pyton
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        # x : [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embed_dim]
        pooled = embedded.mean(dim=1)  # Moyenne sur la dimension de embedding layer
        out = self.fc(pooled)  # [batch_size, num_class]
        return out
````