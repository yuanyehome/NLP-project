import torch
import torch.nn as nn


# 词向量嵌入
class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embedding.weight.requires_grad = True

    def forward(self, inputs):
        outputs = self.embedding(inputs)
        return outputs
