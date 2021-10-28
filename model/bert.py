import torch.nn as nn
from .transformer import TransformerBlock
from .embedding import BERTEmbedding
import numpy
# The way I am looking at #head it seems to be filter in cnns. Would it be a good idea to make sure the dimensions of the heads always equals 3x3


class BERT(nn.Module):

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(
            vocab_size=vocab_size, embed_size=hidden)
#research feedforward expand
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):

        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1)*self.attn_heads*2, 1).unsqueeze(1)
        # Can you help explain this piece

        x = self.embedding(x)

        for transformer in self.transformer_blocks:
            # x = transformer.forward(x, mask)
            x = transformer.forward(x)

        return x
