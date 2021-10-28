import torch.nn as nn
import torch
import math

# You will need to review this class with me, as I have no idea on how positional embedding work so this didn't make much sense


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, maxLen=512):
        super().__init__()

        pe = torch.zeros(maxLen, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, maxLen).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0)/d_model)).exp()  # what is div_term

        pe[:, 0::2] = torch.sin(position * div_term)  # need learn
        pe[:, 1::2] = torch.cos(position * div_term)  # need learn

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # what is this doing
