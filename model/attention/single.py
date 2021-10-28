import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)
                              )/math.sqrt(query.size(-1))

        if(not (mask == None)):
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if(not (dropout == None)):
            p_attn = dropout(p_attn)

        # No point in returning p_attn
        return torch.matmul(p_attn, value), p_attn
