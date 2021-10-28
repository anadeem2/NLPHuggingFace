import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        # Why 3? What does padding do, I don't understand how this seperates sentence 1 & 2
        super().__init__(3, embed_size, padding_idx=0)
        
