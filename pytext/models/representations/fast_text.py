
from typing import Optional

import torch
import torch.nn as nn
from pytext.utils.usage import log_class_usage
from .representation_base import RepresentationBase


class FastTextRepresentation(RepresentationBase):

    class Config(RepresentationBase.Config):
        dropout: float = 0.4
        bias: bool = True
    
    def __init__(self, config:Config, embed_dim: int):
        super().__init__(config)
        self.representation_dim = embed_dim
        self.dropout = nn.Dropout(config.dropout)
        self.dense = nn.Linear(embed_dim, embed_dim, config.bias)
        log_class_usage(__class__)


    def forward(self, embedded_tokens: torch.Tensor, seq_lengths: torch.Tensor):
        rep = self.dropout(embedded_tokens)
        rep = torch.sum(rep, 1) / seq_lengths.unsqueeze(1).float()
        return rep
