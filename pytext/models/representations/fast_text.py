
from typing import Optional

import torch
import torch.nn as nn
from pytext.utils.usage import log_class_usage
from .representation_base import RepresentationBase


class FastTextRepresentation(RepresentationBase):

    class Config(RepresentationBase.Config):
        dropout: float = 0.4
        bias: bool = True
        in_dim: Optional[int] = None
        out_dim: Optional[int] = None                
    
    def __init__(self, config:Config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.dense = nn.Linear(config.in_dim, config.out_dim, config.bias)
        log_class_usage(__class__)

    def forward(self, embedded_tokens: torch.Tensor, seq_lengths: torch.Tensor):
        rep = self.dropout(embedded_tokens)
        rep = torch.sum(rep, 1) / seq_lengths.unsqueeze(1).float()
        return rep
