'''
Part of the code from https://github.com/wangxiang1230/OadTR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import (LearnedPositionalEncoding, FixedPositionalEncoding)
from .attention import SelfAttention


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TemporalTransformer(nn.Module):
    def __init__(self, config, type, positional_encoding="fixed"):
        super().__init__()
        if type == "rgb":
            assert config.data.RGB_features == "DINOv2", "Only DINOv2 features (best) for the video experiment"
            self.embedding_dim = 1536  # DINOv2 features
        elif type == "depth":
            assert config.data.depth_features == "DINOv2", "Only DINOv2 features (best) for the video experiment"
            self.embedding_dim = 1536  # DINOv2 features
        else:
            raise ValueError("Type must be 'rgb' or 'depth'")

        
        
        self.num_heads = 8 # 8 # OadTR uses 8 heads
        self.num_layers = 3 #3 # OadTR uses 3 layers
        assert self.embedding_dim % self.num_heads == 0, 'embedding_dim must be divisible by num_heads'
    
        self.hidden_dim = self.embedding_dim * 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        if positional_encoding == 'learned':
            length = config.data.input_frames + 1 # +1 for cls token
            self.positional_encoding = LearnedPositionalEncoding(length, self.embedding_dim, length)
        elif positional_encoding == 'fixed':
            self.positional_encoding = FixedPositionalEncoding(self.embedding_dim)
        else:
            self.positional_encoding = None

        self.pe_dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)

        self.encoder = TransformerModel(self.embedding_dim, self.num_layers,
                                        self.num_heads, self.hidden_dim, 0.1)

    def forward(self, x):
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # Class token is at the end
            x = torch.cat((x, cls_tokens), dim=1) # (B, S, 1536)
        
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
            x = self.pe_dropout(x)
        
        x = self.encoder(x)
        x = self.dropout(x)

        if self.cls_token is not None:
            x = x[:, -1] # (B,1536)
        else:
            x = x.mean(dim=1)

        return x