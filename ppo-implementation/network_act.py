"""
    This file contains a neural network module for us to
    define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        attn_weights = torch.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        return attn_weights @ V

class FeedForwardNN(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim, discrete, use_attention):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        # super(FeedForwardNN, self).__init__()
        super().__init__()
        
        self.discrete = discrete
        self.use_attention = use_attention
        self.obs_shape = in_dim if isinstance(in_dim, tuple) else (in_dim,)

        if len(self.obs_shape) == 1:  # Flat vector input
            self.encoder = nn.Identity()
            self.hidden_dim = self.obs_shape[0]
        else:  # Image input: use CNN
            c, h, w = self.obs_shape
            self.encoder = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.SiLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.SiLU(),
                nn.Flatten()
            )
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                self.hidden_dim = self.encoder(dummy).shape[1]
      
        self.linear1 = nn.Linear(self.hidden_dim, 256)
        self.norm1 = nn.LayerNorm(256)

        if self.use_attention:
            self.attn = SelfAttention(256)

        self.linear2 = nn.Linear(256, 256)
        self.norm2 = nn.LayerNorm(256)

        self.output = nn.Linear(256, out_dim)


    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        if len(self.obs_shape) == 3 and obs.ndim == 3:
            obs = obs.unsqueeze(0)

        if len(self.obs_shape) == 3:
            obs = obs / 255.0
            if obs.shape[-1] == 3 and obs.shape[1] != 3:
                obs = obs.permute(0, 3, 1, 2)

        x = self.encoder(obs)
        x = self.norm1(F.silu(self.linear1(x)))

        if self.use_attention:
            x = x.unsqueeze(1)                   # (B, 1, 256)
            x, _ = self.attn(x, x, x)            # Self-attention
            x = x.squeeze(1)                     # (B, 256)

        residual = x
        x = self.norm2(F.silu(self.linear2(x)))
        x = x + residual

        return self.output(x)