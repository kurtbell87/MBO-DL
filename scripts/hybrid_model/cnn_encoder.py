"""CNN Encoder for book snapshot spatial features.

Architecture from spec with BatchNorm (required for convergence, per R3):
  Conv1d(2→32, k=3) → BN → ReLU → Conv1d(32→64, k=3) → BN → ReLU → Pool → Linear(64→16)

Input:  (B, 2, 20) — 2 channels (price_offset, size), 20 book levels
Output: (B, 16)    — 16-dim embedding
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=2, embed_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 2, 20)
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 32, 20)
        x = self.relu(self.bn2(self.conv2(x)))   # (B, 64, 20)
        x = self.pool(x).squeeze(-1)             # (B, 64)
        x = self.fc(x)                           # (B, 16)
        return x
