"""CNN Encoder — R3-exact architecture (12,128 params).

Architecture (verified by 9C and 9D):
  Conv1d(2→59, k=3, pad=1) → BN(59) → ReLU
  Conv1d(59→59, k=3, pad=1) → BN(59) → ReLU
  AdaptiveAvgPool1d(1) → Linear(59→16) → ReLU → Linear(16→1)

Input:  (B, 2, 20) — 2 channels (price_offset, size), 20 book levels
Embedding: (B, 16)  — extracted from Linear(59→16) + ReLU
Output: (B, 1)      — scalar return prediction

Parameter count breakdown:
  Conv1d(2→59, k=3):  2*59*3 + 59 = 413
  BN(59):             59*2        = 118
  Conv1d(59→59, k=3): 59*59*3 + 59 = 10,502
  BN(59):             59*2        = 118
  Linear(59→16):      59*16 + 16  = 960
  Linear(16→1):       16*1 + 1    = 17
  Total:                            12,128
"""

import torch
import torch.nn as nn


class CNNEncoderR3(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=59, embed_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Linear(hidden_channels, embed_dim)
        self.fc_out = nn.Linear(embed_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 2, 20)
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 59, 20)
        x = self.relu(self.bn2(self.conv2(x)))   # (B, 59, 20)
        x = self.pool(x).squeeze(-1)             # (B, 59)
        x = self.relu(self.fc_embed(x))          # (B, 16) — embedding extraction point
        x = self.fc_out(x)                       # (B, 1)
        return x

    def embed(self, x):
        """Extract 16-dim embedding (before regression head)."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc_embed(x))
        return x
