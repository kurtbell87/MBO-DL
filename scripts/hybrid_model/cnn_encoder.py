"""CNN encoder for book snapshot spatial features.

Architecture from spec:
  Permute -> (B, 2, 20)
  -> Conv1d(2, 32, k=3, pad=1) -> ReLU
  -> Conv1d(32, 64, k=3, pad=1) -> ReLU
  -> AdaptiveAvgPool1d(1) -> (B, 64)
  -> Linear(64, 16) -> (B, 16)
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 16)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 2, 20)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)          # (B, 64, 1)
        x = x.squeeze(-1)         # (B, 64)
        x = self.fc(x)            # (B, 16)
        return x
