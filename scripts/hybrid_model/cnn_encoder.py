"""PyTorch CNN encoder module (spec Architecture section)."""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Conv1d encoder: (B, 2, 20) -> (B, 16).

    Architecture from spec:
        Permute -> (B, 2, 20)
        Conv1d(2, 32, k=3, pad=1) -> ReLU
        Conv1d(32, 64, k=3, pad=1) -> ReLU
        AdaptiveAvgPool1d(1) -> (B, 64)
        Linear(64, 16) -> (B, 16)

    Total params: ~7.5k
    """

    def __init__(self, embedding_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (B, 2, 20) book snapshot tensor

        Returns:
            (B, embedding_dim) embedding
        """
        # Input is already (B, 2, 20) — no permute needed
        x = self.relu(self.conv1(x))   # (B, 32, 20)
        x = self.relu(self.conv2(x))   # (B, 64, 20)
        x = self.pool(x)               # (B, 64, 1)
        x = x.squeeze(-1)              # (B, 64)
        x = self.fc(x)                 # (B, embedding_dim)
        return x


class CNNWithHead(nn.Module):
    """CNN encoder + linear regression head for Stage 1 training.

    The regression head is Linear(16, 1) predicting fwd_return_h.
    After training, discard the head and use only the encoder.
    """

    def __init__(self, embedding_dim=16):
        super().__init__()
        self.encoder = CNNEncoder(embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, 2, 20)

        Returns:
            (B,) predictions
        """
        emb = self.encoder(x)
        return self.head(emb).squeeze(-1)


class CNNClassifier(nn.Module):
    """CNN encoder + classification head for CNN-only ablation baseline.

    Direct classification on TB labels (no XGBoost, no non-spatial features).
    """

    def __init__(self, embedding_dim=16, num_classes=3):
        super().__init__()
        self.encoder = CNNEncoder(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        emb = self.encoder(x)
        return self.head(emb)
