"""Extract 16-dim embeddings from a frozen CNN encoder."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def extract_embeddings(model, book_arr, batch_size=512):
    """Extract embeddings from a trained CNNWithHead model.

    Args:
        model: trained CNNWithHead (uses model.encoder)
        book_arr: (N, 2, 20) float array

    Returns:
        (N, 16) numpy array of embeddings
    """
    encoder = model.encoder
    encoder.eval()

    book_tensor = torch.tensor(book_arr, dtype=torch.float32)
    ds = TensorDataset(book_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for (batch,) in loader:
            emb = encoder(batch)
            embeddings.append(emb.numpy())

    return np.concatenate(embeddings, axis=0)
