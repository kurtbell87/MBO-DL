"""Stage 1: CNN encoder training on forward return regression."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .cnn_encoder import CNNWithHead, CNNClassifier
from .data_loader import BookDataset


def set_seed(seed=config.CNN_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_cnn_regression(train_book, train_targets, val_book, val_targets,
                         embedding_dim=config.CNN_EMBEDDING_DIM,
                         lr=config.CNN_LR,
                         weight_decay=config.CNN_WEIGHT_DECAY,
                         batch_size=config.CNN_BATCH_SIZE,
                         max_epochs=config.CNN_MAX_EPOCHS,
                         patience=config.CNN_PATIENCE,
                         seed=config.CNN_SEED):
    """Train CNN encoder with regression head on forward returns.

    Args:
        train_book: (N_train, 2, 20) training book arrays
        train_targets: (N_train,) forward returns
        val_book: (N_val, 2, 20) validation book arrays
        val_targets: (N_val,) forward returns
        ... hyperparameters from config

    Returns:
        model: trained CNNWithHead
        history: dict with train_loss, val_loss per epoch
    """
    set_seed(seed)

    train_ds = BookDataset(train_book, train_targets)
    val_ds = BookDataset(val_book, val_targets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CNNWithHead(embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_losses = []
        for book_batch, target_batch in train_loader:
            optimizer.zero_grad()
            pred = model(book_batch)
            loss = criterion(pred, target_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for book_batch, target_batch in val_loader:
                pred = model(book_batch)
                loss = criterion(pred, target_batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def train_cnn_classifier(train_book, train_labels, val_book, val_labels,
                         embedding_dim=config.CNN_EMBEDDING_DIM,
                         lr=config.CNN_LR,
                         weight_decay=config.CNN_WEIGHT_DECAY,
                         batch_size=config.CNN_BATCH_SIZE,
                         max_epochs=config.CNN_MAX_EPOCHS,
                         patience=config.CNN_PATIENCE,
                         seed=config.CNN_SEED):
    """Train CNN with classification head for CNN-only ablation baseline.

    Labels are mapped: {-1 -> 0, 0 -> 1, +1 -> 2} for cross-entropy.
    """
    set_seed(seed)

    # Map labels to 0,1,2
    label_map = {-1: 0, 0: 1, 1: 2}
    train_mapped = np.array([label_map[int(l)] for l in train_labels])
    val_mapped = np.array([label_map[int(l)] for l in val_labels])

    train_ds = BookDataset(train_book, train_mapped.astype(np.float32))
    val_ds = BookDataset(val_book, val_mapped.astype(np.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CNNClassifier(embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for book_batch, label_batch in train_loader:
            optimizer.zero_grad()
            logits = model(book_batch)
            loss = criterion(logits, label_batch.long())
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for book_batch, label_batch in val_loader:
                logits = model(book_batch)
                loss = criterion(logits, label_batch.long())
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
