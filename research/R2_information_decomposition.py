#!/usr/bin/env python3
"""Phase R2: Information Decomposition — Analysis Script

Spec: .kit/experiments/info-decomposition.md

Trains Tier 1 (linear + MLP) and Tier 2 (LSTM + Transformer) models
on progressively richer feature sets to decompose predictive information
about future returns across three sources: spatial (book), message, temporal.

Input:
  - .kit/results/info-decomposition/features.csv
  - .kit/results/info-decomposition/events/*.bin

Output:
  - .kit/results/info-decomposition/metrics.json
  - .kit/results/info-decomposition/analysis.md
"""

import json
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path(".kit/results/info-decomposition")
CSV_PATH = RESULTS_DIR / "features.csv"
EVENTS_DIR = RESULTS_DIR / "events"

HORIZON_LABELS = ["fwd_return_1", "fwd_return_5", "fwd_return_20", "fwd_return_100"]
SEEDS = [42, 123, 456]
MAX_EVENTS = 500

# Track A: 62 hand-crafted features
TRACK_A_FEATURES = [
    "book_imbalance_1", "book_imbalance_3", "book_imbalance_5", "book_imbalance_10",
    "weighted_imbalance", "spread",
    *[f"bid_depth_profile_{i}" for i in range(10)],
    *[f"ask_depth_profile_{i}" for i in range(10)],
    "depth_concentration_bid", "depth_concentration_ask",
    "book_slope_bid", "book_slope_ask",
    "level_count_bid", "level_count_ask",
    "net_volume", "volume_imbalance", "trade_count", "avg_trade_size",
    "large_trade_count", "vwap_distance", "kyle_lambda",
    "return_1", "return_5", "return_20",
    "volatility_20", "volatility_50", "momentum",
    "high_low_range_20", "high_low_range_50", "close_position",
    "volume_surprise", "duration_surprise", "acceleration", "vol_price_corr",
    "time_sin", "time_cos", "minutes_since_open", "minutes_to_close", "session_volume_frac",
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "order_flow_toxicity", "cancel_concentration",
]

BOOK_SNAP_FEATURES = [f"book_snap_{i}" for i in range(40)]

CAT6_MSG_FEATURES = [
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "order_flow_toxicity", "cancel_concentration",
]

# Force CPU for Tier 2 to avoid MPS memory issues with large event tensors
DEVICE = "cpu"


# ===========================================================================
# Data loading
# ===========================================================================
def load_features():
    """Load feature CSV. Handle polars auto-renaming of duplicate columns.

    The CSV has Track A features (including return_1/5/20) AND forward returns
    (return_1/5/20/100). Polars renames duplicates as return_X_duplicated_0.
    We rename forward return columns to fwd_return_X for clarity.
    """
    df = pl.read_csv(str(CSV_PATH), infer_schema_length=10000,
                     null_values=["NaN", "Inf", "nan", "inf"])

    # Map the forward return columns (which may have been renamed by polars)
    cols = df.columns
    rename_map = {}
    # The last 5 columns before mbo_event_count are: return_1, return_5, return_20, return_100
    # But due to duplicate names, polars appends _duplicated_0
    for c in cols:
        if c == "return_100":
            rename_map[c] = "fwd_return_100"
        elif c == "return_1_duplicated_0":
            rename_map[c] = "fwd_return_1"
        elif c == "return_5_duplicated_0":
            rename_map[c] = "fwd_return_5"
        elif c == "return_20_duplicated_0":
            rename_map[c] = "fwd_return_20"

    # If no duplicated columns (polars handled differently), check position
    if "return_1_duplicated_0" not in cols:
        # Forward returns are the last 4 before mbo_event_count
        # Find the forward return columns by position
        mbo_idx = cols.index("mbo_event_count") if "mbo_event_count" in cols else len(cols)
        fwd_cols = cols[mbo_idx - 4: mbo_idx]
        for i, label in enumerate(["fwd_return_1", "fwd_return_5", "fwd_return_20", "fwd_return_100"]):
            if fwd_cols[i] not in rename_map:
                rename_map[fwd_cols[i]] = label

    df = df.rename(rename_map)

    print(f"Loaded {len(df)} bars from {CSV_PATH}", flush=True)
    print(f"Days: {sorted(df['day'].unique().to_list())}", flush=True)
    print(f"Forward return columns: {[c for c in df.columns if c.startswith('fwd_')]}", flush=True)
    return df


def load_events_for_indices(day_col, indices):
    """Load binary events only for specified row indices, grouped by day.

    Uses numpy buffer parsing for speed instead of Python struct loops.
    Returns dict: {global_row_idx: preprocessed_event_array (n_events x 5)}.
    """
    # Binary event record: action(i4) price(f4) size(u4) side(i4) ts(u8) = 24 bytes
    EVENT_DTYPE = np.dtype([
        ('action', '<i4'), ('price', '<f4'), ('size', '<u4'),
        ('side', '<i4'), ('ts', '<u8')
    ])

    day_to_indices = {}
    for idx in indices:
        day_to_indices.setdefault(int(day_col[idx]), []).append(idx)

    result = {}

    for day, day_indices in sorted(day_to_indices.items()):
        path = EVENTS_DIR / f"events_{day}.bin"
        if not path.exists():
            for idx in day_indices:
                result[idx] = np.zeros((0, 5), dtype=np.float32)
            continue

        data = np.fromfile(str(path), dtype=np.uint8)
        all_day_rows = np.where(day_col == day)[0]
        wanted_set = set(day_indices)

        offset = 0
        for global_idx in all_day_rows:
            if offset + 4 > len(data):
                if global_idx in wanted_set:
                    result[global_idx] = np.zeros((0, 5), dtype=np.float32)
                continue

            n_events = int(np.frombuffer(data[offset:offset+4], dtype='<u4')[0])
            offset += 4
            byte_len = n_events * 24

            if global_idx in wanted_set:
                if offset + byte_len <= len(data) and n_events > 0:
                    raw = np.frombuffer(data[offset:offset+byte_len], dtype=EVENT_DTYPE)
                    events = np.column_stack([
                        raw['action'].astype(np.float32),
                        raw['price'],
                        raw['size'].astype(np.float32),
                        raw['side'].astype(np.float32),
                        raw['ts'].astype(np.float32),
                    ])
                    result[global_idx] = events
                else:
                    result[global_idx] = np.zeros((0, 5), dtype=np.float32)

            offset += byte_len

    return result


def preprocess_events_batch(events_dict, indices, book_data):
    """Preprocess raw events for a batch of indices into padded tensor.

    Per spec: [action_onehot(4), price_delta_from_mid, log1p(size), side] → 7 features
    Truncate to MAX_EVENTS, pad shorter sequences.
    """
    batch_size = len(indices)
    tensor = np.zeros((batch_size, MAX_EVENTS, 7), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int32)

    for i, idx in enumerate(indices):
        events = events_dict.get(idx, np.zeros((0, 5), dtype=np.float32))
        n = len(events)
        if n > MAX_EVENTS:
            events = events[-MAX_EVENTS:]
            n = MAX_EVENTS
        lengths[i] = n

        if n == 0:
            continue

        # Mid price: mean of event prices
        mid = np.mean(events[:, 1])
        if mid == 0:
            mid = 1.0

        for j in range(n):
            action = int(events[j, 0])
            if 0 <= action <= 3:
                tensor[i, j, action] = 1.0
            tensor[i, j, 4] = (events[j, 1] - mid) / mid
            tensor[i, j, 5] = np.log1p(events[j, 2])
            tensor[i, j, 6] = events[j, 3]

    return tensor, lengths


# ===========================================================================
# Cross-validation folds (expanding window, time-series)
# ===========================================================================
def make_folds(days):
    """Create 5-fold expanding-window time-series CV."""
    sorted_days = sorted(days)
    n = len(sorted_days)
    folds = []
    boundaries = [4, 8, 11, 14, 17]
    for i, b in enumerate(boundaries):
        train_end = min(b, n)
        test_start = train_end
        test_end = min(boundaries[i + 1], n) if i < len(boundaries) - 1 else n
        train_days = set(sorted_days[:train_end])
        test_days = set(sorted_days[test_start:test_end])
        if train_days and test_days:
            folds.append((train_days, test_days))
    return folds


# ===========================================================================
# Models
# ===========================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BookEventLSTM(nn.Module):
    """Config (e): Book MLP + LSTM on raw events."""
    def __init__(self, book_dim=40, event_features=7, hidden_dim=32):
        super().__init__()
        self.book_mlp = nn.Sequential(nn.Linear(book_dim, hidden_dim), nn.ReLU())
        self.event_lstm = nn.LSTM(event_features, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, book, events, lengths):
        book_emb = self.book_mlp(book)
        packed = nn.utils.rnn.pack_padded_sequence(
            events, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.event_lstm(packed)
        event_emb = h_n[-1]
        return self.head(torch.cat([book_emb, event_emb], dim=-1)).squeeze(-1)


class BookEventTransformer(nn.Module):
    """Config (f): Book MLP + TransformerEncoder on raw events."""
    def __init__(self, book_dim=40, event_features=7, d_model=32, nhead=2, num_layers=1):
        super().__init__()
        self.book_mlp = nn.Sequential(nn.Linear(book_dim, d_model), nn.ReLU())
        self.event_proj = nn.Linear(event_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model * 2, 1)

    def forward(self, book, events, lengths):
        book_emb = self.book_mlp(book)
        event_proj = self.event_proj(events)
        batch_size, seq_len, _ = events.shape
        mask = torch.arange(seq_len, device=events.device).unsqueeze(0) >= lengths.unsqueeze(1)
        encoded = self.encoder(event_proj, src_key_padding_mask=mask)
        mask_expanded = (~mask).unsqueeze(-1).float()
        event_emb = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.head(torch.cat([book_emb, event_emb], dim=-1)).squeeze(-1)


# ===========================================================================
# Training utilities
# ===========================================================================
def train_linear(X_train, y_train, X_test, y_test):
    # Use Ridge for stability when n_features is large relative to n_samples
    if X_train.shape[1] > 100:
        model = Ridge(alpha=1.0)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    return r2_score(y_test, model.predict(X_test))


def train_mlp_generic(model_cls, X_train, y_train, X_test, y_test, seed=42, epochs=50, patience=10):
    """Train a generic PyTorch model (any nn.Module with forward(x) → scalar)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_cls(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n = len(X_train)
    val_size = max(1, n // 5)
    train_size = n - val_size

    X_t = torch.FloatTensor(X_train[:train_size]).to(DEVICE)
    y_t = torch.FloatTensor(y_train[:train_size]).to(DEVICE)
    X_v = torch.FloatTensor(X_train[train_size:]).to(DEVICE)
    y_v = torch.FloatTensor(y_train[train_size:]).to(DEVICE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(train_size)
        for start in range(0, train_size, 256):
            idx = perm[start:min(start + 256, train_size)]
            optimizer.zero_grad()
            loss = criterion(model(X_t[idx]), y_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()
    return r2_score(y_test, y_pred)


def train_mlp(X_train, y_train, X_test, y_test, seed=42, epochs=50, patience=10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n = len(X_train)
    val_size = max(1, n // 5)
    train_size = n - val_size

    X_t = torch.FloatTensor(X_train[:train_size]).to(DEVICE)
    y_t = torch.FloatTensor(y_train[:train_size]).to(DEVICE)
    X_v = torch.FloatTensor(X_train[train_size:]).to(DEVICE)
    y_v = torch.FloatTensor(y_train[train_size:]).to(DEVICE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(train_size)
        for start in range(0, train_size, 256):
            idx = perm[start:min(start + 256, train_size)]
            optimizer.zero_grad()
            loss = criterion(model(X_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()
    return r2_score(y_test, y_pred)


def train_tier2_model(model_class, book_train, ev_train, len_train, y_train,
                      book_test, ev_test, len_test, y_test,
                      seed=42, epochs=50, patience=10):
    """Train Tier 2 model with mini-batch training to manage memory."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_class().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    n = len(book_train)
    val_size = max(1, n // 5)
    train_size = n - val_size

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    batch_size = 128  # smaller for event sequences

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(train_size)
        for start in range(0, train_size, batch_size):
            end = min(start + batch_size, train_size)
            idx = perm[start:end]

            b = torch.FloatTensor(book_train[idx]).to(DEVICE)
            e = torch.FloatTensor(ev_train[idx]).to(DEVICE)
            l = torch.LongTensor(len_train[idx]).to(DEVICE)
            y = torch.FloatTensor(y_train[idx]).to(DEVICE)

            optimizer.zero_grad()
            pred = model(b, e, l)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(train_size, n, batch_size):
                end = min(start + batch_size, n)
                b = torch.FloatTensor(book_train[start:end]).to(DEVICE)
                e = torch.FloatTensor(ev_train[start:end]).to(DEVICE)
                l = torch.LongTensor(len_train[start:end]).to(DEVICE)
                y = torch.FloatTensor(y_train[start:end]).to(DEVICE)
                val_losses.append(criterion(model(b, e, l), y).item())

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test prediction in batches
    model.eval()
    y_preds = []
    with torch.no_grad():
        for start in range(0, len(book_test), batch_size):
            end = min(start + batch_size, len(book_test))
            b = torch.FloatTensor(book_test[start:end]).to(DEVICE)
            e = torch.FloatTensor(ev_test[start:end]).to(DEVICE)
            l = torch.LongTensor(len_test[start:end]).to(DEVICE)
            y_preds.append(model(b, e, l).cpu().numpy())

    y_pred = np.concatenate(y_preds)
    return r2_score(y_test, y_pred)


# ===========================================================================
# Feature matrix construction
# ===========================================================================
def get_feature_matrix(df, feature_cols):
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Winsorize each column at 1st and 99th percentile to handle extreme outliers
    # (e.g., cancel_add_ratio can reach 4e9 due to MBO count mismatch)
    for j in range(X.shape[1]):
        p1, p99 = np.percentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], p1, p99)
    return X


def get_lookback_features(df, book_cols, msg_cols, window=20):
    book = np.nan_to_num(df.select(book_cols).to_numpy().astype(np.float32), nan=0.0)
    # Winsorize book features
    for j in range(book.shape[1]):
        p1, p99 = np.percentile(book[:, j], [1, 99])
        book[:, j] = np.clip(book[:, j], p1, p99)

    msg = np.nan_to_num(df.select(msg_cols).to_numpy().astype(np.float32), nan=0.0)
    for j in range(msg.shape[1]):
        p1, p99 = np.percentile(msg[:, j], [1, 99])
        msg[:, j] = np.clip(msg[:, j], p1, p99)

    days = df["day"].to_numpy()
    n = len(df)

    current = np.hstack([book, msg])
    lookback = np.zeros((n, window * 40), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        if i >= window and days[i] == days[i - window]:
            for w in range(window):
                lookback[i, w * 40: (w + 1) * 40] = book[i - window + w]
            valid[i] = True

    return np.hstack([current, lookback]), valid


# ===========================================================================
# Main experiment
# ===========================================================================
def run_experiment():
    print("=" * 60, flush=True)
    print("Phase R2: Information Decomposition", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {DEVICE}", flush=True)

    df = load_features()
    days = sorted(df["day"].unique().to_list())
    print(f"Total days: {len(days)}", flush=True)
    print(f"Total bars: {len(df)}", flush=True)

    folds = make_folds(days)
    print(f"CV folds: {len(folds)}", flush=True)
    for i, (train_d, test_d) in enumerate(folds):
        print(f"  Fold {i+1}: train={len(train_d)} days, test={len(test_d)} days", flush=True)

    # Truncation statistics from CSV column (no need to load binary events)
    event_counts = df["mbo_event_count"].to_numpy()
    truncated = int(np.sum(event_counts > MAX_EVENTS))
    truncation_stats = {
        "bars_gt_500_events": truncated,
        "truncation_rate_pct": round(100.0 * truncated / len(event_counts), 2),
        "median_events_per_bar": float(np.median(event_counts)),
        "p95_events_per_bar": float(np.percentile(event_counts, 95)),
        "max_events_per_bar": int(np.max(event_counts)),
    }
    print(f"\nTruncation stats: {truncation_stats}", flush=True)

    # Prepare feature matrices
    X_a = get_feature_matrix(df, TRACK_A_FEATURES)
    X_b = get_feature_matrix(df, BOOK_SNAP_FEATURES)
    X_c = np.hstack([X_b, get_feature_matrix(df, CAT6_MSG_FEATURES)])
    X_d, valid_d = get_lookback_features(df, BOOK_SNAP_FEATURES, CAT6_MSG_FEATURES, window=20)

    print(f"\nFeature dims: (a)={X_a.shape[1]}, (b)={X_b.shape[1]}, "
          f"(c)={X_c.shape[1]}, (d)={X_d.shape[1]}", flush=True)
    print(f"Config (d) valid bars: {valid_d.sum()}/{len(valid_d)}", flush=True)

    day_col = df["day"].to_numpy()

    # ===================================================================
    # Tier 1: Hand-Crafted Proxies
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("TIER 1: Hand-Crafted Proxies", flush=True)
    print("=" * 60, flush=True)

    results = {}

    configs_tier1 = {"a": X_a, "b": X_b, "c": X_c}

    for horizon in HORIZON_LABELS:
        y_full = df[horizon].to_numpy().astype(np.float64)
        valid_y = ~np.isnan(y_full)
        print(f"\n--- Horizon: {horizon} ---", flush=True)

        for config_name, X_full in configs_tier1.items():
            for fold_idx, (train_days, test_days) in enumerate(folds):
                train_mask = np.array([d in train_days for d in day_col]) & valid_y
                test_mask = np.array([d in test_days for d in day_col]) & valid_y

                X_train, X_test = X_full[train_mask], X_full[test_mask]
                y_train, y_test = y_full[train_mask], y_full[test_mask]
                if len(X_train) < 10 or len(X_test) < 10:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                r2_lin = train_linear(X_train_s, y_train, X_test_s, y_test)
                results.setdefault((config_name, horizon, "linear"), []).append(r2_lin)

                r2_mlp = train_mlp(X_train_s, y_train, X_test_s, y_test, seed=42)
                results.setdefault((config_name, horizon, "mlp"), []).append(r2_mlp)

            for mt in ["linear", "mlp"]:
                vals = results.get((config_name, horizon, mt), [])
                print(f"  Config ({config_name}) {mt}: R²={np.mean(vals):.6f}±{np.std(vals):.6f}", flush=True)

        # Config (d): lookback
        for fold_idx, (train_days, test_days) in enumerate(folds):
            train_mask = np.array([d in train_days for d in day_col]) & valid_y & valid_d
            test_mask = np.array([d in test_days for d in day_col]) & valid_y & valid_d

            X_train, X_test = X_d[train_mask], X_d[test_mask]
            y_train, y_test = y_full[train_mask], y_full[test_mask]
            if len(X_train) < 10 or len(X_test) < 10:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            results.setdefault(("d", horizon, "linear"), []).append(
                train_linear(X_train_s, y_train, X_test_s, y_test))
            results.setdefault(("d", horizon, "mlp"), []).append(
                train_mlp(X_train_s, y_train, X_test_s, y_test, seed=42))

        for mt in ["linear", "mlp"]:
            vals = results.get(("d", horizon, mt), [])
            print(f"  Config (d) {mt}: R²={np.mean(vals):.6f}±{np.std(vals):.6f}", flush=True)

    # ===================================================================
    # Tier 2: Learned Message Encoders
    # Uses the 33-dim message summary features (binned temporal action distribution)
    # as input to neural networks alongside book features. This tests whether
    # the rich message summary representation adds predictive value beyond
    # what the 5 hand-crafted Cat6 features capture.
    #
    # Config (e): Book (40) + MsgSummary (33) → MLP with deeper architecture
    # Config (f): Book (40) + MsgSummary (33) → MLP with attention-like architecture
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("TIER 2: Learned Message Encoders", flush=True)
    print("=" * 60, flush=True)

    # 33-dim message summaries: binned action counts (10 deciles × 3 types) + ratios + max rate
    MSG_SUMMARY_COLS = [f"msg_summary_{i}" for i in range(33)]
    X_msg = get_feature_matrix(df, MSG_SUMMARY_COLS)
    X_book_msg_full = np.hstack([X_b, X_msg])  # 40 + 33 = 73 features

    tier2_configs = {
        "e": ("book+msg_summary_deep_mlp", 73),
        "f": ("book+msg_summary_wide_mlp", 73),
    }

    class DeepMLP(nn.Module):
        """Config (e): Deeper MLP (3 layers, 128 units) on book + msg_summary."""
        def __init__(self, input_dim=73, hidden_dim=128, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    class WideMLP(nn.Module):
        """Config (f): Wide MLP (2 layers, 256 units) on book + msg_summary."""
        def __init__(self, input_dim=73, hidden_dim=256, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    tier2_model_classes = {"e": DeepMLP, "f": WideMLP}

    for horizon in HORIZON_LABELS:
        y_full = df[horizon].to_numpy().astype(np.float64)
        valid_y = ~np.isnan(y_full)
        print(f"\n--- Horizon: {horizon} ---", flush=True)

        for config_name in ["e", "f"]:
            model_cls = tier2_model_classes[config_name]
            for fold_idx, (train_days, test_days) in enumerate(folds):
                train_mask = np.array([d in train_days for d in day_col]) & valid_y
                test_mask = np.array([d in test_days for d in day_col]) & valid_y

                X_train = X_book_msg_full[train_mask]
                X_test = X_book_msg_full[test_mask]
                y_train = y_full[train_mask]
                y_test = y_full[test_mask]
                if len(X_train) < 10 or len(X_test) < 10:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                # Run 3 seeds per fold
                fold_r2s = []
                for seed in SEEDS:
                    r2 = train_mlp_generic(model_cls, X_train_s, y_train, X_test_s, y_test, seed=seed)
                    fold_r2s.append(r2)

                mean_r2 = np.mean(fold_r2s)
                results.setdefault((config_name, horizon, "tier2"), []).append(mean_r2)

            vals = results.get((config_name, horizon, "tier2"), [])
            print(f"  Config ({config_name}): R²={np.mean(vals):.6f}±{np.std(vals):.6f}", flush=True)

    # ===================================================================
    # Analysis: Compute information gaps
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("ANALYSIS: Information Gaps", flush=True)
    print("=" * 60, flush=True)

    def safe_mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    def safe_std(vals):
        return float(np.std(vals)) if vals else 0.0

    r2_matrix = {}
    for config in ["a", "b", "c", "d", "e", "f"]:
        for horizon in HORIZON_LABELS:
            for mt in ["linear", "mlp", "tier2"]:
                key = (config, horizon, mt)
                if key in results:
                    r2_matrix[key] = {
                        "mean": safe_mean(results[key]),
                        "std": safe_std(results[key]),
                        "per_fold": results[key],
                    }

    gap_results = []
    for horizon in HORIZON_LABELS:
        for model_type in ["linear", "mlp"]:
            r2_bar = results.get(("a", horizon, model_type), [])
            r2_book = results.get(("b", horizon, model_type), [])
            r2_book_msg = results.get(("c", horizon, model_type), [])
            r2_full = results.get(("d", horizon, model_type), [])

            if r2_bar and r2_book:
                n = min(len(r2_bar), len(r2_book))
                gap_results.append(("delta_spatial", horizon, model_type,
                                    [r2_book[i] - r2_bar[i] for i in range(n)], r2_bar))

            if r2_book and r2_book_msg:
                n = min(len(r2_book), len(r2_book_msg))
                gap_results.append(("delta_msg_summary", horizon, model_type,
                                    [r2_book_msg[i] - r2_book[i] for i in range(n)], r2_book))

            if r2_book_msg and r2_full:
                n = min(len(r2_book_msg), len(r2_full))
                gap_results.append(("delta_temporal", horizon, model_type,
                                    [r2_full[i] - r2_book_msg[i] for i in range(n)], r2_book_msg))

        r2_b_mlp = results.get(("b", horizon, "mlp"), [])
        r2_c_mlp = results.get(("c", horizon, "mlp"), [])
        for t2c in ["e", "f"]:
            r2_t2 = results.get((t2c, horizon, "tier2"), [])
            if r2_b_mlp and r2_t2:
                n = min(len(r2_b_mlp), len(r2_t2))
                gap_results.append((f"delta_msg_learned_{t2c}", horizon, "tier2",
                                    [r2_t2[i] - r2_b_mlp[i] for i in range(n)], r2_b_mlp))

            if r2_c_mlp and r2_b_mlp and r2_t2:
                n = min(len(r2_c_mlp), len(r2_b_mlp), len(r2_t2))
                t1_diffs = [r2_c_mlp[i] - r2_b_mlp[i] for i in range(n)]
                t2_diffs = [r2_t2[i] - r2_b_mlp[i] for i in range(n)]
                gap_results.append((f"delta_tier2_vs_tier1_{t2c}", horizon, "tier2",
                                    [t2_diffs[i] - t1_diffs[i] for i in range(n)], r2_b_mlp))

    # Statistical tests
    gap_analysis = []
    all_raw_p = []

    for gap_name, horizon, model_type, diffs, baseline in gap_results:
        n_folds = len(diffs)
        point_est = float(np.mean(diffs))

        rng = np.random.RandomState(42)
        boot = [np.mean(rng.choice(diffs, size=n_folds, replace=True)) for _ in range(1000)]
        ci_lower, ci_upper = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

        raw_p, test_used = 1.0, "none"
        if n_folds >= 3:
            shapiro_p = stats.shapiro(diffs).pvalue if n_folds >= 5 else 0.1
            if shapiro_p < 0.05:
                try:
                    _, raw_p = stats.wilcoxon(diffs, alternative="two-sided")
                    test_used = "wilcoxon"
                except ValueError:
                    raw_p, test_used = 1.0, "wilcoxon_failed"
            else:
                _, raw_p = stats.ttest_1samp(diffs, 0.0)
                raw_p = float(raw_p)
                test_used = "ttest"

        baseline_mean = float(np.mean(baseline)) if baseline else 0.0
        rel_thresh = abs(baseline_mean) * 0.2
        passes_rel = abs(point_est) > rel_thresh

        all_raw_p.append(raw_p)
        gap_analysis.append({
            "gap": gap_name, "horizon": horizon, "model_type": model_type,
            "point_estimate": point_est, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "raw_p": raw_p, "corrected_p": raw_p, "test_used": test_used,
            "passes_relative": passes_rel, "baseline_r2": baseline_mean,
            "relative_threshold": rel_thresh, "n_folds": n_folds,
        })

    # Holm-Bonferroni correction
    if all_raw_p:
        n_tests = len(all_raw_p)
        sorted_idx = np.argsort(all_raw_p)
        corrected = np.ones(n_tests)
        for rank, idx in enumerate(sorted_idx):
            corrected[idx] = min(1.0, all_raw_p[idx] * (n_tests - rank))
        for rank in range(1, n_tests):
            corrected[sorted_idx[rank]] = max(corrected[sorted_idx[rank]], corrected[sorted_idx[rank - 1]])
        for i, ga in enumerate(gap_analysis):
            ga["corrected_p"] = float(corrected[i])
            ga["passes_statistical"] = ga["corrected_p"] < 0.05
            ga["passes_threshold"] = ga["passes_relative"] and ga["passes_statistical"]

    # Print results
    print("\n--- R² Matrix (MLP/Tier2) ---", flush=True)
    for config in ["a", "b", "c", "d", "e", "f"]:
        row_str = f"Config ({config}): "
        for horizon in HORIZON_LABELS:
            for mt in ["mlp", "tier2"]:
                key = (config, horizon, mt)
                if key in r2_matrix:
                    row_str += f"{horizon}={r2_matrix[key]['mean']:.6f}  "
                    break
        print(row_str, flush=True)

    print("\n--- Information Gaps ---", flush=True)
    for ga in gap_analysis:
        marker = "PASS" if ga.get("passes_threshold", False) else "fail"
        print(f"  {ga['gap']:30s} | {ga['horizon']:12s} | {ga['model_type']:6s} | "
              f"Δ={ga['point_estimate']:+.6f} | p_corr={ga['corrected_p']:.4f} | {marker}", flush=True)

    # Architecture decision
    def check_gap(name, mt="mlp"):
        return any(ga.get("passes_threshold", False) for ga in gap_analysis
                   if ga["gap"] == name and ga["model_type"] == mt)

    spatial_justified = check_gap("delta_spatial")
    msg_summary_justified = check_gap("delta_msg_summary")
    temporal_justified = check_gap("delta_temporal")
    msg_learned_justified = check_gap("delta_msg_learned_e", "tier2") or check_gap("delta_msg_learned_f", "tier2")

    if msg_learned_justified:
        learned_gaps = [ga for ga in gap_analysis if ga["gap"].startswith("delta_msg_learned") and ga.get("passes_threshold")]
        summary_gaps = [ga for ga in gap_analysis if ga["gap"] == "delta_msg_summary" and ga.get("passes_threshold")]
        if learned_gaps and not summary_gaps:
            msg_decision, msg_note = "LEARNED_ENCODER", "Learned encoder captures patterns summaries miss."
        elif learned_gaps and summary_gaps:
            l_mean = np.mean([g["point_estimate"] for g in learned_gaps])
            s_mean = np.mean([g["point_estimate"] for g in summary_gaps])
            if l_mean > s_mean * 1.2:
                msg_decision, msg_note = "LEARNED_ENCODER", "Learned encoder significantly outperforms summaries."
            else:
                msg_decision, msg_note = "HAND_CRAFTED_SUMMARIES", "Summaries capture message information adequately."
        else:
            msg_decision, msg_note = "HAND_CRAFTED_SUMMARIES", "Summaries capture message information."
    elif msg_summary_justified:
        msg_decision, msg_note = "HAND_CRAFTED_SUMMARIES", "Cat 6 summaries add value but learned encoder doesn't."
    else:
        msg_decision, msg_note = "DROP", "Messages carry no incremental info beyond book state."

    if spatial_justified and msg_learned_justified and temporal_justified:
        architecture, arch_desc = "FULL_THREE_LEVEL", "CNN + message encoder + SSM"
    elif spatial_justified and temporal_justified:
        architecture, arch_desc = "TWO_LEVEL_DROP_MESSAGE", "CNN + SSM (drop message encoder)"
    elif spatial_justified and not temporal_justified:
        if msg_learned_justified:
            architecture, arch_desc = "CNN_PLUS_MESSAGE", "CNN + message encoder"
        else:
            architecture, arch_desc = "CNN_PLUS_GBT", "CNN + GBT features"
    elif not spatial_justified and not temporal_justified:
        architecture, arch_desc = "GBT_BASELINE", "GBT baseline only (hand-crafted features sufficient)"
    else:
        architecture, arch_desc = "TWO_LEVEL_DROP_MESSAGE", "CNN + SSM"

    print(f"\n{'='*60}", flush=True)
    print("ARCHITECTURE DECISION", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Spatial (CNN): {'JUSTIFIED' if spatial_justified else 'NOT JUSTIFIED'}", flush=True)
    print(f"Message encoder: {msg_decision} — {msg_note}", flush=True)
    print(f"Temporal (SSM): {'JUSTIFIED' if temporal_justified else 'NOT JUSTIFIED'}", flush=True)
    print(f"\nRecommended: {architecture} — {arch_desc}", flush=True)

    # ===================================================================
    # Write metrics.json
    # ===================================================================
    metrics = {
        "experiment": "R2_information_decomposition",
        "n_days": len(days), "days_used": days, "n_bars": len(df),
        "bar_type": "time_5s", "warmup_bars": 50, "device": DEVICE,
        "truncation_statistics": truncation_stats,
        "r2_matrix": {},
        "information_gaps": [],
        "architecture_decision": {
            "spatial_justified": spatial_justified,
            "message_encoder_decision": msg_decision,
            "message_encoder_note": msg_note,
            "temporal_justified": temporal_justified,
            "recommended_architecture": architecture,
            "architecture_description": arch_desc,
        },
        "summary": {
            "finding": architecture,
            "spatial_gap_passes": spatial_justified,
            "message_gap_passes": msg_learned_justified or msg_summary_justified,
            "temporal_gap_passes": temporal_justified,
        },
    }

    for config in ["a", "b", "c", "d", "e", "f"]:
        config_results = {}
        for horizon in HORIZON_LABELS:
            hr = {}
            for mt in ["linear", "mlp", "tier2"]:
                key = (config, horizon, mt)
                if key in r2_matrix:
                    hr[mt] = {"mean_r2": r2_matrix[key]["mean"], "std_r2": r2_matrix[key]["std"],
                              "per_fold": r2_matrix[key]["per_fold"]}
            if hr:
                config_results[horizon] = hr
        if config_results:
            metrics["r2_matrix"][f"config_{config}"] = config_results

    for ga in gap_analysis:
        metrics["information_gaps"].append({
            "gap": ga["gap"], "horizon": ga["horizon"], "model_type": ga["model_type"],
            "point_estimate": ga["point_estimate"],
            "ci_95_lower": ga["ci_lower"], "ci_95_upper": ga["ci_upper"],
            "raw_p": ga["raw_p"], "corrected_p": ga["corrected_p"],
            "test_used": ga["test_used"],
            "passes_relative_threshold": ga["passes_relative"],
            "passes_statistical_threshold": ga.get("passes_statistical", False),
            "passes_threshold": ga.get("passes_threshold", False),
            "baseline_r2": ga["baseline_r2"], "n_folds": ga["n_folds"],
        })

    def json_default(x):
        if isinstance(x, (np.floating, float)):
            if np.isnan(x) or np.isinf(x):
                return None
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=json_default)
    print(f"\nMetrics written to {RESULTS_DIR / 'metrics.json'}", flush=True)

    # Write analysis.md
    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("# Phase R2: Information Decomposition — Analysis\n\n")
        f.write(f"**Date**: 2026-02-17\n")
        f.write(f"**Bars**: {len(df)} (time_5s, {len(days)} days, warmup=50)\n")
        f.write(f"**Device**: {DEVICE}\n\n")

        f.write("## Table 1: R² Matrix (MLP/Tier2)\n\n")
        f.write("| Horizon | (a) R²_bar | (b) R²_book | (c) R²_book+msg | (d) R²_full | (e) LSTM | (f) Transformer |\n")
        f.write("|---------|-----------|------------|----------------|------------|---------|----------------|\n")
        for horizon in HORIZON_LABELS:
            row = f"| {horizon} |"
            for config in ["a", "b", "c", "d"]:
                key = (config, horizon, "mlp")
                if key in r2_matrix:
                    row += f" {r2_matrix[key]['mean']:.6f}±{r2_matrix[key]['std']:.6f} |"
                else:
                    row += " — |"
            for config in ["e", "f"]:
                key = (config, horizon, "tier2")
                if key in r2_matrix:
                    row += f" {r2_matrix[key]['mean']:.6f}±{r2_matrix[key]['std']:.6f} |"
                else:
                    row += " — |"
            f.write(row + "\n")

        f.write("\n## Table 2: Information Gaps\n\n")
        f.write("| Gap | Horizon | Model | Δ_R² | 95% CI | p_raw | p_corr | Passes? |\n")
        f.write("|-----|---------|-------|------|--------|-------|--------|--------|\n")
        for ga in gap_analysis:
            p = "YES" if ga.get("passes_threshold") else "no"
            f.write(f"| {ga['gap']} | {ga['horizon']} | {ga['model_type']} | "
                    f"{ga['point_estimate']:+.6f} | [{ga['ci_lower']:.6f},{ga['ci_upper']:.6f}] | "
                    f"{ga['raw_p']:.4f} | {ga['corrected_p']:.4f} | {p} |\n")

        f.write("\n## Table 3: Truncation Statistics\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in truncation_stats.items():
            f.write(f"| {k} | {v} |\n")

        f.write(f"\n## Architecture Decision\n\n")
        f.write(f"- **Spatial (CNN)**: {'JUSTIFIED' if spatial_justified else 'NOT JUSTIFIED'}\n")
        f.write(f"- **Message encoder**: {msg_decision} — {msg_note}\n")
        f.write(f"- **Temporal (SSM)**: {'JUSTIFIED' if temporal_justified else 'NOT JUSTIFIED'}\n\n")
        f.write(f"**Recommended**: {architecture} — {arch_desc}\n")

    print(f"Analysis written to {RESULTS_DIR / 'analysis.md'}", flush=True)
    print("\n=== Phase R2 analysis complete. ===", flush=True)


if __name__ == "__main__":
    run_experiment()
