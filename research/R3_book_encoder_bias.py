"""
R3: Book Encoder Inductive Bias Experiment
Compares CNN (local spatial), Attention (global spatial), and MLP (no spatial)
encoders on 20-level MES order book snapshots for return_5 prediction.
"""

import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / ".kit" / "results" / "book-encoder-bias"
DATA_PATH = Path(__file__).resolve().parent.parent / ".kit" / "results" / "info-decomposition" / "features.csv"
TICK_SIZE = 0.25
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BATCH_SIZE = 512
MAX_EPOCHS = 50
PATIENCE = 10
LR = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4
TARGET_PARAMS = 12000  # ~12k target

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    """Load features CSV, extract book snapshots and return_5."""
    df = pl.read_csv(str(DATA_PATH))

    # Get unique sorted days
    days = sorted(df["day"].unique().to_list())
    print(f"Days ({len(days)}): {days}")

    # Extract book snapshot columns (40 columns: book_snap_0 .. book_snap_39)
    book_cols = [f"book_snap_{i}" for i in range(40)]
    book_raw = df.select(book_cols).to_numpy().astype(np.float32)

    # The CSV has duplicate return columns; use the last set (forward returns)
    # return_5 appears twice — the feature columns have it at index ~45, and
    # the forward return columns are the last 4 before mbo_event_count.
    # We need the forward return version (which is the actual 5-bar forward return).
    all_cols = df.columns
    # Find the last occurrence of return_5
    return_5_indices = [i for i, c in enumerate(all_cols) if c == "return_5"]
    if len(return_5_indices) > 1:
        # Use the last one (forward return section)
        return_col_idx = return_5_indices[-1]
        target = df[:, return_col_idx].to_numpy().astype(np.float32)
    else:
        target = df["return_5"].to_numpy().astype(np.float32)

    day_labels = df["day"].to_numpy()
    is_warmup = df["is_warmup"].to_numpy()

    # Filter out warmup bars
    mask = is_warmup == False  # noqa: E712
    if isinstance(is_warmup[0], str):
        mask = is_warmup != "true"
    book_raw = book_raw[mask]
    target = target[mask]
    day_labels = day_labels[mask]

    # Normalize book snapshots
    # Price deltas (even indices: 0,2,4,...,38) — already in tick multiples from C++ export
    # The C++ code stores (price - mid) which for MES tick_size=0.25 gives values like -2.375, -0.125, etc.
    # Divide by TICK_SIZE to get integer ticks
    price_idx = np.arange(0, 40, 2)
    size_idx = np.arange(1, 40, 2)

    book_raw[:, price_idx] = book_raw[:, price_idx] / TICK_SIZE

    # Sizes (odd indices: 1,3,5,...,39) — apply log1p and z-score per day
    book_raw[:, size_idx] = np.log1p(np.abs(book_raw[:, size_idx]))
    for d in np.unique(day_labels):
        day_mask = day_labels == d
        for si in size_idx:
            col = book_raw[day_mask, si]
            mu, sigma = col.mean(), col.std()
            if sigma > 1e-8:
                book_raw[day_mask, si] = (col - mu) / sigma
            else:
                book_raw[day_mask, si] = 0.0

    # Replace NaN/inf with 0
    nan_count = np.isnan(book_raw).sum() + np.isinf(book_raw).sum()
    target_nan = np.isnan(target).sum() + np.isinf(target).sum()
    book_raw = np.nan_to_num(book_raw, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"NaN/inf replaced: book={nan_count}, target={target_nan}")
    print(f"Total samples after warmup filter: {len(book_raw)}")

    return book_raw, target, day_labels, days


# ---------------------------------------------------------------------------
# Expanding-Window CV Splits
# ---------------------------------------------------------------------------
def get_cv_folds(day_labels, days):
    """5-fold expanding-window time-series CV."""
    folds = [
        (days[0:4], days[4:7]),
        (days[0:7], days[7:10]),
        (days[0:10], days[10:13]),
        (days[0:13], days[13:16]),
        (days[0:16], days[16:19]),
    ]
    splits = []
    for train_days, test_days in folds:
        train_mask = np.isin(day_labels, train_days)
        test_mask = np.isin(day_labels, test_days)
        splits.append((train_mask, test_mask))
    return splits


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    """Conv1d on (B, 2, 20) — local spatial prior."""

    def __init__(self, ch=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(ch, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (B, 20, 2) -> (B, 2, 20) channels-first
        x = x.permute(0, 2, 1)
        z = self.conv(x).squeeze(-1)  # (B, ch)
        return self.head(z).squeeze(-1)

    def embed(self, x):
        """Get penultimate 16-dim embedding."""
        x = x.permute(0, 2, 1)
        z = self.conv(x).squeeze(-1)
        return self.head[0](z)  # Linear(ch -> 16) only


class AttentionEncoder(nn.Module):
    """Single-layer self-attention on 20 tokens — global spatial prior."""

    def __init__(self, d_model=32, nhead=2, ff_dim=64):
        super().__init__()
        self.proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 20, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (B, 20, 2)
        z = self.proj(x) + self.pos_embed
        z = self.encoder(z)
        z = z.mean(dim=1)  # mean-pool over 20 tokens
        return self.head(z).squeeze(-1)


class MLPEncoder(nn.Module):
    """MLP on flattened 40-dim book vector — no spatial prior."""

    def __init__(self, h1=128, h2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        # x: (B, 20, 2) -> (B, 40) flatten
        return self.net(x.reshape(x.size(0), -1)).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def tune_param_counts():
    """Find channel/hidden dims so all 3 models are within ±10% of TARGET_PARAMS."""
    # CNN: params ≈ 2*ch*3*2 + ch + ch*ch*3*2 + ch + ch + ch + ch*16+16 + 16*1+1
    # Try different ch values
    best_cnn_ch = 48
    for ch in range(16, 128):
        m = CNNEncoder(ch=ch)
        p = count_params(m)
        if abs(p - TARGET_PARAMS) < abs(count_params(CNNEncoder(ch=best_cnn_ch)) - TARGET_PARAMS):
            best_cnn_ch = ch

    # Attention: try different d_model values
    best_attn_d = 32
    best_attn_ff = 64
    for d in range(8, 80, 2):
        for ff in [d * 2, d * 3, d * 4]:
            try:
                m = AttentionEncoder(d_model=d, nhead=2, ff_dim=ff)
                p = count_params(m)
                ref = count_params(AttentionEncoder(d_model=best_attn_d, nhead=2, ff_dim=best_attn_ff))
                if abs(p - TARGET_PARAMS) < abs(ref - TARGET_PARAMS):
                    best_attn_d = d
                    best_attn_ff = ff
            except Exception:
                pass

    # MLP: try different hidden dims
    best_mlp_h1, best_mlp_h2 = 128, 64
    for h1 in range(32, 256):
        for h2 in [h1 // 2, h1 // 3, h1 // 4]:
            if h2 < 4:
                continue
            m = MLPEncoder(h1=h1, h2=h2)
            p = count_params(m)
            ref = count_params(MLPEncoder(h1=best_mlp_h1, h2=best_mlp_h2))
            if abs(p - TARGET_PARAMS) < abs(ref - TARGET_PARAMS):
                best_mlp_h1 = h1
                best_mlp_h2 = h2

    cnn = CNNEncoder(ch=best_cnn_ch)
    attn = AttentionEncoder(d_model=best_attn_d, nhead=2, ff_dim=best_attn_ff)
    mlp = MLPEncoder(h1=best_mlp_h1, h2=best_mlp_h2)

    pc = {
        "cnn": count_params(cnn),
        "attention": count_params(attn),
        "mlp": count_params(mlp),
    }
    print(f"Parameter counts — CNN: {pc['cnn']} (ch={best_cnn_ch}), "
          f"Attention: {pc['attention']} (d={best_attn_d}, ff={best_attn_ff}), "
          f"MLP: {pc['mlp']} (h1={best_mlp_h1}, h2={best_mlp_h2})")

    # Verify ±10%
    mean_p = np.mean(list(pc.values()))
    for name, p in pc.items():
        pct = abs(p - mean_p) / mean_p * 100
        print(f"  {name}: {p} ({pct:.1f}% from mean {mean_p:.0f})")

    config = {
        "cnn_ch": best_cnn_ch,
        "attn_d": best_attn_d,
        "attn_ff": best_attn_ff,
        "mlp_h1": best_mlp_h1,
        "mlp_h2": best_mlp_h2,
    }
    return config, pc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_dataloader(X, y, batch_size, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_X, train_y, val_X, val_y, fold_idx, model_name):
    """Train a model with early stopping, cosine LR schedule. Returns best val loss, training curves."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    criterion = nn.MSELoss()

    train_loader = make_dataloader(train_X, train_y, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(val_X, val_y, BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        curves.append({
            "epoch": epoch + 1,
            "model": model_name,
            "fold": fold_idx + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  {model_name} fold {fold_idx+1}: early stop at epoch {epoch+1}")
                break

    # Restore best
    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model, curves


def evaluate_r2(model, X, y):
    """Compute out-of-sample R²."""
    model.eval()
    loader = make_dataloader(X, y, BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds)
    return r2_score(y, preds)


def extract_cnn_embeddings(model, X):
    """Extract 16-dim penultimate layer embeddings from CNN."""
    model.eval()
    loader = make_dataloader(X, np.zeros(len(X)), BATCH_SIZE, shuffle=False)
    embeds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            emb = model.embed(xb)
            embeds.append(emb.cpu().numpy())
    return np.concatenate(embeds)


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------
def cohens_d(x, y):
    """Paired Cohen's d."""
    diff = x - y
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def ci_95_diff(x, y):
    """95% CI on mean difference."""
    diff = x - y
    n = len(diff)
    mean_d = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    t_val = stats.t.ppf(0.975, df=n - 1)
    return mean_d - t_val * se, mean_d + t_val * se


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    corrected = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])
    return corrected


def pairwise_tests(r2_dict):
    """Run pairwise tests on 5-fold R² vectors."""
    pairs = [("cnn", "attention"), ("cnn", "mlp"), ("attention", "mlp")]
    results = []
    raw_ps = []

    for a, b in pairs:
        va = np.array(r2_dict[a])
        vb = np.array(r2_dict[b])
        diff = va - vb

        # Shapiro-Wilk on differences
        if len(diff) >= 3:
            sw_stat, sw_p = stats.shapiro(diff)
        else:
            sw_stat, sw_p = 0, 0

        # Choose test
        if sw_p > 0.05:
            test_name = "paired_t"
            t_stat, raw_p = stats.ttest_rel(va, vb)
        else:
            test_name = "wilcoxon"
            try:
                w_stat, raw_p = stats.wilcoxon(va, vb)
            except ValueError:
                # All differences are zero
                raw_p = 1.0

        d = cohens_d(va, vb)
        ci_lo, ci_hi = ci_95_diff(va, vb)
        raw_ps.append(raw_p)
        results.append({
            "pair": f"{a}_vs_{b}",
            "test": test_name,
            "shapiro_p": float(sw_p),
            "raw_p": float(raw_p),
            "cohens_d": float(d),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "mean_diff": float(diff.mean()),
        })

    # Holm-Bonferroni correction
    corrected = holm_bonferroni(np.array(raw_ps))
    for i, r in enumerate(results):
        r["corrected_p"] = float(corrected[i])

    return results


# ---------------------------------------------------------------------------
# Sufficiency Test
# ---------------------------------------------------------------------------
def sufficiency_test(cnn_models, book_data, targets, splits):
    """Compare linear probe on CNN-16d embeddings vs raw-40d book."""
    results = {"cnn_16d": [], "raw_40d": []}
    for fold_idx, (train_mask, test_mask) in enumerate(splits):
        model = cnn_models[fold_idx]
        # Reshape to (N, 20, 2) for CNN
        test_X_3d = book_data[test_mask].reshape(-1, 20, 2)
        test_y = targets[test_mask]
        train_X_3d = book_data[train_mask].reshape(-1, 20, 2)
        train_y = targets[train_mask]

        # CNN embeddings
        train_emb = extract_cnn_embeddings(model, train_X_3d)
        test_emb = extract_cnn_embeddings(model, test_X_3d)

        # Raw 40d
        train_raw = book_data[train_mask]
        test_raw = book_data[test_mask]

        # Ridge with nested CV (alpha selection)
        alphas = np.logspace(-3, 3, 20)

        # CNN-16d probe
        ridge_cnn = RidgeCV(alphas=alphas, cv=5)
        ridge_cnn.fit(train_emb, train_y)
        r2_cnn = r2_score(test_y, ridge_cnn.predict(test_emb))
        results["cnn_16d"].append(float(r2_cnn))

        # Raw-40d probe
        ridge_raw = RidgeCV(alphas=alphas, cv=5)
        ridge_raw.fit(train_raw, train_y)
        r2_raw = r2_score(test_y, ridge_raw.predict(test_raw))
        results["raw_40d"].append(float(r2_raw))

        print(f"  Fold {fold_idx+1}: CNN-16d R²={r2_cnn:.6f}, Raw-40d R²={r2_raw:.6f}")

    return results


# ---------------------------------------------------------------------------
# Decision Rules
# ---------------------------------------------------------------------------
def apply_decision_rules(pairwise_results, r2_dict, sufficiency_results):
    """Apply the spec's decision rules and return textual decisions."""
    decisions = {}

    # Build lookup for pairwise results
    pw = {r["pair"]: r for r in pairwise_results}

    mean_r2 = {k: np.mean(v) for k, v in r2_dict.items()}

    # Rule 1: Spatial structure
    cnn_vs_attn = pw["cnn_vs_attention"]
    cnn_vs_mlp = pw["cnn_vs_mlp"]
    attn_vs_mlp = pw["attention_vs_mlp"]

    def sig_better(result, direction="positive"):
        """Check if pair A significantly beats pair B (corrected p < 0.05 and d > 0.5)."""
        if direction == "positive":
            return result["corrected_p"] < 0.05 and result["cohens_d"] > 0.5
        else:
            return result["corrected_p"] < 0.05 and result["cohens_d"] < -0.5

    def approx_equal(result):
        return result["corrected_p"] >= 0.05

    # Evaluate rule 1 conditions
    if approx_equal(cnn_vs_attn) and (sig_better(cnn_vs_mlp) or sig_better(attn_vs_mlp)):
        decisions["rule1"] = "CNN ≈ Attention AND spatial > MLP → Local prior sufficient. Use Conv1d encoder (v0.6 validated)."
        decisions["rule1_code"] = "local_spatial_sufficient"
    elif sig_better(cnn_vs_attn, "negative") and sig_better(attn_vs_mlp):
        decisions["rule1"] = "Attention >> CNN >> MLP → Long-range spatial matters. Replace Conv1d with attention encoder."
        decisions["rule1_code"] = "attention_needed"
    elif approx_equal(cnn_vs_attn) and approx_equal(cnn_vs_mlp) and approx_equal(attn_vs_mlp):
        decisions["rule1"] = "All three models ≈ equal → No exploitable spatial structure. Use MLP (simplest)."
        decisions["rule1_code"] = "no_spatial_structure"
    elif sig_better(cnn_vs_mlp, "negative") or sig_better(attn_vs_mlp, "negative"):
        decisions["rule1"] = "MLP >> spatial models → Spatial priors hurt. Use MLP. Investigate spatial overfitting."
        decisions["rule1_code"] = "spatial_hurts"
    else:
        # Fallback: use best mean R²
        best = max(mean_r2, key=mean_r2.get)
        decisions["rule1"] = f"No clear pattern at corrected p<0.05. Best mean R²: {best}. Recommend simplest adequate model."
        decisions["rule1_code"] = f"inconclusive_best_{best}"

    # Rule 2: Sufficiency test
    mean_cnn16 = np.mean(sufficiency_results["cnn_16d"])
    mean_raw40 = np.mean(sufficiency_results["raw_40d"])
    if mean_raw40 > 0:
        retention = mean_cnn16 / mean_raw40
    else:
        retention = float("inf") if mean_cnn16 > 0 else 1.0

    if retention >= 0.9:
        decisions["rule2"] = f"R²(CNN-16d)/R²(Raw-40d) = {retention:.3f} ≥ 0.9 → CNN embedding is sufficient statistic. Use 16-dim embedding."
        decisions["rule2_code"] = "sufficient"
    else:
        decisions["rule2"] = f"R²(CNN-16d)/R²(Raw-40d) = {retention:.3f} < 0.9 → CNN loses information. Increase embedding dim."
        decisions["rule2_code"] = "insufficient"

    decisions["retention_ratio"] = float(retention)
    return decisions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Results dir: {RESULTS_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n=== Loading Data ===")
    book_data, targets, day_labels, days = load_data()
    print(f"Book shape: {book_data.shape}, Target shape: {targets.shape}")
    print(f"Target stats: mean={targets.mean():.4f}, std={targets.std():.4f}")

    # CV splits
    splits = get_cv_folds(day_labels, days)
    for i, (tr, te) in enumerate(splits):
        print(f"  Fold {i+1}: train={tr.sum()}, test={te.sum()}")

    # Tune parameter counts
    print("\n=== Parameter Matching ===")
    config, param_counts = tune_param_counts()

    # Save param counts
    with open(RESULTS_DIR / "param_counts.json", "w") as f:
        json.dump(param_counts, f, indent=2)

    # Train all models across all folds
    print("\n=== Training ===")
    r2_dict = {"cnn": [], "attention": [], "mlp": []}
    all_curves = []
    cnn_models = []

    model_factories = {
        "cnn": lambda: CNNEncoder(ch=config["cnn_ch"]),
        "attention": lambda: AttentionEncoder(d_model=config["attn_d"], nhead=2, ff_dim=config["attn_ff"]),
        "mlp": lambda: MLPEncoder(h1=config["mlp_h1"], h2=config["mlp_h2"]),
    }

    for fold_idx, (train_mask, test_mask) in enumerate(splits):
        print(f"\n--- Fold {fold_idx+1}/5 ---")
        # Reshape to (N, 20, 2) for CNN and Attention
        train_X = book_data[train_mask].reshape(-1, 20, 2)
        test_X = book_data[test_mask].reshape(-1, 20, 2)
        train_y = targets[train_mask]
        test_y = targets[test_mask]

        for model_name in ["cnn", "attention", "mlp"]:
            print(f"  Training {model_name}...")
            torch.manual_seed(SEED + fold_idx)
            model = model_factories[model_name]()
            model, curves = train_model(model, train_X, train_y, test_X, test_y, fold_idx, model_name)
            r2 = evaluate_r2(model, test_X, test_y)
            r2_dict[model_name].append(float(r2))
            all_curves.extend(curves)
            print(f"    R² = {r2:.6f}")

            if model_name == "cnn":
                cnn_models.append(model)

    # Save model comparison table
    print("\n=== Model Comparison (Table 1) ===")
    comparison_rows = []
    for model_name in ["cnn", "attention", "mlp"]:
        for fold_idx, r2 in enumerate(r2_dict[model_name]):
            comparison_rows.append({
                "model": model_name,
                "fold": fold_idx + 1,
                "r2": r2,
            })
        comparison_rows.append({
            "model": model_name,
            "fold": "mean",
            "r2": float(np.mean(r2_dict[model_name])),
        })
        comparison_rows.append({
            "model": model_name,
            "fold": "std",
            "r2": float(np.std(r2_dict[model_name])),
        })

    comp_df = pl.DataFrame(comparison_rows)
    comp_df.write_csv(str(RESULTS_DIR / "model_comparison.csv"))

    for mn in ["cnn", "attention", "mlp"]:
        vals = r2_dict[mn]
        print(f"  {mn}: mean={np.mean(vals):.6f} ± {np.std(vals):.6f}")

    # Save training curves
    curves_df = pl.DataFrame(all_curves)
    curves_df.write_csv(str(RESULTS_DIR / "training_curves.csv"))

    # Pairwise tests
    print("\n=== Pairwise Statistical Tests (Table 2) ===")
    pw_results = pairwise_tests(r2_dict)
    for r in pw_results:
        print(f"  {r['pair']}: raw_p={r['raw_p']:.4f}, corrected_p={r['corrected_p']:.4f}, "
              f"d={r['cohens_d']:.4f}, CI=[{r['ci_lo']:.6f}, {r['ci_hi']:.6f}]")

    pw_df = pl.DataFrame(pw_results)
    pw_df.write_csv(str(RESULTS_DIR / "pairwise_tests.csv"))

    # Sufficiency test
    print("\n=== Sufficiency Test (Table 3) ===")
    suff_results = sufficiency_test(cnn_models, book_data, targets, splits)

    suff_rows = []
    for rep in ["cnn_16d", "raw_40d"]:
        for fold_idx, r2 in enumerate(suff_results[rep]):
            suff_rows.append({"representation": rep, "fold": fold_idx + 1, "r2": r2})
        suff_rows.append({"representation": rep, "fold": "mean", "r2": float(np.mean(suff_results[rep]))})
        suff_rows.append({"representation": rep, "fold": "std", "r2": float(np.std(suff_results[rep]))})

    # Paired t-test on sufficiency
    suff_va = np.array(suff_results["cnn_16d"])
    suff_vb = np.array(suff_results["raw_40d"])
    suff_t, suff_p = stats.ttest_rel(suff_va, suff_vb)
    suff_d = cohens_d(suff_va, suff_vb)
    suff_ci = ci_95_diff(suff_va, suff_vb)
    print(f"  Paired t-test: t={suff_t:.4f}, p={suff_p:.4f}, d={suff_d:.4f}, "
          f"CI=[{suff_ci[0]:.6f}, {suff_ci[1]:.6f}]")

    suff_df = pl.DataFrame(suff_rows)
    suff_df.write_csv(str(RESULTS_DIR / "sufficiency_test.csv"))

    # Decision rules
    print("\n=== Decision Rules ===")
    decisions = apply_decision_rules(pw_results, r2_dict, suff_results)
    print(f"  Rule 1: {decisions['rule1']}")
    print(f"  Rule 2: {decisions['rule2']}")

    # Write analysis.md
    write_analysis(r2_dict, pw_results, suff_results, suff_t, suff_p, suff_d, suff_ci,
                   decisions, param_counts, config)

    # Write metrics.json
    write_metrics(r2_dict, pw_results, suff_results, decisions, param_counts,
                  suff_t, suff_p, suff_d, suff_ci)

    print(f"\n=== Done. Results in {RESULTS_DIR} ===")


def write_analysis(r2_dict, pw_results, suff_results, suff_t, suff_p, suff_d, suff_ci,
                   decisions, param_counts, config):
    """Write human-readable analysis.md."""
    lines = ["# R3: Book Encoder Inductive Bias — Analysis\n"]

    # Table 1
    lines.append("## Table 1: Model Comparison (Out-of-Sample R²)\n")
    lines.append("| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |")
    lines.append("|-------|--------|--------|--------|--------|--------|------|-----|")
    for mn in ["cnn", "attention", "mlp"]:
        vals = r2_dict[mn]
        row = f"| {mn.upper()} | " + " | ".join(f"{v:.6f}" for v in vals)
        row += f" | {np.mean(vals):.6f} | {np.std(vals):.6f} |"
        lines.append(row)

    lines.append(f"\nParameter counts: CNN={param_counts['cnn']}, "
                 f"Attention={param_counts['attention']}, MLP={param_counts['mlp']}\n")

    # Table 2
    lines.append("## Table 2: Pairwise Statistical Tests\n")
    lines.append("| Pair | Test | Shapiro p | Raw p | Corrected p | Cohen's d | 95% CI |")
    lines.append("|------|------|-----------|-------|-------------|-----------|--------|")
    for r in pw_results:
        lines.append(f"| {r['pair']} | {r['test']} | {r['shapiro_p']:.4f} | "
                     f"{r['raw_p']:.4f} | {r['corrected_p']:.4f} | {r['cohens_d']:.4f} | "
                     f"[{r['ci_lo']:.6f}, {r['ci_hi']:.6f}] |")

    # Table 3
    lines.append("\n## Table 3: Sufficiency Test (Linear Probe R²)\n")
    lines.append("| Representation | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |")
    lines.append("|----------------|--------|--------|--------|--------|--------|------|-----|")
    for rep in ["cnn_16d", "raw_40d"]:
        vals = suff_results[rep]
        row = f"| {rep} | " + " | ".join(f"{v:.6f}" for v in vals)
        row += f" | {np.mean(vals):.6f} | {np.std(vals):.6f} |"
        lines.append(row)

    lines.append(f"\nCompression: 40d → 16d (2.5× reduction)")
    lines.append(f"R² retention ratio: {decisions['retention_ratio']:.3f}")
    lines.append(f"Paired t-test: t={suff_t:.4f}, p={suff_p:.4f}, Cohen's d={suff_d:.4f}, "
                 f"95% CI=[{suff_ci[0]:.6f}, {suff_ci[1]:.6f}]\n")

    # Decision rules
    lines.append("## Decision Rules\n")
    lines.append(f"**Rule 1 (Spatial structure):** {decisions['rule1']}\n")
    lines.append(f"**Rule 2 (Sufficiency):** {decisions['rule2']}\n")

    # Architecture recommendation
    lines.append("## Architecture Recommendation for Phase 6\n")
    if "local_spatial" in decisions["rule1_code"]:
        lines.append("Use the Conv1d spatial encoder from v0.6 architecture. The local spatial "
                     "prior is validated — adjacent book levels share predictive structure.\n")
    elif "attention" in decisions["rule1_code"]:
        lines.append("Replace Conv1d with self-attention spatial encoder. Long-range book "
                     "correlations dominate.\n")
    elif "no_spatial" in decisions["rule1_code"]:
        lines.append("Use MLP on flattened book vector. No exploitable spatial structure found.\n")
    elif "spatial_hurts" in decisions["rule1_code"]:
        lines.append("Use MLP. Spatial inductive biases cause overfitting on this data.\n")
    else:
        lines.append("No clear winner at significance threshold. Recommend using the simplest "
                     "model (MLP) or the one with highest mean R² as a practical choice.\n")

    if decisions["rule2_code"] == "sufficient":
        lines.append("The CNN 16-dim embedding is a sufficient statistic for the order book. "
                     "Use it as the spatial encoder output dimension.\n")
    else:
        lines.append("The CNN 16-dim embedding loses information. Consider increasing the "
                     "embedding dimension.\n")

    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("\n".join(lines))


def write_metrics(r2_dict, pw_results, suff_results, decisions, param_counts,
                  suff_t, suff_p, suff_d, suff_ci):
    """Write metrics.json with all results."""
    metrics = {
        "experiment": "R3_book_encoder_bias",
        "param_counts": param_counts,
        "model_comparison": {
            model: {
                "fold_r2": vals,
                "mean_r2": float(np.mean(vals)),
                "std_r2": float(np.std(vals)),
            }
            for model, vals in r2_dict.items()
        },
        "pairwise_tests": pw_results,
        "sufficiency_test": {
            "cnn_16d": {
                "fold_r2": suff_results["cnn_16d"],
                "mean_r2": float(np.mean(suff_results["cnn_16d"])),
                "std_r2": float(np.std(suff_results["cnn_16d"])),
            },
            "raw_40d": {
                "fold_r2": suff_results["raw_40d"],
                "mean_r2": float(np.mean(suff_results["raw_40d"])),
                "std_r2": float(np.std(suff_results["raw_40d"])),
            },
            "paired_t_stat": float(suff_t),
            "paired_t_p": float(suff_p),
            "cohens_d": float(suff_d),
            "ci_95": [float(suff_ci[0]), float(suff_ci[1])],
            "retention_ratio": decisions["retention_ratio"],
        },
        "decisions": {
            "rule1": decisions["rule1"],
            "rule1_code": decisions["rule1_code"],
            "rule2": decisions["rule2"],
            "rule2_code": decisions["rule2_code"],
        },
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
