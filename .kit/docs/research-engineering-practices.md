# Research Engineering Practices — Kenoma Labs

**Audience:** All research sub-agents (RUN phase, OPS subagent, experiment scripts).
**Authority:** These practices are MANDATORY. Violations waste compute, money, and time.

---

## 1. Python Environment: uv Only

**Always use `uv` for Python package management. No conda. No raw pip.**

```bash
# Install dependencies
uv pip install torch xgboost polars scikit-learn wandb

# Pin versions in requirements.txt with exact pins
uv pip freeze > requirements.txt

# Reproduce environment
uv pip install -r requirements.txt
```

- Every experiment script must have a `requirements.txt` with exact version pins.
- The RUN phase installs dependencies via `uv pip install -r requirements.txt` before execution.
- On cloud instances: install uv first (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then use it for everything.

---

## 2. Parallelize All Independent Fits

**If you can parallelize without violating the data integrity of the experiment, do so.**

Independent work units (CPCV splits, hyperparameter grid points, multi-seed runs) MUST be parallelized across all available GPUs and CPU cores.

### GPU Parallelization Pattern

```python
import torch.multiprocessing as mp

def train_split(split_config, device_id, result_queue):
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    # ... train model on device ...
    result_queue.put((split_config["split_idx"], results))

num_gpus = torch.cuda.device_count()
# Distribute splits across GPUs in round-robin
```

### Rules

- Detect GPU count at runtime: `torch.cuda.device_count()`. Script works on 1 GPU or 4 without code changes.
- One process per GPU, pinned via `torch.cuda.set_device()`.
- No shared mutable state between workers. Each worker returns results via queue or return value.
- CPU cores split across workers: `n_cpu_per_worker = os.cpu_count() // num_workers`.
- XGBoost on GPU: `tree_method="hist", device=f"cuda:{device_id}"`. Always. No CPU XGBoost when a GPU is present.
- **Never parallelize when fold N depends on fold N-1** (expanding window with sequential state, online learning).

---

## 3. Incremental Checkpointing to S3

**Never lose completed work. Save results per-split as they complete.**

Every experiment script MUST implement incremental checkpointing:

```python
CHECKPOINT_KEY = f"s3://{BUCKET}/cloud-runs/{RUN_ID}/checkpoint.json"

def save_checkpoint(completed_splits, results):
    """Save after every completed split."""
    checkpoint = {
        "completed_splits": completed_splits,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }
    # Write to local disk AND S3
    with open("checkpoint.json", "w") as f:
        json.dump(checkpoint, f)
    s3.upload_file("checkpoint.json", BUCKET, checkpoint_key)

def load_checkpoint():
    """On startup, check for existing checkpoint and resume."""
    try:
        s3.download_file(BUCKET, checkpoint_key, "checkpoint.json")
        with open("checkpoint.json") as f:
            return json.load(f)
    except:
        return None
```

### Contract

- After each independent unit of work (split, fold, grid point), write checkpoint to S3.
- On startup, check for existing checkpoint. Skip completed work. Resume from next incomplete unit.
- Checkpoint includes: completed split indices, their results, and a timestamp.
- This is what enables spot instances — a 2-minute interruption warning is plenty of time to save state.

---

## 4. Fail Fast on MPS Locally

**Validate the full pipeline locally before spending on cloud compute.**

Before any cloud launch, run the Minimum Viable Experiment (MVE) locally on the Mac's MPS backend:

- Load a subset of data (1-2 days, not the full dataset).
- Run 1-2 splits end-to-end: data loading → normalization → training → prediction → metrics.
- Verify: no NaN, no shape mismatches, no import errors, no serialization bugs.
- Verify: checkpoint save/load round-trips correctly.
- **Only after local MVE passes do you request cloud compute.**

This catches 90% of bugs for $0 instead of discovering them 30 minutes into an EC2 run.

---

## 5. OPS Subagent for Compute Planning

**Researchers do not choose instances. OPS recommends, human approves.**

Before the RUN phase launches cloud compute, spawn an OPS subagent that:

1. **Analyzes the workload:**
   - Model parameter count → VRAM requirement per model instance
   - Dataset size → storage and memory requirements
   - Number of independent fits → parallelism opportunity
   - Estimated per-fit wall-clock → total compute budget

2. **Recommends infrastructure:**
   - Instance type (smallest that fits: `VRAM_per_model × num_parallel_workers + dataset < GPU_memory`)
   - Spot vs on-demand (spot is default if checkpointing is implemented)
   - Estimated cost: `instance_cost_per_hr × estimated_wall_hours`
   - Estimated wall-clock with parallelization

3. **Instance selection hierarchy:**
   - **G-family first** (g5, g4dn) — A10G and T4 cover most research workloads
   - **P-family only** when a single model needs >24GB VRAM (large transformers, not small CNNs)
   - **CPU-only** (c7a) for pure XGBoost/sklearn workloads with no neural network component
   - Never over-provision: a 12K-parameter CNN does not need an A100

4. **Presents recommendation to human for approval** before any cloud resource is provisioned.

---

## 6. Reproducibility — Non-Negotiable

**Every experiment must be fully reproducible from its artifacts.**

### Every experiment script MUST:

```python
# 1. Set global seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 2. Deterministic CUDA operations
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 3. Per-split seeds (deterministic but unique)
split_seed = SEED + split_idx

# 4. Log environment fingerprint
env_fingerprint = {
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    "xgboost_version": xgb.__version__,
    "polars_version": pl.__version__,
    "numpy_version": np.__version__,
    "uv_frozen_deps": open("requirements.txt").read(),
    "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    "data_sha256": "<hash of input data manifest>",
    "seed": SEED,
}
```

### Store with results:
- `env_fingerprint.json` — full environment and dependency snapshot
- `requirements.txt` — exact pinned dependencies (from `uv pip freeze`)
- The experiment script itself (copy into results directory)

---

## 7. Data Locality: EBS Snapshots

**Use EBS snapshots for datasets. Do not download from S3 at instance startup.**

- All datasets are pre-loaded into EBS snapshots.
- Cloud instances attach the EBS snapshot at launch — data is available immediately on NVMe.
- No waiting 5-10 minutes for S3 downloads before the experiment can start.
- When datasets grow to TB scale: transition to streaming via `mosaic-streaming` or `webdataset` from S3. Do not attempt to fit TBs on a single EBS volume.

---

## 8. Experiment Tracking: Weights & Biases

**All experiments log to W&B. No exceptions.**

```python
import wandb

# Initialize at experiment start
wandb.init(
    project="mbo-dl",
    name=f"{experiment_name}-{run_id}",
    config={
        "experiment": experiment_name,
        "seed": SEED,
        "instance_type": os.environ.get("INSTANCE_TYPE", "local"),
        # ... all hyperparameters ...
    },
)

# Log per-split metrics
wandb.log({
    "split": split_idx,
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "expectancy_base": expectancy,
    "epoch": epochs_trained,
    "wall_seconds": elapsed,
})

# Log final summary
wandb.summary["cpcv_mean_accuracy"] = mean_acc
wandb.summary["outcome"] = "D"

# Save artifacts
wandb.save("checkpoint.json")
wandb.save("metrics.json")

wandb.finish()
```

### What W&B provides:
- **Live metrics dashboard** — watch splits completing in real-time from your browser
- **System metrics** — GPU utilization, VRAM, CPU, temperature (automatic, no code needed)
- **Experiment comparison** — side-by-side runs across different configs
- **Artifact versioning** — track which data, script, and checkpoint produced which result
- **Sweeps** — managed hyperparameter search across spot instances (for future XGBoost tuning)
- **Alerts** — notify on NaN loss, GPU idle, or run failure

---

## 9. Infrastructure Observability: CloudWatch

**CloudWatch handles infrastructure-level logging. W&B handles experiment-level tracking.**

All cloud experiment scripts MUST stream stdout/stderr to CloudWatch:

```python
# At script startup, configure CloudWatch log streaming
# Log group: /kenoma/experiments
# Log stream: {run_id}
```

CloudWatch covers what W&B doesn't:
- Instance boot and provisioning logs
- Dependency installation output
- OOM kills, CUDA driver errors, spot interruption notices
- System-level failures before the experiment script even starts

**Together:** W&B tells you "split 23 accuracy was 0.41". CloudWatch tells you "the instance ran out of disk space before split 23 started."

---

## 10. Fail Fast, Don't Retry Silently

**A broken experiment should stop immediately and report, not waste compute retrying.**

- If a split produces NaN loss: **ABORT the entire run.** Log the failure. Do not skip and continue.
- If accuracy on the first 2 splits is below random (< 0.33): **ABORT.** The pipeline is broken.
- If a dependency fails to install: **ABORT.** Do not attempt workarounds.
- If CUDA is not available on a GPU instance: **ABORT.** Do not fall back to CPU silently.
- Never catch and suppress exceptions during training. Let them propagate, let the run fail, let CloudWatch and W&B capture the stack trace.

**The cost of running 43 more splits on a broken pipeline is always higher than stopping at split 2.**

---

## 11. Spot Instances as Default

**Spot is the default for all cloud runs. On-demand only when checkpointing is not yet implemented for the specific workload.**

With incremental checkpointing (Practice #3), spot interruptions are a non-event:
- Spot interruption gives 2-minute warning.
- Checkpoint saves in <5 seconds.
- New spot instance launches, loads checkpoint, resumes.
- You paid 60-70% less for potentially better GPUs.

### Spot eligibility checklist:
- [x] Incremental checkpointing implemented and tested
- [x] Checkpoint save completes in <30 seconds
- [x] Resume-from-checkpoint verified locally
- [ ] If ANY box is unchecked → use on-demand

---

## 12. OPS Handles Resource Cleanup

**OPS is responsible for the full lifecycle of cloud resources.**

After every cloud run (success or failure), OPS must:
1. **Terminate the instance** (self-terminate on completion; OPS reaps orphans)
2. **Clean up EBS volumes** that are no longer attached
3. **Garbage-collect S3 artifacts** older than retention policy (checkpoints: 7 days, final results: permanent)
4. **Reap stale cloud-run entries** from `cloud-state.json`
5. **Report resource usage** — instances launched, hours consumed, cost estimate

Automated cleanup runs via `cloud-run gc` and `cloud-run reap`. OPS flags anomalies (instances running >24h, orphaned volumes, unexpectedly high S3 usage).

---

## Summary: The Experiment Lifecycle

```
1. FRAME designs the experiment spec
2. Local MVE on MPS — fail fast for $0 (Practice #4)
3. OPS subagent analyzes workload → recommends instance + cost estimate (Practice #5)
4. Human approves the compute plan
5. RUN phase launches on approved infrastructure:
   - uv installs pinned dependencies (Practice #1)
   - EBS snapshot provides data instantly (Practice #7)
   - W&B initialized for experiment tracking (Practice #8)
   - CloudWatch streaming for infra logs (Practice #9)
   - Reproducibility fingerprint logged (Practice #6)
   - Independent fits parallelized across all GPUs (Practice #2)
   - Per-split checkpointing to S3 (Practice #3)
   - Fail fast on any pipeline error (Practice #10)
   - Spot instance with automatic resume on interruption (Practice #11)
6. READ/LOG phase analyzes results from W&B + S3
7. OPS cleans up resources (Practice #12)
```
