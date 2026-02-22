#!/usr/bin/env python3
"""Launcher for e2e-cnn-classification experiment (GPU mode).

Downloads Parquet data and experiment script from S3, runs the experiment,
then copies results to /work/results/ for the bootstrap S3 sync.
"""
import boto3
import os
import subprocess
import sys

s3 = boto3.client("s3")
BUCKET = "kenoma-labs-research"

# 1. Download Parquet data
data_dir = ".kit/results/full-year-export"
os.makedirs(data_dir, exist_ok=True)
paginator = s3.get_paginator("list_objects_v2")
files = []
for page in paginator.paginate(Bucket=BUCKET, Prefix="data/full-year-export/"):
    files.extend(
        [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".parquet")]
    )
print(f"Downloading {len(files)} parquet files to {data_dir}/")
for i, key in enumerate(files):
    s3.download_file(BUCKET, key, f"{data_dir}/{os.path.basename(key)}")
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(files)}")
print(f"Downloaded {len(files)} parquet files")

# 2. Download experiment script
s3.download_file(
    BUCKET,
    "cloud-runs/experiment-scripts/e2e-cnn-classification/run_experiment.py",
    "run_experiment.py",
)
print("Downloaded run_experiment.py")

# 3. Run experiment
os.makedirs(".kit/results/e2e-cnn-classification", exist_ok=True)
rc = subprocess.call([sys.executable, "run_experiment.py"])

# 4. Copy results for bootstrap S3 upload
if os.path.isdir(".kit/results/e2e-cnn-classification"):
    os.system("cp -r .kit/results/e2e-cnn-classification/* results/ 2>/dev/null || true")
    print("Copied results to /work/results/ for S3 sync")

sys.exit(rc)
