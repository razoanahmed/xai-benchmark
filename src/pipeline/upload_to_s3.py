"""Stage 6.5: upload the processed data (data/processed/), the 9 full-scale
trained models (models/), and configs/models.yaml to the S3 bucket in
configs/aws.yaml.

Deliberately skips models/dev/ -- those are dev-scale artifacts (see
models.py's _output_dir) that must never be mistaken for the real,
full-scale models the Stage 6 results table describes.

configs/models.yaml is needed by Stage 7's SageMaker jobs: torch_inference.py
reads it to reconstruct the LSTM/FT-Transformer architecture (hidden size,
layers, etc.) before loading the saved weights. XGBoost's own saved model
file is self-contained and doesn't need it -- easy to miss, since a job
that skips this file only breaks for two of the three model types.

Credentials come from whatever the AWS CLI already has configured on this
machine (`aws configure` / SSO) -- boto3 picks those up automatically.
Nothing in this repo ever holds an access key or secret key.

Usage:
    python -m src.pipeline.upload_to_s3
"""

from __future__ import annotations

from pathlib import Path

import boto3
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AWS_CONFIG_PATH = REPO_ROOT / "configs" / "aws.yaml"


def load_aws_config() -> dict:
    with open(AWS_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _files_to_upload() -> list[tuple[Path, str]]:
    """Returns (local_path, s3_key) pairs, mirroring the repo's relative
    layout under s3://<bucket>/.
    """
    pairs = []

    processed_dir = REPO_ROOT / "data" / "processed"
    for path in sorted(processed_dir.glob("*")):
        if path.name == ".gitkeep" or not path.is_file():
            continue
        pairs.append((path, f"data/processed/{path.name}"))

    models_dir = REPO_ROOT / "models"
    for path in sorted(models_dir.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if "dev" in path.relative_to(models_dir).parts:
            continue  # never upload dev-scale artifacts
        pairs.append((path, f"models/{path.relative_to(models_dir)}"))

    pairs.append((REPO_ROOT / "configs" / "models.yaml", "configs/models.yaml"))

    return pairs


def main() -> None:
    aws_config = load_aws_config()
    bucket = aws_config["s3_bucket"]
    region = aws_config["region"]

    s3 = boto3.client("s3", region_name=region)

    pairs = _files_to_upload()
    total_bytes = sum(path.stat().st_size for path, _ in pairs)
    print(f"Uploading {len(pairs)} files ({total_bytes / 1e6:.1f} MB) to s3://{bucket}/ (region {region})")

    for path, key in pairs:
        size_mb = path.stat().st_size / 1e6
        print(f"  {path.relative_to(REPO_ROOT)} -> s3://{bucket}/{key} ({size_mb:.1f} MB)")
        s3.upload_file(str(path), bucket, key)

    print("Done.")


if __name__ == "__main__":
    main()
