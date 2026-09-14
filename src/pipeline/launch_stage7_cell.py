"""Stage 7: submit one (dataset, model, XAI method) cell as a real
SageMaker Processing Job on the instance type chosen in configs/aws.yaml.

Uses FrameworkProcessor(estimator_cls=PyTorch) rather than plain
SKLearnProcessor: our container needs torch (pre-installed by the PyTorch
framework image, saving a from-scratch torch pip install on every one of
the 27 jobs) plus xgboost/shap/lime/rtdl_revisiting_models, installed via
src/requirements.txt. source_dir=src/ ships the whole pipeline package into
the container; run_stage7_cell.py is the entry point (see its own
docstring for why it can't be src/pipeline/explain.py directly).

data/processed/, models/, and configs/ are mounted as sibling input
channels to the code channel (not nested under it) -- source_dir=src/
uploads src/'s *contents* straight into /opt/ml/processing/input/code/, so
pipeline/ sits one level higher inside the container than locally (no
src/ layer above it). config.py's REPO_ROOT = parents[2] from
pipeline/config.py therefore resolves to /opt/ml/processing/input/ itself,
not .../input/code/ -- mount data/processed/, models/, and configs/ there
to match, and REPO_ROOT-relative paths in the pipeline package resolve
exactly the way they do locally.

configs/models.yaml specifically is required by torch_inference.py to
reconstruct the LSTM/FT-Transformer architecture -- missing this mount
made every LSTM/FT-Transformer cell fail (FileNotFoundError) while
XGBoost cells succeeded, since XGBoost's saved model is self-contained
and never reads this file. Found the hard way during the first real
27-cell sweep attempt; see CLAUDE.md's Stage 7 notes.

Usage:
    python -m src.pipeline.launch_stage7_cell <dataset> <model_type> <method>
"""

from __future__ import annotations

import sys

import boto3
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.processing import FrameworkProcessor
from sagemaker.session import Session

from .config import REPO_ROOT
from .upload_to_s3 import load_aws_config

PYTORCH_FRAMEWORK_VERSION = "2.3"
PYTHON_VERSION = "py311"  # matches the local venv's Python 3.11 exactly, so src/requirements.txt's pinned versions resolve


def launch_cell(dataset: str, model_type: str, method: str, wait: bool = True) -> str:
    aws_config = load_aws_config()
    bucket = aws_config["s3_bucket"]
    region = aws_config["region"]
    role_arn = aws_config["sagemaker_role_arn"]
    instance_type = aws_config["instance_type"]

    sagemaker_session = Session(boto_session=boto3.Session(region_name=region))

    processor = FrameworkProcessor(
        estimator_cls=PyTorch,
        framework_version=PYTORCH_FRAMEWORK_VERSION,
        py_version=PYTHON_VERSION,
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        base_job_name=f"stage7-{dataset}-{model_type}-{method}".replace("_", "-"),
    )

    output_s3_uri = f"s3://{bucket}/results/stage7/{dataset}/{model_type}/{method}"

    print(f"Launching {dataset}/{model_type}/{method} on {instance_type} in {region}...")
    processor.run(
        code="run_stage7_cell.py",
        source_dir=str(REPO_ROOT / "src"),
        inputs=[
            ProcessingInput(source=f"s3://{bucket}/data/processed", destination="/opt/ml/processing/input/data/processed"),
            ProcessingInput(source=f"s3://{bucket}/models", destination="/opt/ml/processing/input/models"),
            ProcessingInput(source=f"s3://{bucket}/configs", destination="/opt/ml/processing/input/configs"),
        ],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination=output_s3_uri)],
        arguments=[dataset, model_type, method, "/opt/ml/processing/output"],
        wait=wait,
        logs=wait,
    )

    job_name = processor.jobs[-1].describe()["ProcessingJobName"]
    print(f"Job name: {job_name}")
    print(f"Output:   {output_s3_uri}")
    return job_name


if __name__ == "__main__":
    launch_cell(sys.argv[1], sys.argv[2], sys.argv[3])
