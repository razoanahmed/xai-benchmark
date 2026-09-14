"""Stage 6.5: launch a trivial SageMaker Processing Job to prove the chain
works end to end (execution role, S3 read/write, container, instance type)
before Stage 7 trusts it with the real 27-cell sweep.

Reads smoke_processing_script.py's input from the sepsis dtype JSON already
uploaded to S3, and writes its one-line output back to S3 under
results/smoke_test/. This script then reads that output back locally to
confirm the round trip actually happened, not just that the job reported
success.

Usage:
    python -m src.pipeline.launch_smoke_test
"""

from __future__ import annotations

import boto3
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.session import Session
from sagemaker.sklearn.processing import SKLearnProcessor

from .config import REPO_ROOT
from .upload_to_s3 import load_aws_config


def main() -> None:
    aws_config = load_aws_config()
    bucket = aws_config["s3_bucket"]
    region = aws_config["region"]
    role_arn = aws_config["sagemaker_role_arn"]
    instance_type = aws_config["instance_type"]

    boto_session = boto3.Session(region_name=region)
    sagemaker_session = Session(boto_session=boto_session)

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=sagemaker_session,
        base_job_name="xai-benchmark-smoke-test",
    )

    input_s3_uri = f"s3://{bucket}/data/processed/sepsis_dtypes.json"
    output_s3_uri = f"s3://{bucket}/results/smoke_test"

    print(f"Launching Processing Job on {instance_type} in {region}...")
    print(f"  input:  {input_s3_uri}")
    print(f"  output: {output_s3_uri}")

    processor.run(
        code=str(REPO_ROOT / "src" / "pipeline" / "smoke_processing_script.py"),
        inputs=[ProcessingInput(source=input_s3_uri, destination="/opt/ml/processing/input")],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination=output_s3_uri)],
        wait=True,
        logs=True,
    )

    s3 = boto3.client("s3", region_name=region)
    obj = s3.get_object(Bucket=bucket, Key="results/smoke_test/summary.txt")
    result_text = obj["Body"].read().decode()
    print(f"\nRead back from S3: {result_text.strip()}")
    print("Smoke test PASSED -- round trip confirmed.")


if __name__ == "__main__":
    main()
