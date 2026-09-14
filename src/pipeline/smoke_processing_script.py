"""Runs INSIDE the SageMaker Processing container, not on this machine.

Trivial by design: reads the tiny sepsis dtype JSON already uploaded to S3
(mounted at /opt/ml/processing/input/), and writes a one-line summary to
/opt/ml/processing/output/. Proves the whole chain works end to end --
container starts, the execution role can read our bucket, the job can
write results back to S3 -- before Stage 7 trusts it with anything real.
"""

import json
from pathlib import Path

INPUT_PATH = Path("/opt/ml/processing/input/sepsis_dtypes.json")
OUTPUT_PATH = Path("/opt/ml/processing/output/summary.txt")


def main() -> None:
    data = json.loads(INPUT_PATH.read_text())
    n_features = len(data["selected_features"])
    summary = f"smoke test OK: read {n_features} selected features for dataset target={data['target']!r}\n"
    print(summary)
    OUTPUT_PATH.write_text(summary)


if __name__ == "__main__":
    main()
