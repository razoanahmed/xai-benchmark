"""SageMaker Processing entry point for one Stage 7 cell.

Not src/pipeline/explain.py directly -- that module uses relative imports
(`from .export import ...`), which only work when it's imported as part of
the pipeline package, not when SageMaker invokes it as a bare script
(`python explain.py`, with no package context). This tiny launcher is the
sibling-of-the-package trick that fixes that: it does a real package import,
so pipeline/'s internal relative imports resolve normally.

XGBoost cells get shap upgraded to 0.51.0 (+ numpy>=2) here, on top of
src/requirements.txt's baseline pin of numpy<2 + shap==0.49.1. That
baseline exists because this container's pre-built torch breaks under
numpy>=2 (see requirements.txt's comment), and shap>=0.50 hard-requires
numpy>=2 -- but shap==0.49.1's TreeExplainer has its own separate bug:
it can't parse some of XGBoost 3.2.0's tree JSON (a scientific-notation
split threshold with no decimal point crashes with "could not convert
string to float"). XGBoost inference never imports torch (see
xgboost_inference.py), so upgrading numpy back to 2.x is completely safe
for exactly this one model type -- verified by sepsis/xgboost/shap
succeeding earlier in the Stage 7 sweep under shap==0.51.0/numpy>=2,
before the lstm/ft_transformer numpy<2 pin was even added.
"""

import subprocess
import sys

if len(sys.argv) > 2 and sys.argv[2] == "xgboost":
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=2", "shap==0.51.0"], check=True)

from pipeline.explain import main

if __name__ == "__main__":
    main()
