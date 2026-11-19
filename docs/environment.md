# Environment

## Pinned dependencies

* `requirements.txt` lists every **direct** dependency with an exact `==` pin (Python 3.13).
* `requirements-lock.txt` is `pip freeze` of the resolved cluster environment: **every**
  transitive package, exactly as the tracked runs used it. It is also attached to every
  training run in MLflow (`environment/requirements-lock.txt`).
* The versions were chosen as the latest stable releases at project start (2026-11-19) and
  verified to resolve together (`pip check`: no broken requirements).

## PyTorch build

* Cluster: `torch==2.14.1+cu126`. The DeepCluster onboarding guide recommends the CUDA 12.6
  build because newer builds (cu128/cu130) drop the TITAN X; cu118 builds stop at torch 2.7.1.
* Laptop and CI: the **same torch version** as a CPU build (`2.14.1+cpu`). CI runners have no
  GPU, and the CUDA wheels add ~4 GB of downloads. The CPU requirements are derived
  mechanically from `requirements.txt` (`sed s/+cu126/+cpu/`), so the two can never drift apart.

## No extra packages

The download uses `urllib` + `zipfile` from the standard library and the FFT features use
NumPy. `scipy` is pinned only because scikit-learn depends on it (pinning it explicitly keeps it
from floating); the project code does not import it.
