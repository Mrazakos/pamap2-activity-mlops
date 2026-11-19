"""Self-contained model bundles and a common prediction interface.

A bundle folder holds everything needed for inference and for tracing the
model back to its run::

    bundle.json   kind, channels, classes, normalisation stats, window geometry,
                  resolved config, MLflow run id, weights SHA-256
    model.pt      PyTorch state dict          (kind = "torch")
    model.joblib  fitted scikit-learn model   (kind = "sklearn")
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import torch
from sklearn.base import ClassifierMixin
from torch import nn

from src.utils import read_json, sha256_file, write_json

FloatArray = npt.NDArray[np.float32]
BUNDLE_FILE = "bundle.json"


@dataclass
class BundleMeta:
    """Metadata stored next to the weights."""

    kind: str
    config: dict[str, Any]
    channels: list[str]
    class_ids: list[int]
    class_names: list[str]
    norm_mean: list[float]
    norm_std: list[float]
    window_length: int
    sample_rate_hz: float
    run_id: str = ""
    weights_sha256: str = ""
    feature_names: list[str] = field(default_factory=list)


def state_dict_sha256(model: nn.Module) -> str:
    """Hash the model weights independently of file format and device.

    Args:
        model: PyTorch model.

    Returns:
        SHA-256 over all state-dict tensors in sorted key order.
    """
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_torch_bundle(directory: Path, model: nn.Module, meta: BundleMeta) -> BundleMeta:
    """Write a PyTorch bundle.

    Args:
        directory: Target folder (created).
        model: Trained model.
        meta: Metadata (``weights_sha256`` is filled in).

    Returns:
        The completed metadata.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / "model.pt")
    meta.kind = "torch"
    meta.weights_sha256 = state_dict_sha256(model)
    write_json(directory / BUNDLE_FILE, asdict(meta))
    return meta


def save_sklearn_bundle(directory: Path, model: ClassifierMixin, meta: BundleMeta) -> BundleMeta:
    """Write a scikit-learn bundle.

    Args:
        directory: Target folder (created).
        model: Fitted estimator.
        meta: Metadata (``weights_sha256`` is filled in).

    Returns:
        The completed metadata.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib", compress=3)
    meta.kind = "sklearn"
    meta.weights_sha256 = sha256_file(directory / "model.joblib")
    write_json(directory / BUNDLE_FILE, asdict(meta))
    return meta


class Predictor:
    """Uniform ``predict_proba`` over torch and scikit-learn bundles.

    Inputs are *normalised* windows ``(n, channels, length)``; use
    :meth:`normalize` to convert raw sensor units first.
    """

    def __init__(self, meta: BundleMeta, model: nn.Module | ClassifierMixin, device: str = "cpu") -> None:
        """Wrap a loaded model.

        Args:
            meta: Bundle metadata.
            model: Loaded model.
            device: Torch device for inference.
        """
        self.meta = meta
        self.model = model
        self.device = torch.device(device)
        self.mean = np.asarray(meta.norm_mean, dtype=np.float32)
        self.std = np.asarray(meta.norm_std, dtype=np.float32)

    @property
    def n_classes(self) -> int:
        """Number of output classes."""
        return len(self.meta.class_ids)

    def normalize(self, raw: FloatArray) -> FloatArray:
        """Standardise raw windows with the training statistics.

        Args:
            raw: ``(n, channels, length)`` in sensor units.

        Returns:
            Normalised windows.
        """
        return ((raw - self.mean[None, :, None]) / self.std[None, :, None]).astype(np.float32)

    def predict_proba(self, x: FloatArray, batch_size: int = 512) -> npt.NDArray[np.float64]:
        """Class probabilities for normalised windows.

        Args:
            x: ``(n, channels, length)``.
            batch_size: Inference batch size (torch only).

        Returns:
            ``(n, n_classes)`` probabilities.
        """
        if isinstance(self.model, nn.Module):
            self.model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(x), batch_size):
                    batch = torch.from_numpy(np.ascontiguousarray(x[start : start + batch_size])).to(
                        self.device
                    )
                    chunks.append(torch.softmax(self.model(batch), dim=1).cpu().double().numpy())
            return np.concatenate(chunks) if chunks else np.empty((0, self.n_classes))
        return self._sklearn_proba(x)

    def _sklearn_proba(self, x: FloatArray) -> npt.NDArray[np.float64]:
        """Compute features, predict and map to the full class set."""
        from src.config import FeatureConfig
        from src.features import features_frame

        feats, _ = features_frame(
            x, self.meta.channels, self.meta.sample_rate_hz, FeatureConfig(**self.meta.config["features"])
        )
        partial = self.model.predict_proba(feats)  # type: ignore[union-attr]
        proba = np.zeros((len(x), self.n_classes))
        proba[:, self.model.classes_] = partial  # type: ignore[union-attr]
        return proba

    def predict(self, x: FloatArray) -> npt.NDArray[np.int64]:
        """Predicted class indices.

        Args:
            x: Normalised windows.

        Returns:
            ``(n,)`` class indices.
        """
        return self.predict_proba(x).argmax(axis=1).astype(np.int64)


def load_bundle(directory: Path, device: str = "cpu") -> Predictor:
    """Load a bundle written by :func:`save_torch_bundle` or :func:`save_sklearn_bundle`.

    Args:
        directory: Bundle folder.
        device: Torch device.

    Returns:
        A ready-to-use predictor.
    """
    from src.config import ModelConfig
    from src.models import build_model

    directory = Path(directory)
    meta = BundleMeta(**read_json(directory / BUNDLE_FILE))
    if meta.kind == "sklearn":
        return Predictor(meta, joblib.load(directory / "model.joblib"))
    model = build_model(ModelConfig(**meta.config["model"]), len(meta.channels), len(meta.class_ids))
    state = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return Predictor(meta, model, device)
