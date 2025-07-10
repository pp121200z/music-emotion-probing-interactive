from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import timesfm
from src.data.hku956_dataset import BiophysicalSignals
from src.config import get_config, Config
from typing import Dict, List, Optional


class SignalPreprocessor:
    """Resample to a fixed length and z-score-normalize each modality."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self._RATES = self.config.dataset.sampling_rates
        self.scalers: dict[str, StandardScaler] = {}

    # ── fitting ────────────────────────────────────────────────────────────────
    def fit(self, data: List[BiophysicalSignals]) -> None:
        """Compute per-channel StandardScaler parameters."""
        for k in self._RATES:
            vals = np.concatenate([getattr(d, k) for d in data if getattr(d, k).size])
            if vals.size:
                self.scalers[k] = StandardScaler().fit(vals[:, None])

    # ── helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _resample(x: np.ndarray, length: int) -> np.ndarray:
        if x.size == 0:
            return np.zeros(length, dtype=x.dtype)
        if x.size == length:
            return x
        return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, x.size), x)

    def _norm(self, x: np.ndarray, k: str) -> np.ndarray:
        scaler = self.scalers.get(k)
        return x if scaler is None else scaler.transform(x[:, None])[:, 0]

    # ── public ────────────────────────────────────────────────────────────────
    def __call__(self, sig: BiophysicalSignals, length: int) -> BiophysicalSignals:
        """Resample to *length* and normalise (except IBI)."""
        return BiophysicalSignals(
            bvp=self._norm(self._resample(sig.bvp, length), "bvp"),
            eda=self._norm(self._resample(sig.eda, length), "eda"),
            hr=self._norm(self._resample(sig.hr, length), "hr"),
            ibi=self._resample(sig.ibi, length),  # keep raw
            temp=self._norm(self._resample(sig.temp, length), "temp"),
        )


class TimesFMFeatureExtractor:
    """Thin wrapper around TimesFM for representation extraction."""

    def __init__(
        self,
        config: Optional[Config] = None,
        context: Optional[int] = None,
        gpu: Optional[bool] = None,
        capture_layers: Optional[bool] = None,
    ):
        self.config = config or get_config()

        # Use config values or provided overrides
        self.context = context or self.config.timesfm.context_length
        use_gpu = gpu if gpu is not None else self.config.timesfm.use_gpu
        self.capture_layers = (
            capture_layers
            if capture_layers is not None
            else self.config.timesfm.capture_layers
        )

        self.selected_layers = self.config.timesfm.selected_layers
        self.output_dim = self.config.timesfm.output_dim

        # Determine backend
        if self.config.timesfm.backend == "auto":
            backend = "gpu" if use_gpu and torch.cuda.is_available() else "cpu"
        else:
            backend = self.config.timesfm.backend

        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=self.config.timesfm.per_core_batch_size,
                horizon_len=self.config.timesfm.horizon_length,
                context_len=self.context,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.config.timesfm.model_id
            ),
        )
        self.layer_out: dict[int, np.ndarray] = {}
        self._hooks = []
        if self.capture_layers:
            self._register_hooks(self.config.timesfm.total_layers)

    def _register_hooks(self, n: int) -> None:
        layers = self.model._model.stacked_transformer.layers[:n]
        for i, layer in enumerate(layers):
            self._hooks.append(
                layer.register_forward_hook(
                    lambda _, __, out, idx=i: self.layer_out.__setitem__(
                        idx,
                        (out[0] if isinstance(out, tuple) else out)
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )
            )

    def __del__(self):
        for h in self._hooks:
            h.remove()

    def _pad_crop(self, x: np.ndarray) -> np.ndarray:
        if x.size < self.context:
            return np.pad(x, (0, self.context - x.size))
        return x[: self.context]

    @staticmethod
    def _flatten(x: np.ndarray) -> np.ndarray:
        if x.ndim >= 2:
            x = x.reshape(-1, x.shape[-1]).mean(0)
        return x.ravel()

    def _fix_dim(self, x: np.ndarray, d: Optional[int] = None) -> np.ndarray:
        d = d or self.config.timesfm.feature_dim
        if x.size < d:
            return np.pad(x, (0, d - x.size))
        return x[:d]

    def encode(
        self, x: np.ndarray, targets: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        feature_dim = self.config.timesfm.feature_dim

        if x.size == 0:
            return {layer: np.random.randn(feature_dim) for layer in (targets or [0])}

        self.layer_out.clear()
        pred, _ = self.model.forecast([self._pad_crop(x)], freq=[0])

        if self.capture_layers and self.layer_out:
            keys = targets or self.layer_out.keys()
            return {
                layer: self._fix_dim(self._flatten(self.layer_out[layer]))
                for layer in keys
            }

        # ✅ 拼接多个层的输出
        if self.selected_layers:
            return {
                -1: np.concatenate([
                    self._fix_dim(pred.flatten(), self.output_dim // len(self.selected_layers))
                    for _ in self.selected_layers
                ])
            }

        return {0: self._fix_dim(pred.flatten(), self.output_dim)}



class FeatureExtractorPipeline:
    """Preprocess → TimesFM → concatenate modalities."""

    def __init__(
        self,
        config: Optional[Config] = None,
        context: Optional[int] = None,
        gpu: Optional[bool] = None,
        capture_layers: Optional[bool] = None,
    ):
        self.config = config or get_config()
        self._CHS = tuple(self.config.dataset.modalities)

        self.preproc = SignalPreprocessor(self.config)
        self.encoder = TimesFMFeatureExtractor(
            self.config, context, gpu, capture_layers
        )
        self.fitted = False
        self.capture_layers = (
            capture_layers
            if capture_layers is not None
            else self.config.timesfm.capture_layers
        )

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, data: List[BiophysicalSignals]) -> None:
        self.preproc.fit(data)
        self.fitted = True

    # ── inference ─────────────────────────────────────────────────────────────
    def __call__(
        self, sig: BiophysicalSignals, layers: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        assert self.fitted, "Call fit() first"
        x = self.preproc(sig, self.encoder.context)
        feats = {ch: self.encoder.encode(getattr(x, ch), layers) for ch in self._CHS}

        if self.capture_layers:
            all_layers = set().union(*feats.values())
            feature_dim = self.config.timesfm.feature_dim
            return {
                layer: np.concatenate(
                    [feats[ch].get(layer, np.zeros(feature_dim)) for ch in self._CHS]
                )
                for layer in all_layers
            }

        return {0: np.concatenate([feats[ch][0] for ch in self._CHS])}
