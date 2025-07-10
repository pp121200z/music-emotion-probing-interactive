import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import joblib


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from src.data.hku956_dataset import HKU956Dataset
from src.models.timesfm_extractor import FeatureExtractorPipeline
from src.config import get_config, Config


@dataclass
class ExperimentResults:
    """Container for a single (layer, model, modality) experiment"""

    layer: Optional[int]
    target: str
    modalities: List[str]
    model_type: str
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    fold_scores: List[Dict[str, float]]


class EmotionProbingExperiment:
    """Emotion classification / probing on HKU956 with TimesFM features"""

    def __init__(
        self,
        data_root: Optional[Path | str] = None,
        results_dir: Optional[Path | str] = None,
        *,
        config: Optional[Config] = None,
        layer_wise: Optional[bool] = None,
        n_folds: Optional[int] = None,
    ) -> None:
        self.config = config or get_config()

        # Use config defaults if not provided
        data_root = data_root or self.config.dataset.data_root
        results_dir = results_dir or self.config.experiment.results_dir
        layer_wise = (
            layer_wise if layer_wise is not None else self.config.experiment.layer_wise
        )
        n_folds = n_folds or self.config.experiment.n_folds

        self.dataset = HKU956Dataset(Path(data_root), self.config)
        self.feature_extractor = FeatureExtractorPipeline(
            config=self.config, capture_layers=layer_wise
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create cache directory for recordings
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.layer_wise = layer_wise
        if self.layer_wise:
            layers_to_probe = self.config.experiment.layers_to_probe
            self.layers = layers_to_probe or list(
                range(self.config.timesfm.total_layers)
            )
        else:
            self.layers = [0]

        self.modalities = self.config.dataset.modalities
        self.n_folds = n_folds

        # Build models from config
        self.models = {}
        for name, model_config in self.config.models.items():
            if model_config.type == "LogisticRegression":
                self.models[name] = LogisticRegression(**model_config.params)
            elif model_config.type == "SVC":
                self.models[name] = SVC(**model_config.params)
            elif model_config.type == "MLPClassifier":
                self.models[name] = MLPClassifier(**model_config.params)
            else:
                raise ValueError(f"Unknown model type: {model_config.type}")

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _cross_validate(
        self, X: np.ndarray, y: np.ndarray, model, k: int
    ) -> Dict[str, float]:
        """Return aggregate metrics & per–fold breakdown."""

        skf = StratifiedKFold(
            n_splits=k,
            shuffle=self.config.experiment.shuffle,
            random_state=self.config.experiment.random_state,
        )
        scaler = StandardScaler()
        metrics: Dict[str, list[float]] = {
            m: [] for m in ("accuracy", "f1", "precision", "recall")
        }

        for train, test in tqdm(skf.split(X, y), total=k, desc="CV folds", leave=False):
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])

            model.fit(X_train, y[train])
            pred = model.predict(X_test)

            metrics["accuracy"].append(accuracy_score(y[test], pred))
            metrics["f1"].append(f1_score(y[test], pred, zero_division=0))
            metrics["precision"].append(precision_score(y[test], pred, zero_division=0))
            metrics["recall"].append(recall_score(y[test], pred, zero_division=0))

        summary = {
            f"{m}_{s}": getattr(np, s)(vals)
            for m, vals in metrics.items()
            for s in ("mean", "std")
        }
        summary["fold_scores"] = [{m: metrics[m][i] for m in metrics} for i in range(k)]
        # Save the model from the last fold
        import joblib

        joblib.dump(model, "saved_probe_model.joblib")
        print("Saved trained probe model to saved_probe_model.joblib")

        return summary

    def _prepare_recordings(self) -> List[Tuple]:
        valence_lbls = self.dataset.get_binary_labels("valence")
        arousal_lbls = self.dataset.get_binary_labels("arousal")
        valid = []
        for pid, song_no, song_id in self.dataset.get_all_valid_recordings():
            key = f"{pid}_{song_no}_{song_id}"
            if key in valence_lbls and key in arousal_lbls:
                valid.append((pid, song_no, song_id))
        return valid

    def _get_cache_filename(self, recordings: List[Tuple]) -> str:
        """Generate cache filename based on recordings and config"""
        layer_str = "layerwise" if self.layer_wise else "simple"
        max_rec = self.config.experiment.max_recordings or "all"
        return f"features_{layer_str}_{max_rec}.pkl"

    def _save_extracted_features(
        self, recordings: List[Tuple], features: Dict[str, Dict[int, np.ndarray]]
    ) -> None:
        """Save extracted features and recordings to cache"""
        cache_file = self.cache_dir / self._get_cache_filename(recordings)
        cache_data = {
            "recordings": recordings,
            "features": features,
            "config_hash": hash(str(self.config)),
            "layer_wise": self.layer_wise,
            "layers": self.layers,
            "modalities": self.modalities,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"Saved extracted features to {cache_file}")

    def _load_extracted_features(
        self, recordings: List[Tuple]
    ) -> Optional[Dict[str, Dict[int, np.ndarray]]]:
        """Load extracted features from cache if available and valid"""
        cache_file = self.cache_dir / self._get_cache_filename(recordings)

        if not cache_file.exists():
            return None

        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        # Verify cache is still valid
        if (
            cache_data["recordings"] == recordings
            and cache_data["layer_wise"] == self.layer_wise
            and cache_data["layers"] == self.layers
            and cache_data["modalities"] == self.modalities
        ):
            print(f"Loaded cached features from {cache_file}")
            return cache_data["features"]

        return None

    # ------------------------------------------------------------------
    # Feature pipeline
    # ------------------------------------------------------------------
    def _extract_features(
        self, recordings: List[Tuple]
    ) -> Dict[str, Dict[int, np.ndarray]]:
        # Try to load from cache first
        cached_features = self._load_extracted_features(recordings)
        if cached_features is not None:
            return cached_features

        # Fit preprocessor
        signals = [self.dataset.load_signals_for_recording(*r) for r in recordings]
        self.feature_extractor.fit([s for s in signals if s is not None])

        features: Dict[str, Dict[int, np.ndarray]] = {}
        for rec, sig in tqdm(
            zip(recordings, signals), total=len(recordings), desc="Extracting features"
        ):
            if sig is None:
                continue
            key = f"{rec[0]}_{rec[1]}_{rec[2]}"
            if self.layer_wise:
                features[key] = self.feature_extractor(sig, self.layers)
            else:
                features[key] = self.feature_extractor(sig)

        # Save to cache
        self._save_extracted_features(recordings, features)
        return features

    def _ablate_modalities(
        self, feats: Dict[str, Dict[int, np.ndarray]], exclude: List[str]
    ) -> Dict[str, Dict[int, np.ndarray]]:
        if not exclude:
            return feats

        feature_dim = self.config.timesfm.feature_dim
        keep_idx = [i for i, m in enumerate(self.modalities) if m not in exclude]

        return {
            k: {
                layer: np.concatenate(
                    [v[i * feature_dim : (i + 1) * feature_dim] for i in keep_idx]
                )
                for layer, v in layers.items()
            }
            for k, layers in feats.items()
        }

    # ------------------------------------------------------------------
    # Core experiment
    # ------------------------------------------------------------------
    def _run_single_setting(
        self,
        feats: Dict[str, Dict[int, np.ndarray]],
        labels: Dict[str, int],
        target: str,
        exclude: List[str],
    ) -> List[ExperimentResults]:
        feats = self._ablate_modalities(feats, exclude)
        keys = [k for k in feats if k in labels]
        res: List[ExperimentResults] = []

        modality_desc = "all" if not exclude else f"without {','.join(exclude)}"

        for layer in tqdm(self.layers, desc=f"{target} ({modality_desc}) - layers"):
            X = np.stack([feats[k][layer] for k in keys])
            y = np.array([labels[k] for k in keys])

            for model_name, model in self.models.items():
                stats = self._cross_validate(X, y, model, self.n_folds)
                model_path = self.results_dir / f"{target}_probe_model.joblib" # target is "valence" or "arousal"
                print(f"saving model to:{model_path}")
                joblib.dump(model, str(model_path)) 
                
                res.append(
                    ExperimentResults(
                        layer if self.layer_wise else None,
                        target,
                        [m for m in self.modalities if m not in exclude],
                        model_name,
                        **stats,
                    )
                )

        return res
    
    def run(self) -> List[ExperimentResults]:
        recordings = self._prepare_recordings()
        max_recordings = self.config.experiment.max_recordings
        if max_recordings is not None:
            recordings = recordings[:max_recordings]

        print(f"Found {len(recordings)} valid recordings")

        feats = self._extract_features(recordings)
        results: List[ExperimentResults] = []

        targets = self.config.experiment.targets
        for target in tqdm(targets, desc="Processing targets"):
            labels = self.dataset.get_binary_labels(target)
            results += self._run_single_setting(feats, labels, target, [])
            for m in tqdm(self.modalities, desc=f"{target} ablation studies", leave=False):
                results += self._run_single_setting(feats, labels, target, [m])

        out_file = self.results_dir / "results.json"
        out_file.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print("Saved results to", out_file)
        return results


    def apply_quick_test_config(self) -> None:
        """Apply quick test configuration if enabled."""
        if self.config.experiment.quick_test.get("enabled", False):
            print("Applying quick test configuration...")
            self.n_folds = self.config.experiment.quick_test["n_folds"]
            if self.layer_wise and "layers" in self.config.experiment.quick_test:
                self.layers = self.config.experiment.quick_test["layers"]
                
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Emotion probing / classification experiment"
    )
    p.add_argument("--data_root", type=Path, help="Dataset root directory")
    p.add_argument("--results_dir", type=Path, help="Results output directory")
    p.add_argument(
        "--layer-wise", action="store_true", help="enable layer-wise probing"
    )
    p.add_argument("--quick-test", action="store_true", help="enable quick test mode")
    p.add_argument(
        "--config", type=Path, default="config.yaml", help="Configuration file path"
    )
    args = p.parse_args()

    # Load config and override with CLI args if provided
    from src.config import load_config

    config = load_config(args.config)

    if args.quick_test:
        config.experiment.quick_test["enabled"] = True

    exp = EmotionProbingExperiment(
        data_root=args.data_root,
        results_dir=args.results_dir,
        config=config,
        layer_wise=args.layer_wise,
    )
    exp.apply_quick_test_config()
    exp.run()
