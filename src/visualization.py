import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import get_config, Config


class ResultsVisualizer:
    def __init__(
        self, results_dir: Optional[Path] = None, config: Optional[Config] = None
    ):
        self.config = config or get_config()
        self.results_dir = Path(results_dir or self.config.experiment.results_dir)
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def load_results(self, result_file: str) -> List[Dict]:
        file_path = self.results_dir / result_file
        if not file_path.exists():
            raise FileNotFoundError(f"Result file not found: {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    def plot_layer_performance(
        self,
        target: str = "valence",
        save_path: Optional[Path] = None,
        model: Optional[str] = None,
    ) -> None:
        results = self.load_results("results.json")

        # Create subfolder for layer performance plots
        if save_path:
            layer_dir = save_path.parent / "layer_performance"
            layer_dir.mkdir(exist_ok=True, parents=True)

        for metric in ["f1_mean", "accuracy_mean"]:
            layer_data = [
                {
                    "layer": r["layer"],
                    "model_type": r["model_type"],
                    "metric_value": r[metric],
                    "metric_std": r[metric.replace("_mean", "_std")],
                }
                for r in results
                if (
                    r["target"] == target
                    and r["layer"] is not None
                    and len(r["modalities"]) == 4
                    and r["model_type"] == model
                    and metric in r
                )
            ]

            df = pd.DataFrame(layer_data)
            if df.empty:
                continue

            plt.figure(figsize=(12, 8))

            for model_type in df["model_type"].unique():
                model_data = df[df["model_type"] == model_type]
                plt.errorbar(
                    model_data["layer"],
                    model_data["metric_value"],
                    yerr=model_data["metric_std"],
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    capsize=5,
                )

            plt.xlabel("Layer Index", fontsize=14)
            plt.ylabel(f"{metric.replace('_', ' ').title()}", fontsize=14)
            plt.ylim(0, 1)

            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            title = f"{target.capitalize()} Classification: {metric.replace('_', ' ').title()} by Layer"
            plt.title(title, fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                metric_name = metric.replace("_mean", "")
                filename = f"{target}_{metric_name}_{model}.png"
                plt.savefig(layer_dir / filename, dpi=300, bbox_inches="tight")

            plt.close()

    def plot_modality_ablation(
        self,
        target: str = "valence",
        save_path: Optional[Path] = None,
        model: Optional[str] = None,
    ) -> None:
        results = self.load_results("results.json")
        all_modalities = {"bvp", "eda", "hr", "temp"}

        # Create subfolder for modality ablation plots
        if save_path:
            ablation_dir = save_path.parent / "modality_ablation"
            ablation_dir.mkdir(exist_ok=True, parents=True)

        for metric in ["f1_mean", "accuracy_mean"]:
            ablation_data = []
            for r in results:
                if (
                    r["target"] == target
                    and r["model_type"] == model
                    and r["layer"] is not None
                    and metric in r
                ):
                    present = set(r["modalities"])
                    missing = all_modalities - present

                    if len(missing) == 0:
                        condition = "All modalities"
                    elif len(missing) == 1:
                        condition = f"Without {list(missing)[0].upper()}"
                    else:
                        continue

                    ablation_data.append(
                        {
                            "layer": r["layer"],
                            "condition": condition,
                            "model_type": r["model_type"],
                            "metric_value": r[metric],
                            "metric_std": r[metric.replace("_mean", "_std")],
                        }
                    )

            df = pd.DataFrame(ablation_data)
            if df.empty:
                continue

            plt.figure(figsize=(12, 8))

            # Group by condition and plot each as a line
            for condition in df["condition"].unique():
                condition_data = df[df["condition"] == condition].sort_values("layer")
                plt.errorbar(
                    condition_data["layer"],
                    condition_data["metric_value"],
                    yerr=condition_data["metric_std"],
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    label=condition,
                    capsize=5,
                )

            plt.xlabel("Layer Index", fontsize=14)
            plt.ylabel(f"{metric.replace('_', ' ').title()}", fontsize=14)
            plt.ylim(0, 1)

            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            title = f"{target.capitalize()} Classification: {metric.replace('_', ' ').title()} Modality Ablation by Layer"
            plt.title(title, fontsize=16)

            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                metric_name = metric.replace("_mean", "")
                filename = f"{target}_{metric_name}_{model}.png"
                plt.savefig(ablation_dir / filename, dpi=300, bbox_inches="tight")

            plt.close()
