"""Run emotion probing experiments with TimesFM on the HKU956 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

from src.experiments.emotion_probing import EmotionProbingExperiment  # noqa: E402
import src.experiments.emotion_probing as ep
print("实际导入的是文件：", ep.__file__)
from src.config import load_config  # noqa: E402

def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser(description="Emotion probing on HKU956")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Configuration file path",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/HKU956"),
        help="HKU956 dataset root directory",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory to save experiment outputs",
    )
    parser.add_argument(
        "--layer_wise",
        action="store_true",
        help="Enable layer-wise probing (slower)",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run a short sanity check",
    )
    return parser.parse_args()


def banner(config, args: argparse.Namespace) -> None:
    """Print a concise run summary."""
    data_root = args.data_root or config.dataset.data_root
    results_dir = args.results_dir or config.experiment.results_dir
    layer_wise = args.layer_wise or config.experiment.layer_wise

    print(f"Data root         : {data_root}")
    print(f"Results directory : {results_dir}")
    print(f"Layer-wise probing: {layer_wise}")
    print(f"Configuration file: {args.config}")


def main() -> int:
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command line overrides
    if args.quick_test:
        config.experiment.quick_test["enabled"] = True

    banner(config, args)

    exp = EmotionProbingExperiment(
        data_root=args.data_root,
        results_dir=args.results_dir,
        config=config,
        layer_wise=args.layer_wise if args.layer_wise else None,
    )
    print("Has method:", hasattr(exp, "apply_quick_test_config"))

    exp.apply_quick_test_config()
    exp.run()


if __name__ == "__main__":
    main()
