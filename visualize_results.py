#!/usr/bin/env python3
"""
Script to visualize emotion probing experiment results
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.visualization import ResultsVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize emotion probing results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots", help="Directory to save plots"
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["valence", "arousal", "both"],
        default="both",
        help="Which emotion target to visualize",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["layer", "ablation", "comparison", "cv", "summary", "all"],
        default="all",
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "svm", "mlp"],
        default="linear",
        help="Model to use for visualization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize visualizer
    visualizer = ResultsVisualizer(results_dir=args.results_dir)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Determine targets
    targets = ["valence", "arousal"] if args.target == "both" else [args.target]

    print("Creating visualization plots...")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Targets: {targets}")
    print(f"Plot type: {args.plot_type}")
    print(f"Model: {args.model}")
    print("-" * 50)

    for target in targets:
        if args.plot_type in ["layer", "all"]:
            visualizer.plot_layer_performance(
                target=target,
                save_path=output_dir
                / f"{target}_layer_performance{f'_{args.model}' if args.model else ''}.png",
                model=args.model,
            )

        if args.plot_type in ["ablation", "all"]:
            visualizer.plot_modality_ablation(
                target=target,
                save_path=output_dir
                / f"{target}_modality_ablation{f'_{args.model}' if args.model else ''}.png",
                model=args.model,
            )

    # List generated files
    plot_files = list(output_dir.glob("*.png"))
    if plot_files:
        print("\nGenerated files:")
        for file in sorted(plot_files):
            print(f"  - {file.name}")


if __name__ == "__main__":
    main()
