#!/usr/bin/env python3
"""
Script to explore HKU956 dataset and display statistics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from hku956_dataset import HKU956Dataset


def main():
    data_root = Path("data/HKU956")

    if not data_root.exists():
        print(f"Error: Dataset not found at {data_root}")
        return 1

    print("=== HKU956 Dataset Exploration ===")

    # Initialize dataset
    dataset = HKU956Dataset(data_root)

    # Display basic statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Total recordings: {stats['total_recordings']}")
    print(f"  Participants: {stats['participants']}")
    print(
        f"  Valence - Positive: {stats['valence_positive']}, Negative: {stats['valence_negative']}"
    )
    print(
        f"  Arousal - Positive: {stats['arousal_positive']}, Negative: {stats['arousal_negative']}"
    )

    # Show sample recordings
    print(f"\nParticipants: {dataset.participants[:10]}...")  # Show first 10

    # Show sample emotion ratings
    print("\nSample emotion ratings:")
    print(dataset.emotion_ratings.head())

    # Test loading a sample recording
    print("\nTesting signal loading...")
    valid_recordings = dataset.get_all_valid_recordings()
    if valid_recordings:
        participant_id, song_no, song_id = valid_recordings[0]
        print(f"Loading signals for: {participant_id}, song {song_no}, ID {song_id}")

        signals = dataset.load_signals_for_recording(participant_id, song_no, song_id)
        if signals:
            print(f"  BVP length: {len(signals.bvp)}")
            print(f"  EDA length: {len(signals.eda)}")
            print(f"  HR length: {len(signals.hr)}")
            print(f"  IBI length: {len(signals.ibi)}")
            print(f"  TEMP length: {len(signals.temp)}")
        else:
            print("  Failed to load signals")

    # Display emotion label distribution
    valence_labels = dataset.get_binary_labels("valence")
    arousal_labels = dataset.get_binary_labels("arousal")

    print("\nLabel Distribution:")
    print(
        f"  Valence: {sum(valence_labels.values())} positive, {len(valence_labels) - sum(valence_labels.values())} negative"
    )
    print(
        f"  Arousal: {sum(arousal_labels.values())} positive, {len(arousal_labels) - sum(arousal_labels.values())} negative"
    )

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
