from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.config import get_config, Config


@dataclass
class BiophysicalSignals:
    """Container for biophysical signal data"""

    bvp: np.ndarray  # Blood Volume Pulse (64Hz)
    eda: np.ndarray  # Electrodermal Activity (4Hz)
    hr: np.ndarray  # Heart Rate (1Hz)
    ibi: np.ndarray  # Inter-beat Interval (variable)
    temp: np.ndarray  # Skin Temperature (4Hz)


@dataclass
class EmotionRating:
    """Container for emotion rating data"""

    participant_id: str
    song_no: int
    song_id: int
    valence_rating: float
    valence: str  # 'positive', 'negative', 'neutral'
    arousal_rating: float
    arousal: str  # 'positive', 'negative', 'neutral'
    play_duration: int


class HKU956Dataset:
    """HKU956 dataset loader and preprocessor"""

    def __init__(
        self, data_root: Optional[Path] = None, config: Optional[Config] = None
    ):
        self.config = config or get_config()

        # Use provided data_root or config default
        if data_root is None:
            data_root = Path(self.config.dataset.data_root)

        self.data_root = Path(data_root)
        self.physio_signals_dir = (
            self.data_root / self.config.dataset.physio_signals_dir
        )
        self.av_ratings_file = self.data_root / self.config.dataset.av_ratings_file

        # Load emotion ratings once
        self.emotion_ratings = self._load_emotion_ratings()

        # Get all participant directories
        self.participants = [
            d.name for d in self.physio_signals_dir.iterdir() if d.is_dir()
        ]

    def _load_emotion_ratings(self) -> pd.DataFrame:
        """Load emotion ratings from CSV"""
        return pd.read_csv(self.av_ratings_file)

    def _load_signal_file(self, file_path: Path) -> np.ndarray:
        """Load a single signal file and return as numpy array"""
        if not file_path.exists():
            return np.array([])

        # Check if file is empty
        if file_path.stat().st_size == 0:
            return np.array([])

        data = pd.read_csv(file_path, header=None)

        # Check if dataframe is empty
        if data.empty or len(data.columns) == 0:
            return np.array([])

        return data.iloc[:, 0].values

    def load_signals_for_recording(
        self, participant_id: str, song_no: int, song_id: int
    ) -> Optional[BiophysicalSignals]:
        """Load all biophysical signals for a specific recording"""
        participant_dir = self.physio_signals_dir / participant_id

        if not participant_dir.exists():
            return None

        # Construct filename: {song_no}_{song_id}.csv
        filename = f"{song_no}_{song_id}.csv"

        # Load each signal type
        bvp = self._load_signal_file(participant_dir / "BVP" / filename)
        eda = self._load_signal_file(participant_dir / "EDA" / filename)
        hr = self._load_signal_file(participant_dir / "HR" / filename)
        ibi = self._load_signal_file(participant_dir / "IBI" / filename)
        temp = self._load_signal_file(participant_dir / "TEMP" / filename)

        # Check if essential signals were loaded successfully (BVP, EDA, HR, TEMP)
        # IBI can be empty as it has variable output
        if any(len(signal) == 0 for signal in [bvp, eda, hr, temp]):
            return None

        return BiophysicalSignals(bvp=bvp, eda=eda, hr=hr, ibi=ibi, temp=temp)

    def get_emotion_rating(
        self, participant_id: str, song_no: int, song_id: int
    ) -> Optional[EmotionRating]:
        """Get emotion rating for a specific recording"""
        mask = (
            (self.emotion_ratings["participant_id"] == participant_id)
            & (self.emotion_ratings["song_no"] == song_no)
            & (self.emotion_ratings["song_id"] == song_id)
        )

        matching_rows = self.emotion_ratings[mask]

        if matching_rows.empty:
            return None

        row = matching_rows.iloc[0]
        return EmotionRating(
            participant_id=row["participant_id"],
            song_no=row["song_no"],
            song_id=row["song_id"],
            valence_rating=row["valence_rating"],
            valence=row["valence"],
            arousal_rating=row["arousal_rating"],
            arousal=row["arousal"],
            play_duration=row["play_duration"],
        )

    def get_binary_labels(self, target: str) -> Dict[str, int]:
        """Get binary labels for valence or arousal"""
        assert target in ["valence", "arousal"], (
            f"Target must be 'valence' or 'arousal', got {target}"
        )

        # Filter out neutral ratings (0)
        non_neutral = self.emotion_ratings[
            self.emotion_ratings[f"{target}_rating"] != 0
        ]

        labels = {}
        for _, row in non_neutral.iterrows():
            key = f"{row['participant_id']}_{row['song_no']}_{row['song_id']}"
            # Positive = 1, Negative = 0
            labels[key] = 1 if row[f"{target}_rating"] > 0 else 0

        return labels

    def get_all_valid_recordings(self) -> List[Tuple[str, int, int]]:
        """Get list of all recordings that have both signals and emotion ratings"""
        valid_recordings = []

        for _, row in self.emotion_ratings.iterrows():
            participant_id = row["participant_id"]
            song_no = row["song_no"]
            song_id = row["song_id"]

            # Check if signals exist for this recording
            signals = self.load_signals_for_recording(participant_id, song_no, song_id)
            if signals is not None:
                valid_recordings.append((participant_id, song_no, song_id))

        return valid_recordings

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        valid_recordings = self.get_all_valid_recordings()

        valence_labels = self.get_binary_labels("valence")
        arousal_labels = self.get_binary_labels("arousal")

        return {
            "total_recordings": len(valid_recordings),
            "participants": len(self.participants),
            "valence_positive": sum(valence_labels.values()),
            "valence_negative": len(valence_labels) - sum(valence_labels.values()),
            "arousal_positive": sum(arousal_labels.values()),
            "arousal_negative": len(arousal_labels) - sum(arousal_labels.values()),
        }
