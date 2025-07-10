# test_predict.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.config import get_config
from src.models.timesfm_extractor import FeatureExtractorPipeline


config = get_config()
feature_extractor = FeatureExtractorPipeline(config=config, capture_layers=False)

csv_path = "mock_physiological_input.csv"
df = pd.read_csv(csv_path)

class Signal:
    pass

signal = Signal()
for modality in config.dataset.modalities:
    if modality in df.columns:
        setattr(signal, modality, df[modality].values)
    else:
        print(f"Warning: input data is lacking {modality}, filled with 0s")
        setattr(signal, modality, np.zeros(128))

feature_extractor.fit([signal])
embedding = feature_extractor(signal).reshape(1, -1)

val_path = Path("results/valence_probe_model.joblib")
aro_path = Path("results/arousal_probe_model.joblib")
if not val_path.exists() or not aro_path.exists():
    raise FileNotFoundError("Cannot find valence/arousal model in results/")

val_model = joblib.load(val_path)
aro_model = joblib.load(aro_path)

val_pred = val_model.predict(embedding)[0]
aro_pred = aro_model.predict(embedding)[0]

print(" Model prediction results:")
print(f"Valence : {'positive' if val_pred == 1 else 'negative'} ({val_pred})")
print(f"Arousal : {'high' if aro_pred == 1 else 'low'} ({aro_pred})")
print("used bio signals:", config.dataset.modalities)

