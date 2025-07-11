# Original Project Overview (from AokiKoshiro)

# Music Emotion Probing

A research project investigating emotion estimation from physiological signals during music listening using TimesFM foundation model representations.

## Overview

This project explores whether internal representations from the TimesFM time-series foundation model can outperform traditional handcrafted feature-based methods for emotion prediction from physiological data.

## Dataset

**HKU956**: 30 participants listening to 956 music tracks with recorded physiological signals:
- Blood Volume Pulse (BVP) - 64 Hz
- Heart Rate (HR) - 1 Hz  
- Electrodermal Activity (EDA) - 1 Hz
- Skin Temperature (TEMP) - 4 Hz

**Labels**: Binary classification for Valence and Arousal (positive/negative)

## Method

1. **Feature Extraction**: Normalized physiological signals are fed into frozen TimesFM model
2. **Probing**: Probes trained on layer-wise representations
3. **Evaluation**: k-fold cross-validation with accuracy and F1 score metrics

## Experiments

- **Layer-wise Probing**: Compare performance across Transformer layers
- **Modality Ablation**: Evaluate contribution of each physiological signal

## Setup

```bash
# Install dependencies
uv sync

# Run experiments
uv run python run_experiment.py

# Visualize results
uv run python visualize_results.py
```



# This Updated Version Overview

## Web Interface for this model
This version includes a frontend for uploading CSV data and visualizing emotion predictions.

## What is updated
This project is forked and developed from AokiKoshiro/music-emotion-probing, with additional features:

1. Real-time prediction via web interface
2. Quadrant-based emotion visualization (arousal vs valence)
3. Supports physiological signal CSV upload

## New Files
- `frontend/`: React-based UI
- `predict_server.py`: Simplified inference API
- Demo-ready setup instructions

## Setup

# Start the backend server

```bash
cd music-emotion-probing-main
python -m venv venv310
.\venv310\Scripts\Activate.ps1
pip install -r requirements.txt
python predict_server.py
```

# Start the frontend server (Vite+React)

```bash
cd frontend
npm install        # only  the first time
npm run dev   # launches at http://localhost:5173/
```



> Developed by Peiyi Zhang, July 2025.
