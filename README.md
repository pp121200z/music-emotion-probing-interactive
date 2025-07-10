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
