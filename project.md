## 1. Research Objectives and Hypotheses

- **Research Objective**
    - To estimate *induced emotions* from physiological signals (e.g., heart rate, electrodermal activity) during music listening, aiming to outperform traditional handcrafted feature-based methods in prediction accuracy.
- **Hypothesis 1**
    - The internal representations of the time-series foundation model, [TimesFM](https://github.com/google-research/timesfm), contain richer emotion-related information than conventional methods, enabling even simple linear [probes](https://arxiv.org/abs/2102.12452) to surpass traditional feature + SVM baselines on the HKU956 dataset.
- **Hypothesis 2**
    - Representations extracted from intermediate layers are more effective for emotion estimation than those from the final layer. This aligns with the widely observed "intermediate layer semantics" hypothesis in Transformer-based models

## 2. Dataset: [**HKU956**](https://www.mdpi.com/2076-3417/12/18/9354)

- Number of participants: 30
- Number of music tracks: 956
- Recorded physiological signals:
    - Blood Volume Pulse (BVP)
    - Heart Rate (HR)
    - Interbeat Interval (IBI)
    - Electrodermal Activity (EDA)
    - Skin Temperature (TEMP)
- Sampling rates:
    - BVP: 64 Hz
    - HR: 1 Hz
    - EDA: 1 Hz
    - TEMP: 4 Hz
- Subjective emotion ratings: Valence and Arousal on a continuous scale from −10 to +10, based on the Self-Assessment Manikin (SAM)
- Data split: 10-fold cross-validation
- Label definitions:
    - Binary classification based on the sign (positive/negative) of Valence
    - Binary classification based on the sign of Arousal (samples near 0 excluded)
- Evaluation metrics: Accuracy and F1 Score

## 3. Model Architecture

- **Foundation Model**
    - [TimesFM 1.0-200M](https://github.com/google-research/timesfm)
    - Model weights are kept frozen (no fine-tuning)
- **Input Preprocessing**
    - Each physiological signal is individually normalized
- **Feature Extraction**
    - The normalized signals are separately input into TimesFM
    - For each Transformer layer, the internal representations (residual stream) are averaged across the time dimension to yield a single vector
- [**Probe](https://arxiv.org/abs/2102.12452) Design**
    - The five feature vectors are concatenated into a single representation
    - Both linear probes (logistic regression) and non-linear probes (MLP) are trained for each layer independently

## 4. Experiments

- **Step 1: Layer-wise Probing**
    - Perform k-fold cross-validation for each Transformer layer
    - Compare the performance across layers to identify the most predictive one
- **Step 2: Modality Ablation Experiment**
    - Using the best-performing layer identified in Step 1
    - Run the same classification pipeline across six conditions: using all physiological signals, and leaving one signal out each time
    - Evaluate the contribution of each modality to emotion prediction
