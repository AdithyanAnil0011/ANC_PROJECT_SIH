# AI/ML-Enabled Adaptive Noise Cancellation System for Defence Environments

An AI/ML-based speech enhancement system designed for defence communication environments with severe and dynamic background noise.

The system combines:

- **Deep-learning-based noise classification**
- **Voice Activity Detection (VAD)**
- **NLMS adaptive noise cancellation**
- **Real-time audio processing**

The objective is to improve speech intelligibility in environments containing noises such as gunshots, shelling, vehicles, helicopters, and fighter aircraft.

---

## Problem Statement

Communication in defence environments can be severely affected by high-intensity and rapidly changing background noise.

Traditional noise cancellation methods may struggle when the noise characteristics change significantly.

This project uses an AI-based noise classifier to identify the dominant noise environment and combines it with adaptive signal processing to suppress background noise while preserving speech.

---

## System Overview

```

text
                Input Audio
                     │
                     ▼
            ┌─────────────────┐
            │ Noise Reference │
            │     Signal      │
            └────────┬────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  CNN Noise          │
          │  Classifier         │
          └─────────┬───────────┘
                    │
                    ▼
       Identify Dominant Noise
                    │
                    ▼
          ┌─────────────────────┐
          │ Voice Activity      │
          │ Detection (VAD)     │
          └─────────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │ VAD-Controlled      │
          │ NLMS Adaptive ANC   │
          └─────────┬───────────┘
                    │
                    ▼
            Enhanced Speech
```
---

Noise Classes

The classifier recognizes five defence-related noise categories:

Gunshot
Shelling
Vehicle
Helicopter
Fighter
Dataset

The training data was constructed using clean speech from LibriSpeech train-clean-5 and defence-related noise samples.

The dataset contains:

5 noise categories
6 SNR conditions
SNR values: −5, 0, 5, 10, 15, 20 dB
6000 balanced noisy-speech mixtures
1200 samples per noise category
1000 samples per SNR condition
200 samples per noise × SNR combination

The dataset was split without overlapping noise sources between training, validation, and test sets.

Dataset Split
Split	Samples
Training	4200
Validation	904
Testing	896
AI Noise Classifier

A lightweight CNN is used to classify the dominant noise type.

Input Processing
Audio
  ↓
16 kHz Mono
  ↓
4-second Segment
  ↓
RMS Normalization
  ↓
Mel Spectrogram
  ↓
Power-to-dB Conversion
  ↓
CNN
CNN Architecture
Input
  ↓
Conv2D (1 → 16)
  ↓
BatchNorm + ReLU + MaxPool
  ↓
Conv2D (16 → 32)
  ↓
BatchNorm + ReLU + MaxPool
  ↓
Conv2D (32 → 64)
  ↓
BatchNorm + ReLU + MaxPool
  ↓
Adaptive Average Pooling
  ↓
Dropout
  ↓
Linear Layer
  ↓
5 Noise Classes

The model contains approximately 18.9K parameters, making it lightweight enough for future embedded deployment.

Classifier Performance

The final classifier achieved:

Test Accuracy: 84.04%

Overall Metrics
Metric	Score
Accuracy	84.04%
Macro Precision	85.06%
Macro Recall	83.80%
Macro F1 Score	83.66%
Per-Class Performance
Noise	Precision	Recall	F1
Gunshot	90.06%	88.11%	89.07%
Shelling	84.90%	90.06%	87.40%
Vehicle	88.33%	61.63%	72.60%
Helicopter	72.65%	93.92%	81.93%
Fighter	89.35%	85.31%	87.28%
SNR-Wise Classification Performance
SNR	Accuracy
−5 dB	81.76%
0 dB	87.76%
5 dB	85.33%
10 dB	86.29%
15 dB	84.13%
20 dB	79.31%

The classifier remains effective across a wide range of noise conditions, including heavily corrupted speech at −5 dB SNR.

Adaptive Noise Cancellation

The project uses an NLMS (Normalized Least Mean Squares) adaptive filter for noise cancellation.

A VAD-controlled version was implemented so that filter coefficients are updated primarily during noise-only regions.

Why VAD?

Updating an adaptive noise canceller continuously can cause the filter to adapt to speech itself.

The VAD therefore separates the signal approximately into:

Speech region → Freeze filter adaptation

Noise-only region → Update NLMS coefficients

This helps reduce speech distortion while allowing the filter to learn the background noise characteristics.

Final End-to-End Demonstration

A complete end-to-end demonstration was implemented using a fixed helicopter-noise scenario.

Scenario
Noise Type : Helicopter
SNR        : −5 dB
Sample ID  : 3740

The CNN correctly identified the noise as:

Helicopter: 97.07%
VAD
Speech      : 58.60%
Noise-only  : 41.40%
VAD-Controlled NLMS
Filter length : 128
Step size     : 0.01
VAD threshold : −40 dB
SNR Improvement
Before ANC : −5.00 dB
After ANC  : +2.90 dB

Improvement: +7.90 dB

The enhanced output was also saved as a WAV file for listening and demonstration.

Repository Structure
ANC_PROJECT_SIH/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── demo/
│   └── audio_samples/
│       └── helicopter/
│           ├── noisy_input.wav
│           ├── enhanced_output.wav
│           └── results.csv
│
├── models/
│   └── defence_noise_classifier_4sec.pth
│
├── results/
│   ├── nlms_batch_results.csv
│   ├── nlms_mu_by_noise.csv
│   ├── nlms_mu_by_snr.csv
│   ├── nlms_mu_summary.csv
│   ├── vad_nlms_batch_results.csv
│   └── vad_threshold_sweep_results.csv
│
└── src/
    ├── defence_noise_dataset.py
    ├── demo_end_to_end.py
    ├── evaluate_defence_classifier.py
    ├── nlms_anc.py
    ├── scenario_loader.py
    ├── snr_wise_classifier.py
    ├── test_defence_classifier_4sec.py
    ├── train_defence_classifier.py
    └── vad_nlms_anc.py
Hardware Used

Development and training were performed on:

GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 4 GB
Framework: PyTorch
CUDA: CUDA-enabled PyTorch environment

The classifier is intentionally lightweight to support future optimization for embedded/edge hardware.

Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ANC_PROJECT_SIH

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Running the Demo

The end-to-end demonstration can be run using:

python src/demo_end_to_end.py

The demo:

Loads a predefined defence-noise scenario.
Extracts the reference noise.
Classifies the noise using the CNN.
Performs VAD.
Runs VAD-controlled NLMS adaptive cancellation.
Calculates SNR before and after enhancement.
Saves the enhanced audio.
Plays the noisy and enhanced signals for comparison.
Model Evaluation

To evaluate the trained classifier:

python src/evaluate_defence_classifier.py

For SNR-wise evaluation:

python src/snr_wise_classifier.py
Important Note on Dataset

The raw datasets and large training data are intentionally excluded from this repository using .gitignore.

The repository contains the source code, trained model, evaluation results, and demonstration audio required to understand and reproduce the implemented pipeline.

Limitations

The current implementation is a research prototype.

Important limitations include:

The current VAD-NLMS evaluation includes an experimental oracle-VAD configuration for controlled analysis.
Real-world defence environments contain highly non-stationary and overlapping noise sources.
The current CNN performs noise classification from a reference noise signal.
Further testing with real field recordings is required.
Embedded deployment will require additional optimization and hardware validation.
Future Work

Planned improvements include:

Real-time microphone-based processing
Embedded deployment on a suitable edge device
Quantization and model compression
More robust real-world VAD
Multi-microphone / beamforming integration
More diverse battlefield noise recordings
Deep-learning-based speech enhancement
End-to-end adaptive noise cancellation
Objective speech-intelligibility evaluation
Project Status

Current status: Working prototype

The complete pipeline has been implemented and demonstrated:

Noise Reference
      ↓
AI Noise Classification
      ↓
Voice Activity Detection
      ↓
Adaptive NLMS Noise Cancellation
      ↓
Enhanced Speech

The current end-to-end demonstration achieves a +7.90 dB SNR improvement for the selected helicopter-noise scenario at −5 dB input SNR.


**One thing before pasting:** replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHu
