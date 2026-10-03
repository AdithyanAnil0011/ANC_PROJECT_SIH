# AI/ML-Enabled Adaptive Noise Cancellation System for Defence Environments

An AI/ML-based speech enhancement system designed for defence communication environments with severe and dynamic background noise.

The system combines:
* Deep-learning-based noise classification
* Voice Activity Detection (VAD)
* Normalized Least Mean Squares (NLMS) adaptive noise cancellation
* Real-time audio processing

The objective is to improve speech intelligibility in environments containing high-intensity noises such as gunshots, shelling, vehicles, helicopters, and fighter aircraft.

---

## Problem Statement

* Communication in defence environments can be severely affected by high-intensity and rapidly changing background noise.
* Traditional noise cancellation methods often struggle when noise characteristics change significantly.
* This project utilizes an AI-based noise classifier to identify the dominant noise environment and combines it with adaptive signal processing to suppress background noise while preserving speech.

---

## System Overview

```text
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

## Noise Classes & Dataset

The classifier recognizes five defence-related noise categories:
1. **Gunshot**
2. **Shelling**
3. **Vehicle**
4. **Helicopter**
5. **Fighter**

The training data was constructed using clean speech from `LibriSpeech train-clean-5` and defence-related noise samples.

### Dataset Specifications
* **Noise Categories:** $5$
* **SNR Conditions:** $6$ ($-5, 0, 5, 10, 15, 20\text{ dB}$)
* **Total Mixtures:** $6,000$ balanced noisy-speech mixtures ($1,200$ samples per noise category; $1,000$ samples per SNR condition; $200$ samples per noise $\times$ SNR combination)

The dataset was split without overlapping noise sources between training, validation, and test sets.

| Split | Samples |
| :--- | :--- |
| **Training** | $4,200$ |
| **Validation** | $904$ |
| **Testing** | $896$ |

---

## AI Noise Classifier

A lightweight CNN is used to classify the dominant noise type.

### Pipeline
1. Audio $\rightarrow$ $16\text{ kHz}$ Mono
2. $4$-second Segment
3. RMS Normalization
4. Mel Spectrogram
5. Power-to-dB Conversion
6. CNN Processing

### CNN Architecture
* Input $\rightarrow$ `Conv2D (1 → 16)` $\rightarrow$ BatchNorm + ReLU + MaxPool
* `Conv2D (16 → 32)` $\rightarrow$ BatchNorm + ReLU + MaxPool
* `Conv2D (32 → 64)` $\rightarrow$ BatchNorm + ReLU + MaxPool
* Adaptive Average Pooling $\rightarrow$ Dropout $\rightarrow$ Linear Layer $\rightarrow$ **5 Noise Classes**

The model contains approximately **18.9K parameters**, making it lightweight enough for embedded deployment.

### Classifier Performance

* **Test Accuracy:** $84.04\%$
* **Macro Precision:** $85.06\%$
* **Macro Recall:** $83.80\%$
* **Macro F1 Score:** $83.66\%$

#### Per-Class Performance
| Noise | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| **Gunshot** | $90.06\%$ | $88.11\%$ | $89.07\%$ |
| **Shelling** | $84.90\%$ | $90.06\%$ | $87.40\%$ |
| **Vehicle** | $88.33\%$ | $61.63\%$ | $72.60\%$ |
| **Helicopter** | $72.65\%$ | $93.92\%$ | $81.93\%$ |
| **Fighter** | $89.35\%$ | $85.31\%$ | $87.28\%$ |

#### SNR-Wise Classification Performance
| SNR | Accuracy |
| :--- | :--- |
| **$-5\text{ dB}$** | $81.76\%$ |
| **$0\text{ dB}$** | $87.76\%$ |
| **$5\text{ dB}$** | $85.33\%$ |
| **$10\text{ dB}$** | $86.29\%$ |
| **$15\text{ dB}$** | $84.13\%$ |
| **$20\text{ dB}$** | $79.31\%$ |

---

## Adaptive Noise Cancellation

The project uses an **NLMS (Normalized Least Mean Squares)** adaptive filter for noise cancellation. A VAD-controlled version was implemented so that filter coefficients are updated primarily during noise-only regions.

### Why VAD?
* Updating an adaptive noise canceller continuously can cause the filter to adapt to speech itself.
* The VAD separates the signal into:
  * **Speech region:** Freeze filter adaptation
  * **Noise-only region:** Update NLMS coefficients

---

## Final End-to-End Demonstration

A complete end-to-end demonstration was implemented using a fixed helicopter-noise scenario.

* **Noise Type:** Helicopter
* **SNR:** $-5\text{ dB}$
* **Sample ID:** $3740$

### Results
* **CNN Classification:** Helicopter ($97.07\%$)
* **VAD Analysis:** Speech ($58.60\%$), Noise-only ($41.40\%$)
* **VAD-Controlled NLMS Filter Parameters:**
  * Filter length: $128$
  * Step size: $0.01$
  * VAD threshold: $-40\text{ dB}$
* **SNR Improvement:**
  * **Before ANC:** $-5.00\text{ dB}$
  * **After ANC:** $+2.90\text{ dB}$
  * **Total Improvement:** **$+7.90\text{ dB}$**

---

## Repository Structure

```text
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
```

---

## Hardware Used

* **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU ($4\text{ GB}$ VRAM)
* **Framework:** PyTorch (CUDA-enabled)

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ANC_PROJECT_SIH.git
   cd ANC_PROJECT_SIH
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment (Windows):**
   ```bash
   .venv\Scripts\activate
   ```
   *(For macOS/Linux use: `source .venv/bin/activate`)*

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Pipeline

### Run the End-to-End Demo
```bash
python src/demo_end_to_end.py
```
*Loads a predefined defence-noise scenario, extracts reference noise, classifies it via CNN, executes VAD, runs VAD-controlled NLMS adaptive cancellation, calculates SNR improvements, and saves/plays the enhanced audio.*

### Evaluate the Trained Classifier
```bash
python src/evaluate_defence_classifier.py
```

### Run SNR-Wise Evaluation
```bash
python src/snr_wise_classifier.py
```

---

## Important Note on Dataset

Raw datasets and large training bundles are excluded from this repository via `.gitignore`. The repository includes source code, pre-trained weights, evaluation results, and sample demonstration audio required to reproduce the pipeline.

---

## Limitations

* The current VAD-NLMS evaluation includes an experimental oracle-VAD configuration for controlled analysis.
* Real-world defence environments contain highly non-stationary and overlapping noise sources.
* The current CNN performs noise classification from a reference noise signal; further testing with live field recordings is required.
* Embedded deployment will require target-hardware optimization and validation.

---

## Future Work

* Real-time microphone-based processing
* Embedded deployment on edge hardware
* Quantization and model compression
* Robust real-world VAD integration
* Multi-microphone / beamforming support
* Diverse battlefield noise recording collection
* Deep-learning-based speech enhancement integration
