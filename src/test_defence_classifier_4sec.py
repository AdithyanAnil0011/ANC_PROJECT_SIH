import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.defence_noise_dataset import DefenceNoiseDataset


# ============================================================
# MODEL
# ============================================================

class DefenceNoiseCNN(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_METADATA = (
    PROJECT_ROOT /
    "dataset_balanced" /
    "test_metadata.csv"
)

CHECKPOINT = (
    PROJECT_ROOT /
    "models" /
    "defence_noise_classifier_4sec.pth"
)

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 4.0

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

BATCH_SIZE = 32

CLASS_NAMES = [
    "gunshot",
    "shelling",
    "vehicle",
    "helicopter",
    "fighter"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("4-SECOND DEFENCE NOISE CLASSIFIER TEST")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DATASET
# ============================================================

test_dataset = DefenceNoiseDataset(
    TEST_METADATA,
    segment_seconds=SEGMENT_SECONDS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# ============================================================
# MEL SPECTROGRAM
# ============================================================

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
).to(DEVICE)


db_transform = torchaudio.transforms.AmplitudeToDB(
    stype="power",
    top_db=80
).to(DEVICE)


# ============================================================
# MODEL
# ============================================================

model = DefenceNoiseCNN(
    num_classes=len(CLASS_NAMES)
).to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


print("\nCheckpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):

    print("Checkpoint keys:")
    print(list(checkpoint.keys()))


# ------------------------------------------------------------
# Load model weights
# ------------------------------------------------------------

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)


model.eval()

print("\nModel loaded successfully.")


# ============================================================
# MODEL PARAMETER CHECK
# ============================================================

print("\nModel parameters:")

total_params = 0

for name, param in model.named_parameters():

    print(
        f"{name:<45} {tuple(param.shape)}"
    )

    total_params += param.numel()


print("\nTotal parameters:", total_params)


# ============================================================
# CHECK ONE TEST SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SINGLE SAMPLE PIPELINE CHECK")
print("=" * 60)

sample = test_dataset[0]

sample_audio = sample["audio"].unsqueeze(0).to(DEVICE)

sample_label = sample["label"].item()

sample_noise = sample["noise_type"]

sample_snr = sample["snr_db"].item()


print("Sample ID      :", sample["sample_id"])
print("True class     :", sample_noise)
print("True label     :", sample_label)
print("SNR            :", sample_snr)
print("Audio shape    :", sample_audio.shape)


with torch.no_grad():

    sample_mel = mel_transform(sample_audio)

    sample_db = db_transform(sample_mel)

    sample_db = db_transform(sample_mel)

    sample_input = sample_db.unsqueeze(1)

 

    sample_input = sample_db.unsqueeze(1)

    sample_output = model(sample_input)

    sample_prob = torch.softmax(
        sample_output,
        dim=1
    )

    sample_prediction = torch.argmax(
        sample_output,
        dim=1
    ).item()


print("Mel shape      :", sample_mel.shape)
print("CNN input      :", sample_input.shape)

print("\nModel logits:")
print(
    sample_output.cpu().numpy()
)

print("\nModel probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<12}: "
        f"{sample_prob[0, i].item() * 100:.2f}%"
    )

print(
    "\nPredicted class:",
    CLASS_NAMES[sample_prediction]
)


# ============================================================
# FULL TEST
# ============================================================

print("\n" + "=" * 60)
print("RUNNING FULL TEST")
print("=" * 60)


all_predictions = []
all_labels = []

all_snrs = []
all_noise_types = []


total_batches = len(test_loader)


with torch.no_grad():

    for batch_idx, batch in enumerate(test_loader):

        audio = batch["audio"].to(DEVICE)

        labels = batch["label"].to(DEVICE)


        # ----------------------------------------------------
        # Audio → Mel
        # ----------------------------------------------------

        mel = mel_transform(audio)


        # ----------------------------------------------------
        # Mel → dB
        # ----------------------------------------------------

        mel = db_transform(mel)


        # ----------------------------------------------------
        # Per-sample normalization
        # ----------------------------------------------------

        mel = mel.unsqueeze(1)




        # ----------------------------------------------------
        # CNN input
        # ----------------------------------------------------

        mel = mel.unsqueeze(1)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        outputs = model(mel)

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_snrs.extend(
            batch["snr_db"].numpy()
        )

        all_noise_types.extend(
            batch["noise_type"]
        )


        if (batch_idx + 1) % 10 == 0:

            print(
                f"Testing batch "
                f"{batch_idx + 1}/{total_batches}"
            )


# ============================================================
# NUMPY
# ============================================================

y_true = np.array(all_labels)

y_pred = np.array(all_predictions)

snrs = np.array(all_snrs)

noise_types = np.array(all_noise_types)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(y_pred == i)

    percentage = (
        count / len(y_pred)
    ) * 100

    print(
        f"{class_name:<12}: "
        f"{count:>4} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
)


print("\n" + "=" * 60)
print("4-SECOND CLASSIFIER TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy         : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision  : "
    f"{precision * 100:.2f}%"
)

print(
    f"Macro Recall     : "
    f"{recall * 100:.2f}%"
)

print(
    f"Macro F1         : "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nConfusion Matrix:")

print(
    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )
)


# ============================================================
# ACCURACY BY SNR
# ============================================================

print("\nAccuracy by SNR:")
print("-" * 40)


for snr in sorted(np.unique(snrs)):

    mask = snrs == snr

    snr_accuracy = accuracy_score(
        y_true[mask],
        y_pred[mask]
    )

    print(
        f"SNR {snr:>3.0f} dB : "
        f"{snr_accuracy * 100:.2f}% "
        f"({mask.sum()} samples)"
    )


# ============================================================
# ACCURACY BY NOISE TYPE
# ============================================================

print("\nAccuracy by Noise Type:")
print("-" * 40)


for noise in CLASS_NAMES:

    mask = noise_types == noise

    if mask.sum() == 0:
        continue

    noise_accuracy = accuracy_score(
        y_true[mask],
        y_pred[mask]
    )

    print(
        f"{noise:<12} : "
        f"{noise_accuracy * 100:.2f}% "
        f"({mask.sum()} samples)"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("Evaluation complete.")
print("=" * 60)