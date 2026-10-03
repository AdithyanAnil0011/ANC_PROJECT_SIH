import torch
import torch.nn as nn
import torchaudio
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from pathlib import Path
from src.defence_noise_dataset import DefenceNoiseDataset


# ============================================================
# MODEL
# ============================================================

class DefenceNoiseCNN(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

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
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = PROJECT_ROOT / "models" / "defence_noise_classifier_4sec.pth"
TEST_METADATA = PROJECT_ROOT / "dataset_balanced" / "test_metadata.csv"

BATCH_SIZE = 32

SNR_VALUES = [-5, 0, 5, 10, 15, 20]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

CLASS_NAMES = checkpoint["class_names"]

SAMPLE_RATE = checkpoint["sample_rate"]
N_FFT = checkpoint["n_fft"]
HOP_LENGTH = checkpoint["hop_length"]
N_MELS = checkpoint["n_mels"]
SEGMENT_SECONDS = checkpoint["segment_seconds"]


print("=" * 65)
print("SNR-WISE DEFENCE NOISE CLASSIFIER EVALUATION")
print("=" * 65)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nCheckpoint parameters:")
print("Sample rate     :", SAMPLE_RATE)
print("FFT             :", N_FFT)
print("Hop length      :", HOP_LENGTH)
print("Mel bins        :", N_MELS)
print("Segment seconds :", SEGMENT_SECONDS)

print("\nClasses:")
print(CLASS_NAMES)


# ============================================================
# LOAD MODEL
# ============================================================

model = DefenceNoiseCNN(
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nModel loaded successfully.")


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
    stype="power"
).to(DEVICE)


# ============================================================
# LOAD TEST DATASET
# ============================================================

dataset = DefenceNoiseDataset(
    TEST_METADATA,
    sample_rate=SAMPLE_RATE,
    segment_seconds=SEGMENT_SECONDS
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTest samples:", len(dataset))


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []
all_snr = []
all_noise_types = []

print("\nRunning inference...")

with torch.no_grad():

    for batch_idx, batch in enumerate(loader):

        audio = batch["audio"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        snr = batch["snr_db"]

        noise_types = batch["noise_type"]

        # ----------------------------------------------------
        # SAME PREPROCESSING AS TRAINING
        # ----------------------------------------------------

        features = mel_transform(audio)

        features = db_transform(features)

        # [B, 64, 126]
        #       ↓
        # [B, 1, 64, 126]

        features = features.unsqueeze(1)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        outputs = model(features)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_snr.extend(
            snr.cpu().numpy()
        )

        all_noise_types.extend(
            noise_types
        )

        if (batch_idx + 1) % 10 == 0:

            print(
                f"Batch "
                f"{batch_idx + 1}/{len(loader)}"
            )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_pred = np.array(all_predictions)

y_true = np.array(all_labels)

snr_array = np.array(all_snr)

noise_array = np.array(all_noise_types)


# ============================================================
# OVERALL ACCURACY
# ============================================================

overall_accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n")
print("=" * 65)
print("OVERALL TEST ACCURACY")
print("=" * 65)

print(
    f"Accuracy: {overall_accuracy * 100:.2f}%"
)


# ============================================================
# SNR-WISE ACCURACY
# ============================================================

print("\n")
print("=" * 65)
print("SNR-WISE ACCURACY")
print("=" * 65)

print(
    f"{'SNR (dB)':>10} | "
    f"{'Samples':>8} | "
    f"{'Correct':>8} | "
    f"{'Accuracy':>10}"
)

print("-" * 50)


snr_results = {}


for snr_value in SNR_VALUES:

    mask = snr_array == snr_value

    snr_true = y_true[mask]

    snr_pred = y_pred[mask]

    count = len(snr_true)

    correct = np.sum(
        snr_true == snr_pred
    )

    if count > 0:

        accuracy = (
            correct / count
        ) * 100

    else:

        accuracy = 0.0

    snr_results[snr_value] = accuracy

    print(
        f"{snr_value:>10} | "
        f"{count:>8} | "
        f"{correct:>8} | "
        f"{accuracy:>9.2f}%"
    )


# ============================================================
# SNR × NOISE TYPE ACCURACY
# ============================================================

print("\n")
print("=" * 65)
print("SNR × NOISE TYPE ACCURACY")
print("=" * 65)

print(
    f"{'Noise':<12}",
    end=""
)

for snr_value in SNR_VALUES:

    print(
        f"{snr_value:>9}dB",
        end=""
    )

print()

print("-" * 67)


for class_index, noise_name in enumerate(CLASS_NAMES):

    print(
        f"{noise_name:<12}",
        end=""
    )

    for snr_value in SNR_VALUES:

        mask = (
            (snr_array == snr_value) &
            (y_true == class_index)
        )

        class_true = y_true[mask]

        class_pred = y_pred[mask]

        if len(class_true) > 0:

            accuracy = (
                np.sum(
                    class_true == class_pred
                ) / len(class_true)
            ) * 100

        else:

            accuracy = 0.0

        print(
            f"{accuracy:>8.1f}%",
            end=""
        )

    print()


# ============================================================
# BEST / WORST SNR
# ============================================================

best_snr = max(
    snr_results,
    key=snr_results.get
)

worst_snr = min(
    snr_results,
    key=snr_results.get
)


print("\n")
print("=" * 65)
print("SNR SUMMARY")
print("=" * 65)

print(
    f"Best SNR condition  : "
    f"{best_snr} dB "
    f"({snr_results[best_snr]:.2f}%)"
)

print(
    f"Worst SNR condition : "
    f"{worst_snr} dB "
    f"({snr_results[worst_snr]:.2f}%)"
)

print(
    f"Overall accuracy    : "
    f"{overall_accuracy * 100:.2f}%"
)

print("\nDone.")