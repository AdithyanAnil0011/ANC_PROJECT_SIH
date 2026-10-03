import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import numpy as np

from defence_noise_dataset import DefenceNoiseDataset


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

BATCH_SIZE = 32

TEST_FILE = "dataset_balanced/test_metadata.csv"
CHECKPOINT = "defence_noise_classifier.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "gunshot",
    "shelling",
    "vehicle",
    "helicopter",
    "fighter"
]


# ============================================================
# MODEL
# ============================================================

class DefenceNoiseCNN(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
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
# LOAD CHECKPOINT
# ============================================================

print("=" * 60)
print("DEFENCE NOISE CLASSIFIER - TEST EVALUATION")
print("=" * 60)

print("\nDevice:", DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model = DefenceNoiseCNN(
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded:", CHECKPOINT)


# ============================================================
# MEL TRANSFORMS
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


def extract_features(audio):

    mel = mel_transform(audio)

    mel_db = db_transform(mel)

    return mel_db


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = DefenceNoiseDataset(
    TEST_FILE,
    sample_rate=SAMPLE_RATE,
    segment_seconds=2.0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_labels = []
all_snrs = []
all_noise_types = []

with torch.no_grad():

    for batch in test_loader:

        audio = batch["audio"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        features = extract_features(audio)

        features = features.unsqueeze(1)

        outputs = model(features)

        predictions = outputs.argmax(dim=1)

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


all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_snrs = np.array(all_snrs)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = (
    np.mean(all_predictions == all_labels) * 100
)

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy:.2f}%")


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (len(CLASS_NAMES), len(CLASS_NAMES)),
    dtype=int
)

for true, pred in zip(
    all_labels,
    all_predictions
):
    confusion[true, pred] += 1


print("\nConfusion Matrix")
print("(Rows = Actual, Columns = Predicted)\n")

print(
    "             "
    + " ".join(
        f"{name[:9]:>10}"
        for name in CLASS_NAMES
    )
)

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{name[:9]:>10} "
        + " ".join(
            f"{confusion[i, j]:10d}"
            for j in range(len(CLASS_NAMES))
        )
    )


# ============================================================
# PER-CLASS METRICS
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

precisions = []
recalls = []
f1_scores = []

for i, name in enumerate(CLASS_NAMES):

    tp = confusion[i, i]

    fp = confusion[:, i].sum() - tp

    fn = confusion[i, :].sum() - tp

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    precisions.append(precision)

    recalls.append(recall)

    f1_scores.append(f1)

    print(
        f"\n{name.upper()}"
    )

    print(
        f"  Precision: {precision * 100:.2f}%"
    )

    print(
        f"  Recall:    {recall * 100:.2f}%"
    )

    print(
        f"  F1 Score:  {f1 * 100:.2f}%"
    )


macro_precision = np.mean(precisions)
macro_recall = np.mean(recalls)
macro_f1 = np.mean(f1_scores)

print("\n" + "-" * 40)

print(
    f"Macro Precision: {macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall:    {macro_recall * 100:.2f}%"
)

print(
    f"Macro F1:        {macro_f1 * 100:.2f}%"
)


# ============================================================
# ACCURACY BY SNR
# ============================================================

print("\n" + "=" * 60)
print("ACCURACY BY SNR")
print("=" * 60)

for snr in sorted(np.unique(all_snrs)):

    mask = all_snrs == snr

    snr_accuracy = (
        np.mean(
            all_predictions[mask]
            == all_labels[mask]
        ) * 100
    )

    print(
        f"SNR {snr:>5.0f} dB : "
        f"{snr_accuracy:.2f}% "
        f"({mask.sum()} samples)"
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)