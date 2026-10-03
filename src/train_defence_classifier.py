import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio

from defence_noise_dataset import DefenceNoiseDataset


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

PATIENCE = 5

TRAIN_FILE = "dataset_balanced/train_metadata.csv"
VAL_FILE = "dataset_balanced/val_metadata.csv"

CHECKPOINT = "defence_noise_classifier_4sec.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "gunshot",
    "shelling",
    "vehicle",
    "helicopter",
    "fighter"
]


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
# CNN MODEL
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
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio):

    mel = mel_transform(audio)

    mel_db = db_transform(mel)

    return mel_db


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("DEFENCE NOISE CLASSIFIER")
print("=" * 60)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


print("\nLoading datasets...")

train_dataset = DefenceNoiseDataset(
    TRAIN_FILE,
    sample_rate=SAMPLE_RATE,
    segment_seconds=4.0
)

val_dataset = DefenceNoiseDataset(
    VAL_FILE,
    sample_rate=SAMPLE_RATE,
    segment_seconds=4.0
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = DefenceNoiseCNN(
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\nModel parameters:",
      sum(p.numel() for p in model.parameters()))

# ============================================================
# ONE-BATCH PIPELINE TEST
# ============================================================

# print("\nTesting one training batch...", flush=True)

# batch = next(iter(train_loader))

# print("Batch loaded.", flush=True)

# audio = batch["audio"].to(DEVICE)
# labels = batch["label"].to(DEVICE)

# print("Audio shape:", audio.shape, flush=True)

# features = extract_features(audio)

# print("Mel shape:", features.shape, flush=True)

# features = features.unsqueeze(1)

# print("CNN input shape:", features.shape, flush=True)

# outputs = model(features)

# print("Output shape:", outputs.shape, flush=True)

# loss = criterion(outputs, labels)

# print("Loss:", loss.item(), flush=True)

# loss.backward()

# print("One batch completed successfully.", flush=True)
# exit()
# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

epochs_without_improvement = 0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx % 10 == 0:
             print(
                 f"Training batch {batch_idx}/{len(train_loader)}",
                 flush=True
             )


        audio = batch["audio"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        # Extract Mel features
        features = extract_features(audio)

        # CNN expects  [B, 1, 64, ~126]
        features = features.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()


        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss /= len(train_loader)

    train_accuracy = (
        100.0 * train_correct / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            audio = batch["audio"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            features = extract_features(audio)

            features = features.unsqueeze(1)

            outputs = model(features)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_loss /= len(val_loader)

    val_accuracy = (
        100.0 * val_correct / val_total
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch + 1:02d}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Acc:    {val_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "class_names": CLASS_NAMES,

                "sample_rate": SAMPLE_RATE,

                "n_fft": N_FFT,

                "hop_length": HOP_LENGTH,

                "n_mels": N_MELS,

                "segment_seconds": 4.0
            },
            CHECKPOINT
        )

        print("✓ Best model saved")


    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping.")

        break


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Best model:", CHECKPOINT)