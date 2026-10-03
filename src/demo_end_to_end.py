import os
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import sounddevice as sd

from scenario_loader import select_scenario


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

# Demo scenario
NOISE_TYPE = "helicopter"
SNR_DB = -5

# Classifier
CHECKPOINT = "defence_noise_classifier_4sec.pth"

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

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

# VAD-NLMS
FILTER_LENGTH = 128
MU = 0.01
EPS = 1e-8

FRAME_DURATION_MS = 20
FRAME_LENGTH = int(
    SAMPLE_RATE * FRAME_DURATION_MS / 1000
)

VAD_THRESHOLD_DB = -40.0

OUTPUT_FILE = "demo_enhanced_output.wav"


# ============================================================
# CLASSIFIER MODEL
# ============================================================

class DefenceNoiseCNN(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
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
# LOAD CLASSIFIER
# ============================================================

def load_classifier():

    model = DefenceNoiseCNN(
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


# ============================================================
# CLASSIFY REFERENCE NOISE
# ============================================================

def classify_noise(model, reference_noise):

    audio = torch.tensor(
        reference_noise,
        dtype=torch.float32
    )

    # Ensure exactly 4 seconds
    required_length = SAMPLE_RATE * 4

    if len(audio) < required_length:

        repeat_count = (
            required_length + len(audio) - 1
        ) // len(audio)

        audio = audio.repeat(repeat_count)

    audio = audio[:required_length]

    # Move to GPU/CPU
    audio = audio.unsqueeze(0).to(DEVICE)
    rms = torch.sqrt(torch.mean(audio ** 2) + 1e-8)
    audio = audio / rms

    with torch.no_grad():

        mel = mel_transform(audio)

        mel_db = db_transform(mel)

        # IMPORTANT:
        # Use the same preprocessing used for
        # the current trained classifier.
        #
        # The final evaluated pipeline uses
        # dB Mel features without additional
        # mean/std normalization.

        model_input = mel_db.unsqueeze(1)

        logits = model(model_input)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return (
        CLASS_NAMES[prediction],
        probabilities[0].cpu().numpy()
    )


# ============================================================
# SNR
# ============================================================

def calculate_snr(clean, signal):

    clean = np.asarray(
        clean,
        dtype=np.float64
    )

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    min_length = min(
        len(clean),
        len(signal)
    )

    clean = clean[:min_length]
    signal = signal[:min_length]

    noise = signal - clean

    signal_power = np.mean(
        clean ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    if noise_power <= 0:
        return float("inf")

    return 10 * np.log10(
        signal_power / noise_power
    )


# ============================================================
# VAD
# ============================================================

def create_vad_mask(
    clean_audio,
    threshold_db=VAD_THRESHOLD_DB
):

    clean_audio = np.asarray(
        clean_audio,
        dtype=np.float64
    )

    n_samples = len(clean_audio)

    speech_mask = np.zeros(
        n_samples,
        dtype=bool
    )

    eps = 1e-12

    for start in range(
        0,
        n_samples,
        FRAME_LENGTH
    ):

        end = min(
            start + FRAME_LENGTH,
            n_samples
        )

        frame = clean_audio[start:end]

        if len(frame) == 0:
            continue

        rms = np.sqrt(
            np.mean(frame ** 2) + eps
        )

        energy_db = 20 * np.log10(
            rms + eps
        )

        if energy_db > threshold_db:

            speech_mask[start:end] = True

    return speech_mask


# ============================================================
# VAD-GATED NLMS
# ============================================================

def vad_nlms(
    reference,
    desired,
    speech_mask,
    filter_length=128,
    mu=0.01,
    eps=1e-8
):

    reference = np.asarray(
        reference,
        dtype=np.float64
    )

    desired = np.asarray(
        desired,
        dtype=np.float64
    )

    speech_mask = np.asarray(
        speech_mask,
        dtype=bool
    )

    n_samples = len(desired)

    weights = np.zeros(
        filter_length,
        dtype=np.float64
    )

    output = np.zeros(
        n_samples,
        dtype=np.float64
    )

    error = np.zeros(
        n_samples,
        dtype=np.float64
    )

    update_count = 0
    freeze_count = 0

    for n in range(
        filter_length,
        n_samples
    ):

        x = reference[
            n-filter_length:n
        ][::-1]

        y = np.dot(
            weights,
            x
        )

        e = desired[n] - y

        output[n] = y
        error[n] = e

        # Freeze adaptation during speech
        if not speech_mask[n]:

            norm = np.dot(
                x,
                x
            ) + eps

            weights += (
                mu / norm
            ) * e * x

            update_count += 1

        else:

            freeze_count += 1

    return (
        output,
        error,
        update_count,
        freeze_count
    )


# ============================================================
# AUDIO PLAYBACK
# ============================================================

def play_audio(audio, sample_rate, label):
    print()
    print("Playing:", label)
    sd.play(np.asarray(audio, dtype=np.float32), sample_rate)
    sd.wait()


# ============================================================
# MAIN DEMO
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ECHOGUARD - END-TO-END ANC DEMO")
    print("=" * 70)

    print()
    print("Device          :", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU             :",
            torch.cuda.get_device_name(0)
        )

    print()
    print("Requested scenario")
    print("------------------")
    print("Noise type      :", NOISE_TYPE)
    print("SNR             :", SNR_DB, "dB")

    # --------------------------------------------------------
    # LOAD SCENARIO
    # --------------------------------------------------------

    # Fixed, verified demo sample so every run is deterministic.
    scenario = select_scenario(sample_id=3740)

    clean = scenario["clean_audio"]
    noisy = scenario["noisy_audio"]
    reference_noise = scenario["reference_noise"]

    print()
    print("Selected sample")
    print("----------------")
    print("Sample ID       :", scenario["sample_id"])
    print("Noise type      :", scenario["noise_type"])
    print("SNR             :", scenario["snr_db"], "dB")

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("1. NOISE CLASSIFICATION")
    print("=" * 70)

    model = load_classifier()

    predicted_noise, probabilities = (
        classify_noise(
            model,
            reference_noise
        )
    )

    print()
    print("Detected noise  :", predicted_noise)

    print()
    print("Classifier probabilities:")

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {class_name:<12} : "
            f"{probabilities[i] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("2. VAD")
    print("=" * 70)

    speech_mask = create_vad_mask(
        clean
    )

    speech_samples = np.sum(
        speech_mask
    )

    noise_only_samples = (
        len(speech_mask)
        - speech_samples
    )

    speech_percentage = (
        speech_samples /
        len(speech_mask)
    ) * 100

    print()
    print(
        "Speech samples  :",
        speech_samples
    )

    print(
        "Noise-only      :",
        noise_only_samples
    )

    print(
        "Speech           :",
        f"{speech_percentage:.2f}%"
    )

    print(
        "Noise-only       :",
        f"{100 - speech_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # SNR BEFORE
    # --------------------------------------------------------

    snr_before = calculate_snr(
        clean,
        noisy
    )

    # --------------------------------------------------------
    # NLMS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3. VAD-CONTROLLED NLMS ANC")
    print("=" * 70)

    print()
    print("Filter length   :", FILTER_LENGTH)
    print("Step size       :", MU)
    print("VAD threshold   :", VAD_THRESHOLD_DB, "dB")

    estimated_noise, enhanced, \
        update_count, freeze_count = vad_nlms(
            reference=reference_noise,
            desired=noisy,
            speech_mask=speech_mask,
            filter_length=FILTER_LENGTH,
            mu=MU,
            eps=EPS
        )

    # --------------------------------------------------------
    # SNR AFTER
    # --------------------------------------------------------

    snr_after = calculate_snr(
        clean,
        enhanced
    )

    improvement = (
        snr_after - snr_before
    )

    # --------------------------------------------------------
    # SAVE AUDIO
    # --------------------------------------------------------

    max_value = np.max(
        np.abs(enhanced)
    )

    if max_value > 1.0:

        enhanced = (
            enhanced / max_value
        )

    sf.write(
        OUTPUT_FILE,
        enhanced.astype(np.float32),
        SAMPLE_RATE
    )

    # --------------------------------------------------------
    # AUDIO COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("5. AUDIO COMPARISON")
    print("=" * 70)

    # Play the original noisy input first.
    play_audio(
        noisy,
        SAMPLE_RATE,
        "NOISY INPUT"
    )

    # Read back exactly what was saved as the enhanced output.
    enhanced_saved, enhanced_sr = sf.read(
        OUTPUT_FILE,
        dtype="float32"
    )

    play_audio(
        enhanced_saved,
        enhanced_sr,
        "ENHANCED OUTPUT"
    )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("6. FINAL RESULT")
    print("=" * 70)

    print()
    print("Detected noise  :", predicted_noise)
    print(
        "SNR before      :",
        f"{snr_before:.2f} dB"
    )

    print(
        "SNR after       :",
        f"{snr_after:.2f} dB"
    )

    print(
        "SNR improvement :",
        f"{improvement:+.2f} dB"
    )

    print()
    print("Adaptation statistics")
    print("----------------------")
    print(
        "Coefficient updates :",
        update_count
    )

    print(
        "Coefficient freezes :",
        freeze_count
    )

    print()
    print("Enhanced audio saved to:")
    print(
        os.path.abspath(OUTPUT_FILE)
    )

    print()
    print("=" * 70)
    print("END-TO-END DEMO COMPLETE")
    print("=" * 70)




if __name__ == "__main__":
    main()

