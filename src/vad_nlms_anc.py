import numpy as np
import soundfile as sf

from src.scenario_loader import select_scenario
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

FILTER_LENGTH = 128
MU = 0.01
EPS = 1e-8

SAMPLE_RATE = 16000

# VAD settings
FRAME_DURATION_MS = 20
FRAME_LENGTH = int(
    SAMPLE_RATE * FRAME_DURATION_MS / 1000
)

# Energy threshold for the simple VAD
VAD_THRESHOLD_DB = -40.0

# Test scenario
NOISE_TYPE = "helicopter"
SNR_DB = 20;

# OUTPUT_FILE = "enhanced_helicopter_vad_nlms.wav"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_ROOT /
    "enhanced_helicopter_vad_nlms.wav"
)


# ============================================================
# SNR CALCULATION
# ============================================================

def calculate_snr(clean, signal):

    clean = np.asarray(clean, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    noise = signal - clean

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 0:
        return float("inf")

    return 10 * np.log10(
        signal_power / noise_power
    )


# ============================================================
# SIMPLE ENERGY-BASED VAD
# ============================================================

def create_vad_mask(clean_audio,
                    frame_length=FRAME_LENGTH,
                    threshold_db=VAD_THRESHOLD_DB):

    clean_audio = np.asarray(
        clean_audio,
        dtype=np.float64
    )

    n_samples = len(clean_audio)

    speech_mask = np.zeros(
        n_samples,
        dtype=bool
    )

    # Small protection against log(0)
    eps = 1e-12

    # Calculate RMS energy for each frame
    for start in range(
        0,
        n_samples,
        frame_length
    ):

        end = min(
            start + frame_length,
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

        # Speech detected
        if energy_db > threshold_db:

            speech_mask[start:end] = True

    return speech_mask


# ============================================================
# VAD-GATED NLMS
# ============================================================

def vad_nlms(reference,
             desired,
             speech_mask,
             filter_length=128,
             mu=0.01,
             eps=1e-8):

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

    # --------------------------------------------------------
    # NLMS processing
    # --------------------------------------------------------

    for n in range(
        filter_length,
        n_samples
    ):

        # Reference vector
        x = reference[
            n-filter_length:n
        ][::-1]

        # Estimated noise
        y = np.dot(
            weights,
            x
        )

        # Error / enhanced signal
        e = desired[n] - y

        output[n] = y
        error[n] = e

        # ----------------------------------------------------
        # VAD PROTECTION
        #
        # If speech is active:
        #     FREEZE coefficient update
        #
        # If speech is inactive:
        #     Allow NLMS adaptation
        # ----------------------------------------------------

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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ECHOGUARD - VAD-GATED NLMS")
    print("=" * 70)

    print(f"Noise type       : {NOISE_TYPE}")
    print(f"Input SNR        : {SNR_DB} dB")
    print(f"Filter length    : {FILTER_LENGTH}")
    print(f"Step size (mu)   : {MU}")
    print(f"VAD frame        : {FRAME_DURATION_MS} ms")
    print(f"VAD threshold    : {VAD_THRESHOLD_DB} dB")

    # ========================================================
    # LOAD SCENARIO
    # ========================================================

    scenario = select_scenario(
        noise_type=NOISE_TYPE,
        snr_db=SNR_DB
    )

    clean = scenario["clean_audio"]
    noisy = scenario["noisy_audio"]
    reference_noise = scenario["reference_noise"]

    print("\nSelected scenario:")
    print(
        f"Sample ID : {scenario['sample_id']}"
    )
    print(
        f"Noise     : {scenario['noise_type']}"
    )
    print(
        f"SNR       : {scenario['snr_db']} dB"
    )

    print("\nAudio:")
    print(
        f"Clean     : {len(clean)} samples"
    )
    print(
        f"Noisy     : {len(noisy)} samples"
    )
    print(
        f"Reference : {len(reference_noise)} samples"
    )

    # ========================================================
    # VERIFY REFERENCE RELATIONSHIP
    # ========================================================

    reconstruction_error = (
        noisy -
        (clean + reference_noise)
    )

    rms_error = np.sqrt(
        np.mean(
            reconstruction_error ** 2
        )
    )

    print("\nReference verification:")
    print(
        f"RMS reconstruction error : "
        f"{rms_error:.8f}"
    )

    # ========================================================
    # CREATE VAD MASK
    # ========================================================

    print("\nCreating VAD mask...")

    speech_mask = create_vad_mask(
        clean_audio=clean
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

    noise_percentage = (
        noise_only_samples /
        len(speech_mask)
    ) * 100

    print(
        f"Speech frames/samples : "
        f"{speech_samples}"
    )

    print(
        f"Noise-only samples    : "
        f"{noise_only_samples}"
    )

    print(
        f"Speech percentage     : "
        f"{speech_percentage:.2f}%"
    )

    print(
        f"Noise-only percentage : "
        f"{noise_percentage:.2f}%"
    )

    # ========================================================
    # SNR BEFORE
    # ========================================================

    snr_before = calculate_snr(
        clean,
        noisy
    )

    print(
        f"\nSNR BEFORE VAD-NLMS : "
        f"{snr_before:.2f} dB"
    )

    # ========================================================
    # RUN VAD-GATED NLMS
    # ========================================================

    print("\nRunning VAD-gated NLMS...")
    print(
        f"Filter length : {FILTER_LENGTH}"
    )
    print(
        f"Step size     : {MU}"
    )

    estimated_noise, enhanced, update_count, freeze_count = (
        vad_nlms(
            reference=reference_noise,
            desired=noisy,
            speech_mask=speech_mask,
            filter_length=FILTER_LENGTH,
            mu=MU,
            eps=EPS
        )
    )

    # ========================================================
    # SNR AFTER
    # ========================================================

    snr_after = calculate_snr(
        clean,
        enhanced
    )

    improvement = (
        snr_after -
        snr_before
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        f"\nSNR AFTER VAD-NLMS  : "
        f"{snr_after:.2f} dB"
    )

    print(
        f"Improvement         : "
        f"{improvement:.2f} dB"
    )

    print("\nAdaptation statistics:")

    print(
        f"Coefficient updates : "
        f"{update_count}"
    )

    print(
        f"Coefficient freezes  : "
        f"{freeze_count}"
    )

    # ========================================================
    # SAVE AUDIO
    # ========================================================

    # Prevent clipping
    max_value = np.max(
        np.abs(enhanced)
    )

    if max_value > 1.0:

        enhanced = (
            enhanced /
            max_value
        )

    sf.write(
        OUTPUT_FILE,
        enhanced.astype(np.float32),
        SAMPLE_RATE
    )

    print(
        f"\nEnhanced audio saved as:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 70)
    print("VAD-NLMS TEST COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()