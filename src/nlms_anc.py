import numpy as np
import soundfile as sf

from src.scenario_loader import select_scenario


# ============================================================
# NLMS ADAPTIVE FILTER
# ============================================================

def nlms(reference, desired, filter_length=128, mu=0.5, eps=1e-8):
    """
    Normalized Least Mean Squares (NLMS) adaptive filter.

    Parameters
    ----------
    reference : np.ndarray
        Reference noise signal.

    desired : np.ndarray
        Noisy signal containing speech + noise.

    filter_length : int
        Number of adaptive filter coefficients.

    mu : float
        Adaptation step size.

    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    output : np.ndarray
        Error signal / enhanced speech estimate.

    weights : np.ndarray
        Final adaptive filter coefficients.
    """

    reference = np.asarray(reference, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)

    n_samples = min(
        len(reference),
        len(desired)
    )

    reference = reference[:n_samples]
    desired = desired[:n_samples]

    # Adaptive filter coefficients
    weights = np.zeros(
        filter_length,
        dtype=np.float64
    )

    # Reference buffer
    reference_buffer = np.zeros(
        filter_length,
        dtype=np.float64
    )

    # Output / error signal
    output = np.zeros(
        n_samples,
        dtype=np.float64
    )

    for n in range(n_samples):

        # Shift previous reference samples
        reference_buffer[1:] = reference_buffer[:-1]

        # Current reference sample
        reference_buffer[0] = reference[n]

        # Estimated noise
        estimated_noise = np.dot(
            weights,
            reference_buffer
        )

        # Error = desired - estimated noise
        error = desired[n] - estimated_noise

        # Store enhanced signal
        output[n] = error

        # Normalized LMS update
        reference_power = np.dot(
            reference_buffer,
            reference_buffer
        )

        weights += (
            mu
            * error
            * reference_buffer
            / (reference_power + eps)
        )

    return output, weights


# ============================================================
# SNR CALCULATION
# ============================================================

def calculate_snr(clean, signal):
    """
    Calculate SNR of signal relative to clean speech.

    signal = clean speech + residual noise

    SNR = 10 log10(signal_power / noise_power)
    """

    clean = np.asarray(
        clean,
        dtype=np.float64
    )

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    n_samples = min(
        len(clean),
        len(signal)
    )

    clean = clean[:n_samples]
    signal = signal[:n_samples]

    noise = signal - clean

    signal_power = np.mean(
        clean ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    if noise_power < 1e-12:
        return float("inf")

    return 10 * np.log10(
        signal_power / noise_power
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("ECHOGUARD NLMS ADAPTIVE NOISE CANCELLATION")
    print("=" * 60)


    # ========================================================
    # SELECT SCENARIO
    # ========================================================

    # Same scenario type as before.
    # You can change these values later.
    noise_type = "helicopter"
    snr_db = -5

    scenario = select_scenario(
        noise_type=noise_type,
        snr_db=snr_db
    )


    # ========================================================
    # GET AUDIO
    # ========================================================

    clean = scenario["clean_audio"]

    noisy = scenario["noisy_audio"]

    # IMPORTANT:
    # Use the exact scaled reference noise generated
    # for this noisy-clean pair.
    reference_noise = scenario["reference_noise"]


    # ========================================================
    # PRINT SCENARIO INFORMATION
    # ========================================================

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

    print(
        f"Sample rate : {scenario['noisy_sr']} Hz"
    )

    print(
        f"Clean samples     : {len(clean)}"
    )

    print(
        f"Noisy samples     : {len(noisy)}"
    )

    print(
        f"Reference samples : {len(reference_noise)}"
    )


    # ========================================================
    # CONVERT TO FLOAT64
    # ========================================================

    clean = np.asarray(
        clean,
        dtype=np.float64
    )

    noisy = np.asarray(
        noisy,
        dtype=np.float64
    )

    reference_noise = np.asarray(
        reference_noise,
        dtype=np.float64
    )


    # ========================================================
    # ALIGN SIGNAL LENGTHS
    # ========================================================

    n_samples = min(
        len(clean),
        len(noisy),
        len(reference_noise)
    )

    clean = clean[:n_samples]

    noisy = noisy[:n_samples]

    reference_noise = reference_noise[:n_samples]


    print(
        f"\nProcessing samples: {n_samples}"
    )


    # ========================================================
    # VERIFY REFERENCE
    # ========================================================

    reconstructed_noisy = (
        clean + reference_noise
    )

    reconstruction_error = (
        noisy - reconstructed_noisy
    )

    max_error = np.max(
        np.abs(reconstruction_error)
    )

    mean_error = np.mean(
        np.abs(reconstruction_error)
    )

    rms_error = np.sqrt(
        np.mean(
            reconstruction_error ** 2
        )
    )


    print("\nReference verification:")

    print(
        f"Maximum error : {max_error:.8f}"
    )

    print(
        f"Mean error    : {mean_error:.8f}"
    )

    print(
        f"RMS error     : {rms_error:.8f}"
    )


    # ========================================================
    # SNR BEFORE NLMS
    # ========================================================

    snr_before = calculate_snr(
        clean,
        noisy
    )

    print(
        f"\nSNR BEFORE NLMS : "
        f"{snr_before:.2f} dB"
    )


    # ========================================================
    # RUN NLMS
    # ========================================================

    print(
        "\nRunning NLMS adaptive filter..."
    )

    print(
        "Filter length : 128"
    )

    print(
        "Step size     : 0.5"
    )


    enhanced, weights = nlms(
        reference=reference_noise,
        desired=noisy,
        filter_length=128,
        mu=0.5,
        eps=1e-8
    )


    # ========================================================
    # SNR AFTER NLMS
    # ========================================================

    snr_after = calculate_snr(
        clean,
        enhanced
    )

    improvement = (
        snr_after - snr_before
    )


    print(
        f"\nSNR AFTER NLMS  : "
        f"{snr_after:.2f} dB"
    )

    print(
        f"Improvement     : "
        f"{improvement:.2f} dB"
    )


    # ========================================================
    # SAVE ENHANCED AUDIO
    # ========================================================

    output_file = (
        f"enhanced_"
        f"{scenario['noise_type']}"
        f"_nlms.wav"
    )

    # Clip before saving
    enhanced_to_save = np.clip(
        enhanced,
        -1.0,
        1.0
    )

    sf.write(
        output_file,
        enhanced_to_save,
        scenario["noisy_sr"]
    )


    print(
        f"\nEnhanced audio saved as:"
        f"\n{output_file}"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("NLMS RESULT")
    print("=" * 60)

    print(
        f"Noise type       : "
        f"{scenario['noise_type']}"
    )

    print(
        f"Input SNR        : "
        f"{snr_before:.2f} dB"
    )

    print(
        f"Output SNR       : "
        f"{snr_after:.2f} dB"
    )

    print(
        f"SNR improvement  : "
        f"{improvement:.2f} dB"
    )

    print("=" * 60)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()