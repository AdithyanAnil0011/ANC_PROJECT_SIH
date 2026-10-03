import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\ANC_PROJECT_SIH")

DATASET_DIR = BASE_DIR / "dataset_balanced"

CLEAN_DIR = DATASET_DIR / "clean"
NOISY_DIR = DATASET_DIR / "noisy"
REFERENCE_DIR = DATASET_DIR / "reference"

METADATA_FILE = DATASET_DIR / "metadata.csv"

# Original defence-noise directory
ORIGINAL_NOISE_DIR = BASE_DIR / "noise"


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(METADATA_FILE)


# ============================================================
# BASIC INFORMATION
# ============================================================

def get_available_noise_types():
    """Return all available defence noise categories."""
    return sorted(metadata["noise_type"].unique().tolist())


def get_available_snr_values():
    """Return all available SNR values."""
    return sorted(metadata["snr_db"].unique().tolist())


# ============================================================
# SELECT SCENARIO
# ============================================================

def select_scenario(noise_type=None, snr_db=None, sample_id=None):
    """
    Select one sample from the balanced ANC dataset.

    Parameters
    ----------
    noise_type : str, optional
        Defence noise category:
        gunshot, shelling, vehicle, helicopter, fighter

    snr_db : int/float, optional
        Desired SNR:
        -5, 0, 5, 10, 15, 20

    sample_id : int, optional
        Exact sample ID to load.

    Returns
    -------
    dict
        Contains:
        - clean_audio
        - noisy_audio
        - reference_noise
        - original_noise_audio
        - metadata
        - file paths
    """

    # --------------------------------------------------------
    # Select by exact sample ID
    # --------------------------------------------------------

    if sample_id is not None:

        selected = metadata[
            metadata["sample_id"] == sample_id
        ]

        if len(selected) == 0:
            raise ValueError(
                f"No sample found with sample_id={sample_id}"
            )

        row = selected.iloc[0]

    # --------------------------------------------------------
    # Select by noise type + SNR
    # --------------------------------------------------------

    else:

        filtered = metadata.copy()

        if noise_type is not None:
            filtered = filtered[
                filtered["noise_type"].str.lower()
                == noise_type.lower()
            ]

        if snr_db is not None:
            filtered = filtered[
                filtered["snr_db"] == snr_db
            ]

        if len(filtered) == 0:

            raise ValueError(
                f"No scenario found for "
                f"noise_type={noise_type}, "
                f"snr_db={snr_db}"
            )

        # Randomly select one valid scenario
        row = filtered.sample(
            n=1,
            random_state=random.randint(0, 100000)
        ).iloc[0]


    # ========================================================
    # FILE PATHS
    # ========================================================

    sample_id = int(row["sample_id"])

    clean_path = CLEAN_DIR / row["clean_file"]

    noisy_path = NOISY_DIR / row["noisy_file"]

    # --------------------------------------------------------
    # Exact scaled reference noise used during mixing
    # --------------------------------------------------------

    if "reference_file" in row.index:

        reference_path = REFERENCE_DIR / row["reference_file"]

    else:

        # Fallback in case an older metadata file is used
        reference_path = (
            REFERENCE_DIR /
            f"reference_{sample_id:05d}.wav"
        )


    # --------------------------------------------------------
    # Original source noise file
    # --------------------------------------------------------

    original_noise_path = None

    if "noise_file" in row.index:

        original_noise_path = (
            ORIGINAL_NOISE_DIR /
            str(row["noise_file"])
        )


    # ========================================================
    # CHECK REQUIRED FILES
    # ========================================================

    if not clean_path.exists():
        raise FileNotFoundError(
            f"Clean file not found:\n{clean_path}"
        )

    if not noisy_path.exists():
        raise FileNotFoundError(
            f"Noisy file not found:\n{noisy_path}"
        )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference noise file not found:\n{reference_path}\n\n"
            f"Make sure the balanced dataset was regenerated "
            f"with reference-noise generation enabled."
        )


    # ========================================================
    # LOAD CLEAN AUDIO
    # ========================================================

    clean_audio, clean_sr = sf.read(clean_path)

    # Stereo → mono
    if clean_audio.ndim > 1:
        clean_audio = np.mean(
            clean_audio,
            axis=1
        )


    # ========================================================
    # LOAD NOISY AUDIO
    # ========================================================

    noisy_audio, noisy_sr = sf.read(noisy_path)

    # Stereo → mono
    if noisy_audio.ndim > 1:
        noisy_audio = np.mean(
            noisy_audio,
            axis=1
        )


    # ========================================================
    # LOAD EXACT REFERENCE NOISE
    # ========================================================

    reference_noise, reference_sr = sf.read(
        reference_path
    )

    # Stereo → mono
    if reference_noise.ndim > 1:
        reference_noise = np.mean(
            reference_noise,
            axis=1
        )


    # ========================================================
    # LOAD ORIGINAL SOURCE NOISE
    # ========================================================

    original_noise_audio = None
    original_noise_sr = None

    if (
        original_noise_path is not None
        and original_noise_path.exists()
    ):

        original_noise_audio, original_noise_sr = (
            sf.read(original_noise_path)
        )

        # Stereo → mono
        if original_noise_audio.ndim > 1:
            original_noise_audio = np.mean(
                original_noise_audio,
                axis=1
            )


    # ========================================================
    # RETURN SCENARIO
    # ========================================================

    return {

        # ----------------------------------------------------
        # Audio signals
        # ----------------------------------------------------

        "clean_audio": clean_audio,
        "clean_sr": clean_sr,

        "noisy_audio": noisy_audio,
        "noisy_sr": noisy_sr,

        # IMPORTANT:
        # This is the exact scaled/aligned noise component
        # used to generate the noisy signal.
        "reference_noise": reference_noise,
        "reference_sr": reference_sr,

        # Original unscaled noise file
        "noise_audio": original_noise_audio,
        "noise_sr": original_noise_sr,

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        "sample_id": sample_id,

        "noise_type": row["noise_type"],

        "snr_db": row["snr_db"],

        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

        "clean_path": clean_path,

        "noisy_path": noisy_path,

        "reference_path": reference_path,

        "noise_path": original_noise_path,

        # ----------------------------------------------------
        # Complete metadata row
        # ----------------------------------------------------

        "metadata": row.to_dict()
    }


# ============================================================
# RANDOM SCENARIO
# ============================================================

def get_random_scenario():
    """
    Select a completely random scenario.
    """

    row = metadata.sample(
        n=1,
        random_state=random.randint(0, 100000)
    ).iloc[0]

    return select_scenario(
        sample_id=int(row["sample_id"])
    )


# ============================================================
# SCENARIO BY NOISE TYPE
# ============================================================

def get_scenario_by_noise(noise_type, snr_db=None):
    """
    Select a scenario for a particular noise type.

    Example:
        get_scenario_by_noise("helicopter", -5)
    """

    return select_scenario(
        noise_type=noise_type,
        snr_db=snr_db
    )


# ============================================================
# VERIFY AUDIO RELATIONSHIP
# ============================================================

def verify_reference_relationship(scenario):
    """
    Verify that:

        noisy ≈ clean + reference_noise

    The reference noise is the exact scaled noise component
    used during dataset generation.

    Small differences are expected because the WAV files are
    stored using PCM quantization.
    """

    clean = scenario["clean_audio"].astype(np.float64)

    noisy = scenario["noisy_audio"].astype(np.float64)

    reference = scenario["reference_noise"].astype(np.float64)


    # --------------------------------------------------------
    # Match lengths
    # --------------------------------------------------------

    min_length = min(
        len(clean),
        len(noisy),
        len(reference)
    )

    clean = clean[:min_length]

    noisy = noisy[:min_length]

    reference = reference[:min_length]


    # --------------------------------------------------------
    # Reconstruct noisy signal
    # --------------------------------------------------------

    reconstructed = clean + reference


    # --------------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------------

    error = noisy - reconstructed

    max_error = np.max(
        np.abs(error)
    )

    mean_error = np.mean(
        np.abs(error)
    )

    rms_error = np.sqrt(
        np.mean(error ** 2)
    )


    print("\n" + "=" * 60)
    print("REFERENCE NOISE VERIFICATION")
    print("=" * 60)

    print(
        f"Sample ID        : {scenario['sample_id']}"
    )

    print(
        f"Noise type       : {scenario['noise_type']}"
    )

    print(
        f"SNR              : {scenario['snr_db']} dB"
    )

    print(
        f"Samples checked  : {min_length}"
    )

    print(
        f"Maximum error    : {max_error:.8f}"
    )

    print(
        f"Mean error       : {mean_error:.8f}"
    )

    print(
        f"RMS error        : {rms_error:.8f}"
    )

    print("=" * 60)

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "rms_error": rms_error
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("ECHOGUARD SCENARIO LOADER TEST")
    print("=" * 60)


    # --------------------------------------------------------
    # Available scenarios
    # --------------------------------------------------------

    print("\nAvailable noise types:")

    print(
        get_available_noise_types()
    )


    print("\nAvailable SNR values:")

    print(
        get_available_snr_values()
    )


    # --------------------------------------------------------
    # Select example scenario
    # --------------------------------------------------------

    scenario = get_scenario_by_noise(
        noise_type="helicopter",
        snr_db=-5
    )


    # --------------------------------------------------------
    # Print scenario information
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Audio information
    # --------------------------------------------------------

    print("\nAudio:")

    print(
        f"Clean: "
        f"{len(scenario['clean_audio'])} samples "
        f"@ {scenario['clean_sr']} Hz"
    )

    print(
        f"Noisy: "
        f"{len(scenario['noisy_audio'])} samples "
        f"@ {scenario['noisy_sr']} Hz"
    )

    print(
        f"Reference noise: "
        f"{len(scenario['reference_noise'])} samples "
        f"@ {scenario['reference_sr']} Hz"
    )


    if scenario["noise_audio"] is not None:

        print(
            f"Original noise: "
            f"{len(scenario['noise_audio'])} samples "
            f"@ {scenario['noise_sr']} Hz"
        )


    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    print("\nFiles:")

    print(
        f"Clean     : {scenario['clean_path']}"
    )

    print(
        f"Noisy     : {scenario['noisy_path']}"
    )

    print(
        f"Reference : {scenario['reference_path']}"
    )

    print(
        f"Original  : {scenario['noise_path']}"
    )


    # --------------------------------------------------------
    # Verify exact reference relationship
    # --------------------------------------------------------

    verify_reference_relationship(
        scenario
    )


    print("\nScenario loader working.")