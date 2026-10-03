import os
import wave

import torch
import pandas as pd

from torch.utils.data import Dataset


class DefenceNoiseDataset(Dataset):

    CLASS_NAMES = [
        "gunshot",
        "shelling",
        "vehicle",
        "helicopter",
        "fighter"
    ]

    CLASS_TO_IDX = {
        name: idx
        for idx, name in enumerate(CLASS_NAMES)
    }

    def __init__(
        self,
        metadata_file,
        reference_dir="dataset_balanced/reference",
        sample_rate=16000,
        segment_seconds=2.0
    ):

        self.df = pd.read_csv(metadata_file)

        self.reference_dir = reference_dir
        self.sample_rate = sample_rate

        self.segment_length = int(
            sample_rate * segment_seconds
        )

    def __len__(self):
        return len(self.df)

    # --------------------------------------------------
    # WAV loader using Python wave module
    # --------------------------------------------------
    def load_audio(self, path):

        with wave.open(path, "rb") as wf:

            sr = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()

            audio_bytes = wf.readframes(num_frames)

        # Convert PCM bytes to tensor
        if sample_width == 2:

            audio =  torch.tensor(
               list(
                   int.from_bytes(
                       audio_bytes[i:i+2],
                       byteorder="little",
                       signed=True
                   )
                   for i in range(0, len(audio_bytes), 2)
               ),
                dtype=torch.float32
            )

            audio = audio / 32768.0

        elif sample_width == 4:

            audio = torch.frombuffer(
                audio_bytes,
                dtype=torch.int32
            ).clone().float()

            audio = audio / 2147483648.0

        elif sample_width == 1:

            audio = torch.frombuffer(
                audio_bytes,
                dtype=torch.uint8
            ).clone().float()

            audio = (audio - 128.0) / 128.0

        else:
            raise RuntimeError(
                f"Unsupported WAV sample width: {sample_width}"
            )

        # Convert stereo/multichannel → mono
        if channels > 1:

            audio = audio.reshape(
                -1,
                channels
            )

            audio = audio.mean(dim=1)

        # Resampling is not expected for our generated dataset,
        # but explicitly verify the sample rate.
        if sr != self.sample_rate:

            raise RuntimeError(
                f"Unexpected sample rate {sr} Hz in {path}. "
                f"Expected {self.sample_rate} Hz."
            )

        return audio

    # --------------------------------------------------
    # RMS normalization
    # --------------------------------------------------
    def normalize_rms(self, audio):

        rms = torch.sqrt(
            torch.mean(audio ** 2) + 1e-8
        )

        audio = audio / rms

        return audio

    # --------------------------------------------------
    # Make exactly 2 seconds
    # --------------------------------------------------
    def fix_length(self, audio):

        length = audio.shape[0]

        if length < self.segment_length:

            repeat_count = (
                self.segment_length + length - 1
            ) // length

            audio = audio.repeat(repeat_count)

        audio = audio[:self.segment_length]

        return audio

    # --------------------------------------------------
    # Get sample
    # --------------------------------------------------
    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        reference_file = row["reference_file"]

        path = os.path.join(
            self.reference_dir,
            reference_file
        )

        audio = self.load_audio(path)

        audio = self.fix_length(audio)

        audio = self.normalize_rms(audio)

        label = self.CLASS_TO_IDX[
            row["noise_type"]
        ]

        return {
            "audio": audio,

            "label": torch.tensor(
                label,
                dtype=torch.long
            ),

            "noise_type": row["noise_type"],

            "snr_db": torch.tensor(
                row["snr_db"],
                dtype=torch.float32
            ),

            "sample_id": row["sample_id"]
        }


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    dataset = DefenceNoiseDataset(
        "dataset_balanced/train_metadata.csv"
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("\nSample information:")
    print("Audio shape :", sample["audio"].shape)
    print("Label       :", sample["label"])
    print("Noise type  :", sample["noise_type"])
    print("SNR         :", sample["snr_db"])
    print("Sample ID   :", sample["sample_id"])

    print(
        "\nExpected audio length:",
        16000 * 2
    )

    print(
        "Actual audio length:",
        sample["audio"].shape[0]
    )

    print(
        "Audio RMS:",
        torch.sqrt(torch.mean(sample["audio"] ** 2)).item()
    )