"""
========================================================================
PHASE 2 — LESSON 64
TRAIN SPEAKER ENCODER FROM REAL AUDIO
========================================================================

PURPOSE
-------
Train a speaker encoder from REAL human speech.

This lesson moves from:

    Lesson 63
        synthetic speaker references
        ↓
    Lesson 64
        real speech → speaker embedding
        ↓
    Later lessons
        metric learning
        speaker verification
        reference-audio voice cloning

IMPORTANT
---------
NO PRETRAINED SPEAKER MODEL IS USED.

We build the speaker encoder ourselves.

If the required sample dataset is missing, this script AUTOMATICALLY
downloads LibriSpeech dev-clean and prepares a small speaker dataset.

DATASET
-------
Default automatic dataset:

    LibriSpeech dev-clean

The original LibriSpeech structure is approximately:

    LibriSpeech/
        dev-clean/
            speaker_id/
                chapter_id/
                    speaker-chapter-utterance.flac

We convert a controlled subset to WAV and reorganize it into:

    speaker_dataset/
        speaker_000/
            0000.wav
            0001.wav
            ...
        speaker_001/
            ...

WHY LIBRISPEECH?
----------------
It contains multiple real speakers and multiple utterances per speaker,
which is exactly what we need to learn speaker identity.

The complete LibriSpeech corpus is very large, but dev-clean is only
about 337 MB, making it practical for this lesson.

LEARNING OBJECTIVES
-------------------
STEP 1
    Automatically obtain real multi-speaker audio.

STEP 2
    Organize utterances by speaker identity.

STEP 3
    Convert audio into log-Mel spectrograms from scratch.

STEP 4
    Build a neural speaker encoder.

STEP 5
    Train the encoder using speaker classification.

STEP 6
    Extract fixed-dimensional speaker embeddings.

STEP 7
    Measure same-speaker versus different-speaker similarity.

STEP 8
    Visualize the learned speaker embedding space.

CORE IDEA
---------
A speaker encoder learns:

    audio
      ↓
    acoustic features
      ↓
    neural encoder
      ↓
    fixed-dimensional vector
      ↓
    speaker embedding

For example:

    Speaker A audio
        ↓
    [ 0.13, -0.42, 0.77, ... ]

    Speaker A another utterance
        ↓
    [ 0.11, -0.39, 0.75, ... ]

These vectors should become close.

For Speaker B:

    Speaker B audio
        ↓
    [ -0.51, 0.82, -0.14, ... ]

which should be farther away.

IMPORTANT LIMITATION
--------------------
Classification loss teaches:

    "Which training speaker is this?"

It does NOT directly optimize:

    "Are two arbitrary utterances from the same speaker close?"

That distinction becomes important in the next stage.

Later we will introduce metric learning:

    contrastive loss
    triplet loss
    cosine embedding objectives
    speaker verification

This lesson therefore establishes the first REAL speaker representation.

========================================================================
"""

import os
import sys
import math
import tarfile
import shutil
import random
import hashlib
import subprocess
import urllib.request
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "lesson64_outputs"

# Final organized dataset used by the training pipeline.
DATASET_DIR = ROOT / "speaker_dataset"

# Temporary downloaded dataset.
DOWNLOAD_DIR = ROOT / "lesson64_data"

LIBRISPEECH_DIR = DOWNLOAD_DIR / "LibriSpeech"
LIBRISPEECH_ARCHIVE = DOWNLOAD_DIR / "dev-clean.tar.gz"

# Official OpenSLR resource.
LIBRISPEECH_URL = (
    "https://www.openslr.org/resources/12/dev-clean.tar.gz"
)

# MD5 published by OpenSLR for dev-clean.
LIBRISPEECH_MD5 = "42e2234ba48799c1f50f24a7926300a1"

# --------------------------------------------------------------------------
# Dataset preparation
# --------------------------------------------------------------------------

# Number of speakers to prepare for this lesson.
MAX_SPEAKERS = 12

# Number of utterances per speaker.
MAX_UTTERANCES_PER_SPEAKER = 20

# Minimum number of utterances required for a speaker.
MIN_UTTERANCES_PER_SPEAKER = 6

# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------

SAMPLE_RATE = 16000

# Each training example is cropped/padded to two seconds.
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

N_MELS = 80

FMIN = 50.0
FMAX = 7600.0

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

EMBEDDING_DIM = 128

BATCH_SIZE = 8

EPOCHS = 30

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.20

NUM_WORKERS = 2

SEED = 42

# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def md5_file(path: Path, chunk_size: int = 1024 * 1024):
    digest = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================================
# DATASET DOWNLOAD
# ============================================================================

def download_file(url: str, destination: Path):
    """
    Download a file with visible progress.

    We intentionally use Python's standard library instead of requiring
    another dataset package.
    """

    ensure_directory(destination.parent)

    print()
    print("=" * 72)
    print("DOWNLOADING SAMPLE DATASET")
    print("=" * 72)
    print(f"URL:")
    print(f"    {url}")
    print()
    print(f"Destination:")
    print(f"    {destination}")
    print()

    if destination.exists():
        print("Archive already exists.")
        print("Skipping download.")

        return

    def progress_hook(block_num, block_size, total_size):
        if total_size <= 0:
            return

        downloaded = block_num * block_size

        percent = min(100.0, downloaded * 100.0 / total_size)

        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)

        print(
            f"\rProgress: {percent:6.2f}% "
            f"({downloaded_mb:8.1f} / {total_mb:8.1f} MB)",
            end="",
            flush=True,
        )

    try:
        urllib.request.urlretrieve(
            url,
            destination,
            reporthook=progress_hook,
        )

        print()

    except Exception:
        print()

        if destination.exists():
            try:
                destination.unlink()
            except Exception:
                pass

        raise


def verify_download(path: Path):
    """
    Verify the archive against the checksum published by OpenSLR.

    This protects the training pipeline from accidentally using a
    corrupted partial download.
    """

    print()
    print("Verifying downloaded archive...")

    actual = md5_file(path)

    print(f"Expected MD5: {LIBRISPEECH_MD5}")
    print(f"Actual MD5:   {actual}")

    if actual != LIBRISPEECH_MD5:
        raise RuntimeError(
            "Downloaded LibriSpeech archive failed MD5 verification.\n"
            f"Expected: {LIBRISPEECH_MD5}\n"
            f"Actual:   {actual}"
        )

    print("MD5 verification passed.")


# ============================================================================
# DATASET EXTRACTION
# ============================================================================

def extract_librispeech():
    """
    Extract the LibriSpeech archive.
    """

    if LIBRISPEECH_DIR.exists():
        dev_clean = LIBRISPEECH_DIR / "dev-clean"

        if dev_clean.exists():
            print()
            print("LibriSpeech archive already extracted.")
            return

    print()
    print("=" * 72)
    print("EXTRACTING LIBRISPEECH")
    print("=" * 72)

    ensure_directory(DOWNLOAD_DIR)

    with tarfile.open(LIBRISPEECH_ARCHIVE, "r:gz") as tar:
        tar.extractall(DOWNLOAD_DIR)

    print("Extraction complete.")


# ============================================================================
# AUDIO CONVERSION
# ============================================================================

def find_ffmpeg():
    """
    Find ffmpeg.

    LibriSpeech dev-clean stores audio as FLAC.

    Our lesson pipeline intentionally uses WAV internally, so we convert
    the selected files using ffmpeg.

    This is only data preparation. It is NOT a pretrained model.
    """

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is not None:
        return ffmpeg

    return None


def convert_flac_to_wav(flac_path: Path, wav_path: Path):
    """
    Convert one FLAC file to mono 16-kHz PCM WAV.

    ffmpeg is preferred because it is reliable for FLAC decoding.
    """

    ffmpeg = find_ffmpeg()

    if ffmpeg is None:
        raise RuntimeError(
            "\n"
            "ffmpeg was not found.\n\n"
            "This lesson automatically downloads LibriSpeech, whose "
            "audio files are FLAC.\n\n"
            "Please install ffmpeg:\n\n"
            "    sudo apt update\n"
            "    sudo apt install ffmpeg\n\n"
            "Then run this lesson again.\n"
        )

    ensure_directory(wav_path.parent)

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(flac_path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]

    subprocess.run(
        command,
        check=True,
    )


# ============================================================================
# PREPARE SPEAKER DATASET
# ============================================================================

def discover_librispeech_speakers():
    """
    Return speaker directories from LibriSpeech dev-clean.

    Expected structure:

        dev-clean/
            1272/
            1462/
            ...
    """

    root = LIBRISPEECH_DIR / "dev-clean"

    if not root.exists():
        raise FileNotFoundError(
            f"Expected LibriSpeech directory does not exist:\n{root}"
        )

    speakers = []

    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue

        flac_files = list(path.rglob("*.flac"))

        if len(flac_files) >= MIN_UTTERANCES_PER_SPEAKER:
            speakers.append((path.name, path, flac_files))

    return speakers


def prepare_speaker_dataset():
    """
    Automatically create speaker_dataset/ from LibriSpeech.

    The resulting structure matches the expected format of this lesson.
    """

    print()
    print("=" * 72)
    print("STEP 1 — PREPARE REAL MULTI-SPEAKER DATA")
    print("=" * 72)

    # If dataset already exists and contains enough speakers, reuse it.
    existing_speakers = []

    if DATASET_DIR.exists():

        for speaker_dir in sorted(DATASET_DIR.iterdir()):

            if not speaker_dir.is_dir():
                continue

            wavs = list(speaker_dir.glob("*.wav"))

            if len(wavs) >= MIN_UTTERANCES_PER_SPEAKER:
                existing_speakers.append((speaker_dir.name, wavs))

    if len(existing_speakers) >= 2:

        print("Existing speaker dataset found.")

        print(
            f"Speakers: {len(existing_speakers)}"
        )

        total_files = sum(
            len(files)
            for _, files in existing_speakers
        )

        print(
            f"Utterances: {total_files}"
        )

        return

    # ----------------------------------------------------------------------
    # Download LibriSpeech if necessary.
    # ----------------------------------------------------------------------

    ensure_directory(DOWNLOAD_DIR)

    if not LIBRISPEECH_ARCHIVE.exists():

        print()
        print(
            "No local speaker dataset found."
        )

        print(
            "Automatically downloading LibriSpeech dev-clean..."
        )

        download_file(
            LIBRISPEECH_URL,
            LIBRISPEECH_ARCHIVE,
        )

    else:

        print()
        print(
            "LibriSpeech archive already downloaded."
        )

    # ----------------------------------------------------------------------
    # Verify archive.
    # ----------------------------------------------------------------------

    verify_download(
        LIBRISPEECH_ARCHIVE
    )

    # ----------------------------------------------------------------------
    # Extract.
    # ----------------------------------------------------------------------

    extract_librispeech()

    # ----------------------------------------------------------------------
    # Discover speakers.
    # ----------------------------------------------------------------------

    speakers = discover_librispeech_speakers()

    print()
    print(
        f"Available speakers with enough utterances: "
        f"{len(speakers)}"
    )

    if len(speakers) < 2:
        raise RuntimeError(
            "LibriSpeech dev-clean does not contain enough speakers."
        )

    selected = speakers[:MAX_SPEAKERS]

    print(
        f"Selecting {len(selected)} speakers."
    )

    # ----------------------------------------------------------------------
    # Recreate organized dataset.
    # ----------------------------------------------------------------------

    if DATASET_DIR.exists():

        print()
        print(
            "Existing incomplete speaker_dataset found."
        )

        print(
            "Rebuilding it from the downloaded dataset..."
        )

        shutil.rmtree(DATASET_DIR)

    ensure_directory(DATASET_DIR)

    total_converted = 0

    print()
    print("=" * 72)
    print("CONVERTING SELECTED AUDIO")
    print("=" * 72)

    for new_index, (original_id, _, flac_files) in enumerate(selected):

        speaker_name = f"speaker_{new_index:03d}"

        speaker_dir = DATASET_DIR / speaker_name

        ensure_directory(speaker_dir)

        chosen_files = sorted(
            flac_files
        )[:MAX_UTTERANCES_PER_SPEAKER]

        print()
        print(
            f"{speaker_name} "
            f"(original LibriSpeech speaker {original_id})"
        )

        print(
            f"  Files: {len(chosen_files)}"
        )

        for utterance_index, flac_path in enumerate(chosen_files):

            wav_path = (
                speaker_dir
                / f"{utterance_index:04d}.wav"
            )

            if wav_path.exists():
                continue

            convert_flac_to_wav(
                flac_path,
                wav_path,
            )

            total_converted += 1

            print(
                f"\r  Converted: "
                f"{utterance_index + 1}/{len(chosen_files)}",
                end="",
                flush=True,
            )

        print()

    print()
    print(
        f"Total converted WAV files: {total_converted}"
    )

    print()
    print(
        "Prepared dataset:"
    )

    print(
        f"    {DATASET_DIR}"
    )


# ============================================================================
# WAV READER
# ============================================================================

def load_wav(path: Path):
    """
    Read PCM WAV using Python's standard library.

    Supported:
        8-bit
        16-bit
        24-bit
        32-bit

    Stereo audio is converted to mono.
    """

    import wave

    with wave.open(str(path), "rb") as wav:

        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()

        raw = wav.readframes(n_frames)

    if sample_width == 1:

        audio = np.frombuffer(
            raw,
            dtype=np.uint8,
        ).astype(np.float32)

        audio = (
            audio - 128.0
        ) / 128.0

    elif sample_width == 2:

        audio = np.frombuffer(
            raw,
            dtype=np.int16,
        ).astype(np.float32)

        audio /= 32768.0

    elif sample_width == 3:

        raw_bytes = np.frombuffer(
            raw,
            dtype=np.uint8,
        )

        usable = (
            len(raw_bytes) // 3
        ) * 3

        raw_bytes = raw_bytes[:usable]

        samples = raw_bytes.reshape(
            -1,
            3,
        )

        values = (
            samples[:, 0].astype(np.int32)
            |
            (samples[:, 1].astype(np.int32) << 8)
            |
            (samples[:, 2].astype(np.int32) << 16)
        )

        negative = (
            values & 0x800000
        ) != 0

        values[negative] -= 0x1000000

        audio = (
            values.astype(np.float32)
            / 8388608.0
        )

    elif sample_width == 4:

        audio = np.frombuffer(
            raw,
            dtype=np.int32,
        ).astype(np.float32)

        audio /= 2147483648.0

    else:

        raise ValueError(
            f"Unsupported WAV sample width: "
            f"{sample_width} bytes"
        )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels,
        )

        audio = np.mean(
            audio,
            axis=1,
        )

    return (
        audio.astype(np.float32),
        sample_rate,
    )


# ============================================================================
# RESAMPLING
# ============================================================================

def resample_linear(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
):
    """
    Simple linear interpolation resampler.

    For this lesson the automatically prepared audio is already 16 kHz,
    but this makes the pipeline robust to other input WAV files.
    """

    if source_rate == target_rate:
        return audio.astype(np.float32)

    if len(audio) < 2:
        return audio.astype(np.float32)

    duration = (
        len(audio)
        / float(source_rate)
    )

    target_length = max(
        1,
        int(round(
            duration * target_rate
        )),
    )

    source_positions = np.linspace(
        0.0,
        len(audio) - 1,
        num=target_length,
    )

    source_indices = np.arange(
        len(audio),
        dtype=np.float64,
    )

    resampled = np.interp(
        source_positions,
        source_indices,
        audio,
    )

    return resampled.astype(np.float32)


# ============================================================================
# AUDIO NORMALIZATION
# ============================================================================

def normalize_audio(audio):
    """
    Peak normalize audio.

    This is intentionally simple.

    We are learning speaker representation, not building a production
    audio mastering pipeline.
    """

    audio = audio.astype(
        np.float32
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:
        audio = audio / peak

    return audio


# ============================================================================
# FIXED-LENGTH AUDIO
# ============================================================================

def crop_or_pad_audio(
    audio,
    target_length,
    training=True,
):
    """
    Convert arbitrary-length audio into a fixed-length segment.

    Training:
        random crop when audio is longer.

    Validation:
        center crop.
    """

    if len(audio) == target_length:
        return audio

    if len(audio) > target_length:

        if training:

            max_start = (
                len(audio)
                - target_length
            )

            start = random.randint(
                0,
                max_start,
            )

        else:

            start = (
                len(audio)
                - target_length
            ) // 2

        return audio[
            start:start + target_length
        ]

    # Pad shorter utterances.

    result = np.zeros(
        target_length,
        dtype=np.float32,
    )

    start = (
        target_length
        - len(audio)
    ) // 2

    result[
        start:start + len(audio)
    ] = audio

    return result


# ============================================================================
# MEL FILTERBANK
# ============================================================================

def hz_to_mel(hz):
    return (
        2595.0
        * np.log10(
            1.0 + hz / 700.0
        )
    )


def mel_to_hz(mel):
    return (
        700.0
        * (
            10.0 ** (mel / 2595.0)
            - 1.0
        )
    )


def create_mel_filterbank(
    sample_rate,
    n_fft,
    n_mels,
    fmin,
    fmax,
):
    """
    Build triangular Mel filterbank from scratch.
    """

    n_freqs = (
        n_fft // 2 + 1
    )

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    fft_bins = np.floor(
        (n_fft + 1)
        * hz_points
        / sample_rate
    ).astype(int)

    filterbank = np.zeros(
        (
            n_mels,
            n_freqs,
        ),
        dtype=np.float32,
    )

    for m in range(n_mels):

        left = fft_bins[m]
        center = fft_bins[m + 1]
        right = fft_bins[m + 2]

        left = max(
            0,
            min(left, n_freqs - 1),
        )

        center = max(
            0,
            min(center, n_freqs - 1),
        )

        right = max(
            0,
            min(right, n_freqs - 1),
        )

        if center > left:

            filterbank[
                m,
                left:center,
            ] = (
                np.arange(
                    left,
                    center,
                )
                - left
            ) / float(
                center - left
            )

        if right > center:

            filterbank[
                m,
                center:right,
            ] = (
                right
                - np.arange(
                    center,
                    right,
                )
            ) / float(
                right - center
            )

    return filterbank


MEL_FILTERBANK = create_mel_filterbank(
    SAMPLE_RATE,
    N_FFT,
    N_MELS,
    FMIN,
    FMAX,
)


# ============================================================================
# LOG-MEL EXTRACTION
# ============================================================================

def waveform_to_log_mel(audio):
    """
    Convert waveform → log-Mel spectrogram.

    This is implemented explicitly instead of using a pretrained
    frontend.

    Pipeline:

        waveform
            ↓
        Hann window
            ↓
        STFT
            ↓
        power spectrum
            ↓
        Mel filterbank
            ↓
        log compression
    """

    waveform = torch.from_numpy(
        audio
    ).float()

    window = torch.hann_window(
        WIN_LENGTH,
    )

    stft = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True,
    )

    magnitude = torch.abs(
        stft
    )

    power = magnitude.pow(2.0)

    mel_filter = torch.from_numpy(
        MEL_FILTERBANK
    ).float()

    mel = torch.matmul(
        mel_filter,
        power,
    )

    mel = torch.clamp(
        mel,
        min=1e-8,
    )

    log_mel = torch.log(
        mel
    )

    return log_mel.numpy().astype(
        np.float32
    )


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def discover_dataset():
    """
    Discover:

        speaker_000/*.wav
        speaker_001/*.wav
        ...

    Returns records:

        {
            "path": Path,
            "speaker": speaker_name
        }
    """

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset preparation failed:\n"
            f"    {DATASET_DIR}"
        )

    speaker_dirs = [
        path
        for path in sorted(
            DATASET_DIR.iterdir()
        )
        if path.is_dir()
    ]

    records = []

    for speaker_dir in speaker_dirs:

        wavs = sorted(
            speaker_dir.glob("*.wav")
        )

        if len(wavs) < MIN_UTTERANCES_PER_SPEAKER:
            continue

        for wav_path in wavs:

            records.append(
                {
                    "path": wav_path,
                    "speaker": speaker_dir.name,
                }
            )

    if len(records) == 0:

        raise RuntimeError(
            "No WAV files were found in:\n"
            f"{DATASET_DIR}"
        )

    return records


# ============================================================================
# TRAIN / VALIDATION SPLIT
# ============================================================================

def split_records(records):
    """
    Split independently inside every speaker.

    This is important.

    We do NOT randomly split all files globally because that could
    accidentally leave some speakers completely absent from validation.
    """

    by_speaker = defaultdict(list)

    for record in records:

        by_speaker[
            record["speaker"]
        ].append(record)

    train_records = []
    val_records = []

    for speaker, items in sorted(
        by_speaker.items()
    ):

        items = list(items)

        random.shuffle(items)

        n_val = max(
            1,
            int(
                len(items)
                * VAL_RATIO
            ),
        )

        # Guarantee at least one training sample.
        n_val = min(
            n_val,
            len(items) - 1,
        )

        val_part = items[:n_val]
        train_part = items[n_val:]

        train_records.extend(
            train_part
        )

        val_records.extend(
            val_part
        )

    return (
        train_records,
        val_records,
    )


# ============================================================================
# SPEAKER DATASET
# ============================================================================

class SpeakerDataset(Dataset):

    def __init__(
        self,
        records,
        speaker_to_index,
        training,
    ):

        self.records = records

        self.speaker_to_index = (
            speaker_to_index
        )

        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        audio, sample_rate = load_wav(
            record["path"]
        )

        audio = resample_linear(
            audio,
            sample_rate,
            SAMPLE_RATE,
        )

        audio = normalize_audio(
            audio
        )

        audio = crop_or_pad_audio(
            audio,
            CLIP_SAMPLES,
            training=self.training,
        )

        mel = waveform_to_log_mel(
            audio
        )

        # Normalize each utterance.
        mel_mean = mel.mean()
        mel_std = mel.std()

        mel = (
            mel - mel_mean
        ) / (
            mel_std + 1e-5
        )

        label = self.speaker_to_index[
            record["speaker"]
        ]

        return (
            torch.from_numpy(mel).float(),
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


# ============================================================================
# SPEAKER ENCODER
# ============================================================================

class SpeakerEncoder(nn.Module):
    """
    Speaker encoder.

    Input:

        [B, 80, T]

    Output:

        [B, embedding_dim]

    Architecture:

        Mel
          ↓
        Conv1D
          ↓
        Conv1D
          ↓
        Conv1D
          ↓
        global average pooling
          ↓
        linear projection
          ↓
        L2-normalized speaker embedding
    """

    def __init__(
        self,
        embedding_dim=128,
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            N_MELS,
            128,
            kernel_size=5,
            padding=2,
        )

        self.bn1 = nn.BatchNorm1d(
            128
        )

        self.conv2 = nn.Conv1d(
            128,
            128,
            kernel_size=5,
            padding=2,
        )

        self.bn2 = nn.BatchNorm1d(
            128
        )

        self.conv3 = nn.Conv1d(
            128,
            256,
            kernel_size=5,
            padding=2,
        )

        self.bn3 = nn.BatchNorm1d(
            256
        )

        self.conv4 = nn.Conv1d(
            256,
            256,
            kernel_size=3,
            padding=1,
        )

        self.bn4 = nn.BatchNorm1d(
            256
        )

        self.projection = nn.Sequential(
            nn.Linear(
                256,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(
                256,
                embedding_dim,
            ),
        )

    def forward(self, mel):

        x = self.conv1(mel)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        # Global temporal pooling.
        x = x.mean(dim=-1)

        embedding = self.projection(
            x
        )

        # Normalize onto unit hypersphere.
        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding


# ============================================================================
# SPEAKER CLASSIFIER
# ============================================================================

class SpeakerClassifier(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_speakers,
    ):

        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dim,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(
                128,
                num_speakers,
            ),
        )

    def forward(self, embedding):

        return self.classifier(
            embedding
        )


# ============================================================================
# MODEL
# ============================================================================

class SpeakerRecognitionModel(nn.Module):

    def __init__(
        self,
        num_speakers,
        embedding_dim=128,
    ):

        super().__init__()

        self.encoder = SpeakerEncoder(
            embedding_dim
        )

        self.classifier = SpeakerClassifier(
            embedding_dim,
            num_speakers,
        )

    def forward(self, mel):

        embedding = self.encoder(
            mel
        )

        logits = self.classifier(
            embedding
        )

        return (
            embedding,
            logits,
        )


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for mel, labels in loader:

        mel = mel.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        embeddings, logits = model(
            mel
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * mel.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_items += (
            mel.size(0)
        )

    return (
        total_loss / max(
            1,
            total_items,
        ),
        total_correct / max(
            1,
            total_items,
        ),
    )


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    embeddings = []
    labels_all = []

    for mel, labels in loader:

        mel = mel.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        speaker_embeddings, logits = model(
            mel
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * mel.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_items += (
            mel.size(0)
        )

        embeddings.append(
            speaker_embeddings.cpu()
        )

        labels_all.append(
            labels.cpu()
        )

    if embeddings:

        embeddings = torch.cat(
            embeddings,
            dim=0,
        )

        labels_all = torch.cat(
            labels_all,
            dim=0,
        )

    else:

        embeddings = torch.empty(
            0,
            EMBEDDING_DIM,
        )

        labels_all = torch.empty(
            0,
            dtype=torch.long,
        )

    return (
        total_loss / max(
            1,
            total_items,
        ),
        total_correct / max(
            1,
            total_items,
        ),
        embeddings,
        labels_all,
    )


# ============================================================================
# COSINE SIMILARITY
# ============================================================================

def cosine_similarity(a, b):

    a = F.normalize(
        a,
        p=2,
        dim=-1,
    )

    b = F.normalize(
        b,
        p=2,
        dim=-1,
    )

    return (
        a * b
    ).sum(dim=-1)


# ============================================================================
# SAME / DIFFERENT SPEAKER ANALYSIS
# ============================================================================

def analyze_embedding_similarity(
    embeddings,
    labels,
):
    """
    Compare:

        same speaker
        vs
        different speaker

    using cosine similarity.
    """

    if len(embeddings) < 2:

        return (
            [],
            [],
        )

    same = []
    different = []

    for i in range(
        len(embeddings)
    ):

        for j in range(
            i + 1,
            len(embeddings),
        ):

            similarity = cosine_similarity(
                embeddings[i:i + 1],
                embeddings[j:j + 1],
            ).item()

            if (
                labels[i]
                == labels[j]
            ):
                same.append(
                    similarity
                )

            else:
                different.append(
                    similarity
                )

    return (
        same,
        different,
    )


# ============================================================================
# EMBEDDING VISUALIZATION
# ============================================================================

def pca_2d(embeddings):
    """
    Small PCA implementation using SVD.

    No sklearn dependency is required.
    """

    x = embeddings.numpy()

    x = (
        x
        - x.mean(axis=0, keepdims=True)
    )

    u, s, vt = np.linalg.svd(
        x,
        full_matrices=False,
    )

    projected = (
        x
        @ vt[:2].T
    )

    return projected


def save_embedding_plot(
    embeddings,
    labels,
    speaker_names,
):
    """
    Visualize learned speaker embeddings in 2D.

    PCA is used only for visualization.
    """

    if len(embeddings) < 2:
        return

    projected = pca_2d(
        embeddings
    )

    plt.figure(
        figsize=(10, 8)
    )

    unique_labels = sorted(
        set(
            labels.tolist()
        )
    )

    for label in unique_labels:

        mask = (
            labels.numpy()
            == label
        )

        plt.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=50,
            alpha=0.75,
            label=speaker_names[label],
        )

    plt.xlabel(
        "PCA dimension 1"
    )

    plt.ylabel(
        "PCA dimension 2"
    )

    plt.title(
        "Lesson 64 — Learned Speaker Embedding Space"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "speaker_embedding_space.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        f"Embedding visualization saved:"
    )

    print(
        f"    {path}"
    )


# ============================================================================
# TRAINING CURVES
# ============================================================================

def save_training_plot(
    history,
):
    epochs = np.arange(
        1,
        len(
            history["train_loss"]
        ) + 1,
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Cross-Entropy Loss"
    )

    plt.title(
        "Lesson 64 — Speaker Encoder Training"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "speaker_training.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Training plot saved:"
    )

    print(
        f"    {path}"
    )


# ============================================================================
# SIMILARITY PLOT
# ============================================================================

def save_similarity_plot(
    same,
    different,
):
    if (
        len(same) == 0
        or len(different) == 0
    ):
        print(
            "Not enough pairs for similarity plot."
        )

        return

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        same,
        bins=30,
        alpha=0.65,
        density=True,
        label="Same Speaker",
    )

    plt.hist(
        different,
        bins=30,
        alpha=0.65,
        density=True,
        label="Different Speaker",
    )

    plt.xlabel(
        "Cosine Similarity"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Speaker Embedding Similarity"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "speaker_similarity.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Similarity plot saved:"
    )

    print(
        f"    {path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    seed_everything(
        SEED
    )

    ensure_directory(
        OUTPUT_DIR
    )

    print("=" * 72)
    print(
        "PHASE 2 — LESSON 64"
    )

    print(
        "TRAIN SPEAKER ENCODER FROM REAL AUDIO"
    )

    print("=" * 72)

    print()
    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ======================================================================
    # STEP 1
    # ======================================================================

    prepare_speaker_dataset()

    # ======================================================================
    # STEP 2
    # ======================================================================

    print()
    print("=" * 72)
    print("STEP 2 — DISCOVER SPEAKER AUDIO")
    print("=" * 72)

    records = discover_dataset()

    speakers = sorted(
        set(
            r["speaker"]
            for r in records
        )
    )

    print(
        f"Speakers: {len(speakers)}"
    )

    print(
        f"Total utterances: {len(records)}"
    )

    for speaker in speakers:

        count = sum(
            r["speaker"] == speaker
            for r in records
        )

        print(
            f"  {speaker}: {count} utterances"
        )

    if len(speakers) < 2:

        raise RuntimeError(
            "At least two speakers are required."
        )

    # ======================================================================
    # STEP 3
    # ======================================================================

    print()
    print("=" * 72)
    print("STEP 3 — TRAIN / VALIDATION SPLIT")
    print("=" * 72)

    train_records, val_records = split_records(
        records
    )

    print(
        f"Training samples: "
        f"{len(train_records)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_records)}"
    )

    # ======================================================================
    # STEP 4
    # ======================================================================

    print()
    print("=" * 72)
    print("STEP 4 — BUILD SPEAKER DATASETS")
    print("=" * 72)

    speaker_to_index = {
        speaker: index
        for index, speaker in enumerate(
            speakers
        )
    }

    index_to_speaker = {
        index: speaker
        for speaker, index
        in speaker_to_index.items()
    }

    train_dataset = SpeakerDataset(
        train_records,
        speaker_to_index,
        training=True,
    )

    val_dataset = SpeakerDataset(
        val_records,
        speaker_to_index,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # ======================================================================
    # STEP 5
    # ======================================================================

    print()
    print("=" * 72)
    print("STEP 5 — BUILD SPEAKER ENCODER")
    print("=" * 72)

    model = SpeakerRecognitionModel(
        num_speakers=len(speakers),
        embedding_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # ======================================================================
    # STEP 6
    # ======================================================================

    print()
    print("=" * 72)
    print("STEP 6 — TRAIN")
    print("=" * 72)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")

    best_checkpoint = (
        OUTPUT_DIR
        / "speaker_encoder_best.pt"
    )

    print()
    print(
        "Starting training..."
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        (
            val_loss,
            val_acc,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_loss
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_acc
        )

        history["val_accuracy"].append(
            val_acc
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.3f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.3f} | "
            f"LR {current_lr:.2e}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "speaker_to_index": speaker_to_index,
                    "embedding_dim": EMBEDDING_DIM,
                    "sample_rate": SAMPLE_RATE,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "best_val_loss": best_val_loss,
                },
                best_checkpoint,
            )

            print(
                f"  -> Best checkpoint saved."
            )

    # ======================================================================
    # STEP 7
    # ======================================================================

    print()
    print("=" * 72)
    print("STEP 7 — LOAD BEST MODEL")
    print("=" * 72)

    checkpoint = torch.load(
        best_checkpoint,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    (
        final_val_loss,
        final_val_acc,
        embeddings,
        labels,
    ) = evaluate(
        model,
        val_loader,
        criterion,
    )

    print(
        f"Best validation loss: "
        f"{final_val_loss:.6f}"
    )

    print(
        f"Best validation accuracy: "
        f"{final_val_acc:.4f}"
    )

    # ======================================================================
    # STEP 8
    # ======================================================================

    print()
    print("=" * 72)
    print(
        "STEP 8 — SPEAKER EMBEDDING ANALYSIS"
    )
    print("=" * 72)

    (
        same,
        different,
    ) = analyze_embedding_similarity(
        embeddings,
        labels,
    )

    if len(same) > 0:

        print(
            f"Same-speaker pairs: "
            f"{len(same)}"
        )

        print(
            f"Mean same-speaker cosine similarity: "
            f"{np.mean(same):.4f}"
        )

    if len(different) > 0:

        print(
            f"Different-speaker pairs: "
            f"{len(different)}"
        )

        print(
            f"Mean different-speaker cosine similarity: "
            f"{np.mean(different):.4f}"
        )

    # ======================================================================
    # STEP 9
    # ======================================================================

    print()
    print("=" * 72)
    print("STEP 9 — VISUALIZATION")
    print("=" * 72)

    save_training_plot(
        history
    )

    save_embedding_plot(
        embeddings,
        labels,
        index_to_speaker,
    )

    save_similarity_plot(
        same,
        different,
    )

    # ======================================================================
    # STEP 10
    # ======================================================================

    final_checkpoint = (
        OUTPUT_DIR
        / "speaker_encoder_final.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "speaker_to_index": speaker_to_index,
            "embedding_dim": EMBEDDING_DIM,
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "best_val_loss": best_val_loss,
        },
        final_checkpoint,
    )

    print()
    print("=" * 72)
    print("LESSON 64 COMPLETE")
    print("=" * 72)

    print()
    print("Dataset:")
    print(
        f"    {DATASET_DIR}"
    )

    print()
    print("Best model:")
    print(
        f"    {best_checkpoint}"
    )

    print()
    print("Final model:")
    print(
        f"    {final_checkpoint}"
    )

    print()
    print("Visualizations:")

    print(
        f"    {OUTPUT_DIR / 'speaker_training.png'}"
    )

    print(
        f"    {OUTPUT_DIR / 'speaker_embedding_space.png'}"
    )

    print(
        f"    {OUTPUT_DIR / 'speaker_similarity.png'}"
    )

    print()
    print("What you learned:")
    print(
        "    REAL AUDIO"
        " -> LOG-MEL"
        " -> SPEAKER ENCODER"
        " -> SPEAKER EMBEDDING"
    )

    print()
    print(
        "Important:"
    )

    print(
        "    This lesson uses speaker classification."
    )

    print(
        "    The next improvement is metric learning,"
    )

    print(
        "    where same-speaker embeddings are explicitly"
    )

    print(
        "    pulled together and different-speaker"
    )

    print(
        "    embeddings are pushed apart."
    )

    print()
    print("=" * 72)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()