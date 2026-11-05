"""
========================================================================
PHASE 2 — LESSON 61
NEURAL VOCODER FROM SCRATCH
========================================================================

PURPOSE
-------
Convert a Mel-spectrogram back into a waveform using a neural vocoder
trained completely from scratch.

NO PRETRAINED MODELS
--------------------
This lesson does NOT use:

    - HiFi-GAN
    - WaveRNN
    - WaveGlow
    - MelGAN
    - BigVGAN
    - pretrained vocoders
    - pretrained speech models

The vocoder is implemented and trained from scratch with PyTorch.

------------------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------------------

STEP 1
------
Load Mel-spectrogram + original waveform pairs.

STEP 2
------
Understand the relationship:

    waveform
        ↓
    STFT / Mel extraction
        ↓
    Mel-spectrogram
        ↓
    neural vocoder
        ↓
    reconstructed waveform

STEP 3
------
Build a convolutional neural vocoder:

    Mel
      ↓
    projection
      ↓
    upsampling blocks
      ↓
    residual blocks
      ↓
    waveform

STEP 4
------
Train using:

    waveform L1 loss
    waveform MSE loss
    multi-resolution STFT loss

STEP 5
------
Compare:

    original waveform
    reconstructed waveform

STEP 6
------
Save:

    checkpoint
    waveform plots
    spectrogram plots
    reconstructed WAV

------------------------------------------------------------------------
IMPORTANT DATASET REQUIREMENT
------------------------------------------------------------------------

The Mel file alone is NOT enough to train a supervised neural vocoder.

We need:

    Mel --------------------> input
    Original waveform ------> target

The script therefore searches recursively for WAV files.

It supports situations such as:

    lesson55_outputs/
        features/
            sample.npz
        wav/
            sample.wav

or:

    lesson55_outputs/
        features/
            sample.npz

    lesson55_outputs/
        audio/
            sample.wav

or:

    dataset/
        wavs/
            sample.wav

or even:

    some_dataset/
        nested/
            folders/
                sample.wav

The WAV is matched primarily by filename stem.

------------------------------------------------------------------------
IMPORTANT
------------------------------------------------------------------------

If no WAV exists anywhere, this lesson cannot perform supervised
vocoder training.

In that case, create/use the original WAV files from Lesson 55.

Do NOT attempt to fabricate waveform targets from the Mel-spectrogram.

------------------------------------------------------------------------
REAL-WORLD EXAMPLE
------------------------------------------------------------------------

A production TTS system normally looks approximately like:

    Text
      ↓
    phonemes
      ↓
    acoustic model
      ↓
    Mel-spectrogram
      ↓
    neural vocoder
      ↓
    speech waveform

The vocoder is the final component that turns the acoustic representation
into audible speech.

------------------------------------------------------------------------
"""

import os
import json
import math
import wave
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("lesson61_outputs")
FEATURE_DIR = Path("lesson55_outputs/features")

# Search roots for WAV files.
#
# We intentionally search recursively instead of assuming a fixed directory.
SEARCH_ROOTS = [
    Path("lesson55_outputs"),
    Path("lesson55_outputs/wav"),
    Path("lesson55_outputs/audio"),
    Path("lesson55_outputs/wavs"),
    Path("wav"),
    Path("wavs"),
    Path("audio"),
    Path("audios"),
    Path("dataset"),
    Path("datasets"),
    Path("data"),
    Path("."),
]

BATCH_SIZE = 1
EPOCHS = 300

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-6

GRAD_CLIP = 1.0

PATIENCE = 40

MAX_AUDIO_SAMPLES = 32768

TARGET_SAMPLE_RATE = 16000

DEFAULT_HOP_LENGTH = 256

N_MELS = 80

# Vocoder upsampling.
#
# 4 × 4 × 4 × 4 = 256
#
# Therefore the network approximately performs:
#
#     Mel frame → 256 waveform samples
#
# corresponding to a hop length of 256.
UPSAMPLE_FACTORS = [4, 4, 4, 4]


# ============================================================================
# RANDOM SEED
# ============================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# WAV UTILITIES
# ============================================================================

def read_wav(path):
    """
    Read PCM WAV using Python's standard wave module.

    Returns:
        waveform: float32 numpy array in [-1, 1]
        sample_rate: integer
    """

    with wave.open(str(path), "rb") as wf:

        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0

    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0

    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        audio = audio / 2147483648.0

    else:
        raise RuntimeError(
            f"Unsupported WAV sample width {sample_width} bytes: {path}"
        )

    if n_channels > 1:

        if len(audio) % n_channels != 0:
            audio = audio[:len(audio) - (len(audio) % n_channels)]

        audio = audio.reshape(-1, n_channels)

        audio = audio.mean(axis=1)

    return audio.astype(np.float32), sample_rate


def write_wav(path, audio, sample_rate):
    """
    Write mono float waveform to 16-bit PCM WAV.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)

    audio = np.clip(audio, -1.0, 1.0)

    pcm = (audio * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def resample_linear(audio, source_sr, target_sr):
    """
    Simple educational linear resampler.

    This is intentionally simple because the purpose of the lesson is
    understanding the neural vocoder rather than implementing a
    production-quality resampling library.
    """

    if source_sr == target_sr:
        return audio.astype(np.float32)

    if len(audio) < 2:
        return audio.astype(np.float32)

    duration = (len(audio) - 1) / float(source_sr)

    new_length = int(round(duration * target_sr)) + 1

    old_x = np.linspace(
        0.0,
        duration,
        len(audio),
        dtype=np.float64,
    )

    new_x = np.linspace(
        0.0,
        duration,
        new_length,
        dtype=np.float64,
    )

    resampled = np.interp(
        new_x,
        old_x,
        audio,
    )

    return resampled.astype(np.float32)


# ============================================================================
# MEL LOADING
# ============================================================================

def load_mel(npz_path):
    """
    Load Mel representation from Lesson 55.

    Supports:

        log_mel
        normalized_log_mel

    and automatically handles either:

        [n_mels, frames]

    or:

        [frames, n_mels]
    """

    data = np.load(npz_path, allow_pickle=True)

    if "normalized_log_mel" in data:
        mel = data["normalized_log_mel"]

    elif "log_mel" in data:
        mel = data["log_mel"]

    else:
        raise RuntimeError(
            f"No Mel representation found in {npz_path}. "
            f"Available fields: {list(data.keys())}"
        )

    mel = np.asarray(mel, dtype=np.float32)

    # Convert to [n_mels, frames]
    if mel.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D Mel array, got shape {mel.shape}: {npz_path}"
        )

    # Try explicit n_mels information.
    n_mels = None

    if "n_mels" in data:
        try:
            n_mels = int(np.asarray(data["n_mels"]).reshape(-1)[0])
        except Exception:
            pass

    if n_mels is None:
        n_mels = N_MELS

    if mel.shape[0] != n_mels and mel.shape[1] == n_mels:
        mel = mel.T

    elif mel.shape[0] != n_mels:

        # Common Mel dimensions.
        common_mel_sizes = {40, 64, 80, 96, 100, 128}

        if mel.shape[1] in common_mel_sizes:
            mel = mel.T

    return mel.astype(np.float32)


def load_hop_length(npz_path):
    """
    Read hop length from the Lesson 55 NPZ when available.
    """

    data = np.load(npz_path, allow_pickle=True)

    if "hop_length" in data:

        try:
            return int(
                np.asarray(data["hop_length"]).reshape(-1)[0]
            )
        except Exception:
            pass

    return DEFAULT_HOP_LENGTH


def load_sample_rate(npz_path):
    """
    Read sample rate from the Lesson 55 NPZ when available.
    """

    data = np.load(npz_path, allow_pickle=True)

    if "sample_rate" in data:

        try:
            return int(
                np.asarray(data["sample_rate"]).reshape(-1)[0]
            )
        except Exception:
            pass

    return TARGET_SAMPLE_RATE


# ============================================================================
# FILE MATCHING
# ============================================================================

def normalize_stem(stem):
    """
    Normalize filenames so that small naming differences can be matched.

    Example:

        sample_001
        sample-001
        sample 001

    become approximately equivalent.
    """

    value = stem.lower().strip()

    replacements = [
        "_mel",
        "_mels",
        "_audio",
        "_wav",
        "_waveform",
        "-mel",
        "-mels",
        "-audio",
        "-wav",
        "-waveform",
    ]

    for token in replacements:
        if value.endswith(token):
            value = value[:-len(token)]

    value = value.replace(" ", "")
    value = value.replace("_", "")
    value = value.replace("-", "")

    return value


def collect_wav_files():
    """
    Recursively collect WAV files.

    Duplicates are removed.

    We deliberately search recursively because Lesson 55 datasets may
    place audio in different directory structures.
    """

    wav_files = set()

    print()
    print("=" * 72)
    print("SEARCHING FOR WAV FILES")
    print("=" * 72)

    for root in SEARCH_ROOTS:

        root = Path(root)

        if not root.exists():
            continue

        try:

            for path in root.rglob("*.wav"):

                if path.is_file():
                    wav_files.add(path.resolve())

        except Exception as exc:

            print(
                f"WARNING: could not search {root}: {exc}"
            )

    wav_files = sorted(wav_files)

    print(f"Found WAV files: {len(wav_files)}")

    for path in wav_files[:30]:
        print(f"  {path}")

    if len(wav_files) > 30:
        print(
            f"  ... and {len(wav_files) - 30} more"
        )

    return wav_files


def find_matching_wav(npz_path, wav_files):
    """
    Find the WAV corresponding to one Mel NPZ.

    Matching priority:

        1. exact filename stem
        2. normalized stem
        3. parent-directory/stem combinations
    """

    npz_stem = npz_path.stem

    # ------------------------------------------------------------
    # Exact stem.
    # ------------------------------------------------------------

    exact = []

    for wav in wav_files:

        if wav.stem.lower() == npz_stem.lower():
            exact.append(wav)

    if len(exact) > 0:
        return exact[0]

    # ------------------------------------------------------------
    # Normalized stem.
    # ------------------------------------------------------------

    target = normalize_stem(npz_stem)

    normalized = []

    for wav in wav_files:

        if normalize_stem(wav.stem) == target:
            normalized.append(wav)

    if len(normalized) > 0:
        return normalized[0]

    # ------------------------------------------------------------
    # Common prefixes.
    # ------------------------------------------------------------

    candidates = []

    for wav in wav_files:

        wav_stem = normalize_stem(wav.stem)

        if (
            wav_stem.startswith(target)
            or target.startswith(wav_stem)
        ):
            candidates.append(wav)

    if len(candidates) == 1:
        return candidates[0]

    return None


# ============================================================================
# RECORD LOADING
# ============================================================================

def load_records():
    """
    Build Mel/WAV training records.
    """

    if not FEATURE_DIR.exists():

        raise RuntimeError(
            f"Mel feature directory does not exist:\n"
            f"    {FEATURE_DIR.resolve()}\n\n"
            f"Run Lesson 55 first."
        )

    npz_files = sorted(
        FEATURE_DIR.glob("*.npz")
    )

    if len(npz_files) == 0:

        raise RuntimeError(
            f"No NPZ Mel files found in:\n"
            f"    {FEATURE_DIR.resolve()}"
        )

    print()
    print("=" * 72)
    print("STEP 1 — LOAD MEL + WAVEFORM PAIRS")
    print("=" * 72)

    print(f"Mel feature directory:")
    print(f"  {FEATURE_DIR.resolve()}")

    print(f"Mel files found: {len(npz_files)}")

    wav_files = collect_wav_files()

    if len(wav_files) == 0:

        raise RuntimeError(
            "\n"
            "No WAV files were found anywhere in the searched directories.\n\n"
            "Lesson 61 requires the ORIGINAL WAV waveform as the training\n"
            "target. A Mel-spectrogram alone cannot provide the waveform\n"
            "target for supervised vocoder training.\n\n"
            "Please check where the original Lesson 55 WAV files are stored."
        )

    records = []

    for npz_path in npz_files:

        try:

            mel = load_mel(npz_path)

            hop_length = load_hop_length(npz_path)

            sample_rate = load_sample_rate(npz_path)

            wav_path = find_matching_wav(
                npz_path,
                wav_files,
            )

            if wav_path is None:

                print(
                    f"WARNING: no matching WAV for:\n"
                    f"    {npz_path}"
                )

                continue

            waveform, wav_sr = read_wav(wav_path)

            if wav_sr != sample_rate:

                waveform = resample_linear(
                    waveform,
                    wav_sr,
                    sample_rate,
                )

            # --------------------------------------------------------
            # Match waveform length to Mel frame count.
            #
            # The approximate expected number of samples is:
            #
            #     Mel frames × hop length
            # --------------------------------------------------------

            expected_samples = mel.shape[1] * hop_length

            if len(waveform) > expected_samples:

                waveform = waveform[:expected_samples]

            elif len(waveform) < expected_samples:

                waveform = np.pad(
                    waveform,
                    (
                        0,
                        expected_samples - len(waveform),
                    ),
                )

            if len(waveform) > MAX_AUDIO_SAMPLES:

                waveform = waveform[
                    :MAX_AUDIO_SAMPLES
                ]

                max_frames = max(
                    1,
                    len(waveform) // hop_length,
                )

                mel = mel[:, :max_frames]

                waveform = waveform[
                    :mel.shape[1] * hop_length
                ]

            if len(waveform) < hop_length:

                print(
                    f"WARNING: waveform too short: {wav_path}"
                )

                continue

            records.append(
                {
                    "mel": mel.astype(np.float32),
                    "waveform": waveform.astype(np.float32),
                    "sample_rate": sample_rate,
                    "hop_length": hop_length,
                    "mel_path": str(npz_path),
                    "wav_path": str(wav_path),
                }
            )

            print()
            print("MATCHED:")
            print(f"  Mel : {npz_path}")
            print(f"  WAV : {wav_path}")
            print(f"  Mel shape : {mel.shape}")
            print(f"  Waveform : {waveform.shape}")
            print(f"  Sample rate : {sample_rate}")
            print(f"  Hop length : {hop_length}")

        except Exception as exc:

            print(
                f"WARNING: skipping sample {npz_path}: {exc}"
            )

    if len(records) == 0:

        raise RuntimeError(
            "\n"
            "No valid Mel/WAV pairs were found.\n\n"
            "The WAV files may exist, but their filenames do not match "
            "the Mel filenames. Check the printed WAV list above."
        )

    print()
    print(f"Valid Mel/WAV pairs: {len(records)}")

    return records


# ============================================================================
# DATASET
# ============================================================================

class VocoderDataset(Dataset):

    def __init__(self, records):

        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        mel = torch.from_numpy(
            record["mel"]
        )

        waveform = torch.from_numpy(
            record["waveform"]
        )

        return {
            "mel": mel,
            "waveform": waveform,
            "sample_rate": record["sample_rate"],
            "hop_length": record["hop_length"],
            "mel_path": record["mel_path"],
            "wav_path": record["wav_path"],
        }


# ============================================================================
# COLLATE
# ============================================================================

def collate_batch(batch):
    """
    For this educational lesson we use batch size 1 by default.

    If multiple samples are provided, they are padded to common lengths.
    """

    max_frames = max(
        item["mel"].shape[1]
        for item in batch
    )

    max_samples = max(
        item["waveform"].shape[0]
        for item in batch
    )

    n_mels = batch[0]["mel"].shape[0]

    mel_batch = torch.zeros(
        len(batch),
        n_mels,
        max_frames,
        dtype=torch.float32,
    )

    waveform_batch = torch.zeros(
        len(batch),
        max_samples,
        dtype=torch.float32,
    )

    mel_lengths = []
    waveform_lengths = []

    for i, item in enumerate(batch):

        mel = item["mel"]
        waveform = item["waveform"]

        mel_batch[
            i,
            :,
            :mel.shape[1]
        ] = mel

        waveform_batch[
            i,
            :waveform.shape[0]
        ] = waveform

        mel_lengths.append(
            mel.shape[1]
        )

        waveform_lengths.append(
            waveform.shape[0]
        )

    return {
        "mel": mel_batch,
        "waveform": waveform_batch,
        "mel_lengths": torch.tensor(
            mel_lengths,
            dtype=torch.long,
        ),
        "waveform_lengths": torch.tensor(
            waveform_lengths,
            dtype=torch.long,
        ),
    }


# ============================================================================
# RESIDUAL BLOCK
# ============================================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation=1,
    ):

        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )

        self.norm1 = nn.GroupNorm(
            8,
            channels,
        )

        self.norm2 = nn.GroupNorm(
            8,
            channels,
        )

    def forward(self, x):

        residual = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = F.leaky_relu(y, 0.2)

        y = self.conv2(y)
        y = self.norm2(y)

        y = y + residual

        y = F.leaky_relu(
            y,
            0.2,
        )

        return y


# ============================================================================
# UPSAMPLING BLOCK
# ============================================================================

class UpsampleBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        factor,
    ):

        super().__init__()

        kernel_size = factor * 2

        self.upsample = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=factor,
            padding=factor // 2,
        )

        self.norm = nn.GroupNorm(
            8,
            out_channels,
        )

        self.residual1 = ResidualBlock(
            out_channels,
            dilation=1,
        )

        self.residual2 = ResidualBlock(
            out_channels,
            dilation=2,
        )

    def forward(self, x):

        x = self.upsample(x)

        x = self.norm(x)

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.residual1(x)

        x = self.residual2(x)

        return x


# ============================================================================
# NEURAL VOCODER
# ============================================================================

class NeuralVocoder(nn.Module):
    """
    Educational Mel-to-waveform neural vocoder.

    Input:

        [B, n_mels, frames]

    Output:

        [B, 1, samples]
    """

    def __init__(
        self,
        n_mels=N_MELS,
        base_channels=128,
    ):

        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv1d(
                n_mels,
                base_channels,
                kernel_size=7,
                padding=3,
            ),
            nn.GroupNorm(
                8,
                base_channels,
            ),
            nn.LeakyReLU(0.2),
        )

        channels = base_channels

        self.upsample_blocks = nn.ModuleList()

        factors = UPSAMPLE_FACTORS

        channel_schedule = [
            128,
            96,
            64,
            32,
        ]

        for factor, out_channels in zip(
            factors,
            channel_schedule,
        ):

            self.upsample_blocks.append(
                UpsampleBlock(
                    channels,
                    out_channels,
                    factor,
                )
            )

            channels = out_channels

        self.output = nn.Sequential(
            nn.Conv1d(
                channels,
                16,
                kernel_size=7,
                padding=3,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv1d(
                16,
                1,
                kernel_size=7,
                padding=3,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        mel,
        target_length=None,
    ):

        x = self.input_projection(mel)

        for block in self.upsample_blocks:
            x = block(x)

        x = self.output(x)

        if target_length is not None:

            if x.shape[-1] > target_length:

                x = x[..., :target_length]

            elif x.shape[-1] < target_length:

                x = F.pad(
                    x,
                    (
                        0,
                        target_length - x.shape[-1],
                    ),
                )

        return x


# ============================================================================
# STFT LOSS
# ============================================================================

def stft_magnitude(
    waveform,
    n_fft,
    hop_length=None,
):

    if hop_length is None:
        hop_length = n_fft // 4

    window = torch.hann_window(
        n_fft,
        device=waveform.device,
    )

    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
    )

    return torch.abs(spec)


def multi_resolution_stft_loss(
    predicted,
    target,
):

    fft_sizes = [
        256,
        512,
        1024,
    ]

    total = 0.0

    for n_fft in fft_sizes:

        pred_mag = stft_magnitude(
            predicted,
            n_fft,
        )

        target_mag = stft_magnitude(
            target,
            n_fft,
        )

        min_time = min(
            pred_mag.shape[-1],
            target_mag.shape[-1],
        )

        pred_mag = pred_mag[
            ...,
            :min_time
        ]

        target_mag = target_mag[
            ...,
            :min_time
        ]

        linear_loss = F.l1_loss(
            pred_mag,
            target_mag,
        )

        log_pred = torch.log(
            pred_mag + 1e-5
        )

        log_target = torch.log(
            target_mag + 1e-5
        )

        log_loss = F.l1_loss(
            log_pred,
            log_target,
        )

        total = total + linear_loss + log_loss

    return total / len(fft_sizes)


# ============================================================================
# VOCODER LOSS
# ============================================================================

def vocoder_loss(
    predicted,
    target,
):

    predicted = predicted.squeeze(1)

    target = target.squeeze(1)

    l1 = F.l1_loss(
        predicted,
        target,
    )

    mse = F.mse_loss(
        predicted,
        target,
    )

    stft = multi_resolution_stft_loss(
        predicted,
        target,
    )

    total = (
        1.0 * l1
        + 0.5 * mse
        + 1.0 * stft
    )

    return total, l1, mse, stft


# ============================================================================
# PARAMETER COUNT
# ============================================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model,
    loader,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_loss = float("inf")

    best_state = None

    history = []

    patience_counter = 0

    print()
    print("=" * 72)
    print("STEP 2 — TRAIN NEURAL VOCODER")
    print("=" * 72)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        epoch_loss = 0.0

        epoch_l1 = 0.0
        epoch_mse = 0.0
        epoch_stft = 0.0

        batches = 0

        for batch in loader:

            mel = batch["mel"].to(
                DEVICE
            )

            target = batch["waveform"].to(
                DEVICE
            )

            target_length = target.shape[-1]

            target = target.unsqueeze(1)

            optimizer.zero_grad(
                set_to_none=True
            )

            predicted = model(
                mel,
                target_length=target_length,
            )

            total, l1, mse, stft = vocoder_loss(
                predicted,
                target,
            )

            if not torch.isfinite(total):

                print(
                    "WARNING: non-finite loss detected. "
                    "Skipping batch."
                )

                continue

            total.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            epoch_loss += total.item()
            epoch_l1 += l1.item()
            epoch_mse += mse.item()
            epoch_stft += stft.item()

            batches += 1

        if batches == 0:
            continue

        epoch_loss /= batches
        epoch_l1 /= batches
        epoch_mse /= batches
        epoch_stft /= batches

        history.append(
            {
                "loss": epoch_loss,
                "l1": epoch_l1,
                "mse": epoch_mse,
                "stft": epoch_stft,
            }
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_loss": best_loss,
                    "epoch": epoch,
                },
                OUTPUT_DIR / "neural_vocoder_best.pt",
            )

        else:

            patience_counter += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:4d}/{EPOCHS} | "
                f"loss={epoch_loss:.6f} | "
                f"L1={epoch_l1:.6f} | "
                f"MSE={epoch_mse:.6f} | "
                f"STFT={epoch_stft:.6f}"
            )

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, history, best_loss


# ============================================================================
# TRAINING PLOT
# ============================================================================

def save_training_plot(history):

    if len(history) == 0:
        return

    losses = [
        x["loss"]
        for x in history
    ]

    l1 = [
        x["l1"]
        for x in history
    ]

    mse = [
        x["mse"]
        for x in history
    ]

    stft = [
        x["stft"]
        for x in history
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        losses,
        label="Total",
    )

    plt.plot(
        l1,
        label="Waveform L1",
    )

    plt.plot(
        mse,
        label="Waveform MSE",
    )

    plt.plot(
        stft,
        label="STFT",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Lesson 61 — Neural Vocoder Training"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "vocoder_training.png",
        dpi=150,
    )

    plt.close()


# ============================================================================
# WAVEFORM PLOT
# ============================================================================

def save_waveform_plot(
    waveform,
    sample_rate,
    filename,
    title,
):

    waveform = np.asarray(
        waveform
    ).reshape(-1)

    time = np.arange(
        len(waveform)
    ) / float(sample_rate)

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        time,
        waveform,
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.title(title)

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150,
    )

    plt.close()


# ============================================================================
# SPECTROGRAM PLOT
# ============================================================================

def save_spectrogram_plot(
    waveform,
    sample_rate,
    filename,
    title,
):

    waveform = np.asarray(
        waveform
    ).reshape(-1)

    tensor = torch.from_numpy(
        waveform
    )

    n_fft = 512

    hop = 128

    window = torch.hann_window(
        n_fft
    )

    spec = torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )

    magnitude = torch.abs(
        spec
    )

    log_spec = torch.log(
        magnitude + 1e-5
    ).numpy()

    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        log_spec,
        aspect="auto",
        origin="lower",
    )

    plt.xlabel("Frame")
    plt.ylabel("Frequency Bin")

    plt.title(title)

    plt.colorbar(
        label="Log Magnitude"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150,
    )

    plt.close()


# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def reconstruct(
    model,
    record,
):

    model.eval()

    mel = torch.from_numpy(
        record["mel"]
    ).unsqueeze(0)

    mel = mel.to(
        DEVICE
    )

    waveform_length = len(
        record["waveform"]
    )

    prediction = model(
        mel,
        target_length=waveform_length,
    )

    prediction = prediction[
        0,
        0,
    ]

    prediction = prediction.detach().cpu().numpy()

    return prediction.astype(
        np.float32
    )


# ============================================================================
# SAVE COMPARISON
# ============================================================================

def save_comparison(
    target,
    predicted,
    sample_rate,
):

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float32,
    )

    length = min(
        len(target),
        len(predicted),
    )

    target = target[:length]
    predicted = predicted[:length]

    # ------------------------------------------------------------
    # Target waveform
    # ------------------------------------------------------------

    save_waveform_plot(
        target,
        sample_rate,
        "target_waveform.png",
        "Target Waveform",
    )

    # ------------------------------------------------------------
    # Predicted waveform
    # ------------------------------------------------------------

    save_waveform_plot(
        predicted,
        sample_rate,
        "reconstructed_waveform.png",
        "Reconstructed Waveform",
    )

    # ------------------------------------------------------------
    # Spectrograms
    # ------------------------------------------------------------

    save_spectrogram_plot(
        target,
        sample_rate,
        "target_spectrogram.png",
        "Target Spectrogram",
    )

    save_spectrogram_plot(
        predicted,
        sample_rate,
        "reconstructed_spectrogram.png",
        "Reconstructed Spectrogram",
    )

    # ------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------

    write_wav(
        OUTPUT_DIR / "target_audio.wav",
        target,
        sample_rate,
    )

    write_wav(
        OUTPUT_DIR / "reconstructed_audio.wav",
        predicted,
        sample_rate,
    )

    # ------------------------------------------------------------
    # Side-by-side waveform
    # ------------------------------------------------------------

    time = np.arange(
        length
    ) / float(sample_rate)

    plt.figure(
        figsize=(12, 7)
    )

    plt.subplot(2, 1, 1)

    plt.plot(
        time,
        target,
    )

    plt.title(
        "Original Target Waveform"
    )

    plt.ylabel("Amplitude")

    plt.grid(
        alpha=0.3
    )

    plt.subplot(2, 1, 2)

    plt.plot(
        time,
        predicted,
    )

    plt.title(
        "Neural Vocoder Reconstruction"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "waveform_comparison.png",
        dpi=150,
    )

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():

    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("PHASE 2 — LESSON 61")
    print("NEURAL VOCODER FROM SCRATCH")
    print("=" * 72)

    print()

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ========================================================================
    # STEP 1
    # ========================================================================

    records = load_records()

    # ========================================================================
    # SHOW DATASET SUMMARY
    # ========================================================================

    print()
    print("=" * 72)
    print("DATASET SUMMARY")
    print("=" * 72)

    for i, record in enumerate(records):

        print()
        print(f"Sample {i}")

        print(
            f"  Mel shape      : {record['mel'].shape}"
        )

        print(
            f"  Waveform shape : {record['waveform'].shape}"
        )

        print(
            f"  Sample rate    : {record['sample_rate']}"
        )

        print(
            f"  Hop length     : {record['hop_length']}"
        )

        print(
            f"  Mel file       : {record['mel_path']}"
        )

        print(
            f"  WAV file       : {record['wav_path']}"
        )

    # ========================================================================
    # STEP 2
    # ========================================================================

    dataset = VocoderDataset(
        records
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    # ========================================================================
    # MODEL
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 2 — BUILD NEURAL VOCODER")
    print("=" * 72)

    model = NeuralVocoder(
        n_mels=records[0]["mel"].shape[0]
    )

    model = model.to(
        DEVICE
    )

    print(
        f"Model parameters: "
        f"{count_parameters(model):,}"
    )

    print()
    print(
        "Architecture:"
    )

    print(
        "  Mel"
    )
    print(
        "   ↓"
    )
    print(
        "  Conv1D projection"
    )
    print(
        "   ↓"
    )
    print(
        "  Upsample ×4"
    )
    print(
        "   ↓"
    )
    print(
        "  Upsample ×4"
    )
    print(
        "   ↓"
    )
    print(
        "  Upsample ×4"
    )
    print(
        "   ↓"
    )
    print(
        "  Upsample ×4"
    )
    print(
        "   ↓"
    )
    print(
        "  Residual blocks"
    )
    print(
        "   ↓"
    )
    print(
        "  Waveform"
    )

    # ========================================================================
    # STEP 3
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 3 — TRAINING")
    print("=" * 72)

    model, history, best_loss = train_model(
        model,
        loader,
    )

    save_training_plot(
        history
    )

    # ========================================================================
    # STEP 4
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 4 — RECONSTRUCT WAVEFORM")
    print("=" * 72)

    record = records[0]

    predicted = reconstruct(
        model,
        record,
    )

    target = record["waveform"]

    sample_rate = record["sample_rate"]

    save_comparison(
        target,
        predicted,
        sample_rate,
    )

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print("=" * 72)
    print("LESSON 61 COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Best training loss: {best_loss:.6f}"
    )

    print()
    print(
        f"Checkpoint:"
    )

    print(
        f"  {OUTPUT_DIR / 'neural_vocoder_best.pt'}"
    )

    print()
    print(
        "Generated files:"
    )

    generated_files = [
        "neural_vocoder_best.pt",
        "vocoder_training.png",
        "target_waveform.png",
        "reconstructed_waveform.png",
        "waveform_comparison.png",
        "target_spectrogram.png",
        "reconstructed_spectrogram.png",
        "target_audio.wav",
        "reconstructed_audio.wav",
    ]

    for filename in generated_files:

        path = OUTPUT_DIR / filename

        if path.exists():
            print(
                f"  {path}"
            )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a training-set reconstruction demonstration."
    )

    print(
        "With only one or very few utterances, the result does NOT "
        "demonstrate generalization."
    )

    print()
    print(
        "The key lesson is:"
    )

    print(
        "    Mel-spectrogram → Neural Vocoder → Waveform"
    )

    print()
    print(
        "The next stage can build on this vocoder for the complete "
        "text-to-speech pipeline."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()