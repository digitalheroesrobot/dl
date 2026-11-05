"""
========================================================================
PHASE 2 — LESSON 42
REBUILD AND EVALUATE NEURAL SPEECH TOKEN REPRESENTATIONS
========================================================================

PURPOSE
-------
Reuse the repaired Lesson 41 VQ-VAE to generate a clean,
consistent neural speech-token dataset.

IMPORTANT:
This lesson does NOT train a new VQ-VAE.
It loads the trained Lesson 41 checkpoint.

LEARNING OBJECTIVES
-------------------

STEP 1 — Load the repaired VQ-VAE
    Restore the encoder and repaired EMA codebook.

STEP 2 — Restore normalization
    Use exactly the same mel normalization statistics
    that were calculated during Lesson 41.

STEP 3 — Extract speech features
    Convert WAV audio into log-mel spectrograms.

STEP 4 — Encode speech
    Convert continuous encoder representations into
    discrete VQ codebook indices.

STEP 5 — Rebuild datasets
    Export train and validation speech-token sequences.

STEP 6 — Evaluate codebook utilization
    Measure:
        - Unique codes
        - Active codes
        - Dead codes
        - Codebook perplexity
        - Token frequency distribution
        - Dominant code percentage

STEP 7 — Visualize results
    Generate:
        - Codebook distribution
        - Token timeline
        - Recording token lengths

STEP 8 — Save dataset
    Save tokens, metadata, statistics and plots.

ARCHITECTURE
------------

WAV AUDIO
    |
    v
16 kHz Audio
    |
    v
STFT
    |
    v
80-bin Log Mel Spectrogram
    |
    v
Lesson 41 Mel Normalization
    |
    v
Trained Convolutional Encoder
    |
    v
64-dimensional Latent Vectors
    |
    v
Repaired EMA VQ Codebook
    |
    v
256 Possible Discrete Tokens
    |
    v
TRAIN / VALIDATION TOKEN DATASET

IMPORTANT TECHNICAL DETAILS
---------------------------

Sample rate:       16,000 Hz
STFT hop:          320 samples
Mel frame period:  20 ms
Encoder stride:    4
Token interval:    80 ms
Codebook size:     256
Latent dimension:  64

The model reconstructs mel spectrograms.
It is NOT yet a production waveform neural codec.

INPUT
-----

dataset/*.wav

lesson41_output/
    best_model.pt
    mel_normalization.pt

OUTPUT
------

lesson42_output/
    train_tokens/
    val_tokens/
    token_metadata.json
    token_statistics.json
    codebook_distribution.png
    token_timeline.png
    recording_token_lengths.png

INSTALL
-------

pip install torch numpy matplotlib soundfile scipy

RUN
---

python lesson42_rebuild_speech_tokens.py

Custom paths:

python lesson42_rebuild_speech_tokens.py \
    --audio-dir dataset \
    --checkpoint lesson41_output/best_model.pt \
    --normalization lesson41_output/mel_normalization.pt \
    --output-dir lesson42_output

========================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import random

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import resample_poly


# ======================================================================
# CONFIGURATION
# ======================================================================

SAMPLE_RATE = 16000

N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 320

N_MELS = 80

LATENT_DIM = 64
CODEBOOK_SIZE = 256

ENCODER_STRIDE = 4

TOKEN_INTERVAL_MS = (
    HOP_LENGTH / SAMPLE_RATE
) * ENCODER_STRIDE * 1000

SEED = 42


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# MEL FILTERBANK
# Must match Lesson 41 exactly.
# ======================================================================

def hz_to_mel(hz):

    return 2595.0 * np.log10(
        1.0 + hz / 700.0
    )


def mel_to_hz(mel):

    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def create_mel_filterbank(
    sample_rate,
    n_fft,
    n_mels,
):

    minimum_mel = hz_to_mel(0)

    maximum_mel = hz_to_mel(
        sample_rate / 2
    )

    mel_points = np.linspace(
        minimum_mel,
        maximum_mel,
        n_mels + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    fft_frequencies = np.linspace(
        0,
        sample_rate / 2,
        n_fft // 2 + 1,
    )

    filterbank = np.zeros(
        (n_mels, len(fft_frequencies)),
        dtype=np.float32,
    )

    for i in range(n_mels):

        left = hz_points[i]

        center = hz_points[i + 1]

        right = hz_points[i + 2]

        ascending = (
            (fft_frequencies - left)
            / max(center - left, 1e-8)
        )

        descending = (
            (right - fft_frequencies)
            / max(right - center, 1e-8)
        )

        filterbank[i] = np.maximum(
            0.0,
            np.minimum(
                ascending,
                descending,
            ),
        )

    return torch.from_numpy(
        filterbank
    )


# ======================================================================
# AUDIO LOADING
# ======================================================================

def load_audio(path):

    audio, source_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    # Convert stereo/multichannel audio to mono.
    audio = audio.mean(axis=1)

    if source_rate != SAMPLE_RATE:

        divisor = math.gcd(
            int(source_rate),
            SAMPLE_RATE,
        )

        up = SAMPLE_RATE // divisor

        down = int(source_rate) // divisor

        audio = resample_poly(
            audio,
            up,
            down,
        ).astype(np.float32)

    waveform = torch.from_numpy(
        np.asarray(
            audio,
            dtype=np.float32,
        )
    )

    waveform = torch.nan_to_num(
        waveform
    )

    peak = waveform.abs().max()

    if peak > 1e-8:

        waveform = waveform / peak

    return waveform


# ======================================================================
# MEL FEATURE EXTRACTION
# Must match Lesson 41.
# ======================================================================

class MelExtractor:

    def __init__(self, device):

        self.device = device

        self.window = torch.hann_window(
            WIN_LENGTH,
            device=device,
        )

        self.filterbank = create_mel_filterbank(
            SAMPLE_RATE,
            N_FFT,
            N_MELS,
        ).to(device)

    @torch.no_grad()
    def __call__(self, waveform):

        waveform = waveform.to(
            self.device
        )

        spectrum = torch.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            return_complex=True,
        )

        power = spectrum.abs().pow(2)

        mel = torch.matmul(
            self.filterbank,
            power,
        )

        log_mel = torch.log(
            mel.clamp_min(1e-6)
        )

        return log_mel


# ======================================================================
# LESSON 41 MODEL ARCHITECTURE
#
# Keep this architecture identical to Lesson 41.
# ======================================================================

class EMAVectorQuantizer(nn.Module):

    def __init__(self):

        super().__init__()

        self.codebook_size = CODEBOOK_SIZE

        self.embedding_dim = LATENT_DIM

        self.register_buffer(
            "embedding",
            torch.randn(
                CODEBOOK_SIZE,
                LATENT_DIM,
            ),
        )

        self.register_buffer(
            "ema_count",
            torch.ones(CODEBOOK_SIZE),
        )

        self.register_buffer(
            "ema_sum",
            torch.randn(
                CODEBOOK_SIZE,
                LATENT_DIM,
            ),
        )

        self.register_buffer(
            "dead_steps",
            torch.zeros(CODEBOOK_SIZE),
        )

        self.register_buffer(
            "initialized",
            torch.tensor(False),
        )


class SpeechMelVQVAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv1d(
                N_MELS,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv1d(
                128,
                LATENT_DIM,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

        self.quantizer = EMAVectorQuantizer()

        self.decoder = nn.Sequential(

            nn.ConvTranspose1d(
                LATENT_DIM,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.GELU(),

            nn.ConvTranspose1d(
                128,
                N_MELS,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    @torch.no_grad()
    def encode(self, mel):

        z = self.encoder(mel)

        flat_z = z.transpose(
            1,
            2,
        )

        distances = (
            flat_z.pow(2).sum(
                dim=-1,
                keepdim=True,
            )
            - 2 * flat_z @ self.quantizer.embedding.T
            + self.quantizer.embedding.pow(2).sum(
                dim=1
            )
        )

        indices = distances.argmin(
            dim=-1
        )

        return indices


# ======================================================================
# DATASET DISCOVERY
# ======================================================================

def find_audio_files(audio_dir):

    if not audio_dir.exists():

        raise FileNotFoundError(
            f"Audio directory not found: {audio_dir}"
        )

    extensions = {
        ".wav",
        ".flac",
    }

    files = sorted(
        path
        for path in audio_dir.iterdir()
        if path.suffix.lower() in extensions
    )

    if len(files) < 2:

        raise RuntimeError(
            "At least two WAV/FLAC files are required."
        )

    return files


def split_files(files, val_ratio=0.2):

    files = list(files)

    # Reproduce Lesson 41's seeded shuffle.
    random.shuffle(files)

    val_count = max(
        1,
        int(round(len(files) * val_ratio)),
    )

    val_count = min(
        val_count,
        len(files) - 1,
    )

    return (
        files[val_count:],
        files[:val_count],
    )


# ======================================================================
# LOAD LESSON 41 CHECKPOINT
# ======================================================================

def load_lesson41_model(
    checkpoint_path,
    normalization_path,
    device,
):

    print("\nLoading Lesson 41 checkpoint...")

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not normalization_path.exists():

        raise FileNotFoundError(
            f"Normalization file not found: "
            f"{normalization_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = SpeechMelVQVAE().to(
        device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    # Verify the repaired codebook was initialized.
    if not bool(
        model.quantizer.initialized.item()
    ):

        raise RuntimeError(
            "Checkpoint contains an uninitialized "
            "VQ codebook."
        )

    normalization = torch.load(
        normalization_path,
        map_location="cpu",
        weights_only=False,
    )

    mean = normalization["mean"].float()

    std = normalization["std"].float()

    if mean.shape != (N_MELS,):

        raise ValueError(
            f"Invalid mean shape: {mean.shape}"
        )

    if std.shape != (N_MELS,):

        raise ValueError(
            f"Invalid std shape: {std.shape}"
        )

    print("Checkpoint loaded successfully.")

    print(
        f"Best training epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint.get('val_loss', 'unknown')}"
    )

    print(
        f"Codebook initialized: "
        f"{bool(model.quantizer.initialized.item())}"
    )

    print(
        f"Codebook shape: "
        f"{tuple(model.quantizer.embedding.shape)}"
    )

    return model, mean, std


# ======================================================================
# SPEECH TOKEN ENCODING
# ======================================================================

@torch.no_grad()
def encode_recording(
    model,
    extractor,
    waveform,
    mean,
    std,
    device,
):

    # --------------------------------------------------------------
    # STEP 1 — Extract log-mel spectrogram.
    # --------------------------------------------------------------

    mel = extractor(
        waveform
    )

    original_frames = mel.shape[1]

    # --------------------------------------------------------------
    # STEP 2 — Normalize using Lesson 41 statistics.
    # --------------------------------------------------------------

    mean_device = mean.to(device)

    std_device = std.to(device)

    normalized = (
        mel - mean_device.unsqueeze(1)
    ) / std_device.unsqueeze(1)

    # --------------------------------------------------------------
    # STEP 3 — Pad to a multiple of encoder stride.
    # --------------------------------------------------------------

    padding = (
        ENCODER_STRIDE
        - original_frames % ENCODER_STRIDE
    ) % ENCODER_STRIDE

    normalized = F.pad(
        normalized,
        (0, padding),
    )

    # --------------------------------------------------------------
    # STEP 4 — Encode with repaired VQ-VAE.
    # --------------------------------------------------------------

    normalized = normalized.unsqueeze(
        0
    )

    tokens = model.encode(
        normalized
    )

    tokens = tokens.squeeze(0)

    tokens = tokens.cpu().numpy().astype(
        np.int64
    )

    return tokens, original_frames


# ======================================================================
# TOKEN STATISTICS
# ======================================================================

def calculate_statistics(
    token_arrays,
    codebook_size,
):

    if not token_arrays:

        raise RuntimeError(
            "No token sequences available."
        )

    all_tokens = np.concatenate(
        token_arrays
    )

    counts = np.bincount(
        all_tokens,
        minlength=codebook_size,
    )

    total_tokens = int(
        counts.sum()
    )

    probabilities = (
        counts / max(total_tokens, 1)
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities + 1e-12
        )
    )

    perplexity = float(
        np.exp(entropy)
    )

    active_codes = int(
        np.count_nonzero(counts)
    )

    dead_codes = (
        codebook_size - active_codes
    )

    dominant_code = int(
        np.argmax(counts)
    )

    dominant_percentage = float(
        counts[dominant_code]
        / max(total_tokens, 1)
        * 100
    )

    return {

        "total_tokens": total_tokens,

        "unique_codes": active_codes,

        "active_codes": active_codes,

        "dead_codes": dead_codes,

        "codebook_size": codebook_size,

        "perplexity": perplexity,

        "dominant_code": dominant_code,

        "dominant_code_percentage": (
            dominant_percentage
        ),

        "code_counts": counts.tolist(),

        "code_probabilities": (
            probabilities.tolist()
        ),

    }


# ======================================================================
# EXPORT TOKEN DATASET
# ======================================================================

@torch.no_grad()
def export_split(
    model,
    extractor,
    files,
    mean,
    std,
    device,
    output_dir,
    split_name,
):

    split_dir = (
        output_dir / f"{split_name}_tokens"
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    token_arrays = []

    print(
        f"\nExporting {split_name} recordings..."
    )

    for path in files:

        print(
            f"\nProcessing: {path.name}"
        )

        waveform = load_audio(path)

        tokens, mel_frames = encode_recording(
            model,
            extractor,
            waveform,
            mean,
            std,
            device,
        )

        output_path = (
            split_dir / f"{path.stem}.npy"
        )

        np.save(
            output_path,
            tokens,
        )

        duration = (
            len(waveform) / SAMPLE_RATE
        )

        record = {

            "filename": path.name,

            "token_file": str(
                output_path.relative_to(output_dir)
            ),

            "duration_seconds": duration,

            "mel_frames": int(
                mel_frames
            ),

            "token_count": int(
                len(tokens)
            ),

            "unique_codes": int(
                len(np.unique(tokens))
            ),

        }

        records.append(record)

        token_arrays.append(tokens)

        print(
            f"  Duration: {duration:.2f}s"
        )

        print(
            f"  Mel frames: {mel_frames}"
        )

        print(
            f"  Tokens: {len(tokens)}"
        )

        print(
            f"  Unique codes: "
            f"{record['unique_codes']}"
        )

    statistics = calculate_statistics(
        token_arrays,
        CODEBOOK_SIZE,
    )

    return records, token_arrays, statistics


# ======================================================================
# VISUALIZATION 1 — CODEBOOK DISTRIBUTION
# ======================================================================

def plot_codebook_distribution(
    train_stats,
    val_stats,
    output_dir,
):

    train_counts = np.array(
        train_stats["code_counts"]
    )

    val_counts = np.array(
        val_stats["code_counts"]
    )

    indices = np.arange(
        CODEBOOK_SIZE
    )

    plt.figure(figsize=(14, 6))

    plt.bar(
        indices,
        train_counts,
        alpha=0.65,
        label="Training",
    )

    plt.bar(
        indices,
        val_counts,
        alpha=0.55,
        label="Validation",
    )

    plt.xlabel("Codebook Index")

    plt.ylabel("Token Frequency")

    plt.title(
        "Lesson 42 — VQ Codebook Distribution"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "codebook_distribution.png",
        dpi=150,
    )

    plt.close()


# ======================================================================
# VISUALIZATION 2 — TOKEN TIMELINE
# ======================================================================

def plot_token_timeline(
    token_arrays,
    records,
    output_dir,
):

    if not token_arrays:

        return

    count = min(
        len(token_arrays),
        4,
    )

    fig, axes = plt.subplots(
        count,
        1,
        figsize=(14, 3 * count),
    )

    if count == 1:

        axes = [axes]

    for i in range(count):

        tokens = token_arrays[i]

        time_axis = (
            np.arange(len(tokens))
            * TOKEN_INTERVAL_MS
            / 1000.0
        )

        axes[i].scatter(
            time_axis,
            tokens,
            s=8,
        )

        axes[i].set_title(
            records[i]["filename"]
        )

        axes[i].set_xlabel(
            "Time (seconds)"
        )

        axes[i].set_ylabel(
            "Token ID"
        )

        axes[i].set_ylim(
            -2,
            CODEBOOK_SIZE + 2,
        )

        axes[i].grid(
            alpha=0.3
        )

    plt.suptitle(
        "Neural Speech Token Timeline"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "token_timeline.png",
        dpi=150,
    )

    plt.close()


# ======================================================================
# VISUALIZATION 3 — RECORDING TOKEN LENGTHS
# ======================================================================

def plot_recording_lengths(
    train_records,
    val_records,
    output_dir,
):

    names = (
        [
            r["filename"]
            for r in train_records
        ]
        +
        [
            r["filename"]
            for r in val_records
        ]
    )

    lengths = (
        [
            r["token_count"]
            for r in train_records
        ]
        +
        [
            r["token_count"]
            for r in val_records
        ]
    )

    splits = (
        ["Train"] * len(train_records)
        +
        ["Validation"] * len(val_records)
    )

    x = np.arange(
        len(names)
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        x,
        lengths,
        label="Token count",
    )

    plt.xticks(
        x,
        names,
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        "Number of Tokens"
    )

    plt.title(
        "Speech Token Length per Recording"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "recording_token_lengths.png",
        dpi=150,
    )

    plt.close()


# ======================================================================
# SAVE REPORT
# ======================================================================

def save_report(
    output_dir,
    train_records,
    val_records,
    train_stats,
    val_stats,
):

    report = {

        "lesson": 42,

        "description": (
            "Rebuilt neural speech-token dataset "
            "using the repaired Lesson 41 VQ-VAE."
        ),

        "configuration": {

            "sample_rate": SAMPLE_RATE,

            "n_fft": N_FFT,

            "hop_length": HOP_LENGTH,

            "n_mels": N_MELS,

            "latent_dim": LATENT_DIM,

            "codebook_size": CODEBOOK_SIZE,

            "encoder_stride": ENCODER_STRIDE,

            "mel_frame_interval_ms": 20,

            "token_interval_ms": (
                TOKEN_INTERVAL_MS
            ),

        },

        "training": {

            "recordings": train_records,

            "statistics": train_stats,

        },

        "validation": {

            "recordings": val_records,

            "statistics": val_stats,

        },

    }

    with open(
        output_dir / "token_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    with open(
        output_dir / "token_statistics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "train": train_stats,
                "validation": val_stats,
            },
            f,
            indent=2,
        )


# ======================================================================
# MAIN
# ======================================================================

def run(args):

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    audio_dir = Path(
        args.audio_dir
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    normalization_path = Path(
        args.normalization
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "LESSON 42 — REBUILD NEURAL SPEECH TOKENS"
    )

    print("=" * 72)

    print(f"\nDevice: {device}")

    if device.type == "cuda":

        print(
            torch.cuda.get_device_name(0)
        )

    print(
        f"Sample rate: {SAMPLE_RATE}"
    )

    print(
        f"Mel bins: {N_MELS}"
    )

    print(
        f"Codebook size: {CODEBOOK_SIZE}"
    )

    print(
        f"Latent dimension: {LATENT_DIM}"
    )

    print(
        f"Token interval: {TOKEN_INTERVAL_MS:.1f} ms"
    )

    # --------------------------------------------------------------
    # STEP 1 — LOAD CHECKPOINT
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print("STEP 1 — LOAD REPAIRED VQ-VAE")

    print("=" * 72)

    model, mean, std = load_lesson41_model(
        checkpoint_path,
        normalization_path,
        device,
    )

    # --------------------------------------------------------------
    # STEP 2 — FIND AUDIO
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print("STEP 2 — LOAD RECORDINGS")

    print("=" * 72)

    files = find_audio_files(
        audio_dir
    )

    train_files, val_files = split_files(
        files,
        args.val_ratio,
    )

    print(
        f"Total recordings: {len(files)}"
    )

    print(
        f"Training recordings: {len(train_files)}"
    )

    print(
        f"Validation recordings: {len(val_files)}"
    )

    # --------------------------------------------------------------
    # STEP 3 — MEL EXTRACTOR
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print("STEP 3 — INITIALIZE MEL EXTRACTOR")

    print("=" * 72)

    extractor = MelExtractor(
        device
    )

    # --------------------------------------------------------------
    # STEP 4 — REBUILD TRAIN TOKENS
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print("STEP 4 — REBUILD TRAINING TOKENS")

    print("=" * 72)

    train_records, train_arrays, train_stats = export_split(
        model,
        extractor,
        train_files,
        mean,
        std,
        device,
        output_dir,
        "train",
    )

    # --------------------------------------------------------------
    # STEP 5 — REBUILD VALIDATION TOKENS
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print("STEP 5 — REBUILD VALIDATION TOKENS")

    print("=" * 72)

    val_records, val_arrays, val_stats = export_split(
        model,
        extractor,
        val_files,
        mean,
        std,
        device,
        output_dir,
        "val",
    )

    # --------------------------------------------------------------
    # STEP 6 — STATISTICAL EVALUATION
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print("STEP 6 — EVALUATE CODEBOOK UTILIZATION")

    print("=" * 72)

    for name, stats in [
        ("TRAIN", train_stats),
        ("VALIDATION", val_stats),
    ]:

        print(f"\n{name}")

        print(
            f"  Total tokens: "
            f"{stats['total_tokens']}"
        )

        print(
            f"  Active codes: "
            f"{stats['active_codes']}/{CODEBOOK_SIZE}"
        )

        print(
            f"  Dead codes: "
            f"{stats['dead_codes']}"
        )

        print(
            f"  Perplexity: "
            f"{stats['perplexity']:.2f}"
        )

        print(
            f"  Dominant code: "
            f"{stats['dominant_code']}"
        )

        print(
            f"  Dominant code frequency: "
            f"{stats['dominant_code_percentage']:.2f}%"
        )

    # --------------------------------------------------------------
    # STEP 7 — VISUALIZATION
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print("STEP 7 — GENERATE VISUALIZATIONS")

    print("=" * 72)

    plot_codebook_distribution(
        train_stats,
        val_stats,
        output_dir,
    )

    plot_token_timeline(
        train_arrays,
        train_records,
        output_dir,
    )

    plot_recording_lengths(
        train_records,
        val_records,
        output_dir,
    )

    print(
        "Visualizations saved."
    )

    # --------------------------------------------------------------
    # STEP 8 — SAVE METADATA
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print("STEP 8 — SAVE DATASET METADATA")

    print("=" * 72)

    save_report(
        output_dir,
        train_records,
        val_records,
        train_stats,
        val_stats,
    )

    # --------------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print("LESSON 42 COMPLETE")

    print("=" * 72)

    print(
        f"\nTotal train tokens: "
        f"{train_stats['total_tokens']}"
    )

    print(
        f"Total validation tokens: "
        f"{val_stats['total_tokens']}"
    )

    print(
        f"Training active codes: "
        f"{train_stats['active_codes']}/{CODEBOOK_SIZE}"
    )

    print(
        f"Validation active codes: "
        f"{val_stats['active_codes']}/{CODEBOOK_SIZE}"
    )

    print(
        f"Training perplexity: "
        f"{train_stats['perplexity']:.2f}"
    )

    print(
        f"Validation perplexity: "
        f"{val_stats['perplexity']:.2f}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        output_dir.resolve()
    )

    print("\nGenerated files:")

    print("  train_tokens/")

    print("  val_tokens/")

    print("  token_metadata.json")

    print("  token_statistics.json")

    print("  codebook_distribution.png")

    print("  token_timeline.png")

    print("  recording_token_lengths.png")

    print("\nNEXT:")

    print(
        "Lesson 43 — Retrain Autoregressive "
        "Speech Token Language Model"
    )

    print("=" * 72)


# ======================================================================
# COMMAND-LINE ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 42: Rebuild neural speech tokens"
        )
    )

    parser.add_argument(
        "--audio-dir",
        type=str,
        default="dataset",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="lesson41_output/best_model.pt",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        default="lesson41_output/mel_normalization.pt",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="lesson42_output",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run(args)