"""
========================================================================
LESSON 39 — REAL SPEECH TOKEN DATASET PIPELINE
========================================================================

COURSE
------
Voice Cloning / TTS From Scratch

PREVIOUS LESSONS
----------------
Lesson 36:
    Neural Speech Codec

Lesson 37:
    Autoregressive Speech Token Model

Lesson 38:
    Improved Speech Token Language Model

THIS LESSON
-----------
We now move from synthetic token sequences to REAL AUDIO.

The goal is to construct the dataset that a speech-token language model
actually needs.
ffmpeg needed
PIPELINE
--------

    WAV / FLAC / MP3
          |
          v
    waveform loading
          |
          v
    mono conversion
          |
          v
    resampling
          |
          v
    amplitude normalization
          |
          v
    speech tokenizer / codec
          |
          v
    discrete token IDs
          |
          v
    fixed-length sequences
          |
          v
    PyTorch Dataset


WHY THIS MATTERS
----------------

The Transformer from Lessons 37 and 38 expects:

    token IDs

Real speech begins as:

    waveform samples

Therefore we need:

    waveform -> discrete representation


TOKEN SEQUENCE
--------------

Suppose the codec produces:

    [14, 21, 21, 35, 91, 91, 42, ...]

The autoregressive model can learn:

    P(x_t | x_<t)


AUDIO PREPROCESSING
-------------------

Different recordings may have:

    different sample rates
    different channels
    different amplitudes
    different durations

We standardize them.

Example:

    44.1 kHz stereo
           |
           v
    16 kHz mono


DATASET DESIGN
--------------

A speech dataset should NOT simply concatenate all recordings together.

We preserve recording boundaries.

For example:

    speaker_01_001.wav
        -> tokens A

    speaker_01_002.wav
        -> tokens B

    speaker_02_001.wav
        -> tokens C


Then each recording is independently converted into training chunks.


DATA LEAKAGE
------------

This is especially important for speech.

If the same recording appears in both:

    training
    validation

the validation score becomes misleading.

Therefore the split happens at the AUDIO FILE level first.

Then token sequences are generated.


DEMONSTRATION TOKENIZER
-----------------------

If --codec is not supplied, this script uses a deterministic
demonstration tokenizer.

It is NOT a production speech codec.

It converts log-mel spectral information into discrete IDs.

This allows us to test:

    audio
      |
      v
    features
      |
      v
    tokens
      |
      v
    dataset


REAL CODEC MODE
---------------

A production system should instead use the codec/tokenizer learned
in previous lessons.

The interface expected by this script is conceptually:

    tokenizer.encode(waveform)

returning:

    integer token IDs


IMPORTANT LESSON 36 ISSUE
-------------------------

If the Lesson 36 VQ model produced:

    Codes 1/256
    Perplexity 1.00

then the codebook has collapsed.

DO NOT use that checkpoint as a meaningful speech tokenizer.

Fix the codec first, or use the demonstration tokenizer while
continuing to develop the data pipeline.


TOKEN RATE
----------

If audio duration is:

    D seconds

and the codec produces:

    T tokens

then:

    token_rate = T / D


This is an important number for TTS.

For example:

    50 tokens / second

means:

    5 seconds -> approximately 250 tokens


DATASET SIZE
------------

For N recordings:

    total_tokens
        =
    sum(tokens_per_recording)


Then fixed-length chunks are created.

Example:

    context length = 256

    token sequence:
        1200 tokens

becomes:

    chunk 1: tokens 0:256
    chunk 2: tokens 256:512
    chunk 3: tokens 512:768
    chunk 4: tokens 768:1024

The final incomplete chunk can be discarded.


WHAT YOU LEARN
--------------

STEP 1
    Find real audio files.

STEP 2
    Load audio robustly.

STEP 3
    Standardize sample rate.

STEP 4
    Convert to mono.

STEP 5
    Normalize amplitude.

STEP 6
    Remove unusable recordings.

STEP 7
    Convert waveform to discrete tokens.

STEP 8
    Save token sequences.

STEP 9
    Build fixed-length chunks.

STEP 10
    Split without leakage.

STEP 11
    Analyze token statistics.

STEP 12
    Prepare the dataset for Lesson 40.


DEPENDENCIES
------------

    pip install torch torchaudio numpy matplotlib soundfile


RUN
---

Create a directory:

    dataset/

Put audio files inside:

    dataset/
        speaker1_001.wav
        speaker1_002.wav
        speaker2_001.wav
        ...

Then:

    python lesson39_real_speech_token_dataset.py \
        --audio-dir dataset


OUTPUT
------

The script creates:

    lesson39_tokens/

        recording_0000.npy
        recording_0001.npy
        ...

        train.npy
        val.npy

        metadata.json


========================================================================
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio


# ======================================================================
# 1. REPRODUCIBILITY
# ======================================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ======================================================================
# 2. CONFIGURATION
# ======================================================================

TARGET_SAMPLE_RATE = 16000

MIN_DURATION_SEC = 0.5

MAX_DURATION_SEC = 30.0

VOCAB_SIZE = 256

TOKEN_HOP_MS = 20

TOKEN_HOP_SAMPLES = int(
    TARGET_SAMPLE_RATE
    * TOKEN_HOP_MS
    / 1000
)

MEL_BINS = 80

FFT_SIZE = 512

WINDOW_SIZE = 400

DEFAULT_CONTEXT_LENGTH = 256

DEFAULT_VAL_RATIO = 0.10

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a"
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# 3. ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 39 - Real Speech Token Dataset Pipeline"
        )
    )

    parser.add_argument(
        "--audio-dir",
        type=str,
        default="dataset",
        help="Directory containing audio files."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="lesson39_tokens",
        help="Output token dataset directory."
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for debugging."
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show waveform/mel/token visualizations."
    )

    return parser.parse_args()


# ======================================================================
# 4. AUDIO FILE DISCOVERY
# ======================================================================

def find_audio_files(
    audio_dir
):

    root = Path(
        audio_dir
    )

    if not root.exists():

        raise FileNotFoundError(
            f"Audio directory does not exist: {root}"
        )

    files = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in (
            SUPPORTED_AUDIO_EXTENSIONS
        ):

            files.append(
                path
            )

    files.sort()

    return files


# ======================================================================
# 5. LOAD AUDIO
# ======================================================================

def load_audio(
    path
):

    """
    Load audio and standardize it to:

        [1, samples]

    at TARGET_SAMPLE_RATE.
    """

    waveform, sample_rate = (
        torchaudio.load(
            str(path)
        )
    )

    # --------------------------------------------------------------
    # Convert stereo/multichannel -> mono
    # --------------------------------------------------------------

    if waveform.ndim != 2:

        raise ValueError(
            f"Unexpected waveform shape: "
            f"{waveform.shape}"
        )

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    # --------------------------------------------------------------
    # Resample
    # --------------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            TARGET_SAMPLE_RATE
        )

    # --------------------------------------------------------------
    # Remove NaN / Inf
    # --------------------------------------------------------------

    waveform = torch.nan_to_num(
        waveform
    )

    # --------------------------------------------------------------
    # Peak normalization
    # --------------------------------------------------------------

    peak = waveform.abs().max()

    if peak > 0:

        waveform = (
            waveform
            / peak
        )

    return waveform


# ======================================================================
# 6. AUDIO VALIDATION
# ======================================================================

def validate_audio(
    waveform
):

    samples = waveform.shape[-1]

    duration = (
        samples
        / TARGET_SAMPLE_RATE
    )

    if duration < MIN_DURATION_SEC:

        return False, (
            f"too_short:{duration:.3f}s"
        )

    if duration > MAX_DURATION_SEC:

        return False, (
            f"too_long:{duration:.3f}s"
        )

    if not torch.isfinite(
        waveform
    ).all():

        return False, "non_finite"

    if waveform.abs().max() < 1e-5:

        return False, "silent"

    rms = torch.sqrt(
        torch.mean(
            waveform ** 2
        )
    )

    if rms < 1e-5:

        return False, "very_low_rms"

    return True, "ok"


# ======================================================================
# 7. LOG-MEL FEATURE EXTRACTOR
# ======================================================================

class LogMelExtractor:

    """
    Convert waveform into log-mel spectral features.

    This is NOT the final neural codec.

    It is a deterministic feature representation used by the
    demonstration tokenizer.
    """

    def __init__(self):

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=FFT_SIZE,
            win_length=WINDOW_SIZE,
            hop_length=TOKEN_HOP_SAMPLES,
            n_mels=MEL_BINS
        )

    def __call__(
        self,
        waveform
    ):

        mel = self.mel(
            waveform
        )

        mel = torch.log(
            mel.clamp_min(1e-5)
        )

        # Normalize each recording.
        mean = mel.mean()

        std = mel.std().clamp_min(
            1e-5
        )

        mel = (
            mel - mean
        ) / std

        return mel


# ======================================================================
# 8. DEMONSTRATION SPEECH TOKENIZER
# ======================================================================

class DemonstrationSpeechTokenizer:

    """
    Deterministic feature-to-token tokenizer.

    IMPORTANT:
        This is NOT a learned neural speech codec.

    It exists so that Lesson 39 can run independently and demonstrate
    the complete real-audio -> discrete-token dataset pipeline.

    Each mel frame becomes one discrete token.

    We convert the mean spectral energy into a 256-level representation.

    This produces meaningful temporal token variation, but it should
    NOT be confused with a production codec such as a neural VQ codec.
    """

    def __init__(
        self,
        vocab_size=VOCAB_SIZE
    ):

        self.vocab_size = vocab_size

        self.mel_extractor = (
            LogMelExtractor()
        )

    @torch.no_grad()
    def encode(
        self,
        waveform
    ):

        mel = self.mel_extractor(
            waveform
        )

        # mel shape:
        #
        # [1, mel_bins, frames]

        # Convert each frame into a scalar
        # spectral descriptor.

        descriptor = mel.mean(
            dim=1
        ).squeeze(0)

        # Squash into [0, 1].

        descriptor = torch.sigmoid(
            descriptor
        )

        tokens = torch.clamp(
            (
                descriptor
                * (self.vocab_size - 1)
            ).round(),
            0,
            self.vocab_size - 1
        )

        return tokens.long()


# ======================================================================
# 9. TOKENIZER INTERFACE
# ======================================================================

class SpeechTokenizerInterface:

    """
    Adapter interface for a real neural codec.

    Your repaired Lesson 36 codec can be connected by implementing:

        encode(waveform)

    and returning:

        torch.LongTensor

    with values in:

        [0, VOCAB_SIZE - 1]
    """

    def encode(
        self,
        waveform
    ):

        raise NotImplementedError(
            "Connect your real Lesson 36 codec here."
        )


# ======================================================================
# 10. TOKEN VALIDATION
# ======================================================================

def validate_tokens(
    tokens,
    vocab_size
):

    if not isinstance(
        tokens,
        torch.Tensor
    ):

        tokens = torch.as_tensor(
            tokens
        )

    tokens = tokens.detach().cpu()

    tokens = tokens.reshape(-1)

    if tokens.numel() == 0:

        raise ValueError(
            "Tokenizer produced zero tokens."
        )

    tokens = tokens.long()

    minimum = int(
        tokens.min().item()
    )

    maximum = int(
        tokens.max().item()
    )

    if minimum < 0:

        raise ValueError(
            f"Negative token ID: {minimum}"
        )

    if maximum >= vocab_size:

        raise ValueError(
            f"Token ID {maximum} exceeds "
            f"vocabulary size {vocab_size}."
        )

    return tokens


# ======================================================================
# 11. SAVE TOKEN SEQUENCE
# ======================================================================

def save_token_sequence(
    path,
    tokens
):

    np.save(
        str(path),
        tokens.numpy().astype(
            np.int64
        )
    )


# ======================================================================
# 12. CREATE FIXED-LENGTH CHUNKS
# ======================================================================

def create_chunks(
    token_sequences,
    context_length
):

    chunks = []

    for tokens in token_sequences:

        length = len(tokens)

        usable = (
            length
            // context_length
        ) * context_length

        if usable == 0:
            continue

        tokens = tokens[
            :usable
        ]

        sequence_chunks = (
            tokens.reshape(
                -1,
                context_length
            )
        )

        chunks.append(
            sequence_chunks
        )

    if not chunks:

        return np.empty(
            (
                0,
                context_length
            ),
            dtype=np.int64
        )

    return np.concatenate(
        chunks,
        axis=0
    )


# ======================================================================
# 13. TRAIN / VALIDATION SPLIT
# ======================================================================

def split_files(
    files,
    val_ratio,
    seed
):

    files = list(
        files
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        files
    )

    if len(files) <= 1:

        return files, []

    val_count = max(
        1,
        int(
            len(files)
            * val_ratio
        )
    )

    val_files = files[
        :val_count
    ]

    train_files = files[
        val_count:
    ]

    return (
        train_files,
        val_files
    )


# ======================================================================
# 14. TOKEN STATISTICS
# ======================================================================

def token_statistics(
    sequences,
    vocab_size
):

    if len(sequences) == 0:

        return None

    flat = np.concatenate(
        sequences
    )

    counts = np.bincount(
        flat,
        minlength=vocab_size
    )

    probabilities = (
        counts
        / counts.sum()
    )

    nonzero = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        nonzero
        * np.log2(
            nonzero
        )
    )

    perplexity = (
        2 ** entropy
    )

    return {
        "total_tokens": int(
            len(flat)
        ),
        "unique_tokens": int(
            np.count_nonzero(
                counts
            )
        ),
        "entropy_bits": float(
            entropy
        ),
        "distribution_perplexity": float(
            perplexity
        ),
        "min_token": int(
            flat.min()
        ),
        "max_token": int(
            flat.max()
        )
    }


# ======================================================================
# 15. PLOT TOKEN DISTRIBUTION
# ======================================================================

def plot_token_distribution(
    sequences,
    vocab_size,
    title
):

    if len(sequences) == 0:

        return

    flat = np.concatenate(
        sequences
    )

    counts = np.bincount(
        flat,
        minlength=vocab_size
    )

    plt.figure(
        figsize=(11, 4)
    )

    plt.bar(
        np.arange(
            vocab_size
        ),
        counts
    )

    plt.xlabel(
        "Token ID"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# 16. AUDIO / MEL / TOKEN VISUALIZATION
# ======================================================================

def visualize_example(
    waveform,
    mel,
    tokens,
    title
):

    waveform = (
        waveform
        .squeeze()
        .cpu()
        .numpy()
    )

    mel = (
        mel
        .squeeze()
        .cpu()
        .numpy()
    )

    tokens = (
        tokens
        .cpu()
        .numpy()
    )

    duration = (
        len(waveform)
        / TARGET_SAMPLE_RATE
    )

    time = np.linspace(
        0,
        duration,
        len(waveform)
    )

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        time,
        waveform
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        f"{title} — waveform"
    )

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        mel,
        aspect="auto",
        origin="lower"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Mel bin"
    )

    plt.title(
        f"{title} — log-mel representation"
    )

    plt.colorbar(
        label="Normalized log energy"
    )

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        tokens
    )

    plt.xlabel(
        "Token frame"
    )

    plt.ylabel(
        "Token ID"
    )

    plt.title(
        f"{title} — discrete speech tokens"
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# 17. MAIN
# ======================================================================

def main():

    args = parse_args()

    print("=" * 72)
    print("LESSON 39 — REAL SPEECH TOKEN DATASET PIPELINE")
    print("=" * 72)

    print()
    print("Device:", DEVICE)
    print(
        "Target sample rate:",
        TARGET_SAMPLE_RATE
    )

    print(
        "Token hop:",
        TOKEN_HOP_MS,
        "ms"
    )

    print(
        "Vocabulary:",
        VOCAB_SIZE
    )

    print(
        "Context length:",
        args.context_length
    )

    # --------------------------------------------------------------
    # STEP 1 — DISCOVER AUDIO
    # --------------------------------------------------------------

    files = find_audio_files(
        args.audio_dir
    )

    if args.max_files is not None:

        files = files[
            :args.max_files
        ]

    print()
    print(
        "Audio files found:",
        len(files)
    )

    if len(files) == 0:

        print()
        print(
            "No audio files found."
        )

        print()
        print(
            "Create:"
        )

        print(
            f"  {args.audio_dir}/"
        )

        print(
            "and place WAV/FLAC/MP3 files inside."
        )

        print()
        print(
            "Example:"
        )

        print(
            "  dataset/"
        )

        print(
            "      sample_001.wav"
        )

        print(
            "      sample_002.wav"
        )

        print()

        return

    for path in files[:10]:

        print(
            " ",
            path
        )

    if len(files) > 10:

        print(
            f" ... and {len(files) - 10} more"
        )

    # --------------------------------------------------------------
    # STEP 2 — TRAIN/VALIDATION SPLIT
    # --------------------------------------------------------------

    train_files, val_files = split_files(
        files,
        args.val_ratio,
        SEED
    )

    print()
    print(
        "Train files:",
        len(train_files)
    )

    print(
        "Validation files:",
        len(val_files)
    )

    # --------------------------------------------------------------
    # STEP 3 — OUTPUT DIRECTORY
    # --------------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    train_token_dir = (
        output_dir
        / "train_recordings"
    )

    val_token_dir = (
        output_dir
        / "val_recordings"
    )

    train_token_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    val_token_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------------
    # STEP 4 — TOKENIZER
    # --------------------------------------------------------------

    tokenizer = (
        DemonstrationSpeechTokenizer(
            vocab_size=VOCAB_SIZE
        )
    )

    print()
    print(
        "Tokenizer:"
    )

    print(
        "  Demonstration deterministic "
        "mel-to-token tokenizer"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is NOT a production neural speech codec."
    )

    print(
        "It is used to verify the real-audio dataset pipeline."
    )

    # --------------------------------------------------------------
    # STEP 5 — PROCESS AUDIO
    # --------------------------------------------------------------

    metadata = []

    train_token_sequences = []

    val_token_sequences = []

    rejected = []

    example_visualized = False

    all_split_files = [
        ("train", path)
        for path in train_files
    ] + [
        ("val", path)
        for path in val_files
    ]

    print()
    print("=" * 72)
    print("PROCESSING AUDIO")
    print("=" * 72)

    for file_index, (
        split,
        path
    ) in enumerate(
        all_split_files
    ):

        print(
            f"[{file_index + 1:04d}/"
            f"{len(all_split_files):04d}] "
            f"{split:5s} "
            f"{path.name}"
        )

        try:

            waveform = load_audio(
                path
            )

            valid, reason = validate_audio(
                waveform
            )

            if not valid:

                rejected.append(
                    {
                        "file": str(path),
                        "reason": reason
                    }
                )

                print(
                    "    REJECT:",
                    reason
                )

                continue

            duration = (
                waveform.shape[-1]
                / TARGET_SAMPLE_RATE
            )

            # ------------------------------------------------------
            # Extract tokens.
            # ------------------------------------------------------

            tokens = tokenizer.encode(
                waveform
            )

            tokens = validate_tokens(
                tokens,
                VOCAB_SIZE
            )

            token_count = len(
                tokens
            )

            token_rate = (
                token_count
                / duration
            )

            # ------------------------------------------------------
            # Save recording-level tokens.
            # ------------------------------------------------------

            if split == "train":

                token_path = (
                    train_token_dir
                    / f"{file_index:06d}.npy"
                )

                train_token_sequences.append(
                    tokens.numpy()
                )

            else:

                token_path = (
                    val_token_dir
                    / f"{file_index:06d}.npy"
                )

                val_token_sequences.append(
                    tokens.numpy()
                )

            save_token_sequence(
                token_path,
                tokens
            )

            metadata.append(
                {
                    "file": str(path),
                    "split": split,
                    "duration_sec": float(
                        duration
                    ),
                    "num_tokens": int(
                        token_count
                    ),
                    "token_rate_hz": float(
                        token_rate
                    ),
                    "token_file": str(
                        token_path
                    )
                }
            )

            print(
                f"    duration={duration:.2f}s "
                f"tokens={token_count} "
                f"rate={token_rate:.2f} tokens/s"
            )

            # ------------------------------------------------------
            # Visualize first valid example.
            # ------------------------------------------------------

            if (
                args.plot
                and not example_visualized
            ):

                mel = (
                    tokenizer
                    .mel_extractor(
                        waveform
                    )
                )

                visualize_example(
                    waveform,
                    mel,
                    tokens,
                    path.name
                )

                example_visualized = True

        except Exception as exc:

            rejected.append(
                {
                    "file": str(path),
                    "reason": (
                        f"exception:{exc}"
                    )
                }
            )

            print(
                "    ERROR:",
                exc
            )

    # --------------------------------------------------------------
    # STEP 6 — CREATE FIXED-LENGTH CHUNKS
    # --------------------------------------------------------------

    train_chunks = create_chunks(
        train_token_sequences,
        args.context_length
    )

    val_chunks = create_chunks(
        val_token_sequences,
        args.context_length
    )

    print()
    print("=" * 72)
    print("FIXED-LENGTH TOKEN DATASET")
    print("=" * 72)

    print()
    print(
        "Train recording sequences:",
        len(train_token_sequences)
    )

    print(
        "Validation recording sequences:",
        len(val_token_sequences)
    )

    print(
        "Train chunks:",
        train_chunks.shape
    )

    print(
        "Validation chunks:",
        val_chunks.shape
    )

    # --------------------------------------------------------------
    # STEP 7 — SAVE DATASET
    # --------------------------------------------------------------

    np.save(
        output_dir / "train.npy",
        train_chunks
    )

    np.save(
        output_dir / "val.npy",
        val_chunks
    )

    # --------------------------------------------------------------
    # STEP 8 — STATISTICS
    # --------------------------------------------------------------

    train_stats = token_statistics(
        train_token_sequences,
        VOCAB_SIZE
    )

    val_stats = token_statistics(
        val_token_sequences,
        VOCAB_SIZE
    )

    print()
    print("=" * 72)
    print("TRAIN TOKEN STATISTICS")
    print("=" * 72)

    if train_stats:

        for key, value in train_stats.items():

            print(
                f"{key:30s}: {value}"
            )

    print()
    print("=" * 72)
    print("VALIDATION TOKEN STATISTICS")
    print("=" * 72)

    if val_stats:

        for key, value in val_stats.items():

            print(
                f"{key:30s}: {value}"
            )

    # --------------------------------------------------------------
    # STEP 9 — DISTRIBUTION PLOT
    # --------------------------------------------------------------

    if args.plot:

        plot_token_distribution(
            train_token_sequences,
            VOCAB_SIZE,
            "Training Speech Token Distribution"
        )

        plot_token_distribution(
            val_token_sequences,
            VOCAB_SIZE,
            "Validation Speech Token Distribution"
        )

    # --------------------------------------------------------------
    # STEP 10 — METADATA
    # --------------------------------------------------------------

    metadata_output = {
        "sample_rate": TARGET_SAMPLE_RATE,
        "vocab_size": VOCAB_SIZE,
        "token_hop_ms": TOKEN_HOP_MS,
        "context_length": args.context_length,
        "num_audio_files": len(files),
        "num_processed": len(metadata),
        "num_rejected": len(rejected),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "train_chunks": int(
            len(train_chunks)
        ),
        "val_chunks": int(
            len(val_chunks)
        ),
        "train_statistics": train_stats,
        "validation_statistics": val_stats,
        "rejected": rejected,
        "tokenizer": (
            "demonstration_mel_to_token"
        )
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata_output,
            f,
            indent=2
        )

    # --------------------------------------------------------------
    # STEP 11 — FINAL REPORT
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("LESSON 39 COMPLETE")
    print("=" * 72)

    print()

    print(
        "Output directory:"
    )

    print(
        " ",
        output_dir.resolve()
    )

    print()

    print(
        "Files:"
    )

    print(
        "  train.npy"
    )

    print(
        "  val.npy"
    )

    print(
        "  metadata.json"
    )

    print(
        "  train_recordings/*.npy"
    )

    print(
        "  val_recordings/*.npy"
    )

    print()

    print(
        "Processed:",
        len(metadata)
    )

    print(
        "Rejected:",
        len(rejected)
    )

    print()

    if len(train_chunks) == 0:

        print(
            "WARNING:"
        )

        print(
            "No complete training chunks were produced."
        )

        print(
            "You need longer recordings or a smaller "
            "context length."
        )

    else:

        print(
            "Training dataset is ready for Lesson 40."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The current tokenizer is a demonstration tokenizer."
    )

    print(
        "For real TTS quality, replace it with the repaired "
        "neural speech codec from Lesson 36."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Lesson 40 — Train the autoregressive model on "
        "real speech tokens."
    )

    print()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()