"""
========================================================================
PHASE 2 — LESSON 59
IMPROVED ACOUSTIC MODEL TRAINING
========================================================================

PURPOSE
-------
Lesson 58 introduced a FastSpeech 2-style acoustic model:

    phonemes
        ↓
    phoneme embedding
        ↓
    Transformer encoder
        ↓
    duration predictor
        ↓
    length regulator
        ↓
    Transformer decoder
        ↓
    Mel spectrogram

Lesson 59 improves the acoustic part of that system.

The goal is NOT to build canonical FastSpeech 2 yet.

Instead, we progressively introduce the important engineering ideas
needed for a practical acoustic model:

    1. Masked Mel reconstruction loss
    2. Combined L1 + MSE acoustic loss
    3. Residual acoustic refinement
    4. Convolutional post-net
    5. Teacher-forced acoustic reconstruction
    6. Predicted-duration acoustic synthesis
    7. Better training diagnostics
    8. Train/validation handling
    9. Robust alignment/Mel frame handling
   10. Visualization of acoustic reconstruction

NO PRETRAINED MODELS
--------------------
This lesson does NOT use:

    - FastSpeech pretrained checkpoints
    - FastSpeech 2 pretrained checkpoints
    - Tacotron
    - Whisper
    - Wav2Vec2
    - HuBERT
    - Montreal Forced Aligner
    - pretrained phonemizers
    - pretrained vocoders
    - pretrained TTS models

Everything important is implemented from scratch.

========================================================================
WHAT YOU LEARN
========================================================================

STEP 1 — MASKED MEL LOSS
------------------------
Training batches contain padded Mel sequences.

Padding must NOT contribute to the loss.

We therefore calculate:

    loss = sum(error * mask) / sum(mask)

rather than averaging over padded frames.

This becomes essential when moving from one utterance to many
utterances with different lengths.

-----------------------------------------------------------------------

STEP 2 — COMBINED ACOUSTIC LOSS
-------------------------------
We use both:

    L1 loss
    MSE loss

L1:

    |y_hat - y|

MSE:

    (y_hat - y)^2

L1 tends to be robust to outliers.

MSE strongly penalizes larger errors.

The combined loss is:

    L_acoustic = lambda_l1 * L1
               + lambda_mse * MSE

-----------------------------------------------------------------------

STEP 3 — RESIDUAL ACOUSTIC REFINEMENT
--------------------------------------
The decoder first predicts a coarse Mel spectrogram:

    Mel_coarse

Then a refinement network predicts:

    Mel_residual

and the final output becomes:

    Mel_final = Mel_coarse + Mel_residual

This is easier for a network than learning the entire acoustic mapping
again from scratch.

The refinement network only needs to learn:

    target - coarse_prediction

-----------------------------------------------------------------------

STEP 4 — POST-NET
-----------------
A convolutional post-net is used to refine local spectral structure.

Conceptually:

    Transformer decoder
            ↓
       coarse Mel
            ↓
       Conv post-net
            ↓
       residual Mel
            ↓
       final Mel

This is inspired by the refinement idea used in classical neural TTS
systems, while keeping the implementation educational and from scratch.

-----------------------------------------------------------------------

STEP 5 — TEACHER-FORCED DURATION
--------------------------------
During training we use ground-truth durations:

    phonemes
       ↓
    ground-truth durations
       ↓
    length regulator
       ↓
    acoustic decoder

This allows us to study acoustic reconstruction independently from
duration prediction errors.

-----------------------------------------------------------------------

STEP 6 — PREDICTED-DURATION INFERENCE
-------------------------------------
At inference:

    phonemes
       ↓
    duration predictor
       ↓
    predicted durations
       ↓
    length regulator
       ↓
    acoustic decoder
       ↓
    Mel spectrogram

This is closer to actual TTS inference.

-----------------------------------------------------------------------

STEP 7 — TWO TYPES OF RECONSTRUCTION
------------------------------------
We explicitly distinguish:

A. Teacher-forced reconstruction

    Uses true phoneme durations.

B. Predicted-duration reconstruction

    Uses the model's predicted durations.

This distinction is important.

If predicted-duration Mel quality is poor, the problem may be:

    duration prediction

rather than:

    acoustic modeling.

-----------------------------------------------------------------------

STEP 8 — VISUAL DIAGNOSTICS
----------------------------
The script saves:

    training_loss.png
    teacher_forced_target_mel.png
    teacher_forced_coarse_mel.png
    teacher_forced_final_mel.png
    teacher_forced_residual.png
    predicted_duration_mel.png
    acoustic_error.png
    duration_comparison.png

These plots allow us to inspect what the model actually learned.

========================================================================
DATA
========================================================================

Expected previous lesson outputs:

    lesson54_outputs/
        *_alignment.json

    lesson55_outputs/
        features/
            *.npz

The alignment JSON contains information similar to:

    aligned_phonemes:
        phoneme
        start_frame
        end_frame
        duration_frames

The Mel NPZ contains:

    log_mel
    normalized_log_mel
    sample_rate
    hop_length

The loader is intentionally robust to a difference between:

    alignment interval length

and:

    total Mel frame count.

For example:

    alignment = 409 frames
    Mel        = 442 frames

We DO NOT stretch the phoneme durations.

Instead, if the alignment interval is valid:

    Mel[alignment_start:alignment_end]

is used.

This preserves the original phoneme timing.

========================================================================
IMPORTANT LIMITATION
========================================================================

If the dataset contains only one utterance:

    training samples = 1
    validation samples = 0

Then this lesson demonstrates the complete pipeline, but it does NOT
demonstrate generalization.

A low training loss in that situation means:

    "the model can fit this example"

not:

    "the model is a good general TTS acoustic model."

========================================================================
OUTPUT DIRECTORY
========================================================================

lesson59_outputs/

    vocabulary.json

    improved_acoustic_model_best.pt

    training_loss.png
    teacher_forced_target_mel.png
    teacher_forced_coarse_mel.png
    teacher_forced_final_mel.png
    teacher_forced_residual.png
    predicted_duration_mel.png
    acoustic_error.png
    duration_comparison.png

========================================================================
"""

import os
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

OUTPUT_DIR = Path("lesson59_outputs")

ALIGNMENT_DIR = Path("lesson54_outputs")
MEL_FEATURE_DIR = Path("lesson55_outputs/features")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

EMBED_DIM = 128
ATTENTION_HEADS = 4

ENCODER_LAYERS = 3
DECODER_LAYERS = 3

FF_DIM = 512

DROPOUT = 0.10

POSTNET_CHANNELS = 128
POSTNET_LAYERS = 5
POSTNET_KERNEL_SIZE = 5

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

BATCH_SIZE = 1

EPOCHS = 300

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

GRAD_CLIP = 1.0

PATIENCE = 40

TRAIN_RATIO = 0.8

# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------

DURATION_LOSS_WEIGHT = 1.0

MEL_L1_WEIGHT = 1.0
MEL_MSE_WEIGHT = 0.5

COARSE_MEL_WEIGHT = 0.5
FINAL_MEL_WEIGHT = 1.0

RESIDUAL_WEIGHT = 0.05

# ---------------------------------------------------------------------------
# Data limits
# ---------------------------------------------------------------------------

MAX_SEQUENCE_LENGTH = 4096

MIN_DURATION = 1
MAX_DURATION = 100

MAX_SYNTHESIS_FRAMES = 2000

PAD_ID = 0

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings improve reproducibility.
    # They can reduce GPU performance slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value, default=None):
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def safe_int(value, default=None):
    try:
        return int(round(float(value)))
    except Exception:
        return default


def find_alignment_files():
    if not ALIGNMENT_DIR.exists():
        raise FileNotFoundError(
            f"Alignment directory does not exist:\n"
            f"  {ALIGNMENT_DIR.resolve()}\n\n"
            f"Run Lesson 54 first."
        )

    files = sorted(ALIGNMENT_DIR.glob("*_alignment.json"))

    if not files:
        files = sorted(ALIGNMENT_DIR.glob("*.json"))

    return files


def find_mel_file(stem: str):
    candidates = [
        MEL_FEATURE_DIR / f"{stem}.npz",
        MEL_FEATURE_DIR / f"{stem}_features.npz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback search.
    matches = list(MEL_FEATURE_DIR.glob(f"{stem}*.npz"))

    if matches:
        return matches[0]

    return None


# ============================================================================
# MEL ORIENTATION
# ============================================================================

def normalize_mel_orientation(mel, npz_data):
    """
    Convert Mel spectrogram into:

        [frames, n_mels]

    Previous lessons may store Mel as either:

        [n_mels, frames]

    or:

        [frames, n_mels]

    Prefer explicit n_mels metadata when available.
    """

    mel = np.asarray(mel)

    if mel.ndim != 2:
        raise ValueError(
            f"Mel must be 2-D, got shape {mel.shape}"
        )

    n_mels = None

    if "n_mels" in npz_data:
        n_mels = safe_int(npz_data["n_mels"])

    if n_mels is not None and n_mels > 0:

        if mel.shape[0] == n_mels:
            return mel.T.astype(np.float32)

        if mel.shape[1] == n_mels:
            return mel.astype(np.float32)

        raise ValueError(
            f"Mel shape {mel.shape} is incompatible with "
            f"n_mels={n_mels}"
        )

    # Common educational Mel sizes.
    common_mel_sizes = {40, 64, 80, 100, 128, 160}

    if mel.shape[0] in common_mel_sizes:
        return mel.T.astype(np.float32)

    if mel.shape[1] in common_mel_sizes:
        return mel.astype(np.float32)

    # Conservative fallback:
    # Mel bins are normally much smaller than the number of frames.
    if mel.shape[0] < mel.shape[1]:
        return mel.T.astype(np.float32)

    return mel.astype(np.float32)


# ============================================================================
# ALIGNMENT PARSING
# ============================================================================

def parse_alignment_items(alignment_data):
    """
    Extract aligned phoneme intervals.

    We prefer explicit:

        start_frame
        end_frame

    because these preserve the original alignment boundaries.

    Duration is derived as:

        end_frame - start_frame

    whenever both boundaries are available.
    """

    if "aligned_phonemes" in alignment_data:
        items = alignment_data["aligned_phonemes"]

    elif "phonemes" in alignment_data:
        items = alignment_data["phonemes"]

    elif isinstance(alignment_data, list):
        items = alignment_data

    else:
        raise ValueError(
            "Could not find aligned_phonemes/phonemes in alignment JSON."
        )

    if not isinstance(items, list):
        raise ValueError("Alignment phoneme list is not a list.")

    parsed = []

    for index, item in enumerate(items):

        if not isinstance(item, dict):
            raise ValueError(
                f"Alignment item {index} is not a dictionary."
            )

        phoneme = (
            item.get("phoneme")
            or item.get("token")
            or item.get("symbol")
        )

        if phoneme is None:
            raise ValueError(
                f"Alignment item {index} has no phoneme field."
            )

        phoneme = str(phoneme)

        start = safe_int(item.get("start_frame"))
        end = safe_int(item.get("end_frame"))

        duration = safe_int(item.get("duration_frames"))

        if start is None or end is None:

            if duration is None:
                raise ValueError(
                    f"Alignment item {index} has neither "
                    f"valid boundaries nor duration."
                )

            if duration < MIN_DURATION:
                raise ValueError(
                    f"Alignment item {index} has invalid duration "
                    f"{duration}."
                )

            # Cannot determine absolute alignment interval.
            # We can construct a relative interval below.
            parsed.append(
                {
                    "phoneme": phoneme,
                    "start_frame": None,
                    "end_frame": None,
                    "duration_frames": duration,
                }
            )

            continue

        if start < 0:
            raise ValueError(
                f"Alignment item {index} has negative start frame."
            )

        if end <= start:
            raise ValueError(
                f"Alignment item {index} has invalid interval "
                f"[{start}, {end})."
            )

        boundary_duration = end - start

        if duration is not None and duration != boundary_duration:
            warnings.warn(
                f"Alignment item {index}: "
                f"duration_frames={duration} but "
                f"end-start={boundary_duration}. "
                f"Using end-start."
            )

        parsed.append(
            {
                "phoneme": phoneme,
                "start_frame": start,
                "end_frame": end,
                "duration_frames": boundary_duration,
            }
        )

    if not parsed:
        raise ValueError("No phoneme alignment items found.")

    return parsed


# ============================================================================
# DATASET RECORD LOADING
# ============================================================================

def load_records(
    alignment_dir=ALIGNMENT_DIR,
    mel_dir=MEL_FEATURE_DIR,
):
    print()
    print("=" * 72)
    print("STEP 1 — LOAD DATA")
    print("=" * 72)

    alignment_files = sorted(alignment_dir.glob("*_alignment.json"))

    if not alignment_files:
        alignment_files = sorted(alignment_dir.glob("*.json"))

    if not alignment_files:
        raise RuntimeError(
            f"No alignment JSON files found in {alignment_dir}"
        )

    records = []

    for alignment_path in alignment_files:

        stem = alignment_path.stem

        if stem.endswith("_alignment"):
            base_stem = stem[:-len("_alignment")]
        else:
            base_stem = stem

        mel_path = find_mel_file(base_stem)

        if mel_path is None:
            print(
                f"WARNING: no Mel file found for "
                f"{alignment_path.name}"
            )
            continue

        try:
            with open(alignment_path, "r", encoding="utf-8") as f:
                alignment_data = json.load(f)

            aligned_items = parse_alignment_items(alignment_data)

            # ---------------------------------------------------------------
            # First determine whether absolute frame boundaries exist.
            # ---------------------------------------------------------------

            has_absolute_boundaries = all(
                item["start_frame"] is not None
                and item["end_frame"] is not None
                for item in aligned_items
            )

            # ---------------------------------------------------------------
            # Load Mel
            # ---------------------------------------------------------------

            with np.load(mel_path, allow_pickle=True) as data:

                if "normalized_log_mel" in data:
                    mel = data["normalized_log_mel"]

                elif "log_mel" in data:
                    mel = data["log_mel"]

                else:
                    raise ValueError(
                        "NPZ contains neither normalized_log_mel "
                        "nor log_mel."
                    )

                mel = normalize_mel_orientation(mel, data)

                sample_rate = (
                    safe_int(data["sample_rate"])
                    if "sample_rate" in data
                    else None
                )

                hop_length = (
                    safe_int(data["hop_length"])
                    if "hop_length" in data
                    else None
                )

            mel_frames = mel.shape[0]
            n_mels = mel.shape[1]

            # ---------------------------------------------------------------
            # Absolute-boundary case
            # ---------------------------------------------------------------

            if has_absolute_boundaries:

                previous_end = None

                for item in aligned_items:

                    start = item["start_frame"]
                    end = item["end_frame"]

                    if start < 0 or end > mel_frames:
                        raise ValueError(
                            f"Alignment interval [{start}, {end}) "
                            f"is outside Mel range [0, {mel_frames})."
                        )

                    if previous_end is not None and start != previous_end:
                        raise ValueError(
                            "Alignment intervals are not continuous: "
                            f"previous_end={previous_end}, "
                            f"current_start={start}."
                        )

                    previous_end = end

                first_start = aligned_items[0]["start_frame"]
                last_end = aligned_items[-1]["end_frame"]

                duration_sum = sum(
                    item["duration_frames"]
                    for item in aligned_items
                )

                interval_length = last_end - first_start

                if interval_length != duration_sum:
                    raise ValueError(
                        f"Alignment interval length "
                        f"{interval_length} != duration sum "
                        f"{duration_sum}."
                    )

                # -----------------------------------------------------------
                # IMPORTANT FIX:
                #
                # If Mel contains extra frames outside the aligned interval,
                # crop the Mel.
                #
                # Example:
                #
                # alignment = 409 frames
                # Mel       = 442 frames
                #
                # We use Mel[0:409].
                #
                # We do NOT stretch the phoneme durations.
                # -----------------------------------------------------------

                if mel_frames != interval_length:

                    print(
                        f"INFO: {base_stem}: cropping Mel frames "
                        f"[{first_start}:{last_end}] from "
                        f"{mel_frames} frames."
                    )

                    mel = mel[first_start:last_end]

                else:

                    # Even if total lengths match, preserve the exact
                    # aligned interval if it does not start at zero.
                    if first_start != 0:
                        print(
                            f"INFO: {base_stem}: extracting aligned "
                            f"Mel interval [{first_start}:{last_end}]."
                        )

                        mel = mel[first_start:last_end]

            # ---------------------------------------------------------------
            # Relative-duration fallback
            # ---------------------------------------------------------------

            else:

                durations = [
                    item["duration_frames"]
                    for item in aligned_items
                ]

                duration_sum = sum(durations)

                if duration_sum > mel_frames:
                    raise ValueError(
                        f"Duration sum {duration_sum} exceeds "
                        f"Mel frames {mel_frames}."
                    )

                if duration_sum < mel_frames:

                    print(
                        f"INFO: {base_stem}: using first "
                        f"{duration_sum} Mel frames from "
                        f"{mel_frames} frames."
                    )

                    mel = mel[:duration_sum]

            # ---------------------------------------------------------------
            # Final consistency check
            # ---------------------------------------------------------------

            durations = [
                int(item["duration_frames"])
                for item in aligned_items
            ]

            phonemes = [
                item["phoneme"]
                for item in aligned_items
            ]

            if any(d < MIN_DURATION for d in durations):
                raise ValueError(
                    f"Found duration smaller than {MIN_DURATION}."
                )

            if len(phonemes) == 0:
                raise ValueError("No phonemes remain.")

            if sum(durations) != mel.shape[0]:
                raise ValueError(
                    f"Final mismatch: duration sum={sum(durations)}, "
                    f"Mel frames={mel.shape[0]}."
                )

            if len(phonemes) > MAX_SEQUENCE_LENGTH:
                raise ValueError(
                    f"Phoneme sequence too long: {len(phonemes)}"
                )

            # ---------------------------------------------------------------
            # Remove NaN/Inf
            # ---------------------------------------------------------------

            if not np.isfinite(mel).all():
                raise ValueError(
                    "Mel contains NaN or infinite values."
                )

            records.append(
                {
                    "id": base_stem,
                    "phonemes": phonemes,
                    "durations": durations,
                    "mel": mel.astype(np.float32),
                    "sample_rate": sample_rate,
                    "hop_length": hop_length,
                }
            )

            print(
                f"Loaded: {base_stem:40s} "
                f"phonemes={len(phonemes):4d} "
                f"frames={mel.shape[0]:5d} "
                f"mels={mel.shape[1]:3d}"
            )

        except Exception as exc:

            print(
                f"WARNING: skipping {alignment_path.name}: {exc}"
            )

    if not records:
        raise RuntimeError(
            "No valid aligned Mel records found.\n"
            "Check Lesson 54 alignment files and Lesson 55 Mel files."
        )

    print()
    print(f"Valid records: {len(records)}")

    return records


# ============================================================================
# VOCABULARY
# ============================================================================

def build_vocabulary(records):
    phonemes = set()

    for record in records:
        phonemes.update(record["phonemes"])

    phonemes = sorted(phonemes)

    # PAD must remain 0.
    vocabulary = {
        "<pad>": PAD_ID
    }

    for phoneme in phonemes:
        if phoneme == "<pad>":
            continue

        vocabulary[phoneme] = len(vocabulary)

    return vocabulary


# ============================================================================
# DATASET
# ============================================================================

class AcousticDataset(Dataset):

    def __init__(self, records, vocabulary):

        self.records = records
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        phoneme_ids = [
            self.vocabulary[p]
            for p in record["phonemes"]
        ]

        durations = np.asarray(
            record["durations"],
            dtype=np.int64,
        )

        mel = np.asarray(
            record["mel"],
            dtype=np.float32,
        )

        return {
            "id": record["id"],
            "phoneme_ids": torch.tensor(
                phoneme_ids,
                dtype=torch.long,
            ),
            "durations": torch.tensor(
                durations,
                dtype=torch.long,
            ),
            "log_durations": torch.log1p(
                torch.tensor(
                    durations,
                    dtype=torch.float32,
                )
            ),
            "mel": torch.tensor(
                mel,
                dtype=torch.float32,
            ),
        }


# ============================================================================
# COLLATE
# ============================================================================

def collate_batch(batch):

    batch_size = len(batch)

    max_phonemes = max(
        item["phoneme_ids"].shape[0]
        for item in batch
    )

    max_frames = max(
        item["mel"].shape[0]
        for item in batch
    )

    n_mels = batch[0]["mel"].shape[1]

    phoneme_ids = torch.full(
        (batch_size, max_phonemes),
        PAD_ID,
        dtype=torch.long,
    )

    durations = torch.zeros(
        (batch_size, max_phonemes),
        dtype=torch.long,
    )

    log_durations = torch.zeros(
        (batch_size, max_phonemes),
        dtype=torch.float32,
    )

    mel = torch.zeros(
        (batch_size, max_frames, n_mels),
        dtype=torch.float32,
    )

    phoneme_mask = torch.ones(
        (batch_size, max_phonemes),
        dtype=torch.bool,
    )

    mel_mask = torch.ones(
        (batch_size, max_frames),
        dtype=torch.bool,
    )

    ids = []

    for i, item in enumerate(batch):

        p_len = item["phoneme_ids"].shape[0]
        f_len = item["mel"].shape[0]

        phoneme_ids[i, :p_len] = item["phoneme_ids"]
        durations[i, :p_len] = item["durations"]
        log_durations[i, :p_len] = item["log_durations"]

        mel[i, :f_len] = item["mel"]

        phoneme_mask[i, :p_len] = False
        mel_mask[i, :f_len] = False

        ids.append(item["id"])

    return {
        "ids": ids,
        "phoneme_ids": phoneme_ids,
        "durations": durations,
        "log_durations": log_durations,
        "mel": mel,
        "phoneme_padding_mask": phoneme_mask,
        "mel_padding_mask": mel_mask,
    }


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embed_dim,
        max_len=4096,
    ):

        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embed_dim,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(
            max_len,
            embed_dim,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):

        length = x.shape[1]

        if length > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length {length} exceeds "
                f"maximum positional encoding length "
                f"{self.pe.shape[1]}."
            )

        return x + self.pe[:, :length]


# ============================================================================
# PHONEME ENCODER
# ============================================================================

class PhonemeEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        heads,
        layers,
        ff_dim,
        dropout,
        max_len,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=PAD_ID,
        )

        self.positional_encoding = PositionalEncoding(
            embed_dim,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        phoneme_ids,
        padding_mask,
    ):

        x = self.embedding(phoneme_ids)

        x = self.positional_encoding(x)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.norm(x)

        return x


# ============================================================================
# DURATION PREDICTOR
# ============================================================================

class DurationPredictor(nn.Module):

    def __init__(
        self,
        embed_dim,
        dropout=0.1,
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(
            embed_dim,
            1,
        )

    def forward(self, x):

        # x:
        # [B, T, C]

        y = x.transpose(1, 2)

        y = self.conv1(y)

        y = y.transpose(1, 2)
        y = self.norm1(y)
        y = F.relu(y)
        y = self.dropout(y)

        y = y.transpose(1, 2)

        y = self.conv2(y)

        y = y.transpose(1, 2)
        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)

        output = self.output(y).squeeze(-1)

        return output


# ============================================================================
# LENGTH REGULATOR
# ============================================================================

class LengthRegulator(nn.Module):

    def __init__(
        self,
        max_duration=MAX_DURATION,
    ):

        super().__init__()

        self.max_duration = max_duration

    def forward(
        self,
        x,
        durations,
    ):

        # x:
        # [B, phonemes, C]
        #
        # durations:
        # [B, phonemes]

        expanded = []

        for batch_index in range(x.shape[0]):

            pieces = []

            for phoneme_index in range(x.shape[1]):

                duration = int(
                    durations[
                        batch_index,
                        phoneme_index,
                    ].item()
                )

                if duration <= 0:
                    continue

                duration = min(
                    duration,
                    self.max_duration,
                )

                vector = x[
                    batch_index,
                    phoneme_index,
                ]

                vector = vector.unsqueeze(0)

                vector = vector.repeat(
                    duration,
                    1,
                )

                pieces.append(vector)

            if pieces:
                sequence = torch.cat(
                    pieces,
                    dim=0,
                )
            else:
                sequence = x.new_zeros(
                    (1, x.shape[-1])
                )

            expanded.append(sequence)

        max_length = max(
            item.shape[0]
            for item in expanded
        )

        output = x.new_zeros(
            (
                x.shape[0],
                max_length,
                x.shape[-1],
            )
        )

        padding_mask = torch.ones(
            (
                x.shape[0],
                max_length,
            ),
            dtype=torch.bool,
            device=x.device,
        )

        for batch_index, sequence in enumerate(expanded):

            length = sequence.shape[0]

            output[
                batch_index,
                :length,
            ] = sequence

            padding_mask[
                batch_index,
                :length,
            ] = False

        return output, padding_mask


# ============================================================================
# MEL DECODER
# ============================================================================

class MelDecoder(nn.Module):

    def __init__(
        self,
        embed_dim,
        heads,
        layers,
        ff_dim,
        dropout,
        n_mels,
        max_len,
    ):

        super().__init__()

        self.positional_encoding = PositionalEncoding(
            embed_dim,
            max_len=max_len,
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.mel_projection = nn.Linear(
            embed_dim,
            n_mels,
        )

    def forward(
        self,
        x,
        padding_mask,
    ):

        x = self.positional_encoding(x)

        x = self.decoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.norm(x)

        mel = self.mel_projection(x)

        return mel


# ============================================================================
# POST-NET
# ============================================================================

class MelPostNet(nn.Module):

    """
    Convolutional residual refinement network.

    Input:

        [B, frames, n_mels]

    Output:

        [B, frames, n_mels]

    The output is a residual correction, not the final Mel itself.
    """

    def __init__(
        self,
        n_mels,
        channels=POSTNET_CHANNELS,
        layers=POSTNET_LAYERS,
        kernel_size=POSTNET_KERNEL_SIZE,
        dropout=DROPOUT,
    ):

        super().__init__()

        modules = []

        padding = kernel_size // 2

        # First layer:
        # Mel -> hidden channels

        modules.append(
            nn.Conv1d(
                n_mels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        )

        modules.append(nn.Tanh())
        modules.append(nn.Dropout(dropout))

        # Middle refinement layers.

        for _ in range(layers - 2):

            modules.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

            modules.append(nn.Tanh())
            modules.append(nn.Dropout(dropout))

        # Final layer:
        # hidden channels -> Mel

        modules.append(
            nn.Conv1d(
                channels,
                n_mels,
                kernel_size=kernel_size,
                padding=padding,
            )
        )

        self.network = nn.Sequential(*modules)

    def forward(self, mel):

        # [B, T, M]
        x = mel.transpose(1, 2)

        residual = self.network(x)

        residual = residual.transpose(1, 2)

        return residual


# ============================================================================
# COMPLETE IMPROVED ACOUSTIC MODEL
# ============================================================================

class ImprovedFastSpeechAcousticModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_mels,
        embed_dim=EMBED_DIM,
        heads=ATTENTION_HEADS,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        max_len=MAX_SEQUENCE_LENGTH,
    ):

        super().__init__()

        self.encoder = PhonemeEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            heads=heads,
            layers=encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_len=max_len,
        )

        self.duration_predictor = DurationPredictor(
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.length_regulator = LengthRegulator(
            max_duration=MAX_DURATION,
        )

        self.decoder = MelDecoder(
            embed_dim=embed_dim,
            heads=heads,
            layers=decoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            n_mels=n_mels,
            max_len=MAX_SYNTHESIS_FRAMES,
        )

        self.postnet = MelPostNet(
            n_mels=n_mels,
            channels=POSTNET_CHANNELS,
            layers=POSTNET_LAYERS,
            kernel_size=POSTNET_KERNEL_SIZE,
            dropout=dropout,
        )

    def predict_durations(
        self,
        encoded,
        phoneme_padding_mask,
    ):

        predicted_log_durations = self.duration_predictor(
            encoded
        )

        # Keep predictions numerically safe.
        predicted_log_durations = predicted_log_durations.clamp(
            min=0.0,
            max=math.log1p(MAX_DURATION),
        )

        predicted_durations = torch.round(
            torch.expm1(predicted_log_durations)
        ).long()

        predicted_durations = predicted_durations.clamp(
            min=MIN_DURATION,
            max=MAX_DURATION,
        )

        # Padding must have zero duration.
        predicted_durations = predicted_durations.masked_fill(
            phoneme_padding_mask,
            0,
        )

        return (
            predicted_log_durations,
            predicted_durations,
        )

    def forward(
        self,
        phoneme_ids,
        phoneme_padding_mask,
        ground_truth_durations=None,
        use_predicted_durations=False,
    ):

        encoded = self.encoder(
            phoneme_ids,
            phoneme_padding_mask,
        )

        predicted_log_durations = self.duration_predictor(
            encoded
        )

        predicted_log_durations = predicted_log_durations.clamp(
            min=0.0,
            max=math.log1p(MAX_DURATION),
        )

        predicted_durations = torch.round(
            torch.expm1(predicted_log_durations)
        ).long()

        predicted_durations = predicted_durations.clamp(
            min=MIN_DURATION,
            max=MAX_DURATION,
        )

        predicted_durations = predicted_durations.masked_fill(
            phoneme_padding_mask,
            0,
        )

        # ---------------------------------------------------------------
        # Choose duration source.
        # ---------------------------------------------------------------

        if (
            ground_truth_durations is not None
            and not use_predicted_durations
        ):

            durations_for_acoustic_model = (
                ground_truth_durations.clone()
            )

            durations_for_acoustic_model = durations_for_acoustic_model.clamp(
                min=0,
                max=MAX_DURATION,
            )

            durations_for_acoustic_model = (
                durations_for_acoustic_model.masked_fill(
                    phoneme_padding_mask,
                    0,
                )
            )

        else:

            durations_for_acoustic_model = (
                predicted_durations
            )

        # ---------------------------------------------------------------
        # Length regulation
        # ---------------------------------------------------------------

        expanded, frame_padding_mask = self.length_regulator(
            encoded,
            durations_for_acoustic_model,
        )

        # ---------------------------------------------------------------
        # Coarse Mel
        # ---------------------------------------------------------------

        coarse_mel = self.decoder(
            expanded,
            frame_padding_mask,
        )

        # ---------------------------------------------------------------
        # Post-net residual
        # ---------------------------------------------------------------

        residual = self.postnet(
            coarse_mel
        )

        final_mel = coarse_mel + residual

        # ---------------------------------------------------------------
        # Safety: zero padded output.
        # ---------------------------------------------------------------

        final_mel = final_mel.masked_fill(
            frame_padding_mask.unsqueeze(-1),
            0.0,
        )

        coarse_mel = coarse_mel.masked_fill(
            frame_padding_mask.unsqueeze(-1),
            0.0,
        )

        residual = residual.masked_fill(
            frame_padding_mask.unsqueeze(-1),
            0.0,
        )

        return {
            "encoded": encoded,
            "predicted_log_durations": predicted_log_durations,
            "predicted_durations": predicted_durations,
            "used_durations": durations_for_acoustic_model,
            "frame_padding_mask": frame_padding_mask,
            "coarse_mel": coarse_mel,
            "residual": residual,
            "final_mel": final_mel,
        }


# ============================================================================
# MASKED LOSSES
# ============================================================================

def masked_l1_loss(
    prediction,
    target,
    padding_mask,
):

    valid = (~padding_mask).unsqueeze(-1)

    error = torch.abs(
        prediction - target
    )

    error = error.masked_fill(
        ~valid,
        0.0,
    )

    denominator = valid.sum().clamp_min(1)

    denominator = denominator * prediction.shape[-1]

    return error.sum() / denominator


def masked_mse_loss(
    prediction,
    target,
    padding_mask,
):

    valid = (~padding_mask).unsqueeze(-1)

    error = (
        prediction - target
    ) ** 2

    error = error.masked_fill(
        ~valid,
        0.0,
    )

    denominator = valid.sum().clamp_min(1)

    denominator = denominator * prediction.shape[-1]

    return error.sum() / denominator


def masked_duration_mse(
    prediction,
    target,
    padding_mask,
):

    valid = ~padding_mask

    error = (
        prediction - target
    ) ** 2

    error = error.masked_fill(
        ~valid,
        0.0,
    )

    denominator = valid.sum().clamp_min(1)

    return error.sum() / denominator


# ============================================================================
# LOSS COMPUTATION
# ============================================================================

def compute_losses(
    outputs,
    batch,
):

    target_mel = batch["mel"]

    mel_padding_mask = outputs["frame_padding_mask"]

    target_durations = batch["log_durations"]

    duration_mask = batch["phoneme_padding_mask"]

    # ---------------------------------------------------------------
    # Duration loss
    # ---------------------------------------------------------------

    duration_loss = masked_duration_mse(
        outputs["predicted_log_durations"],
        target_durations,
        duration_mask,
    )

    # ---------------------------------------------------------------
    # Coarse acoustic loss
    # ---------------------------------------------------------------

    coarse_l1 = masked_l1_loss(
        outputs["coarse_mel"],
        target_mel,
        mel_padding_mask,
    )

    coarse_mse = masked_mse_loss(
        outputs["coarse_mel"],
        target_mel,
        mel_padding_mask,
    )

    coarse_acoustic = (
        MEL_L1_WEIGHT * coarse_l1
        + MEL_MSE_WEIGHT * coarse_mse
    )

    # ---------------------------------------------------------------
    # Final acoustic loss
    # ---------------------------------------------------------------

    final_l1 = masked_l1_loss(
        outputs["final_mel"],
        target_mel,
        mel_padding_mask,
    )

    final_mse = masked_mse_loss(
        outputs["final_mel"],
        target_mel,
        mel_padding_mask,
    )

    final_acoustic = (
        MEL_L1_WEIGHT * final_l1
        + MEL_MSE_WEIGHT * final_mse
    )

    # ---------------------------------------------------------------
    # Residual regularization
    # ---------------------------------------------------------------

    residual_energy = masked_mse_loss(
        outputs["residual"],
        torch.zeros_like(outputs["residual"]),
        mel_padding_mask,
    )

    # ---------------------------------------------------------------
    # Total
    # ---------------------------------------------------------------

    total_loss = (
        DURATION_LOSS_WEIGHT * duration_loss
        + COARSE_MEL_WEIGHT * coarse_acoustic
        + FINAL_MEL_WEIGHT * final_acoustic
        + RESIDUAL_WEIGHT * residual_energy
    )

    return {
        "total": total_loss,
        "duration": duration_loss,
        "coarse_l1": coarse_l1,
        "coarse_mse": coarse_mse,
        "final_l1": final_l1,
        "final_mse": final_mse,
        "residual": residual_energy,
    }


# ============================================================================
# DATASET SPLIT
# ============================================================================

def create_datasets(records, vocabulary):

    full_dataset = AcousticDataset(
        records,
        vocabulary,
    )

    total = len(full_dataset)

    if total < 2:

        print()
        print(
            "WARNING: only one valid utterance is available."
        )
        print(
            "Training and validation split is impossible."
        )
        print(
            "Validation metrics will therefore be unavailable."
        )
        print(
            "This is a pipeline demonstration, not a "
            "generalization experiment."
        )

        return full_dataset, None

    train_size = max(
        1,
        int(total * TRAIN_RATIO),
    )

    val_size = total - train_size

    if val_size < 1:

        train_size = total - 1
        val_size = 1

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_dataset, val_dataset


# ============================================================================
# DATA LOADERS
# ============================================================================

def create_loaders(
    train_dataset,
    val_dataset,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )

    val_loader = None

    if val_dataset is not None:

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=0,
        )

    return train_loader, val_loader


# ============================================================================
# MODEL PARAMETER COUNT
# ============================================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================================
# TRAINING EPOCH
# ============================================================================

def run_epoch(
    model,
    loader,
    optimizer=None,
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    totals = {
        "total": 0.0,
        "duration": 0.0,
        "coarse_l1": 0.0,
        "coarse_mse": 0.0,
        "final_l1": 0.0,
        "final_mse": 0.0,
        "residual": 0.0,
    }

    count = 0

    for batch in loader:

        phoneme_ids = batch[
            "phoneme_ids"
        ].to(DEVICE)

        phoneme_padding_mask = batch[
            "phoneme_padding_mask"
        ].to(DEVICE)

        durations = batch[
            "durations"
        ].to(DEVICE)

        batch_device = {
            key: value.to(DEVICE)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            outputs = model(
                phoneme_ids=phoneme_ids,
                phoneme_padding_mask=phoneme_padding_mask,
                ground_truth_durations=durations,
                use_predicted_durations=False,
            )

            losses = compute_losses(
                outputs,
                batch_device,
            )

            total_loss = losses["total"]

            if training:

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

        for key in totals:
            totals[key] += losses[key].item()

        count += 1

    if count == 0:
        return totals

    return {
        key: value / count
        for key, value in totals.items()
    }


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    vocabulary,
    n_mels,
):

    path = OUTPUT_DIR / "improved_acoustic_model_best.pt"

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocabulary": vocabulary,
        "n_mels": n_mels,
        "embed_dim": EMBED_DIM,
        "attention_heads": ATTENTION_HEADS,
        "encoder_layers": ENCODER_LAYERS,
        "decoder_layers": DECODER_LAYERS,
        "ff_dim": FF_DIM,
        "max_duration": MAX_DURATION,
    }

    torch.save(
        checkpoint,
        path,
    )

    return path


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    vocabulary,
    n_mels,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    history = {
        "train_total": [],
        "train_duration": [],
        "train_coarse_l1": [],
        "train_coarse_mse": [],
        "train_final_l1": [],
        "train_final_mse": [],
        "train_residual": [],
        "val_total": [],
        "val_duration": [],
        "val_final_l1": [],
        "val_final_mse": [],
    }

    best_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    print()
    print("=" * 72)
    print("STEP 4 — TRAIN IMPROVED ACOUSTIC MODEL")
    print("=" * 72)

    for epoch in range(1, EPOCHS + 1):

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
        )

        if val_loader is not None:

            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
            )

            monitored_loss = val_metrics["total"]

        else:

            val_metrics = None

            monitored_loss = train_metrics["total"]

        scheduler.step(monitored_loss)

        # ---------------------------------------------------------------
        # Store training metrics.
        # ---------------------------------------------------------------

        history["train_total"].append(
            train_metrics["total"]
        )

        history["train_duration"].append(
            train_metrics["duration"]
        )

        history["train_coarse_l1"].append(
            train_metrics["coarse_l1"]
        )

        history["train_coarse_mse"].append(
            train_metrics["coarse_mse"]
        )

        history["train_final_l1"].append(
            train_metrics["final_l1"]
        )

        history["train_final_mse"].append(
            train_metrics["final_mse"]
        )

        history["train_residual"].append(
            train_metrics["residual"]
        )

        if val_metrics is not None:

            history["val_total"].append(
                val_metrics["total"]
            )

            history["val_duration"].append(
                val_metrics["duration"]
            )

            history["val_final_l1"].append(
                val_metrics["final_l1"]
            )

            history["val_final_mse"].append(
                val_metrics["final_mse"]
            )

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:4d}/{EPOCHS} | "
                f"train={train_metrics['total']:.6f} | "
                f"duration={train_metrics['duration']:.6f} | "
                f"coarse={train_metrics['coarse_l1']:.6f} | "
                f"final={train_metrics['final_l1']:.6f} | "
                f"lr={current_lr:.2e}"
            )

            if val_metrics is not None:

                print(
                    f"                    "
                    f"val={val_metrics['total']:.6f} | "
                    f"val_final={val_metrics['final_l1']:.6f}"
                )

        # ---------------------------------------------------------------
        # Best checkpoint
        # ---------------------------------------------------------------

        if monitored_loss < best_loss:

            best_loss = monitored_loss
            best_epoch = epoch
            patience_counter = 0

            path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=best_loss,
                vocabulary=vocabulary,
                n_mels=n_mels,
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print()
            print(
                f"Early stopping at epoch {epoch}."
            )

            break

    print()
    print(f"Best monitored loss: {best_loss:.6f}")
    print(f"Best epoch: {best_epoch}")

    checkpoint_path = (
        OUTPUT_DIR /
        "improved_acoustic_model_best.pt"
    )

    if checkpoint_path.exists():

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            f"Best checkpoint loaded: "
            f"{checkpoint_path}"
        )

    return history


# ============================================================================
# TRAINING PLOT
# ============================================================================

def save_training_plot(history):

    plt.figure(figsize=(12, 7))

    epochs = np.arange(
        1,
        len(history["train_total"]) + 1,
    )

    plt.plot(
        epochs,
        history["train_total"],
        label="Train total",
    )

    if history["val_total"]:

        plt.plot(
            epochs,
            history["val_total"],
            label="Validation total",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Lesson 59 — Improved Acoustic Model Training")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    path = OUTPUT_DIR / "training_loss.png"

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Training plot saved: {path}"
    )


# ============================================================================
# MEL VISUALIZATION
# ============================================================================

def save_mel_image(
    mel,
    title,
    filename,
):

    mel = np.asarray(mel)

    plt.figure(figsize=(14, 5))

    plt.imshow(
        mel.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    plt.colorbar(label="Normalized Mel value")

    plt.xlabel("Time frame")
    plt.ylabel("Mel bin")

    plt.title(title)

    plt.tight_layout()

    path = OUTPUT_DIR / filename

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path


# ============================================================================
# DURATION VISUALIZATION
# ============================================================================

def save_duration_comparison(
    phonemes,
    actual,
    predicted,
):

    x = np.arange(
        len(phonemes)
    )

    width = 0.38

    plt.figure(figsize=(15, 6))

    plt.bar(
        x - width / 2,
        actual,
        width,
        label="Actual",
    )

    plt.bar(
        x + width / 2,
        predicted,
        width,
        label="Predicted",
    )

    plt.xticks(
        x,
        phonemes,
        rotation=90,
    )

    plt.xlabel("Phoneme")
    plt.ylabel("Duration (frames)")

    plt.title(
        "Lesson 59 — Duration Prediction Comparison"
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    path = OUTPUT_DIR / "duration_comparison.png"

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path


# ============================================================================
# EVALUATION HELPER
# ============================================================================

def prepare_single_record(
    record,
    vocabulary,
):

    phoneme_ids = torch.tensor(
        [
            vocabulary[p]
            for p in record["phonemes"]
        ],
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    durations = torch.tensor(
        record["durations"],
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    padding_mask = torch.zeros(
        (
            1,
            phoneme_ids.shape[1],
        ),
        dtype=torch.bool,
        device=DEVICE,
    )

    target_mel = torch.tensor(
        record["mel"],
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    return (
        phoneme_ids,
        durations,
        padding_mask,
        target_mel,
    )


# ============================================================================
# PREDICTED DURATION SAFETY
# ============================================================================

def limit_total_predicted_duration(
    durations,
    max_frames=MAX_SYNTHESIS_FRAMES,
):

    durations = durations.clone()

    total = int(
        durations.sum().item()
    )

    if total <= max_frames:
        return durations

    active = durations > 0

    count = int(
        active.sum().item()
    )

    if count == 0:
        return durations

    if max_frames < count:
        # Cannot give every phoneme at least one frame.
        # Keep the first max_frames active phonemes.
        output = torch.zeros_like(durations)

        active_indices = torch.where(active)[0]

        for index in active_indices[:max_frames]:
            output[index] = 1

        return output

    # Reserve one frame per active phoneme.
    base = torch.where(
        active,
        torch.ones_like(durations),
        torch.zeros_like(durations),
    )

    remaining = max_frames - count

    extra = torch.clamp(
        durations - 1,
        min=0,
    )

    extra_sum = int(
        extra.sum().item()
    )

    if extra_sum <= 0:
        return base

    scaled = torch.floor(
        extra.float()
        * (remaining / extra_sum)
    ).long()

    result = base + scaled

    # Distribute any remaining frames.
    difference = max_frames - int(
        result.sum().item()
    )

    if difference > 0:

        indices = torch.where(
            active
        )[0]

        for index in indices:

            if difference <= 0:
                break

            result[index] += 1
            difference -= 1

    return result


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_example(
    model,
    record,
    vocabulary,
):

    print()
    print("=" * 72)
    print("STEP 5 — ACOUSTIC RECONSTRUCTION")
    print("=" * 72)

    model.eval()

    (
        phoneme_ids,
        actual_durations,
        padding_mask,
        target_mel,
    ) = prepare_single_record(
        record,
        vocabulary,
    )

    # ========================================================================
    # A. TEACHER-FORCED RECONSTRUCTION
    # ========================================================================

    teacher_outputs = model(
        phoneme_ids=phoneme_ids,
        phoneme_padding_mask=padding_mask,
        ground_truth_durations=actual_durations,
        use_predicted_durations=False,
    )

    teacher_coarse = (
        teacher_outputs["coarse_mel"][0]
        .cpu()
        .numpy()
    )

    teacher_final = (
        teacher_outputs["final_mel"][0]
        .cpu()
        .numpy()
    )

    teacher_residual = (
        teacher_outputs["residual"][0]
        .cpu()
        .numpy()
    )

    target = (
        target_mel[0]
        .cpu()
        .numpy()
    )

    teacher_common = min(
        target.shape[0],
        teacher_final.shape[0],
    )

    teacher_target = target[
        :teacher_common
    ]

    teacher_coarse_common = teacher_coarse[
        :teacher_common
    ]

    teacher_final_common = teacher_final[
        :teacher_common
    ]

    teacher_residual_common = teacher_residual[
        :teacher_common
    ]

    teacher_mae = np.mean(
        np.abs(
            teacher_final_common
            - teacher_target
        )
    )

    teacher_coarse_mae = np.mean(
        np.abs(
            teacher_coarse_common
            - teacher_target
        )
    )

    print(
        f"Teacher-forced coarse Mel MAE: "
        f"{teacher_coarse_mae:.6f}"
    )

    print(
        f"Teacher-forced final Mel MAE:  "
        f"{teacher_mae:.6f}"
    )

    # ========================================================================
    # B. PREDICTED-DURATION INFERENCE
    # ========================================================================

    inference_outputs = model(
        phoneme_ids=phoneme_ids,
        phoneme_padding_mask=padding_mask,
        ground_truth_durations=None,
        use_predicted_durations=True,
    )

    predicted_durations = (
        inference_outputs[
            "predicted_durations"
        ][0]
        .cpu()
        .numpy()
    )

    predicted_durations_tensor = torch.tensor(
        predicted_durations,
        dtype=torch.long,
    )

    predicted_durations_tensor = (
        limit_total_predicted_duration(
            predicted_durations_tensor,
            MAX_SYNTHESIS_FRAMES,
        )
    )

    # If duration limiting was necessary, run the acoustic decoder again
    # using the limited durations.
    if not torch.equal(
        predicted_durations_tensor,
        torch.tensor(
            predicted_durations,
            dtype=torch.long,
        ),
    ):

        limited_durations = (
            predicted_durations_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        inference_outputs = model(
            phoneme_ids=phoneme_ids,
            phoneme_padding_mask=padding_mask,
            ground_truth_durations=limited_durations,
            use_predicted_durations=False,
        )

        predicted_durations = (
            limited_durations[0]
            .cpu()
            .numpy()
        )

    predicted_mel = (
        inference_outputs[
            "final_mel"
        ][0]
        .cpu()
        .numpy()
    )

    # ========================================================================
    # C. SAVE VISUALIZATIONS
    # ========================================================================

    save_mel_image(
        target,
        "Target Mel Spectrogram",
        "teacher_forced_target_mel.png",
    )

    save_mel_image(
        teacher_coarse_common,
        "Teacher-Forced Coarse Mel Prediction",
        "teacher_forced_coarse_mel.png",
    )

    save_mel_image(
        teacher_final_common,
        "Teacher-Forced Final Mel Prediction",
        "teacher_forced_final_mel.png",
    )

    save_mel_image(
        teacher_residual_common,
        "Post-Net Residual Correction",
        "teacher_forced_residual.png",
    )

    save_mel_image(
        predicted_mel,
        "Predicted-Duration Mel Spectrogram",
        "predicted_duration_mel.png",
    )

    # ========================================================================
    # D. ERROR MAP
    # ========================================================================

    error = np.abs(
        teacher_final_common
        - teacher_target
    )

    save_mel_image(
        error,
        "Teacher-Forced Absolute Mel Reconstruction Error",
        "acoustic_error.png",
    )

    # ========================================================================
    # E. DURATION COMPARISON
    # ========================================================================

    save_duration_comparison(
        record["phonemes"],
        record["durations"],
        predicted_durations,
    )

    # ========================================================================
    # F. PRINT DURATION TABLE
    # ========================================================================

    print()
    print(
        f"{'Phoneme':15s}"
        f"{'Actual':>10s}"
        f"{'Predicted':>12s}"
    )

    print("-" * 40)

    for phoneme, actual, predicted in zip(
        record["phonemes"],
        record["durations"],
        predicted_durations,
    ):

        print(
            f"{phoneme:15s}"
            f"{actual:10d}"
            f"{predicted:12d}"
        )

    print()

    print(
        f"Actual total frames:     "
        f"{sum(record['durations'])}"
    )

    print(
        f"Predicted total frames:  "
        f"{int(predicted_durations.sum())}"
    )

    print(
        f"Teacher-forced MAE:      "
        f"{teacher_mae:.6f}"
    )

    return {
        "teacher_coarse_mae": teacher_coarse_mae,
        "teacher_final_mae": teacher_mae,
        "predicted_durations": predicted_durations,
    }


# ============================================================================
# SAVE VOCABULARY
# ============================================================================

def save_vocabulary(vocabulary):

    path = OUTPUT_DIR / "vocabulary.json"

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            vocabulary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Vocabulary saved: {path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    set_seed()

    ensure_output_dir()

    print("=" * 72)
    print("PHASE 2 — LESSON 59")
    print("IMPROVED ACOUSTIC MODEL TRAINING")
    print("=" * 72)

    print()
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ========================================================================
    # STEP 1 — DATA
    # ========================================================================

    records = load_records()

    # ========================================================================
    # STEP 2 — VOCABULARY
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 2 — BUILD PHONEME VOCABULARY")
    print("=" * 72)

    vocabulary = build_vocabulary(
        records
    )

    print(
        f"Phoneme vocabulary size: "
        f"{len(vocabulary)}"
    )

    save_vocabulary(
        vocabulary
    )

    # ========================================================================
    # STEP 3 — DATASET
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 3 — CREATE DATASET")
    print("=" * 72)

    train_dataset, val_dataset = create_datasets(
        records,
        vocabulary,
    )

    print(
        f"Training samples:   "
        f"{len(train_dataset)}"
    )

    if val_dataset is not None:
        print(
            f"Validation samples: "
            f"{len(val_dataset)}"
        )
    else:
        print(
            "Validation samples: 0"
        )

    train_loader, val_loader = create_loaders(
        train_dataset,
        val_dataset,
    )

    # Determine Mel dimensionality.
    n_mels = records[0]["mel"].shape[1]

    print(
        f"Mel bins: {n_mels}"
    )

    # ========================================================================
    # STEP 4 — MODEL
    # ========================================================================

    print()
    print("=" * 72)
    print("MODEL CONFIGURATION")
    print("=" * 72)

    model = ImprovedFastSpeechAcousticModel(
        vocab_size=len(vocabulary),
        n_mels=n_mels,
        embed_dim=EMBED_DIM,
        heads=ATTENTION_HEADS,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        max_len=MAX_SEQUENCE_LENGTH,
    ).to(DEVICE)

    parameters = count_parameters(
        model
    )

    print(
        f"Model parameters: "
        f"{parameters:,}"
    )

    print()
    print(
        "Architecture:"
    )
    print(
        "  Phoneme Embedding"
    )
    print(
        "        ↓"
    )
    print(
        "  Transformer Encoder"
    )
    print(
        "        ↓"
    )
    print(
        "  Duration Predictor"
    )
    print(
        "        ↓"
    )
    print(
        "  Length Regulator"
    )
    print(
        "        ↓"
    )
    print(
        "  Transformer Acoustic Decoder"
    )
    print(
        "        ↓"
    )
    print(
        "  Coarse Mel"
    )
    print(
        "        ↓"
    )
    print(
        "  Convolutional Post-Net"
    )
    print(
        "        ↓"
    )
    print(
        "  Residual Correction"
    )
    print(
        "        ↓"
    )
    print(
        "  Final Mel"
    )

    # ========================================================================
    # TRAIN
    # ========================================================================

    print()
    print(
        "Starting training..."
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocabulary=vocabulary,
        n_mels=n_mels,
    )

    # ========================================================================
    # TRAINING VISUALIZATION
    # ========================================================================

    save_training_plot(
        history
    )

    # ========================================================================
    # FINAL EXAMPLE
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 6 — FINAL EXAMPLE")
    print("=" * 72)

    # Use the first original record so the visualization is deterministic.
    example_record = records[0]

    evaluation = evaluate_example(
        model=model,
        record=example_record,
        vocabulary=vocabulary,
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 72)
    print("LESSON 59 COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Best checkpoint:"
    )

    print(
        OUTPUT_DIR /
        "improved_acoustic_model_best.pt"
    )

    print()
    print(
        "Important interpretation:"
    )

    print(
        "  Teacher-forced reconstruction tests the acoustic model"
    )

    print(
        "  using correct phoneme durations."
    )

    print(
        "  Predicted-duration synthesis additionally depends on"
    )

    print(
        "  the duration predictor."
    )

    if len(records) == 1:

        print()
        print(
            "WARNING:"
        )

        print(
            "Only one utterance was available."
        )

        print(
            "The reported training result demonstrates pipeline"
        )

        print(
            "learning but does not demonstrate generalization."
        )

    print()
    print(
        "Next conceptual step:"
    )

    print(
        "  Add explicit pitch and energy modeling."
    )

    print()
    print(
        "That leads into Lesson 60."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()