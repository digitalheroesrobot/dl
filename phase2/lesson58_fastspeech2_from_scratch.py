"""
File: lesson58_fastspeech2_from_scratch.py

PHASE 2 — LESSON 58
FASTSPEECH 2-STYLE ACOUSTIC MODEL FROM SCRATCH
======================================================================

OBJECTIVE
---------
Integrate the components developed in previous lessons into a single
neural acoustic model.

PREVIOUS LESSONS
----------------
Lesson 53: Length Regulator
Lesson 54: Phoneme-to-Audio Forced Alignment
Lesson 55: Real Mel Dataset Builder
Lesson 56: Duration Prediction
Lesson 57: Improved Duration Prediction

LESSON 58
---------
Build an integrated acoustic model that converts phoneme sequences
into Mel spectrograms.

ARCHITECTURE
------------

                    PHONEME IDs
                         |
                         v
                  PHONEME EMBEDDING
                         |
                         v
                 POSITIONAL ENCODING
                         |
                         v
                TRANSFORMER ENCODER
                         |
                         v
                   ENCODER OUTPUT
                         |
                    +----+----+
                    |         |
                    v         v
               DURATION    DURATION
               PREDICTOR   SUPERVISION
                    |         |
                    |         |
                    |    LENGTH REGULATOR
                    |         |
                    +----+----+
                         |
                         v
                 FRAME-LEVEL FEATURES
                         |
                         v
                   MEL DECODER
                         |
                         v
                   MEL PREDICTION
                         |
                         v
                MEL RECONSTRUCTION

TRAINING
--------
Training uses aligned durations from Lesson 54.

The Length Regulator expands phoneme representations according
to ground-truth durations.

The decoder learns to reconstruct the real Mel spectrogram.

The duration predictor learns:

    log(1 + duration_frames)

IMPORTANT DATA ALIGNMENT FIX
----------------------------
Lesson 54 alignment and Lesson 55 Mel extraction may use slightly
different boundary/frame conventions.

For example:

    duration sum = 409
    Mel frames   = 442

This does NOT necessarily mean the alignment is unusable.

If the alignment explicitly covers:

    start_frame = 0
    end_frame   = 409

while the Mel contains 442 frames, the first 409 frames are the
aligned acoustic region.

This implementation therefore:

    1. validates the phoneme boundaries,
    2. computes the aligned interval,
    3. crops the Mel spectrogram to that interval,
    4. preserves the original phoneme durations.

It NEVER proportionally rescales phoneme durations merely to force
a frame-count match.

INFERENCE
---------
At inference time the duration predictor generates durations.

The Length Regulator expands the phoneme representations using
predicted durations.

The decoder generates the Mel spectrogram.

IMPORTANT SCOPE
---------------
This is a FastSpeech 2-style educational acoustic model.

It implements:

    - Phoneme embedding
    - Transformer encoder
    - Duration predictor
    - Length regulator
    - Transformer Mel decoder
    - Mel reconstruction
    - Duration supervision

Pitch and energy predictors are deliberately deferred until they
have appropriate acoustic supervision.

NO PRETRAINED MODELS
--------------------
No pretrained models.
No pretrained TTS models.
No pretrained phonemizers.
No external forced aligner.

DATA INPUTS
-----------
Lesson 54:

    lesson54_outputs/*_alignment.json

Expected alignment fields:

    aligned_phonemes:
        phoneme
        start_frame
        end_frame
        duration_frames

Lesson 55:

    lesson55_outputs/features/*.npz

Expected NPZ fields:

    log_mel
    normalized_log_mel
    sample_rate
    hop_length

MEL FORMAT
----------
Lesson 55 normally stores:

    (n_mels, frames)

This script internally converts it to:

    (frames, n_mels)

======================================================================
WHAT YOU LEARN
======================================================================

STEP 1
------
Load and validate real aligned acoustic data.

STEP 2
------
Build a phoneme vocabulary.

STEP 3
------
Convert phoneme symbols into integer IDs.

STEP 4
------
Encode phoneme sequences with a Transformer.

STEP 5
------
Predict phoneme durations.

STEP 6
------
Expand phoneme representations into frame-level features.

STEP 7
------
Decode frame-level features into Mel spectrograms.

STEP 8
------
Train duration and acoustic reconstruction objectives jointly.

STEP 9
------
Run inference using predicted durations.

STEP 10
-------
Compare the generated Mel spectrogram with the real Mel.

REAL-WORLD CONNECTION
---------------------
This is the core architecture behind modern non-autoregressive
text-to-speech systems.

The major idea is:

    text/phonemes
        ->
    durations
        ->
    frame sequence
        ->
    acoustic representation
        ->
    waveform vocoder

Lesson 58 focuses on the acoustic part of this pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


# ======================================================================
# STEP 1 — CONFIGURATION
# ======================================================================

SEED = 42

EMBED_DIM = 128

ATTENTION_HEADS = 4

ENCODER_LAYERS = 3

DECODER_LAYERS = 3

FF_DIM = 512

DROPOUT = 0.1

BATCH_SIZE = 1

EPOCHS = 300

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 1.0

PATIENCE = 40

MAX_SEQUENCE_LENGTH = 4096

PAD_ID = 0

UNK_ID = 1

DURATION_LOSS_WEIGHT = 1.0

MEL_LOSS_WEIGHT = 1.0

# Maximum duration used during inference for a single phoneme.
MAX_PREDICTED_DURATION = 100

# Safety limit for generated Mel sequences.
MAX_SYNTHESIS_FRAMES = 2000


# ======================================================================
# STEP 2 — REPRODUCIBILITY
# ======================================================================

def set_seed():
    """Set deterministic random seeds."""

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ======================================================================
# STEP 3 — DEVICE
# ======================================================================

def get_device():
    """Select CUDA when available."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ======================================================================
# STEP 4 — POSITIONAL ENCODING
# ======================================================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Input:
        (B, T, C)

    Output:
        (B, T, C)
    """

    def __init__(
        self,
        dimension,
        max_length=MAX_SEQUENCE_LENGTH,
    ):
        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                dimension,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / dimension)
        )

        encoding = torch.zeros(
            1,
            max_length,
            dimension,
        )

        encoding[0, :, 0::2] = torch.sin(
            position * div_term
        )

        odd_dimension_count = encoding[:, :, 1::2].shape[-1]

        if odd_dimension_count > 0:
            encoding[0, :, 1::2] = torch.cos(
                position
                * div_term[:odd_dimension_count]
            )

        self.register_buffer(
            "encoding",
            encoding,
        )

    def forward(self, x):
        length = x.size(1)

        if length > self.encoding.size(1):
            raise ValueError(
                "Sequence exceeds positional encoding limit: "
                f"{length} > {self.encoding.size(1)}"
            )

        return x + self.encoding[:, :length]


# ======================================================================
# STEP 5 — PHONEME EMBEDDING
# ======================================================================

class PhonemeEmbedding(nn.Module):
    """Convert phoneme IDs into continuous representations."""

    def __init__(
        self,
        vocabulary_size,
        embed_dim,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size,
            embed_dim,
            padding_idx=PAD_ID,
        )

        self.position = PositionalEncoding(
            embed_dim
        )

        self.scale = math.sqrt(embed_dim)

    def forward(self, phoneme_ids):
        x = self.embedding(
            phoneme_ids
        ) * self.scale

        return self.position(x)


# ======================================================================
# STEP 6 — TRANSFORMER PHONEME ENCODER
# ======================================================================

class PhonemeEncoder(nn.Module):
    """Encode phoneme-level linguistic information."""

    def __init__(self):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=ENCODER_LAYERS,
        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

    def forward(
        self,
        x,
        padding_mask,
    ):
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        return self.norm(x)


# ======================================================================
# STEP 7 — DURATION PREDICTOR
# ======================================================================

class DurationPredictor(nn.Module):
    """
    Predict log(1 + phoneme duration).

    This implementation keeps the tensor layout explicit:

        Conv1d expects:
            (B, C, T)

        Transformer output:
            (B, T, C)
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            EMBED_DIM,
            EMBED_DIM,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.LayerNorm(
            EMBED_DIM
        )

        self.conv2 = nn.Conv1d(
            EMBED_DIM,
            EMBED_DIM,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.LayerNorm(
            EMBED_DIM
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.output = nn.Linear(
            EMBED_DIM,
            1,
        )

    def forward(self, x):
        # --------------------------------------------------------------
        # Input:
        #     (B, T, C)
        #
        # Conv1d:
        #     (B, C, T)
        # --------------------------------------------------------------

        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = F.relu(x)

        x = x.transpose(1, 2)

        x = self.norm1(x)

        x = self.dropout(x)

        x = x.transpose(1, 2)

        x = self.conv2(x)

        x = F.relu(x)

        x = x.transpose(1, 2)

        x = self.norm2(x)

        x = self.dropout(x)

        prediction = self.output(x)

        return prediction.squeeze(-1)


# ======================================================================
# STEP 8 — LENGTH REGULATOR
# ======================================================================

class LengthRegulator(nn.Module):
    """
    Expand each phoneme representation according to its duration.

    Example:

        phonemes:
            A B C

        durations:
            2 3 1

        output:

            A A B B B C
    """

    def forward(
        self,
        encoder_output,
        durations,
    ):
        batch_outputs = []

        lengths = []

        for batch_index in range(
            encoder_output.size(0)
        ):
            expanded = []

            for phoneme_index in range(
                encoder_output.size(1)
            ):
                duration = int(
                    durations[
                        batch_index,
                        phoneme_index,
                    ].item()
                )

                if duration > 0:
                    vector = encoder_output[
                        batch_index,
                        phoneme_index,
                    ]

                    expanded.append(
                        vector.unsqueeze(0).expand(
                            duration,
                            -1,
                        )
                    )

            if expanded:
                output = torch.cat(
                    expanded,
                    dim=0,
                )

            else:
                output = encoder_output.new_zeros(
                    1,
                    encoder_output.size(-1),
                )

            batch_outputs.append(output)

            lengths.append(
                output.size(0)
            )

        max_length = max(lengths)

        padded_outputs = encoder_output.new_zeros(
            len(batch_outputs),
            max_length,
            encoder_output.size(-1),
        )

        padding_mask = torch.ones(
            len(batch_outputs),
            max_length,
            dtype=torch.bool,
            device=encoder_output.device,
        )

        for index, output in enumerate(
            batch_outputs
        ):
            length = output.size(0)

            padded_outputs[
                index,
                :length
            ] = output

            padding_mask[
                index,
                :length
            ] = False

        return (
            padded_outputs,
            padding_mask,
            lengths,
        )


# ======================================================================
# STEP 9 — MEL DECODER
# ======================================================================

class MelDecoder(nn.Module):
    """
    Convert frame-level hidden representations into Mel bins.
    """

    def __init__(
        self,
        n_mels,
    ):
        super().__init__()

        self.position = PositionalEncoding(
            EMBED_DIM
        )

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerEncoder(
            layer,
            num_layers=DECODER_LAYERS,
        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

        self.projection = nn.Linear(
            EMBED_DIM,
            n_mels,
        )

    def forward(
        self,
        x,
        padding_mask,
    ):
        x = self.position(x)

        x = self.decoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.norm(x)

        return self.projection(x)


# ======================================================================
# STEP 10 — COMPLETE ACOUSTIC MODEL
# ======================================================================

class FastSpeech2Educational(nn.Module):
    """
    Educational FastSpeech 2-style acoustic model.

    Important:
    The model is intentionally smaller/simpler than a production
    FastSpeech 2 implementation.

    Pitch and energy predictors are not included yet.
    """

    def __init__(
        self,
        vocabulary_size,
        n_mels,
    ):
        super().__init__()

        self.embedding = PhonemeEmbedding(
            vocabulary_size,
            EMBED_DIM,
        )

        self.encoder = PhonemeEncoder()

        self.duration_predictor = DurationPredictor()

        self.length_regulator = LengthRegulator()

        self.decoder = MelDecoder(
            n_mels
        )

    def forward(
        self,
        phoneme_ids,
        ground_truth_durations=None,
        phoneme_padding_mask=None,
    ):
        if phoneme_padding_mask is None:
            phoneme_padding_mask = (
                phoneme_ids == PAD_ID
            )

        # --------------------------------------------------------------
        # 1. Phoneme embedding
        # --------------------------------------------------------------

        x = self.embedding(
            phoneme_ids
        )

        # --------------------------------------------------------------
        # 2. Transformer encoder
        # --------------------------------------------------------------

        encoder_output = self.encoder(
            x,
            phoneme_padding_mask,
        )

        # --------------------------------------------------------------
        # 3. Predict log durations
        # --------------------------------------------------------------

        predicted_log_durations = (
            self.duration_predictor(
                encoder_output
            )
        )

        # --------------------------------------------------------------
        # 4. Select durations
        # --------------------------------------------------------------

        if ground_truth_durations is not None:

            durations = ground_truth_durations

        else:

            durations = torch.round(
                torch.expm1(
                    predicted_log_durations.clamp(
                        min=0.0,
                        max=math.log1p(
                            MAX_PREDICTED_DURATION
                        ),
                    )
                )
            ).clamp(
                min=1,
                max=MAX_PREDICTED_DURATION,
            ).long()

            durations = durations.masked_fill(
                phoneme_padding_mask,
                0,
            )

        # --------------------------------------------------------------
        # 5. Length regulator
        # --------------------------------------------------------------

        expanded, frame_padding_mask, lengths = (
            self.length_regulator(
                encoder_output,
                durations,
            )
        )

        # --------------------------------------------------------------
        # Safety check
        # --------------------------------------------------------------

        if expanded.size(1) > MAX_SYNTHESIS_FRAMES:

            raise RuntimeError(
                "Predicted duration sequence is too long: "
                f"{expanded.size(1)} frames. "
                f"Maximum allowed is {MAX_SYNTHESIS_FRAMES}. "
                "The duration predictor may need more training."
            )

        # --------------------------------------------------------------
        # 6. Mel decoder
        # --------------------------------------------------------------

        predicted_mel = self.decoder(
            expanded,
            frame_padding_mask,
        )

        return {
            "predicted_mel": predicted_mel,
            "predicted_log_durations": (
                predicted_log_durations
            ),
            "used_durations": durations,
            "frame_padding_mask": frame_padding_mask,
            "frame_lengths": lengths,
        }


# ======================================================================
# STEP 11 — MEL ORIENTATION
# ======================================================================

def normalize_mel_orientation(
    mel,
    npz_data,
):
    """
    Convert Mel matrix into:

        (frames, n_mels)

    The Lesson 55 metadata is preferred over a simple shape heuristic.
    """

    if mel.ndim != 2:
        raise ValueError(
            f"Invalid Mel shape: {mel.shape}"
        )

    expected_n_mels = None

    if "n_mels" in npz_data:
        try:
            expected_n_mels = int(
                np.asarray(
                    npz_data["n_mels"]
                ).reshape(-1)[0]
            )
        except Exception:
            expected_n_mels = None

    if expected_n_mels is not None:

        if mel.shape[0] == expected_n_mels:
            # (n_mels, frames)
            return mel.T

        if mel.shape[1] == expected_n_mels:
            # Already (frames, n_mels)
            return mel

        raise ValueError(
            f"Mel shape {mel.shape} does not match "
            f"metadata n_mels={expected_n_mels}."
        )

    # Fallback for Lesson 55-style data.
    #
    # Typical:
    #     n_mels = 80
    #
    # and frames > 80.

    if mel.shape[0] <= 128 and mel.shape[1] > mel.shape[0]:
        return mel.T

    if mel.shape[1] <= 128 and mel.shape[0] > mel.shape[1]:
        return mel

    # If both dimensions are ambiguous, assume the standard
    # Lesson 55 orientation.
    return mel.T


# ======================================================================
# STEP 12 — EXTRACT ALIGNMENT INFORMATION
# ======================================================================

def extract_alignment(
    alignment,
):
    """
    Extract:

        phonemes
        durations
        start_frames
        end_frames

    from Lesson 54 alignment JSON.

    Primary schema:

        aligned_phonemes:
            phoneme
            start_frame
            end_frame
            duration_frames
    """

    aligned = alignment.get(
        "aligned_phonemes",
        [],
    )

    phonemes = []

    durations = []

    start_frames = []

    end_frames = []

    for item in aligned:

        phoneme = str(
            item.get(
                "phoneme",
                "",
            )
        ).strip()

        if not phoneme:
            continue

        start = item.get(
            "start_frame",
            None,
        )

        end = item.get(
            "end_frame",
            None,
        )

        duration = item.get(
            "duration_frames",
            None,
        )

        # --------------------------------------------------------------
        # Prefer explicit start/end boundaries.
        # --------------------------------------------------------------

        if start is not None and end is not None:

            start = int(
                round(float(start))
            )

            end = int(
                round(float(end))
            )

            duration_from_bounds = end - start

            if duration_from_bounds <= 0:
                continue

            duration = duration_from_bounds

        elif duration is not None:

            duration = int(
                round(float(duration))
            )

            if duration <= 0:
                continue

            # Boundary information is unavailable.
            start = None
            end = None

        else:

            continue

        phonemes.append(
            phoneme
        )

        durations.append(
            duration
        )

        start_frames.append(
            start
        )

        end_frames.append(
            end
        )

    return (
        phonemes,
        durations,
        start_frames,
        end_frames,
    )


# ======================================================================
# STEP 13 — VALIDATE AND ALIGN MEL/DURATIONS
# ======================================================================

def align_mel_to_durations(
    utterance_id,
    mel,
    phonemes,
    durations,
    start_frames,
    end_frames,
):
    """
    Make the acoustic target consistent with the alignment.

    Critical behavior:

        duration_sum != mel_frames

    does NOT automatically invalidate the sample.

    Example:

        Mel frames:
            442

        aligned interval:
            [0, 409)

        duration sum:
            409

    Result:

        Mel -> Mel[0:409]

    No duration rescaling is performed.
    """

    mel_frames = mel.shape[0]

    duration_sum = int(
        sum(durations)
    )

    if duration_sum <= 0:
        raise ValueError(
            f"{utterance_id}: non-positive duration sum."
        )

    if len(phonemes) != len(durations):
        raise ValueError(
            f"{utterance_id}: phoneme/duration count mismatch."
        )

    # --------------------------------------------------------------
    # Case 1:
    # Exact match.
    # --------------------------------------------------------------

    if duration_sum == mel_frames:

        return mel

    print(
        f"WARNING: {utterance_id}: "
        f"duration sum={duration_sum}, "
        f"Mel frames={mel_frames}"
    )

    # --------------------------------------------------------------
    # Explicit boundaries are the safest solution.
    # --------------------------------------------------------------

    boundaries_available = (
        len(start_frames) == len(phonemes)
        and len(end_frames) == len(phonemes)
        and all(
            start is not None and end is not None
            for start, end in zip(
                start_frames,
                end_frames,
            )
        )
    )

    if boundaries_available:

        previous_end = None

        for index, (
            start,
            end,
        ) in enumerate(
            zip(
                start_frames,
                end_frames,
            )
        ):

            if start < 0:
                raise ValueError(
                    f"{utterance_id}: negative start frame "
                    f"at phoneme {index}."
                )

            if end <= start:
                raise ValueError(
                    f"{utterance_id}: invalid boundary "
                    f"{start}:{end}."
                )

            if end > mel_frames:
                raise ValueError(
                    f"{utterance_id}: alignment boundary "
                    f"{end} exceeds Mel length {mel_frames}."
                )

            if previous_end is not None:

                if start != previous_end:

                    raise ValueError(
                        f"{utterance_id}: alignment contains a gap "
                        f"or overlap between frames "
                        f"{previous_end} and {start}."
                    )

            previous_end = end

        first_start = start_frames[0]

        last_end = end_frames[-1]

        aligned_interval_length = (
            last_end - first_start
        )

        if aligned_interval_length != duration_sum:

            raise ValueError(
                f"{utterance_id}: boundary interval "
                f"{aligned_interval_length} does not equal "
                f"duration sum {duration_sum}."
            )

        if (
            first_start == 0
            and last_end == mel_frames
        ):
            raise ValueError(
                f"{utterance_id}: full Mel has {mel_frames} "
                f"frames but alignment has {duration_sum}."
            )

        print(
            f"INFO: {utterance_id}: "
            f"cropping Mel frames "
            f"[{first_start}:{last_end}] "
            f"from {mel_frames} frames."
        )

        cropped = mel[
            first_start:last_end
        ]

        if cropped.shape[0] != duration_sum:

            raise ValueError(
                f"{utterance_id}: cropped Mel has "
                f"{cropped.shape[0]} frames but "
                f"durations sum to {duration_sum}."
            )

        return cropped

    # --------------------------------------------------------------
    # Case 2:
    # No explicit boundaries.
    #
    # Only allow a conservative prefix crop when the duration sum
    # is smaller than the Mel length.
    #
    # We do NOT rescale durations.
    # --------------------------------------------------------------

    if duration_sum < mel_frames:

        print(
            f"INFO: {utterance_id}: "
            f"explicit boundaries unavailable; "
            f"using first {duration_sum} Mel frames "
            f"because duration sum is shorter than the Mel."
        )

        cropped = mel[
            :duration_sum
        ]

        if cropped.shape[0] == duration_sum:
            return cropped

    raise ValueError(
        f"{utterance_id}: cannot safely reconcile "
        f"duration sum={duration_sum} with "
        f"Mel frames={mel_frames}."
    )


# ======================================================================
# STEP 14 — LOAD REAL DATASET
# ======================================================================

def load_records(
    features_dir,
    alignment_dir,
):
    """
    Load aligned phoneme/Mel pairs.

    This is the main fix for the Lesson 58 failure.

    The old implementation did:

        duration_sum != mel_frames
            -> skip

    The new implementation attempts to identify the aligned interval
    and crop the Mel target appropriately.
    """

    features_dir = Path(
        features_dir
    )

    alignment_dir = Path(
        alignment_dir
    )

    if not features_dir.exists():

        raise FileNotFoundError(
            f"Missing feature directory: {features_dir}"
        )

    if not alignment_dir.exists():

        raise FileNotFoundError(
            f"Missing alignment directory: {alignment_dir}"
        )

    records = []

    alignment_files = sorted(
        alignment_dir.glob(
            "*_alignment.json"
        )
    )

    if not alignment_files:

        raise RuntimeError(
            "No alignment JSON files found."
        )

    print(
        f"Found {len(alignment_files)} alignment files."
    )

    for alignment_path in alignment_files:

        utterance_id = alignment_path.name.replace(
            "_alignment.json",
            "",
        )

        feature_path = (
            features_dir
            / f"{utterance_id}.npz"
        )

        if not feature_path.exists():

            print(
                f"WARNING: Missing feature file for "
                f"{utterance_id}: {feature_path}"
            )

            continue

        try:

            with open(
                alignment_path,
                "r",
                encoding="utf-8",
            ) as file:

                alignment = json.load(file)

            (
                phonemes,
                durations,
                start_frames,
                end_frames,
            ) = extract_alignment(
                alignment
            )

            if not phonemes:

                print(
                    f"WARNING: {utterance_id}: "
                    "no valid phonemes found."
                )

                continue

            with np.load(
                feature_path,
                allow_pickle=False,
            ) as data:

                if "normalized_log_mel" in data:

                    mel = data[
                        "normalized_log_mel"
                    ].astype(
                        np.float32
                    )

                elif "log_mel" in data:

                    mel = data[
                        "log_mel"
                    ].astype(
                        np.float32
                    )

                else:

                    raise ValueError(
                        "NPZ does not contain "
                        "'normalized_log_mel' or 'log_mel'."
                    )

                mel = normalize_mel_orientation(
                    mel,
                    data,
                )

            # ----------------------------------------------------------
            # Reconcile alignment and Mel frame boundaries.
            # ----------------------------------------------------------

            mel = align_mel_to_durations(
                utterance_id,
                mel,
                phonemes,
                durations,
                start_frames,
                end_frames,
            )

            records.append(
                {
                    "id": utterance_id,
                    "phonemes": phonemes,
                    "durations": durations,
                    "mel": mel,
                }
            )

            print(
                f"Loaded {utterance_id}: "
                f"{len(phonemes)} phonemes | "
                f"{mel.shape[0]} frames | "
                f"{mel.shape[1]} Mel bins"
            )

        except Exception as exc:

            print(
                f"WARNING: Skipping {utterance_id}: "
                f"{exc}"
            )

    if not records:

        raise RuntimeError(
            "No valid aligned Mel records found. "
            "Check frame counts, alignment boundaries, "
            "and file names."
        )

    return records


# ======================================================================
# STEP 15 — VOCABULARY
# ======================================================================

def build_vocabulary(
    records,
):
    """Build a phoneme vocabulary."""

    symbols = sorted(
        {
            phoneme
            for record in records
            for phoneme in record["phonemes"]
        }
    )

    vocabulary = {
        "<PAD>": PAD_ID,
        "<UNK>": UNK_ID,
    }

    for symbol in symbols:

        if symbol not in vocabulary:

            vocabulary[symbol] = len(
                vocabulary
            )

    return vocabulary


# ======================================================================
# STEP 16 — DATASET
# ======================================================================

class AcousticDataset(Dataset):
    """PyTorch dataset for phoneme/duration/Mel training."""

    def __init__(
        self,
        records,
        vocabulary,
    ):
        self.records = records

        self.vocabulary = vocabulary

    def __len__(self):
        return len(
            self.records
        )

    def __getitem__(
        self,
        index,
    ):
        record = self.records[index]

        ids = [
            self.vocabulary.get(
                phoneme,
                UNK_ID,
            )
            for phoneme in record["phonemes"]
        ]

        durations = np.asarray(
            record["durations"],
            dtype=np.int64,
        )

        mel = record["mel"]

        return {
            "id": record["id"],

            "phoneme_ids": torch.tensor(
                ids,
                dtype=torch.long,
            ),

            "durations": torch.tensor(
                durations,
                dtype=torch.long,
            ),

            "log_durations": torch.tensor(
                np.log1p(durations),
                dtype=torch.float32,
            ),

            "mel": torch.tensor(
                mel,
                dtype=torch.float32,
            ),
        }


# ======================================================================
# STEP 17 — BATCH COLLATION
# ======================================================================

def collate_batch(
    batch,
):
    """Pad phoneme and Mel sequences for batching."""

    batch_size = len(batch)

    max_phonemes = max(
        len(item["phoneme_ids"])
        for item in batch
    )

    max_frames = max(
        len(item["mel"])
        for item in batch
    )

    n_mels = batch[0]["mel"].shape[-1]

    phoneme_ids = torch.zeros(
        batch_size,
        max_phonemes,
        dtype=torch.long,
    )

    durations = torch.zeros(
        batch_size,
        max_phonemes,
        dtype=torch.long,
    )

    log_durations = torch.zeros(
        batch_size,
        max_phonemes,
        dtype=torch.float32,
    )

    phoneme_mask = torch.ones(
        batch_size,
        max_phonemes,
        dtype=torch.bool,
    )

    mel = torch.zeros(
        batch_size,
        max_frames,
        n_mels,
        dtype=torch.float32,
    )

    mel_mask = torch.ones(
        batch_size,
        max_frames,
        dtype=torch.bool,
    )

    ids = []

    for i, item in enumerate(batch):

        p = len(
            item["phoneme_ids"]
        )

        f = len(
            item["mel"]
        )

        phoneme_ids[
            i,
            :p
        ] = item["phoneme_ids"]

        durations[
            i,
            :p
        ] = item["durations"]

        log_durations[
            i,
            :p
        ] = item["log_durations"]

        phoneme_mask[
            i,
            :p
        ] = False

        mel[
            i,
            :f
        ] = item["mel"]

        mel_mask[
            i,
            :f
        ] = False

        ids.append(
            item["id"]
        )

    return {
        "ids": ids,
        "phoneme_ids": phoneme_ids,
        "durations": durations,
        "log_durations": log_durations,
        "phoneme_mask": phoneme_mask,
        "mel": mel,
        "mel_mask": mel_mask,
    }


# ======================================================================
# STEP 18 — MASKED LOSSES
# ======================================================================

def duration_loss(
    predictions,
    targets,
    mask,
):
    """Masked MSE on log durations."""

    valid = ~mask

    if not valid.any():

        return predictions.sum() * 0.0

    return F.mse_loss(
        predictions[valid],
        targets[valid],
    )


def mel_loss(
    predictions,
    targets,
    mask,
):
    """Masked L1 reconstruction loss for Mel spectrograms."""

    valid = ~mask

    valid = valid.unsqueeze(
        -1
    ).expand_as(
        predictions
    )

    if not valid.any():

        return predictions.sum() * 0.0

    return F.l1_loss(
        predictions[valid],
        targets[valid],
    )


# ======================================================================
# STEP 19 — TRAINING
# ======================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):
    """Run one training epoch."""

    model.train()

    total_loss = 0.0

    total_mel = 0.0

    total_duration = 0.0

    for batch in loader:

        phonemes = batch[
            "phoneme_ids"
        ].to(device)

        durations = batch[
            "durations"
        ].to(device)

        log_durations = batch[
            "log_durations"
        ].to(device)

        phoneme_mask = batch[
            "phoneme_mask"
        ].to(device)

        target_mel = batch[
            "mel"
        ].to(device)

        mel_mask = batch[
            "mel_mask"
        ].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            phonemes,
            ground_truth_durations=durations,
            phoneme_padding_mask=phoneme_mask,
        )

        predicted_mel = output[
            "predicted_mel"
        ]

        predicted_duration = output[
            "predicted_log_durations"
        ]

        # --------------------------------------------------------------
        # Mel reconstruction
        # --------------------------------------------------------------

        loss_mel = mel_loss(
            predicted_mel,
            target_mel,
            mel_mask,
        )

        # --------------------------------------------------------------
        # Duration prediction
        # --------------------------------------------------------------

        loss_duration = duration_loss(
            predicted_duration,
            log_durations,
            phoneme_mask,
        )

        # --------------------------------------------------------------
        # Combined objective
        # --------------------------------------------------------------

        loss = (
            MEL_LOSS_WEIGHT * loss_mel
            +
            DURATION_LOSS_WEIGHT * loss_duration
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        total_loss += loss.item()

        total_mel += loss_mel.item()

        total_duration += loss_duration.item()

    count = max(
        len(loader),
        1,
    )

    return {
        "total": total_loss / count,
        "mel": total_mel / count,
        "duration": total_duration / count,
    }


# ======================================================================
# STEP 20 — VALIDATION
# ======================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    """Evaluate the model using ground-truth durations."""

    model.eval()

    total_loss = 0.0

    total_mel = 0.0

    total_duration = 0.0

    count = 0

    for batch in loader:

        phonemes = batch[
            "phoneme_ids"
        ].to(device)

        durations = batch[
            "durations"
        ].to(device)

        log_durations = batch[
            "log_durations"
        ].to(device)

        phoneme_mask = batch[
            "phoneme_mask"
        ].to(device)

        target_mel = batch[
            "mel"
        ].to(device)

        mel_mask = batch[
            "mel_mask"
        ].to(device)

        output = model(
            phonemes,
            ground_truth_durations=durations,
            phoneme_padding_mask=phoneme_mask,
        )

        loss_mel = mel_loss(
            output["predicted_mel"],
            target_mel,
            mel_mask,
        )

        loss_duration = duration_loss(
            output["predicted_log_durations"],
            log_durations,
            phoneme_mask,
        )

        loss = (
            MEL_LOSS_WEIGHT * loss_mel
            +
            DURATION_LOSS_WEIGHT * loss_duration
        )

        total_loss += loss.item()

        total_mel += loss_mel.item()

        total_duration += loss_duration.item()

        count += 1

    count = max(
        count,
        1,
    )

    return {
        "total": total_loss / count,
        "mel": total_mel / count,
        "duration": total_duration / count,
    }


# ======================================================================
# STEP 21 — INFERENCE
# ======================================================================

@torch.no_grad()
def synthesize_mel(
    model,
    phoneme_ids,
    device,
):
    """
    Generate a Mel spectrogram using predicted durations.
    """

    model.eval()

    phoneme_ids = torch.tensor(
        phoneme_ids,
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    padding_mask = (
        phoneme_ids == PAD_ID
    )

    output = model(
        phoneme_ids,
        ground_truth_durations=None,
        phoneme_padding_mask=padding_mask,
    )

    mel = output[
        "predicted_mel"
    ][0]

    durations = output[
        "used_durations"
    ][0]

    return (
        mel.cpu().numpy(),
        durations.cpu().numpy(),
    )


# ======================================================================
# STEP 22 — VISUALIZATION
# ======================================================================

def save_loss_plot(
    history,
    output_dir,
):
    """Save training/validation loss curves."""

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["train"],
        label="Training",
    )

    if history["validation"]:

        plt.plot(
            history["validation"],
            label="Validation",
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Combined Loss"
    )

    plt.title(
        "FastSpeech 2 Acoustic Model Training"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "training_loss.png",
        dpi=150,
    )

    plt.close()


def save_mel_plot(
    mel,
    filename,
    title,
):
    """Save a Mel spectrogram visualization."""

    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        mel.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    plt.xlabel(
        "Time Frames"
    )

    plt.ylabel(
        "Mel Frequency Bins"
    )

    plt.title(
        title
    )

    plt.colorbar(
        label="Normalized Mel value"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150,
    )

    plt.close()


def save_comparison(
    actual,
    predicted,
    output_dir,
):
    """Save actual, predicted, and error Mel images."""

    save_mel_plot(
        actual,
        output_dir / "actual_mel.png",
        "Actual Mel Spectrogram",
    )

    save_mel_plot(
        predicted,
        output_dir / "predicted_mel.png",
        "Predicted Mel Spectrogram",
    )

    common_frames = min(
        actual.shape[0],
        predicted.shape[0],
    )

    error = np.abs(
        actual[:common_frames]
        -
        predicted[:common_frames]
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        error.T,
        origin="lower",
        aspect="auto",
    )

    plt.xlabel(
        "Time Frames"
    )

    plt.ylabel(
        "Mel Frequency Bins"
    )

    plt.title(
        "Absolute Mel Reconstruction Error"
    )

    plt.colorbar(
        label="Absolute error"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "mel_reconstruction_error.png",
        dpi=150,
    )

    plt.close()


# ======================================================================
# STEP 23 — PRINT DATASET STATISTICS
# ======================================================================

def print_dataset_statistics(
    records,
):
    """Print useful Lesson 58 dataset statistics."""

    total_frames = sum(
        record["mel"].shape[0]
        for record in records
    )

    total_phonemes = sum(
        len(record["phonemes"])
        for record in records
    )

    durations = [
        duration
        for record in records
        for duration in record["durations"]
    ]

    print(
        "\nDataset statistics:"
    )

    print(
        "  Utterances:",
        len(records),
    )

    print(
        "  Total phonemes:",
        total_phonemes,
    )

    print(
        "  Total Mel frames:",
        total_frames,
    )

    if durations:

        print(
            "  Minimum phoneme duration:",
            min(durations),
        )

        print(
            "  Maximum phoneme duration:",
            max(durations),
        )

        print(
            "  Mean phoneme duration:",
            f"{np.mean(durations):.2f}",
        )


# ======================================================================
# STEP 24 — MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features_dir",
        default="lesson55_outputs/features",
    )

    parser.add_argument(
        "--alignment_dir",
        default="lesson54_outputs",
    )

    parser.add_argument(
        "--output_dir",
        default="lesson58_output",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()

    if args.epochs < 1:

        raise ValueError(
            "Epochs must be positive."
        )

    if args.batch_size < 1:

        raise ValueError(
            "Batch size must be positive."
        )

    set_seed()

    device = get_device()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 72
    )

    print(
        "PHASE 2 — LESSON 58"
    )

    print(
        "FASTSPEECH 2-STYLE ACOUSTIC MODEL"
    )

    print(
        "=" * 72
    )

    print(
        "\nDevice:",
        device,
    )

    # ------------------------------------------------------------------
    # STEP 1 — LOAD DATA
    # ------------------------------------------------------------------

    print(
        "\nSTEP 1 — LOAD DATA"
    )

    records = load_records(
        args.features_dir,
        args.alignment_dir,
    )

    print_dataset_statistics(
        records
    )

    # ------------------------------------------------------------------
    # STEP 2 — BUILD VOCABULARY
    # ------------------------------------------------------------------

    print(
        "\nSTEP 2 — BUILD VOCABULARY"
    )

    vocabulary = build_vocabulary(
        records
    )

    n_mels = records[0]["mel"].shape[-1]

    print(
        "Vocabulary size:",
        len(vocabulary),
    )

    print(
        "Mel bins:",
        n_mels,
    )

    with open(
        output_dir / "vocabulary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            vocabulary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # STEP 3 — DATA SPLIT
    # ------------------------------------------------------------------

    print(
        "\nSTEP 3 — DATA SPLIT"
    )

    indices = list(
        range(len(records))
    )

    random.shuffle(
        indices
    )

    if len(records) >= 2:

        validation_count = max(
            1,
            int(len(records) * 0.1),
        )

        validation_count = min(
            validation_count,
            len(records) - 1,
        )

        validation_indices = set(
            indices[:validation_count]
        )

        train_records = [
            record
            for i, record in enumerate(records)
            if i not in validation_indices
        ]

        validation_records = [
            record
            for i, record in enumerate(records)
            if i in validation_indices
        ]

    else:

        train_records = records

        validation_records = []

        print(
            "WARNING: Only one utterance available."
        )

        print(
            "Validation metrics will be unavailable."
        )

    train_dataset = AcousticDataset(
        train_records,
        vocabulary,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    validation_loader = None

    if validation_records:

        validation_dataset = AcousticDataset(
            validation_records,
            vocabulary,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

    print(
        "Training utterances:",
        len(train_records),
    )

    print(
        "Validation utterances:",
        len(validation_records),
    )

    # ------------------------------------------------------------------
    # STEP 4 — BUILD MODEL
    # ------------------------------------------------------------------

    print(
        "\nSTEP 4 — BUILD FASTSPEECH MODEL"
    )

    model = FastSpeech2Educational(
        vocabulary_size=len(vocabulary),
        n_mels=n_mels,
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Model parameters: {parameters:,}"
    )

    # ------------------------------------------------------------------
    # STEP 5 — OPTIMIZER
    # ------------------------------------------------------------------

    print(
        "\nSTEP 5 — OPTIMIZER"
    )

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
    )

    # ------------------------------------------------------------------
    # STEP 6 — TRAINING
    # ------------------------------------------------------------------

    print(
        "\nSTEP 6 — TRAIN ACOUSTIC MODEL"
    )

    history = {
        "train": [],
        "validation": [],
        "mel": [],
        "duration": [],
    }

    best_loss = float(
        "inf"
    )

    patience_counter = 0

    checkpoint_path = (
        output_dir
        /
        "fastspeech2_best.pt"
    )

    start_time = time.time()

    for epoch in range(
        args.epochs
    ):

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        history["train"].append(
            train_metrics["total"]
        )

        history["mel"].append(
            train_metrics["mel"]
        )

        history["duration"].append(
            train_metrics["duration"]
        )

        if validation_loader is not None:

            validation_metrics = evaluate(
                model,
                validation_loader,
                device,
            )

            monitored_loss = (
                validation_metrics["total"]
            )

            history["validation"].append(
                monitored_loss
            )

        else:

            monitored_loss = (
                train_metrics["total"]
            )

        scheduler.step(
            monitored_loss
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        if epoch == 0 or (epoch + 1) % 5 == 0:

            if validation_loader is not None:

                print(
                    f"Epoch {epoch + 1:03d}/{args.epochs} | "
                    f"Train: {train_metrics['total']:.5f} | "
                    f"Val: {monitored_loss:.5f} | "
                    f"Mel: {train_metrics['mel']:.5f} | "
                    f"Duration: {train_metrics['duration']:.5f} | "
                    f"LR: {current_lr:.2e}"
                )

            else:

                print(
                    f"Epoch {epoch + 1:03d}/{args.epochs} | "
                    f"Train: {train_metrics['total']:.5f} | "
                    f"Mel: {train_metrics['mel']:.5f} | "
                    f"Duration: {train_metrics['duration']:.5f} | "
                    f"LR: {current_lr:.2e}"
                )

        if monitored_loss < best_loss - 1e-6:

            best_loss = monitored_loss

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "vocabulary":
                        vocabulary,

                    "n_mels":
                        n_mels,

                    "embed_dim":
                        EMBED_DIM,

                    "best_loss":
                        best_loss,

                    "history":
                        history,
                },
                checkpoint_path,
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping."
            )

            break

    elapsed = (
        time.time()
        -
        start_time
    )

    print(
        f"\nTraining time: {elapsed:.2f} seconds"
    )

    print(
        f"Best monitored loss: {best_loss:.6f}"
    )

    if validation_loader is None:

        print(
            "NOTE: This is training loss, "
            "not validation loss."
        )

    # ------------------------------------------------------------------
    # STEP 7 — RESTORE BEST MODEL
    # ------------------------------------------------------------------

    print(
        "\nSTEP 7 — RESTORE BEST CHECKPOINT"
    )

    if not checkpoint_path.exists():

        raise RuntimeError(
            "Best checkpoint was not created."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ------------------------------------------------------------------
    # STEP 8 — TRAINING VISUALIZATION
    # ------------------------------------------------------------------

    print(
        "\nSTEP 8 — SAVE TRAINING PLOTS"
    )

    save_loss_plot(
        history,
        output_dir,
    )

    # ------------------------------------------------------------------
    # STEP 9 — INFERENCE
    # ------------------------------------------------------------------

    print(
        "\nSTEP 9 — MEL INFERENCE"
    )

    example = records[0]

    example_ids = [
        vocabulary.get(
            phoneme,
            UNK_ID,
        )
        for phoneme in example["phonemes"]
    ]

    predicted_mel, predicted_durations = (
        synthesize_mel(
            model,
            example_ids,
            device,
        )
    )

    actual_mel = example["mel"]

    print(
        "Example utterance:",
        example["id"],
    )

    print(
        "Actual Mel shape:",
        actual_mel.shape,
    )

    print(
        "Predicted Mel shape:",
        predicted_mel.shape,
    )

    print(
        "\nPhoneme duration predictions:"
    )

    for (
        phoneme,
        actual,
        predicted,
    ) in zip(
        example["phonemes"],
        example["durations"],
        predicted_durations,
    ):

        print(
            f"{phoneme:10s} "
            f"Actual: {actual:4d} "
            f"Predicted: {int(predicted):4d}"
        )

    print(
        "\nActual total frames:",
        int(sum(example["durations"])),
    )

    print(
        "Predicted total frames:",
        int(predicted_durations.sum()),
    )

    # ------------------------------------------------------------------
    # STEP 10 — MEL COMPARISON
    # ------------------------------------------------------------------

    print(
        "\nSTEP 10 — MEL RECONSTRUCTION"
    )

    save_comparison(
        actual_mel,
        predicted_mel,
        output_dir,
    )

    common_frames = min(
        len(actual_mel),
        len(predicted_mel),
    )

    reconstruction_mae = np.mean(
        np.abs(
            actual_mel[:common_frames]
            -
            predicted_mel[:common_frames]
        )
    )

    print(
        f"Mel reconstruction MAE: "
        f"{reconstruction_mae:.6f}"
    )

    print(
        "Common comparison frames:",
        common_frames,
    )

    if (
        len(actual_mel)
        !=
        len(predicted_mel)
    ):

        print(
            "NOTE: Predicted and actual Mel lengths differ "
            "because inference uses predicted durations."
        )

        print(
            "The MAE therefore compares only their common "
            "initial frame region."
        )

    # ------------------------------------------------------------------
    # STEP 11 — SUMMARY
    # ------------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "LESSON 58 — RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        "Utterances:",
        len(records),
    )

    print(
        "Vocabulary:",
        len(vocabulary),
    )

    print(
        "Mel dimensions:",
        n_mels,
    )

    print(
        "Model parameters:",
        f"{parameters:,}",
    )

    print(
        "Best checkpoint:",
        checkpoint_path,
    )

    print(
        "Mel reconstruction MAE:",
        f"{reconstruction_mae:.6f}",
    )

    print(
        "\nOUTPUT FILES"
    )

    for path in sorted(
        output_dir.iterdir()
    ):

        if path.is_file():

            print(
                " ",
                path.name,
            )

    print(
        "\nLESSON 58 COMPLETED."
    )

    print(
        "=" * 72
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()