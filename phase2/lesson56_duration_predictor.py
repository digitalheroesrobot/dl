"""
File: lesson56_duration_predictor.py

PHASE 2 — LESSON 56
NEURAL DURATION PREDICTOR FROM SCRATCH
======================================================================

PURPOSE
-------
Train a neural network to predict the number of acoustic frames
assigned to each phoneme.

This lesson connects the outputs of:

    LESSON 54:
        Phoneme-to-audio forced alignment.
        Produces phoneme-level duration labels.

    LESSON 55:
        Real speech Mel dataset builder.
        Produces acoustic features and utterance metadata.

DATA FLOW
---------

    Lesson 54 alignment JSON
              |
              v
       Phoneme sequence
              |
              v
       Duration labels
              |
              |
              +--------------------+
                                   |
    Lesson 55 NPZ                  |
         |                         |
         v                         |
    Mel spectrogram                |
    Transcript                     |
    Sample rate                    |
         |                         |
         +------------+------------+
                      |
                      v
             Matched utterance
                      |
                      v
             Duration Predictor
                      |
                      v
             Predicted durations

======================================================================
IMPORTANT DATASET FORMAT
======================================================================

Lesson 55:

    lesson55_outputs/
        features/
            sample.npz

Lesson 54:

    lesson54_outputs/
        sample_alignment.json

Lesson 54 JSON format:

    {
        "phoneme_sequence": ["DH", "IH", "S", ...],

        "aligned_phonemes": [
            {
                "phoneme": "DH",
                "start_frame": 0,
                "end_frame": 3,
                "duration_frames": 3
            },
            ...
        ]
    }

The script matches:

    sample.npz
        <
    sample_alignment.json

ONE-UTTERANCE MODE
------------------

If only one matched utterance is available:

    - Train on that utterance.
    - Monitor training loss.
    - Save the best training checkpoint.
    - Do not report a fake validation score.
    - Warn that this is a demonstration, not generalization.

MULTI-UTTERANCE MODE
--------------------

If at least two utterances are available:

    - Split by utterance.
    - Train on training utterances.
    - Evaluate on separate validation utterances.

======================================================================
NEURAL ARCHITECTURE
======================================================================

    Phoneme IDs
         |
         v
    Embedding Layer
         |
         v
    Bidirectional LSTM
         |
         v
    Linear Layer
         |
         v
    Predicted Log-Duration

Target:

    log_duration = log(1 + duration_frames)

Inference:

    duration_frames = exp(predicted_log_duration) - 1

======================================================================
INSTALLATION
======================================================================

    pip install torch numpy matplotlib

======================================================================
RUN
======================================================================

    python lesson56_duration_predictor.py

With explicit paths:

    python lesson56_duration_predictor.py \
        --data_dir lesson55_outputs \
        --alignment_dir lesson54_outputs \
        --epochs 100

Numerical tests:

    python lesson56_duration_predictor.py --self_test_only

======================================================================
OUTPUT
======================================================================

    lesson56_output/
        duration_predictor_best.pt
        training_history.json
        duration_training.png
        duration_prediction_example.png
        phoneme_duration_predictions.json

======================================================================
"""

import json
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

SEED = 42

DEFAULT_DATA_DIR = "lesson55_outputs"
DEFAULT_ALIGNMENT_DIR = "lesson54_outputs"
DEFAULT_OUTPUT_DIR = "lesson56_output"

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
DROPOUT = 0.15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

BATCH_SIZE = 16
EPOCHS = 100

VALIDATION_RATIO = 0.10
EARLY_STOPPING_PATIENCE = 15

PAD_ID = 0

MAX_DURATION = 1000.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# 2. REPRODUCIBILITY
# ======================================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# 3. DURATION TRANSFORMATIONS
# ======================================================================

def duration_to_log(duration):

    duration = np.asarray(
        duration,
        dtype=np.float32
    )

    if np.any(duration < 0):
        raise ValueError(
            "Duration cannot be negative."
        )

    return np.log1p(duration)


def log_to_duration(log_duration):

    log_duration = np.asarray(
        log_duration,
        dtype=np.float32
    )

    log_duration = np.clip(
        log_duration,
        0.0,
        np.log1p(MAX_DURATION)
    )

    return np.maximum(
        np.expm1(log_duration),
        0.0
    )


# ======================================================================
# 4. PATH AND FILE DISCOVERY
# ======================================================================

def discover_npz_files(data_dir):

    data_dir = Path(data_dir)

    if not data_dir.exists():

        raise FileNotFoundError(
            f"Lesson 55 directory does not exist: {data_dir}"
        )

    files = sorted(
        data_dir.rglob("*.npz")
    )

    return files


def discover_alignment_files(alignment_dir):

    alignment_dir = Path(alignment_dir)

    if not alignment_dir.exists():

        raise FileNotFoundError(
            f"Lesson 54 directory does not exist: "
            f"{alignment_dir}"
        )

    files = sorted(
        alignment_dir.rglob("*alignment.json")
    )

    return files


def get_utterance_id(path, is_alignment=False):

    """
    Convert file names into common utterance identifiers.

    Examples:

        sample.npz
            -> sample

        sample_alignment.json
            -> sample

        recording_001_alignment.json
            -> recording_001
    """

    stem = Path(path).stem

    if is_alignment and stem.endswith("_alignment"):

        stem = stem[:-len("_alignment")]

    return stem


# ======================================================================
# 5. LOAD LESSON 55 MEL DATA
# ======================================================================

def load_mel_metadata(npz_path):

    """
    Load Lesson 55 NPZ metadata.

    The Mel spectrogram itself is not used as a duration target.

    It is used to verify that the alignment belongs to the
    corresponding acoustic utterance.
    """

    with np.load(
        npz_path,
        allow_pickle=False
    ) as data:

        available_keys = list(data.files)

        if "log_mel" not in data.files:

            raise ValueError(
                f"Missing log_mel in {npz_path}. "
                f"Available keys: {available_keys}"
            )

        if "normalized_log_mel" in data.files:

            mel = np.asarray(
                data["normalized_log_mel"]
            )

        else:

            mel = np.asarray(
                data["log_mel"]
            )

        sample_rate = int(
            np.asarray(
                data["sample_rate"]
            ).item()
        )

        hop_length = int(
            np.asarray(
                data["hop_length"]
            ).item()
        )

        transcript = ""

        if "transcript" in data.files:

            transcript_value = data["transcript"]

            transcript = str(
                np.asarray(
                    transcript_value
                ).item()
            )

    if mel.ndim != 2:

        raise ValueError(
            f"Expected 2D Mel spectrogram in {npz_path}, "
            f"got shape {mel.shape}"
        )

    if mel.size == 0:

        raise ValueError(
            f"Empty Mel spectrogram: {npz_path}"
        )

    if not np.all(np.isfinite(mel)):

        raise ValueError(
            f"Mel spectrogram contains NaN or Inf: {npz_path}"
        )

    # Lesson 55 convention:
    # [mel_bins, time_frames]
    #
    # For the normal Lesson 55 output, the last axis is time.
    mel_frames = int(mel.shape[-1])

    return {
        "mel_frames": mel_frames,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "transcript": transcript,
        "mel_shape": list(mel.shape),
    }


# ======================================================================
# 6. LOAD LESSON 54 ALIGNMENT JSON
# ======================================================================

def extract_phoneme_alignment(alignment_path):

    """
    Read phoneme-level alignment records from Lesson 54.

    Expected structure:

        {
            "phoneme_sequence": [...],

            "aligned_phonemes": [
                {
                    "phoneme": "DH",
                    "start_frame": 0,
                    "end_frame": 3,
                    "duration_frames": 3
                }
            ]
        }

    The actual duration target is duration_frames.

    If duration_frames is absent, it is calculated as:

        end_frame - start_frame
    """

    with open(
        alignment_path,
        "r",
        encoding="utf-8"
    ) as f:

        alignment = json.load(f)

    records = alignment.get(
        "aligned_phonemes"
    )

    if not isinstance(records, list):

        raise ValueError(
            f"Missing aligned_phonemes list in {alignment_path}"
        )

    phonemes = []
    durations = []

    for index, record in enumerate(records):

        if not isinstance(record, dict):

            raise ValueError(
                f"Invalid alignment record at index {index}"
            )

        phoneme = record.get(
            "phoneme"
        )

        if phoneme is None:

            raise ValueError(
                f"Missing phoneme at alignment record {index}"
            )

        phoneme = str(phoneme).strip()

        if not phoneme:

            raise ValueError(
                f"Empty phoneme at alignment record {index}"
            )

        duration = record.get(
            "duration_frames"
        )

        if duration is None:

            start_frame = record.get(
                "start_frame"
            )

            end_frame = record.get(
                "end_frame"
            )

            if start_frame is None or end_frame is None:

                raise ValueError(
                    f"Missing duration and frame boundaries "
                    f"at record {index}"
                )

            duration = (
                float(end_frame) - float(start_frame)
            )

        duration = float(duration)

        if not np.isfinite(duration):

            raise ValueError(
                f"Invalid duration at record {index}"
            )

        if duration < 0:

            raise ValueError(
                f"Negative duration at record {index}"
            )

        if duration > MAX_DURATION:

            raise ValueError(
                f"Duration exceeds MAX_DURATION at record {index}: "
                f"{duration}"
            )

        phonemes.append(phoneme)
        durations.append(duration)

    if not phonemes:

        raise ValueError(
            f"No phoneme alignment records in {alignment_path}"
        )

    phonemes = np.asarray(
        phonemes,
        dtype=str
    )

    durations = np.asarray(
        durations,
        dtype=np.float32
    )

    # Verify the top-level phoneme sequence when available.
    top_level_sequence = alignment.get(
        "phoneme_sequence"
    )

    if isinstance(top_level_sequence, list):

        top_level_sequence = [
            str(item).strip()
            for item in top_level_sequence
        ]

        if len(top_level_sequence) != len(phonemes):

            print(
                "\nWARNING: Top-level phoneme_sequence length "
                "does not match aligned_phonemes length."
            )

            print(
                "Using aligned_phonemes as the authoritative sequence."
            )

    return {
        "phonemes": phonemes,
        "durations": durations,
        "alignment_type": alignment.get(
            "alignment_type",
            "unknown"
        ),
        "sample_rate": alignment.get(
            "sample_rate",
            alignment.get("sample_rate_for_mel")
        ),
        "hop_length": alignment.get(
            "hop_length"
        ),
        "frame_rate": alignment.get(
            "frame_rate",
            alignment.get("mel_frame_rate")
        ),
        "source": str(alignment_path),
    }


# ======================================================================
# 7. MATCH LESSON 54 AND LESSON 55 FILES
# ======================================================================

def load_matched_dataset(
    data_dir,
    alignment_dir
):

    """
    Match:

        Lesson 55: sample.npz

        Lesson 54: sample_alignment.json

    Both files must describe the same utterance.
    """

    npz_files = discover_npz_files(
        data_dir
    )

    alignment_files = discover_alignment_files(
        alignment_dir
    )

    print("\nDISCOVERING DATASETS")
    print("=" * 65)

    print(
        f"Lesson 55 NPZ files: {len(npz_files)}"
    )

    print(
        f"Lesson 54 alignment JSON files: "
        f"{len(alignment_files)}"
    )

    npz_lookup = {}

    for path in npz_files:

        utterance_id = get_utterance_id(
            path
        )

        npz_lookup[utterance_id] = path

    alignment_lookup = {}

    for path in alignment_files:

        utterance_id = get_utterance_id(
            path,
            is_alignment=True
        )

        alignment_lookup[utterance_id] = path

    common_ids = sorted(
        set(npz_lookup.keys())
        & set(alignment_lookup.keys())
    )

    missing_alignment = sorted(
        set(npz_lookup.keys())
        - set(alignment_lookup.keys())
    )

    missing_mel = sorted(
        set(alignment_lookup.keys())
        - set(npz_lookup.keys())
    )

    print(
        f"Matched utterances: {len(common_ids)}"
    )

    if missing_alignment:

        print(
            "\nLesson 55 files without Lesson 54 alignment:"
        )

        for item in missing_alignment[:10]:
            print(f"  {item}")

    if missing_mel:

        print(
            "\nLesson 54 alignments without Lesson 55 NPZ:"
        )

        for item in missing_mel[:10]:
            print(f"  {item}")

    if not common_ids:

        raise RuntimeError(
            "\nNo matching utterances found.\n\n"
            "Expected files such as:\n"
            "  lesson55_outputs/features/sample.npz\n"
            "  lesson54_outputs/sample_alignment.json\n\n"
            "Check both directories and filenames."
        )

    examples = []

    for utterance_id in common_ids:

        npz_path = npz_lookup[utterance_id]

        alignment_path = alignment_lookup[utterance_id]

        print(
            f"\nLoading utterance: {utterance_id}"
        )

        try:

            mel_metadata = load_mel_metadata(
                npz_path
            )

            alignment = extract_phoneme_alignment(
                alignment_path
            )

            phonemes = alignment["phonemes"]

            durations = alignment["durations"]

            total_aligned_frames = float(
                durations.sum()
            )

            mel_frames = mel_metadata[
                "mel_frames"
            ]

            frame_difference = abs(
                total_aligned_frames - mel_frames
            )

            print(
                f"  Phonemes: {len(phonemes)}"
            )

            print(
                f"  Aligned frames: {total_aligned_frames:.1f}"
            )

            print(
                f"  Mel frames: {mel_frames}"
            )

            print(
                f"  Sample rate: {mel_metadata['sample_rate']}"
            )

            print(
                f"  Hop length: {mel_metadata['hop_length']}"
            )

            if frame_difference > 2:

                print(
                    "  WARNING: Alignment and Mel frame counts "
                    "differ by more than 2 frames."
                )

                print(
                    "  Check that Lesson 54 and Lesson 55 use "
                    "the same sample rate and hop length."
                )

            else:

                print(
                    "  [PASS] Alignment and Mel frame counts are consistent."
                )

            examples.append(
                {
                    "utterance_id": utterance_id,
                    "phonemes": phonemes,
                    "durations": durations,
                    "mel_metadata": mel_metadata,
                    "alignment_metadata": alignment,
                    "npz_path": str(npz_path),
                    "alignment_path": str(alignment_path),
                }
            )

        except Exception as exc:

            print(
                f"  [SKIP] Could not load utterance: {exc}"
            )

    if not examples:

        raise RuntimeError(
            "No valid matched utterances could be loaded."
        )

    print("\nDATASET LOADING COMPLETE")
    print("=" * 65)

    print(
        f"Valid matched utterances: {len(examples)}"
    )

    return examples


# ======================================================================
# 8. BUILD PHONEME VOCABULARY
# ======================================================================

def build_phoneme_vocabulary(examples):

    """
    Construct a vocabulary from the aligned phoneme symbols.

    ID 0 is reserved for padding.

    Example:

        {
            "<PAD>": 0,
            "AA": 1,
            "AE": 2,
            "AH": 3,
            ...
        }

    The vocabulary is saved in the checkpoint.
    """

    unique_phonemes = sorted(
        {
            phoneme
            for example in examples
            for phoneme in example["phonemes"]
        }
    )

    vocabulary = {
        "<PAD>": PAD_ID
    }

    for phoneme in unique_phonemes:

        vocabulary[phoneme] = (
            len(vocabulary)
        )

    reverse_vocabulary = {
        index: phoneme
        for phoneme, index in vocabulary.items()
    }

    return vocabulary, reverse_vocabulary


def encode_examples(
    examples,
    vocabulary
):

    encoded = []

    for example in examples:

        phoneme_ids = np.asarray(
            [
                vocabulary[phoneme]
                for phoneme in example["phonemes"]
            ],
            dtype=np.int64
        )

        encoded.append(
            {
                **example,
                "phoneme_ids": phoneme_ids,
            }
        )

    return encoded


# ======================================================================
# 9. DATASET STATISTICS
# ======================================================================

def print_dataset_statistics(
    examples,
    vocabulary
):

    all_durations = np.concatenate(
        [
            example["durations"]
            for example in examples
        ]
    )

    all_phonemes = [
        phoneme
        for example in examples
        for phoneme in example["phonemes"]
    ]

    sequence_lengths = [
        len(example["phonemes"])
        for example in examples
    ]

    print("\nDATASET STATISTICS")
    print("=" * 65)

    print(
        f"Number of utterances: {len(examples)}"
    )

    print(
        f"Total phoneme tokens: {len(all_phonemes)}"
    )

    print(
        f"Unique phonemes: {len(vocabulary) - 1}"
    )

    print(
        f"Average phoneme sequence length: "
        f"{np.mean(sequence_lengths):.2f}"
    )

    print(
        f"Average duration: {np.mean(all_durations):.2f} frames"
    )

    print(
        f"Median duration: {np.median(all_durations):.2f} frames"
    )

    print(
        f"Minimum duration: {np.min(all_durations):.2f} frames"
    )

    print(
        f"Maximum duration: {np.max(all_durations):.2f} frames"
    )

    print(
        f"Total aligned frames: {np.sum(all_durations):.1f}"
    )

    print("=" * 65)


# ======================================================================
# 10. TRAIN / VALIDATION SPLIT
# ======================================================================

def split_dataset(examples):

    """
    Split by utterance, not by phoneme.

    This prevents phonemes from the same utterance appearing
    in both training and validation datasets.
    """

    if len(examples) < 2:

        print(
            "\nWARNING: Only one matched utterance is available."
        )

        print(
            "Using ONE-UTTERANCE DEMONSTRATION MODE."
        )

        print(
            "There is no independent validation dataset."
        )

        print(
            "Training loss will be monitored instead."
        )

        return examples, []

    indices = list(
        range(len(examples))
    )

    random.shuffle(indices)

    validation_size = max(
        1,
        int(len(indices) * VALIDATION_RATIO)
    )

    validation_size = min(
        validation_size,
        len(indices) - 1
    )

    validation_indices = indices[
        :validation_size
    ]

    training_indices = indices[
        validation_size:
    ]

    train_examples = [
        examples[i]
        for i in training_indices
    ]

    validation_examples = [
        examples[i]
        for i in validation_indices
    ]

    return train_examples, validation_examples


# ======================================================================
# 11. PYTORCH DATASET
# ======================================================================

class DurationDataset(Dataset):

    def __init__(self, examples):

        self.examples = examples

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        phoneme_ids = torch.tensor(
            example["phoneme_ids"],
            dtype=torch.long
        )

        durations = torch.tensor(
            example["durations"],
            dtype=torch.float32
        )

        log_durations = torch.log1p(
            durations
        )

        return {
            "phoneme_ids": phoneme_ids,
            "durations": durations,
            "log_durations": log_durations,
            "length": len(phoneme_ids),
            "utterance_id": example["utterance_id"],
        }


# ======================================================================
# 12. COLLATE FUNCTION
# ======================================================================

def collate_duration_batch(batch):

    lengths = torch.tensor(
        [
            item["length"]
            for item in batch
        ],
        dtype=torch.long
    )

    max_length = int(
        lengths.max()
    )

    batch_size = len(batch)

    phoneme_ids = torch.full(
        (batch_size, max_length),
        PAD_ID,
        dtype=torch.long
    )

    durations = torch.zeros(
        (batch_size, max_length),
        dtype=torch.float32
    )

    log_durations = torch.zeros(
        (batch_size, max_length),
        dtype=torch.float32
    )

    mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool
    )

    utterance_ids = []

    for i, item in enumerate(batch):

        length = item["length"]

        phoneme_ids[i, :length] = (
            item["phoneme_ids"]
        )

        durations[i, :length] = (
            item["durations"]
        )

        log_durations[i, :length] = (
            item["log_durations"]
        )

        mask[i, :length] = True

        utterance_ids.append(
            item["utterance_id"]
        )

    return {
        "phoneme_ids": phoneme_ids,
        "durations": durations,
        "log_durations": log_durations,
        "lengths": lengths,
        "mask": mask,
        "utterance_ids": utterance_ids,
    }


# ======================================================================
# 13. NEURAL DURATION PREDICTOR
# ======================================================================

class DurationPredictor(nn.Module):

    """
    Architecture:

        Phoneme IDs
             |
             v
        Embedding
             |
             v
        Bidirectional LSTM
             |
             v
        Linear layer
             |
             v
        ReLU
             |
             v
        Linear layer
             |
             v
        Log-duration prediction

    Input:

        [batch_size, sequence_length]

    Output:

        [batch_size, sequence_length]

    One predicted duration per phoneme.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_ID,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.output = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                1
            ),
        )

    def forward(
        self,
        phoneme_ids,
        lengths
    ):

        embedded = self.embedding(
            phoneme_ids
        )

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.lstm(
            packed
        )

        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=phoneme_ids.shape[1],
        )

        lstm_output = self.dropout(
            lstm_output
        )

        predictions = self.output(
            lstm_output
        )

        return predictions.squeeze(-1)


# ======================================================================
# 14. MASKED DURATION LOSS
# ======================================================================

def masked_duration_loss(
    predictions,
    targets,
    mask
):

    squared_error = (
        predictions - targets
    ) ** 2

    masked_error = (
        squared_error * mask.float()
    )

    valid_count = mask.sum().clamp_min(1)

    return (
        masked_error.sum() / valid_count
    )


# ======================================================================
# 15. EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    total_squared_error = 0.0
    total_phonemes = 0

    absolute_errors = []

    for batch in loader:

        phoneme_ids = batch[
            "phoneme_ids"
        ].to(device)

        targets = batch[
            "log_durations"
        ].to(device)

        mask = batch[
            "mask"
        ].to(device)

        lengths = batch[
            "lengths"
        ]

        predictions = model(
            phoneme_ids,
            lengths
        )

        loss = masked_duration_loss(
            predictions,
            targets,
            mask
        )

        valid_count = int(
            mask.sum().item()
        )

        total_squared_error += (
            loss.item() * valid_count
        )

        total_phonemes += valid_count

        predicted_durations = log_to_duration(
            predictions.cpu().numpy()
        )

        target_durations = batch[
            "durations"
        ].numpy()

        valid_mask = batch[
            "mask"
        ].numpy()

        errors = np.abs(
            predicted_durations[valid_mask]
            - target_durations[valid_mask]
        )

        absolute_errors.extend(
            errors.tolist()
        )

    mean_loss = (
        total_squared_error
        / max(total_phonemes, 1)
    )

    mae = (
        float(np.mean(absolute_errors))
        if absolute_errors
        else float("nan")
    )

    return mean_loss, mae


# ======================================================================
# 16. TRAINING
# ======================================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    epochs,
    learning_rate,
    output_dir,
    vocabulary,
    reverse_vocabulary,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    has_validation = (
        validation_loader is not None
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_mae_frames": [],
    }

    best_loss = float("inf")

    patience_counter = 0

    checkpoint_path = (
        output_dir
        / "duration_predictor_best.pt"
    )

    print("\nSTARTING TRAINING")
    print("=" * 65)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Training batches: {len(train_loader)}"
    )

    print(
        f"Validation enabled: {has_validation}"
    )

    print(
        f"Epochs: {epochs}"
    )

    print("=" * 65)

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0.0
        total_phonemes = 0

        for batch in train_loader:

            phoneme_ids = batch[
                "phoneme_ids"
            ].to(DEVICE)

            targets = batch[
                "log_durations"
            ].to(DEVICE)

            mask = batch[
                "mask"
            ].to(DEVICE)

            lengths = batch[
                "lengths"
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                phoneme_ids,
                lengths
            )

            loss = masked_duration_loss(
                predictions,
                targets,
                mask
            )

            if not torch.isfinite(loss):

                raise RuntimeError(
                    "Training loss became NaN or Inf."
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            valid_count = int(
                mask.sum().item()
            )

            total_loss += (
                loss.item() * valid_count
            )

            total_phonemes += valid_count

        train_loss = (
            total_loss
            / max(total_phonemes, 1)
        )

        if has_validation:

            validation_loss, validation_mae = evaluate(
                model,
                validation_loader,
                DEVICE
            )

            monitor_loss = validation_loss

            scheduler.step(
                validation_loss
            )

        else:

            # Demonstration mode:
            # no independent validation dataset exists.
            validation_loss = None
            validation_mae = None

            monitor_loss = train_loss

            scheduler.step(
                train_loss
            )

        history["train_loss"].append(
            float(train_loss)
        )

        history["validation_loss"].append(
            (
                float(validation_loss)
                if validation_loss is not None
                else None
            )
        )

        history["validation_mae_frames"].append(
            (
                float(validation_mae)
                if validation_mae is not None
                else None
            )
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if has_validation:

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {validation_loss:.6f} | "
                f"Val MAE: {validation_mae:.3f} frames | "
                f"LR: {current_lr:.2e}"
            )

        else:

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        if monitor_loss < best_loss:

            best_loss = monitor_loss

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),

                    "vocab_size": model.embedding.num_embeddings,

                    "embedding_dim": EMBEDDING_DIM,

                    "hidden_dim": HIDDEN_DIM,

                    "num_layers": NUM_LSTM_LAYERS,

                    "dropout": DROPOUT,

                    "phoneme_vocabulary": vocabulary,

                    "reverse_vocabulary": reverse_vocabulary,

                    "best_loss": best_loss,

                    "has_validation": has_validation,
                },
                checkpoint_path
            )

        else:

            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    history_path = (
        output_dir / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    print("\nTRAINING FINISHED")
    print("=" * 65)

    print(
        f"Best monitored loss: {best_loss:.6f}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    if not has_validation:

        print(
            "NOTE: This is a training-set demonstration result, "
            "not a validation result."
        )

    return history, checkpoint_path


# ======================================================================
# 17. LOAD BEST CHECKPOINT
# ======================================================================

def load_best_checkpoint(
    checkpoint_path,
    device
):
    """
    Load a checkpoint generated by this training script.

    weights_only=False is used because this locally generated
    checkpoint contains additional vocabulary metadata that may
    include NumPy scalar objects.

    IMPORTANT:
        Only use this setting for checkpoints you trust.
        Never load arbitrary checkpoints from untrusted sources.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = DurationPredictor(
        vocab_size=int(checkpoint["vocab_size"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("\nCheckpoint loaded successfully.")
    print(
        f"Vocabulary size: {checkpoint['vocab_size']}"
    )
    print(
        f"Best monitored loss: {checkpoint['best_loss']:.6f}"
    )

    return model, checkpoint


# ======================================================================
# 18. DURATION INFERENCE
# ======================================================================

@torch.no_grad()
def predict_durations(
    model,
    phoneme_ids,
    device=DEVICE
):

    phoneme_ids = np.asarray(
        phoneme_ids,
        dtype=np.int64
    )

    if phoneme_ids.ndim != 1:

        raise ValueError(
            "phoneme_ids must be a one-dimensional array."
        )

    if len(phoneme_ids) == 0:

        raise ValueError(
            "Cannot predict an empty phoneme sequence."
        )

    if np.any(phoneme_ids <= PAD_ID):

        raise ValueError(
            "Input phoneme IDs must be greater than PAD_ID."
        )

    input_tensor = torch.tensor(
        phoneme_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    lengths = torch.tensor(
        [len(phoneme_ids)],
        dtype=torch.long
    )

    model.eval()

    predicted_log = model(
        input_tensor,
        lengths
    )

    predicted_frames = log_to_duration(
        predicted_log.squeeze(0).cpu().numpy()
    )

    predicted_frames = np.rint(
        predicted_frames
    ).astype(np.int64)

    predicted_frames = np.maximum(
        predicted_frames,
        0
    )

    return predicted_frames


# ======================================================================
# 19. PLOT TRAINING HISTORY
# ======================================================================

def plot_training_history(
    history,
    output_dir
):

    plt.figure(figsize=(10, 5))

    epochs = np.arange(
        1,
        len(history["train_loss"]) + 1
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training loss"
    )

    validation_losses = history[
        "validation_loss"
    ]

    valid_epochs = [
        i + 1
        for i, value in enumerate(validation_losses)
        if value is not None
    ]

    valid_losses = [
        value
        for value in validation_losses
        if value is not None
    ]

    if valid_losses:

        plt.plot(
            valid_epochs,
            valid_losses,
            label="Validation loss"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Log-duration MSE")

    plt.title(
        "Lesson 56 — Duration Predictor Training"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    path = (
        Path(output_dir)
        / "duration_training.png"
    )

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    print(
        f"Training plot saved: {path}"
    )


# ======================================================================
# 20. PLOT DURATION COMPARISON
# ======================================================================

def plot_duration_comparison(
    target,
    predicted,
    output_dir
):

    positions = np.arange(
        len(target)
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        positions,
        target,
        marker="o",
        label="Ground truth"
    )

    plt.plot(
        positions,
        predicted,
        marker="x",
        label="Predicted"
    )

    plt.xlabel("Phoneme position")
    plt.ylabel("Duration (frames)")

    plt.title(
        "Ground Truth vs Predicted Phoneme Durations"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    path = (
        Path(output_dir)
        / "duration_prediction_example.png"
    )

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    print(
        f"Prediction plot saved: {path}"
    )


# ======================================================================
# 21. SAVE PREDICTION EXAMPLE
# ======================================================================

def save_prediction_example(
    model,
    example,
    vocabulary,
    reverse_vocabulary,
    output_dir
):

    phoneme_ids = example[
        "phoneme_ids"
    ]

    target = example[
        "durations"
    ]

    predicted = predict_durations(
        model,
        phoneme_ids
    )

    phoneme_names = [
        reverse_vocabulary[int(index)]
        for index in phoneme_ids
    ]

    results = {
        "utterance_id": example["utterance_id"],

        "phonemes": phoneme_names,

        "phoneme_ids": phoneme_ids.tolist(),

        "ground_truth_durations": target.tolist(),

        "predicted_durations": predicted.tolist(),

        "ground_truth_total_frames": float(
            np.sum(target)
        ),

        "predicted_total_frames": int(
            np.sum(predicted)
        ),

        "sample_rate": example["mel_metadata"]["sample_rate"],

        "hop_length": example["mel_metadata"]["hop_length"],
    }

    output_path = (
        Path(output_dir)
        / "phoneme_duration_predictions.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    plot_duration_comparison(
        target,
        predicted,
        output_dir
    )

    print("\nDURATION PREDICTION EXAMPLE")
    print("=" * 75)

    for i, phoneme in enumerate(phoneme_names):

        print(
            f"{phoneme:10s} | "
            f"ID: {phoneme_ids[i]:4d} | "
            f"Target: {target[i]:7.1f} frames | "
            f"Predicted: {predicted[i]:5d} frames"
        )

    print("-" * 75)

    print(
        f"Ground truth total: {np.sum(target):.1f} frames"
    )

    print(
        f"Predicted total:    {np.sum(predicted)} frames"
    )

    print(
        f"Prediction results saved: {output_path}"
    )

    print("=" * 75)


# ======================================================================
# 22. NUMERICAL TESTS
# ======================================================================

def run_tests():

    print("\nRUNNING NUMERICAL TESTS")
    print("=" * 65)

    # --------------------------------------------------------------
    # TEST 1 — Log-duration conversion
    # --------------------------------------------------------------

    durations = np.array(
        [0, 1, 2, 5, 10, 20],
        dtype=np.float32
    )

    log_durations = duration_to_log(
        durations
    )

    recovered = log_to_duration(
        log_durations
    )

    assert np.allclose(
        recovered,
        durations,
        atol=1e-4
    )

    print(
        "  [PASS] Log-duration transformation"
    )

    # --------------------------------------------------------------
    # TEST 2 — Alignment JSON parsing
    # --------------------------------------------------------------

    import tempfile

    test_alignment = {
        "phoneme_sequence": [
            "DH",
            "IH",
            "S"
        ],

        "aligned_phonemes": [
            {
                "phoneme": "DH",
                "start_frame": 0,
                "end_frame": 3,
                "duration_frames": 3
            },
            {
                "phoneme": "IH",
                "start_frame": 3,
                "end_frame": 7,
                "duration_frames": 4
            },
            {
                "phoneme": "S",
                "start_frame": 7,
                "end_frame": 10,
                "duration_frames": 3
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:

        test_path = (
            Path(tmp) / "test_alignment.json"
        )

        with open(
            test_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                test_alignment,
                f
            )

        parsed = extract_phoneme_alignment(
            test_path
        )

        assert np.array_equal(
            parsed["phonemes"],
            ["DH", "IH", "S"]
        )

        assert np.array_equal(
            parsed["durations"],
            [3, 4, 3]
        )

    print(
        "  [PASS] Lesson 54 alignment JSON parsing"
    )

    # --------------------------------------------------------------
    # TEST 3 — Vocabulary
    # --------------------------------------------------------------

    test_examples = [
        {
            "phonemes": np.array(
                ["DH", "IH", "S"]
            )
        },
        {
            "phonemes": np.array(
                ["S", "AE", "M"]
            )
        },
    ]

    vocabulary, reverse = build_phoneme_vocabulary(
        test_examples
    )

    assert vocabulary["<PAD>"] == PAD_ID

    assert vocabulary["DH"] > PAD_ID

    assert reverse[vocabulary["DH"]] == "DH"

    print(
        "  [PASS] Phoneme vocabulary construction"
    )

    # --------------------------------------------------------------
    # TEST 4 — Batch padding
    # --------------------------------------------------------------

    batch_examples = [
        {
            "phoneme_ids": np.array([2, 5, 8]),
            "durations": np.array(
                [3, 4, 5],
                dtype=np.float32
            ),
            "utterance_id": "a",
        },
        {
            "phoneme_ids": np.array([4, 7]),
            "durations": np.array(
                [2, 6],
                dtype=np.float32
            ),
            "utterance_id": "b",
        },
    ]

    dataset = DurationDataset(
        batch_examples
    )

    batch = collate_duration_batch(
        [
            dataset[0],
            dataset[1],
        ]
    )

    assert batch["phoneme_ids"].shape == (2, 3)

    assert batch["mask"].sum().item() == 5

    assert batch["phoneme_ids"][1, 2].item() == PAD_ID

    print(
        "  [PASS] Variable-length batch padding"
    )

    # --------------------------------------------------------------
    # TEST 5 — Model output
    # --------------------------------------------------------------

    model = DurationPredictor(
        vocab_size=20,
        embedding_dim=32,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )

    output = model(
        batch["phoneme_ids"],
        batch["lengths"]
    )

    assert output.shape == (2, 3)

    assert torch.isfinite(output).all()

    print(
        "  [PASS] Neural duration predictor output"
    )

    # --------------------------------------------------------------
    # TEST 6 — Masked loss
    # --------------------------------------------------------------

    loss = masked_duration_loss(
        output,
        batch["log_durations"],
        batch["mask"]
    )

    assert torch.isfinite(loss)

    assert loss.item() >= 0

    print(
        "  [PASS] Masked duration loss"
    )

    # --------------------------------------------------------------
    # TEST 7 — Backpropagation
    # --------------------------------------------------------------

    model.zero_grad()

    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    print(
        "  [PASS] Backpropagation"
    )

    # --------------------------------------------------------------
    # TEST 8 — Inference
    # --------------------------------------------------------------

    model.eval()

    prediction = predict_durations(
        model,
        np.array([2, 5, 8]),
        device=torch.device("cpu")
    )

    assert prediction.shape == (3,)

    assert np.all(prediction >= 0)

    print(
        "  [PASS] Duration inference"
    )

    print("\nALL NUMERICAL TESTS PASSED.")
    print("=" * 65)


# ======================================================================
# 23. MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 56 — Neural Duration Predictor "
            "using Lesson 54 alignment and Lesson 55 Mel data."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR
    )

    parser.add_argument(
        "--alignment_dir",
        type=str,
        default=DEFAULT_ALIGNMENT_DIR
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE
    )

    parser.add_argument(
        "--self_test_only",
        action="store_true"
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 70)
    print("PHASE 2 — LESSON 56")
    print("NEURAL DURATION PREDICTOR FROM SCRATCH")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Lesson 55 dataset: {args.data_dir}"
    )

    print(
        f"Lesson 54 alignment: {args.alignment_dir}"
    )

    print(
        f"Output: {args.output_dir}"
    )

    print("=" * 70)

    # --------------------------------------------------------------
    # STEP 1 — Numerical tests
    # --------------------------------------------------------------

    run_tests()

    if args.self_test_only:

        print(
            "\nSelf-test completed successfully."
        )

        return

    # --------------------------------------------------------------
    # STEP 2 — Match Lesson 54 and Lesson 55
    # --------------------------------------------------------------

    examples = load_matched_dataset(
        data_dir=args.data_dir,
        alignment_dir=args.alignment_dir,
    )

    # --------------------------------------------------------------
    # STEP 3 — Build phoneme vocabulary
    # --------------------------------------------------------------

    vocabulary, reverse_vocabulary = (
        build_phoneme_vocabulary(
            examples
        )
    )

    examples = encode_examples(
        examples,
        vocabulary
    )

    # --------------------------------------------------------------
    # STEP 4 — Dataset statistics
    # --------------------------------------------------------------

    print_dataset_statistics(
        examples,
        vocabulary
    )

    # --------------------------------------------------------------
    # STEP 5 — Train / validation split
    # --------------------------------------------------------------

    train_examples, validation_examples = (
        split_dataset(
            examples
        )
    )

    print("\nDATASET SPLIT")
    print("=" * 65)

    print(
        f"Training utterances: {len(train_examples)}"
    )

    print(
        f"Validation utterances: {len(validation_examples)}"
    )

    # --------------------------------------------------------------
    # STEP 6 — PyTorch datasets
    # --------------------------------------------------------------

    train_dataset = DurationDataset(
        train_examples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_duration_batch,
        num_workers=0,
    )

    validation_loader = None

    if validation_examples:

        validation_dataset = DurationDataset(
            validation_examples
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_duration_batch,
            num_workers=0,
        )

    # --------------------------------------------------------------
    # STEP 7 — Model
    # --------------------------------------------------------------

    vocab_size = len(vocabulary)

    model = DurationPredictor(
        vocab_size=vocab_size
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nMODEL CONFIGURATION")
    print("=" * 65)

    print(
        f"Vocabulary size: {vocab_size}"
    )

    print(
        f"Embedding dimension: {EMBEDDING_DIM}"
    )

    print(
        f"Hidden dimension: {HIDDEN_DIM}"
    )

    print(
        f"LSTM layers: {NUM_LSTM_LAYERS}"
    )

    print(
        f"Model parameters: {parameter_count:,}"
    )

    print("=" * 65)

    # --------------------------------------------------------------
    # STEP 8 — Training
    # --------------------------------------------------------------

    history, checkpoint_path = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        vocabulary=vocabulary,
        reverse_vocabulary=reverse_vocabulary,
    )

    # --------------------------------------------------------------
    # STEP 9 — Training visualization
    # --------------------------------------------------------------

    plot_training_history(
        history,
        args.output_dir
    )

    # --------------------------------------------------------------
    # STEP 10 — Load best model
    # --------------------------------------------------------------

    best_model, checkpoint = load_best_checkpoint(
        checkpoint_path,
        DEVICE
    )

    # --------------------------------------------------------------
    # STEP 11 — Final evaluation
    # --------------------------------------------------------------

    if validation_loader is not None:

        final_loss, final_mae = evaluate(
            best_model,
            validation_loader,
            DEVICE
        )

        print("\nFINAL VALIDATION RESULTS")
        print("=" * 65)

        print(
            f"Log-duration MSE: {final_loss:.6f}"
        )

        print(
            f"Duration MAE: {final_mae:.3f} frames"
        )

        print("=" * 65)

    else:

        print("\nVALIDATION")
        print("=" * 65)

        print(
            "No independent validation set is available."
        )

        print(
            "This run demonstrates the training pipeline "
            "using one utterance."
        )

        print("=" * 65)

    # --------------------------------------------------------------
    # STEP 12 — Example inference
    # --------------------------------------------------------------

    example = (
        validation_examples[0]
        if validation_examples
        else train_examples[0]
    )

    save_prediction_example(
        model=best_model,
        example=example,
        vocabulary=vocabulary,
        reverse_vocabulary=reverse_vocabulary,
        output_dir=args.output_dir,
    )

    # --------------------------------------------------------------
    # STEP 13 — Completion
    # --------------------------------------------------------------

    print("\nLESSON 56 COMPLETED.")
    print("=" * 65)

    print(
        "Neural phoneme duration predictor trained."
    )

    print(
        f"Matched utterances: {len(examples)}"
    )

    print(
        f"Phoneme vocabulary: {vocab_size - 1}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"Output directory: {args.output_dir}"
    )

    if len(examples) == 1:

        print(
            "\nNEXT DATASET STEP:"
        )

        print(
            "Collect and align more utterances before "
            "using this model as a general-purpose predictor."
        )

    print("=" * 65)


if __name__ == "__main__":

    main()