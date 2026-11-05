"""
File: lesson57_improved_duration_prediction.py

PHASE 2 — LESSON 57
IMPROVED PHONEME DURATION PREDICTION
======================================================================

OBJECTIVE
---------
Improve the duration predictor developed in Lesson 56.

Lesson 56:
    Phoneme IDs
        |
        v
    Embedding
        |
        v
    Bidirectional LSTM
        |
        v
    Duration Regression

Lesson 57:
    Phoneme IDs
        |
        v
    Embedding
        |
        v
    Positional Encoding
        |
        v
    Transformer Encoder
        |
        v
    Duration Regression

WHY IMPROVE IT?
---------------
A BiLSTM processes sequences recurrently.

A Transformer uses self-attention to model relationships between
phonemes across an entire input sequence.

For example:

    PHONEMES:

        k ae t

    DURATIONS:

        4  7  5

The duration of a phoneme can depend on:

    - Its phonetic identity.
    - Its neighboring phonemes.
    - Its position in the word.
    - Its position in the sentence.
    - The surrounding phonetic context.

The Transformer can learn these contextual relationships.

IMPORTANT:
----------
This is an educational duration-prediction model.

It is NOT a complete FastSpeech 2 implementation.

The current model does not yet explicitly predict pitch and energy.

----------------------------------------------------------------------
STEP 1 — DATA
----------------------------------------------------------------------

Lesson 55:

    lesson55_outputs/
        features/
            utterance_001.npz
            utterance_002.npz

Each NPZ contains:

    log_mel
    normalized_log_mel
    sample_rate
    original_sample_rate
    hop_length
    n_fft
    n_mels
    duration
    transcript

Lesson 54:

    lesson54_outputs/
        utterance_001_alignment.json
        utterance_002_alignment.json

Alignment JSON contains:

    aligned_phonemes:
        phoneme
        start_frame
        end_frame
        duration_frames

----------------------------------------------------------------------
STEP 2 — TARGET
----------------------------------------------------------------------

Predict log duration:

    y = log(1 + duration_frames)

Why?

Phoneme durations have a skewed distribution.

Log transformation makes regression more numerically stable.

During inference:

    predicted_duration = exp(predicted_log_duration) - 1

----------------------------------------------------------------------
STEP 3 — ARCHITECTURE
----------------------------------------------------------------------

    Phoneme IDs
         |
         v
    Embedding(128)
         |
         v
    Positional Encoding
         |
         v
    Transformer Encoder
         |
         v
    Linear(128 -> 128)
         |
         v
    ReLU
         |
         v
    Linear(128 -> 1)
         |
         v
    Predicted Log Duration

----------------------------------------------------------------------
STEP 4 — TRAINING
----------------------------------------------------------------------

Loss:

    Masked MSE

Optimizer:

    AdamW

Additional techniques:

    - Gradient clipping
    - Padding masks
    - Deterministic random seed
    - Best checkpoint saving
    - Validation monitoring
    - Early stopping

----------------------------------------------------------------------
STEP 5 — COMPARISON
----------------------------------------------------------------------

Compare:

    1. BiLSTM baseline
    2. Transformer duration predictor

Metrics:

    - Training loss
    - Validation loss
    - Mean absolute duration error
    - Duration prediction visualization

IMPORTANT VALIDATION NOTE:
--------------------------

If there is only one aligned utterance, a separate utterance-level
validation set cannot be constructed.

In that situation, validation metrics are reported as unavailable.
Training-set results must not be presented as validation results.

NO PRETRAINED MODELS
--------------------

This implementation uses:

    - PyTorch
    - NumPy
    - Matplotlib

No pretrained Transformer.
No pretrained phoneme encoder.
No pretrained TTS model.

======================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

SEED = 42

EMBED_DIM = 128

TRANSFORMER_HEADS = 4

TRANSFORMER_LAYERS = 3

TRANSFORMER_FF_DIM = 512

DROPOUT = 0.15

BATCH_SIZE = 16

EPOCHS = 200

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 1.0

PATIENCE = 25

MIN_DELTA = 1e-5

VALIDATION_RATIO = 0.10

PAD_ID = 0

MAX_POSITION_LENGTH = 4096


# ======================================================================
# 2. REPRODUCIBILITY
# ======================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ======================================================================
# 3. DEVICE
# ======================================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ======================================================================
# 4. PHONEME VOCABULARY
# ======================================================================

def build_vocabulary(records):

    phonemes = set()

    for record in records:

        for phoneme in record["phonemes"]:

            phonemes.add(phoneme)

    vocabulary = {

        "<PAD>": 0,

        "<UNK>": 1,

    }

    for phoneme in sorted(phonemes):

        if phoneme not in vocabulary:

            vocabulary[phoneme] = len(vocabulary)

    return vocabulary


# ======================================================================
# 5. LOAD ALIGNMENT DATA
# ======================================================================

def load_alignment_records(
    features_dir,
    alignment_dir,
):

    features_dir = Path(features_dir)

    alignment_dir = Path(alignment_dir)

    if not features_dir.exists():

        raise FileNotFoundError(
            f"Feature directory not found: {features_dir}"
        )

    if not alignment_dir.exists():

        raise FileNotFoundError(
            f"Alignment directory not found: {alignment_dir}"
        )

    records = []

    alignment_files = sorted(
        alignment_dir.glob("*_alignment.json")
    )

    if not alignment_files:

        raise RuntimeError(
            f"No alignment JSON files found in {alignment_dir}"
        )

    print("\nLoading Lesson 54/55 data...")

    for alignment_path in alignment_files:

        utterance_id = alignment_path.name.replace(
            "_alignment.json",
            "",
        )

        feature_path = (
            features_dir /
            f"{utterance_id}.npz"
        )

        if not feature_path.exists():

            print(
                f"Skipping {utterance_id}: "
                "matching Mel feature file not found."
            )

            continue

        with open(
            alignment_path,
            "r",
            encoding="utf-8",
        ) as file:

            alignment = json.load(file)

        aligned_phonemes = alignment.get(
            "aligned_phonemes",
            [],
        )

        phonemes = []

        durations = []

        for item in aligned_phonemes:

            phoneme = str(
                item.get("phoneme", "")
            ).strip()

            duration = item.get(
                "duration_frames",
                None,
            )

            if duration is None:

                start = item.get(
                    "start_frame",
                    0,
                )

                end = item.get(
                    "end_frame",
                    0,
                )

                duration = end - start

            duration = float(duration)

            if not phoneme:

                continue

            if not np.isfinite(duration):

                continue

            if duration <= 0:

                continue

            phonemes.append(phoneme)

            durations.append(duration)

        if not phonemes:

            print(
                f"Skipping {utterance_id}: "
                "no valid phoneme durations."
            )

            continue

        try:

            with np.load(
                feature_path,
                allow_pickle=False,
            ) as features:

                if "log_mel" not in features:

                    raise KeyError(
                        "NPZ does not contain log_mel."
                    )

                mel = features["log_mel"]

                if mel.ndim != 2:

                    raise ValueError(
                        f"Unexpected Mel shape: {mel.shape}"
                    )

                mel_frames = int(
                    mel.shape[-1]
                )

                sample_rate = int(
                    features["sample_rate"]
                )

                hop_length = int(
                    features["hop_length"]
                )

        except Exception as error:

            print(
                f"Skipping {utterance_id}: "
                f"feature error: {error}"
            )

            continue

        records.append({

            "id": utterance_id,

            "phonemes": phonemes,

            "durations": np.asarray(
                durations,
                dtype=np.float32,
            ),

            "mel_frames": mel_frames,

            "sample_rate": sample_rate,

            "hop_length": hop_length,

            "feature_path": str(feature_path),

            "alignment_path": str(alignment_path),

        })

        print(
            f"  {utterance_id}: "
            f"{len(phonemes)} phonemes | "
            f"{mel_frames} Mel frames"
        )

    if not records:

        raise RuntimeError(
            "No valid paired Lesson 54/55 records found."
        )

    print(
        f"\nValid utterances: {len(records)}"
    )

    return records


# ======================================================================
# 6. TRAIN / VALIDATION SPLIT
# ======================================================================

def split_records(
    records,
    validation_ratio,
):

    indices = list(
        range(len(records))
    )

    random.Random(SEED).shuffle(
        indices
    )

    if len(records) < 2:

        print(
            "\nWARNING: Only one utterance is available."
        )

        print(
            "Validation metrics will be unavailable."
        )

        return records, []

    validation_count = max(
        1,
        int(
            len(records) * validation_ratio
        ),
    )

    validation_count = min(
        validation_count,
        len(records) - 1,
    )

    validation_indices = set(
        indices[:validation_count]
    )

    train_records = []

    validation_records = []

    for index, record in enumerate(records):

        if index in validation_indices:

            validation_records.append(record)

        else:

            train_records.append(record)

    print(
        f"\nTraining utterances: {len(train_records)}"
    )

    print(
        f"Validation utterances: {len(validation_records)}"
    )

    return train_records, validation_records


# ======================================================================
# 7. PYTORCH DATASET
# ======================================================================

class DurationDataset(Dataset):

    def __init__(
        self,
        records,
        vocabulary,
    ):

        self.records = records

        self.vocabulary = vocabulary

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        phoneme_ids = [

            self.vocabulary.get(
                phoneme,
                1,
            )

            for phoneme in record["phonemes"]

        ]

        durations = record["durations"]

        # Log-duration target.

        log_durations = np.log1p(
            durations
        )

        return {

            "id": record["id"],

            "phoneme_ids": torch.tensor(
                phoneme_ids,
                dtype=torch.long,
            ),

            "durations": torch.tensor(
                durations,
                dtype=torch.float32,
            ),

            "log_durations": torch.tensor(
                log_durations,
                dtype=torch.float32,
            ),

        }


# ======================================================================
# 8. VARIABLE-LENGTH COLLATE
# ======================================================================

def collate_batch(batch):

    batch_size = len(batch)

    max_length = max(

        len(item["phoneme_ids"])

        for item in batch

    )

    phoneme_ids = torch.zeros(

        batch_size,

        max_length,

        dtype=torch.long,

    )

    log_durations = torch.zeros(

        batch_size,

        max_length,

        dtype=torch.float32,

    )

    durations = torch.zeros(

        batch_size,

        max_length,

        dtype=torch.float32,

    )

    padding_mask = torch.ones(

        batch_size,

        max_length,

        dtype=torch.bool,

    )

    ids = []

    for index, item in enumerate(batch):

        length = len(
            item["phoneme_ids"]
        )

        phoneme_ids[index, :length] = (
            item["phoneme_ids"]
        )

        log_durations[index, :length] = (
            item["log_durations"]
        )

        durations[index, :length] = (
            item["durations"]
        )

        padding_mask[index, :length] = False

        ids.append(
            item["id"]
        )

    return {

        "ids": ids,

        "phoneme_ids": phoneme_ids,

        "log_durations": log_durations,

        "durations": durations,

        "padding_mask": padding_mask,

    }


# ======================================================================
# 9. SINUSOIDAL POSITIONAL ENCODING
# ======================================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        dimension,
        max_length=MAX_POSITION_LENGTH,
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

            * (
                -math.log(10000.0)
                / dimension
            )

        )

        encoding = torch.zeros(

            1,

            max_length,

            dimension,

        )

        encoding[0, :, 0::2] = torch.sin(
            position * div_term
        )

        encoding[0, :, 1::2] = torch.cos(
            position * div_term[:encoding[:, :, 1::2].shape[-1]]
        )

        self.register_buffer(
            "encoding",
            encoding,
        )

    def forward(self, x):

        length = x.size(1)

        if length > self.encoding.size(1):

            raise ValueError(
                f"Sequence length {length} exceeds "
                f"maximum positional encoding length "
                f"{self.encoding.size(1)}."
            )

        return (
            x + self.encoding[:, :length]
        )


# ======================================================================
# 10. LESSON 56 BASELINE — BiLSTM
# ======================================================================

class BiLSTMDurationPredictor(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embed_dim=EMBED_DIM,
    ):

        super().__init__()

        self.embedding = nn.Embedding(

            vocabulary_size,

            embed_dim,

            padding_idx=PAD_ID,

        )

        self.lstm = nn.LSTM(

            input_size=embed_dim,

            hidden_size=128,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.15,

        )

        self.regressor = nn.Sequential(

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                1,
            ),

        )

    def forward(
        self,
        phoneme_ids,
        padding_mask=None,
    ):

        x = self.embedding(
            phoneme_ids
        )

        x, _ = self.lstm(x)

        return self.regressor(
            x
        ).squeeze(-1)


# ======================================================================
# 11. LESSON 57 — TRANSFORMER DURATION PREDICTOR
# ======================================================================

class TransformerDurationPredictor(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embed_dim=EMBED_DIM,
        heads=TRANSFORMER_HEADS,
        layers=TRANSFORMER_LAYERS,
        ff_dim=TRANSFORMER_FF_DIM,
        dropout=DROPOUT,
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

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.regressor = nn.Sequential(

            nn.Linear(
                embed_dim,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                1,
            ),

        )

    def forward(
        self,
        phoneme_ids,
        padding_mask=None,
    ):

        x = self.embedding(
            phoneme_ids
        )

        x = self.position(x)

        x = self.encoder(

            x,

            src_key_padding_mask=padding_mask,

        )

        x = self.norm(x)

        predicted_log_durations = (
            self.regressor(x).squeeze(-1)
        )

        return predicted_log_durations


# ======================================================================
# 12. MASKED LOSS
# ======================================================================

def masked_mse_loss(
    predictions,
    targets,
    padding_mask,
):

    valid_mask = ~padding_mask

    squared_error = (
        predictions - targets
    ) ** 2

    selected_error = squared_error[
        valid_mask
    ]

    return selected_error.mean()


# ======================================================================
# 13. DURATION METRICS
# ======================================================================

def calculate_metrics(
    predictions,
    targets,
    padding_mask,
):

    valid_mask = ~padding_mask

    predicted_log = predictions[
        valid_mask
    ]

    target_log = targets[
        valid_mask
    ]

    predicted_frames = torch.expm1(
        predicted_log
    ).clamp_min(0)

    target_frames = torch.expm1(
        target_log
    ).clamp_min(0)

    mae = torch.mean(
        torch.abs(
            predicted_frames - target_frames
        )
    )

    relative_error = torch.mean(

        torch.abs(
            predicted_frames - target_frames
        )

        /

        target_frames.clamp_min(1.0)

    ) * 100.0

    return {

        "mae_frames": mae.item(),

        "relative_error_percent": (
            relative_error.item()
        ),

    }


# ======================================================================
# 14. TRAIN / VALIDATION EPOCH
# ======================================================================

def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training,
):

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0

    total_mae = 0.0

    total_relative_error = 0.0

    batches = 0

    for batch in loader:

        phoneme_ids = batch[
            "phoneme_ids"
        ].to(device)

        targets = batch[
            "log_durations"
        ].to(device)

        padding_mask = batch[
            "padding_mask"
        ].to(device)

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            predictions = model(

                phoneme_ids,

                padding_mask,

            )

            loss = masked_mse_loss(

                predictions,

                targets,

                padding_mask,

            )

            if training:

                loss.backward()

                nn.utils.clip_grad_norm_(

                    model.parameters(),

                    GRAD_CLIP,

                )

                optimizer.step()

        metrics = calculate_metrics(

            predictions.detach(),

            targets,

            padding_mask,

        )

        total_loss += loss.item()

        total_mae += metrics[
            "mae_frames"
        ]

        total_relative_error += metrics[
            "relative_error_percent"
        ]

        batches += 1

    divisor = max(
        batches,
        1,
    )

    return {

        "loss": total_loss / divisor,

        "mae_frames": total_mae / divisor,

        "relative_error_percent": (
            total_relative_error / divisor
        ),

    }


# ======================================================================
# 15. MODEL TRAINING
# ======================================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    output_dir,
    model_name,
    vocabulary,
    epochs=EPOCHS,
):

    model = model.to(device)

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=8,

    )

    history = {

        "train_loss": [],

        "validation_loss": [],

        "train_mae": [],

        "validation_mae": [],

    }

    best_loss = float("inf")

    patience_counter = 0

    best_checkpoint = (
        output_dir /
        f"{model_name}_best.pt"
    )

    print(
        f"\nStarting {model_name} training..."
    )

    for epoch in range(epochs):

        start_time = time.time()

        train_metrics = run_epoch(

            model,

            train_loader,

            optimizer,

            device,

            training=True,

        )

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["train_mae"].append(
            train_metrics["mae_frames"]
        )

        if validation_loader is not None:

            validation_metrics = run_epoch(

                model,

                validation_loader,

                optimizer,

                device,

                training=False,

            )

            monitored_loss = (
                validation_metrics["loss"]
            )

            history["validation_loss"].append(
                monitored_loss
            )

            history["validation_mae"].append(
                validation_metrics["mae_frames"]
            )

        else:

            # Do not pretend training loss is validation loss.

            monitored_loss = train_metrics["loss"]

        scheduler.step(
            monitored_loss
        )

        elapsed = time.time() - start_time

        if validation_loader is not None:

            message = (

                f"Epoch {epoch + 1:03d}/{epochs} | "

                f"Train: {train_metrics['loss']:.5f} | "

                f"Val: {monitored_loss:.5f} | "

                f"Train MAE: "
                f"{train_metrics['mae_frames']:.2f} frames | "

                f"Val MAE: "
                f"{validation_metrics['mae_frames']:.2f} frames | "

                f"{elapsed:.2f}s"

            )

        else:

            message = (

                f"Epoch {epoch + 1:03d}/{epochs} | "

                f"Train: {train_metrics['loss']:.5f} | "

                f"Train MAE: "
                f"{train_metrics['mae_frames']:.2f} frames | "

                f"{elapsed:.2f}s"

            )

        if (

            epoch == 0

            or (epoch + 1) % 5 == 0

        ):

            print(message)

        improved = (
            monitored_loss < best_loss - MIN_DELTA
        )

        if improved:

            best_loss = monitored_loss

            patience_counter = 0

            torch.save(

                {

                    "model_state_dict": (
                        model.state_dict()
                    ),

                    "model_name": model_name,

                    "vocabulary": vocabulary,

                    "vocabulary_size": len(vocabulary),

                    "embed_dim": EMBED_DIM,

                    "transformer_heads": (
                        TRANSFORMER_HEADS
                    ),

                    "transformer_layers": (
                        TRANSFORMER_LAYERS
                    ),

                    "best_loss": best_loss,

                    "history": history,

                },

                best_checkpoint,

            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    print(
        f"\nBest monitored loss: {best_loss:.6f}"
    )

    print(
        f"Checkpoint: {best_checkpoint}"
    )

    if validation_loader is None:

        print(
            "NOTE: Best monitored loss is training loss, "
            "not validation loss."
        )

    # Restore best checkpoint.

    checkpoint = torch.load(

        best_checkpoint,

        map_location=device,

        weights_only=False,

    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, history


# ======================================================================
# 16. INFERENCE
# ======================================================================

@torch.no_grad()
def predict_durations(
    model,
    phonemes,
    vocabulary,
    device,
):

    model.eval()

    ids = [

        vocabulary.get(
            phoneme,
            1,
        )

        for phoneme in phonemes

    ]

    input_tensor = torch.tensor(

        ids,

        dtype=torch.long,

    ).unsqueeze(0).to(device)

    padding_mask = torch.zeros(

        1,

        len(ids),

        dtype=torch.bool,

        device=device,

    )

    predicted_log = model(

        input_tensor,

        padding_mask,

    )

    predicted_frames = torch.expm1(
        predicted_log
    ).clamp_min(0)

    predicted_frames = torch.round(
        predicted_frames
    )

    return predicted_frames[0].cpu().numpy()


# ======================================================================
# 17. VISUALIZATION
# ======================================================================

def save_training_plot(
    history,
    model_name,
    output_dir,
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(

        history["train_loss"],

        label="Training loss",

    )

    if history["validation_loss"]:

        plt.plot(

            history["validation_loss"],

            label="Validation loss",

        )

    plt.xlabel("Epoch")

    plt.ylabel("Masked MSE")

    plt.title(
        f"{model_name} Duration Prediction Loss"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        output_dir /
        f"{model_name}_training.png",

        dpi=150,

    )

    plt.close()


def save_comparison_plot(
    baseline_history,
    transformer_history,
    output_dir,
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(

        baseline_history["train_loss"],

        label="BiLSTM training",

    )

    plt.plot(

        transformer_history["train_loss"],

        label="Transformer training",

    )

    if baseline_history["validation_loss"]:

        plt.plot(

            baseline_history["validation_loss"],

            linestyle="--",

            label="BiLSTM validation",

        )

    if transformer_history["validation_loss"]:

        plt.plot(

            transformer_history["validation_loss"],

            linestyle="--",

            label="Transformer validation",

        )

    plt.xlabel("Epoch")

    plt.ylabel("Masked MSE")

    plt.title(
        "Lesson 56 vs Lesson 57"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        output_dir /
        "model_comparison.png",

        dpi=150,

    )

    plt.close()


def save_prediction_plot(
    phonemes,
    true_durations,
    predicted_durations,
    output_dir,
):

    count = min(

        len(phonemes),

        len(true_durations),

        len(predicted_durations),

        40,

    )

    positions = np.arange(
        count
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.bar(

        positions - 0.2,

        true_durations[:count],

        width=0.4,

        label="Aligned duration",

    )

    plt.bar(

        positions + 0.2,

        predicted_durations[:count],

        width=0.4,

        label="Predicted duration",

    )

    plt.xticks(

        positions,

        phonemes[:count],

        rotation=60,

    )

    plt.xlabel("Phoneme")

    plt.ylabel("Duration (Mel frames)")

    plt.title(
        "Phoneme Duration Prediction"
    )

    plt.grid(
        axis="y"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        output_dir /
        "duration_prediction_example.png",

        dpi=150,

    )

    plt.close()


# ======================================================================
# 18. PARAMETER COUNT
# ======================================================================

def count_parameters(model):

    return sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )


# ======================================================================
# 19. MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--features_dir",

        type=str,

        default="lesson55_outputs/features",

    )

    parser.add_argument(

        "--alignment_dir",

        type=str,

        default="lesson54_outputs",

    )

    parser.add_argument(

        "--output_dir",

        type=str,

        default="lesson57_output",

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

    parser.add_argument(

        "--validation_ratio",

        type=float,

        default=VALIDATION_RATIO,

    )

    args = parser.parse_args()

    if args.epochs < 1:

        raise ValueError(
            "epochs must be positive."
        )

    if args.batch_size < 1:

        raise ValueError(
            "batch_size must be positive."
        )

    if not 0.0 <= args.validation_ratio < 1.0:

        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    set_seed(SEED)

    device = get_device()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    print("=" * 72)

    print(
        "PHASE 2 — LESSON 57"
    )

    print(
        "IMPROVED PHONEME DURATION PREDICTION"
    )

    print("=" * 72)

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------------
    # STEP 1 — LOAD DATA
    # --------------------------------------------------------------

    print("\nSTEP 1 — LOAD REAL SPEECH DATA")

    records = load_alignment_records(

        args.features_dir,

        args.alignment_dir,

    )

    # --------------------------------------------------------------
    # STEP 2 — VOCABULARY
    # --------------------------------------------------------------

    print("\nSTEP 2 — BUILD PHONEME VOCABULARY")

    vocabulary = build_vocabulary(
        records
    )

    print(
        f"Vocabulary size: {len(vocabulary)}"
    )

    with open(

        output_dir / "phoneme_vocabulary.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            vocabulary,

            file,

            indent=2,

            ensure_ascii=False,

        )

    # --------------------------------------------------------------
    # STEP 3 — DATA SPLIT
    # --------------------------------------------------------------

    print("\nSTEP 3 — TRAIN / VALIDATION SPLIT")

    train_records, validation_records = split_records(

        records,

        args.validation_ratio,

    )

    train_dataset = DurationDataset(

        train_records,

        vocabulary,

    )

    validation_dataset = DurationDataset(

        validation_records,

        vocabulary,

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        collate_fn=collate_batch,

    )

    validation_loader = None

    if len(validation_dataset) > 0:

        validation_loader = DataLoader(

            validation_dataset,

            batch_size=args.batch_size,

            shuffle=False,

            collate_fn=collate_batch,

        )

    # --------------------------------------------------------------
    # STEP 4 — BUILD MODELS
    # --------------------------------------------------------------

    print("\nSTEP 4 — BUILD MODELS")

    baseline = BiLSTMDurationPredictor(

        vocabulary_size=len(vocabulary)

    )

    transformer = TransformerDurationPredictor(

        vocabulary_size=len(vocabulary)

    )

    print(
        "\nBiLSTM parameters:",
        f"{count_parameters(baseline):,}",
    )

    print(
        "Transformer parameters:",
        f"{count_parameters(transformer):,}",
    )

    # --------------------------------------------------------------
    # STEP 5 — TRAIN LESSON 56 BASELINE
    # --------------------------------------------------------------

    print("\nSTEP 5 — TRAIN BiLSTM BASELINE")

    baseline, baseline_history = train_model(

        model=baseline,

        train_loader=train_loader,

        validation_loader=validation_loader,

        device=device,

        output_dir=output_dir,

        model_name="bilstm_baseline",

        vocabulary=vocabulary,

        epochs=args.epochs,

    )

    save_training_plot(

        baseline_history,

        "BiLSTM",

        output_dir,

    )

    # --------------------------------------------------------------
    # STEP 6 — TRAIN LESSON 57 TRANSFORMER
    # --------------------------------------------------------------

    print("\nSTEP 6 — TRAIN TRANSFORMER")

    transformer, transformer_history = train_model(

        model=transformer,

        train_loader=train_loader,

        validation_loader=validation_loader,

        device=device,

        output_dir=output_dir,

        model_name="transformer_duration_predictor",

        vocabulary=vocabulary,

        epochs=args.epochs,

    )

    save_training_plot(

        transformer_history,

        "Transformer",

        output_dir,

    )

    # --------------------------------------------------------------
    # STEP 7 — COMPARE TRAINING
    # --------------------------------------------------------------

    print("\nSTEP 7 — COMPARE MODELS")

    save_comparison_plot(

        baseline_history,

        transformer_history,

        output_dir,

    )

    # --------------------------------------------------------------
    # STEP 8 — INFERENCE
    # --------------------------------------------------------------

    print("\nSTEP 8 — DURATION PREDICTION")

    example = records[0]

    predicted_durations = predict_durations(

        transformer,

        example["phonemes"],

        vocabulary,

        device,

    )

    true_durations = example["durations"]

    print(
        f"\nExample utterance: {example['id']}"
    )

    print(
        "\nPhoneme | Actual frames | Predicted frames"
    )

    print("-" * 52)

    for phoneme, actual, predicted in zip(

        example["phonemes"],

        true_durations,

        predicted_durations,

    ):

        print(

            f"{phoneme:10s} | "

            f"{actual:13.1f} | "

            f"{predicted:16.1f}"

        )

    # --------------------------------------------------------------
    # STEP 9 — VISUALIZATION
    # --------------------------------------------------------------

    save_prediction_plot(

        example["phonemes"],

        true_durations,

        predicted_durations,

        output_dir,

    )

    # --------------------------------------------------------------
    # STEP 10 — SUMMARY
    # --------------------------------------------------------------

    print("\n" + "=" * 72)

    print(
        "LESSON 57 — RESULTS"
    )

    print("=" * 72)

    print(
        f"\nUtterances: {len(records)}"
    )

    print(
        f"Phoneme vocabulary: {len(vocabulary)}"
    )

    print(
        f"Training samples: {len(train_records)}"
    )

    print(
        f"Validation samples: {len(validation_records)}"
    )

    if validation_loader is not None:

        baseline_eval = run_epoch(

            baseline,

            validation_loader,

            None,

            device,

            training=False,

        )

        transformer_eval = run_epoch(

            transformer,

            validation_loader,

            None,

            device,

            training=False,

        )

        print("\nVALIDATION COMPARISON")

        print(
            f"BiLSTM MAE: "
            f"{baseline_eval['mae_frames']:.3f} frames"
        )

        print(
            f"Transformer MAE: "
            f"{transformer_eval['mae_frames']:.3f} frames"
        )

        print(
            f"BiLSTM relative error: "
            f"{baseline_eval['relative_error_percent']:.2f}%"
        )

        print(
            f"Transformer relative error: "
            f"{transformer_eval['relative_error_percent']:.2f}%"
        )

    else:

        print(
            "\nValidation comparison unavailable: "
            "insufficient utterance count."
        )

        print(
            "Do not interpret training metrics as "
            "generalization performance."
        )

    print("\nOUTPUT FILES")

    for path in sorted(output_dir.iterdir()):

        if path.is_file():

            print(
                f"  {path.name}"
            )

    print("\nLESSON 57 COMPLETED.")

    print("=" * 72)


if __name__ == "__main__":

    main()