"""
File: lesson52_duration_prediction.py

PHASE 2 — LESSON 52
DURATION PREDICTION FOR MULTILINGUAL TEXT-TO-SPEECH
===================================================

PURPOSE
-------
Learn how a TTS system predicts how long each phoneme should last.

In a typical neural TTS pipeline:

    Text
      |
      v
    Text Normalization
      |
      v
    Phonemization
      |
      v
    Phoneme Encoder
      |
      v
    Duration Predictor
      |
      v
    Length Regulator
      |
      v
    Acoustic Model
      |
      v
    Mel Spectrogram
      |
      v
    Neural Vocoder
      |
      v
    Waveform

WHAT YOU LEARN
--------------

STEP 1 — Multilingual Phoneme Vocabulary
    Automatically construct a vocabulary from the actual tokens
    used in the language templates.

STEP 2 — Synthetic Duration Dataset
    Generate example phoneme sequences and their durations.

STEP 3 — Padding and Batching
    Prepare variable-length sequences for Transformer training.

STEP 4 — Positional Encoding
    Give the Transformer information about phoneme positions.

STEP 5 — Transformer Phoneme Encoder
    Extract contextual representations from phoneme sequences.

STEP 6 — Duration Predictor
    Predict the duration of each phoneme.

STEP 7 — Log-Duration Loss
    Train using:

        target = log(1 + duration)

    This helps stabilize duration prediction.

STEP 8 — Length Regulator
    Expand phoneme representations according to predicted durations.

STEP 9 — Training and Validation
    Train the model and monitor its validation loss.

STEP 10 — Inference
    Predict durations for a new phoneme sequence.

STEP 11 — Numerical Tests
    Verify vocabulary, tensor dimensions, duration prediction,
    padding behavior, and length regulation.

IMPORTANT
---------
This lesson uses synthetic durations for educational purposes.

The generated durations are NOT real speech alignments.

A production TTS system requires real phoneme-to-audio alignment,
for example from a forced aligner or an acoustic model.

The vocabulary is constructed automatically to prevent missing-token
errors. However, vocabulary coverage does not guarantee correct
phonetic transcription.

Dependencies:
    pip install torch numpy matplotlib

Run:
    python lesson52_duration_prediction.py
"""

import os
import math
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# STEP 0 — CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = "lesson52_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "duration_predictor.pt"
)

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.1

NUM_TRAIN_SAMPLES = 1200
NUM_VAL_SAMPLES = 200

MAX_SEQUENCE_LENGTH = 40

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 72)
print("PHASE 2 — LESSON 52: DURATION PREDICTION")
print("=" * 72)

print(f"Device: {DEVICE}")


# ============================================================
# STEP 1 — MULTILINGUAL PHONEME VOCABULARY
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SILENCE_TOKEN = "SP"

BASE_PHONEMES = [
    PAD_TOKEN,
    UNK_TOKEN,
    SILENCE_TOKEN,

    # English ARPAbet-style consonants
    "B", "CH", "D", "DH", "F", "G", "HH",
    "JH", "K", "L", "M", "N", "NG", "P",
    "R", "S", "SH", "T", "TH", "V", "W",
    "Y", "Z", "ZH",

    # English vowels
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY", "OW",
    "OY", "UH", "UW",

    # Additional symbols
    "AX", "AXR", "IX", "UX",

    # Example Chinese pinyin-like units
    "NI", "HAO", "WO", "SHI", "JIE",
    "ZHONG", "GUO", "REN", "AI", "YIN",

    # Example Japanese mora-like units
    "KO", "N", "NI", "CHI", "WA",
    "TO", "KYO", "U", "DE", "SU",

    # Example Russian units
    "PR", "I", "VET", "MIR", "DA",
    "NYE", "T", "R", "A",
]


# Each template is a phoneme-token sequence.
# These are simplified educational examples, not a complete G2P system.

LANGUAGE_TEMPLATES = {
    "english": [
        ["HH", "EH", "L", "OW"],
        ["W", "ER", "L", "D"],
        ["HH", "AY"],
        ["G", "UH", "D"],
        ["M", "AO", "R", "N", "IH", "NG"],
        ["T", "IY", "S", "IY"],
        ["R", "OW", "B", "AA", "T"],
        ["S", "P", "IY", "CH"],
        ["DH", "EH", "T", "AH"],
        ["K", "AE", "T"],
    ],

    "chinese": [
        ["NI", "HAO"],
        ["WO", "AI", "NI"],
        ["ZHONG", "GUO"],
        ["REN", "AI"],
        ["YIN", "HAO"],
        ["SHI", "JIE"],
    ],

    "japanese": [
        ["KO", "N", "NI", "CHI", "WA"],
        ["TO", "KYO", "U"],
        ["NI", "HO", "N"],
        ["A", "RI", "GA", "TO", "U"],
        ["RO", "BO", "TTO"],
    ],

    "russian": [
        ["PR", "I", "VET"],
        ["MIR"],
        ["DA"],
        ["NYE", "T"],
        ["R", "A"],
    ],
}


def build_vocabulary(base_phonemes, language_templates):
    """
    Build vocabulary automatically.

    Every token found in the language templates is added if missing
    from the base inventory.

    This prevents KeyError when a template contains a token that
    was not manually added to the base vocabulary.
    """

    vocabulary = list(base_phonemes)

    automatically_added = []

    for language, templates in language_templates.items():

        for sequence in templates:

            for phoneme in sequence:

                if phoneme not in vocabulary:
                    vocabulary.append(phoneme)
                    automatically_added.append(phoneme)

    # Remove duplicates while preserving order.
    vocabulary = list(dict.fromkeys(vocabulary))

    return vocabulary, automatically_added


PHONEMES, AUTOMATICALLY_ADDED = build_vocabulary(
    BASE_PHONEMES,
    LANGUAGE_TEMPLATES
)

PHONEME_TO_ID = {
    phoneme: index
    for index, phoneme in enumerate(PHONEMES)
}

ID_TO_PHONEME = {
    index: phoneme
    for phoneme, index in PHONEME_TO_ID.items()
}

PAD_ID = PHONEME_TO_ID[PAD_TOKEN]
UNK_ID = PHONEME_TO_ID[UNK_TOKEN]

VOCAB_SIZE = len(PHONEMES)


def validate_vocabulary():
    """Verify that every template token exists in the vocabulary."""

    missing = []

    for language, templates in LANGUAGE_TEMPLATES.items():

        for sequence in templates:

            for phoneme in sequence:

                if phoneme not in PHONEME_TO_ID:
                    missing.append(
                        (language, phoneme)
                    )

    if missing:
        raise ValueError(
            f"Missing phonemes in vocabulary: {missing}"
        )

    print("Vocabulary validation: PASSED")


validate_vocabulary()

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Automatically added tokens: {AUTOMATICALLY_ADDED}")


# ============================================================
# STEP 2 — SYNTHETIC DURATION GENERATION
# ============================================================

def generate_phoneme_duration(phoneme):
    """
    Generate an educational synthetic duration.

    Pause:
        5–12 frames

    Vowel:
        4–9 frames

    Other phonemes:
        1–4 frames

    These are arbitrary educational values.
    """

    if phoneme == SILENCE_TOKEN:
        return random.randint(5, 12)

    vowels = {
        "AA", "AE", "AH", "AO", "AW", "AY",
        "EH", "ER", "EY", "IH", "IY", "OW",
        "OY", "UH", "UW", "AX", "AXR", "IX",
        "UX", "A", "I", "U", "E", "O"
    }

    if phoneme in vowels:
        return random.randint(4, 9)

    return random.randint(1, 4)


def generate_sample():
    """
    Generate one multilingual phoneme sequence.

    Multiple templates may be concatenated to create longer examples.
    """

    languages = list(LANGUAGE_TEMPLATES.keys())

    language = random.choice(languages)

    templates = LANGUAGE_TEMPLATES[language]

    sequence = []

    number_of_phrases = random.randint(1, 3)

    for _ in range(number_of_phrases):

        phrase = random.choice(templates)

        sequence.extend(phrase)

        if random.random() < 0.7:
            sequence.append(SILENCE_TOKEN)

    if sequence and sequence[-1] == SILENCE_TOKEN:
        sequence.pop()

    durations = [
        generate_phoneme_duration(p)
        for p in sequence
    ]

    return {
        "language": language,
        "phonemes": sequence,
        "durations": durations,
    }


# ============================================================
# STEP 3 — DATASET
# ============================================================

class DurationDataset(Dataset):

    def __init__(self, number_of_samples):

        self.samples = []

        for _ in range(number_of_samples):

            sample = generate_sample()

            phoneme_ids = []

            for phoneme in sample["phonemes"]:

                if phoneme not in PHONEME_TO_ID:
                    phoneme_id = UNK_ID
                else:
                    phoneme_id = PHONEME_TO_ID[phoneme]

                phoneme_ids.append(phoneme_id)

            self.samples.append({
                "language": sample["language"],
                "phonemes": sample["phonemes"],

                "phoneme_ids": phoneme_ids,

                "durations": sample["durations"],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # Return tensors directly.
        return {
            "language": sample["language"],

            "phonemes": sample["phonemes"],

            "phoneme_ids": torch.tensor(
                sample["phoneme_ids"],
                dtype=torch.long
            ),

            "durations": torch.tensor(
                sample["durations"],
                dtype=torch.long
            ),
        }


train_dataset = DurationDataset(
    NUM_TRAIN_SAMPLES
)

val_dataset = DurationDataset(
    NUM_VAL_SAMPLES
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# ============================================================
# STEP 4 — COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    """
    Pad variable-length sequences.

    phoneme_ids:
        [batch, sequence_length]

    durations:
        [batch, sequence_length]

    padding_mask:
        True means padding.
    """

    batch_size = len(batch)

    max_length = max(
        len(item["phoneme_ids"])
        for item in batch
    )

    phoneme_ids = torch.full(
        (batch_size, max_length),
        PAD_ID,
        dtype=torch.long
    )

    durations = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long
    )

    padding_mask = torch.ones(
        (batch_size, max_length),
        dtype=torch.bool
    )

    for i, item in enumerate(batch):

        length = len(item["phoneme_ids"])

        phoneme_ids[i, :length] = item["phoneme_ids"]

        durations[i, :length] = item["durations"]

        padding_mask[i, :length] = False

    return {
        "phoneme_ids": phoneme_ids,
        "durations": durations,
        "padding_mask": padding_mask,
        "phonemes": [item["phonemes"] for item in batch],
        "languages": [item["language"] for item in batch],
    }


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# STEP 5 — POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embed_dim,
        max_length=500,
        dropout=0.1
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(
            max_length,
            embed_dim
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        sequence_length = x.size(1)

        x = x + self.pe[:, :sequence_length, :]

        return self.dropout(x)


# ============================================================
# STEP 6 — TRANSFORMER DURATION PREDICTOR
# ============================================================

class DurationPredictor(nn.Module):
    """
    FastSpeech-style convolutional duration predictor.

    Input:
        [B, T, C]

    Output:
        [B, T]

    B = batch size
    T = phoneme sequence length
    C = embedding dimension
    """

    def __init__(
        self,
        embed_dim,
        dropout=0.1
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout2 = nn.Dropout(dropout)

        self.output_layer = nn.Conv1d(
            embed_dim,
            1,
            kernel_size=1
        )

    def forward(self, x):
        """
        Input: [B, T, C]
        """

        # ----------------------------------------------------
        # First convolution
        # ----------------------------------------------------

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        x = self.conv1(x)

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        # LayerNorm operates on the final dimension.
        x = self.norm1(x)

        x = F.relu(x)

        x = self.dropout1(x)

        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        x = self.conv2(x)

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        x = self.norm2(x)

        x = F.relu(x)

        x = self.dropout2(x)

        # ----------------------------------------------------
        # Duration prediction
        # ----------------------------------------------------

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        x = self.output_layer(x)

        # [B, 1, T] -> [B, T]
        x = x.squeeze(1)

        return x


class DurationModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
        pad_id=0
    ):

        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_id
        )

        self.position = PositionalEncoding(
            embed_dim,
            max_length=MAX_SEQUENCE_LENGTH + 20,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.duration_predictor = DurationPredictor(
            embed_dim,
            dropout
        )

    def encode(
        self,
        phoneme_ids,
        padding_mask=None
    ):

        x = self.embedding(phoneme_ids)

        x = self.position(x)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        return x

    def forward(
        self,
        phoneme_ids,
        padding_mask=None
    ):

        features = self.encode(
            phoneme_ids,
            padding_mask
        )

        log_durations = self.duration_predictor(
            features
        )

        if padding_mask is not None:

            log_durations = log_durations.masked_fill(
                padding_mask,
                0.0
            )

        return log_durations

    @torch.no_grad()
    def predict_durations(self, phoneme_ids):

        self.eval()

        padding_mask = phoneme_ids.eq(
            self.pad_id
        )

        log_durations = self.forward(
            phoneme_ids,
            padding_mask
        )

        # Reverse log1p transformation.
        durations = torch.expm1(
            log_durations
        )

        durations = torch.round(
            durations
        )

        durations = durations.clamp(
            min=1,
            max=100
        )

        # Padding must have zero duration.
        durations = durations.masked_fill(
            padding_mask,
            0
        )

        return durations.long()


model = DurationModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    pad_id=PAD_ID
).to(DEVICE)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Model parameters: {parameter_count:,}")


# ============================================================
# STEP 7 — LOSS FUNCTION
# ============================================================

def duration_loss(
    predicted_log_durations,
    target_durations,
    padding_mask
):
    """
    MSE loss in log-duration space.

        target = log(1 + duration)

    Padding positions are excluded from the loss.
    """

    target_log_durations = torch.log1p(
        target_durations.float()
    )

    loss_per_token = F.mse_loss(
        predicted_log_durations,
        target_log_durations,
        reduction="none"
    )

    valid_mask = ~padding_mask

    loss_per_token = loss_per_token * valid_mask.float()

    denominator = valid_mask.float().sum().clamp(
        min=1.0
    )

    return loss_per_token.sum() / denominator


# ============================================================
# STEP 8 — TRAINING AND VALIDATION
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4
)

train_losses = []
val_losses = []

best_val_loss = float("inf")


def run_validation():

    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for batch in val_loader:

            phoneme_ids = batch["phoneme_ids"].to(DEVICE)

            durations = batch["durations"].to(DEVICE)

            padding_mask = batch["padding_mask"].to(DEVICE)

            predicted = model(
                phoneme_ids,
                padding_mask
            )

            loss = duration_loss(
                predicted,
                durations,
                padding_mask
            )

            total_loss += loss.item()

            total_batches += 1

    return total_loss / max(total_batches, 1)


print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch in train_loader:

        phoneme_ids = batch["phoneme_ids"].to(DEVICE)

        durations = batch["durations"].to(DEVICE)

        padding_mask = batch["padding_mask"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        predicted_log_durations = model(
            phoneme_ids,
            padding_mask
        )

        loss = duration_loss(
            predicted_log_durations,
            durations,
            padding_mask
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / max(
        len(train_loader),
        1
    )

    val_loss = run_validation()

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f} | "
        f"LR: {current_lr:.6f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "phonemes": PHONEMES,
            "phoneme_to_id": PHONEME_TO_ID,
            "config": {
                "embed_dim": EMBED_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "ff_dim": FF_DIM,
                "vocab_size": VOCAB_SIZE,
                "pad_id": PAD_ID,
            }
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )


print("\nTraining completed.")

print(f"Best validation loss: {best_val_loss:.5f}")

print(f"Checkpoint saved: {CHECKPOINT_PATH}")


# ============================================================
# STEP 9 — LOAD BEST CHECKPOINT
# ============================================================

def load_checkpoint(path, model):

    try:
        # Explicit weights_only=False supports checkpoints containing
        # vocabulary and configuration metadata.
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False
        )

    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            path,
            map_location=DEVICE
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return checkpoint


checkpoint = load_checkpoint(
    CHECKPOINT_PATH,
    model
)

print(
    f"Loaded best checkpoint from epoch "
    f"{checkpoint['epoch']}"
)


# ============================================================
# STEP 10 — LENGTH REGULATOR
# ============================================================

class LengthRegulator(nn.Module):
    """
    Expand each phoneme feature according to its duration.

    Input:
        features:  [B, T, C]
        durations: [B, T]

    Output:
        expanded:  [B, max_expanded_length, C]
        lengths:   [B]

    Example:

        features:
            A, B, C

        durations:
            2, 3, 1

        expanded:
            A, A, B, B, B, C
    """

    def forward(
        self,
        features,
        durations
    ):

        if features.dim() != 3:
            raise ValueError(
                "features must have shape [B, T, C]"
            )

        if durations.dim() != 2:
            raise ValueError(
                "durations must have shape [B, T]"
            )

        if features.shape[:2] != durations.shape:
            raise ValueError(
                "Feature and duration dimensions do not match."
            )

        durations = durations.long().clamp(
            min=0
        )

        batch_size = features.size(0)

        feature_dim = features.size(2)

        expanded_sequences = []

        expanded_lengths = []

        for batch_index in range(batch_size):

            repeated_features = []

            for time_index in range(features.size(1)):

                duration = int(
                    durations[
                        batch_index,
                        time_index
                    ].item()
                )

                if duration > 0:

                    repeated = features[
                        batch_index,
                        time_index
                    ].unsqueeze(0).expand(
                        duration,
                        feature_dim
                    )

                    repeated_features.append(
                        repeated
                    )

            if repeated_features:

                expanded = torch.cat(
                    repeated_features,
                    dim=0
                )

            else:

                expanded = features.new_zeros(
                    (0, feature_dim)
                )

            expanded_sequences.append(expanded)

            expanded_lengths.append(
                expanded.size(0)
            )

        max_length = max(
            expanded_lengths,
            default=0
        )

        output = features.new_zeros(
            batch_size,
            max_length,
            feature_dim
        )

        for batch_index, sequence in enumerate(
            expanded_sequences
        ):

            length = sequence.size(0)

            if length > 0:

                output[
                    batch_index,
                    :length
                ] = sequence

        lengths = torch.tensor(
            expanded_lengths,
            dtype=torch.long,
            device=features.device
        )

        return output, lengths


length_regulator = LengthRegulator()


# ============================================================
# STEP 11 — INFERENCE
# ============================================================

print("\n" + "=" * 72)
print("DURATION PREDICTION — INFERENCE")
print("=" * 72)

inference_phonemes = [
    "HH", "EH", "L", "OW",
    "SP",
    "W", "ER", "L", "D"
]

# Check every inference token before conversion.
missing_tokens = [
    phoneme
    for phoneme in inference_phonemes
    if phoneme not in PHONEME_TO_ID
]

if missing_tokens:
    raise ValueError(
        f"Inference tokens missing from vocabulary: {missing_tokens}"
    )

inference_ids = [
    PHONEME_TO_ID[phoneme]
    for phoneme in inference_phonemes
]

# Create the tensor once.
# Shape before batching: [T]
test_ids = torch.tensor(
    inference_ids,
    dtype=torch.long,
    device=DEVICE
)

# Add batch dimension.
# Shape after batching: [1, T]
test_ids = test_ids.unsqueeze(0)

model.eval()

with torch.no_grad():

    predicted_durations = model.predict_durations(
        test_ids
    )

predicted_durations = predicted_durations[0].cpu()

total_frames = int(
    predicted_durations.sum().item()
)

# Educational example: assume 62.5 frames per second.
FRAME_RATE = 62.5

estimated_duration = total_frames / FRAME_RATE

print("\nPhonemes:")
print(inference_phonemes)

print("\nPredicted durations:")
print(predicted_durations.tolist())

print(f"\nTotal frames: {total_frames}")

print(
    f"Estimated duration: {estimated_duration:.3f} seconds"
)


# ============================================================
# STEP 12 — LENGTH REGULATOR INFERENCE
# ============================================================

with torch.no_grad():

    # Reuse the existing batched tensor.
    padding_mask = test_ids.eq(PAD_ID)

    features = model.encode(
        test_ids,
        padding_mask
    )

    # predicted_durations currently has shape [T].
    # Add a batch dimension to obtain [1, T].
    test_durations = (
        predicted_durations
        .unsqueeze(0)
        .long()
        .to(DEVICE)
    )

    expanded_features, expanded_lengths = length_regulator(
        features,
        test_durations
    )

print("\nLength Regulator:")

print(
    f"Input phoneme features: "
    f"{tuple(features.shape)}"
)

print(
    f"Predicted durations: "
    f"{tuple(test_durations.shape)}"
)

print(
    f"Expanded features: "
    f"{tuple(expanded_features.shape)}"
)

print(
    f"Expanded sequence length: "
    f"{expanded_lengths.tolist()}"
)


# ============================================================
# STEP 13 — VISUALIZATION: TRAINING CURVES
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "Duration Predictor Training"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

training_plot_path = os.path.join(
    OUTPUT_DIR,
    "training_curves.png"
)

plt.savefig(
    training_plot_path,
    dpi=150
)

plt.show()

print(f"Saved: {training_plot_path}")


# ============================================================
# STEP 14 — VISUALIZATION: PHONEME DURATIONS
# ============================================================

phoneme_positions = np.arange(
    len(inference_phonemes)
)

plt.figure(figsize=(12, 5))

plt.bar(
    phoneme_positions,
    predicted_durations.numpy()
)

plt.xticks(
    phoneme_positions,
    inference_phonemes
)

plt.xlabel("Phoneme")

plt.ylabel("Predicted Duration (frames)")

plt.title(
    "Predicted Duration for Each Phoneme"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

duration_plot_path = os.path.join(
    OUTPUT_DIR,
    "predicted_durations.png"
)

plt.savefig(
    duration_plot_path,
    dpi=150
)

plt.show()

print(f"Saved: {duration_plot_path}")


# ============================================================
# STEP 15 — VISUALIZATION: LENGTH REGULATOR
# ============================================================

expanded_sequence = []

for phoneme, duration in zip(
    inference_phonemes,
    predicted_durations.tolist()
):

    expanded_sequence.extend(
        [phoneme] * duration
    )

plt.figure(figsize=(14, 4))

plt.bar(
    np.arange(len(expanded_sequence)),
    np.ones(len(expanded_sequence))
)

plt.yticks([])

plt.xlabel("Expanded Frame")

plt.title(
    "Length Regulator — Phoneme Expansion"
)

plt.tight_layout()

length_plot_path = os.path.join(
    OUTPUT_DIR,
    "length_regulator.png"
)

plt.savefig(
    length_plot_path,
    dpi=150
)

plt.show()

print(f"Saved: {length_plot_path}")


# ============================================================
# STEP 16 — NUMERICAL TESTS
# ============================================================

print("\n" + "=" * 72)
print("NUMERICAL TESTS")
print("=" * 72)


def run_test(name, condition):

    if condition:

        print(f"[PASS] {name}")

    else:

        raise AssertionError(
            f"[FAIL] {name}"
        )


# Test 1: vocabulary has unique entries.
run_test(
    "Vocabulary contains unique tokens",
    len(PHONEMES) == len(set(PHONEMES))
)


# Test 2: PAD token exists.
run_test(
    "PAD token exists",
    PAD_TOKEN in PHONEME_TO_ID
)


# Test 3: every language template is covered.
all_template_tokens = {
    phoneme
    for templates in LANGUAGE_TEMPLATES.values()
    for sequence in templates
    for phoneme in sequence
}

run_test(
    "All template tokens exist in vocabulary",
    all(
        token in PHONEME_TO_ID
        for token in all_template_tokens
    )
)


# Test 4: inference tensor shape.
run_test(
    "Inference tensor has batch dimension",
    test_ids.dim() == 2 and test_ids.shape[0] == 1
)


# Test 5: prediction shape.
run_test(
    "Predicted duration shape matches phoneme sequence",
    predicted_durations.shape[0] == len(inference_phonemes)
)


# Test 6: durations are positive.
run_test(
    "All predicted durations are positive",
    bool(torch.all(predicted_durations > 0))
)


# Test 7: total duration.
run_test(
    "Total duration equals sum of phoneme durations",
    total_frames == int(predicted_durations.sum().item())
)


# Test 8: length regulator output length.
run_test(
    "Length regulator output matches total frames",
    expanded_features.shape[1] == total_frames
)


# Test 9: length regulator batch size.
run_test(
    "Length regulator preserves batch size",
    expanded_features.shape[0] == 1
)


# Test 10: dataset tensor types.
sample = train_dataset[0]

run_test(
    "Dataset phoneme IDs are tensors",
    isinstance(sample["phoneme_ids"], torch.Tensor)
)


run_test(
    "Dataset durations are tensors",
    isinstance(sample["durations"], torch.Tensor)
)


# Test 11: padding behavior.
padding_test_ids = torch.tensor(
    [[
        PHONEME_TO_ID["HH"],
        PHONEME_TO_ID["EH"],
        PAD_ID,
        PAD_ID
    ]],
    dtype=torch.long,
    device=DEVICE
)

padding_test_predictions = model.predict_durations(
    padding_test_ids
)

run_test(
    "Padding positions receive zero duration",
    bool(
        torch.all(
            padding_test_predictions[0, 2:] == 0
        )
    )
)


# Test 12: checkpoint exists.
run_test(
    "Model checkpoint was saved",
    os.path.exists(CHECKPOINT_PATH)
)


# Test 13: finite model output.
run_test(
    "Predicted durations are finite",
    bool(
        torch.isfinite(
            predicted_durations.float()
        ).all()
    )
)


print("\n" + "=" * 72)
print("ALL TESTS PASSED")
print("=" * 72)


# ============================================================
# STEP 17 — FINAL SUMMARY
# ============================================================

print("\nLESSON 52 SUMMARY")
print("-" * 72)

print(f"Vocabulary size:       {VOCAB_SIZE}")

print(f"Model parameters:      {parameter_count:,}")

print(f"Training samples:      {NUM_TRAIN_SAMPLES}")

print(f"Validation samples:    {NUM_VAL_SAMPLES}")

print(f"Training epochs:       {EPOCHS}")

print(f"Best validation loss:  {best_val_loss:.5f}")

print(f"Predicted frames:      {total_frames}")

print(f"Estimated duration:    {estimated_duration:.3f} seconds")

print("\nGenerated files:")

print(f"  Checkpoint:          {CHECKPOINT_PATH}")

print(f"  Training curves:     {training_plot_path}")

print(f"  Duration plot:       {duration_plot_path}")

print(f"  Length regulator:    {length_plot_path}")

print("\nLesson 52 completed successfully.")