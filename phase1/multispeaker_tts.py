"""
File: phase1_lesson18_multispeaker_tts.py

PHASE 1 — LESSON 18
BUILD A MULTI-SPEAKER TTS ACOUSTIC MODEL FROM SCRATCH
=====================================================

PURPOSE
-------
Implement a simplified FastSpeech-style, multi-speaker TTS model.

Components:
    1. Phoneme vocabulary and embedding
    2. Transformer text encoder
    3. Speaker ID embedding
    4. Phoneme duration predictor
    5. Length regulator
    6. Speaker conditioning
    7. Transformer Mel decoder
    8. Mel reconstruction loss
    9. Duration prediction loss
   10. Training and validation
   11. Multi-speaker inference
   12. Mel spectrogram visualization
   13. Checkpoint saving

IMPORTANT
---------
This lesson uses synthetic phoneme-to-Mel training examples.

The Mel targets are artificial acoustic patterns, NOT real speech.
The model does not learn genuine pronunciation.

The model is trained from random initialization.
No pretrained TTS model is used.

The output is a predicted Mel spectrogram, not a waveform.
A neural vocoder is needed to convert it into audio.

ARCHITECTURE
------------
Phoneme IDs
    |
    v
Phoneme Embedding
    |
    v
Transformer Text Encoder
    |
    +--------------------------+
    |                          |
    v                          v
Duration Predictor       Speaker ID Embedding
    |                          |
    v                          |
Length Regulator               |
    |                          |
    +------------+-------------+
                 |
                 v
         Speaker Conditioning
                 |
                 v
           Mel Decoder
                 |
                 v
        Predicted Mel Spectrogram

TRAINING
--------
    L_total = L_mel + lambda_duration * L_duration

    L_mel:
        Masked L1 Mel reconstruction loss.

    L_duration:
        Smooth L1 loss between predicted and target log durations.

    During training:
        Ground-truth durations are used by the length regulator.

    During inference:
        Predicted durations are used.

RUN
---
    python phase1_lesson18_multispeaker_tts.py

Optional:
    python phase1_lesson18_multispeaker_tts.py --epochs 40
    python phase1_lesson18_multispeaker_tts.py --speakers 8
    python phase1_lesson18_multispeaker_tts.py --samples 1000
    python phase1_lesson18_multispeaker_tts.py --no-plot

DEPENDENCIES
------------
    pip install numpy matplotlib torch

OUTPUT
------
    lesson18_multispeaker_tts.pt
    lesson18_training_curves.png
    lesson18_speaker_comparison.png
"""


import argparse
import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# CONFIGURATION
# ============================================================

N_MELS = 80

HIDDEN_DIM = 128

EMBEDDING_DIM = 128

NUM_HEADS = 4

NUM_ENCODER_LAYERS = 3

NUM_DECODER_LAYERS = 3

MAX_PHONEME_SEQUENCE_LENGTH = 16

MIN_DURATION = 2

MAX_DURATION = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# PHONEME VOCABULARY
# ============================================================

"""
A small educational English phoneme inventory.

ID 0 is reserved for padding.

This is not a complete IPA inventory.

The vocabulary is deliberately small so that we can concentrate
on the architecture rather than linguistic frontend engineering.

A future multilingual frontend will need separate or unified
phoneme inventories and appropriate language-specific processing.
"""

PHONEMES = [
    "PAD",
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH"
]

PHONEME_TO_ID = {
    phoneme: index
    for index, phoneme in enumerate(PHONEMES)
}

ID_TO_PHONEME = {
    index: phoneme
    for index, phoneme in enumerate(PHONEMES)
}

VOCAB_SIZE = len(PHONEMES)


# ============================================================
# SYNTHETIC SPEAKER PROFILES
# ============================================================

def create_speaker_profiles(
    number_of_speakers,
    seed=123
):
    """
    Generate artificial speaker characteristics.

    Each speaker receives a stable spectral profile.

    These profiles are used only to create synthetic Mel targets.
    They are not physically accurate models of human voices.
    """

    rng = np.random.default_rng(seed)

    profiles = []

    for speaker_id in range(number_of_speakers):

        profile = {

            "speaker_id": speaker_id,

            "spectral_offset": rng.normal(
                0.0,
                0.45,
                size=N_MELS
            ).astype(np.float32),

            "spectral_slope": rng.uniform(
                -0.5,
                0.5
            ),

            "amplitude": rng.uniform(
                0.75,
                1.25
            ),

            "resonance_center": rng.uniform(
                0.2,
                0.8
            )
        }

        # Smooth speaker spectral offset.
        kernel = np.ones(
            7,
            dtype=np.float32
        ) / 7.0

        profile["spectral_offset"] = np.convolve(
            profile["spectral_offset"],
            kernel,
            mode="same"
        ).astype(np.float32)

        profiles.append(profile)

    return profiles


# ============================================================
# SYNTHETIC PHONEME-TO-MEL TEMPLATES
# ============================================================

def create_phoneme_templates(seed=321):
    """
    Create one artificial spectral template per phoneme.

    The template represents a simplified spectral pattern.

    Real speech does not have one fixed Mel spectrum per phoneme.
    The actual spectrum depends on context, speaker, coarticulation,
    pitch, stress, and many other factors.

    These templates only make the toy learning problem possible.
    """

    rng = np.random.default_rng(seed)

    templates = rng.normal(
        0.0,
        0.7,
        size=(VOCAB_SIZE, N_MELS)
    ).astype(np.float32)

    templates[0] = 0.0

    # Smooth the frequency dimension.
    kernel = np.ones(
        5,
        dtype=np.float32
    ) / 5.0

    for index in range(1, VOCAB_SIZE):

        templates[index] = np.convolve(
            templates[index],
            kernel,
            mode="same"
        )

    return templates


# ============================================================
# SYNTHETIC TRAINING EXAMPLE GENERATOR
# ============================================================

def generate_example(
    speaker_id,
    profile,
    phoneme_templates,
    rng
):
    """
    Generate:

        phoneme_ids: (phoneme_count,)
        durations:   (phoneme_count,)
        mel:         (mel_frames, n_mels)

    The target Mel pattern is based on:

        phoneme spectral template
        + speaker-specific spectral offset
        + frequency slope
        + temporal modulation
        + small noise
    """

    sequence_length = rng.integers(
        5,
        MAX_PHONEME_SEQUENCE_LENGTH + 1
    )

    phoneme_ids = rng.integers(
        1,
        VOCAB_SIZE,
        size=sequence_length
    )

    durations = rng.integers(
        MIN_DURATION,
        MAX_DURATION + 1,
        size=sequence_length
    )

    mel_frames = []

    frequency_axis = np.linspace(
        -1.0,
        1.0,
        N_MELS
    )

    speaker_offset = profile[
        "spectral_offset"
    ]

    speaker_slope = profile[
        "spectral_slope"
    ]

    speaker_amplitude = profile[
        "amplitude"
    ]

    resonance_center = profile[
        "resonance_center"
    ]

    for phoneme_id, duration in zip(
        phoneme_ids,
        durations
    ):

        base = phoneme_templates[
            phoneme_id
        ].copy()

        # Speaker-specific spectral contribution.
        speaker_component = (
            speaker_offset
            + speaker_slope * frequency_axis
        )

        # Simplified speaker-dependent resonance.
        resonance = np.exp(
            -0.5
            * (
                (
                    np.linspace(0.0, 1.0, N_MELS)
                    - resonance_center
                ) / 0.15
            ) ** 2
        )

        speaker_component += (
            0.5 * resonance
        )

        for frame in range(duration):

            relative_position = (
                frame / max(duration - 1, 1)
            )

            # Small temporal evolution inside a phoneme.
            temporal_modulation = (
                0.10
                * np.sin(
                    2.0
                    * np.pi
                    * relative_position
                )
            )

            frame_mel = (
                speaker_amplitude
                * base
                + speaker_component
                + temporal_modulation
            )

            noise = rng.normal(
                0.0,
                0.04,
                size=N_MELS
            )

            frame_mel += noise

            mel_frames.append(
                frame_mel.astype(np.float32)
            )

    mel = np.stack(
        mel_frames
    )

    return (
        phoneme_ids.astype(np.int64),
        durations.astype(np.int64),
        mel.astype(np.float32),
        speaker_id
    )


# ============================================================
# DATASET
# ============================================================

class MultiSpeakerTTSDataset(Dataset):

    def __init__(
        self,
        number_of_speakers=8,
        number_of_samples=800,
        seed=42
    ):

        rng = np.random.default_rng(seed)

        self.profiles = create_speaker_profiles(
            number_of_speakers,
            seed=seed + 1
        )

        templates = create_phoneme_templates(
            seed=seed + 2
        )

        self.examples = []

        print("\nGenerating synthetic multi-speaker TTS dataset...")

        for sample_index in range(number_of_samples):

            speaker_id = int(
                rng.integers(
                    0,
                    number_of_speakers
                )
            )

            profile = self.profiles[
                speaker_id
            ]

            example = generate_example(
                speaker_id,
                profile,
                templates,
                rng
            )

            self.examples.append(example)

        print("Dataset generation complete.")

        print("Number of speakers:", number_of_speakers)

        print("Number of examples:", number_of_samples)

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):

        phonemes, durations, mel, speaker_id = (
            self.examples[index]
        )

        return {
            "phonemes": torch.tensor(
                phonemes,
                dtype=torch.long
            ),

            "durations": torch.tensor(
                durations,
                dtype=torch.long
            ),

            "mel": torch.tensor(
                mel,
                dtype=torch.float32
            ),

            "speaker_id": torch.tensor(
                speaker_id,
                dtype=torch.long
            )
        }


# ============================================================
# BATCH COLLATION
# ============================================================

def collate_batch(batch):
    """
    Pad variable-length phoneme and Mel sequences.

    Phonemes:
        (B, max_phoneme_length)

    Durations:
        (B, max_phoneme_length)

    Mel:
        (B, max_mel_frames, N_MELS)

    Lengths are retained so that padding is excluded from losses.
    """

    batch_size = len(batch)

    phoneme_lengths = torch.tensor(
        [
            len(item["phonemes"])
            for item in batch
        ],
        dtype=torch.long
    )

    mel_lengths = torch.tensor(
        [
            item["mel"].shape[0]
            for item in batch
        ],
        dtype=torch.long
    )

    max_phonemes = int(
        phoneme_lengths.max()
    )

    max_mel_frames = int(
        mel_lengths.max()
    )

    phonemes = torch.zeros(
        batch_size,
        max_phonemes,
        dtype=torch.long
    )

    durations = torch.zeros(
        batch_size,
        max_phonemes,
        dtype=torch.long
    )

    mel = torch.zeros(
        batch_size,
        max_mel_frames,
        N_MELS,
        dtype=torch.float32
    )

    speaker_ids = torch.zeros(
        batch_size,
        dtype=torch.long
    )

    for index, item in enumerate(batch):

        p = item["phonemes"]
        d = item["durations"]
        m = item["mel"]

        phonemes[
            index,
            :len(p)
        ] = p

        durations[
            index,
            :len(d)
        ] = d

        mel[
            index,
            :m.shape[0]
        ] = m

        speaker_ids[index] = item[
            "speaker_id"
        ]

    return {
        "phonemes": phonemes,
        "durations": durations,
        "mel": mel,
        "speaker_ids": speaker_ids,
        "phoneme_lengths": phoneme_lengths,
        "mel_lengths": mel_lengths
    }


# ============================================================
# SINUSOIDAL POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        dimension,
        max_length=2048
    ):

        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                dimension,
                2,
                dtype=torch.float32
            )
            * (-math.log(10000.0) / dimension)
        )

        pe = torch.zeros(
            max_length,
            dimension
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return x + self.pe[
            :,
            :x.shape[1],
            :
        ]


# ============================================================
# TRANSFORMER TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_DIM,
            padding_idx=0
        )

        self.position = PositionalEncoding(
            HIDDEN_DIM
        )

        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=NUM_ENCODER_LAYERS
        )

        self.norm = nn.LayerNorm(
            HIDDEN_DIM
        )

    def forward(
        self,
        phonemes,
        phoneme_lengths
    ):

        padding_mask = (
            torch.arange(
                phonemes.shape[1],
                device=phonemes.device
            )[None, :]
            >= phoneme_lengths[:, None]
        )

        x = self.embedding(phonemes)

        x = self.position(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.norm(x)

        return x, padding_mask


# ============================================================
# DURATION PREDICTOR
# ============================================================

class DurationPredictor(nn.Module):
    """
    Predict log(1 + duration).

    Predicting logarithmic durations is often more stable because
    phoneme durations have a skewed distribution.
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                HIDDEN_DIM,
                HIDDEN_DIM,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.LayerNorm(HIDDEN_DIM),

            nn.Conv1d(
                HIDDEN_DIM,
                HIDDEN_DIM,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.LayerNorm(HIDDEN_DIM),

            nn.Linear(
                HIDDEN_DIM,
                1
            )
        )

    def forward(self, x):

        # Transformer representation:
        # (B, phoneme_length, hidden_dim)

        x = x.transpose(1, 2)

        x = self.network[0](x)
        x = self.network[1](x)

        x = x.transpose(1, 2)
        x = self.network[2](x)

        x = x.transpose(1, 2)

        x = self.network[3](x)
        x = self.network[4](x)

        x = x.transpose(1, 2)
        x = self.network[5](x)

        x = self.network[6](x)

        return x.squeeze(-1)


# ============================================================
# LENGTH REGULATOR
# ============================================================

def length_regulator(
    encoder_output,
    durations,
    phoneme_lengths
):
    """
    Expand phoneme-level representations into frame-level
    representations.

    Example:

        Encoder output:
            [h1, h2, h3]

        Durations:
            [2, 3, 1]

        Expanded:
            [h1, h1, h2, h2, h2, h3]

    The operation uses repeat_interleave.

    This is the core temporal expansion operation in
    duration-based non-autoregressive TTS.
    """

    batch_size = encoder_output.shape[0]

    expanded_sequences = []

    lengths = []

    for batch_index in range(batch_size):

        valid_length = int(
            phoneme_lengths[batch_index].item()
        )

        valid_durations = durations[
            batch_index,
            :valid_length
        ]

        expanded = torch.repeat_interleave(
            encoder_output[
                batch_index,
                :valid_length
            ],
            valid_durations,
            dim=0
        )

        expanded_sequences.append(
            expanded
        )

        lengths.append(
            expanded.shape[0]
        )

    max_length = max(lengths)

    output = encoder_output.new_zeros(
        batch_size,
        max_length,
        HIDDEN_DIM
    )

    for index, expanded in enumerate(
        expanded_sequences
    ):

        output[
            index,
            :expanded.shape[0]
        ] = expanded

    return output, torch.tensor(
        lengths,
        device=encoder_output.device,
        dtype=torch.long
    )


# ============================================================
# MEL DECODER
# ============================================================

class MelDecoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.position = PositionalEncoding(
            HIDDEN_DIM
        )

        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=NUM_DECODER_LAYERS
        )

        self.norm = nn.LayerNorm(
            HIDDEN_DIM
        )

        self.output_projection = nn.Linear(
            HIDDEN_DIM,
            N_MELS
        )

    def forward(
        self,
        x,
        frame_lengths
    ):

        padding_mask = (
            torch.arange(
                x.shape[1],
                device=x.device
            )[None, :]
            >= frame_lengths[:, None]
        )

        x = self.position(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.norm(x)

        mel = self.output_projection(x)

        return mel


# ============================================================
# MULTI-SPEAKER TTS MODEL
# ============================================================

class MultiSpeakerTTS(nn.Module):

    def __init__(
        self,
        number_of_speakers
    ):

        super().__init__()

        self.text_encoder = TextEncoder()

        self.duration_predictor = DurationPredictor()

        self.speaker_embedding = nn.Embedding(
            number_of_speakers,
            EMBEDDING_DIM
        )

        self.speaker_projection = nn.Linear(
            EMBEDDING_DIM,
            HIDDEN_DIM
        )

        self.decoder = MelDecoder()

    def encode(self, phonemes, phoneme_lengths):

        encoder_output, padding_mask = (
            self.text_encoder(
                phonemes,
                phoneme_lengths
            )
        )

        predicted_log_durations = (
            self.duration_predictor(
                encoder_output
            )
        )

        return (
            encoder_output,
            predicted_log_durations,
            padding_mask
        )

    def forward(
        self,
        phonemes,
        phoneme_lengths,
        speaker_ids,
        ground_truth_durations=None
    ):
        """
        Training:
            ground_truth_durations is provided.

        Inference:
            ground_truth_durations is None, so predicted
            durations are converted to integer frame counts.
        """

        (
            encoder_output,
            predicted_log_durations,
            phoneme_padding_mask
        ) = self.encode(
            phonemes,
            phoneme_lengths
        )

        if ground_truth_durations is not None:

            durations = ground_truth_durations

        else:

            durations = torch.expm1(
                predicted_log_durations
            )

            durations = torch.round(
                durations
            ).long()

            durations = durations.clamp(
                min=1,
                max=30
            )

            durations = durations.masked_fill(
                phoneme_padding_mask,
                0
            )

        expanded, frame_lengths = length_regulator(
            encoder_output,
            durations,
            phoneme_lengths
        )

        # Obtain speaker vector.
        speaker_vector = self.speaker_embedding(
            speaker_ids
        )

        speaker_vector = self.speaker_projection(
            speaker_vector
        )

        # Broadcast speaker vector across all Mel frames.
        speaker_vector = speaker_vector.unsqueeze(1)

        # Speaker conditioning.
        conditioned = (
            expanded + speaker_vector
        )

        predicted_mel = self.decoder(
            conditioned,
            frame_lengths
        )

        return {
            "mel": predicted_mel,

            "predicted_log_durations":
                predicted_log_durations,

            "durations": durations,

            "frame_lengths": frame_lengths,

            "speaker_vector": speaker_vector.squeeze(1)
        }


# ============================================================
# MASKED MEL LOSS
# ============================================================

def masked_mel_loss(
    predicted_mel,
    target_mel,
    mel_lengths
):
    """
    Ignore padded Mel frames.

    predicted_mel:
        (B, max_frames, N_MELS)

    target_mel:
        (B, max_frames, N_MELS)
    """

    frame_mask = (
        torch.arange(
            predicted_mel.shape[1],
            device=predicted_mel.device
        )[None, :]
        < mel_lengths[:, None]
    )

    frame_mask = frame_mask.unsqueeze(-1)

    absolute_error = torch.abs(
        predicted_mel - target_mel
    )

    absolute_error = absolute_error * frame_mask

    denominator = (
        frame_mask.sum() * N_MELS
    ).clamp_min(1)

    return absolute_error.sum() / denominator


# ============================================================
# DURATION LOSS
# ============================================================

def duration_loss(
    predicted_log_durations,
    target_durations,
    phoneme_lengths
):
    """
    Compare log(1 + predicted duration) with log(1 + target duration).

    Padding positions are excluded.
    """

    target_log_durations = torch.log1p(
        target_durations.float()
    )

    mask = (
        torch.arange(
            predicted_log_durations.shape[1],
            device=predicted_log_durations.device
        )[None, :]
        < phoneme_lengths[:, None]
    )

    loss = F.smooth_l1_loss(
        predicted_log_durations,
        target_log_durations,
        reduction="none"
    )

    loss = loss * mask

    return (
        loss.sum()
        / mask.sum().clamp_min(1)
    )


# ============================================================
# TRAINING AND VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer=None
):

    training = optimizer is not None

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0
    total_mel_loss = 0.0
    total_duration_loss = 0.0

    batches = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for batch in loader:

            phonemes = batch["phonemes"].to(DEVICE)

            durations = batch["durations"].to(DEVICE)

            mel = batch["mel"].to(DEVICE)

            speaker_ids = batch["speaker_ids"].to(DEVICE)

            phoneme_lengths = batch[
                "phoneme_lengths"
            ].to(DEVICE)

            mel_lengths = batch[
                "mel_lengths"
            ].to(DEVICE)

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            output = model(
                phonemes=phonemes,
                phoneme_lengths=phoneme_lengths,
                speaker_ids=speaker_ids,
                ground_truth_durations=durations
            )

            loss_mel = masked_mel_loss(
                output["mel"],
                mel,
                mel_lengths
            )

            loss_duration = duration_loss(
                output["predicted_log_durations"],
                durations,
                phoneme_lengths
            )

            loss = (
                loss_mel
                + 0.5 * loss_duration
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            total_loss += loss.item()

            total_mel_loss += loss_mel.item()

            total_duration_loss += (
                loss_duration.item()
            )

            batches += 1

    return {
        "total": total_loss / max(batches, 1),

        "mel": total_mel_loss / max(batches, 1),

        "duration": (
            total_duration_loss / max(batches, 1)
        )
    }


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def synthesize_mel(
    model,
    phoneme_ids,
    speaker_id
):
    """
    Generate a Mel spectrogram using predicted durations.

    This is acoustic-model inference.

    No ground-truth durations are supplied.
    """

    model.eval()

    phonemes = torch.tensor(
        phoneme_ids,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    phoneme_lengths = torch.tensor(
        [len(phoneme_ids)],
        dtype=torch.long,
        device=DEVICE
    )

    speaker_ids = torch.tensor(
        [speaker_id],
        dtype=torch.long,
        device=DEVICE
    )

    output = model(
        phonemes=phonemes,
        phoneme_lengths=phoneme_lengths,
        speaker_ids=speaker_ids,
        ground_truth_durations=None
    )

    mel = output["mel"][0]

    duration_values = output[
        "durations"
    ][0]

    frame_length = int(
        output["frame_lengths"][0]
    )

    mel = mel[:frame_length]

    return (
        mel.cpu().numpy(),
        duration_values.cpu().numpy()
    )


# ============================================================
# VISUALIZATION — TRAINING
# ============================================================

def plot_training_history(history):

    epochs = np.arange(
        1,
        len(history["train_total"]) + 1
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    axes[0].plot(
        epochs,
        history["train_total"],
        label="Train"
    )

    axes[0].plot(
        epochs,
        history["val_total"],
        label="Validation"
    )

    axes[0].set_title("Total Loss")
    axes[0].legend()

    axes[1].plot(
        epochs,
        history["train_mel"],
        label="Train"
    )

    axes[1].plot(
        epochs,
        history["val_mel"],
        label="Validation"
    )

    axes[1].set_title("Mel Reconstruction Loss")
    axes[1].legend()

    axes[2].plot(
        epochs,
        history["train_duration"],
        label="Train"
    )

    axes[2].plot(
        epochs,
        history["val_duration"],
        label="Validation"
    )

    axes[2].set_title("Duration Loss")
    axes[2].legend()

    for axis in axes:

        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True)

    plt.tight_layout()

    plt.savefig(
        "lesson18_training_curves.png",
        dpi=150
    )

    plt.show()


# ============================================================
# VISUALIZATION — SPEAKER COMPARISON
# ============================================================

def plot_speaker_comparison(
    mel_a,
    mel_b,
    speaker_a,
    speaker_b
):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8)
    )

    axes[0].imshow(
        mel_a.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )

    axes[0].set_title(
        f"Same Text — Speaker {speaker_a}"
    )

    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Mel Channel")

    axes[1].imshow(
        mel_b.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )

    axes[1].set_title(
        f"Same Text — Speaker {speaker_b}"
    )

    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Mel Channel")

    plt.tight_layout()

    plt.savefig(
        "lesson18_speaker_comparison.png",
        dpi=150
    )

    plt.show()


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train(
    number_of_speakers=8,
    number_of_samples=800,
    epochs=30,
    batch_size=16,
    learning_rate=2e-4,
    no_plot=False
):

    dataset = MultiSpeakerTTSDataset(
        number_of_speakers=number_of_speakers,
        number_of_samples=number_of_samples
    )

    train_size = int(
        0.85 * len(dataset)
    )

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )

    model = MultiSpeakerTTS(
        number_of_speakers=number_of_speakers
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=1e-6
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4
    )

    history = {
        "train_total": [],
        "val_total": [],
        "train_mel": [],
        "val_mel": [],
        "train_duration": [],
        "val_duration": []
    }

    print("\nMulti-Speaker TTS Training")
    print("--------------------------")
    print("Device:", DEVICE)
    print("Speakers:", number_of_speakers)
    print("Training examples:", train_size)
    print("Validation examples:", val_size)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        start_time = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None
        )

        scheduler.step(
            val_metrics["total"]
        )

        history["train_total"].append(
            train_metrics["total"]
        )

        history["val_total"].append(
            val_metrics["total"]
        )

        history["train_mel"].append(
            train_metrics["mel"]
        )

        history["val_mel"].append(
            val_metrics["mel"]
        )

        history["train_duration"].append(
            train_metrics["duration"]
        )

        history["val_duration"].append(
            val_metrics["duration"]
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train: {train_metrics['total']:.4f} | "
            f"Val: {val_metrics['total']:.4f} | "
            f"Mel: {val_metrics['mel']:.4f} | "
            f"Duration: {val_metrics['duration']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["total"] < best_val_loss:

            best_val_loss = val_metrics["total"]

            checkpoint = {

                "model_state_dict": model.state_dict(),

                "number_of_speakers": number_of_speakers,

                "vocab_size": VOCAB_SIZE,

                "phonemes": PHONEMES,

                "n_mels": N_MELS,

                "hidden_dim": HIDDEN_DIM,

                "embedding_dim": EMBEDDING_DIM,

                "best_validation_loss": best_val_loss,

                "speaker_profiles": dataset.profiles
            }

            torch.save(
                checkpoint,
                "lesson18_multispeaker_tts.pt"
            )

    print("\nTraining complete.")

    print(
        "Best validation loss:",
        round(best_val_loss, 5)
    )

    print(
        "Saved checkpoint:",
        "lesson18_multispeaker_tts.pt"
    )

    # ========================================================
    # INFERENCE DEMONSTRATION
    # ========================================================

    model.eval()

    # Same phoneme sequence for both speakers.
    example_phonemes = [
        PHONEME_TO_ID["HH"],
        PHONEME_TO_ID["EH"],
        PHONEME_TO_ID["L"],
        PHONEME_TO_ID["OW"],
        PHONEME_TO_ID["W"],
        PHONEME_TO_ID["EH"],
        PHONEME_TO_ID["L"],
        PHONEME_TO_ID["K"],
        PHONEME_TO_ID["AH"],
        PHONEME_TO_ID["M"],
    ]

    speaker_a = 0

    speaker_b = min(
        1,
        number_of_speakers - 1
    )

    mel_a, durations_a = synthesize_mel(
        model,
        example_phonemes,
        speaker_a
    )

    mel_b, durations_b = synthesize_mel(
        model,
        example_phonemes,
        speaker_b
    )

    print("\nInference demonstration")
    print("-----------------------")

    print("Phoneme sequence:")

    print([
        ID_TO_PHONEME[index]
        for index in example_phonemes
    ])

    print("\nSpeaker A:", speaker_a)

    print("Predicted durations:", durations_a.tolist())

    print("Generated Mel shape:", mel_a.shape)

    print("\nSpeaker B:", speaker_b)

    print("Predicted durations:", durations_b.tolist())

    print("Generated Mel shape:", mel_b.shape)

    if not no_plot:

        plot_training_history(history)

        plot_speaker_comparison(
            mel_a,
            mel_b,
            speaker_a,
            speaker_b
        )

    return model, dataset, history


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train a multi-speaker TTS acoustic model."
    )

    parser.add_argument(
        "--speakers",
        type=int,
        default=8
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=800
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--no-plot",
        action="store_true"
    )

    args = parser.parse_args()

    if args.speakers < 2:

        parser.error(
            "--speakers must be at least 2"
        )

    if args.samples < 20:

        parser.error(
            "--samples must be at least 20"
        )

    if args.epochs < 1:

        parser.error(
            "--epochs must be at least 1"
        )

    if args.batch_size < 1:

        parser.error(
            "--batch-size must be at least 1"
        )

    set_seed(42)

    train(
        number_of_speakers=args.speakers,
        number_of_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        no_plot=args.no_plot
    )


if __name__ == "__main__":

    main()