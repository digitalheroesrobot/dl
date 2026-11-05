"""
File: phase1_lesson22_learned_reference_attention.py

PHASE 1 — VOICE CLONING FROM SCRATCH
============================================================

LESSON 22 — LEARNED REFERENCE ATTENTION
             AND STRONGER SPEAKER CONDITIONING
============================================================

PURPOSE
-------
Lesson 21 introduced few-shot voice cloning:

    reference_1
    reference_2
    ...
    reference_K
          |
          v
    speaker encoder
          |
          v
    aggregate embeddings
          |
          v
    speaker representation
          |
          v
    TTS

Lesson 22 makes the aggregation more intelligent.

Instead of treating every reference frame equally, the model
learns attention weights over:

    reference utterances
    AND
    frames inside each utterance

The basic idea becomes:

    reference audio
          |
          v
    frame-level speaker features
          |
          v
    learned attention
          |
          v
    speaker embedding
          |
          v
    speaker-conditioned TTS

This is much closer to the type of mechanism needed by a
real reference-based zero-shot / few-shot voice cloning model.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------

STEP 1 — Frame-level speaker features

Instead of immediately averaging the entire utterance:

    Mel
      ↓
    CNN
      ↓
    frame embeddings

we retain:

    e_1, e_2, ..., e_T

for each reference.

STEP 2 — Temporal attention

The model learns:

    alpha_t = softmax(score(e_t))

and computes:

    e_utterance = sum_t alpha_t e_t

Thus, not every frame contributes equally.

STEP 3 — Multiple-reference attention

For K reference utterances:

    e_1
    e_2
    ...
    e_K

the model can learn:

    beta_1
    beta_2
    ...
    beta_K

and compute:

    e_speaker = sum_k beta_k e_k

STEP 4 — Query-based attention

A learned speaker query is used to determine which acoustic
features are useful for identifying the speaker.

Conceptually:

    q = speaker query

    score_t = q^T W e_t

    alpha_t = softmax(score_t)

STEP 5 — Speaker conditioning

The speaker embedding is injected into the TTS network.

Instead of only adding one vector:

    text + speaker

we use learned conditioning layers.

Conceptually:

    h = TextEncoder(text)

    gamma, beta = SpeakerConditioner(speaker)

    h' = gamma * h + beta

This is a simplified form of feature-wise modulation.

STEP 6 — Speaker classification

The embedding is trained to identify training speakers.

STEP 7 — Contrastive learning

Same-speaker references should be close.

Different speakers should be separated.

STEP 8 — Reference dropout

The model randomly receives:

    1 reference
    2 references
    3 references
    ...
    K references

during training.

STEP 9 — Attention visualization

We visualize which reference frames receive high attention.

This is extremely useful for understanding what the network
actually uses.

------------------------------------------------------------
CORE MATHEMATICS
------------------------------------------------------------

Suppose a reference utterance produces frame features:

    E = [e_1, e_2, ..., e_T]

where:

    e_t ∈ R^D

A learned attention score is:

    s_t = v^T tanh(W e_t + b)

Then:

    alpha_t =
        exp(s_t)
        --------------------
        sum_j exp(s_j)

The utterance representation is:

    e_ref =
        sum_t alpha_t e_t

For multiple references:

    e_ref_1
    e_ref_2
    ...
    e_ref_K

we calculate:

    beta_k = softmax(g(e_ref_k))

and:

    e_speaker =
        sum_k beta_k e_ref_k

Finally:

    e_speaker =
        e_speaker / ||e_speaker||_2

------------------------------------------------------------
WHY ATTENTION?
------------------------------------------------------------

Consider a reference recording:

    [silence][speech][speech][noise][speech][silence]

A simple mean treats every frame equally.

That is undesirable.

Ideally:

    silence  -> low weight
    noise    -> low weight
    useful speech -> high weight

The model should learn something approximately like:

    frame 1   0.01
    frame 2   0.03
    frame 3   0.18
    frame 4   0.24
    frame 5   0.30
    frame 6   0.20
    frame 7   0.03
    frame 8   0.01

The exact numbers are not important.

The important concept is:

    NOT ALL REFERENCE AUDIO IS EQUALLY INFORMATIVE.

------------------------------------------------------------
STRONGER SPEAKER CONDITIONING
------------------------------------------------------------

Lesson 21 approximately used:

    h = text_features + speaker_vector

Lesson 22 uses feature modulation:

    gamma = f_gamma(speaker)
    beta  = f_beta(speaker)

    h' = gamma * h + beta

This gives the speaker representation a stronger influence
on the acoustic decoder.

In a production architecture, more sophisticated conditioning
could be used:

    cross-attention
    adaptive layer normalization
    FiLM
    conditional normalization
    prefix conditioning
    speaker tokens
    cross-modal attention

This lesson uses FiLM-style conditioning because it is simple
enough to understand while demonstrating the principle.

------------------------------------------------------------
ARCHITECTURE
------------------------------------------------------------

                Reference A
                     |
                     v
              Frame Encoder
                     |
                     v
             Temporal Attention
                     |
                     v
                 e_ref_A
                     |
                     |
                Reference B
                     |
                     v
              Frame Encoder
                     |
                     v
             Temporal Attention
                     |
                     v
                 e_ref_B
                     |
                     |
                    ...
                     |
                     v
             Reference Attention
                     |
                     v
              Speaker Embedding
                     |
              +------+------+
              |             |
              v             v
       Speaker Classifier  FiLM
                            |
Target Text                 |
    |                       |
    v                       |
Text Encoder ---------------+
    |
    v
Conditioned Decoder
    |
    v
Target Mel

------------------------------------------------------------
SYNTHETIC DATA
------------------------------------------------------------

This lesson continues to use synthetic speech-like Mel
spectrograms.

Examples of language content:

English:
    "Hello, how are you?"

Chinese:
    "今天天气很好。"

Japanese:
    "今日はとてもいい天気です。"

Russian:
    "Сегодня очень хорошая погода."

The text examples are conceptual only.

The actual training data uses synthetic phoneme IDs.

------------------------------------------------------------
IMPORTANT LIMITATIONS
------------------------------------------------------------

This is still an educational model.

It does NOT yet provide production-quality voice cloning.

Real voice cloning requires:

    real speech datasets
    phoneme/token frontend
    high-quality Mel extraction
    speaker encoder training
    robust augmentation
    duration/alignment modeling
    expressive prosody modeling
    neural vocoder
    waveform synthesis
    speaker verification
    perceptual evaluation

The synthetic speaker model is deliberately simplified.

------------------------------------------------------------
EXPERIMENTS
------------------------------------------------------------

Experiment A:
    Compare mean pooling vs learned attention.

Experiment B:
    Measure speaker similarity with 1–5 references.

Experiment C:
    Test noisy references.

Experiment D:
    Inspect temporal attention weights.

Experiment E:
    Evaluate held-out speakers.

Experiment F:
    Visualize speaker embeddings using PCA.

------------------------------------------------------------
ROADMAP
------------------------------------------------------------

Lesson 20
    Disentangled voice cloning

Lesson 21
    Few-shot voice cloning

Lesson 22
    Learned reference attention
        <-- YOU ARE HERE

Lesson 23
    Neural vocoder integration

Lesson 24
    End-to-end voice cloning

Lesson 25+
    Advanced generative speech architectures
    training
    evaluation
    deployment

============================================================
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

OUTPUT_DIR = Path(
    "lesson22_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

N_MELS = 80

VOCAB_SIZE = 64

PAD_ID = 0

MODEL_DIM = 128

SPEAKER_EMBED_DIM = 64

MAX_PHONEMES = 16

NUM_TRAIN_SPEAKERS = 8

NUM_HELDOUT_SPEAKERS = 2

TRAIN_SAMPLES = 640

VAL_SAMPLES = 128

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 2e-3

MAX_REFERENCES = 5

DROPOUT = 0.10

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# SYNTHETIC SPEAKER PROFILE
# ============================================================

@dataclass
class SpeakerProfile:

    spectral_offset: np.ndarray

    resonance_strength: float

    brightness: float

    harmonic_tilt: float

    temporal_style: float


def create_speaker_profile(
    speaker_id: int,
    seed_offset: int = 0,
) -> SpeakerProfile:

    rng = np.random.default_rng(
        SEED
        + seed_offset
        + speaker_id * 1009
    )

    spectral_offset = rng.normal(
        0.0,
        0.08,
        size=N_MELS,
    )

    resonance_strength = rng.uniform(
        0.6,
        1.4,
    )

    brightness = rng.uniform(
        0.75,
        1.25,
    )

    harmonic_tilt = rng.uniform(
        -0.35,
        0.35,
    )

    temporal_style = rng.uniform(
        0.7,
        1.3,
    )

    return SpeakerProfile(
        spectral_offset=spectral_offset.astype(
            np.float32
        ),
        resonance_strength=float(
            resonance_strength
        ),
        brightness=float(
            brightness
        ),
        harmonic_tilt=float(
            harmonic_tilt
        ),
        temporal_style=float(
            temporal_style
        ),
    )


# ============================================================
# PHONEME TEMPLATES
# ============================================================

def create_phoneme_templates():

    rng = np.random.default_rng(
        SEED + 123
    )

    templates = rng.normal(
        0.0,
        0.25,
        size=(
            VOCAB_SIZE,
            N_MELS,
        ),
    ).astype(
        np.float32
    )

    templates[
        PAD_ID
    ] = 0.0

    mel_axis = np.linspace(
        0.0,
        1.0,
        N_MELS,
    ).astype(
        np.float32
    )

    for phoneme_id in range(
        1,
        VOCAB_SIZE,
    ):

        center = rng.uniform(
            0.10,
            0.90,
        )

        width = rng.uniform(
            0.05,
            0.25,
        )

        gaussian = np.exp(
            -0.5
            * (
                (mel_axis - center)
                / width
            ) ** 2
        )

        templates[
            phoneme_id
        ] += (
            0.8
            * gaussian
        )

    return templates


PHONEME_TEMPLATES = (
    create_phoneme_templates()
)


# ============================================================
# SYNTHETIC UTTERANCE GENERATION
# ============================================================

def make_target_utterance(
    phonemes: np.ndarray,
    speaker: SpeakerProfile,
    rng: np.random.Generator,
):

    durations = []

    for _ in phonemes:

        durations.append(
            int(
                rng.integers(
                    4,
                    9,
                )
            )
        )

    durations = np.asarray(
        durations,
        dtype=np.int64,
    )

    frames = []

    mel_axis = np.linspace(
        0.0,
        1.0,
        N_MELS,
    ).astype(
        np.float32
    )

    for phoneme_id, duration in zip(
        phonemes,
        durations,
    ):

        base = PHONEME_TEMPLATES[
            int(phoneme_id)
        ].copy()

        speaker_curve = (
            speaker.resonance_strength
            * np.sin(
                2.0
                * np.pi
                * (
                    mel_axis
                    * (
                        1.0
                        + speaker.brightness
                    )
                )
            )
        )

        brightness_curve = (
            speaker.brightness
            * mel_axis
        )

        harmonic_curve = (
            speaker.harmonic_tilt
            * np.cos(
                2.0
                * np.pi
                * mel_axis
            )
        )

        spectral = (
            base
            + speaker.spectral_offset
            + 0.12
            * speaker_curve
            + 0.08
            * brightness_curve
            + 0.08
            * harmonic_curve
        )

        t = np.linspace(
            0.0,
            1.0,
            duration,
        ).astype(
            np.float32
        )

        envelope = (
            0.85
            + 0.15
            * np.sin(
                np.pi * t
            )
        )

        for frame_index in range(
            duration
        ):

            noise = rng.normal(
                0.0,
                0.025,
                size=N_MELS,
            ).astype(
                np.float32
            )

            frame = (
                spectral.copy()
                + noise
            )

            frame *= envelope[
                frame_index
            ]

            frames.append(
                frame
            )

    mel = np.stack(
        frames,
        axis=0,
    ).astype(
        np.float32
    )

    mel = np.tanh(
        mel
    )

    return (
        mel.astype(np.float32),
        durations.astype(np.int64),
    )


# ============================================================
# REFERENCE AUGMENTATION
# ============================================================

def augment_reference(
    mel: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
):

    x = mel.copy()

    amplitude = rng.uniform(
        0.90,
        1.10,
    )

    x *= amplitude

    noise_std = (
        rng.uniform(
            0.005,
            0.035,
        )
        * strength
    )

    x += rng.normal(
        0.0,
        noise_std,
        size=x.shape,
    ).astype(
        np.float32
    )

    frequency_axis = np.linspace(
        -1.0,
        1.0,
        x.shape[1],
    ).astype(
        np.float32
    )

    tilt = (
        rng.uniform(
            -0.05,
            0.05,
        )
        * strength
    )

    x += (
        tilt
        * frequency_axis[
            None,
            :
        ]
    )

    return np.clip(
        x,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

class AttentionVoiceDataset(
    Dataset
):

    def __init__(
        self,
        num_samples,
        speaker_ids,
        seed_offset,
        augment_references,
    ):

        self.num_samples = (
            num_samples
        )

        self.speaker_ids = (
            speaker_ids
        )

        self.seed_offset = (
            seed_offset
        )

        self.augment_references = (
            augment_references
        )

    def __len__(self):

        return self.num_samples

    def __getitem__(
        self,
        index,
    ):

        rng = np.random.default_rng(
            SEED
            + self.seed_offset
            + index * 7919
        )

        speaker_id = int(
            rng.choice(
                self.speaker_ids
            )
        )

        speaker = (
            create_speaker_profile(
                speaker_id,
                seed_offset=5000,
            )
        )

        references = []

        for _ in range(
            MAX_REFERENCES
        ):

            length = int(
                rng.integers(
                    5,
                    MAX_PHONEMES + 1,
                )
            )

            phonemes = rng.integers(
                1,
                VOCAB_SIZE,
                size=length,
            )

            mel, _ = (
                make_target_utterance(
                    phonemes,
                    speaker,
                    rng,
                )
            )

            if self.augment_references:

                mel = augment_reference(
                    mel,
                    rng,
                    strength=1.0,
                )

            references.append(
                mel
            )

        # Target content is independent
        # from reference content.

        target_length = int(
            rng.integers(
                6,
                MAX_PHONEMES + 1,
            )
        )

        target_phonemes = rng.integers(
            1,
            VOCAB_SIZE,
            size=target_length,
        )

        target_mel, target_durations = (
            make_target_utterance(
                target_phonemes,
                speaker,
                rng,
            )
        )

        return {
            "references": references,
            "target_phonemes":
                target_phonemes.astype(
                    np.int64
                ),
            "target_mel":
                target_mel.astype(
                    np.float32
                ),
            "target_durations":
                target_durations.astype(
                    np.int64
                ),
            "speaker_id":
                speaker_id,
        }


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    batch_size = len(
        batch
    )

    # --------------------------------------------------------
    # Reference sequences
    # --------------------------------------------------------

    references = []

    reference_masks = []

    for k in range(
        MAX_REFERENCES
    ):

        max_length = max(
            item[
                "references"
            ][k].shape[0]
            for item in batch
        )

        tensor = torch.zeros(
            batch_size,
            max_length,
            N_MELS,
            dtype=torch.float32,
        )

        mask = torch.zeros(
            batch_size,
            max_length,
            dtype=torch.bool,
        )

        for b, item in enumerate(
            batch
        ):

            x = torch.from_numpy(
                item[
                    "references"
                ][k]
            )

            length = x.shape[0]

            tensor[
                b,
                :length
            ] = x

            mask[
                b,
                :length
            ] = True

        references.append(
            tensor
        )

        reference_masks.append(
            mask
        )

    # --------------------------------------------------------
    # Target phonemes
    # --------------------------------------------------------

    max_phonemes = max(
        len(
            item[
                "target_phonemes"
            ]
        )
        for item in batch
    )

    target_phonemes = torch.full(
        (
            batch_size,
            max_phonemes,
        ),
        PAD_ID,
        dtype=torch.long,
    )

    phoneme_mask = torch.zeros(
        batch_size,
        max_phonemes,
        dtype=torch.bool,
    )

    target_durations = torch.zeros(
        batch_size,
        max_phonemes,
        dtype=torch.long,
    )

    for b, item in enumerate(
        batch
    ):

        n = len(
            item[
                "target_phonemes"
            ]
        )

        target_phonemes[
            b,
            :n
        ] = torch.from_numpy(
            item[
                "target_phonemes"
            ]
        )

        target_durations[
            b,
            :n
        ] = torch.from_numpy(
            item[
                "target_durations"
            ]
        )

        phoneme_mask[
            b,
            :n
        ] = True

    # --------------------------------------------------------
    # Target Mel
    # --------------------------------------------------------

    max_frames = max(
        item[
            "target_mel"
        ].shape[0]
        for item in batch
    )

    target_mel = torch.zeros(
        batch_size,
        max_frames,
        N_MELS,
        dtype=torch.float32,
    )

    target_mel_mask = torch.zeros(
        batch_size,
        max_frames,
        dtype=torch.bool,
    )

    for b, item in enumerate(
        batch
    ):

        n = item[
            "target_mel"
        ].shape[0]

        target_mel[
            b,
            :n
        ] = torch.from_numpy(
            item[
                "target_mel"
            ]
        )

        target_mel_mask[
            b,
            :n
        ] = True

    speaker_ids = torch.tensor(
        [
            item[
                "speaker_id"
            ]
            for item in batch
        ],
        dtype=torch.long,
    )

    return {
        "references":
            references,

        "reference_masks":
            reference_masks,

        "target_phonemes":
            target_phonemes,

        "phoneme_mask":
            phoneme_mask,

        "target_durations":
            target_durations,

        "target_mel":
            target_mel,

        "target_mel_mask":
            target_mel_mask,

        "speaker_ids":
            speaker_ids,
    }


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(
    nn.Module
):

    def __init__(
        self,
        dim,
        max_len=512,
    ):

        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                dim,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / dim
            )
        )

        pe = torch.zeros(
            max_len,
            dim,
        )

        pe[
            :,
            0::2
        ] = torch.sin(
            position
            * div_term
        )

        pe[
            :,
            1::2
        ] = torch.cos(
            position
            * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
        self,
        x,
    ):

        return (
            x
            + self.pe[
                :,
                :x.shape[1]
            ]
        )


# ============================================================
# FRAME ENCODER
# ============================================================

class FrameEncoder(
    nn.Module
):

    def __init__(
        self,
        mel_dim,
        embedding_dim,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                mel_dim,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                embedding_dim,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )

    def forward(
        self,
        mel,
    ):

        x = mel.transpose(
            1,
            2,
        )

        x = self.network(
            x
        )

        return x.transpose(
            1,
            2,
        )


# ============================================================
# TEMPORAL ATTENTION
# ============================================================

class TemporalAttention(
    nn.Module
):

    def __init__(
        self,
        embedding_dim,
    ):

        super().__init__()

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.query = nn.Parameter(
            torch.randn(
                embedding_dim
            )
        )

        self.scale = (
            embedding_dim
            ** -0.5
        )

    def forward(
        self,
        features,
        mask,
    ):
        """
        features:
            [B, T, D]

        mask:
            [B, T]
        """

        keys = torch.tanh(
            self.key(
                features
            )
        )

        values = self.value(
            features
        )

        query = self.query.view(
            1,
            1,
            -1,
        )

        scores = (
            keys
            * query
        ).sum(
            dim=-1
        )

        scores *= self.scale

        scores = scores.masked_fill(
            ~mask,
            -1e9,
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        pooled = (
            values
            * weights.unsqueeze(-1)
        ).sum(
            dim=1
        )

        pooled = F.normalize(
            pooled,
            p=2,
            dim=-1,
        )

        return (
            pooled,
            weights,
        )


# ============================================================
# REFERENCE-LEVEL ATTENTION
# ============================================================

class ReferenceAttention(
    nn.Module
):

    def __init__(
        self,
        embedding_dim,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                embedding_dim,
                1,
            ),
        )

    def forward(
        self,
        embeddings,
        reference_mask,
    ):
        """
        embeddings:
            [B, K, D]

        reference_mask:
            [B, K]
        """

        scores = self.network(
            embeddings
        ).squeeze(-1)

        scores = scores.masked_fill(
            ~reference_mask,
            -1e9,
        )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        speaker_embedding = (
            embeddings
            * weights.unsqueeze(-1)
        ).sum(
            dim=1
        )

        speaker_embedding = F.normalize(
            speaker_embedding,
            p=2,
            dim=-1,
        )

        return (
            speaker_embedding,
            weights,
        )


# ============================================================
# FULL SPEAKER ENCODER
# ============================================================

class LearnedSpeakerEncoder(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()

        self.frame_encoder = (
            FrameEncoder(
                N_MELS,
                SPEAKER_EMBED_DIM,
            )
        )

        self.temporal_attention = (
            TemporalAttention(
                SPEAKER_EMBED_DIM
            )
        )

        self.reference_attention = (
            ReferenceAttention(
                SPEAKER_EMBED_DIM
            )
        )

    def encode_one_reference(
        self,
        mel,
        mask,
    ):

        frame_features = (
            self.frame_encoder(
                mel
            )
        )

        embedding, weights = (
            self.temporal_attention(
                frame_features,
                mask,
            )
        )

        return (
            embedding,
            weights,
            frame_features,
        )

    def forward(
        self,
        references,
        reference_masks,
    ):

        utterance_embeddings = []

        temporal_weights = []

        frame_features = []

        for mel, mask in zip(
            references,
            reference_masks,
        ):

            (
                embedding,
                weights,
                features,
            ) = self.encode_one_reference(
                mel,
                mask,
            )

            utterance_embeddings.append(
                embedding
            )

            temporal_weights.append(
                weights
            )

            frame_features.append(
                features
            )

        utterance_embeddings = (
            torch.stack(
                utterance_embeddings,
                dim=1,
            )
        )

        batch_size = (
            utterance_embeddings.shape[0]
        )

        reference_mask = torch.ones(
            batch_size,
            utterance_embeddings.shape[1],
            dtype=torch.bool,
            device=utterance_embeddings.device,
        )

        (
            speaker_embedding,
            reference_weights,
        ) = self.reference_attention(
            utterance_embeddings,
            reference_mask,
        )

        return {
            "speaker_embedding":
                speaker_embedding,

            "utterance_embeddings":
                utterance_embeddings,

            "temporal_weights":
                temporal_weights,

            "reference_weights":
                reference_weights,

            "frame_features":
                frame_features,
        }


# ============================================================
# TEXT ENCODER
# ============================================================

class TextEncoder(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        model_dim,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            model_dim,
            padding_idx=PAD_ID,
        )

        self.position = (
            PositionalEncoding(
                model_dim
            )
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=4,
                dim_feedforward=
                    model_dim * 4,
                dropout=DROPOUT,
                batch_first=True,
                activation="gelu",
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                layer,
                num_layers=2,
            )
        )

    def forward(
        self,
        tokens,
    ):

        x = self.embedding(
            tokens
        )

        x = self.position(
            x
        )

        padding_mask = (
            tokens == PAD_ID
        )

        return self.encoder(
            x,
            src_key_padding_mask=
                padding_mask,
        )


# ============================================================
# DURATION PREDICTOR
# ============================================================

class DurationPredictor(
    nn.Module
):

    def __init__(
        self,
        model_dim,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                model_dim,
                model_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                model_dim,
                model_dim // 2,
            ),

            nn.ReLU(),

            nn.Linear(
                model_dim // 2,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.network(
            x
        ).squeeze(-1)


# ============================================================
# LENGTH REGULATOR
# ============================================================

def length_regulate(
    x,
    durations,
):

    outputs = []

    for b in range(
        x.shape[0]
    ):

        pieces = []

        for n in range(
            x.shape[1]
        ):

            duration = int(
                durations[
                    b,
                    n,
                ].item()
            )

            if duration <= 0:
                continue

            pieces.append(
                x[
                    b,
                    n,
                ].unsqueeze(0).repeat(
                    duration,
                    1,
                )
            )

        if not pieces:

            pieces.append(
                x[
                    b,
                    :1,
                ]
            )

        outputs.append(
            torch.cat(
                pieces,
                dim=0,
            )
        )

    return outputs


# ============================================================
# FILM SPEAKER CONDITIONER
# ============================================================

class FiLMSpeakerConditioner(
    nn.Module
):

    def __init__(
        self,
        speaker_dim,
        model_dim,
    ):

        super().__init__()

        self.gamma = nn.Sequential(

            nn.Linear(
                speaker_dim,
                model_dim,
            ),

            nn.Tanh(),
        )

        self.beta = nn.Linear(
            speaker_dim,
            model_dim,
        )

    def forward(
        self,
        speaker_embedding,
    ):

        gamma = (
            1.0
            + self.gamma(
                speaker_embedding
            )
        )

        beta = self.beta(
            speaker_embedding
        )

        return (
            gamma,
            beta,
        )


# ============================================================
# TTS MODEL
# ============================================================

class AttentionVoiceCloningTTS(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()

        self.speaker_encoder = (
            LearnedSpeakerEncoder()
        )

        self.text_encoder = (
            TextEncoder(
                VOCAB_SIZE,
                MODEL_DIM,
            )
        )

        self.duration_predictor = (
            DurationPredictor(
                MODEL_DIM
            )
        )

        self.speaker_projection = (
            nn.Linear(
                SPEAKER_EMBED_DIM,
                MODEL_DIM,
            )
        )

        self.film = (
            FiLMSpeakerConditioner(
                SPEAKER_EMBED_DIM,
                MODEL_DIM,
            )
        )

        decoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=MODEL_DIM,
                nhead=4,
                dim_feedforward=
                    MODEL_DIM * 4,
                dropout=DROPOUT,
                batch_first=True,
                activation="gelu",
            )
        )

        self.decoder = (
            nn.TransformerEncoder(
                decoder_layer,
                num_layers=2,
            )
        )

        self.mel_projection = (
            nn.Linear(
                MODEL_DIM,
                N_MELS,
            )
        )

        self.speaker_classifier = (
            nn.Linear(
                SPEAKER_EMBED_DIM,
                NUM_TRAIN_SPEAKERS,
            )
        )

    def forward(
        self,
        references,
        reference_masks,
        target_phonemes,
        target_durations,
    ):

        speaker_outputs = (
            self.speaker_encoder(
                references,
                reference_masks,
            )
        )

        speaker_embedding = (
            speaker_outputs[
                "speaker_embedding"
            ]
        )

        text_features = (
            self.text_encoder(
                target_phonemes
            )
        )

        # ----------------------------------------------------
        # Strong speaker conditioning.
        # ----------------------------------------------------

        gamma, beta = self.film(
            speaker_embedding
        )

        text_features = (
            text_features
            * gamma.unsqueeze(1)
            + beta.unsqueeze(1)
        )

        speaker_bias = (
            self.speaker_projection(
                speaker_embedding
            )
            .unsqueeze(1)
        )

        text_features = (
            text_features
            + speaker_bias
        )

        expanded = length_regulate(
            text_features,
            target_durations,
        )

        max_length = max(
            x.shape[0]
            for x in expanded
        )

        batch_size = len(
            expanded
        )

        padded = torch.zeros(
            batch_size,
            max_length,
            MODEL_DIM,
            device=text_features.device,
        )

        mask = torch.zeros(
            batch_size,
            max_length,
            dtype=torch.bool,
            device=text_features.device,
        )

        for b, x in enumerate(
            expanded
        ):

            length = x.shape[0]

            padded[
                b,
                :length
            ] = x

            mask[
                b,
                :length
            ] = True

        decoded = self.decoder(
            padded,
            src_key_padding_mask=
                ~mask,
        )

        mel_prediction = (
            self.mel_projection(
                decoded
            )
        )

        speaker_logits = (
            self.speaker_classifier(
                speaker_embedding
            )
        )

        return {
            **speaker_outputs,

            "mel_prediction":
                mel_prediction,

            "mel_mask":
                mask,

            "speaker_logits":
                speaker_logits,
        }


# ============================================================
# LOSSES
# ============================================================

def masked_mel_l1(
    prediction,
    target,
    target_mask,
):

    time_length = min(
        prediction.shape[1],
        target.shape[1],
    )

    prediction = prediction[
        :,
        :time_length,
    ]

    target = target[
        :,
        :time_length,
    ]

    mask = target_mask[
        :,
        :time_length,
    ]

    error = (
        prediction
        - target
    ).abs()

    mask = (
        mask
        .unsqueeze(-1)
        .float()
    )

    numerator = (
        error
        * mask
    ).sum()

    denominator = (
        mask.sum()
        * prediction.shape[-1]
    ).clamp_min(1.0)

    return (
        numerator
        / denominator
    )


def duration_loss(
    prediction,
    target,
    mask,
):

    prediction = prediction[
        mask
    ]

    target = target[
        mask
    ].float()

    target_log = torch.log(
        target.clamp_min(1.0)
    )

    return F.smooth_l1_loss(
        prediction,
        target_log,
    )


def speaker_classification_loss(
    logits,
    speaker_ids,
):

    return F.cross_entropy(
        logits,
        speaker_ids,
    )


def reference_consistency_loss(
    embeddings,
):

    """
    embeddings:
        [B, K, D]
    """

    prototype = F.normalize(
        embeddings.mean(
            dim=1
        ),
        p=2,
        dim=-1,
    )

    prototype = (
        prototype
        .unsqueeze(1)
        .expand_as(
            embeddings
        )
    )

    cosine = F.cosine_similarity(
        embeddings,
        prototype,
        dim=-1,
    )

    return (
        1.0
        - cosine
    ).mean()


def contrastive_speaker_loss(
    embeddings,
    speaker_ids,
    temperature=0.10,
):

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    similarity = (
        embeddings
        @ embeddings.T
    ) / temperature

    same_speaker = (
        speaker_ids.unsqueeze(0)
        ==
        speaker_ids.unsqueeze(1)
    )

    diagonal = torch.eye(
        embeddings.shape[0],
        dtype=torch.bool,
        device=embeddings.device,
    )

    valid = ~diagonal

    losses = []

    for i in range(
        embeddings.shape[0]
    ):

        logits = similarity[
            i
        ][valid[i]]

        positives = (
            same_speaker[
                i
            ][valid[i]]
        )

        if positives.sum() == 0:
            continue

        log_prob = (
            logits
            - torch.logsumexp(
                logits,
                dim=0,
            )
        )

        losses.append(
            -log_prob[
                positives
            ].mean()
        )

    if not losses:

        return torch.tensor(
            0.0,
            device=embeddings.device,
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
):

    model.train()

    totals = {
        "loss": 0.0,
        "mel": 0.0,
        "duration": 0.0,
        "speaker": 0.0,
        "consistency": 0.0,
        "contrastive": 0.0,
    }

    correct = 0

    count = 0

    for batch in loader:

        references = [
            x.to(DEVICE)
            for x in batch[
                "references"
            ]
        ]

        reference_masks = [
            x.to(DEVICE)
            for x in batch[
                "reference_masks"
            ]
        ]

        target_phonemes = (
            batch[
                "target_phonemes"
            ].to(DEVICE)
        )

        target_durations = (
            batch[
                "target_durations"
            ].to(DEVICE)
        )

        target_mel = (
            batch[
                "target_mel"
            ].to(DEVICE)
        )

        target_mel_mask = (
            batch[
                "target_mel_mask"
            ].to(DEVICE)
        )

        phoneme_mask = (
            batch[
                "phoneme_mask"
            ].to(DEVICE)
        )

        speaker_ids = (
            batch[
                "speaker_ids"
            ].to(DEVICE)
        )

        # ----------------------------------------------------
        # Random reference dropout.
        # ----------------------------------------------------

        num_references = random.randint(
            1,
            MAX_REFERENCES,
        )

        outputs = model(
            references[
                :num_references
            ],
            reference_masks[
                :num_references
            ],
            target_phonemes,
            target_durations,
        )

        mel_loss = masked_mel_l1(
            outputs[
                "mel_prediction"
            ],
            target_mel,
            target_mel_mask,
        )

        # ----------------------------------------------------
        # Duration prediction.
        # ----------------------------------------------------

        text_features = (
            model.text_encoder(
                target_phonemes
            )
        )

        duration_prediction = (
            model.duration_predictor(
                text_features
            )
        )

        dur_loss = duration_loss(
            duration_prediction,
            target_durations,
            phoneme_mask,
        )

        speaker_loss = (
            speaker_classification_loss(
                outputs[
                    "speaker_logits"
                ],
                speaker_ids,
            )
        )

        consistency_loss = (
            reference_consistency_loss(
                outputs[
                    "utterance_embeddings"
                ]
            )
        )

        contrastive_loss = (
            contrastive_speaker_loss(
                outputs[
                    "speaker_embedding"
                ],
                speaker_ids,
            )
        )

        loss = (
            mel_loss
            + 0.15 * dur_loss
            + 0.20 * speaker_loss
            + 0.20 * consistency_loss
            + 0.20 * contrastive_loss
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        totals["loss"] += (
            loss.item()
        )

        totals["mel"] += (
            mel_loss.item()
        )

        totals["duration"] += (
            dur_loss.item()
        )

        totals["speaker"] += (
            speaker_loss.item()
        )

        totals["consistency"] += (
            consistency_loss.item()
        )

        totals["contrastive"] += (
            contrastive_loss.item()
        )

        prediction = (
            outputs[
                "speaker_logits"
            ].argmax(
                dim=-1
            )
        )

        correct += (
            prediction == speaker_ids
        ).sum().item()

        count += (
            speaker_ids.shape[0]
        )

    n = len(loader)

    return {
        key:
            value / n
        for key, value
        in totals.items()
    } | {
        "accuracy":
            correct / max(
                count,
                1,
            )
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0

    total_mel = 0.0

    correct = 0

    count = 0

    for batch in loader:

        references = [
            x.to(DEVICE)
            for x in batch[
                "references"
            ]
        ]

        reference_masks = [
            x.to(DEVICE)
            for x in batch[
                "reference_masks"
            ]
        ]

        target_phonemes = (
            batch[
                "target_phonemes"
            ].to(DEVICE)
        )

        target_durations = (
            batch[
                "target_durations"
            ].to(DEVICE)
        )

        target_mel = (
            batch[
                "target_mel"
            ].to(DEVICE)
        )

        target_mel_mask = (
            batch[
                "target_mel_mask"
            ].to(DEVICE)
        )

        speaker_ids = (
            batch[
                "speaker_ids"
            ].to(DEVICE)
        )

        outputs = model(
            references,
            reference_masks,
            target_phonemes,
            target_durations,
        )

        mel_loss = masked_mel_l1(
            outputs[
                "mel_prediction"
            ],
            target_mel,
            target_mel_mask,
        )

        speaker_loss = (
            speaker_classification_loss(
                outputs[
                    "speaker_logits"
                ],
                speaker_ids,
            )
        )

        loss = (
            mel_loss
            + 0.20
            * speaker_loss
        )

        total_loss += (
            loss.item()
        )

        total_mel += (
            mel_loss.item()
        )

        prediction = (
            outputs[
                "speaker_logits"
            ].argmax(
                dim=-1
            )
        )

        correct += (
            prediction == speaker_ids
        ).sum().item()

        count += (
            speaker_ids.shape[0]
        )

    n = len(loader)

    return {
        "loss":
            total_loss / n,

        "mel":
            total_mel / n,

        "accuracy":
            correct
            / max(
                count,
                1,
            ),
    }


# ============================================================
# ATTENTION EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_reference_counts(
    model,
    speaker_ids,
    repetitions=20,
):

    model.eval()

    results = {
        k: []
        for k in range(
            1,
            MAX_REFERENCES + 1,
        )
    }

    for speaker_id in speaker_ids:

        for repetition in range(
            repetitions
        ):

            rng = np.random.default_rng(
                200000
                + speaker_id * 100
                + repetition
            )

            speaker = (
                create_speaker_profile(
                    speaker_id,
                    seed_offset=5000,
                )
            )

            embeddings = []

            for _ in range(
                MAX_REFERENCES
            ):

                length = int(
                    rng.integers(
                        5,
                        MAX_PHONEMES + 1,
                    )
                )

                phonemes = rng.integers(
                    1,
                    VOCAB_SIZE,
                    size=length,
                )

                mel, _ = (
                    make_target_utterance(
                        phonemes,
                        speaker,
                        rng,
                    )
                )

                x = torch.from_numpy(
                    mel
                ).unsqueeze(0).to(
                    DEVICE
                )

                mask = torch.ones(
                    1,
                    mel.shape[0],
                    dtype=torch.bool,
                    device=DEVICE,
                )

                output = (
                    model.speaker_encoder
                    .encode_one_reference(
                        x,
                        mask,
                    )
                )

                embeddings.append(
                    output[0]
                )

            embeddings = torch.cat(
                embeddings,
                dim=0,
            )

            for k in range(
                1,
                MAX_REFERENCES + 1,
            ):

                subset = (
                    embeddings[
                        :k
                    ]
                )

                prototype = F.normalize(
                    subset.mean(
                        dim=0,
                        keepdim=True,
                    ),
                    p=2,
                    dim=-1,
                )

                similarity = (
                    F.cosine_similarity(
                        subset,
                        prototype,
                        dim=-1,
                    )
                    .mean()
                    .item()
                )

                results[k].append(
                    similarity
                )

    return results


# ============================================================
# NOISE ROBUSTNESS
# ============================================================

@torch.no_grad()
def evaluate_noise_robustness(
    model,
    speaker_ids,
    strengths,
    repetitions=15,
):

    model.eval()

    results = {
        s: []
        for s in strengths
    }

    for speaker_id in speaker_ids:

        for repetition in range(
            repetitions
        ):

            rng = np.random.default_rng(
                300000
                + speaker_id * 100
                + repetition
            )

            speaker = (
                create_speaker_profile(
                    speaker_id,
                    seed_offset=5000,
                )
            )

            clean_embeddings = []

            noisy_mels = []

            for _ in range(
                MAX_REFERENCES
            ):

                length = int(
                    rng.integers(
                        5,
                        MAX_PHONEMES + 1,
                    )
                )

                phonemes = rng.integers(
                    1,
                    VOCAB_SIZE,
                    size=length,
                )

                mel, _ = (
                    make_target_utterance(
                        phonemes,
                        speaker,
                        rng,
                    )
                )

                x = torch.from_numpy(
                    mel
                ).unsqueeze(0).to(
                    DEVICE
                )

                mask = torch.ones(
                    1,
                    mel.shape[0],
                    dtype=torch.bool,
                    device=DEVICE,
                )

                output = (
                    model.speaker_encoder
                    .encode_one_reference(
                        x,
                        mask,
                    )
                )

                clean_embeddings.append(
                    output[0]
                )

                noisy_mels.append(
                    mel
                )

            clean_embeddings = (
                torch.cat(
                    clean_embeddings,
                    dim=0,
                )
            )

            clean_prototype = (
                F.normalize(
                    clean_embeddings.mean(
                        dim=0,
                        keepdim=True,
                    ),
                    p=2,
                    dim=-1,
                )
            )

            for strength in strengths:

                noisy_embeddings = []

                for mel in noisy_mels:

                    noisy = (
                        augment_reference(
                            mel,
                            rng,
                            strength,
                        )
                    )

                    x = (
                        torch.from_numpy(
                            noisy
                        )
                        .unsqueeze(0)
                        .to(DEVICE)
                    )

                    mask = torch.ones(
                        1,
                        noisy.shape[0],
                        dtype=torch.bool,
                        device=DEVICE,
                    )

                    output = (
                        model.speaker_encoder
                        .encode_one_reference(
                            x,
                            mask,
                        )
                    )

                    noisy_embeddings.append(
                        output[0]
                    )

                noisy_embeddings = (
                    torch.cat(
                        noisy_embeddings,
                        dim=0,
                    )
                )

                noisy_prototype = (
                    F.normalize(
                        noisy_embeddings.mean(
                            dim=0,
                            keepdim=True,
                        ),
                        p=2,
                        dim=-1,
                    )
                )

                similarity = (
                    F.cosine_similarity(
                        clean_prototype,
                        noisy_prototype,
                        dim=-1,
                    )
                    .item()
                )

                results[
                    strength
                ].append(
                    similarity
                )

    return results


# ============================================================
# PCA
# ============================================================

def compute_pca(
    x,
    components=2,
):

    x = x - x.mean(
        axis=0,
        keepdims=True,
    )

    _, _, vh = np.linalg.svd(
        x,
        full_matrices=False,
    )

    return (
        x
        @ vh[:components].T
    )


# ============================================================
# HELD-OUT SPEAKER EMBEDDINGS
# ============================================================

@torch.no_grad()
def collect_speaker_embeddings(
    model,
    speaker_ids,
    references_per_speaker=5,
):

    model.eval()

    embeddings = []

    labels = []

    for speaker_id in speaker_ids:

        rng = np.random.default_rng(
            400000
            + speaker_id
        )

        speaker = (
            create_speaker_profile(
                speaker_id,
                seed_offset=5000,
            )
        )

        for _ in range(
            references_per_speaker
        ):

            length = int(
                rng.integers(
                    5,
                    MAX_PHONEMES + 1,
                )
            )

            phonemes = rng.integers(
                1,
                VOCAB_SIZE,
                size=length,
            )

            mel, _ = (
                make_target_utterance(
                    phonemes,
                    speaker,
                    rng,
                )
            )

            x = torch.from_numpy(
                mel
            ).unsqueeze(0).to(
                DEVICE
            )

            mask = torch.ones(
                1,
                mel.shape[0],
                dtype=torch.bool,
                device=DEVICE,
            )

            embedding = (
                model.speaker_encoder
                .encode_one_reference(
                    x,
                    mask,
                )[0]
            )

            embeddings.append(
                embedding.squeeze(0)
                .cpu()
                .numpy()
            )

            labels.append(
                speaker_id
            )

    return (
        np.asarray(
            embeddings
        ),
        np.asarray(
            labels
        ),
    )


# ============================================================
# PLOT TRAINING
# ============================================================

def plot_training(
    history,
):

    epochs = np.arange(
        1,
        len(
            history[
                "train_loss"
            ]
        ) + 1,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history[
            "train_loss"
        ],
        label="Train loss",
    )

    plt.plot(
        epochs,
        history[
            "val_loss"
        ],
        label="Validation loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Lesson 22 — Training Curves"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "lesson22_training_curves.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# PLOT REFERENCE COUNT
# ============================================================

def plot_reference_count(
    results,
):

    counts = list(
        results.keys()
    )

    means = [
        np.mean(
            results[k]
        )
        for k in counts
    ]

    stds = [
        np.std(
            results[k]
        )
        for k in counts
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.errorbar(
        counts,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
    )

    plt.xlabel(
        "Number of references"
    )

    plt.ylabel(
        "Cosine similarity"
    )

    plt.title(
        "Few-Shot Speaker Stability"
    )

    plt.xticks(
        counts
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "lesson22_reference_count.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# PLOT NOISE ROBUSTNESS
# ============================================================

def plot_noise(
    results,
):

    levels = list(
        results.keys()
    )

    means = [
        np.mean(
            results[x]
        )
        for x in levels
    ]

    stds = [
        np.std(
            results[x]
        )
        for x in levels
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.errorbar(
        levels,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
    )

    plt.xlabel(
        "Reference augmentation strength"
    )

    plt.ylabel(
        "Cosine similarity"
    )

    plt.title(
        "Learned Attention — Reference Robustness"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "lesson22_noise_robustness.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# PLOT TEMPORAL ATTENTION
# ============================================================

def plot_temporal_attention(
    weights,
    mask,
):

    weights = (
        weights
        .detach()
        .cpu()
        .numpy()
    )

    mask = (
        mask
        .detach()
        .cpu()
        .numpy()
    )

    length = int(
        mask.sum()
    )

    weights = weights[
        :length
    ]

    plt.figure(
        figsize=(11, 4)
    )

    plt.plot(
        np.arange(length),
        weights,
        linewidth=2,
    )

    plt.xlabel(
        "Reference frame"
    )

    plt.ylabel(
        "Attention weight"
    )

    plt.title(
        "Temporal Speaker Attention"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "lesson22_temporal_attention.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# PLOT REFERENCE ATTENTION
# ============================================================

def plot_reference_attention(
    weights,
):

    weights = (
        weights
        .detach()
        .cpu()
        .numpy()
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.imshow(
        weights,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        label="Attention weight"
    )

    plt.xlabel(
        "Reference index"
    )

    plt.ylabel(
        "Batch sample"
    )

    plt.title(
        "Reference-Level Attention"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "lesson22_reference_attention.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# PLOT PCA
# ============================================================

def plot_pca(
    embeddings,
    labels,
):

    projected = compute_pca(
        embeddings
    )

    plt.figure(
        figsize=(9, 7)
    )

    for speaker_id in sorted(
        np.unique(labels)
    ):

        mask = (
            labels
            == speaker_id
        )

        plt.scatter(
            projected[
                mask,
                0
            ],
            projected[
                mask,
                1
            ],
            label=f"Speaker {speaker_id}",
            alpha=0.75,
        )

    plt.xlabel(
        "PCA component 1"
    )

    plt.ylabel(
        "PCA component 2"
    )

    plt.title(
        "Speaker Embedding PCA"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "lesson22_speaker_pca.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 1 — LESSON 22"
    )

    print(
        "Learned Reference Attention "
        "and Stronger Speaker Conditioning"
    )

    print("=" * 72)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Output: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_dataset = (
        AttentionVoiceDataset(
            num_samples=TRAIN_SAMPLES,
            speaker_ids=list(
                range(
                    NUM_TRAIN_SPEAKERS
                )
            ),
            seed_offset=0,
            augment_references=True,
        )
    )

    val_dataset = (
        AttentionVoiceDataset(
            num_samples=VAL_SAMPLES,
            speaker_ids=list(
                range(
                    NUM_TRAIN_SPEAKERS
                )
            ),
            seed_offset=100000,
            augment_references=False,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = (
        AttentionVoiceCloningTTS()
        .to(DEVICE)
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
            )
        )

        val_metrics = validate(
            model,
            val_loader,
        )

        scheduler.step()

        history[
            "train_loss"
        ].append(
            train_metrics[
                "loss"
            ]
        )

        history[
            "val_loss"
        ].append(
            val_metrics[
                "loss"
            ]
        )

        history[
            "train_accuracy"
        ].append(
            train_metrics[
                "accuracy"
            ]
        )

        history[
            "val_accuracy"
        ].append(
            val_metrics[
                "accuracy"
            ]
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} | "
            f"train="
            f"{train_metrics['loss']:.4f} | "
            f"val="
            f"{val_metrics['loss']:.4f} | "
            f"mel="
            f"{train_metrics['mel']:.4f} | "
            f"speaker_acc="
            f"{val_metrics['accuracy']:.3f} | "
            f"attention_consistency="
            f"{train_metrics['consistency']:.4f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    checkpoint_path = (
        OUTPUT_DIR
        / "lesson22_learned_attention.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "config": {
                "n_mels":
                    N_MELS,

                "vocab_size":
                    VOCAB_SIZE,

                "model_dim":
                    MODEL_DIM,

                "speaker_embed_dim":
                    SPEAKER_EMBED_DIM,

                "max_references":
                    MAX_REFERENCES,
            },
        },
        checkpoint_path,
    )

    print()

    print(
        f"Checkpoint saved: "
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------
    # TRAINING PLOT
    # --------------------------------------------------------

    plot_training(
        history
    )

    # --------------------------------------------------------
    # REFERENCE COUNT
    # --------------------------------------------------------

    print()

    print(
        "Evaluating reference-count "
        "robustness..."
    )

    reference_results = (
        evaluate_reference_counts(
            model,
            list(
                range(
                    NUM_TRAIN_SPEAKERS
                )
            ),
        )
    )

    for k in reference_results:

        mean = np.mean(
            reference_results[k]
        )

        std = np.std(
            reference_results[k]
        )

        print(
            f"  K={k}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    plot_reference_count(
        reference_results
    )

    # --------------------------------------------------------
    # NOISE
    # --------------------------------------------------------

    print()

    print(
        "Evaluating reference robustness..."
    )

    noise_levels = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ]

    noise_results = (
        evaluate_noise_robustness(
            model,
            list(
                range(
                    NUM_TRAIN_SPEAKERS
                )
            ),
            noise_levels,
        )
    )

    for level in noise_results:

        mean = np.mean(
            noise_results[level]
        )

        std = np.std(
            noise_results[level]
        )

        print(
            f"  strength={level:.1f}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    plot_noise(
        noise_results
    )

    # --------------------------------------------------------
    # ATTENTION VISUALIZATION
    # --------------------------------------------------------

    print()

    print(
        "Visualizing attention..."
    )

    batch = next(
        iter(val_loader)
    )

    references = [
        x.to(DEVICE)
        for x in batch[
            "references"
        ]
    ]

    reference_masks = [
        x.to(DEVICE)
        for x in batch[
            "reference_masks"
        ]
    ]

    outputs = model.speaker_encoder(
        references,
        reference_masks,
    )

    plot_reference_attention(
        outputs[
            "reference_weights"
        ]
    )

    # First sample, first reference.
    temporal_weights = (
        outputs[
            "temporal_weights"
        ][0][0]
    )

    temporal_mask = (
        reference_masks[0][0]
    )

    plot_temporal_attention(
        temporal_weights,
        temporal_mask,
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    print()

    print(
        "Generating speaker embedding PCA..."
    )

    all_ids = list(
        range(
            NUM_TRAIN_SPEAKERS
            + NUM_HELDOUT_SPEAKERS
        )
    )

    embeddings, labels = (
        collect_speaker_embeddings(
            model,
            all_ids,
            references_per_speaker=5,
        )
    )

    plot_pca(
        embeddings,
        labels,
    )

    # --------------------------------------------------------
    # HELD-OUT SPEAKER CHECK
    # --------------------------------------------------------

    print()

    print(
        "Held-out speakers:"
    )

    heldout_ids = list(
        range(
            NUM_TRAIN_SPEAKERS,
            NUM_TRAIN_SPEAKERS
            + NUM_HELDOUT_SPEAKERS,
        )
    )

    for speaker_id in heldout_ids:

        mask = (
            labels
            == speaker_id
        )

        speaker_embeddings = (
            embeddings[
                mask
            ]
        )

        prototype = (
            speaker_embeddings.mean(
                axis=0,
                keepdims=True,
            )
        )

        prototype /= (
            np.linalg.norm(
                prototype,
                axis=1,
                keepdims=True,
            )
            + 1e-8
        )

        normalized = (
            speaker_embeddings
            / (
                np.linalg.norm(
                    speaker_embeddings,
                    axis=1,
                    keepdims=True,
                )
                + 1e-8
            )
        )

        similarities = (
            normalized
            @ prototype.T
        ).squeeze(-1)

        print(
            f"  Speaker {speaker_id}: "
            f"{similarities.mean():.4f} "
            f"± "
            f"{similarities.std():.4f}"
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 72)

    print(
        "LESSON 22 COMPLETE"
    )

    print("=" * 72)

    print()

    print(
        "Architecture learned:"
    )

    print(
        "  Mel"
    )

    print(
        "   ↓"
    )

    print(
        "  Frame Encoder"
    )

    print(
        "   ↓"
    )

    print(
        "  Temporal Attention"
    )

    print(
        "   ↓"
    )

    print(
        "  Utterance Embedding"
    )

    print(
        "   ↓"
    )

    print(
        "  Reference Attention"
    )

    print(
        "   ↓"
    )

    print(
        "  Speaker Embedding"
    )

    print(
        "   ↓"
    )

    print(
        "  FiLM Speaker Conditioning"
    )

    print(
        "   ↓"
    )

    print(
        "  TTS Decoder"
    )

    print(
        "   ↓"
    )

    print(
        "  Target Mel"
    )

    print()

    print(
        "Generated files:"
    )

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        print(
            f"  {path.name}"
        )

    print()

    print(
        "Next:"
    )

    print(
        "Lesson 23 — Neural Vocoder Integration"
    )


if __name__ == "__main__":

    main()