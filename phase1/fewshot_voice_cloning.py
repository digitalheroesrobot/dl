"""
File: phase1_lesson21_fewshot_voice_cloning.py

PHASE 1 — VOICE CLONING FROM SCRATCH
============================================================

LESSON 21 — FEW-SHOT VOICE CLONING AND REFERENCE-AUDIO ROBUSTNESS
============================================================

PURPOSE
-------
Lesson 20 introduced disentangled voice cloning:

    reference audio -> speaker embedding
    target text     -> linguistic representation
    speaker + text  -> target Mel spectrogram

Lesson 21 makes the speaker representation more robust.

Instead of relying on one reference utterance:

    reference_1 -> speaker embedding
    reference_2 -> speaker embedding
    reference_3 -> speaker embedding
             ...
                  -> aggregate
                  -> robust speaker embedding
                  -> TTS decoder

This is the basic idea behind FEW-SHOT voice cloning.

The system should learn:

    WHAT IS SAID
        !=
    WHO IS SPEAKING

and should remain relatively stable when the reference audio changes.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------

STEP 1 — Few-shot speaker representation
    Use multiple reference utterances from the same speaker.

STEP 2 — Per-utterance speaker embeddings
    Encode each reference independently:

        e_i = SpeakerEncoder(reference_i)

STEP 3 — Reference aggregation
    Combine embeddings:

        e = normalize(mean(e_1, ..., e_K))

    where K is the number of reference utterances.

STEP 4 — Reference dropout
    During training, randomly use different numbers of references.

    This prevents the model from depending on one specific utterance.

STEP 5 — Reference augmentation
    Simulate imperfect real-world recordings:

        additive noise
        amplitude variation
        spectral variation

STEP 6 — Content independence
    Reference utterances contain different phoneme sequences.

    The speaker encoder should still produce similar embeddings.

STEP 7 — Few-shot evaluation
    Compare:

        1 reference
        2 references
        3 references
        5 references

    and measure speaker embedding consistency.

STEP 8 — Robustness experiment
    Compare clean and corrupted references.

------------------------------------------------------------
CORE MATHEMATICS
------------------------------------------------------------

For reference utterance r_i:

    e_i = f_theta(r_i)

where:

    f_theta = speaker encoder

For K references:

    e_bar = (1/K) * sum_i e_i

Then normalize:

    e = e_bar / ||e_bar||_2

The cosine similarity between two speaker representations is:

    cos(e_a, e_b)
        =
    e_a^T e_b
    ----------------
    ||e_a|| ||e_b||

Because embeddings are L2-normalized:

    cos(e_a, e_b) = e_a^T e_b

------------------------------------------------------------
WHY MULTIPLE REFERENCES HELP
------------------------------------------------------------

A single utterance contains many factors:

    speaker identity
    phonetic content
    pitch
    speaking style
    recording conditions
    noise
    loudness
    microphone characteristics

Ideally:

    speaker embedding
        ≈
    speaker identity

but in practice:

    embedding
        =
    speaker identity
    + residual content information
    + recording variation
    + noise

Multiple references help average away some of those unwanted factors.

For example:

    reference 1:
        "Hello, how are you?"

    reference 2:
        "The weather is very nice today."

    reference 3:
        "I would like a cup of tea."

The linguistic content is different, but the speaker is the same.

------------------------------------------------------------
WHY REFERENCE DROPOUT?
------------------------------------------------------------

If the model always sees exactly five references:

    K = 5

it may fail when deployment provides:

    K = 1

Reference dropout trains the model using different K values.

Example:

    K = 1
    K = 2
    K = 3
    K = 5

This teaches the system to operate under different amounts
of available speaker information.

------------------------------------------------------------
ARCHITECTURE
------------------------------------------------------------

                    Reference 1
                         |
                         v
                  Speaker Encoder
                         |
                         v
                       e_1
                         |
                         |
                    Reference 2
                         |
                         v
                  Speaker Encoder
                         |
                         v
                       e_2
                         |
                         |
                    Reference 3
                         |
                         v
                  Speaker Encoder
                         |
                         v
                       e_3
                         |
                         v
                +----------------+
                | Mean Aggregate |
                +----------------+
                         |
                         v
                  Speaker Vector
                         |
                         |
Target Text ------------+
    |
    v
Text Encoder
    |
    v
Text Representation
    |
    +-------------------+
                        |
                        v
                 TTS Decoder
                        |
                        v
                  Target Mel

------------------------------------------------------------
DATA
------------------------------------------------------------

This lesson intentionally uses SYNTHETIC speech-like Mel data.

Languages represented by text examples:

    English:
        "Hello, how are you?"

    Chinese:
        "今天天气很好。"

    Japanese:
        "今日はとてもいい天気です。"

    Russian:
        "Сегодня очень хорошая погода."

No Korean/Hangul examples are used.

The synthetic data allows the architecture and learning behavior
to be studied without requiring a large speech corpus.

IMPORTANT:
This is NOT a production voice-cloning system.

A real system needs:

    real speech
    speaker-balanced datasets
    phoneme/token processing
    real Mel extraction
    alignment
    augmentation
    speaker verification objectives
    vocoder
    evaluation
    privacy/safety controls

------------------------------------------------------------
LESSON 21 EXPERIMENTS
------------------------------------------------------------

Experiment A:
    How does the number of references affect speaker similarity?

Experiment B:
    Does aggregation reduce reference-specific variation?

Experiment C:
    How robust is the speaker representation to noise?

Experiment D:
    Can one model operate with different K?

Experiment E:
    Does few-shot cloning work on held-out speakers?

------------------------------------------------------------
EXPECTED RESULT
------------------------------------------------------------

Usually:

    1 reference
        lower robustness

    2 references
        improved stability

    3 references
        improved stability

    5 references
        generally strongest stability

The exact result depends on optimization and random seed.

The important learning objective is understanding WHY this happens.

------------------------------------------------------------
LIMITATIONS
------------------------------------------------------------

1. Synthetic Mel features are not real speech.

2. Speaker profiles are generated algorithmically.

3. The "speaker identity" is therefore simplified.

4. No neural vocoder is included.

5. No waveform generation is performed.

6. The model uses ground-truth durations during training.

7. The experiment is educational rather than production-ready.

------------------------------------------------------------
ROADMAP
------------------------------------------------------------

Lesson 18
    Multi-speaker TTS

Lesson 19
    Zero-shot voice cloning

Lesson 20
    Disentangled voice cloning

Lesson 21
    Few-shot voice cloning
        <-- YOU ARE HERE

Lesson 22
    Learned reference attention / speaker conditioning

Lesson 23
    Neural vocoder integration

Lesson 24
    End-to-end voice cloning

Lesson 25+
    More realistic generative speech architectures
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("lesson21_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# SYNTHETIC SPEAKER MODEL
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
        SEED + seed_offset + speaker_id * 1009
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
        spectral_offset=spectral_offset.astype(np.float32),
        resonance_strength=float(resonance_strength),
        brightness=float(brightness),
        harmonic_tilt=float(harmonic_tilt),
        temporal_style=float(temporal_style),
    )


# ============================================================
# PHONEME TEMPLATES
# ============================================================

def create_phoneme_templates() -> np.ndarray:

    rng = np.random.default_rng(SEED + 123)

    templates = rng.normal(
        0.0,
        0.25,
        size=(VOCAB_SIZE, N_MELS),
    ).astype(np.float32)

    # Padding token should have a neutral template.
    templates[PAD_ID] = 0.0

    # Create smooth spectral structure.
    mel_axis = np.linspace(
        0.0,
        1.0,
        N_MELS,
    ).astype(np.float32)

    for phoneme_id in range(1, VOCAB_SIZE):

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
            * ((mel_axis - center) / width) ** 2
        )

        templates[phoneme_id] += (
            0.8 * gaussian
        )

    return templates


PHONEME_TEMPLATES = create_phoneme_templates()


# ============================================================
# SYNTHETIC UTTERANCE GENERATOR
# ============================================================

def make_target_utterance(
    phonemes: np.ndarray,
    speaker: SpeakerProfile,
    rng: np.random.Generator,
):
    """
    Create a synthetic Mel spectrogram and durations.

    The same speaker produces different acoustic realizations
    for different phoneme sequences.
    """

    durations = []

    for _ in phonemes:
        duration = int(
            rng.integers(
                4,
                9,
            )
        )

        durations.append(duration)

    durations = np.asarray(
        durations,
        dtype=np.int64,
    )

    frames = []

    mel_axis = np.linspace(
        0.0,
        1.0,
        N_MELS,
    ).astype(np.float32)

    for phoneme_id, duration in zip(
        phonemes,
        durations,
    ):

        base = PHONEME_TEMPLATES[
            int(phoneme_id)
        ].copy()

        # Speaker-dependent spectral shape.
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
            + 0.12 * speaker_curve
            + 0.08 * brightness_curve
            + 0.08 * harmonic_curve
        )

        # Temporal envelope.
        t = np.linspace(
            0.0,
            1.0,
            duration,
        ).astype(np.float32)

        envelope = (
            0.85
            + 0.15
            * np.sin(
                np.pi * t
            )
        )

        for frame_index in range(duration):

            local = spectral.copy()

            # Small frame-level variation.
            noise = rng.normal(
                0.0,
                0.025,
                size=N_MELS,
            ).astype(np.float32)

            local += noise

            local *= envelope[
                frame_index
            ]

            frames.append(local)

    mel = np.stack(
        frames,
        axis=0,
    ).astype(np.float32)

    # Normalize synthetic Mel values.
    mel = np.tanh(mel)

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
) -> np.ndarray:
    """
    Simulate imperfect reference recordings.

    Augmentations:
        amplitude variation
        additive noise
        spectral tilt
    """

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

    noise = rng.normal(
        0.0,
        noise_std,
        size=x.shape,
    ).astype(np.float32)

    x += noise

    # Simple frequency-axis tilt.
    frequency_axis = np.linspace(
        -1.0,
        1.0,
        x.shape[1],
    ).astype(np.float32)

    tilt = rng.uniform(
        -0.05,
        0.05,
    ) * strength

    x += tilt * frequency_axis[None, :]

    return np.clip(
        x,
        -1.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class FewShotVoiceDataset(Dataset):

    def __init__(
        self,
        num_samples: int,
        speaker_ids: list[int],
        seed_offset: int,
        augment_references: bool,
    ):

        self.num_samples = num_samples
        self.speaker_ids = speaker_ids
        self.seed_offset = seed_offset
        self.augment_references = augment_references

    def __len__(self):
        return self.num_samples

    def __getitem__(
        self,
        index: int,
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

        speaker = create_speaker_profile(
            speaker_id,
            seed_offset=5000,
        )

        # ----------------------------------------------------
        # Generate several DIFFERENT reference utterances.
        # ----------------------------------------------------

        references = []

        for reference_index in range(
            MAX_REFERENCES
        ):

            reference_length = int(
                rng.integers(
                    5,
                    MAX_PHONEMES + 1,
                )
            )

            reference_phonemes = rng.integers(
                1,
                VOCAB_SIZE,
                size=reference_length,
            )

            reference_mel, _ = (
                make_target_utterance(
                    reference_phonemes,
                    speaker,
                    rng,
                )
            )

            if self.augment_references:

                # Make some references imperfect.
                reference_mel = augment_reference(
                    reference_mel,
                    rng,
                    strength=1.0,
                )

            references.append(
                reference_mel
            )

        # ----------------------------------------------------
        # Target utterance.
        #
        # IMPORTANT:
        # target content is independently generated.
        # ----------------------------------------------------

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
            "target_phonemes": target_phonemes.astype(
                np.int64
            ),
            "target_mel": target_mel.astype(
                np.float32
            ),
            "target_durations": target_durations.astype(
                np.int64
            ),
            "speaker_id": speaker_id,
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    batch_size = len(batch)

    # --------------------------------------------------------
    # References:
    #
    # references[b][k] = [T, N_MELS]
    # --------------------------------------------------------

    max_ref_lengths = []

    for k in range(MAX_REFERENCES):

        length = max(
            item["references"][k].shape[0]
            for item in batch
        )

        max_ref_lengths.append(
            length
        )

    references = []

    reference_masks = []

    for k in range(MAX_REFERENCES):

        max_len = max_ref_lengths[k]

        tensor = torch.zeros(
            batch_size,
            max_len,
            N_MELS,
            dtype=torch.float32,
        )

        mask = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
        )

        for b, item in enumerate(batch):

            x = torch.from_numpy(
                item["references"][k]
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
    # Target phonemes.
    # --------------------------------------------------------

    max_phonemes = max(
        len(item["target_phonemes"])
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

    for b, item in enumerate(batch):

        n = len(
            item["target_phonemes"]
        )

        target_phonemes[
            b,
            :n
        ] = torch.from_numpy(
            item["target_phonemes"]
        )

        target_durations[
            b,
            :n
        ] = torch.from_numpy(
            item["target_durations"]
        )

        phoneme_mask[
            b,
            :n
        ] = True

    # --------------------------------------------------------
    # Target Mel.
    # --------------------------------------------------------

    max_target_frames = max(
        item["target_mel"].shape[0]
        for item in batch
    )

    target_mel = torch.zeros(
        batch_size,
        max_target_frames,
        N_MELS,
        dtype=torch.float32,
    )

    target_mel_mask = torch.zeros(
        batch_size,
        max_target_frames,
        dtype=torch.bool,
    )

    for b, item in enumerate(batch):

        n = item["target_mel"].shape[0]

        target_mel[
            b,
            :n
        ] = torch.from_numpy(
            item["target_mel"]
        )

        target_mel_mask[
            b,
            :n
        ] = True

    speaker_ids = torch.tensor(
        [
            item["speaker_id"]
            for item in batch
        ],
        dtype=torch.long,
    )

    return {
        "references": references,
        "reference_masks": reference_masks,
        "target_phonemes": target_phonemes,
        "phoneme_mask": phoneme_mask,
        "target_durations": target_durations,
        "target_mel": target_mel,
        "target_mel_mask": target_mel_mask,
        "speaker_ids": speaker_ids,
    }


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_len: int = 512,
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

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x):

        return x + self.pe[
            :, :x.shape[1]
        ]


# ============================================================
# SPEAKER ENCODER
# ============================================================

class SpeakerEncoder(nn.Module):

    def __init__(
        self,
        mel_dim: int,
        embedding_dim: int,
    ):

        super().__init__()

        self.conv = nn.Sequential(
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
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.projection = nn.Linear(
            128,
            embedding_dim,
        )

    def forward(
        self,
        mel,
        mask,
    ):

        # mel:
        #   [B, T, M]

        x = mel.transpose(
            1,
            2,
        )

        x = self.conv(x)

        x = x.transpose(
            1,
            2,
        )

        mask_f = mask.unsqueeze(
            -1
        ).float()

        summed = (
            x * mask_f
        ).sum(dim=1)

        lengths = mask_f.sum(
            dim=1
        ).clamp_min(1.0)

        pooled = (
            summed / lengths
        )

        embedding = self.projection(
            pooled
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding


# ============================================================
# MULTI-REFERENCE AGGREGATOR
# ============================================================

class ReferenceAggregator(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
    ):

        super().__init__()

        self.attention = nn.Sequential(
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

        Returns:
            [B, D]
        """

        scores = self.attention(
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

        aggregated = (
            embeddings
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        aggregated = F.normalize(
            aggregated,
            p=2,
            dim=-1,
        )

        return aggregated, weights


# ============================================================
# TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            model_dim,
            padding_idx=PAD_ID,
        )

        self.position = PositionalEncoding(
            model_dim
        )

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 4,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
        )

    def forward(
        self,
        tokens,
    ):

        x = self.embedding(
            tokens
        )

        x = self.position(x)

        padding_mask = (
            tokens == PAD_ID
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        return x


# ============================================================
# DURATION PREDICTOR
# ============================================================

class DurationPredictor(nn.Module):

    def __init__(
        self,
        model_dim: int,
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

    def forward(self, x):

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
    """
    x:
        [B, N, D]

    durations:
        [B, N]

    Returns:
        list of [T_i, D]
    """

    outputs = []

    for batch_index in range(
        x.shape[0]
    ):

        expanded = []

        for phoneme_index in range(
            x.shape[1]
        ):

            duration = int(
                durations[
                    batch_index,
                    phoneme_index,
                ].item()
            )

            if duration <= 0:
                continue

            expanded.append(
                x[
                    batch_index,
                    phoneme_index,
                ].unsqueeze(0).repeat(
                    duration,
                    1,
                )
            )

        if len(expanded) == 0:

            expanded.append(
                x[
                    batch_index,
                    :1,
                ]
            )

        outputs.append(
            torch.cat(
                expanded,
                dim=0,
            )
        )

    return outputs


# ============================================================
# DISENTANGLED FEW-SHOT TTS
# ============================================================

class FewShotVoiceCloningTTS(nn.Module):

    def __init__(self):

        super().__init__()

        self.speaker_encoder = SpeakerEncoder(
            N_MELS,
            SPEAKER_EMBED_DIM,
        )

        self.reference_aggregator = (
            ReferenceAggregator(
                SPEAKER_EMBED_DIM
            )
        )

        self.text_encoder = TextEncoder(
            VOCAB_SIZE,
            MODEL_DIM,
        )

        self.duration_predictor = (
            DurationPredictor(
                MODEL_DIM
            )
        )

        self.speaker_projection = nn.Linear(
            SPEAKER_EMBED_DIM,
            MODEL_DIM,
        )

        decoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=MODEL_DIM,
                nhead=4,
                dim_feedforward=MODEL_DIM * 4,
                dropout=DROPOUT,
                batch_first=True,
                activation="gelu",
            )
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=2,
        )

        self.mel_projection = nn.Linear(
            MODEL_DIM,
            N_MELS,
        )

        self.speaker_classifier = nn.Linear(
            SPEAKER_EMBED_DIM,
            NUM_TRAIN_SPEAKERS,
        )

    def encode_references(
        self,
        references,
        reference_masks,
        num_references=None,
    ):

        if num_references is None:

            num_references = len(
                references
            )

        embeddings = []

        for k in range(
            num_references
        ):

            embedding = (
                self.speaker_encoder(
                    references[k],
                    reference_masks[k],
                )
            )

            embeddings.append(
                embedding
            )

        embeddings = torch.stack(
            embeddings,
            dim=1,
        )

        batch_size = (
            embeddings.shape[0]
        )

        reference_mask = torch.ones(
            batch_size,
            num_references,
            dtype=torch.bool,
            device=embeddings.device,
        )

        speaker_embedding, weights = (
            self.reference_aggregator(
                embeddings,
                reference_mask,
            )
        )

        return (
            speaker_embedding,
            embeddings,
            weights,
        )

    def synthesize(
        self,
        target_phonemes,
        target_durations,
        speaker_embedding,
    ):

        text_features = self.text_encoder(
            target_phonemes
        )

        speaker_condition = (
            self.speaker_projection(
                speaker_embedding
            )
            .unsqueeze(1)
        )

        conditioned_text = (
            text_features
            + speaker_condition
        )

        expanded = length_regulate(
            conditioned_text,
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
            device=conditioned_text.device,
        )

        mask = torch.zeros(
            batch_size,
            max_length,
            dtype=torch.bool,
            device=conditioned_text.device,
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
            src_key_padding_mask=~mask,
        )

        mel = self.mel_projection(
            decoded
        )

        return mel, mask

    def forward(
        self,
        references,
        reference_masks,
        target_phonemes,
        target_durations,
    ):

        (
            speaker_embedding,
            individual_embeddings,
            attention_weights,
        ) = self.encode_references(
            references,
            reference_masks,
        )

        mel_prediction, mel_mask = (
            self.synthesize(
                target_phonemes,
                target_durations,
                speaker_embedding,
            )
        )

        speaker_logits = (
            self.speaker_classifier(
                speaker_embedding
            )
        )

        return {
            "speaker_embedding": speaker_embedding,
            "individual_embeddings": individual_embeddings,
            "attention_weights": attention_weights,
            "mel_prediction": mel_prediction,
            "mel_mask": mel_mask,
            "speaker_logits": speaker_logits,
        }


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def masked_mel_l1(
    prediction,
    target,
    target_mask,
):

    # --------------------------------------------------------
    # Defensive alignment.
    #
    # Both tensors should normally have identical time length
    # because target durations are used by the length regulator.
    #
    # We nevertheless crop to the common length so variable
    # batches can never produce a shape mismatch.
    # --------------------------------------------------------

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
        prediction - target
    ).abs()

    mask = mask.unsqueeze(-1).float()

    numerator = (
        error * mask
    ).sum()

    denominator = (
        mask.sum()
        * prediction.shape[-1]
    ).clamp_min(1.0)

    return numerator / denominator


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
    Encourage all references in the same sample
    to have similar speaker embeddings.

    embeddings:
        [B, K, D]
    """

    mean_embedding = F.normalize(
        embeddings.mean(dim=1),
        p=2,
        dim=-1,
    )

    mean_expanded = (
        mean_embedding
        .unsqueeze(1)
        .expand_as(embeddings)
    )

    cosine = F.cosine_similarity(
        embeddings,
        mean_expanded,
        dim=-1,
    )

    return (
        1.0 - cosine
    ).mean()


def batch_speaker_contrastive_loss(
    embeddings,
    speaker_ids,
    temperature=0.10,
):
    """
    Simple speaker-level contrastive loss.

    embeddings:
        [B, D]

    Same speaker -> positive.
    Different speakers -> negative.
    """

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    similarity = (
        embeddings
        @ embeddings.T
    ) / temperature

    labels = speaker_ids.unsqueeze(0) == (
        speaker_ids.unsqueeze(1)
    )

    # Remove self comparisons.
    diagonal = torch.eye(
        embeddings.shape[0],
        dtype=torch.bool,
        device=embeddings.device,
    )

    labels = labels & ~diagonal

    logits_mask = ~diagonal

    losses = []

    for i in range(
        embeddings.shape[0]
    ):

        valid_logits = similarity[
            i
        ][logits_mask[i]]

        valid_labels = labels[
            i
        ][logits_mask[i]]

        if valid_labels.sum() == 0:
            continue

        log_prob = (
            valid_logits
            - torch.logsumexp(
                valid_logits,
                dim=0,
            )
        )

        positive_log_prob = (
            log_prob[valid_labels]
        ).mean()

        losses.append(
            -positive_log_prob
        )

    if len(losses) == 0:

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

    total_loss = 0.0
    total_mel = 0.0
    total_duration = 0.0
    total_speaker = 0.0
    total_consistency = 0.0
    total_contrastive = 0.0

    correct = 0
    count = 0

    for batch in loader:

        references = [
            x.to(DEVICE)
            for x in batch["references"]
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

        # ----------------------------------------------------
        # Reference dropout.
        #
        # Randomly choose how many references are visible.
        #
        # This teaches the model to work with:
        #
        #   1, 2, 3, ..., MAX_REFERENCES
        #
        # references.
        # ----------------------------------------------------

        num_references = random.randint(
            1,
            MAX_REFERENCES,
        )

        (
            speaker_embedding,
            individual_embeddings,
            attention_weights,
        ) = model.encode_references(
            references[
                :num_references
            ],
            reference_masks[
                :num_references
            ],
        )

        mel_prediction, mel_mask = (
            model.synthesize(
                target_phonemes,
                target_durations,
                speaker_embedding,
            )
        )

        speaker_logits = (
            model.speaker_classifier(
                speaker_embedding
            )
        )

        mel_loss = masked_mel_l1(
            mel_prediction,
            target_mel,
            target_mel_mask,
        )

        duration_prediction = (
            model.duration_predictor(
                model.text_encoder(
                    target_phonemes
                )
            )
        )

        duration_loss_value = (
            duration_loss(
                duration_prediction,
                target_durations,
                batch[
                    "phoneme_mask"
                ].to(DEVICE),
            )
        )

        speaker_loss = (
            speaker_classification_loss(
                speaker_logits,
                speaker_ids,
            )
        )

        consistency_loss = (
            reference_consistency_loss(
                individual_embeddings
            )
        )

        contrastive_loss = (
            batch_speaker_contrastive_loss(
                speaker_embedding,
                speaker_ids,
            )
        )

        loss = (
            mel_loss
            + 0.15
            * duration_loss_value
            + 0.20
            * speaker_loss
            + 0.20
            * consistency_loss
            + 0.20
            * contrastive_loss
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

        total_loss += loss.item()
        total_mel += mel_loss.item()
        total_duration += (
            duration_loss_value.item()
        )
        total_speaker += (
            speaker_loss.item()
        )
        total_consistency += (
            consistency_loss.item()
        )
        total_contrastive += (
            contrastive_loss.item()
        )

        predictions = (
            speaker_logits.argmax(
                dim=-1
            )
        )

        correct += (
            predictions == speaker_ids
        ).sum().item()

        count += speaker_ids.shape[0]

    n = len(loader)

    return {
        "loss": total_loss / n,
        "mel": total_mel / n,
        "duration": total_duration / n,
        "speaker": total_speaker / n,
        "consistency": total_consistency / n,
        "contrastive": total_contrastive / n,
        "accuracy": correct / max(
            count,
            1,
        ),
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
            for x in batch["references"]
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

        # Validation uses all references.
        (
            speaker_embedding,
            individual_embeddings,
            attention_weights,
        ) = model.encode_references(
            references,
            reference_masks,
        )

        mel_prediction, mel_mask = (
            model.synthesize(
                target_phonemes,
                target_durations,
                speaker_embedding,
            )
        )

        logits = model.speaker_classifier(
            speaker_embedding
        )

        mel_loss = masked_mel_l1(
            mel_prediction,
            target_mel,
            target_mel_mask,
        )

        speaker_loss = (
            speaker_classification_loss(
                logits,
                speaker_ids,
            )
        )

        loss = (
            mel_loss
            + 0.20
            * speaker_loss
        )

        total_loss += loss.item()
        total_mel += mel_loss.item()

        predictions = logits.argmax(
            dim=-1
        )

        correct += (
            predictions == speaker_ids
        ).sum().item()

        count += speaker_ids.shape[0]

    n = len(loader)

    return {
        "loss": total_loss / n,
        "mel": total_mel / n,
        "accuracy": correct / max(
            count,
            1,
        ),
    }


# ============================================================
# EMBEDDING EVALUATION
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
                900000
                + speaker_id * 100
                + repetition
            )

            speaker = create_speaker_profile(
                speaker_id,
                seed_offset=5000,
            )

            references = []
            masks = []

            for k in range(
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

                references.append(
                    mel
                )

            # ------------------------------------------------
            # Encode each reference independently.
            # ------------------------------------------------

            individual_embeddings = []

            for mel in references:

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
                    model.speaker_encoder(
                        x,
                        mask,
                    )
                )

                individual_embeddings.append(
                    embedding
                )

            individual_embeddings = torch.cat(
                individual_embeddings,
                dim=0,
            )

            # ------------------------------------------------
            # Reference-count experiments.
            # ------------------------------------------------

            for k in range(
                1,
                MAX_REFERENCES + 1,
            ):

                subset = (
                    individual_embeddings[
                        :k
                    ]
                )

                aggregate = F.normalize(
                    subset.mean(
                        dim=0,
                        keepdim=True,
                    ),
                    p=2,
                    dim=-1,
                )

                similarities = (
                    subset
                    @ aggregate.T
                ).squeeze(-1)

                score = (
                    similarities.mean()
                    .item()
                )

                results[k].append(
                    score
                )

    return results


# ============================================================
# NOISY REFERENCE EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_noise_robustness(
    model,
    speaker_ids,
    noise_levels,
    repetitions=15,
):

    model.eval()

    results = {
        level: []
        for level in noise_levels
    }

    for speaker_id in speaker_ids:

        for repetition in range(
            repetitions
        ):

            rng = np.random.default_rng(
                700000
                + speaker_id * 100
                + repetition
            )

            speaker = create_speaker_profile(
                speaker_id,
                seed_offset=5000,
            )

            clean_embeddings = []

            for ref_index in range(
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

                embedding = (
                    model.speaker_encoder(
                        x,
                        mask,
                    )
                )

                clean_embeddings.append(
                    embedding
                )

            clean_embeddings = torch.cat(
                clean_embeddings,
                dim=0,
            )

            clean_reference = F.normalize(
                clean_embeddings.mean(
                    dim=0,
                    keepdim=True,
                ),
                p=2,
                dim=-1,
            )

            for noise_level in noise_levels:

                noisy_embeddings = []

                for ref_index in range(
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

                    noisy = augment_reference(
                        mel,
                        rng,
                        strength=noise_level,
                    )

                    x = torch.from_numpy(
                        noisy
                    ).unsqueeze(0).to(
                        DEVICE
                    )

                    mask = torch.ones(
                        1,
                        noisy.shape[0],
                        dtype=torch.bool,
                        device=DEVICE,
                    )

                    embedding = (
                        model.speaker_encoder(
                            x,
                            mask,
                        )
                    )

                    noisy_embeddings.append(
                        embedding
                    )

                noisy_embeddings = torch.cat(
                    noisy_embeddings,
                    dim=0,
                )

                noisy_reference = F.normalize(
                    noisy_embeddings.mean(
                        dim=0,
                        keepdim=True,
                    ),
                    p=2,
                    dim=-1,
                )

                similarity = (
                    F.cosine_similarity(
                        clean_reference,
                        noisy_reference,
                        dim=-1,
                    )
                    .item()
                )

                results[
                    noise_level
                ].append(
                    similarity
                )

    return results


# ============================================================
# HELD-OUT SPEAKER TEST
# ============================================================

@torch.no_grad()
def evaluate_heldout_speakers(
    model,
    speaker_ids,
):

    model.eval()

    scores = []

    for speaker_id in speaker_ids:

        rng = np.random.default_rng(
            1000000
            + speaker_id
        )

        speaker = create_speaker_profile(
            speaker_id,
            seed_offset=5000,
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

            embedding = (
                model.speaker_encoder(
                    x,
                    mask,
                )
            )

            embeddings.append(
                embedding
            )

        embeddings = torch.cat(
            embeddings,
            dim=0,
        )

        prototype = F.normalize(
            embeddings.mean(
                dim=0,
                keepdim=True,
            ),
            p=2,
            dim=-1,
        )

        similarity = F.cosine_similarity(
            embeddings,
            prototype,
            dim=-1,
        ).mean().item()

        scores.append(
            similarity
        )

        print(
            f"  Held-out speaker {speaker_id}: "
            f"mean reference similarity = "
            f"{similarity:.4f}"
        )

    return scores


# ============================================================
# PLOTS
# ============================================================

def plot_training_curves(
    history,
):

    epochs = np.arange(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Lesson 21 — Few-Shot Voice Cloning Training"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "lesson21_training_curves.png",
        dpi=160,
    )

    plt.close()


def plot_reference_count_results(
    results,
):

    counts = list(
        results.keys()
    )

    means = [
        np.mean(results[k])
        for k in counts
    ]

    stds = [
        np.std(results[k])
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
        "Number of reference utterances"
    )

    plt.ylabel(
        "Mean cosine similarity"
    )

    plt.title(
        "Few-Shot Speaker Embedding Stability"
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
        / "lesson21_reference_count.png",
        dpi=160,
    )

    plt.close()


def plot_noise_results(
    results,
):

    levels = list(
        results.keys()
    )

    means = [
        np.mean(results[x])
        for x in levels
    ]

    stds = [
        np.std(results[x])
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
        "Reference-Audio Robustness"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "lesson21_noise_robustness.png",
        dpi=160,
    )

    plt.close()


def plot_attention_weights(
    weights,
):

    weights = weights.detach().cpu().numpy()

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
        "Learned Reference Aggregation Weights"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "lesson21_reference_attention.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PHASE 1 — LESSON 21"
    )

    print(
        "Few-Shot Voice Cloning & Reference Robustness"
    )

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Output directory: {OUTPUT_DIR.resolve()}"
    )

    print()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = FewShotVoiceDataset(
        num_samples=TRAIN_SAMPLES,
        speaker_ids=list(
            range(
                NUM_TRAIN_SPEAKERS
            )
        ),
        seed_offset=0,
        augment_references=True,
    )

    val_dataset = FewShotVoiceDataset(
        num_samples=VAL_SAMPLES,
        speaker_ids=list(
            range(
                NUM_TRAIN_SPEAKERS
            )
        ),
        seed_offset=100000,
        augment_references=False,
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

    model = FewShotVoiceCloningTTS().to(
        DEVICE
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mel": [],
        "val_mel": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "consistency": [],
        "contrastive": [],
    }

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_metrics = validate(
            model,
            val_loader,
        )

        scheduler.step()

        history[
            "train_loss"
        ].append(
            train_metrics["loss"]
        )

        history[
            "val_loss"
        ].append(
            val_metrics["loss"]
        )

        history[
            "train_mel"
        ].append(
            train_metrics["mel"]
        )

        history[
            "val_mel"
        ].append(
            val_metrics["mel"]
        )

        history[
            "train_accuracy"
        ].append(
            train_metrics["accuracy"]
        )

        history[
            "val_accuracy"
        ].append(
            val_metrics["accuracy"]
        )

        history[
            "consistency"
        ].append(
            train_metrics["consistency"]
        )

        history[
            "contrastive"
        ].append(
            train_metrics["contrastive"]
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_metrics['loss']:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"mel={train_metrics['mel']:.4f} | "
            f"speaker_acc="
            f"{val_metrics['accuracy']:.3f} | "
            f"K-consistency="
            f"{train_metrics['consistency']:.4f}"
        )

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = (
        OUTPUT_DIR
        / "lesson21_fewshot_voice_cloning.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "n_mels": N_MELS,
                "vocab_size": VOCAB_SIZE,
                "model_dim": MODEL_DIM,
                "speaker_embed_dim": SPEAKER_EMBED_DIM,
                "max_references": MAX_REFERENCES,
            },
        },
        checkpoint_path,
    )

    print()
    print(
        f"Checkpoint saved to: "
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------
    # TRAINING CURVES
    # --------------------------------------------------------

    plot_training_curves(
        history
    )

    # --------------------------------------------------------
    # REFERENCE COUNT EXPERIMENT
    # --------------------------------------------------------

    print()
    print(
        "Evaluating few-shot reference counts..."
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

    plot_reference_count_results(
        reference_results
    )

    # --------------------------------------------------------
    # NOISE ROBUSTNESS
    # --------------------------------------------------------

    print()
    print(
        "Evaluating noisy references..."
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
            f"  Noise strength={level:.1f}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    plot_noise_results(
        noise_results
    )

    # --------------------------------------------------------
    # ATTENTION VISUALIZATION
    # --------------------------------------------------------

    print()
    print(
        "Visualizing learned reference weights..."
    )

    batch = next(
        iter(val_loader)
    )

    references = [
        x.to(DEVICE)
        for x in batch["references"]
    ]

    reference_masks = [
        x.to(DEVICE)
        for x in batch[
            "reference_masks"
        ]
    ]

    (
        speaker_embedding,
        individual_embeddings,
        attention_weights,
    ) = model.encode_references(
        references,
        reference_masks,
    )

    plot_attention_weights(
        attention_weights
    )

    # --------------------------------------------------------
    # HELD-OUT SPEAKER TEST
    # --------------------------------------------------------

    print()
    print(
        "Held-out speaker evaluation..."
    )

    heldout_ids = list(
        range(
            NUM_TRAIN_SPEAKERS,
            NUM_TRAIN_SPEAKERS
            + NUM_HELDOUT_SPEAKERS,
        )
    )

    heldout_scores = (
        evaluate_heldout_speakers(
            model,
            heldout_ids,
        )
    )

    print(
        f"Mean held-out similarity: "
        f"{np.mean(heldout_scores):.4f}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "LESSON 21 COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        "Key ideas demonstrated:"
    )

    print(
        "  1. Multiple reference utterances"
    )

    print(
        "  2. Per-reference speaker embeddings"
    )

    print(
        "  3. Learned reference aggregation"
    )

    print(
        "  4. Reference-count robustness"
    )

    print(
        "  5. Reference augmentation"
    )

    print(
        "  6. Speaker consistency loss"
    )

    print(
        "  7. Speaker contrastive learning"
    )

    print(
        "  8. Held-out speaker evaluation"
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
        "Next lesson:"
    )

    print(
        "Lesson 22 — Learned Reference Attention "
        "and Stronger Speaker Conditioning"
    )


if __name__ == "__main__":
    main()