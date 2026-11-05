
"""
===============================================================================
PHASE 1 — LESSON 20
VOICE CLONING WITH SPEAKER / CONTENT DISENTANGLEMENT
===============================================================================

FILE
----
phase1_lesson20_disentangled_voice_cloning.py

RUN
---
python phase1_lesson20_disentangled_voice_cloning.py

Quick experiment:
    python phase1_lesson20_disentangled_voice_cloning.py --epochs 15

Faster CPU experiment:
    python phase1_lesson20_disentangled_voice_cloning.py --epochs 5 \
        --train-samples 400

Disable plots:
    python phase1_lesson20_disentangled_voice_cloning.py --no-plot

===============================================================================
LESSON PURPOSE
===============================================================================

Lesson 19 introduced:

    reference audio
          |
          v
    speaker encoder
          |
          v
    speaker embedding
          |
          +----------------------+
                                 |
    target text -> TTS ----------+
                                 |
                                 v
                              target Mel

However, there is an important problem.

The reference audio contains BOTH:

    1. speaker identity
    2. linguistic/content information

For example:

    Speaker A says:
        "hello world"

The reference signal contains information about:

    Speaker A
    + the words "hello world"

But we want the speaker embedding to represent primarily:

    Speaker A

rather than:

    Speaker A + "hello world"

This lesson explicitly teaches the model to separate those factors.

===============================================================================
CORE IDEA
===============================================================================

We create multiple utterances for the same synthetic speaker:

    Speaker 0:
        utterance A
        utterance B
        utterance C

    Speaker 1:
        utterance A
        utterance B
        utterance C

The linguistic content changes.

The speaker remains constant.

Therefore:

    speaker_encoder(reference_A)
        should be close to
    speaker_encoder(reference_B)

even though their phoneme sequences are different.

At the same time:

    speaker_encoder(Speaker 0)
        should be far from
    speaker_encoder(Speaker 1)

===============================================================================
WHAT YOU LEARN
===============================================================================

STEP 1
------
Content versus speaker identity.

STEP 2
------
Why reference audio contains unwanted linguistic information.

STEP 3
------
Speaker embedding consistency.

STEP 4
------
Contrastive learning.

STEP 5
------
Positive and negative speaker pairs.

STEP 6
------
Cosine similarity and cosine distance.

STEP 7
------
Speaker classification as an auxiliary objective.

STEP 8
------
Speaker-conditioned acoustic generation.

STEP 9
------
Held-out speaker evaluation.

STEP 10
-------
Testing whether the same speaker remains consistent across different text.

===============================================================================
MATHEMATICAL MODEL
===============================================================================

Let:

    R = reference acoustic features
    T = target text
    S = speaker identity
    Y = target acoustic representation

We want:

    E_s(R) -> speaker embedding

and:

    G(T, E_s(R)) -> Y

Ideally:

    E_s(R_1) ≈ E_s(R_2)

when:

    speaker(R_1) == speaker(R_2)

even if:

    text(R_1) != text(R_2)

-------------------------------------------------------------------------------
CONTRASTIVE OBJECTIVE
-------------------------------------------------------------------------------

For a positive pair:

    same speaker

we want:

    cosine(E_1, E_2) -> 1

For a negative pair:

    different speaker

we want:

    cosine(E_1, E_2) -> below a margin

A simple contrastive loss is:

    L_positive = 1 - cosine(E1, E2)

    L_negative = max(0, cosine(E1, E2) - margin)

Therefore:

    L_contrastive =
        positive_loss + negative_loss

-------------------------------------------------------------------------------
SPEAKER CLASSIFICATION
-------------------------------------------------------------------------------

We additionally use:

    speaker_embedding -> classifier -> speaker ID

This is only a training signal.

At inference:

    speaker ID is NOT required.

The classifier can be discarded.

This is important because the objective is not:

    audio -> speaker number

The objective is:

    audio -> reusable speaker representation

-------------------------------------------------------------------------------
TOTAL TRAINING OBJECTIVE
-------------------------------------------------------------------------------

We use:

    L =
        L_mel
        + lambda_duration * L_duration
        + lambda_speaker * L_speaker
        + lambda_contrastive * L_contrastive

where:

    L_mel
        = acoustic reconstruction

    L_duration
        = target phoneme duration prediction

    L_speaker
        = auxiliary speaker classification

    L_contrastive
        = speaker embedding consistency

===============================================================================
IMPORTANT DESIGN CHANGE FROM LESSON 19
===============================================================================

Lesson 19 used:

    one reference -> one embedding

Lesson 20 explicitly uses:

    reference A
    reference B

from the SAME speaker.

The encoder sees different linguistic content but must produce compatible
speaker representations.

This is the central disentanglement experiment.

===============================================================================
SYNTHETIC DATA
===============================================================================

This lesson still uses synthetic Mel spectrograms.

Therefore:

    THIS IS NOT A REAL HUMAN VOICE CLONER.

It is an architectural learning system.

The synthetic generator gives each speaker:

    spectral identity
    resonance profile
    brightness
    harmonic characteristics
    temporal characteristics

while phonemes contribute linguistic/acoustic content.

===============================================================================
REAL TTS CONNECTION
===============================================================================

For a real system:

    reference waveform
          |
          v
    acoustic frontend
          |
          v
    speaker encoder
          |
          v
    speaker embedding
          |
          +-----------------------+
                                  |
    text -> tokenizer -> encoder -+
                                  |
                                  v
                             acoustic model
                                  |
                                  v
                              Mel spectrogram
                                  |
                                  v
                              vocoder
                                  |
                                  v
                              waveform

===============================================================================
MULTILINGUAL EXAMPLES
===============================================================================

English:
    "The robot picked up the object."

Mandarin Chinese:
    "机器人拿起了物体。"

Japanese:
    "ロボットが物体を持ち上げました。"

Russian:
    "Робот поднял предмет."

The educational model does NOT perform real multilingual phonemization here.
The examples are conceptual language examples for the course.

===============================================================================
LIMITATIONS
===============================================================================

This lesson does NOT implement:

    real waveform synthesis
    real phoneme conversion
    real forced alignment
    real multilingual G2P
    real human speaker embeddings
    neural vocoder
    natural prosody
    emotion modeling
    pitch prediction
    energy prediction

Those are introduced progressively in later lessons.

===============================================================================
OUTPUT FILES
===============================================================================

lesson20_disentangled_tts.pt
lesson20_training_curves.png
lesson20_same_speaker_similarity.png
lesson20_embedding_similarity.png
lesson20_embedding_pca.png
lesson20_reference_examples.png

===============================================================================
DEPENDENCIES
===============================================================================

pip install numpy matplotlib torch
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

N_MELS = 80

VOCAB_SIZE = 64

PAD_ID = 0

MODEL_DIM = 128

SPEAKER_EMBED_DIM = 64

MAX_PHONEMES = 16

NUM_TRAIN_SPEAKERS = 8

NUM_HELDOUT_SPEAKERS = 2

DEFAULT_TRAIN_SAMPLES = 640

DEFAULT_VAL_SAMPLES = 128

DEFAULT_BATCH_SIZE = 32

DEFAULT_EPOCHS = 20

DEFAULT_LR = 2e-3

CONTRASTIVE_MARGIN = 0.35


# =============================================================================
# SYNTHETIC SPEAKER PROFILE
# =============================================================================

@dataclass
class SpeakerProfile:
    spectral_offset: np.ndarray
    resonance_strength: np.ndarray
    brightness: float
    harmonic_tilt: float
    temporal_style: float


# =============================================================================
# CREATE SYNTHETIC SPEAKER
# =============================================================================

def create_speaker_profile(
    speaker_id: int,
    n_mels: int = N_MELS,
) -> SpeakerProfile:

    rng = np.random.RandomState(
        1000 + speaker_id
    )

    spectral_offset = rng.normal(
        0.0,
        0.15,
        size=n_mels,
    ).astype(np.float32)

    kernel = (
        np.ones(
            7,
            dtype=np.float32,
        )
        / 7.0
    )

    spectral_offset = np.convolve(
        spectral_offset,
        kernel,
        mode="same",
    ).astype(np.float32)

    resonance_strength = rng.uniform(
        0.7,
        1.3,
        size=4,
    ).astype(np.float32)

    brightness = float(
        rng.uniform(
            0.75,
            1.25,
        )
    )

    harmonic_tilt = float(
        rng.uniform(
            -0.4,
            0.4,
        )
    )

    temporal_style = float(
        rng.uniform(
            0.85,
            1.15,
        )
    )

    return SpeakerProfile(
        spectral_offset=spectral_offset,
        resonance_strength=resonance_strength,
        brightness=brightness,
        harmonic_tilt=harmonic_tilt,
        temporal_style=temporal_style,
    )


# =============================================================================
# SYNTHETIC PHONEME TEMPLATES
# =============================================================================

def create_phoneme_templates(
    vocab_size: int = VOCAB_SIZE,
    n_mels: int = N_MELS,
) -> np.ndarray:

    rng = np.random.RandomState(
        1234
    )

    templates = rng.normal(
        0.0,
        0.30,
        size=(
            vocab_size,
            n_mels,
        ),
    ).astype(np.float32)

    for i in range(vocab_size):

        kernel = (
            np.ones(
                5,
                dtype=np.float32,
            )
            / 5.0
        )

        templates[i] = np.convolve(
            templates[i],
            kernel,
            mode="same",
        )

    return templates


PHONEME_TEMPLATES = (
    create_phoneme_templates()
)


# =============================================================================
# SYNTHETIC MEL GENERATION
# =============================================================================

def make_target_utterance(
    phonemes: np.ndarray,
    durations: np.ndarray,
    speaker: SpeakerProfile,
    noise_scale: float = 0.025,
) -> np.ndarray:

    frames = []

    n_mels = len(
        speaker.spectral_offset
    )

    mel_axis = np.linspace(
        0.0,
        1.0,
        n_mels,
        dtype=np.float32,
    )

    for phoneme_id, duration in zip(
        phonemes,
        durations,
    ):

        base = PHONEME_TEMPLATES[
            int(phoneme_id)
        ].copy()

        speaker_component = (
            speaker.spectral_offset.copy()
        )

        brightness_curve = (
            1.0
            + speaker.brightness
            * 0.25
            * (mel_axis - 0.5)
        )

        resonance = np.ones(
            n_mels,
            dtype=np.float32,
        )

        centers = np.array(
            [
                0.20,
                0.38,
                0.60,
                0.80,
            ],
            dtype=np.float32,
        )

        widths = np.array(
            [
                0.08,
                0.07,
                0.10,
                0.08,
            ],
            dtype=np.float32,
        )

        for strength, center, width in zip(
            speaker.resonance_strength,
            centers,
            widths,
        ):

            resonance += (
                0.20
                * strength
                * np.exp(
                    -0.5
                    * (
                        (mel_axis - center)
                        / width
                    ) ** 2
                )
            )

        acoustic_shape = (
            base * brightness_curve
            + speaker_component
        )

        acoustic_shape *= resonance

        for frame_index in range(
            int(duration)
        ):

            phase = (
                frame_index
                / max(
                    1,
                    int(duration) - 1,
                )
            )

            temporal_modulation = (
                0.08
                * np.sin(
                    2.0
                    * np.pi
                    * phase
                    + speaker.temporal_style
                )
            )

            frame = (
                acoustic_shape
                + temporal_modulation
            )

            frame += np.random.normal(
                0.0,
                noise_scale,
                size=n_mels,
            ).astype(np.float32)

            frames.append(
                frame.astype(
                    np.float32
                )
            )

    if not frames:

        return np.zeros(
            (
                1,
                n_mels,
            ),
            dtype=np.float32,
        )

    return np.stack(
        frames,
        axis=0,
    )


# =============================================================================
# RANDOM PHONEME SEQUENCE
# =============================================================================

def sample_phoneme_sequence(
    rng: np.random.RandomState,
    min_length: int = 5,
    max_length: int = MAX_PHONEMES,
) -> np.ndarray:

    length = rng.randint(
        min_length,
        max_length + 1,
    )

    return rng.randint(
        1,
        VOCAB_SIZE,
        size=length,
        dtype=np.int64,
    )


# =============================================================================
# RANDOM DURATIONS
# =============================================================================

def sample_durations(
    rng: np.random.RandomState,
    length: int,
    speaker: SpeakerProfile,
) -> np.ndarray:

    base = rng.randint(
        3,
        8,
        size=length,
    ).astype(np.float32)

    base *= speaker.temporal_style

    durations = np.clip(
        np.rint(base),
        1,
        10,
    ).astype(np.int64)

    return durations


# =============================================================================
# DATA SAMPLE
# =============================================================================

@dataclass
class Sample:

    reference_a: np.ndarray
    reference_a_length: int

    reference_b: np.ndarray
    reference_b_length: int

    phonemes: np.ndarray

    target_mel: np.ndarray

    target_durations: np.ndarray

    speaker_id: int


# =============================================================================
# SYNTHETIC DATASET
# =============================================================================

class DisentanglementDataset(
    Dataset
):

    def __init__(
        self,
        num_samples: int,
        speaker_ids: List[int],
        seed: int,
    ):

        super().__init__()

        self.samples: List[
            Sample
        ] = []

        for index in range(
            num_samples
        ):

            rng = np.random.RandomState(
                seed + index
            )

            speaker_id = int(
                rng.choice(
                    speaker_ids
                )
            )

            speaker = (
                create_speaker_profile(
                    speaker_id
                )
            )

            # =================================================================
            # REFERENCE A
            # =================================================================

            phonemes_a = (
                sample_phoneme_sequence(
                    rng,
                    6,
                    14,
                )
            )

            durations_a = (
                sample_durations(
                    rng,
                    len(phonemes_a),
                    speaker,
                )
            )

            reference_a = (
                make_target_utterance(
                    phonemes_a,
                    durations_a,
                    speaker,
                    noise_scale=0.04,
                )
            )

            # =================================================================
            # REFERENCE B
            #
            # Different linguistic content.
            # Same speaker.
            # =================================================================

            phonemes_b = (
                sample_phoneme_sequence(
                    rng,
                    6,
                    14,
                )
            )

            # Ensure different text representation.
            if np.array_equal(
                phonemes_a,
                phonemes_b,
            ):

                phonemes_b = np.roll(
                    phonemes_b,
                    1,
                )

            durations_b = (
                sample_durations(
                    rng,
                    len(phonemes_b),
                    speaker,
                )
            )

            reference_b = (
                make_target_utterance(
                    phonemes_b,
                    durations_b,
                    speaker,
                    noise_scale=0.04,
                )
            )

            # =================================================================
            # TARGET
            #
            # Different content again.
            # Same speaker.
            # =================================================================

            target_phonemes = (
                sample_phoneme_sequence(
                    rng,
                    5,
                    MAX_PHONEMES,
                )
            )

            target_durations = (
                sample_durations(
                    rng,
                    len(target_phonemes),
                    speaker,
                )
            )

            target_mel = (
                make_target_utterance(
                    target_phonemes,
                    target_durations,
                    speaker,
                    noise_scale=0.02,
                )
            )

            expected_frames = int(
                target_durations.sum()
            )

            assert (
                target_mel.shape[0]
                == expected_frames
            )

            self.samples.append(
                Sample(
                    reference_a=(
                        reference_a
                        .astype(
                            np.float32
                        )
                    ),
                    reference_a_length=(
                        reference_a.shape[0]
                    ),
                    reference_b=(
                        reference_b
                        .astype(
                            np.float32
                        )
                    ),
                    reference_b_length=(
                        reference_b.shape[0]
                    ),
                    phonemes=(
                        target_phonemes
                        .astype(
                            np.int64
                        )
                    ),
                    target_mel=(
                        target_mel
                        .astype(
                            np.float32
                        )
                    ),
                    target_durations=(
                        target_durations
                        .astype(
                            np.float32
                        )
                    ),
                    speaker_id=speaker_id,
                )
            )

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ):

        return self.samples[index]


# =============================================================================
# COLLATE
# =============================================================================

def collate_fn(
    batch: List[Sample],
):

    batch_size = len(batch)

    max_ref_a = max(
        x.reference_a_length
        for x in batch
    )

    max_ref_b = max(
        x.reference_b_length
        for x in batch
    )

    max_text = max(
        len(x.phonemes)
        for x in batch
    )

    max_target = max(
        len(x.target_mel)
        for x in batch
    )

    reference_a = torch.zeros(
        batch_size,
        N_MELS,
        max_ref_a,
    )

    reference_a_mask = torch.zeros(
        batch_size,
        max_ref_a,
        dtype=torch.bool,
    )

    reference_b = torch.zeros(
        batch_size,
        N_MELS,
        max_ref_b,
    )

    reference_b_mask = torch.zeros(
        batch_size,
        max_ref_b,
        dtype=torch.bool,
    )

    phonemes = torch.full(
        (
            batch_size,
            max_text,
        ),
        PAD_ID,
        dtype=torch.long,
    )

    phoneme_mask = torch.zeros(
        batch_size,
        max_text,
        dtype=torch.bool,
    )

    durations = torch.zeros(
        batch_size,
        max_text,
    )

    target_mel = torch.zeros(
        batch_size,
        max_target,
        N_MELS,
    )

    target_mask = torch.zeros(
        batch_size,
        max_target,
        dtype=torch.bool,
    )

    speaker_ids = torch.zeros(
        batch_size,
        dtype=torch.long,
    )

    for i, sample in enumerate(
        batch
    ):

        # Reference A
        t = sample.reference_a_length

        reference_a[
            i,
            :,
            :t,
        ] = torch.from_numpy(
            sample.reference_a.T
        )

        reference_a_mask[
            i,
            :t,
        ] = True

        # Reference B
        t = sample.reference_b_length

        reference_b[
            i,
            :,
            :t,
        ] = torch.from_numpy(
            sample.reference_b.T
        )

        reference_b_mask[
            i,
            :t,
        ] = True

        # Text
        t = len(
            sample.phonemes
        )

        phonemes[
            i,
            :t,
        ] = torch.from_numpy(
            sample.phonemes
        )

        phoneme_mask[
            i,
            :t,
        ] = True

        durations[
            i,
            :t,
        ] = torch.from_numpy(
            sample.target_durations
        )

        # Target Mel
        t = len(
            sample.target_mel
        )

        target_mel[
            i,
            :t,
        ] = torch.from_numpy(
            sample.target_mel
        )

        target_mask[
            i,
            :t,
        ] = True

        speaker_ids[
            i
        ] = sample.speaker_id

    return {
        "reference_a": reference_a,
        "reference_a_mask": reference_a_mask,
        "reference_b": reference_b,
        "reference_b_mask": reference_b_mask,
        "phonemes": phonemes,
        "phoneme_mask": phoneme_mask,
        "durations": durations,
        "target_mel": target_mel,
        "target_mask": target_mask,
        "speaker_ids": speaker_ids,
    }


# =============================================================================
# MASKED MEAN
# =============================================================================

def masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    mask_f = (
        mask.float()
        .unsqueeze(-1)
    )

    numerator = (
        x * mask_f
    ).sum(dim=1)

    denominator = (
        mask_f.sum(dim=1)
        .clamp_min(1.0)
    )

    return (
        numerator
        / denominator
    )


# =============================================================================
# SPEAKER ENCODER
# =============================================================================

class SpeakerEncoder(
    nn.Module
):

    def __init__(
        self,
        n_mels: int,
        embedding_dim: int,
        num_speakers: int,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                n_mels,
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

        self.projection = nn.Sequential(

            nn.Linear(
                128,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                embedding_dim,
            ),
        )

        # Training-only speaker classifier.
        self.classifier = nn.Linear(
            embedding_dim,
            num_speakers,
        )

    def forward(
        self,
        mel: torch.Tensor,
        mask: torch.Tensor,
    ):

        x = self.network(
            mel
        )

        x = x.transpose(
            1,
            2,
        )

        pooled = masked_mean(
            x,
            mask,
        )

        embedding = self.projection(
            pooled
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        logits = self.classifier(
            embedding
        )

        return (
            embedding,
            logits,
        )


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(
    nn.Module
):

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

        odd_width = pe[
            :,
            1::2,
        ].shape[1]

        pe[:, 1::2] = torch.cos(
            position
            * div_term[
                :odd_width
            ]
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        return (
            x
            + self.pe[
                :,
                :x.size(1),
            ]
        )


# =============================================================================
# TEXT ENCODER
# =============================================================================

class TextEncoder(
    nn.Module
):

    def __init__(
        self,
        vocab_size: int,
        dim: int,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            dim,
            padding_idx=PAD_ID,
        )

        self.position = (
            PositionalEncoding(
                dim
            )
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=dim * 4,
                dropout=0.1,
                batch_first=True,
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
        phonemes: torch.Tensor,
        padding_mask: torch.Tensor,
    ):

        x = self.embedding(
            phonemes
        )

        x = self.position(
            x
        )

        return self.encoder(
            x,
            src_key_padding_mask=(
                padding_mask
            ),
        )


# =============================================================================
# DURATION PREDICTOR
# =============================================================================

class DurationPredictor(
    nn.Module
):

    def __init__(
        self,
        dim: int,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                dim,
                dim,
            ),

            nn.ReLU(),

            nn.Linear(
                dim,
                dim // 2,
            ),

            nn.ReLU(),

            nn.Linear(
                dim // 2,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        return self.net(
            x
        ).squeeze(-1)


# =============================================================================
# LENGTH REGULATOR
# =============================================================================

def length_regulate(
    encoded: torch.Tensor,
    durations: torch.Tensor,
    mask: torch.Tensor,
):

    outputs = []

    lengths = []

    for b in range(
        encoded.size(0)
    ):

        pieces = []

        for t in range(
            encoded.size(1)
        ):

            if not bool(
                mask[
                    b,
                    t,
                ].item()
            ):
                continue

            duration = max(
                1,
                int(
                    round(
                        durations[
                            b,
                            t,
                        ].item()
                    )
                ),
            )

            pieces.append(
                encoded[
                    b,
                    t,
                ]
                .unsqueeze(0)
                .repeat(
                    duration,
                    1,
                )
            )

        if pieces:

            expanded = torch.cat(
                pieces,
                dim=0,
            )

        else:

            expanded = (
                encoded.new_zeros(
                    1,
                    encoded.size(-1),
                )
            )

        outputs.append(
            expanded
        )

        lengths.append(
            expanded.size(0)
        )

    max_length = max(
        lengths
    )

    padded = (
        encoded.new_zeros(
            encoded.size(0),
            max_length,
            encoded.size(-1),
        )
    )

    for b, expanded in enumerate(
        outputs
    ):

        padded[
            b,
            :expanded.size(0),
        ] = expanded

    lengths = torch.tensor(
        lengths,
        device=encoded.device,
    )

    return (
        padded,
        lengths,
    )


# =============================================================================
# DISENTANGLED TTS MODEL
# =============================================================================

class DisentangledTTS(
    nn.Module
):

    def __init__(
        self,
        vocab_size: int,
        n_mels: int,
        model_dim: int,
        speaker_embedding_dim: int,
        num_speakers: int,
    ):

        super().__init__()

        self.speaker_encoder = (
            SpeakerEncoder(
                n_mels,
                speaker_embedding_dim,
                num_speakers,
            )
        )

        self.text_encoder = (
            TextEncoder(
                vocab_size,
                model_dim,
            )
        )

        self.duration_predictor = (
            DurationPredictor(
                model_dim
            )
        )

        self.speaker_projection = (
            nn.Linear(
                speaker_embedding_dim,
                model_dim,
            )
        )

        decoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=4,
                dim_feedforward=model_dim * 4,
                dropout=0.1,
                batch_first=True,
            )
        )

        self.decoder = (
            nn.TransformerEncoder(
                decoder_layer,
                num_layers=2,
            )
        )

        self.mel_projection = nn.Linear(
            model_dim,
            n_mels,
        )

    def encode_speaker(
        self,
        mel: torch.Tensor,
        mask: torch.Tensor,
    ):

        return self.speaker_encoder(
            mel,
            mask,
        )

    def forward(
        self,
        reference_a: torch.Tensor,
        reference_a_mask: torch.Tensor,
        phonemes: torch.Tensor,
        phoneme_mask: torch.Tensor,
        durations: torch.Tensor,
    ):

        # ---------------------------------------------------------------------
        # Speaker representation
        # ---------------------------------------------------------------------

        (
            speaker_embedding,
            speaker_logits,
        ) = self.speaker_encoder(
            reference_a,
            reference_a_mask,
        )

        # ---------------------------------------------------------------------
        # Text representation
        # ---------------------------------------------------------------------

        text_features = (
            self.text_encoder(
                phonemes,
                ~phoneme_mask,
            )
        )

        # ---------------------------------------------------------------------
        # Duration prediction
        # ---------------------------------------------------------------------

        duration_log = (
            self.duration_predictor(
                text_features
            )
        )

        # ---------------------------------------------------------------------
        # Phoneme -> acoustic frames
        # ---------------------------------------------------------------------

        expanded, frame_lengths = (
            length_regulate(
                text_features,
                durations,
                phoneme_mask,
            )
        )

        # ---------------------------------------------------------------------
        # Speaker conditioning
        # ---------------------------------------------------------------------

        speaker_condition = (
            self.speaker_projection(
                speaker_embedding
            )
        )

        speaker_condition = (
            speaker_condition.unsqueeze(1)
        )

        expanded = (
            expanded
            + speaker_condition
        )

        # ---------------------------------------------------------------------
        # Frame mask
        # ---------------------------------------------------------------------

        max_frames = expanded.size(1)

        frame_mask = (
            torch.arange(
                max_frames,
                device=expanded.device,
            ).unsqueeze(0)
            >= frame_lengths.unsqueeze(1)
        )

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------

        decoded = self.decoder(
            expanded,
            src_key_padding_mask=frame_mask,
        )

        # ---------------------------------------------------------------------
        # Mel
        # ---------------------------------------------------------------------

        mel = self.mel_projection(
            decoded
        )

        return {
            "mel": mel,
            "duration_log": duration_log,
            "speaker_embedding": speaker_embedding,
            "speaker_logits": speaker_logits,
            "frame_lengths": frame_lengths,
        }


# =============================================================================
# MASKED MEL LOSS
# =============================================================================

def masked_mel_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
):

    length = min(
        prediction.size(1),
        target.size(1),
        mask.size(1),
    )

    prediction = prediction[
        :,
        :length,
    ]

    target = target[
        :,
        :length,
    ]

    mask = mask[
        :,
        :length,
    ]

    mask_f = (
        mask.float()
        .unsqueeze(-1)
    )

    error = (
        torch.abs(
            prediction - target
        )
        * mask_f
    )

    return (
        error.sum()
        / mask_f.sum().clamp_min(1.0)
    )


# =============================================================================
# DURATION LOSS
# =============================================================================

def duration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
):

    target_log = torch.log(
        target.clamp_min(1.0)
    )

    loss = F.smooth_l1_loss(
        prediction,
        target_log,
        reduction="none",
    )

    mask_f = mask.float()

    return (
        (loss * mask_f).sum()
        / mask_f.sum().clamp_min(1.0)
    )


# =============================================================================
# CONTRASTIVE SPEAKER LOSS
# =============================================================================

def contrastive_speaker_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    speaker_ids: torch.Tensor,
    margin: float = CONTRASTIVE_MARGIN,
):

    similarity = F.cosine_similarity(
        embedding_a,
        embedding_b,
        dim=-1,
    )

    same = (
        speaker_ids
        == speaker_ids
    )

    # -------------------------------------------------------------------------
    # This placeholder is replaced below because A/B always belong to the
    # same speaker in the dataset.
    #
    # The actual negative-pair term is generated by pairing A from the current
    # sample with B from another sample.
    # -------------------------------------------------------------------------

    positive_loss = (
        1.0 - similarity
    ).mean()

    return positive_loss


# =============================================================================
# BATCH CONTRASTIVE LOSS
# =============================================================================

def batch_contrastive_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    speaker_ids: torch.Tensor,
    margin: float = CONTRASTIVE_MARGIN,
):

    # -------------------------------------------------------------------------
    # Positive pairs
    #
    # A_i and B_i are guaranteed to be the same speaker.
    # -------------------------------------------------------------------------

    positive_similarity = (
        F.cosine_similarity(
            embedding_a,
            embedding_b,
            dim=-1,
        )
    )

    positive_loss = (
        1.0
        - positive_similarity
    ).mean()

    # -------------------------------------------------------------------------
    # Negative pairs
    #
    # Shift B by one batch position.
    #
    # Since the batch contains multiple speakers, most shifted pairs are
    # different speakers.
    #
    # We explicitly select positions whose speaker ID differs.
    # -------------------------------------------------------------------------

    shifted_b = torch.roll(
        embedding_b,
        shifts=1,
        dims=0,
    )

    shifted_ids = torch.roll(
        speaker_ids,
        shifts=1,
        dims=0,
    )

    negative_mask = (
        speaker_ids
        != shifted_ids
    )

    if negative_mask.any():

        negative_similarity = (
            F.cosine_similarity(
                embedding_a[
                    negative_mask
                ],
                shifted_b[
                    negative_mask
                ],
                dim=-1,
            )
        )

        negative_loss = F.relu(
            negative_similarity
            - margin
        ).mean()

    else:

        negative_loss = (
            embedding_a.sum()
            * 0.0
        )

    return (
        positive_loss
        + negative_loss
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model: DisentangledTTS,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
):

    model.train()

    totals = {
        "loss": 0.0,
        "mel": 0.0,
        "duration": 0.0,
        "speaker": 0.0,
        "contrastive": 0.0,
    }

    total_examples = 0

    for batch in loader:

        reference_a = batch[
            "reference_a"
        ].to(DEVICE)

        reference_a_mask = batch[
            "reference_a_mask"
        ].to(DEVICE)

        reference_b = batch[
            "reference_b"
        ].to(DEVICE)

        reference_b_mask = batch[
            "reference_b_mask"
        ].to(DEVICE)

        phonemes = batch[
            "phonemes"
        ].to(DEVICE)

        phoneme_mask = batch[
            "phoneme_mask"
        ].to(DEVICE)

        durations = batch[
            "durations"
        ].to(DEVICE)

        target_mel = batch[
            "target_mel"
        ].to(DEVICE)

        target_mask = batch[
            "target_mask"
        ].to(DEVICE)

        speaker_ids = batch[
            "speaker_ids"
        ].to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        # ---------------------------------------------------------------------
        # Main TTS forward pass
        # ---------------------------------------------------------------------

        output = model(
            reference_a=reference_a,
            reference_a_mask=reference_a_mask,
            phonemes=phonemes,
            phoneme_mask=phoneme_mask,
            durations=durations,
        )

        # ---------------------------------------------------------------------
        # Speaker embedding from second reference.
        # ---------------------------------------------------------------------

        embedding_b, logits_b = (
            model.encode_speaker(
                reference_b,
                reference_b_mask,
            )
        )

        # ---------------------------------------------------------------------
        # Mel reconstruction
        # ---------------------------------------------------------------------

        mel_loss = masked_mel_l1(
            output["mel"],
            target_mel,
            target_mask,
        )

        # ---------------------------------------------------------------------
        # Duration
        # ---------------------------------------------------------------------

        dur_loss = duration_loss(
            output["duration_log"],
            durations,
            phoneme_mask,
        )

        # ---------------------------------------------------------------------
        # Speaker classification
        #
        # Both references must identify the same speaker.
        # ---------------------------------------------------------------------

        speaker_loss_a = (
            F.cross_entropy(
                output["speaker_logits"],
                speaker_ids,
            )
        )

        speaker_loss_b = (
            F.cross_entropy(
                logits_b,
                speaker_ids,
            )
        )

        speaker_loss = (
            0.5
            * (
                speaker_loss_a
                + speaker_loss_b
            )
        )

        # ---------------------------------------------------------------------
        # Disentanglement / consistency
        # ---------------------------------------------------------------------

        contrastive_loss = (
            batch_contrastive_loss(
                output[
                    "speaker_embedding"
                ],
                embedding_b,
                speaker_ids,
            )
        )

        # ---------------------------------------------------------------------
        # Total
        # ---------------------------------------------------------------------

        loss = (
            mel_loss
            + 0.15 * dur_loss
            + 0.20 * speaker_loss
            + 0.25 * contrastive_loss
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite loss detected."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        batch_size = (
            reference_a.size(0)
        )

        totals["loss"] += (
            loss.item()
            * batch_size
        )

        totals["mel"] += (
            mel_loss.item()
            * batch_size
        )

        totals["duration"] += (
            dur_loss.item()
            * batch_size
        )

        totals["speaker"] += (
            speaker_loss.item()
            * batch_size
        )

        totals["contrastive"] += (
            contrastive_loss.item()
            * batch_size
        )

        total_examples += batch_size

    return {
        key: value
        / max(
            1,
            total_examples,
        )
        for key, value in totals.items()
    }


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model: DisentangledTTS,
    loader: DataLoader,
):

    model.eval()

    totals = {
        "loss": 0.0,
        "mel": 0.0,
        "duration": 0.0,
        "speaker": 0.0,
        "contrastive": 0.0,
    }

    correct = 0
    total = 0

    total_examples = 0

    for batch in loader:

        reference_a = batch[
            "reference_a"
        ].to(DEVICE)

        reference_a_mask = batch[
            "reference_a_mask"
        ].to(DEVICE)

        reference_b = batch[
            "reference_b"
        ].to(DEVICE)

        reference_b_mask = batch[
            "reference_b_mask"
        ].to(DEVICE)

        phonemes = batch[
            "phonemes"
        ].to(DEVICE)

        phoneme_mask = batch[
            "phoneme_mask"
        ].to(DEVICE)

        durations = batch[
            "durations"
        ].to(DEVICE)

        target_mel = batch[
            "target_mel"
        ].to(DEVICE)

        target_mask = batch[
            "target_mask"
        ].to(DEVICE)

        speaker_ids = batch[
            "speaker_ids"
        ].to(DEVICE)

        output = model(
            reference_a=reference_a,
            reference_a_mask=reference_a_mask,
            phonemes=phonemes,
            phoneme_mask=phoneme_mask,
            durations=durations,
        )

        embedding_b, logits_b = (
            model.encode_speaker(
                reference_b,
                reference_b_mask,
            )
        )

        mel_loss = masked_mel_l1(
            output["mel"],
            target_mel,
            target_mask,
        )

        dur_loss = duration_loss(
            output["duration_log"],
            durations,
            phoneme_mask,
        )

        speaker_loss_a = (
            F.cross_entropy(
                output["speaker_logits"],
                speaker_ids,
            )
        )

        speaker_loss_b = (
            F.cross_entropy(
                logits_b,
                speaker_ids,
            )
        )

        speaker_loss = (
            0.5
            * (
                speaker_loss_a
                + speaker_loss_b
            )
        )

        contrastive_loss = (
            batch_contrastive_loss(
                output[
                    "speaker_embedding"
                ],
                embedding_b,
                speaker_ids,
            )
        )

        loss = (
            mel_loss
            + 0.15 * dur_loss
            + 0.20 * speaker_loss
            + 0.25 * contrastive_loss
        )

        predictions = (
            output[
                "speaker_logits"
            ].argmax(
                dim=-1
            )
        )

        correct += (
            predictions
            == speaker_ids
        ).sum().item()

        total += (
            speaker_ids.numel()
        )

        batch_size = (
            reference_a.size(0)
        )

        for key, value in [
            ("loss", loss),
            ("mel", mel_loss),
            ("duration", dur_loss),
            ("speaker", speaker_loss),
            (
                "contrastive",
                contrastive_loss,
            ),
        ]:

            totals[key] += (
                value.item()
                * batch_size
            )

        total_examples += batch_size

    result = {
        key: value
        / max(
            1,
            total_examples,
        )
        for key, value in totals.items()
    }

    result[
        "speaker_accuracy"
    ] = (
        correct
        / max(
            1,
            total,
        )
    )

    return result


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_embedding(
    model: DisentangledTTS,
    mel: np.ndarray,
):

    model.eval()

    x = torch.from_numpy(
        mel.T
    ).float()

    x = x.unsqueeze(
        0
    ).to(DEVICE)

    mask = torch.ones(
        1,
        mel.shape[0],
        dtype=torch.bool,
        device=DEVICE,
    )

    embedding, _ = (
        model.encode_speaker(
            x,
            mask,
        )
    )

    return (
        embedding[0]
        .cpu()
        .numpy()
    )


# =============================================================================
# CREATE REFERENCE
# =============================================================================

def create_reference(
    speaker_id: int,
    seed: int,
):

    rng = np.random.RandomState(
        seed
    )

    speaker = (
        create_speaker_profile(
            speaker_id
        )
    )

    phonemes = (
        sample_phoneme_sequence(
            rng,
            6,
            14,
        )
    )

    durations = (
        sample_durations(
            rng,
            len(phonemes),
            speaker,
        )
    )

    return make_target_utterance(
        phonemes,
        durations,
        speaker,
        noise_scale=0.025,
    )


# =============================================================================
# COSINE
# =============================================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
):

    a = a / max(
        np.linalg.norm(a),
        1e-8,
    )

    b = b / max(
        np.linalg.norm(b),
        1e-8,
    )

    return float(
        np.dot(
            a,
            b,
        )
    )


# =============================================================================
# PLOT TRAINING
# =============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
):

    epochs = np.arange(
        1,
        len(
            history["train_loss"]
        ) + 1,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="train total",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="validation total",
    )

    plt.plot(
        epochs,
        history["train_mel"],
        label="train Mel",
    )

    plt.plot(
        epochs,
        history["val_mel"],
        label="validation Mel",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Lesson 20 — Training Curves"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "lesson20_training_curves.png",
        dpi=150,
    )

    plt.show()


# =============================================================================
# SAME-SPEAKER SIMILARITY
# =============================================================================

def plot_same_speaker_similarity(
    model: DisentangledTTS,
    speaker_ids: List[int],
):

    within = []
    between = []

    for speaker_id in speaker_ids:

        references = [
            create_reference(
                speaker_id,
                100000
                + speaker_id * 10
                + j,
            )
            for j in range(3)
        ]

        embeddings = [
            extract_embedding(
                model,
                x,
            )
            for x in references
        ]

        for i in range(
            len(embeddings)
        ):

            for j in range(
                i + 1,
                len(embeddings),
            ):

                within.append(
                    cosine_similarity(
                        embeddings[i],
                        embeddings[j],
                    )
                )

    # Different-speaker pairs
    for i in range(
        len(speaker_ids)
    ):

        for j in range(
            i + 1,
            len(speaker_ids),
        ):

            a = extract_embedding(
                model,
                create_reference(
                    speaker_ids[i],
                    200000 + i,
                ),
            )

            b = extract_embedding(
                model,
                create_reference(
                    speaker_ids[j],
                    300000 + j,
                ),
            )

            between.append(
                cosine_similarity(
                    a,
                    b,
                )
            )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        within,
        bins=15,
        alpha=0.7,
        label="same speaker",
    )

    plt.hist(
        between,
        bins=15,
        alpha=0.7,
        label="different speakers",
    )

    plt.xlabel(
        "Cosine similarity"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Speaker Embedding: "
        "Same vs Different Speaker"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "lesson20_same_speaker_similarity.png",
        dpi=150,
    )

    plt.show()

    print()
    print(
        "Mean same-speaker similarity: "
        f"{np.mean(within):.4f}"
    )

    print(
        "Mean different-speaker similarity: "
        f"{np.mean(between):.4f}"
    )


# =============================================================================
# EMBEDDING MATRIX
# =============================================================================

def plot_embedding_similarity(
    embeddings: np.ndarray,
    labels: List[str],
):

    normalized = (
        embeddings
        / np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        ).clip(
            min=1e-8
        )
    )

    matrix = (
        normalized
        @ normalized.T
    )

    plt.figure(
        figsize=(9, 8)
    )

    plt.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    plt.colorbar(
        label="Cosine similarity"
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(labels)),
        labels,
    )

    plt.title(
        "Lesson 20 — Speaker Embedding Similarity"
    )

    plt.tight_layout()

    plt.savefig(
        "lesson20_embedding_similarity.png",
        dpi=150,
    )

    plt.show()


# =============================================================================
# PCA
# =============================================================================

def pca_2d(
    embeddings: np.ndarray,
):

    centered = (
        embeddings
        - embeddings.mean(
            axis=0,
            keepdims=True,
        )
    )

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    return (
        centered
        @ vh[:2].T
    )


# =============================================================================
# PCA PLOT
# =============================================================================

def plot_embedding_pca(
    embeddings: np.ndarray,
    labels: List[str],
):

    points = pca_2d(
        embeddings
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=80,
    )

    for i, label in enumerate(
        labels
    ):

        plt.annotate(
            label,
            (
                points[i, 0],
                points[i, 1],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel(
        "PCA component 1"
    )

    plt.ylabel(
        "PCA component 2"
    )

    plt.title(
        "Lesson 20 — Speaker Embedding PCA"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        "lesson20_embedding_pca.png",
        dpi=150,
    )

    plt.show()


# =============================================================================
# REFERENCE VISUALIZATION
# =============================================================================

def plot_reference_examples(
    model: DisentangledTTS,
    speaker_id: int,
):

    references = [
        create_reference(
            speaker_id,
            500000 + i,
        )
        for i in range(3)
    ]

    embeddings = [
        extract_embedding(
            model,
            reference,
        )
        for reference in references
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
    )

    for i, (
        reference,
        embedding,
    ) in enumerate(
        zip(
            references,
            embeddings,
        )
    ):

        ax = axes[i]

        image = ax.imshow(
            reference.T,
            origin="lower",
            aspect="auto",
        )

        norm = np.linalg.norm(
            embedding
        )

        ax.set_title(
            f"Same speaker {speaker_id} — "
            f"reference {i + 1} — "
            f"embedding norm={norm:.3f}"
        )

        ax.set_xlabel(
            "Time frame"
        )

        ax.set_ylabel(
            "Mel bin"
        )

        fig.colorbar(
            image,
            ax=ax,
        )

    fig.tight_layout()

    fig.savefig(
        "lesson20_reference_examples.png",
        dpi=150,
    )

    plt.show()


# =============================================================================
# HELD-OUT ZERO-SHOT TEST
# =============================================================================

@torch.no_grad()
def zero_shot_test(
    model: DisentangledTTS,
    speaker_id: int,
):

    model.eval()

    reference = create_reference(
        speaker_id,
        900000 + speaker_id,
    )

    reference_tensor = (
        torch.from_numpy(
            reference.T
        )
        .float()
        .unsqueeze(0)
        .to(DEVICE)
    )

    reference_mask = torch.ones(
        1,
        reference.shape[0],
        dtype=torch.bool,
        device=DEVICE,
    )

    # New target text.
    rng = np.random.RandomState(
        910000 + speaker_id
    )

    target_phonemes = (
        sample_phoneme_sequence(
            rng,
            7,
            12,
        )
    )

    speaker = (
        create_speaker_profile(
            speaker_id
        )
    )

    target_durations = (
        sample_durations(
            rng,
            len(target_phonemes),
            speaker,
        )
    )

    target = make_target_utterance(
        target_phonemes,
        target_durations,
        speaker,
        noise_scale=0.02,
    )

    phoneme_tensor = torch.from_numpy(
        target_phonemes
    ).long().unsqueeze(0).to(
        DEVICE
    )

    phoneme_mask = torch.ones(
        1,
        len(target_phonemes),
        dtype=torch.bool,
        device=DEVICE,
    )

    duration_tensor = (
        torch.from_numpy(
            target_durations
        )
        .float()
        .unsqueeze(0)
        .to(DEVICE)
    )

    output = model(
        reference_a=reference_tensor,
        reference_a_mask=reference_mask,
        phonemes=phoneme_tensor,
        phoneme_mask=phoneme_mask,
        durations=duration_tensor,
    )

    generated = (
        output["mel"][0]
        .cpu()
        .numpy()
    )

    return (
        reference,
        target,
        generated,
    )


# =============================================================================
# PLOT ZERO-SHOT TEST
# =============================================================================

def plot_zero_shot_test(
    model: DisentangledTTS,
    speaker_id: int,
):

    reference, target, generated = (
        zero_shot_test(
            model,
            speaker_id,
        )
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 10),
    )

    axes[0].imshow(
        reference.T,
        origin="lower",
        aspect="auto",
    )

    axes[0].set_title(
        f"Reference audio — held-out speaker {speaker_id}"
    )

    axes[1].imshow(
        target.T,
        origin="lower",
        aspect="auto",
    )

    axes[1].set_title(
        "Ground-truth target Mel"
    )

    axes[2].imshow(
        generated.T,
        origin="lower",
        aspect="auto",
    )

    axes[2].set_title(
        "Generated Mel conditioned on reference speaker"
    )

    for ax in axes:

        ax.set_xlabel(
            "Time frame"
        )

        ax.set_ylabel(
            "Mel bin"
        )

    fig.tight_layout()

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 20 — "
            "Speaker/Content Disentanglement"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=DEFAULT_VAL_SAMPLES,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "PHASE 1 — LESSON 20"
    )
    print(
        "SPEAKER / CONTENT DISENTANGLEMENT"
    )
    print("=" * 78)

    print(
        f"Device: {DEVICE}"
    )

    # =========================================================================
    # SPEAKER SPLIT
    # =========================================================================

    train_speakers = list(
        range(
            NUM_TRAIN_SPEAKERS
        )
    )

    heldout_speakers = list(
        range(
            NUM_TRAIN_SPEAKERS,
            NUM_TRAIN_SPEAKERS
            + NUM_HELDOUT_SPEAKERS,
        )
    )

    print()
    print(
        f"Training speakers: "
        f"{train_speakers}"
    )

    print(
        f"Held-out speakers: "
        f"{heldout_speakers}"
    )

    # =========================================================================
    # DATA
    # =========================================================================

    train_dataset = (
        DisentanglementDataset(
            num_samples=args.train_samples,
            speaker_ids=train_speakers,
            seed=100,
        )
    )

    val_dataset = (
        DisentanglementDataset(
            num_samples=args.val_samples,
            speaker_ids=train_speakers,
            seed=10000,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # =========================================================================
    # MODEL
    # =========================================================================

    model = DisentangledTTS(
        vocab_size=VOCAB_SIZE,
        n_mels=N_MELS,
        model_dim=MODEL_DIM,
        speaker_embedding_dim=(
            SPEAKER_EMBED_DIM
        ),
        num_speakers=(
            NUM_TRAIN_SPEAKERS
        ),
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    # =========================================================================
    # HISTORY
    # =========================================================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mel": [],
        "val_mel": [],
        "train_contrastive": [],
        "val_contrastive": [],
        "train_speaker": [],
        "val_speaker": [],
        "val_accuracy": [],
    }

    best_val = float("inf")

    # =========================================================================
    # TRAIN
    # =========================================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
            )
        )

        val_metrics = (
            evaluate(
                model,
                val_loader,
            )
        )

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
            "train_contrastive"
        ].append(
            train_metrics[
                "contrastive"
            ]
        )

        history[
            "val_contrastive"
        ].append(
            val_metrics[
                "contrastive"
            ]
        )

        history[
            "train_speaker"
        ].append(
            train_metrics[
                "speaker"
            ]
        )

        history[
            "val_speaker"
        ].append(
            val_metrics[
                "speaker"
            ]
        )

        history[
            "val_accuracy"
        ].append(
            val_metrics[
                "speaker_accuracy"
            ]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['loss']:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"mel={val_metrics['mel']:.4f} | "
            f"contrastive="
            f"{val_metrics['contrastive']:.4f} | "
            f"speaker_acc="
            f"{val_metrics['speaker_accuracy']:.3f}"
        )

        # ---------------------------------------------------------------------
        # BEST CHECKPOINT
        # ---------------------------------------------------------------------

        if (
            val_metrics["loss"]
            < best_val
        ):

            best_val = (
                val_metrics["loss"]
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "vocab_size":
                        VOCAB_SIZE,

                    "n_mels":
                        N_MELS,

                    "model_dim":
                        MODEL_DIM,

                    "speaker_embedding_dim":
                        SPEAKER_EMBED_DIM,

                    "num_training_speakers":
                        NUM_TRAIN_SPEAKERS,
                },
                "lesson20_disentangled_tts.pt",
            )

    # =========================================================================
    # LOAD BEST MODEL
    # =========================================================================

    checkpoint = torch.load(
        "lesson20_disentangled_tts.pt",
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        "  lesson20_disentangled_tts.pt"
    )

    # =========================================================================
    # CORE DISENTANGLEMENT TEST
    # =========================================================================

    print()
    print("=" * 78)
    print(
        "CORE DISENTANGLEMENT TEST"
    )
    print("=" * 78)

    for speaker_id in train_speakers[:3]:

        reference_a = create_reference(
            speaker_id,
            400000 + speaker_id,
        )

        reference_b = create_reference(
            speaker_id,
            500000 + speaker_id,
        )

        embedding_a = extract_embedding(
            model,
            reference_a,
        )

        embedding_b = extract_embedding(
            model,
            reference_b,
        )

        similarity = cosine_similarity(
            embedding_a,
            embedding_b,
        )

        print(
            f"Speaker {speaker_id}: "
            f"same-speaker cosine="
            f"{similarity:.4f}"
        )

    # =========================================================================
    # HELD-OUT SPEAKER TEST
    # =========================================================================

    print()
    print("=" * 78)
    print(
        "HELD-OUT ZERO-SHOT SPEAKER TEST"
    )
    print("=" * 78)

    heldout_embeddings = []

    heldout_labels = []

    for speaker_id in heldout_speakers:

        reference_1 = create_reference(
            speaker_id,
            600000 + speaker_id,
        )

        reference_2 = create_reference(
            speaker_id,
            700000 + speaker_id,
        )

        embedding_1 = extract_embedding(
            model,
            reference_1,
        )

        embedding_2 = extract_embedding(
            model,
            reference_2,
        )

        same_similarity = (
            cosine_similarity(
                embedding_1,
                embedding_2,
            )
        )

        heldout_embeddings.append(
            embedding_1
        )

        heldout_labels.append(
            f"heldout-{speaker_id}"
        )

        print(
            f"Held-out speaker {speaker_id}: "
            f"reference/reference cosine="
            f"{same_similarity:.4f}"
        )

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    if not args.no_plot:

        plot_training_curves(
            history
        )

        plot_same_speaker_similarity(
            model,
            train_speakers,
        )

        all_embeddings = []

        all_labels = []

        # Training speakers
        for speaker_id in train_speakers:

            reference = create_reference(
                speaker_id,
                800000 + speaker_id,
            )

            embedding = extract_embedding(
                model,
                reference,
            )

            all_embeddings.append(
                embedding
            )

            all_labels.append(
                f"train-{speaker_id}"
            )

        # Held-out speakers
        for label, embedding in zip(
            heldout_labels,
            heldout_embeddings,
        ):

            all_embeddings.append(
                embedding
            )

            all_labels.append(
                label
            )

        all_embeddings = np.stack(
            all_embeddings,
            axis=0,
        )

        plot_embedding_similarity(
            all_embeddings,
            all_labels,
        )

        plot_embedding_pca(
            all_embeddings,
            all_labels,
        )

        # Same speaker, different reference content.
        plot_reference_examples(
            model,
            train_speakers[0],
        )

        # Held-out speaker zero-shot generation.
        plot_zero_shot_test(
            model,
            heldout_speakers[0],
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 78)
    print(
        "LESSON 20 COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        "The model was trained with:"
    )

    print(
        "  reference A -> speaker embedding"
    )

    print(
        "  reference B -> speaker embedding"
    )

    print(
        "  same speaker / different content"
    )

    print(
        "  speaker classification"
    )

    print(
        "  contrastive speaker consistency"
    )

    print(
        "  text-conditioned Mel generation"
    )

    print()
    print(
        "The critical inference path is:"
    )

    print(
        "  reference audio"
    )

    print(
        "       ↓"
    )

    print(
        "  speaker encoder"
    )

    print(
        "       ↓"
    )

    print(
        "  speaker embedding"
    )

    print(
        "       +"
    )

    print(
        "  NEW target text"
    )

    print(
        "       ↓"
    )

    print(
        "  acoustic model"
    )

    print(
        "       ↓"
    )

    print(
        "  generated Mel"
    )

    print()
    print(
        "Speaker ID is NOT required during inference."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is still a synthetic educational system."
    )

    print(
        "It does not produce natural human speech."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
