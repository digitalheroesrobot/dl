"""
===============================================================================
PHASE 1 — LESSON 19
ZERO-SHOT VOICE CLONING ARCHITECTURE
===============================================================================

FILE
----
zero_shot_voice_cloning.py

PURPOSE
-------
Build an educational reference-audio-conditioned TTS system.

Lesson 18 used:

    text + speaker_id -> TTS -> Mel

Lesson 19 changes this to:

    reference audio
          |
          v
    speaker encoder
          |
          v
    speaker embedding
          |
          +------------------+
                             |
    target text -> TTS ------+
                             |
                             v
                            Mel

The important idea is:

    SPEAKER IDENTITY
        comes from reference audio

    LINGUISTIC CONTENT
        comes from target text

===============================================================================
WHAT YOU LEARN
===============================================================================

1. Why fixed speaker-ID embeddings are not zero-shot.

2. Reference-audio speaker encoders.

3. Fixed-dimensional speaker embeddings.

4. L2-normalized speaker embeddings.

5. Auxiliary speaker classification.

6. Multiple reference utterances.

7. Speaker embedding aggregation.

8. Zero-shot speaker conditioning.

9. Held-out speaker evaluation.

10. Content/identity separation.

===============================================================================
IMPORTANT
===============================================================================

This is an EDUCATIONAL synthetic system.

It does NOT clone real human voices.

The synthetic speakers are generated from artificial acoustic profiles.

A real voice-cloning system requires:

    real speech
    transcripts
    text normalization
    phoneme/token representation
    speaker encoder
    acoustic model
    duration/alignment
    pitch/prosody
    neural vocoder
    evaluation
    sufficient training data

===============================================================================
CRITICAL VARIABLE-LENGTH FIX
===============================================================================

The previous version had this problem:

    padded phoneme
        duration = 0

was accidentally converted to:

    max(1, 0) = 1

This created extra predicted Mel frames.

The corrected length_regulate() receives:

    phoneme_mask

and completely ignores padded phonemes.

Therefore:

    real phonemes -> expanded
    PAD phonemes  -> ignored

This guarantees that teacher-forced duration expansion matches the target
Mel length.

===============================================================================
DEPENDENCIES
===============================================================================

pip install numpy matplotlib torch

RUN:

python zero_shot_voice_cloning.py

Quick test:

python zero_shot_voice_cloning.py --epochs 5

No visualization:

python zero_shot_voice_cloning.py --epochs 5 --no-plot
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

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
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

N_MELS = 80

VOCAB_SIZE = 64

TEXT_EMBED_DIM = 96

MODEL_DIM = 128

SPEAKER_EMBED_DIM = 64

MAX_PHONEMES = 16

PAD_ID = 0

NUM_TRAIN_SPEAKERS = 8

NUM_HELDOUT_SPEAKERS = 2

DEFAULT_TRAIN_SAMPLES = 640

DEFAULT_VAL_SAMPLES = 128

DEFAULT_BATCH_SIZE = 32

DEFAULT_EPOCHS = 20

DEFAULT_LR = 2e-3


# =============================================================================
# SYNTHETIC SPEAKER PROFILE
# =============================================================================

@dataclass
class SpeakerProfile:
    """
    Artificial acoustic identity used only for this educational experiment.
    """

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
        loc=0.0,
        scale=0.15,
        size=n_mels,
    ).astype(np.float32)

    kernel = np.ones(
        7,
        dtype=np.float32,
    ) / 7.0

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

    rng = np.random.RandomState(1234)

    templates = rng.normal(
        0.0,
        0.3,
        size=(vocab_size, n_mels),
    ).astype(np.float32)

    for i in range(vocab_size):

        kernel = np.ones(
            5,
            dtype=np.float32,
        ) / 5.0

        templates[i] = np.convolve(
            templates[i],
            kernel,
            mode="same",
        )

    return templates


PHONEME_TEMPLATES = create_phoneme_templates()


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

        for local_frame in range(
            int(duration)
        ):

            phase = (
                local_frame
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
            (1, n_mels),
            dtype=np.float32,
        )

    return np.stack(
        frames,
        axis=0,
    )


# =============================================================================
# SAMPLE PHONEME SEQUENCE
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
# SAMPLE DURATIONS
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

    reference_mel: np.ndarray

    reference_length: int

    phonemes: np.ndarray

    target_mel: np.ndarray

    target_durations: np.ndarray

    speaker_id: int


# =============================================================================
# DATASET
# =============================================================================

class SyntheticVoiceCloningDataset(
    Dataset
):

    def __init__(
        self,
        num_samples: int,
        speaker_ids: List[int],
        seed: int,
    ):

        super().__init__()

        self.samples: List[Sample] = []

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

            speaker = create_speaker_profile(
                speaker_id
            )

            # -------------------------------------------------------------
            # REFERENCE UTTERANCE
            # -------------------------------------------------------------

            reference_phonemes = (
                sample_phoneme_sequence(
                    rng,
                    min_length=7,
                    max_length=14,
                )
            )

            reference_durations = (
                sample_durations(
                    rng,
                    len(reference_phonemes),
                    speaker,
                )
            )

            reference_mel = (
                make_target_utterance(
                    reference_phonemes,
                    reference_durations,
                    speaker,
                    noise_scale=0.035,
                )
            )

            # -------------------------------------------------------------
            # TARGET UTTERANCE
            # -------------------------------------------------------------

            target_phonemes = (
                sample_phoneme_sequence(
                    rng,
                    min_length=5,
                    max_length=MAX_PHONEMES,
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

            # -------------------------------------------------------------
            # Ensure reference and target text are different.
            # -------------------------------------------------------------

            if np.array_equal(
                reference_phonemes,
                target_phonemes,
            ):

                target_phonemes = np.roll(
                    target_phonemes,
                    1,
                )

                target_mel = (
                    make_target_utterance(
                        target_phonemes,
                        target_durations,
                        speaker,
                        noise_scale=0.02,
                    )
                )

            # -------------------------------------------------------------
            # Dataset consistency check.
            # -------------------------------------------------------------

            expected_target_frames = int(
                target_durations.sum()
            )

            actual_target_frames = (
                target_mel.shape[0]
            )

            assert (
                expected_target_frames
                == actual_target_frames
            ), (
                "Target length mismatch: "
                f"expected={expected_target_frames}, "
                f"actual={actual_target_frames}"
            )

            self.samples.append(
                Sample(
                    reference_mel=(
                        reference_mel.astype(
                            np.float32
                        )
                    ),
                    reference_length=(
                        len(reference_mel)
                    ),
                    phonemes=(
                        target_phonemes.astype(
                            np.int64
                        )
                    ),
                    target_mel=(
                        target_mel.astype(
                            np.float32
                        )
                    ),
                    target_durations=(
                        target_durations.astype(
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
    ) -> Sample:

        return self.samples[index]


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def collate_fn(
    batch: List[Sample],
):

    batch_size = len(batch)

    max_ref = max(
        sample.reference_length
        for sample in batch
    )

    max_text = max(
        len(sample.phonemes)
        for sample in batch
    )

    max_target = max(
        len(sample.target_mel)
        for sample in batch
    )

    reference_mel = torch.zeros(
        batch_size,
        N_MELS,
        max_ref,
        dtype=torch.float32,
    )

    reference_mask = torch.zeros(
        batch_size,
        max_ref,
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

    target_mel = torch.zeros(
        batch_size,
        max_target,
        N_MELS,
        dtype=torch.float32,
    )

    target_mel_mask = torch.zeros(
        batch_size,
        max_target,
        dtype=torch.bool,
    )

    durations = torch.zeros(
        batch_size,
        max_text,
        dtype=torch.float32,
    )

    speaker_ids = torch.zeros(
        batch_size,
        dtype=torch.long,
    )

    for i, sample in enumerate(batch):

        ref_t = sample.reference_length

        reference_mel[
            i,
            :,
            :ref_t,
        ] = torch.from_numpy(
            sample.reference_mel.T
        )

        reference_mask[
            i,
            :ref_t,
        ] = True

        text_t = len(
            sample.phonemes
        )

        phonemes[
            i,
            :text_t,
        ] = torch.from_numpy(
            sample.phonemes
        )

        phoneme_mask[
            i,
            :text_t,
        ] = True

        target_t = len(
            sample.target_mel
        )

        target_mel[
            i,
            :target_t,
            :,
        ] = torch.from_numpy(
            sample.target_mel
        )

        target_mel_mask[
            i,
            :target_t,
        ] = True

        durations[
            i,
            :text_t,
        ] = torch.from_numpy(
            sample.target_durations
        )

        speaker_ids[
            i
        ] = sample.speaker_id

    return {
        "reference_mel": reference_mel,
        "reference_mask": reference_mask,
        "phonemes": phonemes,
        "phoneme_mask": phoneme_mask,
        "target_mel": target_mel,
        "target_mel_mask": target_mel_mask,
        "durations": durations,
        "speaker_ids": speaker_ids,
    }


# =============================================================================
# MASKED MEAN
# =============================================================================

def masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:

    mask_f = (
        mask.float()
        .unsqueeze(-1)
    )

    numerator = (
        x * mask_f
    ).sum(dim=dim)

    denominator = (
        mask_f.sum(dim=dim)
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

        self.conv = nn.Sequential(

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

        # Training-only auxiliary classifier.
        self.classifier = nn.Linear(
            embedding_dim,
            num_speakers,
        )

    def forward(
        self,
        reference_mel: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        h = self.conv(
            reference_mel
        )

        h = h.transpose(
            1,
            2,
        )

        pooled = masked_mean(
            h,
            reference_mask,
            dim=1,
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
        d_model: int,
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
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model,
        )

        pe[
            :,
            0::2,
        ] = torch.sin(
            position
            * div_term
        )

        pe[
            :,
            1::2,
        ] = torch.cos(
            position
            * div_term[
                :pe[:, 1::2].shape[1]
            ]
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

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
    ) -> torch.Tensor:

        x = self.embedding(
            phonemes
        )

        x = self.position(
            x
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        return x


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

        self.network = nn.Sequential(

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
    ) -> torch.Tensor:

        return self.network(
            x
        ).squeeze(-1)


# =============================================================================
# LENGTH REGULATOR
# =============================================================================

def length_regulate(
    encoded: torch.Tensor,
    durations: torch.Tensor,
    phoneme_mask: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Expand phoneme representations into acoustic frames.

    IMPORTANT:

    padded phonemes are ignored completely.

    If:

        phoneme_mask[b, t] == False

    the corresponding duration is NOT converted to one frame.
    """

    batch_size = encoded.size(0)

    outputs = []

    lengths = []

    for b in range(
        batch_size
    ):

        sequence = []

        for t in range(
            encoded.size(1)
        ):

            # -------------------------------------------------------------
            # FIX:
            # Ignore padding positions.
            # -------------------------------------------------------------

            if not bool(
                phoneme_mask[
                    b,
                    t,
                ].item()
            ):
                continue

            duration = int(
                round(
                    durations[
                        b,
                        t,
                    ].item()
                )
            )

            duration = max(
                1,
                duration,
            )

            repeated = (
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

            sequence.append(
                repeated
            )

        if sequence:

            expanded = torch.cat(
                sequence,
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
            batch_size,
            max_length,
            encoded.size(-1),
        )
    )

    for b, item in enumerate(
        outputs
    ):

        padded[
            b,
            :item.size(0),
        ] = item

    frame_lengths = torch.tensor(
        lengths,
        device=encoded.device,
        dtype=torch.long,
    )

    return (
        padded,
        frame_lengths,
    )


# =============================================================================
# ZERO-SHOT TTS
# =============================================================================

class ZeroShotTTS(
    nn.Module
):

    def __init__(
        self,
        vocab_size: int,
        n_mels: int,
        model_dim: int,
        speaker_embedding_dim: int,
        num_training_speakers: int,
    ):

        super().__init__()

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

        self.speaker_encoder = (
            SpeakerEncoder(
                n_mels=n_mels,
                embedding_dim=(
                    speaker_embedding_dim
                ),
                num_speakers=(
                    num_training_speakers
                ),
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
                dim_feedforward=(
                    model_dim * 4
                ),
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

        self.mel_projection = (
            nn.Linear(
                model_dim,
                n_mels,
            )
        )

    def forward(
        self,
        reference_mel: torch.Tensor,
        reference_mask: torch.Tensor,
        phonemes: torch.Tensor,
        phoneme_padding_mask: torch.Tensor,
        phoneme_mask: torch.Tensor,
        teacher_durations: torch.Tensor,
    ):

        # -----------------------------------------------------------------
        # Reference audio -> speaker embedding
        # -----------------------------------------------------------------

        (
            speaker_embedding,
            speaker_logits,
        ) = self.speaker_encoder(
            reference_mel,
            reference_mask,
        )

        # -----------------------------------------------------------------
        # Text -> linguistic representation
        # -----------------------------------------------------------------

        text_features = (
            self.text_encoder(
                phonemes,
                phoneme_padding_mask,
            )
        )

        # -----------------------------------------------------------------
        # Duration prediction
        # -----------------------------------------------------------------

        duration_log_prediction = (
            self.duration_predictor(
                text_features
            )
        )

        # -----------------------------------------------------------------
        # Length regulation
        #
        # IMPORTANT:
        # Pass phoneme_mask.
        # -----------------------------------------------------------------

        (
            expanded,
            frame_lengths,
        ) = length_regulate(
            text_features,
            teacher_durations,
            phoneme_mask,
        )

        # -----------------------------------------------------------------
        # Speaker conditioning
        # -----------------------------------------------------------------

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

        # -----------------------------------------------------------------
        # Frame mask
        # -----------------------------------------------------------------

        max_frames = (
            expanded.size(1)
        )

        frame_padding_mask = (
            torch.arange(
                max_frames,
                device=expanded.device,
            ).unsqueeze(0)
            >= frame_lengths.unsqueeze(1)
        )

        # -----------------------------------------------------------------
        # Acoustic decoder
        # -----------------------------------------------------------------

        decoded = self.decoder(
            expanded,
            src_key_padding_mask=(
                frame_padding_mask
            ),
        )

        # -----------------------------------------------------------------
        # Mel prediction
        # -----------------------------------------------------------------

        mel = self.mel_projection(
            decoded
        )

        return {
            "mel": mel,
            "duration_log": (
                duration_log_prediction
            ),
            "speaker_embedding": (
                speaker_embedding
            ),
            "speaker_logits": (
                speaker_logits
            ),
            "frame_lengths": (
                frame_lengths
            ),
        }


# =============================================================================
# MASKED MEL L1
# =============================================================================

def masked_mel_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate masked Mel L1 loss.

    The lengths should already match because length_regulate() uses the
    target durations.

    A common-length crop is retained as a defensive measure.
    """

    common_length = min(
        prediction.size(1),
        target.size(1),
        mask.size(1),
    )

    prediction = prediction[
        :,
        :common_length,
        :,
    ]

    target = target[
        :,
        :common_length,
        :,
    ]

    mask = mask[
        :,
        :common_length,
    ]

    mask_f = (
        mask.float()
        .unsqueeze(-1)
    )

    loss = (
        torch.abs(
            prediction
            - target
        )
        * mask_f
    )

    return (
        loss.sum()
        / mask_f.sum().clamp_min(
            1.0
        )
    )


# =============================================================================
# DURATION LOSS
# =============================================================================

def duration_loss(
    predicted_log_duration: torch.Tensor,
    target_duration: torch.Tensor,
    phoneme_mask: torch.Tensor,
) -> torch.Tensor:

    target_log = torch.log(
        target_duration.clamp_min(
            1.0
        )
    )

    loss = F.smooth_l1_loss(
        predicted_log_duration,
        target_log,
        reduction="none",
    )

    mask = phoneme_mask.float()

    return (
        (loss * mask).sum()
        / mask.sum().clamp_min(
            1.0
        )
    )


# =============================================================================
# TRAIN
# =============================================================================

def train_one_epoch(
    model: ZeroShotTTS,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
):

    model.train()

    total = 0.0
    total_mel = 0.0
    total_duration = 0.0
    total_speaker = 0.0

    for batch in loader:

        reference_mel = (
            batch[
                "reference_mel"
            ].to(DEVICE)
        )

        reference_mask = (
            batch[
                "reference_mask"
            ].to(DEVICE)
        )

        phonemes = (
            batch[
                "phonemes"
            ].to(DEVICE)
        )

        phoneme_mask = (
            batch[
                "phoneme_mask"
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

        durations = (
            batch[
                "durations"
            ].to(DEVICE)
        )

        speaker_ids = (
            batch[
                "speaker_ids"
            ].to(DEVICE)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            reference_mel=reference_mel,
            reference_mask=reference_mask,
            phonemes=phonemes,
            phoneme_padding_mask=(
                ~phoneme_mask
            ),
            phoneme_mask=phoneme_mask,
            teacher_durations=durations,
        )

        if torch.isnan(
            output["mel"]
        ).any():

            raise RuntimeError(
                "NaN detected in Mel prediction."
            )

        mel_loss = masked_mel_l1(
            output["mel"],
            target_mel,
            target_mel_mask,
        )

        dur_loss = duration_loss(
            output[
                "duration_log"
            ],
            durations,
            phoneme_mask,
        )

        speaker_loss = F.cross_entropy(
            output[
                "speaker_logits"
            ],
            speaker_ids,
        )

        loss = (
            mel_loss
            + 0.15 * dur_loss
            + 0.25 * speaker_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = (
            reference_mel.size(0)
        )

        total += (
            loss.item()
            * batch_size
        )

        total_mel += (
            mel_loss.item()
            * batch_size
        )

        total_duration += (
            dur_loss.item()
            * batch_size
        )

        total_speaker += (
            speaker_loss.item()
            * batch_size
        )

    n = len(
        loader.dataset
    )

    return {
        "loss": total / n,
        "mel": total_mel / n,
        "duration": total_duration / n,
        "speaker": total_speaker / n,
    }


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model: ZeroShotTTS,
    loader: DataLoader,
):

    model.eval()

    total = 0.0
    total_mel = 0.0
    total_duration = 0.0
    total_speaker = 0.0

    correct = 0
    count = 0

    for batch in loader:

        reference_mel = (
            batch[
                "reference_mel"
            ].to(DEVICE)
        )

        reference_mask = (
            batch[
                "reference_mask"
            ].to(DEVICE)
        )

        phonemes = (
            batch[
                "phonemes"
            ].to(DEVICE)
        )

        phoneme_mask = (
            batch[
                "phoneme_mask"
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

        durations = (
            batch[
                "durations"
            ].to(DEVICE)
        )

        speaker_ids = (
            batch[
                "speaker_ids"
            ].to(DEVICE)
        )

        output = model(
            reference_mel=reference_mel,
            reference_mask=reference_mask,
            phonemes=phonemes,
            phoneme_padding_mask=(
                ~phoneme_mask
            ),
            phoneme_mask=phoneme_mask,
            teacher_durations=durations,
        )

        mel_loss = masked_mel_l1(
            output["mel"],
            target_mel,
            target_mel_mask,
        )

        dur_loss = duration_loss(
            output[
                "duration_log"
            ],
            durations,
            phoneme_mask,
        )

        speaker_loss = F.cross_entropy(
            output[
                "speaker_logits"
            ],
            speaker_ids,
        )

        loss = (
            mel_loss
            + 0.15 * dur_loss
            + 0.25 * speaker_loss
        )

        predictions = (
            output[
                "speaker_logits"
            ]
            .argmax(dim=-1)
        )

        correct += (
            predictions
            == speaker_ids
        ).sum().item()

        count += speaker_ids.numel()

        batch_size = (
            reference_mel.size(0)
        )

        total += (
            loss.item()
            * batch_size
        )

        total_mel += (
            mel_loss.item()
            * batch_size
        )

        total_duration += (
            dur_loss.item()
            * batch_size
        )

        total_speaker += (
            speaker_loss.item()
            * batch_size
        )

    n = len(
        loader.dataset
    )

    return {
        "loss": total / n,
        "mel": total_mel / n,
        "duration": total_duration / n,
        "speaker": total_speaker / n,
        "speaker_accuracy": (
            correct / max(
                1,
                count,
            )
        ),
    }


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_embedding(
    model: ZeroShotTTS,
    reference_mel: np.ndarray,
) -> np.ndarray:

    model.eval()

    tensor = torch.from_numpy(
        reference_mel.T
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)

    mask = torch.ones(
        1,
        reference_mel.shape[0],
        dtype=torch.bool,
        device=DEVICE,
    )

    embedding, _ = (
        model.speaker_encoder(
            tensor,
            mask,
        )
    )

    return (
        embedding[0]
        .cpu()
        .numpy()
    )


# =============================================================================
# MULTI-REFERENCE EMBEDDING
# =============================================================================

@torch.no_grad()
def extract_multi_reference_embedding(
    model: ZeroShotTTS,
    reference_mels: List[np.ndarray],
) -> np.ndarray:

    embeddings = []

    for reference in reference_mels:

        embedding = extract_embedding(
            model,
            reference,
        )

        embeddings.append(
            embedding
        )

    mean_embedding = np.mean(
        np.stack(
            embeddings,
            axis=0,
        ),
        axis=0,
    )

    norm = np.linalg.norm(
        mean_embedding
    )

    return (
        mean_embedding
        / max(
            norm,
            1e-8,
        )
    )


# =============================================================================
# COSINE SIMILARITY
# =============================================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a_norm = (
        a
        / max(
            np.linalg.norm(a),
            1e-8,
        )
    )

    b_norm = (
        b
        / max(
            np.linalg.norm(b),
            1e-8,
        )
    )

    return float(
        np.dot(
            a_norm,
            b_norm,
        )
    )


# =============================================================================
# CREATE HELD-OUT REFERENCE
# =============================================================================

def create_reference_for_speaker(
    speaker_id: int,
    seed: int,
) -> np.ndarray:

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
            min_length=8,
            max_length=13,
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
# TRAINING CURVES
# =============================================================================

def plot_training_curves(
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
        figsize=(10, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="train",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="validation",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Lesson 19 — Zero-Shot TTS Training"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "lesson19_training_curves.png",
        dpi=150,
    )

    plt.show()


# =============================================================================
# SIMILARITY MATRIX
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

    similarity = (
        normalized
        @ normalized.T
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        similarity,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    plt.colorbar(
        label="Cosine similarity"
    )

    plt.xticks(
        range(
            len(labels)
        ),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(
            len(labels)
        ),
        labels,
    )

    plt.title(
        "Lesson 19 — Speaker Embedding Similarity"
    )

    plt.tight_layout()

    plt.savefig(
        "lesson19_embedding_similarity.png",
        dpi=150,
    )

    plt.show()


# =============================================================================
# PCA
# =============================================================================

def pca_2d(
    embeddings: np.ndarray,
) -> np.ndarray:

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

    components = vh[:2]

    return (
        centered
        @ components.T
    )


# =============================================================================
# PCA PLOT
# =============================================================================

def plot_embedding_pca(
    embeddings: np.ndarray,
    labels: List[str],
):

    coordinates = pca_2d(
        embeddings
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=80,
    )

    for i, label in enumerate(
        labels
    ):

        plt.annotate(
            label,
            (
                coordinates[i, 0],
                coordinates[i, 1],
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
        "Lesson 19 — Speaker Embedding PCA"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        "lesson19_embedding_pca.png",
        dpi=150,
    )

    plt.show()


# =============================================================================
# REFERENCE MEL PLOT
# =============================================================================

def plot_zero_shot_references(
    reference_mels: List[np.ndarray],
    embeddings: List[np.ndarray],
    speaker_ids: List[int],
):

    count = len(
        reference_mels
    )

    fig, axes = plt.subplots(
        count,
        1,
        figsize=(
            10,
            3 * count,
        ),
        squeeze=False,
    )

    for i, (
        mel,
        embedding,
        speaker_id,
    ) in enumerate(
        zip(
            reference_mels,
            embeddings,
            speaker_ids,
        )
    ):

        ax = axes[
            i,
            0,
        ]

        image = ax.imshow(
            mel.T,
            origin="lower",
            aspect="auto",
        )

        embedding_norm = np.linalg.norm(
            embedding
        )

        ax.set_title(
            "Held-out speaker "
            f"{speaker_id} | "
            f"embedding norm = "
            f"{embedding_norm:.3f}"
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
        "lesson19_heldout_references.png",
        dpi=150,
    )

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 19 — "
            "Zero-Shot Voice Cloning"
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
    print("PHASE 1 — LESSON 19")
    print("ZERO-SHOT VOICE CLONING ARCHITECTURE")
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
    # DATASETS
    # =========================================================================

    train_dataset = (
        SyntheticVoiceCloningDataset(
            num_samples=args.train_samples,
            speaker_ids=train_speakers,
            seed=100,
        )
    )

    val_dataset = (
        SyntheticVoiceCloningDataset(
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

    model = ZeroShotTTS(
        vocab_size=VOCAB_SIZE,
        n_mels=N_MELS,
        model_dim=MODEL_DIM,
        speaker_embedding_dim=SPEAKER_EMBED_DIM,
        num_training_speakers=NUM_TRAIN_SPEAKERS,
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
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
        "train_speaker": [],
        "val_speaker": [],
        "val_accuracy": [],
    }

    best_val = float(
        "inf"
    )

    # =========================================================================
    # TRAINING
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
            "train_speaker"
        ].append(
            train_metrics["speaker"]
        )

        history[
            "val_speaker"
        ].append(
            val_metrics["speaker"]
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
            f"speaker_acc="
            f"{val_metrics['speaker_accuracy']:.3f}"
        )

        # ---------------------------------------------------------------------
        # SAVE BEST CHECKPOINT
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
                "lesson19_zero_shot_tts.pt",
            )

    print()

    print(
        "Best checkpoint saved:"
    )

    print(
        "  lesson19_zero_shot_tts.pt"
    )

    # =========================================================================
    # LOAD BEST CHECKPOINT
    # =========================================================================

    checkpoint = torch.load(
        "lesson19_zero_shot_tts.pt",
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # =========================================================================
    # HELD-OUT SPEAKER EXPERIMENT
    # =========================================================================

    print()
    print("=" * 78)
    print("HELD-OUT SPEAKER EXPERIMENT")
    print("=" * 78)

    heldout_references = []

    heldout_embeddings = []

    for speaker_id in heldout_speakers:

        references = [
            create_reference_for_speaker(
                speaker_id,
                seed=(
                    50000
                    + speaker_id * 10
                    + j
                ),
            )
            for j in range(3)
        ]

        single_embedding = (
            extract_embedding(
                model,
                references[0],
            )
        )

        multi_embedding = (
            extract_multi_reference_embedding(
                model,
                references,
            )
        )

        heldout_references.append(
            references[0]
        )

        heldout_embeddings.append(
            multi_embedding
        )

        single_norm = np.linalg.norm(
            single_embedding
        )

        multi_norm = np.linalg.norm(
            multi_embedding
        )

        single_multi_similarity = (
            cosine_similarity(
                single_embedding,
                multi_embedding,
            )
        )

        print(
            f"Held-out speaker {speaker_id}: "
            f"single_norm={single_norm:.4f}, "
            f"multi_norm={multi_norm:.4f}, "
            f"single/multi cosine="
            f"{single_multi_similarity:.4f}"
        )

    # =========================================================================
    # CROSS-SPEAKER SIMILARITY
    # =========================================================================

    if len(
        heldout_embeddings
    ) >= 2:

        cross_similarity = (
            cosine_similarity(
                heldout_embeddings[0],
                heldout_embeddings[1],
            )
        )

        print()

        print(
            "Cosine similarity between "
            "different held-out speakers: "
            f"{cross_similarity:.4f}"
        )

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    if not args.no_plot:

        plot_training_curves(
            history
        )

        all_embeddings = []

        all_labels = []

        # ---------------------------------------------------------------------
        # Training speakers
        # ---------------------------------------------------------------------

        for speaker_id in train_speakers:

            reference = (
                create_reference_for_speaker(
                    speaker_id,
                    seed=70000 + speaker_id,
                )
            )

            embedding = (
                extract_embedding(
                    model,
                    reference,
                )
            )

            all_embeddings.append(
                embedding
            )

            all_labels.append(
                f"train-{speaker_id}"
            )

        # ---------------------------------------------------------------------
        # Held-out speakers
        # ---------------------------------------------------------------------

        for speaker_id, embedding in zip(
            heldout_speakers,
            heldout_embeddings,
        ):

            all_embeddings.append(
                embedding
            )

            all_labels.append(
                f"heldout-{speaker_id}"
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

        plot_zero_shot_references(
            heldout_references,
            heldout_embeddings,
            heldout_speakers,
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 78)
    print("LESSON 19 COMPLETE")
    print("=" * 78)

    print()
    print("Architecture:")
    print()
    print("  reference audio")
    print("       ↓")
    print("  speaker encoder")
    print("       ↓")
    print("  speaker embedding")
    print("       +")
    print("  target phonemes")
    print("       ↓")
    print("  text encoder + duration model")
    print("       ↓")
    print("  speaker-conditioned decoder")
    print("       ↓")
    print("  target Mel spectrogram")

    print()
    print(
        "Speaker ID is used only as an auxiliary "
        "training target."
    )

    print(
        "Inference uses reference audio rather "
        "than a speaker ID."
    )

    print()
    print(
        "Variable-length padding bug: FIXED"
    )

    print(
        "Nested f-string syntax bug: FIXED"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()