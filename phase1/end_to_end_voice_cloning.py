"""
========================================================================
PHASE 1 — LESSON 24
END-TO-END VOICE CLONING
========================================================================

PURPOSE
-------
Integrate the major components learned throughout Phase 1 into one
complete educational voice-cloning pipeline.

PIPELINE:

    Reference Speech
          |
          v
    Speaker Encoder
          |
          v
    Speaker Embedding
          |
          +-----------------------------+
          |                             |
          v                             v
      Target Text                  Speaker Identity
          |                             |
          v                             |
      Text Encoder                      |
          |                             |
          v                             |
    Duration Predictor                  |
          |                             |
          v                             |
    Length Regulation                   |
          |                             |
          +-------------+---------------+
                        |
                        v
                Acoustic Decoder
                        |
                        v
                  Mel Spectrogram
                        |
                        v
                 Neural Vocoder
                        |
                        v
                 Generated Waveform


------------------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------------------

STEP 1 — Speaker identity extraction

    Reference audio -> speaker embedding

STEP 2 — Text/content representation

    Target text -> token sequence -> text representation

STEP 3 — Speaker/content conditioning

    content representation
              +
    speaker embedding
              |
              v
       acoustic model

STEP 4 — Duration modeling

    Token sequence and Mel frames have different temporal resolutions.

STEP 5 — Length regulation

    Token-level features -> frame-level features

STEP 6 — Mel prediction

    Frame features -> Mel spectrogram

STEP 7 — Neural vocoding

    Mel spectrogram -> waveform

STEP 8 — End-to-end inference

    Reference speaker
          +
    target content
          |
          v
    synthesized waveform


------------------------------------------------------------------------
IMPORTANT EDUCATIONAL LIMITATION
------------------------------------------------------------------------

This lesson uses SYNTHETIC speech-like data.

It demonstrates the architecture and training mechanics, but it is NOT
a production voice-cloning system.

A real system requires:

    - real speech datasets
    - phoneme / linguistic frontend
    - speaker-balanced data
    - reliable alignment
    - real speaker embeddings
    - stronger acoustic models
    - production neural vocoder
    - prosody modeling
    - robust duration prediction
    - large-scale training
    - multilingual text processing
    - evaluation


------------------------------------------------------------------------
IMPORTANT FIXES IN THIS VERSION
------------------------------------------------------------------------

FIX 1
-----

The vocoder DataLoader previously used PyTorch's default collator.

That failed because token sequences have variable lengths:

    [9]
    [5]
    [11]

The default collator attempted:

    torch.stack(...)

and crashed.

This version uses:

    collate_fn=collate_batch


FIX 2
-----

The previous version used:

    torchaudio.save()

Modern torchaudio can route audio saving through TorchCodec.

That requires compatible FFmpeg shared libraries.

Your environment currently reports:

    libavutil.so.61 not found
    libavutil.so.60 not found
    libavutil.so.59 not found
    ...

Therefore this lesson does NOT use torchaudio for WAV output.

Instead we use Python's standard-library:

    wave

This requires:

    NO FFmpeg
    NO TorchCodec
    NO torchaudio.save()


------------------------------------------------------------------------
OUTPUT
------------------------------------------------------------------------

    lesson24_end_to_end_voice_cloning.pt

    lesson24_best_acoustic_model.pt

    lesson24_reference.wav
    lesson24_target.wav
    lesson24_generated.wav

    lesson24_reference_mel.png
    lesson24_target_mel.png
    lesson24_generated_mel.png
    lesson24_waveforms.png
    lesson24_training_curves.png
    lesson24_speaker_similarity.png


------------------------------------------------------------------------
NEXT
------------------------------------------------------------------------

Lesson 25:

    FEW-SHOT VOICE CLONING

Topics:

    1. Multiple reference utterances
    2. Reference aggregation
    3. Mean speaker embedding
    4. Attention-based reference aggregation
    5. Reference dropout
    6. Noisy reference audio
    7. Robust speaker identity extraction
    8. Few-shot / zero-shot evaluation
"""

import math
import os
import random
import wave

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:

    # ------------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------------

    sample_rate: int = 16000

    n_fft: int = 512

    hop_length: int = 256

    win_length: int = 512

    n_mels: int = 80

    mel_frames: int = 32

    waveform_length: int = (
        32 * 256
    )

    # ------------------------------------------------------------------------
    # Speakers
    # ------------------------------------------------------------------------

    num_speakers: int = 12

    train_samples: int = 768

    val_samples: int = 128

    # ------------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------------

    vocab_size: int = 64

    max_phonemes: int = 12

    # ------------------------------------------------------------------------
    # Acoustic model
    # ------------------------------------------------------------------------

    model_dim: int = 128

    speaker_embedding_dim: int = 64

    num_heads: int = 4

    num_transformer_layers: int = 2

    # ------------------------------------------------------------------------
    # Acoustic training
    # ------------------------------------------------------------------------

    batch_size: int = 16

    epochs: int = 20

    learning_rate: float = 2e-3

    # ------------------------------------------------------------------------
    # Vocoder
    # ------------------------------------------------------------------------

    vocoder_epochs: int = 6

    vocoder_learning_rate: float = 2e-3

    vocoder_batch_size: int = 16

    # ------------------------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------------------------

    duration_weight: float = 0.15

    speaker_weight: float = 0.20

    consistency_weight: float = 0.25

    # ------------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------------

    seed: int = 42

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


CFG = Config()

DEVICE = torch.device(
    CFG.device
)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ============================================================================
# MEL FILTERBANK
# ============================================================================

def hz_to_mel(freq):

    return 2595.0 * math.log10(
        1.0 + freq / 700.0
    )


def mel_to_hz(mel):

    return 700.0 * (
        10.0 ** (mel / 2595.0)
        - 1.0
    )


def create_mel_filterbank():

    f_min = 0.0

    f_max = (
        CFG.sample_rate / 2.0
    )

    mel_min = hz_to_mel(
        f_min
    )

    mel_max = hz_to_mel(
        f_max
    )

    mel_points = torch.linspace(
        mel_min,
        mel_max,
        CFG.n_mels + 2,
    )

    hz_points = torch.tensor(
        [
            mel_to_hz(
                float(x)
            )
            for x in mel_points
        ],
        dtype=torch.float32,
    )

    bins = torch.floor(
        (CFG.n_fft + 1)
        * hz_points
        / CFG.sample_rate
    ).long()

    filterbank = torch.zeros(
        CFG.n_mels,
        CFG.n_fft // 2 + 1,
    )

    max_bin = CFG.n_fft // 2

    for m in range(
        1,
        CFG.n_mels + 1,
    ):

        left = int(
            bins[m - 1]
        )

        center = int(
            bins[m]
        )

        right = int(
            bins[m + 1]
        )

        left = max(
            0,
            min(
                left,
                max_bin,
            ),
        )

        center = max(
            left + 1,
            min(
                center,
                max_bin,
            ),
        )

        right = max(
            center + 1,
            min(
                right,
                max_bin + 1,
            ),
        )

        for k in range(
            left,
            center,
        ):

            if (
                0
                <= k
                < filterbank.shape[1]
            ):

                filterbank[
                    m - 1,
                    k
                ] = (
                    k - left
                ) / max(
                    center - left,
                    1,
                )

        for k in range(
            center,
            right,
        ):

            if (
                0
                <= k
                < filterbank.shape[1]
            ):

                filterbank[
                    m - 1,
                    k
                ] = (
                    right - k
                ) / max(
                    right - center,
                    1,
                )

    return filterbank


MEL_FILTERBANK = (
    create_mel_filterbank()
)


# ============================================================================
# WAVEFORM -> MEL
# ============================================================================

def waveform_to_mel(
    waveform,
):

    if waveform.dim() == 1:

        waveform = waveform.unsqueeze(
            0
        )

    elif waveform.dim() == 3:

        waveform = waveform.squeeze(
            1
        )

    if waveform.dim() != 2:

        raise ValueError(
            "waveform_to_mel expects "
            "[T], [B,T], or [B,1,T]. "
            f"Received {tuple(waveform.shape)}"
        )

    window = torch.hann_window(
        CFG.win_length,
        device=waveform.device,
    )

    spec = torch.stft(
        waveform,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        win_length=CFG.win_length,
        window=window,
        center=True,
        return_complex=True,
    )

    magnitude = spec.abs()

    mel_filterbank = (
        MEL_FILTERBANK.to(
            waveform.device
        )
    )

    mel = torch.matmul(
        mel_filterbank,
        magnitude,
    )

    mel = torch.log(
        mel.clamp_min(
            1e-5
        )
    )

    if mel.shape[-1] > CFG.mel_frames:

        mel = mel[
            ...,
            :CFG.mel_frames
        ]

    elif mel.shape[-1] < CFG.mel_frames:

        mel = F.pad(
            mel,
            (
                0,
                CFG.mel_frames
                - mel.shape[-1],
            ),
        )

    return mel


# ============================================================================
# SYNTHETIC SPEAKER PROFILE
# ============================================================================

class SpeakerProfile:

    def __init__(
        self,
        speaker_id,
    ):

        rng = np.random.RandomState(
            speaker_id
        )

        self.base_f0 = (
            105.0
            + rng.uniform(
                0,
                150,
            )
        )

        self.harmonic_tilt = (
            0.5
            + rng.uniform(
                0.0,
                0.9,
            )
        )

        self.brightness = (
            rng.uniform(
                0.5,
                1.5,
            )
        )

        self.resonance = (
            rng.uniform(
                0.5,
                1.5,
            )
        )

        self.temporal_style = (
            rng.uniform(
                0.5,
                1.5,
            )
        )


# ============================================================================
# SYNTHETIC SPEECH
# ============================================================================

def generate_synthetic_speech(
    speaker_id,
    seed,
):

    rng = np.random.RandomState(
        seed
    )

    profile = SpeakerProfile(
        speaker_id
    )

    length = CFG.waveform_length

    t = (
        torch.arange(
            length,
            dtype=torch.float32,
        )
        / CFG.sample_rate
    )

    # ------------------------------------------------------------------------
    # Fundamental frequency
    # ------------------------------------------------------------------------

    f0 = (
        profile.base_f0
        + 15.0
        * torch.sin(
            2.0
            * math.pi
            * (
                2.0
                + profile.temporal_style
            )
            * t
        )
    )

    # ------------------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------------------

    phase = torch.cumsum(
        2.0
        * math.pi
        * f0
        / CFG.sample_rate,
        dim=0,
    )

    # ------------------------------------------------------------------------
    # Harmonic synthesis
    # ------------------------------------------------------------------------

    waveform = torch.zeros_like(
        t
    )

    for harmonic in range(
        1,
        16,
    ):

        amplitude = (
            1.0
            / (
                harmonic
                ** profile.harmonic_tilt
            )
        )

        amplitude *= (
            profile.brightness
            ** (
                -0.15
                * harmonic
            )
        )

        waveform += (
            amplitude
            * torch.sin(
                harmonic * phase
            )
        )

    # ------------------------------------------------------------------------
    # Resonance
    # ------------------------------------------------------------------------

    resonance = (
        0.65
        + 0.25
        * profile.resonance
        * torch.sin(
            2.0
            * math.pi
            * 3.0
            * t
            + rng.uniform(
                0,
                2 * math.pi,
            )
        )
    )

    waveform *= resonance

    # ------------------------------------------------------------------------
    # Temporal envelope
    # ------------------------------------------------------------------------

    envelope = (
        0.60
        + 0.30
        * torch.sin(
            2.0
            * math.pi
            * (
                1.5
                + profile.temporal_style
            )
            * t
            + rng.uniform(
                0,
                2 * math.pi,
            )
        )
    )

    envelope = envelope.clamp(
        min=0.05
    )

    waveform *= envelope

    # ------------------------------------------------------------------------
    # Non-linearity
    # ------------------------------------------------------------------------

    waveform = torch.tanh(
        waveform * 0.25
    )

    # ------------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------------

    noise = (
        torch.randn(
            length
        )
        * 0.002
    )

    waveform += noise

    # ------------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------------

    waveform /= (
        waveform.abs().max()
        + 1e-8
    )

    return waveform


# ============================================================================
# SYNTHETIC TEXT TOKENS
# ============================================================================

def generate_token_sequence(
    seed,
):

    rng = np.random.RandomState(
        seed
    )

    length = rng.randint(
        5,
        CFG.max_phonemes + 1,
    )

    tokens = rng.randint(
        1,
        CFG.vocab_size,
        size=length,
    )

    return torch.tensor(
        tokens,
        dtype=torch.long,
    )


# ============================================================================
# SYNTHETIC DURATIONS
# ============================================================================

def generate_durations(
    token_count,
    seed,
):

    rng = np.random.RandomState(
        seed
    )

    durations = rng.randint(
        1,
        5,
        size=token_count,
    )

    durations = durations.astype(
        np.int64
    )

    while durations.sum() > CFG.mel_frames:

        candidates = np.where(
            durations > 1
        )[0]

        if len(candidates) == 0:

            break

        idx = rng.choice(
            candidates
        )

        durations[idx] -= 1

    while durations.sum() < CFG.mel_frames:

        idx = rng.randint(
            0,
            token_count,
        )

        durations[idx] += 1

    return torch.tensor(
        durations,
        dtype=torch.long,
    )


# ============================================================================
# DATASET
# ============================================================================

class VoiceCloningDataset(
    Dataset
):

    def __init__(
        self,
        num_samples,
        seed_offset=0,
    ):

        self.num_samples = (
            num_samples
        )

        self.seed_offset = (
            seed_offset
        )

    def __len__(self):

        return self.num_samples

    def __getitem__(
        self,
        index,
    ):

        seed = (
            self.seed_offset
            + index
        )

        speaker_id = (
            seed
            % CFG.num_speakers
        )

        # --------------------------------------------------------------------
        # Reference
        # --------------------------------------------------------------------

        reference_waveform = (
            generate_synthetic_speech(
                speaker_id,
                seed * 17 + 1,
            )
        )

        reference_mel = (
            waveform_to_mel(
                reference_waveform
            )
            .squeeze(0)
        )

        # --------------------------------------------------------------------
        # Second reference
        # --------------------------------------------------------------------

        reference_waveform_2 = (
            generate_synthetic_speech(
                speaker_id,
                seed * 17 + 2,
            )
        )

        reference_mel_2 = (
            waveform_to_mel(
                reference_waveform_2
            )
            .squeeze(0)
        )

        # --------------------------------------------------------------------
        # Target text
        # --------------------------------------------------------------------

        tokens = (
            generate_token_sequence(
                seed * 31 + 7
            )
        )

        durations = (
            generate_durations(
                len(tokens),
                seed * 41 + 9,
            )
        )

        # --------------------------------------------------------------------
        # Target waveform
        # --------------------------------------------------------------------

        target_waveform = (
            generate_synthetic_speech(
                speaker_id,
                seed * 17 + 3,
            )
        )

        target_mel = (
            waveform_to_mel(
                target_waveform
            )
            .squeeze(0)
        )

        return {

            "reference_mel":
                reference_mel.float(),

            "reference_mel_2":
                reference_mel_2.float(),

            "tokens":
                tokens,

            "durations":
                durations,

            "target_mel":
                target_mel.float(),

            "speaker_id":
                torch.tensor(
                    speaker_id,
                    dtype=torch.long,
                ),

            "target_waveform":
                target_waveform.float(),

            "reference_waveform":
                reference_waveform.float(),
        }


# ============================================================================
# CUSTOM COLLATE FUNCTION
# ============================================================================

def collate_batch(
    batch,
):

    batch_size = len(batch)

    max_tokens = max(
        len(item["tokens"])
        for item in batch
    )

    tokens = torch.zeros(
        batch_size,
        max_tokens,
        dtype=torch.long,
    )

    token_mask = torch.zeros(
        batch_size,
        max_tokens,
        dtype=torch.bool,
    )

    durations = torch.zeros(
        batch_size,
        max_tokens,
        dtype=torch.float32,
    )

    for i, item in enumerate(
        batch
    ):

        length = len(
            item["tokens"]
        )

        tokens[
            i,
            :length
        ] = item["tokens"]

        token_mask[
            i,
            :length
        ] = True

        durations[
            i,
            :length
        ] = item[
            "durations"
        ].float()

    return {

        "reference_mel":
            torch.stack(
                [
                    item[
                        "reference_mel"
                    ]
                    for item in batch
                ],
                dim=0,
            ),

        "reference_mel_2":
            torch.stack(
                [
                    item[
                        "reference_mel_2"
                    ]
                    for item in batch
                ],
                dim=0,
            ),

        "tokens":
            tokens,

        "token_mask":
            token_mask,

        "durations":
            durations,

        "target_mel":
            torch.stack(
                [
                    item[
                        "target_mel"
                    ]
                    for item in batch
                ],
                dim=0,
            ),

        "speaker_id":
            torch.stack(
                [
                    item[
                        "speaker_id"
                    ]
                    for item in batch
                ],
                dim=0,
            ),

        "target_waveform":
            torch.stack(
                [
                    item[
                        "target_waveform"
                    ]
                    for item in batch
                ],
                dim=0,
            ),

        "reference_waveform":
            torch.stack(
                [
                    item[
                        "reference_waveform"
                    ]
                    for item in batch
                ],
                dim=0,
            ),
    }


# ============================================================================
# DATA VALIDATION
# ============================================================================

def verify_batch_shapes(
    batch,
):

    assert (
        batch[
            "reference_mel"
        ].ndim == 3
    )

    assert (
        batch[
            "reference_mel"
        ].shape[1]
        == CFG.n_mels
    )

    assert (
        batch[
            "reference_mel"
        ].shape[2]
        == CFG.mel_frames
    )

    assert (
        batch[
            "target_mel"
        ].ndim == 3
    )

    assert (
        batch[
            "target_mel"
        ].shape[1]
        == CFG.n_mels
    )

    assert (
        batch[
            "target_mel"
        ].shape[2]
        == CFG.mel_frames
    )

    assert (
        batch[
            "target_waveform"
        ].ndim == 2
    )

    assert (
        batch[
            "target_waveform"
        ].shape[1]
        == CFG.waveform_length
    )

    assert (
        batch[
            "tokens"
        ].ndim == 2
    )

    assert (
        batch[
            "token_mask"
        ].shape
        == batch[
            "tokens"
        ].shape
    )

    assert (
        batch[
            "durations"
        ].shape
        == batch[
            "tokens"
        ].shape
    )


# ============================================================================
# SPEAKER ENCODER
# ============================================================================

class SpeakerEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(
                CFG.n_mels,
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
            CFG.speaker_embedding_dim,
        )

    def forward(
        self,
        mel,
    ):

        x = self.conv(
            mel
        )

        x = x.mean(
            dim=-1
        )

        x = self.projection(
            x
        )

        x = F.normalize(
            x,
            dim=-1,
        )

        return x


# ============================================================================
# TEXT ENCODER
# ============================================================================

class TextEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            CFG.vocab_size,
            CFG.model_dim,
            padding_idx=0,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=CFG.model_dim,
                nhead=CFG.num_heads,
                dim_feedforward=
                    CFG.model_dim * 4,
                batch_first=True,
                dropout=0.1,
                activation="gelu",
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=
                    CFG.num_transformer_layers,
            )
        )

    def forward(
        self,
        tokens,
        token_mask,
    ):

        x = self.embedding(
            tokens
        )

        padding_mask = ~token_mask

        x = self.encoder(
            x,
            src_key_padding_mask=
                padding_mask,
        )

        return x


# ============================================================================
# DURATION PREDICTOR
# ============================================================================

class DurationPredictor(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                CFG.model_dim,
                CFG.model_dim,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv1d(
                CFG.model_dim,
                CFG.model_dim,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv1d(
                CFG.model_dim,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x,
    ):

        x = x.transpose(
            1,
            2,
        )

        x = self.network(
            x
        )

        x = x.squeeze(
            1
        )

        return F.softplus(
            x
        )


# ============================================================================
# LENGTH REGULATOR
# ============================================================================

def length_regulate(
    x,
    durations,
    target_length,
):

    batch_size = x.shape[0]

    outputs = []

    for b in range(
        batch_size
    ):

        pieces = []

        for token_index in range(
            x.shape[1]
        ):

            duration = int(
                round(
                    float(
                        durations[
                            b,
                            token_index,
                        ]
                    )
                )
            )

            if duration <= 0:

                continue

            pieces.append(
                x[
                    b,
                    token_index,
                ]
                .unsqueeze(0)
                .repeat(
                    duration,
                    1,
                )
            )

        if len(pieces) == 0:

            expanded = torch.zeros(
                target_length,
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )

        else:

            expanded = torch.cat(
                pieces,
                dim=0,
            )

        if (
            expanded.shape[0]
            > target_length
        ):

            expanded = expanded[
                :target_length
            ]

        elif (
            expanded.shape[0]
            < target_length
        ):

            pad = torch.zeros(
                target_length
                - expanded.shape[0],
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )

            expanded = torch.cat(
                [
                    expanded,
                    pad,
                ],
                dim=0,
            )

        outputs.append(
            expanded
        )

    return torch.stack(
        outputs,
        dim=0,
    )


# ============================================================================
# INFERENCE DURATION NORMALIZATION
# ============================================================================

def make_inference_durations(
    predicted_durations,
    target_length,
):

    durations = (
        predicted_durations
        .detach()
        .cpu()
        .numpy()
    )

    result = []

    for row in durations:

        row = np.maximum(
            np.round(
                row
            ).astype(
                np.int64
            ),
            1,
        )

        while (
            row.sum()
            > target_length
        ):

            candidates = np.where(
                row > 1
            )[0]

            if len(candidates) == 0:

                break

            idx = candidates[
                np.argmax(
                    row[
                        candidates
                    ]
                )
            ]

            row[idx] -= 1

        while (
            row.sum()
            < target_length
        ):

            idx = int(
                np.argmax(
                    row
                )
            )

            row[idx] += 1

        result.append(
            row
        )

    return torch.tensor(
        np.stack(
            result
        ),
        dtype=torch.float32,
        device=predicted_durations.device,
    )


# ============================================================================
# ACOUSTIC DECODER
# ============================================================================

class AcousticDecoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.speaker_projection = (
            nn.Linear(
                CFG.speaker_embedding_dim,
                CFG.model_dim,
            )
        )

        self.decoder = nn.Sequential(

            nn.Conv1d(
                CFG.model_dim,
                CFG.model_dim,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                CFG.model_dim,
                CFG.model_dim,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                CFG.model_dim,
                CFG.model_dim,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )

        self.mel_projection = nn.Conv1d(
            CFG.model_dim,
            CFG.n_mels,
            kernel_size=1,
        )

    def forward(
        self,
        frame_features,
        speaker_embedding,
    ):

        speaker = (
            self.speaker_projection(
                speaker_embedding
            )
            .unsqueeze(1)
        )

        x = (
            frame_features
            + speaker
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.decoder(
            x
        )

        mel = self.mel_projection(
            x
        )

        return mel


# ============================================================================
# SPEAKER CLASSIFIER
# ============================================================================

class SpeakerClassifier(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.classifier = nn.Linear(
            CFG.speaker_embedding_dim,
            CFG.num_speakers,
        )

    def forward(
        self,
        embedding,
    ):

        return self.classifier(
            embedding
        )


# ============================================================================
# COMPLETE ACOUSTIC MODEL
# ============================================================================

class EndToEndVoiceCloningModel(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.speaker_encoder = (
            SpeakerEncoder()
        )

        self.text_encoder = (
            TextEncoder()
        )

        self.duration_predictor = (
            DurationPredictor()
        )

        self.acoustic_decoder = (
            AcousticDecoder()
        )

        self.speaker_classifier = (
            SpeakerClassifier()
        )

    def forward(
        self,
        reference_mel,
        tokens,
        token_mask,
        target_frames,
        durations=None,
    ):

        speaker_embedding = (
            self.speaker_encoder(
                reference_mel
            )
        )

        text_features = (
            self.text_encoder(
                tokens,
                token_mask,
            )
        )

        predicted_durations = (
            self.duration_predictor(
                text_features
            )
        )

        if durations is None:

            durations_for_regulator = (
                predicted_durations
            )

        else:

            durations_for_regulator = (
                durations
            )

        frame_features = (
            length_regulate(
                text_features,
                durations_for_regulator,
                target_frames,
            )
        )

        predicted_mel = (
            self.acoustic_decoder(
                frame_features,
                speaker_embedding,
            )
        )

        speaker_logits = (
            self.speaker_classifier(
                speaker_embedding
            )
        )

        return {

            "speaker_embedding":
                speaker_embedding,

            "predicted_durations":
                predicted_durations,

            "predicted_mel":
                predicted_mel,

            "speaker_logits":
                speaker_logits,
        }


# ============================================================================
# LOSSES
# ============================================================================

def masked_duration_loss(
    predicted,
    target,
    mask,
):

    return F.smooth_l1_loss(
        predicted[mask],
        target[mask],
    )


def cosine_consistency_loss(
    embedding_a,
    embedding_b,
):

    similarity = (
        F.cosine_similarity(
            embedding_a,
            embedding_b,
            dim=-1,
        )
    )

    return (
        1.0
        - similarity
    ).mean()


# ============================================================================
# TRAIN ONE EPOCH
# ============================================================================

def train_epoch(
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

    for batch in loader:

        reference_mel = (
            batch[
                "reference_mel"
            ].to(
                DEVICE,
                non_blocking=True,
            )
        )

        reference_mel_2 = (
            batch[
                "reference_mel_2"
            ].to(
                DEVICE,
                non_blocking=True,
            )
        )

        tokens = (
            batch[
                "tokens"
            ].to(
                DEVICE,
                non_blocking=True,
            )
        )

        token_mask = (
            batch[
                "token_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )
        )

        durations = (
            batch[
                "durations"
            ].to(
                DEVICE,
                non_blocking=True,
            )
        )

        target_mel = (
            batch[
                "target_mel"
            ].to(
                DEVICE,
                non_blocking=True,
            )
        )

        speaker_id = (
            batch[
                "speaker_id"
            ].to(
                DEVICE,
                non_blocking=True,
            )
        )

        output = model(
            reference_mel=
                reference_mel,

            tokens=
                tokens,

            token_mask=
                token_mask,

            target_frames=
                CFG.mel_frames,

            durations=
                durations,
        )

        mel_loss = F.l1_loss(
            output[
                "predicted_mel"
            ],
            target_mel,
        )

        duration_loss = (
            masked_duration_loss(
                output[
                    "predicted_durations"
                ],
                durations,
                token_mask,
            )
        )

        speaker_loss = (
            F.cross_entropy(
                output[
                    "speaker_logits"
                ],
                speaker_id,
            )
        )

        embedding_2 = (
            model.speaker_encoder(
                reference_mel_2
            )
        )

        consistency_loss = (
            cosine_consistency_loss(
                output[
                    "speaker_embedding"
                ],
                embedding_2,
            )
        )

        loss = (
            mel_loss

            + CFG.duration_weight
            * duration_loss

            + CFG.speaker_weight
            * speaker_loss

            + CFG.consistency_weight
            * consistency_loss
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

        total_mel += (
            mel_loss.item()
        )

        total_duration += (
            duration_loss.item()
        )

        total_speaker += (
            speaker_loss.item()
        )

        total_consistency += (
            consistency_loss.item()
        )

    n = max(
        len(loader),
        1,
    )

    return {

        "loss":
            total_loss / n,

        "mel":
            total_mel / n,

        "duration":
            total_duration / n,

        "speaker":
            total_speaker / n,

        "consistency":
            total_consistency / n,
    }


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def validate(
    model,
    loader,
):

    model.eval()

    total_mel = 0.0

    total_speaker = 0.0

    for batch in loader:

        reference_mel = (
            batch[
                "reference_mel"
            ].to(DEVICE)
        )

        tokens = (
            batch[
                "tokens"
            ].to(DEVICE)
        )

        token_mask = (
            batch[
                "token_mask"
            ].to(DEVICE)
        )

        durations = (
            batch[
                "durations"
            ].to(DEVICE)
        )

        target_mel = (
            batch[
                "target_mel"
            ].to(DEVICE)
        )

        speaker_id = (
            batch[
                "speaker_id"
            ].to(DEVICE)
        )

        output = model(
            reference_mel=
                reference_mel,

            tokens=
                tokens,

            token_mask=
                token_mask,

            target_frames=
                CFG.mel_frames,

            durations=
                durations,
        )

        mel_loss = F.l1_loss(
            output[
                "predicted_mel"
            ],
            target_mel,
        )

        speaker_accuracy = (
            (
                output[
                    "speaker_logits"
                ].argmax(
                    dim=-1
                )
                == speaker_id
            )
            .float()
            .mean()
        )

        total_mel += (
            mel_loss.item()
        )

        total_speaker += (
            speaker_accuracy.item()
        )

    n = max(
        len(loader),
        1,
    )

    return {

        "mel":
            total_mel / n,

        "speaker_accuracy":
            total_speaker / n,
    }


# ============================================================================
# SMALL NEURAL VOCODER
# ============================================================================

class SmallVocoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.input = nn.Conv1d(
            CFG.n_mels,
            128,
            kernel_size=7,
            padding=3,
        )

        self.up1 = nn.ConvTranspose1d(
            128,
            64,
            kernel_size=16,
            stride=8,
            padding=4,
        )

        self.up2 = nn.ConvTranspose1d(
            64,
            32,
            kernel_size=16,
            stride=8,
            padding=4,
        )

        self.up3 = nn.ConvTranspose1d(
            32,
            16,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.output = nn.Conv1d(
            16,
            1,
            kernel_size=7,
            padding=3,
        )

    def forward(
        self,
        mel,
    ):

        x = self.input(
            mel
        )

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.up1(
            x
        )

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.up2(
            x
        )

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.up3(
            x
        )

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.output(
            x
        )

        x = torch.tanh(
            x
        )

        expected_length = (
            mel.shape[-1]
            * CFG.hop_length
        )

        if (
            x.shape[-1]
            > expected_length
        ):

            x = x[
                ...,
                :expected_length
            ]

        elif (
            x.shape[-1]
            < expected_length
        ):

            x = F.pad(
                x,
                (
                    0,
                    expected_length
                    - x.shape[-1],
                ),
            )

        return x


# ============================================================================
# TRAIN VOCODER
# ============================================================================

def train_vocoder(
    dataset,
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        "VOCODER WARM-UP"
    )

    print(
        "=" * 72
        + "\n"
    )

    vocoder = (
        SmallVocoder()
        .to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        vocoder.parameters(),
        lr=CFG.vocoder_learning_rate,
        weight_decay=1e-4,
    )

    # ------------------------------------------------------------------------
    # CRITICAL FIX
    #
    # DO NOT use the default DataLoader collator.
    #
    # The dataset contains variable-length token tensors.
    #
    # The custom collate_batch() pads those tensors correctly.
    # ------------------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=CFG.vocoder_batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    print(
        "Vocoder DataLoader:"
    )

    print(
        f"  Samples: {len(dataset)}"
    )

    print(
        f"  Batch size: "
        f"{CFG.vocoder_batch_size}"
    )

    print(
        f"  Batches: "
        f"{len(loader)}"
    )

    print(
        "  Custom collate: ENABLED"
    )

    print()

    for epoch in range(
        1,
        CFG.vocoder_epochs + 1,
    ):

        vocoder.train()

        running_loss = 0.0

        for batch in loader:

            mel = (
                batch[
                    "target_mel"
                ].to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            target = (
                batch[
                    "target_waveform"
                ].to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            target = target.unsqueeze(
                1
            )

            generated = vocoder(
                mel
            )

            length = min(
                generated.shape[-1],
                target.shape[-1],
            )

            generated = (
                generated[
                    ...,
                    :length
                ]
            )

            target = (
                target[
                    ...,
                    :length
                ]
            )

            loss = F.l1_loss(
                generated,
                target,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                vocoder.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
            )

        average_loss = (
            running_loss
            / max(
                len(loader),
                1,
            )
        )

        print(
            f"Vocoder epoch "
            f"{epoch:02d}/"
            f"{CFG.vocoder_epochs} "
            f"| L1={average_loss:.6f}"
        )

    return vocoder


# ============================================================================
# PURE PYTHON WAV WRITER
# ============================================================================

def save_waveform(
    waveform,
    filename,
):
    """
    Save a mono float waveform as a 16-bit PCM WAV file.

    IMPORTANT:

    This intentionally does NOT use:

        torchaudio.save()
        TorchCodec
        FFmpeg

    Therefore the Lesson 24 script does not depend on FFmpeg for
    saving its generated WAV files.

    Input:

        [T]

    or:

        [1,T]
    """

    waveform = (
        waveform
        .detach()
        .cpu()
        .float()
    )

    # ------------------------------------------------------------------------
    # Remove unnecessary dimensions.
    # ------------------------------------------------------------------------

    if waveform.ndim == 3:

        waveform = waveform.squeeze()

    if waveform.ndim == 2:

        if waveform.shape[0] == 1:

            waveform = waveform[0]

        elif waveform.shape[1] == 1:

            waveform = waveform[:, 0]

        else:

            raise ValueError(
                "save_waveform currently "
                "expects mono audio. "
                f"Received shape: "
                f"{tuple(waveform.shape)}"
            )

    if waveform.ndim != 1:

        raise ValueError(
            "save_waveform expects "
            "[T], [1,T], or [B,1,T]. "
            f"Received shape: "
            f"{tuple(waveform.shape)}"
        )

    # ------------------------------------------------------------------------
    # Clamp.
    # ------------------------------------------------------------------------

    waveform = waveform.clamp(
        -1.0,
        1.0,
    )

    # ------------------------------------------------------------------------
    # Convert float [-1,1] to signed int16.
    # ------------------------------------------------------------------------

    pcm = (
        waveform.numpy()
        * 32767.0
    )

    pcm = np.clip(
        pcm,
        -32768,
        32767,
    ).astype(
        np.int16
    )

    # ------------------------------------------------------------------------
    # Write WAV using Python standard library.
    # ------------------------------------------------------------------------

    with wave.open(
        filename,
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            1
        )

        wav_file.setsampwidth(
            2
        )

        wav_file.setframerate(
            CFG.sample_rate
        )

        wav_file.writeframes(
            pcm.tobytes()
        )


# ============================================================================
# MEL PLOT
# ============================================================================

def plot_mel(
    mel,
    title,
    path,
):

    mel = (
        mel.detach()
        .cpu()
        .numpy()
    )

    plt.figure(
        figsize=(10, 4)
    )

    plt.imshow(
        mel,
        origin="lower",
        aspect="auto",
    )

    plt.title(
        title
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Mel bin"
    )

    plt.colorbar(
        label="Log magnitude"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================================
# WAVEFORM PLOT
# ============================================================================

def plot_waveforms(
    target,
    generated,
    path,
):

    target = (
        target.detach()
        .cpu()
        .numpy()
    )

    generated = (
        generated.detach()
        .cpu()
        .numpy()
    )

    length = min(
        len(target),
        len(generated),
    )

    target = target[
        :length
    ]

    generated = generated[
        :length
    ]

    time = (
        np.arange(length)
        / CFG.sample_rate
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        target,
        label="Target",
        alpha=0.75,
    )

    plt.plot(
        time,
        generated,
        label="Generated",
        alpha=0.75,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "End-to-End Voice Cloning Waveform"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================================
# TRAINING CURVES
# ============================================================================

def plot_training_curves(
    history,
    path,
):

    epochs = np.arange(
        1,
        len(history["loss"]) + 1,
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        epochs,
        history["loss"],
        label="Total",
    )

    plt.plot(
        epochs,
        history["mel"],
        label="Mel",
    )

    plt.plot(
        epochs,
        history["duration"],
        label="Duration",
    )

    plt.plot(
        epochs,
        history["speaker"],
        label="Speaker",
    )

    plt.plot(
        epochs,
        history["consistency"],
        label="Consistency",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "End-to-End Acoustic Model Training"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================================
# SPEAKER SIMILARITY
# ============================================================================

def plot_speaker_similarity(
    embedding_a,
    embedding_b,
    path,
):

    similarity = (
        F.cosine_similarity(
            embedding_a,
            embedding_b,
            dim=-1,
        )
    )

    values = (
        similarity
        .detach()
        .cpu()
        .numpy()
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.hist(
        values,
        bins=15,
    )

    plt.xlabel(
        "Cosine Similarity"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Reference Speaker Embedding Consistency"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():

    set_seed(
        CFG.seed
    )

    print(
        "=" * 72
    )

    print(
        "PHASE 1 — LESSON 24"
    )

    print(
        "END-TO-END VOICE CLONING"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Speakers: "
        f"{CFG.num_speakers}"
    )

    print(
        f"Mel bins: "
        f"{CFG.n_mels}"
    )

    print(
        f"Mel frames: "
        f"{CFG.mel_frames}"
    )

    print(
        f"Waveform samples: "
        f"{CFG.waveform_length}"
    )

    print(
        f"Waveform duration: "
        f"{CFG.waveform_length / CFG.sample_rate:.3f} sec"
    )

    print(
        "Audio writer: Python wave "
        "(FFmpeg/TorchCodec independent)"
    )

    # ========================================================================
    # DATASETS
    # ========================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CREATING DATASETS"
    )

    print(
        "=" * 72
    )

    train_dataset = (
        VoiceCloningDataset(
            CFG.train_samples,
            seed_offset=0,
        )
    )

    val_dataset = (
        VoiceCloningDataset(
            CFG.val_samples,
            seed_offset=100000,
        )
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # ========================================================================
    # DATALOADERS
    # ========================================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    # ========================================================================
    # VERIFY DATA
    # ========================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "VERIFYING DATA PIPELINE"
    )

    print(
        "=" * 72
    )

    first_batch = next(
        iter(train_loader)
    )

    verify_batch_shapes(
        first_batch
    )

    print(
        f"Reference Mel: "
        f"{tuple(first_batch['reference_mel'].shape)}"
    )

    print(
        f"Target Mel: "
        f"{tuple(first_batch['target_mel'].shape)}"
    )

    print(
        f"Tokens: "
        f"{tuple(first_batch['tokens'].shape)}"
    )

    print(
        f"Token mask: "
        f"{tuple(first_batch['token_mask'].shape)}"
    )

    print(
        f"Durations: "
        f"{tuple(first_batch['durations'].shape)}"
    )

    print(
        f"Target waveform: "
        f"{tuple(first_batch['target_waveform'].shape)}"
    )

    print(
        "\nData pipeline check: PASSED"
    )

    # ========================================================================
    # MODEL
    # ========================================================================

    model = (
        EndToEndVoiceCloningModel()
        .to(DEVICE)
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ACOUSTIC MODEL"
    )

    print(
        "=" * 72
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # ========================================================================
    # OPTIMIZER
    # ========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=1e-4,
    )

    # ========================================================================
    # HISTORY
    # ========================================================================

    history = {

        "loss": [],

        "mel": [],

        "duration": [],

        "speaker": [],

        "consistency": [],
    }

    validation_history = {

        "mel": [],

        "speaker_accuracy": [],
    }

    # ========================================================================
    # ACOUSTIC TRAINING
    # ========================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ACOUSTIC MODEL TRAINING"
    )

    print(
        "=" * 72
        + "\n"
    )

    best_val_mel = float(
        "inf"
    )

    best_checkpoint_path = (
        "lesson24_best_acoustic_model.pt"
    )

    for epoch in range(
        1,
        CFG.epochs + 1,
    ):

        metrics = train_epoch(
            model,
            train_loader,
            optimizer,
        )

        validation = validate(
            model,
            val_loader,
        )

        for key in history:

            history[key].append(
                metrics[key]
            )

        validation_history[
            "mel"
        ].append(
            validation["mel"]
        )

        validation_history[
            "speaker_accuracy"
        ].append(
            validation[
                "speaker_accuracy"
            ]
        )

        if (
            validation["mel"]
            < best_val_mel
        ):

            best_val_mel = (
                validation["mel"]
            )

            torch.save(
                {
                    "config":
                        CFG.__dict__,

                    "model":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "validation_mel":
                        best_val_mel,
                },
                best_checkpoint_path,
            )

        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{CFG.epochs} "
            f"| Loss="
            f"{metrics['loss']:.4f} "
            f"| Mel="
            f"{metrics['mel']:.4f} "
            f"| Dur="
            f"{metrics['duration']:.4f} "
            f"| Spk="
            f"{metrics['speaker']:.4f} "
            f"| Cons="
            f"{metrics['consistency']:.4f} "
            f"| ValMel="
            f"{validation['mel']:.4f} "
            f"| SpkAcc="
            f"{validation['speaker_accuracy']:.3f}"
        )

    # ========================================================================
    # VOCODER
    # ========================================================================

    vocoder = train_vocoder(
        train_dataset
    )

    # ========================================================================
    # INFERENCE
    # ========================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "END-TO-END INFERENCE"
    )

    print(
        "=" * 72
    )

    model.eval()

    vocoder.eval()

    sample = val_dataset[0]

    reference_mel = (
        sample[
            "reference_mel"
        ]
        .unsqueeze(0)
        .to(DEVICE)
    )

    reference_mel_2 = (
        sample[
            "reference_mel_2"
        ]
        .unsqueeze(0)
        .to(DEVICE)
    )

    tokens = (
        sample[
            "tokens"
        ]
        .unsqueeze(0)
        .to(DEVICE)
    )

    token_mask = torch.ones(
        1,
        tokens.shape[1],
        dtype=torch.bool,
        device=DEVICE,
    )

    # ------------------------------------------------------------------------
    # Predict durations
    # ------------------------------------------------------------------------

    with torch.no_grad():

        duration_pass = model(
            reference_mel=
                reference_mel,

            tokens=
                tokens,

            token_mask=
                token_mask,

            target_frames=
                CFG.mel_frames,

            durations=None,
        )

    predicted_durations = (
        make_inference_durations(
            duration_pass[
                "predicted_durations"
            ],
            CFG.mel_frames,
        )
    )

    # ------------------------------------------------------------------------
    # Acoustic inference
    # ------------------------------------------------------------------------

    with torch.no_grad():

        output = model(
            reference_mel=
                reference_mel,

            tokens=
                tokens,

            token_mask=
                token_mask,

            target_frames=
                CFG.mel_frames,

            durations=
                predicted_durations,
        )

        generated_mel = (
            output[
                "predicted_mel"
            ]
        )

        generated_waveform = (
            vocoder(
                generated_mel
            )
        )

        speaker_embedding_a = (
            model.speaker_encoder(
                reference_mel
            )
        )

        speaker_embedding_b = (
            model.speaker_encoder(
                reference_mel_2
            )
        )

    # ========================================================================
    # TARGET
    # ========================================================================

    target_mel = (
        sample[
            "target_mel"
        ]
    )

    target_waveform = (
        sample[
            "target_waveform"
        ]
    )

    generated_waveform = (
        generated_waveform[
            0,
            0,
        ]
    )

    # ========================================================================
    # REPORT INFERENCE SHAPES
    # ========================================================================

    print(
        "\nInference tensors:"
    )

    print(
        f"  Reference Mel: "
        f"{tuple(reference_mel.shape)}"
    )

    print(
        f"  Tokens: "
        f"{tuple(tokens.shape)}"
    )

    print(
        f"  Predicted durations: "
        f"{predicted_durations.cpu().numpy().astype(int).tolist()}"
    )

    print(
        f"  Generated Mel: "
        f"{tuple(generated_mel.shape)}"
    )

    print(
        f"  Generated waveform: "
        f"{tuple(generated_waveform.shape)}"
    )

    # ========================================================================
    # SAVE COMPLETE CHECKPOINT
    # ========================================================================

    checkpoint = {

        "config":
            CFG.__dict__,

        "model":
            model.state_dict(),

        "vocoder":
            vocoder.state_dict(),

        "history":
            history,

        "validation_history":
            validation_history,

        "best_validation_mel":
            best_val_mel,
    }

    torch.save(
        checkpoint,
        "lesson24_end_to_end_voice_cloning.pt",
    )

    # ========================================================================
    # SAVE WAV FILES
    # ========================================================================

    print(
        "\nSaving WAV files..."
    )

    save_waveform(
        sample[
            "reference_waveform"
        ],
        "lesson24_reference.wav",
    )

    print(
        "  Saved lesson24_reference.wav"
    )

    save_waveform(
        target_waveform,
        "lesson24_target.wav",
    )

    print(
        "  Saved lesson24_target.wav"
    )

    save_waveform(
        generated_waveform,
        "lesson24_generated.wav",
    )

    print(
        "  Saved lesson24_generated.wav"
    )

    # ========================================================================
    # MEL PLOTS
    # ========================================================================

    plot_mel(
        reference_mel[0],
        "Reference Speaker Mel",
        "lesson24_reference_mel.png",
    )

    plot_mel(
        target_mel,
        "Target Mel",
        "lesson24_target_mel.png",
    )

    plot_mel(
        generated_mel[0],
        "Generated Mel",
        "lesson24_generated_mel.png",
    )

    # ========================================================================
    # WAVEFORM PLOT
    # ========================================================================

    plot_waveforms(
        target_waveform,
        generated_waveform,
        "lesson24_waveforms.png",
    )

    # ========================================================================
    # TRAINING CURVES
    # ========================================================================

    plot_training_curves(
        history,
        "lesson24_training_curves.png",
    )

    # ========================================================================
    # SPEAKER SIMILARITY
    # ========================================================================

    plot_speaker_similarity(
        speaker_embedding_a,
        speaker_embedding_b,
        "lesson24_speaker_similarity.png",
    )

    # ========================================================================
    # METRICS
    # ========================================================================

    mel_error = F.l1_loss(
        generated_mel[0],
        target_mel.to(DEVICE),
    ).item()

    speaker_similarity = (
        F.cosine_similarity(
            speaker_embedding_a,
            speaker_embedding_b,
            dim=-1,
        )
        .item()
    )

    waveform_length = min(
        len(target_waveform),
        len(generated_waveform),
    )

    waveform_error = F.l1_loss(
        generated_waveform[
            :waveform_length
        ],
        target_waveform[
            :waveform_length
        ].to(DEVICE),
    ).item()

    # ========================================================================
    # FINAL REPORT
    # ========================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "LESSON 24 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Best validation Mel L1: "
        f"{best_val_mel:.6f}"
    )

    print(
        f"Final Mel reconstruction L1: "
        f"{mel_error:.6f}"
    )

    print(
        f"Waveform L1: "
        f"{waveform_error:.6f}"
    )

    print(
        f"Same-speaker embedding cosine similarity: "
        f"{speaker_similarity:.4f}"
    )

    print(
        "\nGenerated files:"
    )

    output_files = [

        "lesson24_end_to_end_voice_cloning.pt",

        "lesson24_best_acoustic_model.pt",

        "lesson24_reference.wav",

        "lesson24_target.wav",

        "lesson24_generated.wav",

        "lesson24_reference_mel.png",

        "lesson24_target_mel.png",

        "lesson24_generated_mel.png",

        "lesson24_waveforms.png",

        "lesson24_training_curves.png",

        "lesson24_speaker_similarity.png",
    ]

    for filename in output_files:

        print(
            f"  {filename}"
        )

    # ========================================================================
    # ARCHITECTURE
    # ========================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "COMPLETE VOICE-CLONING PIPELINE"
    )

    print(
        "=" * 72
    )

    print(
        """
REFERENCE AUDIO
      |
      v
+----------------------+
|   Speaker Encoder    |
+----------------------+
      |
      v
Speaker Embedding
      |
      |                         TARGET TEXT
      |                            |
      |                            v
      |                      Token Sequence
      |                            |
      |                            v
      |                    +---------------+
      |                    | Text Encoder  |
      |                    +---------------+
      |                            |
      |                            v
      |                      Text Features
      |                            |
      |                            v
      |                    Duration Predictor
      |                            |
      |                            v
      |                     Length Regulation
      |                            |
      +-------------+--------------+
                    |
                    v
             Acoustic Decoder
                    |
                    v
              Mel Spectrogram
                    |
                    v
              Neural Vocoder
                    |
                    v
                 WAVEFORM
"""
    )

    # ========================================================================
    # FIX REPORT
    # ========================================================================

    print(
        "=" * 72
    )

    print(
        "LESSON 24 FIXES"
    )

    print(
        "=" * 72
    )

    print(
        """
FIX 1 — Variable-length token DataLoader

    Problem:

        token_1 = [9]
        token_2 = [5]

    Default PyTorch collator:

        torch.stack(...)

    Result:

        RuntimeError

    Solution:

        collate_fn=collate_batch


FIX 2 — TorchCodec / FFmpeg WAV saving

    Problem:

        torchaudio.save()
             |
             v
        TorchCodec
             |
             v
        FFmpeg shared libraries
             |
             v
        libavutil.so.* missing

    Solution:

        Python standard-library wave module

    The lesson now saves:

        16-bit PCM
        mono
        16 kHz
        WAV

    without:

        FFmpeg
        TorchCodec
        torchaudio.save()


FIX 3 — Inference durations

    Training:

        ground-truth durations

    Inference:

        predicted durations
              |
              v
        integer normalization
              |
              v
        length regulation


FIX 4 — Shape validation

    The script validates:

        Mel dimensions
        token dimensions
        token masks
        durations
        waveform length


The neural model itself was not the cause of the latest crash.
The crash happened only when saving the generated WAV file.
"""
    )

    # ========================================================================
    # NEXT LESSON
    # ========================================================================

    print(
        "=" * 72
    )

    print(
        "NEXT LESSON"
    )

    print(
        "=" * 72
    )

    print(
        """
LESSON 25 — FEW-SHOT VOICE CLONING

We will improve the reference side.

Instead of:

        Reference
            |
            v
        Speaker Encoder
            |
            v
        Speaker Embedding

we will use:

        Reference 1 ----+
                       |
        Reference 2 ----+----> Reference Aggregator
                       |
        Reference 3 ----+
                       |
                       v
                Speaker Embedding
                       |
                       v
                 Acoustic Model
                       |
                       v
                      Mel
                       |
                       v
                    Vocoder
                       |
                       v
                    Speech

Topics:

    1. Multiple reference utterances
    2. Mean embedding aggregation
    3. Attention-based aggregation
    4. Reference dropout
    5. Noisy reference audio
    6. Robust speaker identity extraction
    7. Few-shot evaluation
    8. Zero-shot speaker transfer
"""
    )

    print(
        "=" * 72
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()