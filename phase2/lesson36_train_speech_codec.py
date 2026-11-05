
"""
======================================================================
PHASE 2 — LESSON 36
STABLE NEURAL SPEECH CODEC WITH EMA VECTOR QUANTIZATION
======================================================================

PURPOSE
-------
Build a small speech codec that converts:

    waveform
        ↓
    encoder
        ↓
    continuous latent representation
        ↓
    vector quantization
        ↓
    discrete speech tokens
        ↓
    decoder
        ↓
    reconstructed waveform


WHY WE ARE FIXING THE PREVIOUS VERSION
--------------------------------------
The previous training run showed:

    Step 1:
        Codes 256/256
        Perplexity 177.65

    Step 100:
        Codes 1/256
        Perplexity 1.00

This means the VQ codebook collapsed.

The run then produced:

    Non-finite values detected in input waveform.

This lesson therefore introduces a more robust VQ implementation.


KEY CHANGE
----------
Instead of directly optimizing codebook vectors using ordinary
gradient descent, we use an EMA (Exponential Moving Average)
vector quantizer.

The codebook update becomes conceptually:

    cluster_mean
         ↓
    EMA update
         ↓
    codebook vector


This is similar to the stabilization idea used by VQ-VAE-style
systems.


WHAT YOU LEARN
--------------

STEP 1 — Neural encoder
    Compress waveform into a shorter latent sequence.

STEP 2 — Bounded latent representation
    tanh keeps latent values in a controlled numerical range.

STEP 3 — Vector quantization
    Each latent vector selects the nearest codebook entry.

STEP 4 — EMA codebook
    Codebook vectors are updated from the statistics of the
    encoder latents instead of being directly pushed around
    by ordinary gradient descent.

STEP 5 — Commitment loss
    Encourage the encoder to stay close to the selected code.

STEP 6 — Straight-through estimator
    Allow gradients to pass through the discrete quantization.

STEP 7 — Dead-code replacement
    Rarely used codebook entries are periodically replaced with
    real latent vectors.

STEP 8 — Numerical safety
    NaN/Inf values are detected and sanitized before they can
    destroy the training run.

STEP 9 — Discrete speech representation
    The final output is a sequence such as:

        [12, 12, 47, 47, 83, 91, 91, 104, ...]

    These IDs will become the "speech language" modeled by the
    autoregressive model in Lesson 37.


ARCHITECTURE
------------

Audio:

    8192 samples
       ↓
    Conv stride 2
       ↓
    Conv stride 2
       ↓
    Conv stride 2
       ↓
    Conv stride 2
       ↓
    512 latent frames
       ↓
    64-dimensional latent vectors
       ↓
    256-entry codebook
       ↓
    discrete token sequence
       ↓
    decoder
       ↓
    reconstructed audio


IMPORTANT
---------
This is an EDUCATIONAL speech codec.

It is not intended to reproduce the exact internal architecture
of Chatterbox or any other production TTS system.

The purpose is to understand the principles needed for a
Chatterbox-like system.


OUTPUT
------
lesson36_outputs/
    lesson36_speech_codec.pt
    lesson36_codec_training.png
    lesson36_codec_reconstruction.png
    lesson36_codec_detail.png
    lesson36_codec_spectrogram.png
    lesson36_codec_tokens.png
    lesson36_codebook_usage.png
    lesson36_original.wav
    lesson36_reconstructed.wav


NEXT LESSON
-----------
Lesson 37 — Autoregressive Speech Model

We will learn:

    P(s_t | s_1, ..., s_{t-1})

where:

    s_t = discrete speech token

That will be the first actual generative model operating on
speech tokens.


======================================================================
"""

from __future__ import annotations

import argparse
import math
import random
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# CONFIGURATION
# ======================================================================

SAMPLE_RATE = 16000

AUDIO_LENGTH = 8192

LATENT_DIM = 64

CODEBOOK_SIZE = 256

DOWNSAMPLE_FACTOR = 16

BATCH_SIZE = 16

DEFAULT_STEPS = 1800

LEARNING_RATE = 5e-5

COMMITMENT_COST = 0.25

EMA_DECAY = 0.99

DEAD_CODE_THRESHOLD = 1.0

DEAD_CODE_REPLACEMENT_INTERVAL = 100

SEED = 42


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed: int = SEED):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ======================================================================
# NUMERICAL SAFETY
# ======================================================================

def check_finite(
    tensor: torch.Tensor,
    name: str,
):

    if not torch.isfinite(tensor).all():

        raise FloatingPointError(
            f"Non-finite values detected in {name}."
        )


def sanitize_waveform(
    waveform: torch.Tensor,
):

    """
    Make absolutely sure that the waveform entering the model
    contains only finite values.

    This is a defensive measure.

    It should normally make no change to the synthetic dataset.
    """

    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    waveform = waveform.clamp(
        -1.0,
        1.0,
    )

    maximum = waveform.abs().amax(
        dim=-1,
        keepdim=True,
    )

    waveform = waveform / maximum.clamp_min(
        1e-6
    )

    return waveform


# ======================================================================
# SYNTHETIC SPEECH-LIKE AUDIO
# ======================================================================

def generate_speech_like_waveform(
    length: int = AUDIO_LENGTH,
    sample_rate: int = SAMPLE_RATE,
):

    """
    Generate a finite speech-like waveform.

    This is NOT real speech.

    It provides structured acoustic signals so that the codec
    can learn the mechanics of compression and discrete
    representation without requiring an external dataset.
    """

    t = (
        torch.arange(
            length,
            dtype=torch.float32,
        )
        / float(sample_rate)
    )

    waveform = torch.zeros(
        length,
        dtype=torch.float32,
    )

    duration_total = float(
        length
    ) / float(sample_rate)

    num_segments = random.randint(
        3,
        7,
    )

    for _ in range(num_segments):

        start = random.uniform(
            0.0,
            max(
                0.001,
                duration_total - 0.08,
            ),
        )

        duration = random.uniform(
            0.05,
            0.20,
        )

        end = min(
            start + duration,
            duration_total,
        )

        frequency = random.uniform(
            90.0,
            350.0,
        )

        harmonic_strength = random.uniform(
            0.2,
            0.6,
        )

        phase = random.uniform(
            0.0,
            2.0 * math.pi,
        )

        local_t = t - start

        active = (
            (t >= start)
            & (t <= end)
        )

        if not active.any():

            continue

        duration_safe = max(
            end - start,
            1e-4,
        )

        x = (
            local_t[active]
            / duration_safe
        ).clamp(
            0.0,
            1.0,
        )

        envelope = (
            torch.sin(
                math.pi * x
            )
            .clamp_min(0.0)
            ** 1.5
        )

        fundamental = torch.sin(
            2.0
            * math.pi
            * frequency
            * local_t[active]
            + phase
        )

        harmonic2 = torch.sin(
            2.0
            * math.pi
            * frequency
            * 2.0
            * local_t[active]
            + phase
        )

        harmonic3 = torch.sin(
            2.0
            * math.pi
            * frequency
            * 3.0
            * local_t[active]
            + phase
        )

        component = (
            fundamental
            + harmonic_strength
            * harmonic2
            + 0.2
            * harmonic_strength
            * harmonic3
        )

        waveform[active] += (
            envelope
            * component
        )

    # Very small noise floor.

    noise = (
        0.01
        * torch.randn(
            length,
            dtype=torch.float32,
        )
    )

    waveform += noise

    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = waveform.abs().max()

    waveform = waveform / maximum.clamp_min(
        1e-6
    )

    waveform = waveform.clamp(
        -1.0,
        1.0,
    )

    return waveform


class SyntheticSpeechDataset(
    Dataset
):

    def __init__(
        self,
        length=10000,
        audio_length=AUDIO_LENGTH,
    ):

        self.length = length

        self.audio_length = (
            audio_length
        )

    def __len__(self):

        return self.length

    def __getitem__(
        self,
        index,
    ):

        waveform = (
            generate_speech_like_waveform(
                length=self.audio_length
            )
        )

        waveform = torch.nan_to_num(
            waveform,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        return waveform


# ======================================================================
# WAV READER
# ======================================================================

def read_wav_file(
    path: str,
    target_sample_rate: int = SAMPLE_RATE,
):

    with wave.open(
        path,
        "rb",
    ) as wav:

        channels = wav.getnchannels()

        sample_width = (
            wav.getsampwidth()
        )

        sample_rate = (
            wav.getframerate()
        )

        num_frames = (
            wav.getnframes()
        )

        raw = wav.readframes(
            num_frames
        )

    if sample_width == 1:

        data = torch.tensor(
            list(raw),
            dtype=torch.float32,
        )

        data = (
            data - 128.0
        ) / 128.0

    elif sample_width == 2:

        data = torch.frombuffer(
            raw,
            dtype=torch.int16,
        ).clone().float()

        data /= 32768.0

    elif sample_width == 4:

        data = torch.frombuffer(
            raw,
            dtype=torch.int32,
        ).clone().float()

        data /= 2147483648.0

    else:

        raise RuntimeError(
            "Unsupported WAV sample width."
        )

    if channels > 1:

        data = data.view(
            -1,
            channels,
        )

        data = data.mean(
            dim=1
        )

    data = torch.nan_to_num(
        data,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    if sample_rate != target_sample_rate:

        data = data.view(
            1,
            1,
            -1,
        )

        new_length = int(
            data.shape[-1]
            * target_sample_rate
            / sample_rate
        )

        data = F.interpolate(
            data,
            size=new_length,
            mode="linear",
            align_corners=False,
        )

        data = data.reshape(
            -1
        )

    return sanitize_waveform(
        data
    )


class WAVDataset(
    Dataset
):

    def __init__(
        self,
        directory,
        audio_length=AUDIO_LENGTH,
        sample_rate=SAMPLE_RATE,
    ):

        self.files = sorted(
            list(
                Path(directory).glob(
                    "*.wav"
                )
            )
            + list(
                Path(directory).glob(
                    "*.WAV"
                )
            )
        )

        if not self.files:

            raise RuntimeError(
                f"No WAV files found in "
                f"{directory}"
            )

        self.audio_length = (
            audio_length
        )

        self.sample_rate = (
            sample_rate
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(
        self,
        index,
    ):

        waveform = read_wav_file(
            str(
                self.files[index]
            ),
            self.sample_rate,
        )

        if waveform.numel() < self.audio_length:

            waveform = F.pad(
                waveform,
                (
                    0,
                    self.audio_length
                    - waveform.numel(),
                ),
            )

        else:

            maximum_start = (
                waveform.numel()
                - self.audio_length
            )

            if maximum_start > 0:

                start = random.randint(
                    0,
                    maximum_start,
                )

            else:

                start = 0

            waveform = waveform[
                start:
                start + self.audio_length
            ]

        waveform = sanitize_waveform(
            waveform
        )

        return waveform.float()


# ======================================================================
# ENCODER
# ======================================================================

class SpeechEncoder(
    nn.Module
):

    def __init__(
        self,
        latent_dim=LATENT_DIM,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.GELU(),

            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.GELU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.GELU(),

            nn.Conv1d(
                128,
                latent_dim,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
        )

    def forward(
        self,
        waveform,
    ):

        z = self.features(
            waveform
        )

        # Bound latent space.

        z = torch.tanh(
            z
        )

        return z


# ======================================================================
# DECODER
# ======================================================================

class SpeechDecoder(
    nn.Module
):

    def __init__(
        self,
        latent_dim=LATENT_DIM,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.ConvTranspose1d(
                latent_dim,
                128,
                kernel_size=8,
                stride=2,
                padding=3,
            ),

            nn.GELU(),

            nn.ConvTranspose1d(
                128,
                64,
                kernel_size=8,
                stride=2,
                padding=3,
            ),

            nn.GELU(),

            nn.ConvTranspose1d(
                64,
                32,
                kernel_size=8,
                stride=2,
                padding=3,
            ),

            nn.GELU(),

            nn.ConvTranspose1d(
                32,
                1,
                kernel_size=8,
                stride=2,
                padding=3,
            ),

            nn.Tanh(),
        )

    def forward(
        self,
        latent,
    ):

        return self.net(
            latent
        )


# ======================================================================
# EMA VECTOR QUANTIZER
# ======================================================================

class EMAVectorQuantizer(
    nn.Module
):

    """
    EMA VQ.

    The codebook is NOT optimized directly with Adam.

    Instead:

        encoder latents
              ↓
        nearest code
              ↓
        cluster statistics
              ↓
        exponential moving average
              ↓
        codebook update


    This prevents the codebook from being violently moved by
    the reconstruction gradient.
    """

    def __init__(
        self,
        num_embeddings=CODEBOOK_SIZE,
        embedding_dim=LATENT_DIM,
        commitment_cost=COMMITMENT_COST,
        decay=EMA_DECAY,
    ):

        super().__init__()

        self.num_embeddings = (
            num_embeddings
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.commitment_cost = (
            commitment_cost
        )

        self.decay = decay

        self.codebook = nn.Embedding(
            num_embeddings,
            embedding_dim,
        )

        nn.init.uniform_(
            self.codebook.weight,
            -0.5,
            0.5,
        )

        # EMA cluster counts.

        self.register_buffer(
            "ema_cluster_size",
            torch.ones(
                num_embeddings
            ),
        )

        # EMA latent sums.

        self.register_buffer(
            "ema_weight",
            self.codebook.weight.detach().clone(),
        )

        self.initialized = False

    @torch.no_grad()
    def initialize_from_latents(
        self,
        z,
    ):

        flat_z = (
            z.permute(
                0,
                2,
                1,
            )
            .contiguous()
            .view(
                -1,
                self.embedding_dim,
            )
        )

        flat_z = flat_z[
            torch.isfinite(
                flat_z
            ).all(
                dim=1
            )
        ]

        if flat_z.numel() == 0:

            raise FloatingPointError(
                "No finite latent vectors "
                "available for codebook "
                "initialization."
            )

        if (
            flat_z.shape[0]
            < self.num_embeddings
        ):

            repeat_count = math.ceil(
                self.num_embeddings
                / flat_z.shape[0]
            )

            flat_z = flat_z.repeat(
                repeat_count,
                1,
            )

        indices = torch.randperm(
            flat_z.shape[0],
            device=flat_z.device,
        )[
            :self.num_embeddings
        ]

        selected = flat_z[
            indices
        ].clone()

        self.codebook.weight.data.copy_(
            selected
        )

        self.ema_weight.data.copy_(
            selected
        )

        self.ema_cluster_size.fill_(
            1.0
        )

        self.initialized = True

    def forward(
        self,
        z,
    ):

        batch, dim, time = z.shape

        flat_z = (
            z.permute(
                0,
                2,
                1,
            )
            .contiguous()
            .view(
                -1,
                dim,
            )
        )

        check_finite(
            flat_z,
            "encoder latent",
        )

        codebook = (
            self.codebook.weight
        )

        check_finite(
            codebook,
            "codebook",
        )

        distances = (
            flat_z.pow(2).sum(
                dim=1,
                keepdim=True,
            )
            + codebook.pow(2).sum(
                dim=1
            )
            - 2.0
            * flat_z
            @ codebook.t()
        )

        distances = torch.clamp(
            distances,
            min=0.0,
        )

        check_finite(
            distances,
            "VQ distances",
        )

        token_ids = distances.argmin(
            dim=1
        )

        quantized = self.codebook(
            token_ids
        )

        check_finite(
            quantized,
            "quantized vectors",
        )

        # --------------------------------------------------------------
        # EMA UPDATE
        # --------------------------------------------------------------

        if self.training:

            with torch.no_grad():

                one_hot = F.one_hot(
                    token_ids,
                    num_classes=self.num_embeddings,
                ).float()

                cluster_size = (
                    one_hot.sum(
                        dim=0
                    )
                )

                dw = (
                    one_hot.t()
                    @ flat_z
                )

                self.ema_cluster_size.mul_(
                    self.decay
                ).add_(
                    cluster_size,
                    alpha=1.0 - self.decay,
                )

                self.ema_weight.mul_(
                    self.decay
                ).add_(
                    dw,
                    alpha=1.0 - self.decay,
                )

                # Laplace smoothing.

                n = (
                    self.ema_cluster_size.sum()
                )

                smoothed_counts = (
                    (
                        self.ema_cluster_size
                        + 1e-5
                    )
                    / (
                        n
                        + self.num_embeddings
                        * 1e-5
                    )
                    * n
                )

                new_codebook = (
                    self.ema_weight
                    / smoothed_counts.unsqueeze(
                        1
                    ).clamp_min(
                        1e-6
                    )
                )

                new_codebook = torch.nan_to_num(
                    new_codebook,
                    nan=0.0,
                    posinf=0.5,
                    neginf=-0.5,
                )

                new_codebook = new_codebook.clamp(
                    -1.0,
                    1.0,
                )

                self.codebook.weight.data.copy_(
                    new_codebook
                )

        # --------------------------------------------------------------
        # COMMITMENT LOSS
        # --------------------------------------------------------------

        commitment_loss = F.mse_loss(
            flat_z,
            quantized.detach(),
        )

        vq_loss = (
            self.commitment_cost
            * commitment_loss
        )

        check_finite(
            vq_loss,
            "VQ loss",
        )

        quantized = (
            quantized
            .view(
                batch,
                time,
                dim,
            )
            .permute(
                0,
                2,
                1,
            )
            .contiguous()
        )

        # Straight-through estimator.

        quantized_st = (
            z
            + (
                quantized
                - z
            ).detach()
        )

        token_ids = token_ids.view(
            batch,
            time,
        )

        return (
            quantized_st,
            token_ids,
            vq_loss,
        )

    @torch.no_grad()
    def replace_dead_codes(
        self,
        z,
    ):

        """
        Replace extremely rarely used codes with real encoder
        vectors.

        This prevents permanently dead codebook entries.
        """

        flat_z = (
            z.permute(
                0,
                2,
                1,
            )
            .contiguous()
            .view(
                -1,
                self.embedding_dim,
            )
        )

        finite_mask = torch.isfinite(
            flat_z
        ).all(
            dim=1
        )

        flat_z = flat_z[
            finite_mask
        ]

        if flat_z.numel() == 0:

            return 0

        dead = (
            self.ema_cluster_size
            < DEAD_CODE_THRESHOLD
        )

        dead_indices = (
            dead.nonzero(
                as_tuple=False
            )
            .flatten()
        )

        if dead_indices.numel() == 0:

            return 0

        if (
            flat_z.shape[0]
            < dead_indices.numel()
        ):

            repeat_count = math.ceil(
                dead_indices.numel()
                / flat_z.shape[0]
            )

            flat_z = flat_z.repeat(
                repeat_count,
                1,
            )

        random_indices = torch.randperm(
            flat_z.shape[0],
            device=flat_z.device,
        )[
            :dead_indices.numel()
        ]

        replacement = flat_z[
            random_indices
        ].clone()

        replacement = replacement.clamp(
            -1.0,
            1.0,
        )

        self.codebook.weight.data[
            dead_indices
        ] = replacement

        self.ema_weight.data[
            dead_indices
        ] = replacement

        self.ema_cluster_size[
            dead_indices
        ] = 1.0

        return int(
            dead_indices.numel()
        )


# ======================================================================
# COMPLETE SPEECH CODEC
# ======================================================================

class SpeechCodec(
    nn.Module
):

    def __init__(
        self,
        latent_dim=LATENT_DIM,
        codebook_size=CODEBOOK_SIZE,
    ):

        super().__init__()

        self.encoder = SpeechEncoder(
            latent_dim
        )

        self.quantizer = EMAVectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
        )

        self.decoder = SpeechDecoder(
            latent_dim
        )

    def encode(
        self,
        waveform,
    ):

        if waveform.ndim == 2:

            waveform = waveform.unsqueeze(
                1
            )

        z = self.encoder(
            waveform
        )

        (
            quantized,
            tokens,
            vq_loss,
        ) = self.quantizer(
            z
        )

        return (
            tokens,
            quantized,
            vq_loss,
        )

    def decode_tokens(
        self,
        tokens,
    ):

        embeddings = (
            self.quantizer.codebook(
                tokens
            )
        )

        embeddings = (
            embeddings.permute(
                0,
                2,
                1,
            )
            .contiguous()
        )

        return self.decoder(
            embeddings
        )

    def forward(
        self,
        waveform,
    ):

        if waveform.ndim == 2:

            waveform = waveform.unsqueeze(
                1
            )

        z = self.encoder(
            waveform
        )

        (
            quantized,
            tokens,
            vq_loss,
        ) = self.quantizer(
            z
        )

        reconstructed = (
            self.decoder(
                quantized
            )
        )

        return (
            reconstructed,
            tokens,
            vq_loss,
            z,
        )


# ======================================================================
# DATASET
# ======================================================================

def create_dataset(
    data_dir,
):

    if data_dir is not None:

        print(
            f"Using WAV dataset: "
            f"{data_dir}"
        )

        return WAVDataset(
            directory=data_dir,
            audio_length=AUDIO_LENGTH,
            sample_rate=SAMPLE_RATE,
        )

    print(
        "Using synthetic speech-like dataset."
    )

    return SyntheticSpeechDataset(
        length=10000,
        audio_length=AUDIO_LENGTH,
    )


# ======================================================================
# CODEBOOK STATISTICS
# ======================================================================

def codebook_statistics(
    tokens,
    codebook_size,
):

    flat = (
        tokens
        .detach()
        .reshape(-1)
    )

    counts = torch.bincount(
        flat,
        minlength=codebook_size,
    ).float()

    total = counts.sum().clamp_min(
        1.0
    )

    probabilities = (
        counts
        / total
    )

    nonzero = (
        probabilities > 0
    )

    entropy = -(
        probabilities[nonzero]
        * probabilities[nonzero].log()
    ).sum()

    perplexity = torch.exp(
        entropy
    )

    used = int(
        (
            counts > 0
        )
        .sum()
        .item()
    )

    return (
        used,
        float(
            perplexity.item()
        ),
        counts.cpu(),
    )


# ======================================================================
# APPROXIMATE SNR
# ======================================================================

def approximate_snr(
    original,
    reconstructed,
):

    original = torch.nan_to_num(
        original
    )

    reconstructed = torch.nan_to_num(
        reconstructed
    )

    signal_power = (
        original.pow(2)
        .mean()
    )

    noise_power = (
        (
            original
            - reconstructed
        )
        .pow(2)
        .mean()
        .clamp_min(1e-10)
    )

    ratio = (
        signal_power
        / noise_power
    ).clamp_min(
        1e-10
    )

    return float(
        (
            10.0
            * torch.log10(
                ratio
            )
        ).item()
    )


# ======================================================================
# TRAINING
# ======================================================================

def train_codec(
    model,
    dataloader,
    optimizer,
    device,
    steps,
):

    model.train()

    iterator = iter(
        dataloader
    )

    reconstruction_history = []

    vq_history = []

    total_history = []

    codebook_initialized = False

    for step in range(
        1,
        steps + 1,
    ):

        try:

            waveform = next(
                iterator
            )

        except StopIteration:

            iterator = iter(
                dataloader
            )

            waveform = next(
                iterator
            )

        # --------------------------------------------------------------
        # SANITIZE BEFORE GPU
        # --------------------------------------------------------------

        waveform = torch.nan_to_num(
            waveform,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        waveform = sanitize_waveform(
            waveform
        )

        check_finite(
            waveform,
            "CPU input waveform",
        )

        waveform = waveform.to(
            device,
            non_blocking=True,
        )

        check_finite(
            waveform,
            "GPU input waveform",
        )

        if waveform.ndim == 2:

            waveform = waveform.unsqueeze(
                1
            )

        # --------------------------------------------------------------
        # INITIAL CODEBOOK
        # --------------------------------------------------------------

        if not codebook_initialized:

            with torch.no_grad():

                initial_z = (
                    model.encoder(
                        waveform
                    )
                )

                check_finite(
                    initial_z,
                    "initial latent",
                )

                model.quantizer.initialize_from_latents(
                    initial_z
                )

            codebook_initialized = True

            print(
                "Initialized codebook "
                "from encoder latents."
            )

        # --------------------------------------------------------------
        # FORWARD
        # --------------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            reconstructed,
            tokens,
            vq_loss,
            latent,
        ) = model(
            waveform
        )

        check_finite(
            latent,
            "latent",
        )

        check_finite(
            reconstructed,
            "reconstructed waveform",
        )

        min_length = min(
            waveform.shape[-1],
            reconstructed.shape[-1],
        )

        target = waveform[
            ...,
            :min_length,
        ]

        output = reconstructed[
            ...,
            :min_length,
        ]

        reconstruction_loss = F.l1_loss(
            output,
            target,
        )

        total_loss = (
            reconstruction_loss
            + vq_loss
        )

        check_finite(
            reconstruction_loss,
            "reconstruction loss",
        )

        check_finite(
            total_loss,
            "total loss",
        )

        # --------------------------------------------------------------
        # BACKPROPAGATION
        # --------------------------------------------------------------

        total_loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.5,
            )
        )

        if not torch.isfinite(
            grad_norm
        ):

            raise FloatingPointError(
                "Gradient norm became "
                "NaN or Inf."
            )

        optimizer.step()

        # --------------------------------------------------------------
        # DEAD CODE REPLACEMENT
        # --------------------------------------------------------------

        replaced = 0

        if (
            step
            % DEAD_CODE_REPLACEMENT_INTERVAL
            == 0
        ):

            replaced = (
                model.quantizer.replace_dead_codes(
                    latent.detach()
                )
            )

        # --------------------------------------------------------------
        # HISTORY
        # --------------------------------------------------------------

        reconstruction_history.append(
            reconstruction_loss.item()
        )

        vq_history.append(
            vq_loss.item()
        )

        total_history.append(
            total_loss.item()
        )

        # --------------------------------------------------------------
        # PROGRESS
        # --------------------------------------------------------------

        if (
            step == 1
            or step % 100 == 0
            or step == steps
        ):

            (
                used,
                perplexity,
                _,
            ) = codebook_statistics(
                tokens,
                CODEBOOK_SIZE,
            )

            print(
                f"Step {step:5d}/{steps} | "
                f"Recon "
                f"{reconstruction_loss.item():.5f} | "
                f"VQ "
                f"{vq_loss.item():.5f} | "
                f"Total "
                f"{total_loss.item():.5f} | "
                f"Codes "
                f"{used:3d}/{CODEBOOK_SIZE} | "
                f"Perplexity "
                f"{perplexity:.2f}"
                + (
                    f" | Dead replaced "
                    f"{replaced}"
                    if replaced > 0
                    else ""
                )
            )

    return {
        "reconstruction":
            reconstruction_history,
        "vq":
            vq_history,
        "total":
            total_history,
    }


# ======================================================================
# PLOTS
# ======================================================================

def plot_training(
    history,
    path,
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["reconstruction"],
        label="Reconstruction",
    )

    plt.plot(
        history["vq"],
        label="VQ",
    )

    plt.plot(
        history["total"],
        label="Total",
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Lesson 36 — Speech Codec Training"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def plot_reconstruction(
    original,
    reconstructed,
    path,
):

    original = (
        original.detach()
        .cpu()
        .flatten()
    )

    reconstructed = (
        reconstructed.detach()
        .cpu()
        .flatten()
    )

    length = min(
        original.numel(),
        reconstructed.numel(),
    )

    original = original[:length]

    reconstructed = (
        reconstructed[:length]
    )

    time = (
        torch.arange(length)
        / SAMPLE_RATE
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        time.numpy(),
        original.numpy(),
        label="Original",
        alpha=0.8,
    )

    plt.plot(
        time.numpy(),
        reconstructed.numpy(),
        label="Reconstructed",
        alpha=0.7,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Speech Codec — Reconstruction"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def plot_detail(
    original,
    reconstructed,
    path,
):

    original = (
        original.detach()
        .cpu()
        .flatten()
    )

    reconstructed = (
        reconstructed.detach()
        .cpu()
        .flatten()
    )

    length = min(
        original.numel(),
        reconstructed.numel(),
        2000,
    )

    original = original[:length]

    reconstructed = (
        reconstructed[:length]
    )

    time = (
        torch.arange(length)
        / SAMPLE_RATE
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time.numpy(),
        original.numpy(),
        label="Original",
    )

    plt.plot(
        time.numpy(),
        reconstructed.numpy(),
        label="Reconstructed",
        alpha=0.7,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Waveform Detail"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def compute_spectrogram(
    waveform,
):

    waveform = (
        waveform.detach()
        .cpu()
        .flatten()
    )

    waveform = torch.nan_to_num(
        waveform
    )

    window = torch.hann_window(
        512
    )

    spec = torch.stft(
        waveform,
        n_fft=512,
        hop_length=128,
        win_length=512,
        window=window,
        return_complex=True,
    )

    return torch.log1p(
        spec.abs()
    )


def plot_spectrograms(
    original,
    reconstructed,
    path,
):

    original_spec = (
        compute_spectrogram(
            original
        )
    )

    reconstructed_spec = (
        compute_spectrogram(
            reconstructed
        )
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
    )

    axes[0].imshow(
        original_spec.numpy(),
        origin="lower",
        aspect="auto",
    )

    axes[0].set_title(
        "Original Spectrogram"
    )

    axes[1].imshow(
        reconstructed_spec.numpy(),
        origin="lower",
        aspect="auto",
    )

    axes[1].set_title(
        "Reconstructed Spectrogram"
    )

    axes[1].set_xlabel(
        "Time"
    )

    axes[0].set_ylabel(
        "Frequency"
    )

    axes[1].set_ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def plot_tokens(
    tokens,
    path,
):

    tokens = (
        tokens.detach()
        .cpu()
        .flatten()
    )

    plt.figure(
        figsize=(12, 4)
    )

    plt.step(
        range(tokens.numel()),
        tokens.numpy(),
        where="mid",
    )

    plt.xlabel(
        "Token position"
    )

    plt.ylabel(
        "Token ID"
    )

    plt.title(
        "Discrete Speech Tokens"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def plot_codebook_usage(
    counts,
    path,
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        range(len(counts)),
        counts.numpy(),
    )

    plt.xlabel(
        "Codebook index"
    )

    plt.ylabel(
        "Assignments"
    )

    plt.title(
        "Codebook Usage"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ======================================================================
# WAV WRITER
# ======================================================================

def save_audio(
    waveform,
    path,
):

    waveform = (
        waveform.detach()
        .cpu()
        .flatten()
    )

    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    waveform = waveform.clamp(
        -1.0,
        1.0,
    )

    pcm = (
        waveform
        * 32767.0
    ).round().to(
        torch.int16
    )

    with wave.open(
        str(path),
        "wb",
    ) as wav:

        wav.setnchannels(1)

        wav.setsampwidth(2)

        wav.setframerate(
            SAMPLE_RATE
        )

        wav.writeframes(
            pcm.numpy().tobytes()
        )

    print(
        f"Saved WAV: {path}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 36 — "
            "Stable EMA neural speech codec"
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing WAV files."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="lesson36_outputs",
    )

    args = parser.parse_args()

    set_seed()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "PHASE 2 — LESSON 36"
    )

    print(
        "STABLE EMA NEURAL SPEECH CODEC"
    )

    print("=" * 72)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Sample rate: {SAMPLE_RATE}"
    )

    print(
        f"Audio length: {AUDIO_LENGTH}"
    )

    print(
        f"Latent dimension: {LATENT_DIM}"
    )

    print(
        f"Codebook size: {CODEBOOK_SIZE}"
    )

    print(
        f"Learning rate: {args.lr}"
    )

    print(
        f"EMA decay: {EMA_DECAY}"
    )

    print(
        f"Commitment cost: "
        f"{COMMITMENT_COST}"
    )

    # --------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------

    dataset = create_dataset(
        args.data_dir
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    model = SpeechCodec(
        latent_dim=LATENT_DIM,
        codebook_size=CODEBOOK_SIZE,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(
            0.9,
            0.99,
        ),
        weight_decay=1e-4,
        eps=1e-8,
    )

    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------

    print()

    print(
        "Starting training..."
    )

    print()

    history = train_codec(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        steps=args.steps,
    )

    # --------------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------------

    model.eval()

    with torch.no_grad():

        evaluation_batch = next(
            iter(dataloader)
        )

        evaluation_batch = (
            sanitize_waveform(
                evaluation_batch
            )
        )

        evaluation_batch = (
            evaluation_batch.to(
                device
            )
        )

        if evaluation_batch.ndim == 2:

            evaluation_batch = (
                evaluation_batch.unsqueeze(
                    1
                )
            )

        (
            reconstructed,
            tokens,
            vq_loss,
            latent,
        ) = model(
            evaluation_batch
        )

        min_length = min(
            evaluation_batch.shape[-1],
            reconstructed.shape[-1],
        )

        original = (
            evaluation_batch[
                :1,
                :,
                :min_length,
            ]
        )

        reconstructed_one = (
            reconstructed[
                :1,
                :,
                :min_length,
            ]
        )

        reconstructed_one = (
            torch.nan_to_num(
                reconstructed_one
            )
        )

        l1_error = F.l1_loss(
            reconstructed_one,
            original,
        ).item()

        mse_error = F.mse_loss(
            reconstructed_one,
            original,
        ).item()

        snr = approximate_snr(
            original,
            reconstructed_one,
        )

        (
            used_codes,
            perplexity,
            counts,
        ) = codebook_statistics(
            tokens,
            CODEBOOK_SIZE,
        )

    # --------------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------------

    print()

    print("=" * 72)

    print(
        "EVALUATION"
    )

    print("=" * 72)

    print(
        f"L1 reconstruction error : "
        f"{l1_error:.6f}"
    )

    print(
        f"MSE reconstruction error : "
        f"{mse_error:.6f}"
    )

    print(
        f"Approximate SNR          : "
        f"{snr:.2f} dB"
    )

    print(
        f"Used codebook entries    : "
        f"{used_codes}/{CODEBOOK_SIZE}"
    )

    print(
        f"Codebook perplexity      : "
        f"{perplexity:.2f}"
    )

    print(
        f"Final VQ loss            : "
        f"{vq_loss.item():.6f}"
    )

    # --------------------------------------------------------------
    # FINAL SANITY CHECK
    # --------------------------------------------------------------

    if not math.isfinite(
        l1_error
    ):

        raise FloatingPointError(
            "Final L1 error is non-finite."
        )

    if not math.isfinite(
        mse_error
    ):

        raise FloatingPointError(
            "Final MSE is non-finite."
        )

    if not math.isfinite(
        perplexity
    ):

        raise FloatingPointError(
            "Final perplexity is non-finite."
        )

    # --------------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------------

    checkpoint_path = (
        output_dir
        / "lesson36_speech_codec.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "sample_rate":
                SAMPLE_RATE,

            "audio_length":
                AUDIO_LENGTH,

            "latent_dim":
                LATENT_DIM,

            "codebook_size":
                CODEBOOK_SIZE,

            "downsample_factor":
                DOWNSAMPLE_FACTOR,

            "codebook_perplexity":
                perplexity,

            "used_codes":
                used_codes,
        },
        checkpoint_path,
    )

    print()

    print(
        f"Saved checkpoint: "
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------

    plot_training(
        history,
        output_dir
        / "lesson36_codec_training.png",
    )

    plot_reconstruction(
        original,
        reconstructed_one,
        output_dir
        / "lesson36_codec_reconstruction.png",
    )

    plot_detail(
        original,
        reconstructed_one,
        output_dir
        / "lesson36_codec_detail.png",
    )

    plot_spectrograms(
        original.squeeze(),
        reconstructed_one.squeeze(),
        output_dir
        / "lesson36_codec_spectrogram.png",
    )

    plot_tokens(
        tokens[:1],
        output_dir
        / "lesson36_codec_tokens.png",
    )

    plot_codebook_usage(
        counts,
        output_dir
        / "lesson36_codebook_usage.png",
    )

    # --------------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------------

    save_audio(
        original.squeeze(),
        output_dir
        / "lesson36_original.wav",
    )

    save_audio(
        reconstructed_one.squeeze(),
        output_dir
        / "lesson36_reconstructed.wav",
    )

    # --------------------------------------------------------------
    # TOKEN EXAMPLE
    # --------------------------------------------------------------

    print()

    print("=" * 72)

    print(
        "EXAMPLE DISCRETE SPEECH TOKENS"
    )

    print("=" * 72)

    print(
        tokens[
            0,
            :80,
        ]
        .detach()
        .cpu()
        .tolist()
    )

    # --------------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------------

    print()

    print("=" * 72)

    print(
        "LESSON 36 COMPLETE"
    )

    print("=" * 72)

    print()

    print(
        "Waveform"
    )

    print(
        "    ↓"
    )

    print(
        "Encoder"
    )

    print(
        "    ↓"
    )

    print(
        "Continuous latent"
    )

    print(
        "    ↓"
    )

    print(
        "EMA Vector Quantization"
    )

    print(
        "    ↓"
    )

    print(
        "Discrete speech tokens"
    )

    print(
        "    ↓"
    )

    print(
        "Decoder"
    )

    print(
        "    ↓"
    )

    print(
        "Reconstructed waveform"
    )

    print()

    print(
        "Next: Lesson 37 — "
        "Autoregressive Speech Model"
    )

    print(
        "Lesson 37 will learn:"
    )

    print(
        "P(token_t | token_1, ..., token_{t-1})"
    )


if __name__ == "__main__":

    main()
