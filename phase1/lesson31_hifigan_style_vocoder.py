"""
========================================================================
PHASE 1 — LESSON 31
HiFi-GAN-STYLE NEURAL VOCODER FROM SCRATCH
========================================================================

PURPOSE
-------
Build a simplified HiFi-GAN-style neural vocoder from scratch.

The goal is NOT to reproduce the original HiFi-GAN implementation exactly.

Instead, this lesson teaches the core ideas behind neural vocoders:

    MEL SPECTROGRAM
          |
          v
    GENERATOR
          |
          v
    WAVEFORM
          |
          +----------------------+
          |                      |
          v                      v
   MULTI-SCALE             MULTI-PERIOD
   DISCRIMINATOR            DISCRIMINATOR
          |                      |
          +----------+-----------+
                     |
                     v
              ADVERSARIAL LOSS


WHAT YOU LEARN
--------------

STEP 1
------
Convert waveform -> STFT -> Mel spectrogram.

STEP 2
------
Understand why a vocoder maps acoustic features back to waveform.

STEP 3
------
Build a waveform generator using:

    - convolution
    - transposed convolution
    - residual blocks
    - dilated convolutions

STEP 4
------
Build a simplified Multi-Scale Discriminator.

STEP 5
------
Build a simplified Multi-Period Discriminator.

STEP 6
------
Train the generator using:

    adversarial loss
    +
    mel reconstruction loss
    +
    feature matching loss

STEP 7
------
Visualize:

    - target waveform
    - generated waveform
    - target mel
    - generated mel

REAL-WORLD EXAMPLE
------------------
A production TTS system may generate an 80-bin mel spectrogram from:

    text
      |
      v
    acoustic model
      |
      v
    mel spectrogram
      |
      v
    neural vocoder
      |
      v
    speech waveform

HiFi-GAN is one important example of this general approach.

IMPORTANT
---------
This implementation is educational and intentionally smaller than
production HiFi-GAN.

The synthetic dataset is used so that the complete training pipeline
can be studied without requiring a large speech corpus.

DEVICE SAFETY
-------------
The master STFT window and Mel filterbank remain on CPU.

They are moved to the waveform's device inside waveform_to_logmel().

This prevents:

    "stft input and window must be on the same device"

VISUALIZATION SAFETY
--------------------
Mel tensors may have shape:

    [B, M, T]

while matplotlib expects:

    [M, T]

The helper function squeeze_mel_for_plot() handles both formats.

========================================================================
"""

import math
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

SAMPLE_RATE = 16000

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
N_MELS = 80

DURATION = 0.256

NUM_SAMPLES = 192
BATCH_SIZE = 8

EPOCHS = 40

LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4

LAMBDA_ADV = 1.0
LAMBDA_MEL = 45.0
LAMBDA_FM = 2.0

OUTPUT_DIR = Path("outputs/lesson31")

CHECKPOINT_PATH = OUTPUT_DIR / "hifigan_style_generator.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# RANDOM SEED
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# MEL FILTERBANK
# ============================================================================

def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def create_mel_filterbank(
    sample_rate,
    n_fft,
    n_mels,
    f_min=0.0,
    f_max=None,
):
    if f_max is None:
        f_max = sample_rate / 2.0

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = torch.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
    )

    hz_points = torch.tensor(
        [
            mel_to_hz(float(m))
            for m in mel_points
        ],
        dtype=torch.float32,
    )

    bins = torch.floor(
        (n_fft + 1) * hz_points / sample_rate
    ).long()

    filterbank = torch.zeros(
        n_mels,
        n_fft // 2 + 1,
        dtype=torch.float32,
    )

    for m in range(1, n_mels + 1):

        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])

        if center > left:

            filterbank[m - 1, left:center] = (
                torch.arange(left, center)
                - left
            ).float() / float(center - left)

        if right > center:

            filterbank[m - 1, center:right] = (
                right
                - torch.arange(center, right)
            ).float() / float(right - center)

    return filterbank


MEL_FILTERBANK = create_mel_filterbank(
    SAMPLE_RATE,
    N_FFT,
    N_MELS,
)

WINDOW = torch.hann_window(WIN_LENGTH)


# ============================================================================
# WAVEFORM -> LOG MEL
# ============================================================================

def waveform_to_logmel(waveform):

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    device = waveform.device

    window = WINDOW.to(
        device=device,
        dtype=waveform.dtype,
    )

    mel_filterbank = MEL_FILTERBANK.to(
        device=device,
        dtype=waveform.dtype,
    )

    stft = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True,
    )

    magnitude = stft.abs()

    mel = torch.matmul(
        mel_filterbank,
        magnitude,
    )

    mel = torch.clamp(
        mel,
        min=1e-5,
    )

    log_mel = torch.log(mel)

    return log_mel


# ============================================================================
# SYNTHETIC SPEECH-LIKE WAVEFORM
# ============================================================================

def generate_speech_like_waveform(
    duration=DURATION,
    sample_rate=SAMPLE_RATE,
):

    num_samples = int(
        duration * sample_rate
    )

    t = torch.arange(
        num_samples,
        dtype=torch.float32,
    ) / sample_rate

    base_frequencies = [
        110.0,
        130.0,
        150.0,
        180.0,
        200.0,
        220.0,
        250.0,
        280.0,
    ]

    f0 = random.choice(
        base_frequencies
    )

    vibrato = (
        3.0
        * torch.sin(
            2.0
            * math.pi
            * 5.0
            * t
        )
    )

    instantaneous_frequency = (
        f0 + vibrato
    )

    phase = (
        2.0
        * math.pi
        * torch.cumsum(
            instantaneous_frequency,
            dim=0,
        )
        / sample_rate
    )

    waveform = (
        0.55 * torch.sin(phase)
        + 0.25 * torch.sin(2.0 * phase)
        + 0.12 * torch.sin(3.0 * phase)
        + 0.06 * torch.sin(4.0 * phase)
    )

    envelope = torch.ones_like(
        waveform
    )

    attack = int(
        0.02 * sample_rate
    )

    release = int(
        0.04 * sample_rate
    )

    if attack > 0:

        envelope[:attack] = (
            torch.linspace(
                0.0,
                1.0,
                attack,
            )
        )

    if release > 0:

        envelope[-release:] = (
            torch.linspace(
                1.0,
                0.0,
                release,
            )
        )

    waveform = (
        waveform * envelope
    )

    noise = (
        0.01
        * torch.randn_like(waveform)
    )

    waveform = waveform + noise

    waveform = waveform / (
        waveform.abs().max()
        + 1e-8
    )

    return waveform


# ============================================================================
# DATASET
# ============================================================================

class SyntheticSpeechDataset(
    Dataset
):

    def __init__(
        self,
        num_samples=NUM_SAMPLES,
    ):

        self.mels = []
        self.waveforms = []

        print(
            "Generating synthetic speech dataset..."
        )

        for _ in range(num_samples):

            waveform = (
                generate_speech_like_waveform()
            )

            mel = waveform_to_logmel(
                waveform
            )

            self.waveforms.append(
                waveform
            )

            self.mels.append(
                mel.squeeze(0)
            )

        self.waveforms = torch.stack(
            self.waveforms
        )

        self.mels = torch.stack(
            self.mels
        )

        print(
            f"Dataset waveforms: "
            f"{self.waveforms.shape}"
        )

        print(
            f"Dataset mels: "
            f"{self.mels.shape}"
        )

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, index):

        return (
            self.mels[index],
            self.waveforms[index],
        )


# ============================================================================
# RESIDUAL DILATED BLOCK
# ============================================================================

class ResidualDilatedBlock(
    nn.Module
):

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilations=(1, 3, 5),
    ):

        super().__init__()

        self.blocks = nn.ModuleList()

        for dilation in dilations:

            padding = (
                (kernel_size - 1)
                * dilation
                // 2
            )

            self.blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(
                        0.1
                    ),

                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),

                    nn.LeakyReLU(
                        0.1
                    ),

                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=1,
                    ),
                )
            )

    def forward(self, x):

        for block in self.blocks:

            residual = x

            x = block(x)

            x = x + residual

        return x


# ============================================================================
# VOCODER GENERATOR
# ============================================================================

class VocoderGenerator(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.input = nn.Conv1d(
            N_MELS,
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

        self.res1 = ResidualDilatedBlock(
            64
        )

        self.up2 = nn.ConvTranspose1d(
            64,
            32,
            kernel_size=16,
            stride=8,
            padding=4,
        )

        self.res2 = ResidualDilatedBlock(
            32
        )

        self.up3 = nn.ConvTranspose1d(
            32,
            16,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.res3 = ResidualDilatedBlock(
            16
        )

        self.output = nn.Sequential(

            nn.LeakyReLU(
                0.1
            ),

            nn.Conv1d(
                16,
                1,
                kernel_size=7,
                padding=3,
            ),

            nn.Tanh(),
        )

    def forward(self, mel):

        x = self.input(mel)

        x = F.leaky_relu(
            x,
            0.1,
        )

        x = self.up1(x)
        x = self.res1(x)

        x = self.up2(x)
        x = self.res2(x)

        x = self.up3(x)
        x = self.res3(x)

        x = self.output(x)

        return x


# ============================================================================
# SCALE DISCRIMINATOR
# ============================================================================

class ScaleDiscriminator(
    nn.Module
):

    def __init__(
        self,
        channels=32,
    ):

        super().__init__()

        self.layers = nn.ModuleList([

            nn.Conv1d(
                1,
                channels,
                15,
                padding=7,
            ),

            nn.Conv1d(
                channels,
                channels * 2,
                41,
                stride=4,
                groups=4,
                padding=20,
            ),

            nn.Conv1d(
                channels * 2,
                channels * 4,
                41,
                stride=4,
                groups=4,
                padding=20,
            ),

            nn.Conv1d(
                channels * 4,
                channels * 8,
                41,
                stride=4,
                groups=4,
                padding=20,
            ),

            nn.Conv1d(
                channels * 8,
                channels * 16,
                5,
                padding=2,
            ),
        ])

        self.final = nn.Conv1d(
            channels * 16,
            1,
            3,
            padding=1,
        )

    def forward(self, x):

        features = []

        for layer in self.layers:

            x = layer(x)

            x = F.leaky_relu(
                x,
                0.1,
            )

            features.append(x)

        score = self.final(x)

        features.append(score)

        return score, features


# ============================================================================
# MULTI-SCALE DISCRIMINATOR
# ============================================================================

class MultiScaleDiscriminator(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(16),
            ScaleDiscriminator(16),
            ScaleDiscriminator(16),
        ])

    def forward(self, x):

        outputs = []

        current = x

        for i, discriminator in enumerate(
            self.discriminators
        ):

            score, features = (
                discriminator(current)
            )

            outputs.append(
                (score, features)
            )

            if i < 2:

                current = F.avg_pool1d(
                    current,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )

        return outputs


# ============================================================================
# PERIOD DISCRIMINATOR
# ============================================================================

class PeriodDiscriminator(
    nn.Module
):

    def __init__(
        self,
        period,
        channels=16,
    ):

        super().__init__()

        self.period = period

        self.layers = nn.ModuleList([

            nn.Conv2d(
                1,
                channels,
                (5, 1),
                (3, 1),
                padding=(2, 0),
            ),

            nn.Conv2d(
                channels,
                channels * 2,
                (5, 1),
                (3, 1),
                padding=(2, 0),
            ),

            nn.Conv2d(
                channels * 2,
                channels * 4,
                (5, 1),
                (3, 1),
                padding=(2, 0),
            ),

            nn.Conv2d(
                channels * 4,
                channels * 8,
                (5, 1),
                (3, 1),
                padding=(2, 0),
            ),
        ])

        self.final = nn.Conv2d(
            channels * 8,
            1,
            kernel_size=(3, 1),
            padding=(1, 0),
        )

    def forward(self, x):

        batch, channels, length = (
            x.shape
        )

        pad_length = (
            self.period
            - length % self.period
        ) % self.period

        if pad_length > 0:

            x = F.pad(
                x,
                (0, pad_length),
                mode="reflect",
            )

        length = x.shape[-1]

        x = x.view(
            batch,
            channels,
            length // self.period,
            self.period,
        )

        features = []

        for layer in self.layers:

            x = layer(x)

            x = F.leaky_relu(
                x,
                0.1,
            )

            features.append(x)

        score = self.final(x)

        features.append(score)

        return score, features


# ============================================================================
# MULTI-PERIOD DISCRIMINATOR
# ============================================================================

class MultiPeriodDiscriminator(
    nn.Module
):

    def __init__(self):

        super().__init__()

        periods = [
            2,
            3,
            5,
            7,
        ]

        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p)
            for p in periods
        ])

    def forward(self, x):

        outputs = []

        for discriminator in (
            self.discriminators
        ):

            outputs.append(
                discriminator(x)
            )

        return outputs


# ============================================================================
# ADVERSARIAL LOSSES
# ============================================================================

def discriminator_loss(
    real_outputs,
    fake_outputs,
):

    loss = 0.0

    for (
        real_score,
        _,
    ), (
        fake_score,
        _,
    ) in zip(
        real_outputs,
        fake_outputs,
    ):

        loss = loss + (
            torch.mean(
                (real_score - 1.0) ** 2
            )
            + torch.mean(
                fake_score ** 2
            )
        )

    return loss


def generator_adversarial_loss(
    fake_outputs,
):

    loss = 0.0

    for fake_score, _ in fake_outputs:

        loss = loss + torch.mean(
            (fake_score - 1.0) ** 2
        )

    return loss


# ============================================================================
# FEATURE MATCHING LOSS
# ============================================================================

def feature_matching_loss(
    real_outputs,
    fake_outputs,
):

    loss = 0.0

    for (
        _,
        real_features,
    ), (
        _,
        fake_features,
    ) in zip(
        real_outputs,
        fake_outputs,
    ):

        for real_feature, fake_feature in zip(
            real_features,
            fake_features,
        ):

            loss = loss + F.l1_loss(
                fake_feature,
                real_feature.detach(),
            )

    return loss


# ============================================================================
# MEL PLOT HELPER
# ============================================================================

def squeeze_mel_for_plot(mel):
    """
    Convert a Mel tensor into a 2-D [M, T] tensor.

    Supported inputs:

        [M, T]
        [1, M, T]
        [B, M, T]

    For visualization we use the first batch item.
    """

    if isinstance(mel, np.ndarray):

        array = mel

    else:

        array = mel.detach().cpu().numpy()

    array = np.asarray(array)

    if array.ndim == 4:

        # Unexpected [B, C, M, T]
        array = array[0, 0]

    elif array.ndim == 3:

        # [B, M, T]
        array = array[0]

    elif array.ndim == 2:

        # Already [M, T]
        pass

    elif array.ndim == 1:

        array = array[None, :]

    else:

        raise ValueError(
            "Unsupported mel shape: "
            f"{array.shape}"
        )

    if array.ndim != 2:

        raise ValueError(
            "Mel data must be 2-D after "
            f"processing, got {array.shape}"
        )

    return array


# ============================================================================
# WAVEFORM PLOT HELPER
# ============================================================================

def squeeze_waveform_for_plot(
    waveform
):

    if isinstance(
        waveform,
        np.ndarray,
    ):

        array = waveform

    else:

        array = (
            waveform
            .detach()
            .cpu()
            .numpy()
        )

    array = np.asarray(array)

    if array.ndim == 3:

        # [B, C, T]
        array = array[0, 0]

    elif array.ndim == 2:

        # Could be [B, T] or [C, T]
        array = array[0]

    elif array.ndim == 1:

        pass

    else:

        raise ValueError(
            "Unsupported waveform shape: "
            f"{array.shape}"
        )

    return array


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(
    generator,
    msd,
    mpd,
    loader,
    optimizer_g,
    optimizer_d,
):

    generator.train()
    msd.train()
    mpd.train()

    total_g = 0.0
    total_d = 0.0
    total_mel = 0.0

    for mel, real_waveform in loader:

        mel = mel.to(
            DEVICE,
            non_blocking=True,
        )

        real_waveform = (
            real_waveform
            .to(
                DEVICE,
                non_blocking=True,
            )
            .unsqueeze(1)
        )

        # --------------------------------------------------------------
        # GENERATOR
        # --------------------------------------------------------------

        fake_waveform = generator(
            mel
        )

        # Match waveform length.
        target_length = min(
            fake_waveform.shape[-1],
            real_waveform.shape[-1],
        )

        fake_waveform = (
            fake_waveform[
                ...,
                :target_length
            ]
        )

        real_waveform = (
            real_waveform[
                ...,
                :target_length
            ]
        )

        # --------------------------------------------------------------
        # DISCRIMINATORS
        # --------------------------------------------------------------

        with torch.no_grad():

            real_msd = msd(
                real_waveform
            )

            real_mpd = mpd(
                real_waveform
            )

        fake_msd = msd(
            fake_waveform.detach()
        )

        fake_mpd = mpd(
            fake_waveform.detach()
        )

        loss_d_msd = discriminator_loss(
            real_msd,
            fake_msd,
        )

        loss_d_mpd = discriminator_loss(
            real_mpd,
            fake_mpd,
        )

        loss_d = (
            loss_d_msd
            + loss_d_mpd
        )

        optimizer_d.zero_grad(
            set_to_none=True
        )

        loss_d.backward()

        optimizer_d.step()

        # --------------------------------------------------------------
        # GENERATOR ADVERSARIAL STEP
        # --------------------------------------------------------------

        fake_msd = msd(
            fake_waveform
        )

        fake_mpd = mpd(
            fake_waveform
        )

        real_msd = msd(
            real_waveform
        )

        real_mpd = mpd(
            real_waveform
        )

        loss_adv_msd = (
            generator_adversarial_loss(
                fake_msd
            )
        )

        loss_adv_mpd = (
            generator_adversarial_loss(
                fake_mpd
            )
        )

        loss_adv = (
            loss_adv_msd
            + loss_adv_mpd
        )

        # --------------------------------------------------------------
        # FEATURE MATCHING
        # --------------------------------------------------------------

        loss_fm_msd = (
            feature_matching_loss(
                real_msd,
                fake_msd,
            )
        )

        loss_fm_mpd = (
            feature_matching_loss(
                real_mpd,
                fake_mpd,
            )
        )

        loss_fm = (
            loss_fm_msd
            + loss_fm_mpd
        )

        # --------------------------------------------------------------
        # MEL RECONSTRUCTION
        # --------------------------------------------------------------

        fake_mel = waveform_to_logmel(
            fake_waveform.squeeze(1)
        )

        target_mel = mel[
            ...,
            :fake_mel.shape[-1]
        ]

        mel_loss = F.l1_loss(
            fake_mel,
            target_mel,
        )

        # --------------------------------------------------------------
        # TOTAL GENERATOR LOSS
        # --------------------------------------------------------------

        loss_g = (
            LAMBDA_ADV * loss_adv
            + LAMBDA_MEL * mel_loss
            + LAMBDA_FM * loss_fm
        )

        optimizer_g.zero_grad(
            set_to_none=True
        )

        loss_g.backward()

        optimizer_g.step()

        total_g += (
            loss_g.item()
        )

        total_d += (
            loss_d.item()
        )

        total_mel += (
            mel_loss.item()
        )

    n = len(loader)

    return (
        total_g / n,
        total_d / n,
        total_mel / n,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "PHASE 1 — LESSON 31"
    )
    print(
        "HiFi-GAN-Style Neural Vocoder"
    )
    print("=" * 72)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Sample rate: {SAMPLE_RATE}"
    )

    print(
        f"FFT: {N_FFT}"
    )

    print(
        f"Hop length: {HOP_LENGTH}"
    )

    print(
        f"Mel bins: {N_MELS}"
    )

    print(
        f"Training samples: {NUM_SAMPLES}"
    )

    # ------------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------------

    dataset = SyntheticSpeechDataset(
        NUM_SAMPLES
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    # ------------------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------------------

    generator = (
        VocoderGenerator()
        .to(DEVICE)
    )

    msd = (
        MultiScaleDiscriminator()
        .to(DEVICE)
    )

    mpd = (
        MultiPeriodDiscriminator()
        .to(DEVICE)
    )

    # ------------------------------------------------------------------------
    # PARAMETER COUNTS
    # ------------------------------------------------------------------------

    generator_parameters = sum(
        p.numel()
        for p in generator.parameters()
        if p.requires_grad
    )

    msd_parameters = sum(
        p.numel()
        for p in msd.parameters()
        if p.requires_grad
    )

    mpd_parameters = sum(
        p.numel()
        for p in mpd.parameters()
        if p.requires_grad
    )

    print(
        f"Generator parameters: "
        f"{generator_parameters:,}"
    )

    print(
        f"MSD parameters: "
        f"{msd_parameters:,}"
    )

    print(
        f"MPD parameters: "
        f"{mpd_parameters:,}"
    )

    # ------------------------------------------------------------------------
    # OPTIMIZERS
    # ------------------------------------------------------------------------

    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=LEARNING_RATE_G,
        betas=(0.8, 0.99),
    )

    optimizer_d = torch.optim.AdamW(
        list(msd.parameters())
        + list(mpd.parameters()),
        lr=LEARNING_RATE_D,
        betas=(0.8, 0.99),
    )

    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    print()
    print(
        "Starting training..."
    )
    print()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        (
            loss_g,
            loss_d,
            loss_mel,
        ) = train_one_epoch(
            generator,
            msd,
            mpd,
            loader,
            optimizer_g,
            optimizer_d,
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS} | "
            f"G: {loss_g:.4f} | "
            f"D: {loss_d:.4f} | "
            f"Mel: {loss_mel:.6f}"
        )

    # ------------------------------------------------------------------------
    # SAVE GENERATOR
    # ------------------------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                generator.state_dict(),

            "sample_rate":
                SAMPLE_RATE,

            "n_fft":
                N_FFT,

            "hop_length":
                HOP_LENGTH,

            "n_mels":
                N_MELS,
        },
        CHECKPOINT_PATH,
    )

    print()
    print(
        f"Saved generator checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    # ------------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------------

    generator.eval()

    with torch.no_grad():

        sample_mel = dataset.mels[
            0
        ].unsqueeze(0).to(
            DEVICE
        )

        target_waveform = (
            dataset.waveforms[
                0
            ].unsqueeze(0).to(
                DEVICE
            )
        )

        generated_waveform = (
            generator(
                sample_mel
            )
            .squeeze(1)
        )

        target_length = min(
            target_waveform.shape[-1],
            generated_waveform.shape[-1],
        )

        target_waveform = (
            target_waveform[
                ...,
                :target_length
            ]
        )

        generated_waveform = (
            generated_waveform[
                ...,
                :target_length
            ]
        )

        generated_mel = (
            waveform_to_logmel(
                generated_waveform
            )
        )

        target_mel = (
            sample_mel[
                ...,
                :generated_mel.shape[-1]
            ]
        )

        mel_reconstruction_error = (
            F.l1_loss(
                generated_mel,
                target_mel,
            )
            .item()
        )

    print()
    print(
        f"Mel reconstruction error: "
        f"{mel_reconstruction_error:.6f}"
    )

    # ------------------------------------------------------------------------
    # CONVERT FOR PLOTTING
    # ------------------------------------------------------------------------

    target_waveform_np = (
        squeeze_waveform_for_plot(
            target_waveform
        )
    )

    generated_waveform_np = (
        squeeze_waveform_for_plot(
            generated_waveform
        )
    )

    target_mel_np = (
        squeeze_mel_for_plot(
            target_mel
        )
    )

    generated_mel_np = (
        squeeze_mel_for_plot(
            generated_mel
        )
    )

    print()
    print(
        "Visualization shapes:"
    )

    print(
        f"Target waveform: "
        f"{target_waveform_np.shape}"
    )

    print(
        f"Generated waveform: "
        f"{generated_waveform_np.shape}"
    )

    print(
        f"Target mel: "
        f"{target_mel_np.shape}"
    )

    print(
        f"Generated mel: "
        f"{generated_mel_np.shape}"
    )

    # ------------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------------

    time_target = (
        np.arange(
            len(target_waveform_np)
        )
        / SAMPLE_RATE
    )

    time_generated = (
        np.arange(
            len(generated_waveform_np)
        )
        / SAMPLE_RATE
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
    )

    # ------------------------------------------------------------------------
    # TARGET WAVEFORM
    # ------------------------------------------------------------------------

    axes[0, 0].plot(
        time_target,
        target_waveform_np,
    )

    axes[0, 0].set_title(
        "Target Waveform"
    )

    axes[0, 0].set_xlabel(
        "Time (seconds)"
    )

    axes[0, 0].set_ylabel(
        "Amplitude"
    )

    # ------------------------------------------------------------------------
    # GENERATED WAVEFORM
    # ------------------------------------------------------------------------

    axes[0, 1].plot(
        time_generated,
        generated_waveform_np,
    )

    axes[0, 1].set_title(
        "Generated Waveform"
    )

    axes[0, 1].set_xlabel(
        "Time (seconds)"
    )

    axes[0, 1].set_ylabel(
        "Amplitude"
    )

    # ------------------------------------------------------------------------
    # TARGET MEL
    # ------------------------------------------------------------------------

    axes[1, 0].imshow(
        target_mel_np,
        aspect="auto",
        origin="lower",
    )

    axes[1, 0].set_title(
        "Target Log-Mel Spectrogram"
    )

    axes[1, 0].set_xlabel(
        "Time Frames"
    )

    axes[1, 0].set_ylabel(
        "Mel Bin"
    )

    # ------------------------------------------------------------------------
    # GENERATED MEL
    # ------------------------------------------------------------------------

    axes[1, 1].imshow(
        generated_mel_np,
        aspect="auto",
        origin="lower",
    )

    axes[1, 1].set_title(
        "Generated Log-Mel Spectrogram"
    )

    axes[1, 1].set_xlabel(
        "Time Frames"
    )

    axes[1, 1].set_ylabel(
        "Mel Bin"
    )

    plt.suptitle(
        "Lesson 31 — HiFi-GAN-Style Neural Vocoder",
        fontsize=15,
    )

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / "lesson31_vocoder_result.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    print()
    print(
        f"Saved visualization:"
    )

    print(
        plot_path
    )

    plt.show()

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "LESSON 31 COMPLETE"
    )
    print("=" * 72)

    print(
        """
You built a simplified HiFi-GAN-style neural vocoder.

Pipeline:

    waveform
       |
       v
     STFT
       |
       v
    Mel filterbank
       |
       v
    log-mel
       |
       v
    Generator
       |
       v
    waveform

Training used:

    1. Mel reconstruction loss
    2. Adversarial loss
    3. Feature matching loss

Discriminators:

    Multi-Scale Discriminator
    Multi-Period Discriminator

The visualization now safely converts:

    [B, 80, T]
        ->
    [80, T]

before passing the Mel spectrogram to matplotlib.

NEXT LESSON

Lesson 32 — Text / Linguistic Representation

The next major step is to replace artificial synthetic content IDs
with actual linguistic information.

We will begin connecting:

    TEXT
      |
      v
    linguistic representation
      |
      v
    acoustic representation
      |
      v
    neural vocoder
      |
      v
    waveform

This is the bridge from a standalone vocoder toward a
real text-to-speech architecture.
"""
    )


if __name__ == "__main__":
    main()