"""
File: phase1_lesson16_neural_vocoder_training.py

PHASE 1 — LESSON 16
BUILD A NEURAL VOCODER FROM SCRATCH
===================================

PURPOSE
-------
Build and train a small Mel-conditioned GAN vocoder using PyTorch.

This lesson moves from signal-processing reconstruction (Griffin-Lim)
to neural waveform generation.

The complete training pipeline includes:

    1. Synthetic paired waveform/Mel dataset generation
    2. Mel spectrogram extraction
    3. Mel-conditioned waveform generator
    4. Transposed-convolution upsampling
    5. Residual convolution blocks
    6. Waveform discriminator
    7. Multi-resolution STFT loss
    8. Least-squares adversarial loss
    9. Feature-matching loss
   10. Generator/discriminator optimization
   11. Checkpoint saving
   12. WAV export and visualization

IMPORTANT LIMITATION
--------------------
The training signals are procedural, speech-like harmonic signals.
They are NOT recordings of human speech.

This experiment teaches the neural vocoder training process. It does
not produce a production-quality natural-speech vocoder.

The generator and discriminator are initialized randomly and trained
from scratch. No pretrained neural vocoder is used.

RUN
---
    python phase1_lesson16_neural_vocoder_training.py

Optional:
    python phase1_lesson16_neural_vocoder_training.py --epochs 20
    python phase1_lesson16_neural_vocoder_training.py --samples 256
    python phase1_lesson16_neural_vocoder_training.py --batch-size 4
    python phase1_lesson16_neural_vocoder_training.py --no-plot

OUTPUT
------
    lesson16_vocoder_checkpoint.pt
    lesson16_generated.wav
    lesson16_training_curves.png
    lesson16_waveform_comparison.png

DEPENDENCIES
------------
    pip install numpy matplotlib soundfile torch


============================================================
THEORY 1 — THE NEURAL VOCODER PROBLEM
============================================================

A TTS acoustic model normally produces a Mel spectrogram:

    M = AcousticModel(text, speaker)

A vocoder converts this representation into an audio waveform:

    x = Vocoder(M)

The Mel spectrogram has a much lower temporal resolution than the
waveform.

For example:

    Sample rate = 24,000 Hz
    Hop length  = 256 samples

The Mel spectrogram advances once every:

    256 / 24000 = 0.010667 seconds

That is approximately 10.67 milliseconds per frame.

The vocoder must generate 256 waveform samples for every Mel frame.

This is a temporal upsampling problem.


============================================================
THEORY 2 — GENERATOR
============================================================

The generator is a function:

    x_hat = G(M)

Input:

    M: (batch, mel_channels, mel_frames)

Output:

    x_hat: (batch, 1, mel_frames * hop_length)

The generator consists of:

    Initial convolution
    Transposed-convolution upsampling
    Residual convolution blocks
    Final waveform projection

A transposed convolution increases temporal resolution.

For a transposed convolution:

    output_length =
        (input_length - 1) * stride
        - 2 * padding
        + kernel_size

With:

    stride = 4
    kernel = 8
    padding = 2

We obtain:

    output_length = 4 * input_length

Four stages produce:

    4^4 = 256 times upsampling.


============================================================
THEORY 3 — RESIDUAL BLOCKS
============================================================

A residual block learns a transformation F(x), then adds the input:

    y = x + F(x)

This creates a shorter gradient path and makes deeper convolutional
networks easier to optimize.

We use dilated convolutions to increase the receptive field.

For dilation d:

    effective_kernel_size = 1 + (kernel_size - 1) * d

A kernel of size 3 with dilation 5 has an effective width of:

    1 + (3 - 1) * 5 = 11 samples


============================================================
THEORY 4 — DISCRIMINATOR
============================================================

The discriminator learns to distinguish real and generated waveforms.

    D(real_waveform) -> approximately 1
    D(fake_waveform) -> approximately 0

The generator attempts to produce audio that the discriminator
classifies as real.

We use a least-squares GAN objective.

Discriminator:

    L_D = 0.5 * (
        mean((D(real) - 1)^2)
        +
        mean((D(fake) - 0)^2)
    )

Generator adversarial loss:

    L_G_adv = mean((D(fake) - 1)^2)


============================================================
THEORY 5 — MULTI-RESOLUTION STFT LOSS
============================================================

Different STFT configurations reveal different properties.

Small FFT:
    Better temporal detail.

Large FFT:
    Better frequency resolution.

We compare the generated waveform with the real waveform at
multiple FFT resolutions.

Spectral convergence:

    L_sc = || |STFT(x)| - |STFT(x_hat)| ||_F
           / (|| |STFT(x)| ||_F + epsilon)

Log-magnitude loss:

    L_mag = mean(
        abs(
            log(|STFT(x)| + epsilon)
            -
            log(|STFT(x_hat)| + epsilon)
        )
    )

The combined multi-resolution loss is:

    L_MRSTFT = mean_over_resolutions(L_sc + L_mag)


============================================================
THEORY 6 — FEATURE MATCHING
============================================================

The discriminator contains intermediate feature maps.

Instead of comparing only its final output, feature matching compares
the internal activations produced by real and generated audio.

    L_FM = sum_l mean(abs(D_l(real) - D_l(fake)))

The real feature maps are detached from the gradient graph.

This gives the generator a richer learning signal.


============================================================
THEORY 7 — IMPORTANT PRACTICAL LIMITATIONS
============================================================

1. Synthetic audio is not human speech.
2. The discriminator is a small educational discriminator.
3. The generator is much smaller than HiFi-GAN.
4. The dataset is intentionally limited.
5. Training may be unstable with very small datasets.
6. Waveform L1 is phase-sensitive.
7. A production vocoder needs a large paired speech dataset.
8. Real speech training should use carefully matched waveform and
   Mel preprocessing.
9. Real TTS quality also depends on the acoustic model and text
   frontend, not just the vocoder.

This lesson is a controlled neural-network experiment.
"""


import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 24000
HOP_LENGTH = 256
N_FFT = 1024
N_MELS = 80

SEGMENT_FRAMES = 32
SEGMENT_SAMPLES = SEGMENT_FRAMES * HOP_LENGTH

EPS = 1e-7

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
# SYNTHETIC SPEECH-LIKE SIGNAL GENERATION
# ============================================================

def generate_synthetic_waveform(
    length,
    sample_rate,
    rng
):
    """
    Generate a controlled speech-like harmonic waveform.

    This is NOT a human speech synthesizer.

    The signal contains:
        - A fundamental frequency
        - Harmonic components
        - Simplified spectral-envelope peaks
        - Amplitude variation
        - A small noise component

    The purpose is to create a dataset with a learnable relationship
    between a waveform and its Mel spectrogram.
    """

    t = np.arange(length, dtype=np.float32) / sample_rate

    f0 = rng.uniform(90.0, 230.0)

    vibrato_rate = rng.uniform(2.0, 6.0)
    vibrato_depth = rng.uniform(0.002, 0.012)

    instantaneous_f0 = (
        f0
        * (
            1.0
            + vibrato_depth
            * np.sin(2.0 * np.pi * vibrato_rate * t)
        )
    )

    phase = (
        2.0
        * np.pi
        * np.cumsum(instantaneous_f0)
        / sample_rate
    )

    # Simplified spectral envelope.
    formants = [
        rng.uniform(400, 900),
        rng.uniform(1000, 2000),
        rng.uniform(2000, 3500),
    ]

    bandwidths = [
        rng.uniform(100, 250),
        rng.uniform(150, 350),
        rng.uniform(200, 500),
    ]

    signal = np.zeros(length, dtype=np.float32)

    max_harmonics = min(
        100,
        int((sample_rate / 2) / max(f0, 1.0))
    )

    for harmonic in range(1, max_harmonics + 1):

        frequency = harmonic * f0

        if frequency >= sample_rate / 2:
            break

        # Simplified formant weighting.
        envelope = 0.0

        for center, bandwidth in zip(formants, bandwidths):

            envelope += np.exp(
                -0.5
                * ((frequency - center) / bandwidth) ** 2
            )

        # A small baseline prevents all harmonics from disappearing.
        envelope += 0.08

        # Harmonic roll-off.
        harmonic_rolloff = 1.0 / (harmonic ** 1.15)

        amplitude = envelope * harmonic_rolloff

        signal += (
            amplitude
            * np.sin(harmonic * phase)
        ).astype(np.float32)

    # Smooth amplitude envelope.
    envelope = (
        0.65
        + 0.25 * np.sin(2.0 * np.pi * 2.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 0.7 * t)
    )

    envelope = np.clip(envelope, 0.15, 1.0)

    signal *= envelope.astype(np.float32)

    # Small unvoiced-like noise component.
    noise_level = rng.uniform(0.005, 0.035)

    noise = rng.normal(
        0.0,
        noise_level,
        size=length
    ).astype(np.float32)

    signal += noise

    # Random amplitude.
    signal *= rng.uniform(0.5, 1.0)

    # Avoid clipping.
    peak = np.max(np.abs(signal)) + EPS
    signal = signal / peak * 0.85

    return signal.astype(np.float32)


# ============================================================
# MEL FILTERBANK
# ============================================================

def hz_to_mel(frequency):
    return 2595.0 * np.log10(
        1.0 + frequency / 700.0
    )


def mel_to_hz(mel):
    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def create_mel_filterbank(
    sample_rate,
    n_fft,
    n_mels,
    fmin=0.0,
    fmax=None
):
    """
    Create a triangular Mel filterbank.

    Output shape:

        (n_mels, n_fft // 2 + 1)
    """

    if fmax is None:
        fmax = sample_rate / 2

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2
    )

    hz_points = mel_to_hz(mel_points)

    frequencies = np.linspace(
        0,
        sample_rate / 2,
        n_fft // 2 + 1
    )

    filters = np.zeros(
        (n_mels, len(frequencies)),
        dtype=np.float32
    )

    for m in range(n_mels):

        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]

        rising = (
            (frequencies - left)
            / max(center - left, EPS)
        )

        falling = (
            (right - frequencies)
            / max(right - center, EPS)
        )

        filters[m] = np.maximum(
            0.0,
            np.minimum(rising, falling)
        )

    return torch.tensor(filters, dtype=torch.float32)


MEL_FILTERBANK = create_mel_filterbank(
    SAMPLE_RATE,
    N_FFT,
    N_MELS
)


# ============================================================
# WAVEFORM -> LOG MEL
# ============================================================

def waveform_to_logmel(waveform, mel_filterbank):
    """
    waveform:
        (batch, samples)

    returns:
        (batch, n_mels, frames)

    center=True gives one extra STFT frame for the segment length.
    We crop to SEGMENT_FRAMES so that the generator's output length
    exactly matches the target waveform length.
    """

    window = torch.hann_window(
        N_FFT,
        device=waveform.device
    )

    stft = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True
    )

    power = stft.abs().pow(2)

    mel_filterbank = mel_filterbank.to(
        waveform.device
    )

    mel = torch.matmul(
        mel_filterbank,
        power
    )

    mel = mel[:, :, :SEGMENT_FRAMES]

    logmel = torch.log(
        mel + EPS
    )

    return logmel


# ============================================================
# DATASET
# ============================================================

class SyntheticVocoderDataset(Dataset):

    def __init__(
        self,
        number_of_samples=128,
        seed=42
    ):

        self.waveforms = []

        rng = np.random.default_rng(seed)

        print(
            f"Generating {number_of_samples} "
            "synthetic training signals..."
        )

        for _ in range(number_of_samples):

            waveform = generate_synthetic_waveform(
                SEGMENT_SAMPLES,
                SAMPLE_RATE,
                rng
            )

            self.waveforms.append(waveform)

        self.waveforms = np.stack(
            self.waveforms
        )

        self.waveforms = torch.tensor(
            self.waveforms,
            dtype=torch.float32
        )

        # Calculate paired Mel spectrograms.
        with torch.no_grad():

            logmel = waveform_to_logmel(
                self.waveforms,
                MEL_FILTERBANK
            )

        # Global normalization is calculated from this dataset.
        self.mel_mean = logmel.mean()
        self.mel_std = logmel.std().clamp_min(1e-5)

        self.logmels = (
            logmel - self.mel_mean
        ) / self.mel_std

        print("Dataset ready.")
        print("Waveform shape:", tuple(self.waveforms.shape))
        print("Mel shape:", tuple(self.logmels.shape))

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, index):

        return (
            self.logmels[index],
            self.waveforms[index]
        )


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):
    """
    Residual convolution block.

    y = x + F(x)

    Dilated convolutions increase the receptive field without
    requiring very large kernels.
    """

    def __init__(
        self,
        channels,
        dilation
    ):

        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=padding
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):

        residual = x

        y = self.activation(x)
        y = self.conv1(y)

        y = self.activation(y)
        y = self.conv2(y)

        return residual + y


# ============================================================
# GENERATOR
# ============================================================

class NeuralVocoderGenerator(nn.Module):
    """
    Mel-conditioned waveform generator.

    Input:
        (B, 80, T)

    Output:
        (B, 1, T * 256)

    Four transposed convolutions each multiply time resolution by 4.
    """

    def __init__(self, mel_channels=80):

        super().__init__()

        self.initial = nn.Conv1d(
            mel_channels,
            256,
            kernel_size=7,
            padding=3
        )

        self.upsample_layers = nn.ModuleList()

        self.residual_stages = nn.ModuleList()

        channel_pairs = [
            (256, 128),
            (128, 64),
            (64, 32),
            (32, 16),
        ]

        for in_channels, out_channels in channel_pairs:

            self.upsample_layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=8,
                    stride=4,
                    padding=2
                )
            )

            self.residual_stages.append(
                nn.Sequential(
                    ResidualBlock(out_channels, dilation=1),
                    ResidualBlock(out_channels, dilation=3),
                    ResidualBlock(out_channels, dilation=5),
                )
            )

        self.final = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(
                16,
                1,
                kernel_size=7,
                padding=3
            ),
            nn.Tanh()
        )

    def forward(self, mel):

        x = self.initial(mel)

        for upsample, residual_blocks in zip(
            self.upsample_layers,
            self.residual_stages
        ):

            x = F.leaky_relu(
                x,
                negative_slope=0.1
            )

            x = upsample(x)

            x = residual_blocks(x)

        return self.final(x)


# ============================================================
# DISCRIMINATOR
# ============================================================

class WaveformDiscriminator(nn.Module):
    """
    Small waveform discriminator.

    It returns:
        final prediction
        intermediate feature maps

    Feature maps are used by the feature-matching loss.
    """

    def __init__(self):

        super().__init__()

        channels = [
            1,
            16,
            64,
            128,
            256
        ]

        layers = []

        for index in range(len(channels) - 1):

            in_channels = channels[index]
            out_channels = channels[index + 1]

            layers.append(
                spectral_norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=15,
                        stride=4 if index > 0 else 1,
                        padding=7
                    )
                )
            )

        self.layers = nn.ModuleList(layers)

        self.final = spectral_norm(
            nn.Conv1d(
                256,
                1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, waveform):

        features = []

        x = waveform

        for layer in self.layers:

            x = layer(x)

            x = F.leaky_relu(
                x,
                negative_slope=0.1
            )

            features.append(x)

        prediction = self.final(x)

        features.append(prediction)

        return prediction, features


# ============================================================
# MULTI-RESOLUTION STFT LOSS
# ============================================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    Compare real and generated waveforms at multiple STFT resolutions.

    Three resolutions:
        FFT 256
        FFT 512
        FFT 1024

    This combines temporal and frequency-domain information.
    """

    def __init__(self):

        super().__init__()

        self.resolutions = [
            (256, 64),
            (512, 128),
            (1024, 256),
        ]

    def forward(
        self,
        real_waveform,
        generated_waveform
    ):

        total_loss = 0.0

        for n_fft, hop_length in self.resolutions:

            window = torch.hann_window(
                n_fft,
                device=real_waveform.device
            )

            real_spec = torch.stft(
                real_waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True
            ).abs()

            generated_spec = torch.stft(
                generated_waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True
            ).abs()

            # Spectral convergence.
            spectral_convergence = (
                torch.norm(
                    real_spec - generated_spec,
                    p="fro"
                )
                /
                (
                    torch.norm(real_spec, p="fro")
                    + EPS
                )
            )

            # Log-magnitude loss.
            log_magnitude = F.l1_loss(
                torch.log(real_spec + EPS),
                torch.log(generated_spec + EPS)
            )

            total_loss += (
                spectral_convergence
                + log_magnitude
            )

        return total_loss / len(self.resolutions)


# ============================================================
# FEATURE MATCHING LOSS
# ============================================================

def feature_matching_loss(
    real_features,
    fake_features
):

    loss = 0.0

    for real_feature, fake_feature in zip(
        real_features,
        fake_features
    ):

        loss += F.l1_loss(
            fake_feature,
            real_feature.detach()
        )

    return loss / len(real_features)


# ============================================================
# TRAINING UTILITIES
# ============================================================

def save_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    epoch,
    path,
    mel_mean,
    mel_std
):

    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "mel_mean": mel_mean.detach().cpu(),
            "mel_std": mel_std.detach().cpu(),
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_fft": N_FFT,
            "n_mels": N_MELS,
        },
        path
    )


# ============================================================
# TRAINING
# ============================================================

def train(
    epochs=10,
    number_of_samples=128,
    batch_size=8,
    learning_rate=2e-4,
    no_plot=False
):

    print("\nDevice:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    dataset = SyntheticVocoderDataset(
        number_of_samples=number_of_samples
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    generator = NeuralVocoderGenerator().to(DEVICE)

    discriminator = WaveformDiscriminator().to(DEVICE)

    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.8, 0.99)
    )

    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.8, 0.99)
    )

    mrstft_loss = MultiResolutionSTFTLoss().to(DEVICE)

    history = {
        "generator": [],
        "discriminator": [],
        "mrstft": [],
        "adversarial": [],
        "feature_matching": [],
    }

    generator.train()
    discriminator.train()

    print("\nStarting GAN training...\n")

    for epoch in range(1, epochs + 1):

        epoch_g = 0.0
        epoch_d = 0.0
        epoch_mrstft = 0.0
        epoch_adv = 0.0
        epoch_fm = 0.0

        batches = 0

        start_time = time.time()

        for mel, real_waveform in loader:

            mel = mel.to(
                DEVICE,
                non_blocking=True
            )

            real_waveform = real_waveform.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # STEP 1 — GENERATE FAKE WAVEFORM
            # ------------------------------------------------

            generated_waveform = generator(mel)

            # Shape consistency check.
            if generated_waveform.shape != real_waveform.unsqueeze(1).shape:

                raise RuntimeError(
                    "Generator output shape mismatch: "
                    f"{generated_waveform.shape} vs "
                    f"{real_waveform.unsqueeze(1).shape}"
                )

            # ------------------------------------------------
            # STEP 2 — TRAIN DISCRIMINATOR
            # ------------------------------------------------

            optimizer_d.zero_grad(set_to_none=True)

            real_input = real_waveform.unsqueeze(1)

            real_prediction, _ = discriminator(
                real_input
            )

            fake_prediction, _ = discriminator(
                generated_waveform.detach()
            )

            loss_d_real = torch.mean(
                (real_prediction - 1.0) ** 2
            )

            loss_d_fake = torch.mean(
                fake_prediction ** 2
            )

            loss_d = 0.5 * (
                loss_d_real + loss_d_fake
            )

            loss_d.backward()

            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=10.0
            )

            optimizer_d.step()

            # ------------------------------------------------
            # STEP 3 — TRAIN GENERATOR
            # ------------------------------------------------

            optimizer_g.zero_grad(set_to_none=True)

            # Recalculate discriminator outputs after its update.
            # Discriminator parameters are not updated during this
            # generator backward pass.
            with torch.no_grad():

                _, real_features = discriminator(
                    real_input
                )

            fake_prediction, fake_features = discriminator(
                generated_waveform
            )

            # Adversarial loss.
            loss_adv = torch.mean(
                (fake_prediction - 1.0) ** 2
            )

            # Feature matching.
            loss_fm = feature_matching_loss(
                real_features,
                fake_features
            )

            # Multi-resolution spectral loss.
            loss_spec = mrstft_loss(
                real_waveform,
                generated_waveform.squeeze(1)
            )

            # Waveform reconstruction loss.
            loss_wave = F.l1_loss(
                generated_waveform.squeeze(1),
                real_waveform
            )

            # Combined generator objective.
            loss_g = (
                10.0 * loss_spec
                + 0.5 * loss_wave
                + 0.1 * loss_adv
                + 2.0 * loss_fm
            )

            loss_g.backward()

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=10.0
            )

            optimizer_g.step()

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            epoch_g += loss_g.item()
            epoch_d += loss_d.item()
            epoch_mrstft += loss_spec.item()
            epoch_adv += loss_adv.item()
            epoch_fm += loss_fm.item()

            batches += 1

        if batches == 0:

            raise RuntimeError(
                "No training batches. Increase --samples "
                "or reduce --batch-size."
            )

        epoch_g /= batches
        epoch_d /= batches
        epoch_mrstft /= batches
        epoch_adv /= batches
        epoch_fm /= batches

        history["generator"].append(epoch_g)
        history["discriminator"].append(epoch_d)
        history["mrstft"].append(epoch_mrstft)
        history["adversarial"].append(epoch_adv)
        history["feature_matching"].append(epoch_fm)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"G: {epoch_g:.4f} | "
            f"D: {epoch_d:.4f} | "
            f"MR-STFT: {epoch_mrstft:.4f} | "
            f"ADV: {epoch_adv:.4f} | "
            f"FM: {epoch_fm:.4f} | "
            f"{elapsed:.1f}s"
        )

        save_checkpoint(
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            epoch,
            "lesson16_vocoder_checkpoint.pt",
            dataset.mel_mean,
            dataset.mel_std
        )

    print("\nTraining finished.")

    # ========================================================
    # GENERATE AN EXAMPLE
    # ========================================================

    generator.eval()

    with torch.no_grad():

        # Use a held-out-style example from the synthetic dataset.
        # This is an evaluation example, not an independent real
        # speech recording.
        mel, real_waveform = dataset[
            min(len(dataset) - 1, len(dataset) // 2)
        ]

        mel = mel.unsqueeze(0).to(DEVICE)

        generated = generator(mel)

        generated = generated.squeeze().cpu().numpy()

        real = real_waveform.numpy()

    # Save generated waveform.
    generated = np.clip(
        generated,
        -1.0,
        1.0
    )

    sf.write(
        "lesson16_generated.wav",
        generated,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    print(
        "Generated WAV saved:",
        "lesson16_generated.wav"
    )

    # Save training curves.
    if not no_plot:

        plot_training_history(history)

        plot_waveform_comparison(
            real,
            generated,
            mel.squeeze(0).cpu().numpy()
        )

    return generator, discriminator, history


# ============================================================
# VISUALIZATION — TRAINING HISTORY
# ============================================================

def plot_training_history(history):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8)
    )

    axes[0, 0].plot(
        history["generator"],
        label="Generator"
    )

    axes[0, 0].set_title("Generator Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(
        history["discriminator"],
        label="Discriminator"
    )

    axes[0, 1].set_title("Discriminator Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)

    axes[1, 0].plot(
        history["mrstft"],
        label="Multi-resolution STFT"
    )

    axes[1, 0].set_title("Spectral Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        history["adversarial"],
        label="Adversarial"
    )

    axes[1, 1].plot(
        history["feature_matching"],
        label="Feature matching"
    )

    axes[1, 1].set_title("GAN Auxiliary Losses")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    plt.savefig(
        "lesson16_training_curves.png",
        dpi=150
    )

    plt.show()


# ============================================================
# VISUALIZATION — WAVEFORM AND MEL COMPARISON
# ============================================================

def plot_waveform_comparison(
    real,
    generated,
    mel
):

    duration = len(real) / SAMPLE_RATE

    times = np.arange(len(real)) / SAMPLE_RATE

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10)
    )

    axes[0].plot(
        times,
        real,
        linewidth=0.7
    )

    axes[0].set_title("Reference Synthetic Waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(
        times,
        generated[:len(real)],
        linewidth=0.7
    )

    axes[1].set_title("Generated Waveform")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    image = axes[2].imshow(
        mel,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )

    axes[2].set_title(
        "Input Normalized Log-Mel Spectrogram"
    )

    axes[2].set_xlabel("Mel Frame")
    axes[2].set_ylabel("Mel Channel")

    fig.colorbar(
        image,
        ax=axes[2],
        label="Normalized log-Mel"
    )

    plt.tight_layout()

    plt.savefig(
        "lesson16_waveform_comparison.png",
        dpi=150
    )

    plt.show()

    print(f"Audio duration: {duration:.3f} seconds")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train an educational neural vocoder."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=128
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
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

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.samples < 2:
        parser.error("--samples must be at least 2")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if args.samples < args.batch_size:
        parser.error(
            "--samples must be >= --batch-size"
        )

    if args.samples // args.batch_size < 1:
        parser.error("No complete training batches.")

    set_seed(42)

    train(
        epochs=args.epochs,
        number_of_samples=args.samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        no_plot=args.no_plot
    )


if __name__ == "__main__":
    main()