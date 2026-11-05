"""
======================================================================
PHASE 1 — LESSON 30
MEL-SPECTROGRAM -> WAVEFORM NEURAL VOCODER
======================================================================

PURPOSE
-------
Lesson 29 taught us how to produce a real log-mel spectrogram:

    speaker + content
            |
            v
      acoustic model
            |
            v
         log-mel

But a mel spectrogram is not directly playable as speech.

A TTS system ultimately needs:

    mel spectrogram
            |
            v
       neural vocoder
            |
            v
         waveform

This lesson builds a small neural vocoder from scratch.

We will train it on synthetic speech-like signals so that the entire
experiment remains self-contained.

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1
------
Create deterministic synthetic speech-like waveforms.

STEP 2
------
Convert waveform -> log-mel spectrogram.

STEP 3
------
Build a neural vocoder:

    log-mel
       |
       v
    Conv1D
       |
       v
    upsampling
       |
       v
    Conv1D
       |
       v
    waveform

STEP 4
------
Train using waveform reconstruction loss.

Basic objective:

    L_wave =
        mean(|y_hat - y|)

This directly encourages the generated waveform to resemble
the target waveform.

STEP 5
------
Add a spectral loss.

Waveform samples can be slightly misaligned while still sounding
similar.

Therefore we also compare their STFT magnitudes:

    L_spec =
        mean(
            |STFT(y_hat)| -
            |STFT(y)|
        )

Combined:

    L =
        lambda_wave * L_wave
        +
        lambda_spec * L_spec

This is much closer to how practical neural audio systems are
trained.

STEP 6
------
Evaluate:

    target waveform
           vs
    generated waveform

and:

    target mel
           vs
    generated mel

STEP 7
------
Understand the connection to modern TTS.

The full conceptual pipeline is now:

    text/content
          +
       speaker
          |
          v
    acoustic model
          |
          v
       log-mel
          |
          v
     neural vocoder
          |
          v
       waveform

----------------------------------------------------------------------
IMPORTANT
----------------------------------------------------------------------

This is an educational neural vocoder.

It is intentionally much smaller than production systems such as
HiFi-GAN, WaveRNN, WaveNet, etc.

The objective is to understand the architecture and learning
mechanism before implementing a more sophisticated vocoder.

----------------------------------------------------------------------
WHY MEL -> WAVEFORM IS HARD
----------------------------------------------------------------------

A mel spectrogram loses information.

For example, the magnitude spectrogram does not uniquely specify
the phase of the original waveform.

Therefore:

    waveform -> mel

is relatively straightforward.

But:

    mel -> waveform

is an inverse problem.

A neural vocoder learns a plausible waveform conditioned on the
acoustic representation.

That is why the vocoder is such an important part of TTS.

----------------------------------------------------------------------
REAL-WORLD EXAMPLE
----------------------------------------------------------------------

In a real voice-cloning system:

    reference speech
          |
          v
    speaker encoder
          |
          v
    speaker embedding
          |
          +----------------+
                           |
    text -> text encoder   |
                           v
                     acoustic model
                           |
                           v
                         mel
                           |
                           v
                     neural vocoder
                           |
                           v
                        speech

This lesson implements the final:

    mel -> speech

component from scratch.

======================================================================
"""

import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

SAMPLE_RATE = 16000

DURATION = 0.512
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

N_MELS = 80

F_MIN = 50.0
F_MAX = 7600.0

NUM_SPEAKERS = 8
NUM_CONTENTS = 16

TRAIN_SAMPLES = 256

BATCH_SIZE = 16

EPOCHS = 60

LEARNING_RATE = 2e-3

LAMBDA_WAVE = 1.0
LAMBDA_SPEC = 0.5

OUTPUT_DIR = "outputs/lesson30"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ======================================================================
# STEP 1
# SYNTHETIC SPEECH-LIKE WAVEFORM
# ======================================================================

def speaker_parameters(
    speaker_id,
):
    """
    Each speaker has a different acoustic identity.

    In a real system this information would come from actual speaker
    characteristics.
    """

    f0 = (
        100.0
        + 18.0 * speaker_id
    )

    brightness = (
        0.60
        + 0.045 * speaker_id
    )

    harmonic_strength = (
        0.40
        + 0.035 * (speaker_id % 5)
    )

    amplitude = (
        0.70
        + 0.025 * (speaker_id % 4)
    )

    return (
        f0,
        brightness,
        harmonic_strength,
        amplitude,
    )


def content_parameters(
    content_id,
):
    """
    Content controls the acoustic trajectory.
    """

    start_frequency = (
        160.0
        + 25.0 * (content_id % 8)
    )

    end_frequency = (
        start_frequency
        + 80.0
        + 10.0 * (content_id % 6)
    )

    modulation = (
        1.5
        + 0.15 * (content_id % 5)
    )

    return (
        start_frequency,
        end_frequency,
        modulation,
    )


def generate_waveform(
    speaker_id,
    content_id,
    noise_level=0.001,
):
    """
    Generate one synthetic speech-like waveform.

    Output:
        [NUM_SAMPLES]
    """

    (
        f0,
        brightness,
        harmonic_strength,
        amplitude,
    ) = speaker_parameters(
        speaker_id
    )

    (
        start_frequency,
        end_frequency,
        modulation,
    ) = content_parameters(
        content_id
    )

    t = (
        torch.arange(
            NUM_SAMPLES,
            dtype=torch.float32,
        )
        / SAMPLE_RATE
    )

    progress = t / DURATION

    content_frequency = (
        start_frequency
        + (
            end_frequency
            - start_frequency
        )
        * progress
    )

    content_frequency += (
        10.0
        * torch.sin(
            2.0
            * math.pi
            * modulation
            * t
        )
    )

    # Speaker controls the base pitch.

    instantaneous_frequency = (
        f0
        + 0.30 * content_frequency
    )

    phase = (
        2.0
        * math.pi
        * torch.cumsum(
            instantaneous_frequency,
            dim=0,
        )
        / SAMPLE_RATE
    )

    # Fundamental.

    waveform = torch.sin(
        phase
    )

    # Speaker-dependent harmonic structure.

    for harmonic in range(
        2,
        8,
    ):

        harmonic_amplitude = (
            harmonic_strength
            * brightness
            ** (harmonic - 1)
            / harmonic
        )

        waveform += (
            harmonic_amplitude
            * torch.sin(
                harmonic * phase
            )
        )

    # Content-dependent envelope.

    envelope = (
        0.65
        + 0.35
        * torch.sin(
            2.0
            * math.pi
            * (
                2.0
                + 0.15 * content_id
            )
            * t
        )
        ** 2
    )

    # Attack.

    attack = torch.clamp(
        t / 0.025,
        0.0,
        1.0,
    )

    # Release.

    release = torch.clamp(
        (DURATION - t) / 0.04,
        0.0,
        1.0,
    )

    envelope *= (
        attack
        * release
    )

    waveform *= (
        amplitude
        * envelope
    )

    waveform += (
        noise_level
        * torch.randn_like(
            waveform
        )
    )

    waveform /= (
        waveform.abs().max()
        + 1e-8
    )

    return waveform


# ======================================================================
# STEP 2
# STFT AND MEL FILTERBANK
# ======================================================================

def hz_to_mel(
    frequency,
):
    return (
        2595.0
        * torch.log10(
            1.0
            + frequency / 700.0
        )
    )


def mel_to_hz(
    mel,
):
    return (
        700.0
        * (
            10.0
            ** (mel / 2595.0)
            - 1.0
        )
    )


def create_mel_filterbank(
    sample_rate,
    n_fft,
    n_mels,
    f_min,
    f_max,
):
    """
    Create triangular Mel filters.
    """

    num_frequency_bins = (
        n_fft // 2 + 1
    )

    mel_min = hz_to_mel(
        torch.tensor(
            f_min
        )
    )

    mel_max = hz_to_mel(
        torch.tensor(
            f_max
        )
    )

    mel_points = torch.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    frequencies = (
        torch.arange(
            num_frequency_bins
        )
        * sample_rate
        / n_fft
    )

    filterbank = torch.zeros(
        n_mels,
        num_frequency_bins,
    )

    for m in range(
        1,
        n_mels + 1,
    ):

        left = hz_points[m - 1]
        center = hz_points[m]
        right = hz_points[m + 1]

        left_slope = (
            frequencies - left
        ) / (
            center
            - left
            + 1e-8
        )

        right_slope = (
            right - frequencies
        ) / (
            right
            - center
            + 1e-8
        )

        filterbank[m - 1] = (
            torch.clamp(
                torch.minimum(
                    left_slope,
                    right_slope,
                ),
                min=0.0,
            )
        )

    return filterbank


MEL_FILTERBANK = (
    create_mel_filterbank(
        SAMPLE_RATE,
        N_FFT,
        N_MELS,
        F_MIN,
        F_MAX,
    )
    .to(DEVICE)
)


WINDOW = torch.hann_window(
    WIN_LENGTH,
    device=DEVICE,
)


def waveform_to_logmel(
    waveform,
):
    """
    waveform:
        [samples]

    returns:
        [mel, time]
    """

    waveform = waveform.to(
        DEVICE
    )

    stft = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
        return_complex=True,
    )

    magnitude = stft.abs()

    mel = (
        MEL_FILTERBANK
        @ magnitude
    )

    log_mel = torch.log(
        mel + 1e-5
    )

    return log_mel


# ======================================================================
# DATASET
# ======================================================================

def build_dataset():
    """
    Generate waveform/mel pairs.

    The vocoder only needs:

        mel -> waveform

    It does NOT need speaker IDs or content IDs during inference.
    """

    waveforms = []
    mels = []

    for _ in range(
        TRAIN_SAMPLES
    ):

        speaker_id = random.randrange(
            NUM_SPEAKERS
        )

        content_id = random.randrange(
            NUM_CONTENTS
        )

        waveform = generate_waveform(
            speaker_id,
            content_id,
        )

        mel = waveform_to_logmel(
            waveform
        )

        waveforms.append(
            waveform
        )

        mels.append(
            mel.cpu()
        )

    waveforms = torch.stack(
        waveforms
    )

    mels = torch.stack(
        mels
    )

    return (
        waveforms,
        mels,
    )


# ======================================================================
# STEP 3
# NEURAL VOCODER
# ======================================================================

"""
The mel spectrogram has a much lower temporal resolution than the
waveform.

For our configuration:

    waveform:
        16000 samples / second

    mel:
        approximately 125 frames / second

Therefore the vocoder must increase temporal resolution.

Conceptually:

    [B, 80, T_mel]
             |
             v
       Conv1D
             |
             v
      Upsample x2
             |
             v
      Upsample x2
             |
             v
      Upsample x2
             |
             v
      Upsample x2
             |
             v
       Conv1D
             |
             v
      waveform

Because:

    2 x 2 x 2 x 2 = 16

and:

    128 * 16 = 2048

This does not exactly equal the STFT hop size, so the network will
generate a convenient intermediate waveform resolution and we will
resample/interpolate to the target waveform length.

A production vocoder would normally design the upsampling ratios
very carefully around the hop length.
"""


class ResidualBlock(
    nn.Module
):

    def __init__(
        self,
        channels,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return (
            x
            + self.block(x)
        )


class NeuralVocoder(
    nn.Module
):

    def __init__(
        self,
        n_mels,
    ):
        super().__init__()

        self.input = nn.Conv1d(
            n_mels,
            128,
            kernel_size=7,
            padding=3,
        )

        self.up1 = nn.ConvTranspose1d(
            128,
            96,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.up2 = nn.ConvTranspose1d(
            96,
            64,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.up3 = nn.ConvTranspose1d(
            64,
            48,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.res1 = ResidualBlock(
            96
        )

        self.res2 = ResidualBlock(
            64
        )

        self.res3 = ResidualBlock(
            48
        )

        self.output = nn.Sequential(

            nn.Conv1d(
                48,
                32,
                kernel_size=7,
                padding=3,
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Conv1d(
                32,
                1,
                kernel_size=7,
                padding=3,
            ),

            nn.Tanh(),
        )

    def forward(
        self,
        mel,
        target_length=None,
    ):
        """
        mel:

            [B, n_mels, T]

        waveform:

            [B, 1, samples]
        """

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

        x = self.res1(
            x
        )

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.up2(
            x
        )

        x = self.res2(
            x
        )

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.up3(
            x
        )

        x = self.res3(
            x
        )

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.output(
            x
        )

        if target_length is not None:

            x = F.interpolate(
                x,
                size=target_length,
                mode="linear",
                align_corners=False,
            )

        return x


# ======================================================================
# STEP 4
# WAVEFORM LOSS
# ======================================================================

def waveform_loss(
    prediction,
    target,
):
    """
    L1 waveform reconstruction loss.
    """

    return F.l1_loss(
        prediction,
        target,
    )


# ======================================================================
# STEP 5
# SPECTRAL LOSS
# ======================================================================

def spectral_loss(
    prediction,
    target,
):
    """
    Compare STFT magnitude.

    This gives the model a frequency-domain objective in addition
    to direct waveform reconstruction.
    """

    prediction = (
        prediction.squeeze(1)
    )

    target = (
        target.squeeze(1)
    )

    prediction_stft = torch.stft(
        prediction,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
        return_complex=True,
    )

    target_stft = torch.stft(
        target,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
        return_complex=True,
    )

    prediction_mag = (
        prediction_stft.abs()
    )

    target_mag = (
        target_stft.abs()
    )

    return F.l1_loss(
        prediction_mag,
        target_mag,
    )


# ======================================================================
# BATCHING
# ======================================================================

def get_batch(
    waveforms,
    mels,
    indices,
):
    """
    Return:

        mel:
            [B, 80, T]

        waveform:
            [B, 1, samples]
    """

    batch_waveforms = (
        waveforms[indices]
        .to(DEVICE)
    )

    batch_mels = (
        mels[indices]
        .to(DEVICE)
    )

    batch_waveforms = (
        batch_waveforms
        .unsqueeze(1)
    )

    return (
        batch_mels,
        batch_waveforms,
    )


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_results(
    target_waveform,
    generated_waveform,
    target_mel,
    generated_mel,
):
    """
    Plot target/generated waveform and mel representations.
    """

    target_waveform = (
        target_waveform
        .detach()
        .cpu()
        .numpy()
    )

    generated_waveform = (
        generated_waveform
        .detach()
        .cpu()
        .numpy()
    )

    target_mel = (
        target_mel
        .detach()
        .cpu()
        .numpy()
    )

    generated_mel = (
        generated_mel
        .detach()
        .cpu()
        .numpy()
    )

    time = (
        np.arange(
            len(target_waveform)
        )
        / SAMPLE_RATE
    )

    # --------------------------------------------------------------
    # Waveform
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
    )

    axes[0].plot(
        time,
        target_waveform,
    )

    axes[0].set_title(
        "Target Waveform"
    )

    axes[0].set_xlabel(
        "Time (seconds)"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[1].plot(
        time,
        generated_waveform,
    )

    axes[1].set_title(
        "Generated Waveform"
    )

    axes[1].set_xlabel(
        "Time (seconds)"
    )

    axes[1].set_ylabel(
        "Amplitude"
    )

    plt.tight_layout()

    waveform_path = os.path.join(
        OUTPUT_DIR,
        "target_vs_generated_waveform.png",
    )

    plt.savefig(
        waveform_path,
        dpi=150,
    )

    plt.show()

    # --------------------------------------------------------------
    # Mel
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
    )

    axes[0].imshow(
        target_mel,
        origin="lower",
        aspect="auto",
    )

    axes[0].set_title(
        "Target Log-Mel Spectrogram"
    )

    axes[0].set_ylabel(
        "Mel Bin"
    )

    axes[1].imshow(
        generated_mel,
        origin="lower",
        aspect="auto",
    )

    axes[1].set_title(
        "Generated Log-Mel Spectrogram"
    )

    axes[1].set_xlabel(
        "Time Frame"
    )

    axes[1].set_ylabel(
        "Mel Bin"
    )

    plt.tight_layout()

    mel_path = os.path.join(
        OUTPUT_DIR,
        "target_vs_generated_mel.png",
    )

    plt.savefig(
        mel_path,
        dpi=150,
    )

    plt.show()

    print()
    print(
        f"Saved: {waveform_path}"
    )

    print(
        f"Saved: {mel_path}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 72)
    print("PHASE 1 — LESSON 30")
    print("MEL-SPECTROGRAM -> WAVEFORM NEURAL VOCODER")
    print("=" * 72)

    print()
    print(
        f"Device:      {DEVICE}"
    )

    print(
        f"Sample rate: {SAMPLE_RATE}"
    )

    print(
        f"Duration:    {DURATION:.3f} sec"
    )

    print(
        f"Mel bins:    {N_MELS}"
    )

    print(
        f"FFT size:    {N_FFT}"
    )

    print(
        f"Hop length:  {HOP_LENGTH}"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    print()
    print(
        "Generating training dataset..."
    )

    waveforms, mels = (
        build_dataset()
    )

    print(
        f"Waveforms: {tuple(waveforms.shape)}"
    )

    print(
        f"Mel data:  {tuple(mels.shape)}"
    )

    num_frames = (
        mels.shape[-1]
    )

    print(
        f"Mel frames: {num_frames}"
    )

    # --------------------------------------------------------------
    # Vocoder
    # --------------------------------------------------------------

    vocoder = NeuralVocoder(
        N_MELS
    ).to(DEVICE)

    parameters = sum(
        parameter.numel()
        for parameter in vocoder.parameters()
    )

    print()
    print(
        f"Vocoder parameters: {parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        vocoder.parameters(),
        lr=LEARNING_RATE,
        betas=(0.8, 0.99),
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    losses = []
    waveform_losses = []
    spectral_losses = []

    print()
    print(
        "Training neural vocoder..."
    )
    print()

    dataset_size = len(
        waveforms
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        permutation = torch.randperm(
            dataset_size
        )

        epoch_total = 0.0
        epoch_wave = 0.0
        epoch_spec = 0.0

        for start in range(
            0,
            dataset_size,
            BATCH_SIZE,
        ):

            indices = permutation[
                start:
                start + BATCH_SIZE
            ]

            mel_batch, waveform_batch = (
                get_batch(
                    waveforms,
                    mels,
                    indices,
                )
            )

            generated = vocoder(
                mel_batch,
                target_length=NUM_SAMPLES,
            )

            loss_wave = (
                waveform_loss(
                    generated,
                    waveform_batch,
                )
            )

            loss_spec = (
                spectral_loss(
                    generated,
                    waveform_batch,
                )
            )

            loss = (
                LAMBDA_WAVE
                * loss_wave
                +
                LAMBDA_SPEC
                * loss_spec
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                vocoder.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = (
                len(indices)
            )

            epoch_total += (
                loss.item()
                * batch_size
            )

            epoch_wave += (
                loss_wave.item()
                * batch_size
            )

            epoch_spec += (
                loss_spec.item()
                * batch_size
            )

        epoch_total /= dataset_size
        epoch_wave /= dataset_size
        epoch_spec /= dataset_size

        losses.append(
            epoch_total
        )

        waveform_losses.append(
            epoch_wave
        )

        spectral_losses.append(
            epoch_spec
        )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Total {epoch_total:.6f} | "
                f"Wave {epoch_wave:.6f} | "
                f"Spec {epoch_spec:.6f}"
            )

    # --------------------------------------------------------------
    # Training curves
    # --------------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        losses,
        label="Total",
    )

    plt.plot(
        waveform_losses,
        label="Waveform",
    )

    plt.plot(
        spectral_losses,
        label="Spectral",
    )

    plt.title(
        "Neural Vocoder Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    loss_path = os.path.join(
        OUTPUT_DIR,
        "training_loss.png",
    )

    plt.savefig(
        loss_path,
        dpi=150,
    )

    plt.show()

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "EVALUATION"
    )

    print(
        "=" * 72
    )

    vocoder.eval()

    test_index = 0

    test_mel = (
        mels[
            test_index
        ]
        .unsqueeze(0)
        .to(DEVICE)
    )

    target_waveform = (
        waveforms[
            test_index
        ]
        .to(DEVICE)
    )

    with torch.no_grad():

        generated_waveform = (
            vocoder(
                test_mel,
                target_length=NUM_SAMPLES,
            )
            .squeeze(0)
            .squeeze(0)
        )

    # --------------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------------

    waveform_mae = F.l1_loss(
        generated_waveform,
        target_waveform,
    ).item()

    waveform_mse = F.mse_loss(
        generated_waveform,
        target_waveform,
    ).item()

    generated_mel = (
        waveform_to_logmel(
            generated_waveform
        )
    )

    mel_error = F.l1_loss(
        generated_mel,
        test_mel.squeeze(0),
    ).item()

    print()
    print(
        f"Waveform MAE: {waveform_mae:.6f}"
    )

    print(
        f"Waveform MSE: {waveform_mse:.6f}"
    )

    print(
        f"Mel reconstruction error: "
        f"{mel_error:.6f}"
    )

    # --------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------

    plot_results(
        target_waveform,
        generated_waveform,
        test_mel.squeeze(0),
        generated_mel,
    )

    # --------------------------------------------------------------
    # Save model
    # --------------------------------------------------------------

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        "neural_vocoder.pt",
    )

    torch.save(
        {
            "model_state_dict":
                vocoder.state_dict(),
            "sample_rate":
                SAMPLE_RATE,
            "n_fft":
                N_FFT,
            "hop_length":
                HOP_LENGTH,
            "n_mels":
                N_MELS,
        },
        checkpoint_path,
    )

    # --------------------------------------------------------------
    # Final explanation
    # --------------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "LESSON 30 COMPLETE"
    )

    print(
        "=" * 72
    )

    print()
    print(
        "You have now implemented:"
    )

    print()
    print(
        "  1. Waveform -> STFT"
    )

    print(
        "  2. STFT -> Mel"
    )

    print(
        "  3. Mel -> Neural Vocoder"
    )

    print(
        "  4. Waveform reconstruction loss"
    )

    print(
        "  5. Spectral reconstruction loss"
    )

    print(
        "  6. Joint waveform + spectral training"
    )

    print(
        "  7. Mel reconstruction evaluation"
    )

    print()
    print(
        "The learned pipeline is now:"
    )

    print()
    print(
        "        LOG-MEL"
    )

    print(
        "           |"
    )

    print(
        "           v"
    )

    print(
        "    NEURAL VOCODER"
    )

    print(
        "           |"
    )

    print(
        "           v"
    )

    print(
        "       WAVEFORM"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        f"  {checkpoint_path}"
    )

    print()
    print(
        "Next lesson:"
    )

    print(
        "  Improve the vocoder with multi-scale"
    )

    print(
        "  spectral analysis and adversarial learning."
    )

    print()
    print(
        "This moves us toward a HiFi-GAN-like"
    )

    print(
        "neural vocoder architecture."
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()