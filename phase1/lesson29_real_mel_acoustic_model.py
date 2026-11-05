"""
======================================================================
PHASE 1 — LESSON 29
REAL MEL-SPECTROGRAM ACOUSTIC REPRESENTATION
======================================================================

PURPOSE
-------
Lesson 28 introduced speaker-reference conditioning using synthetic
acoustic vectors.

This lesson moves one important step closer to a real TTS system:

    CONTENT + SPEAKER
          |
          v
    ACOUSTIC MODEL
          |
          v
    LOG-MEL SPECTROGRAM

The model now predicts an actual time-frequency acoustic representation
rather than an arbitrary low-dimensional acoustic vector.

This is the representation that later lessons can feed into a neural
vocoder.

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1
------
Generate synthetic speech-like waveforms.

Why:
    We need audio signals from which we can calculate real
    spectrograms, while keeping the lesson completely self-contained.

Real-world connection:
    A real TTS dataset contains waveform recordings such as:

        wav_0001.wav
        wav_0002.wav
        wav_0003.wav

    Here we generate them ourselves so the experiment is reproducible.

----------------------------------------------------------------------
STEP 2
------
Compute STFT.

For waveform x[n]:

    X(k, t) = STFT{x[n]}

The magnitude is:

    |X(k,t)|

This gives us a time-frequency representation.

Real-world connection:
    Speech recognition, TTS, source separation, music analysis,
    and vocoders all frequently work with time-frequency features.

----------------------------------------------------------------------
STEP 3
------
Construct a Mel filterbank.

Human hearing is approximately logarithmic in frequency.

The Mel scale is commonly defined as:

    mel(f) = 2595 log10(1 + f / 700)

We map linear-frequency STFT bins into mel-frequency bands.

----------------------------------------------------------------------
STEP 4
------
Compute log-mel spectrogram.

Instead of using raw mel energy:

    M

we use:

    log(M + epsilon)

This compresses the enormous dynamic range of speech energy.

This is one of the most important representations in traditional
and modern neural TTS pipelines.

----------------------------------------------------------------------
STEP 5
------
Build a speaker encoder.

Reference acoustic information is converted into:

    speaker_embedding

The purpose is to represent characteristics that should remain
relatively stable across utterances:

    speaker identity
    vocal characteristics
    timbre

----------------------------------------------------------------------
STEP 6
------
Build a content encoder.

A content ID is converted into:

    content_embedding

In a real system, this could eventually come from:

    text
       |
       v
    tokenizer
       |
       v
    text encoder

or a speech/content encoder.

For this lesson we use synthetic content IDs so that the experiment
remains focused on acoustic modeling.

----------------------------------------------------------------------
STEP 7
------
Train a conditional acoustic model.

The model learns:

    P(mel | content, speaker)

Conceptually:

    content embedding
             +
    speaker embedding
             |
             v
       acoustic network
             |
             v
        mel spectrogram

----------------------------------------------------------------------
STEP 8
------
Test speaker transfer.

Keep content fixed:

    content A

Change speaker:

    speaker 1 -> speaker 2

Expected:

    content remains similar
    speaker-dependent acoustic characteristics change

----------------------------------------------------------------------
STEP 9
------
Test content transfer.

Keep speaker fixed:

    speaker 1

Change content:

    content A -> content B

Expected:

    speaker characteristics remain similar
    acoustic content changes

----------------------------------------------------------------------
STEP 10
-------
Visualize real log-mel spectrograms.

This is important because eventually the neural vocoder will consume
this type of representation.

----------------------------------------------------------------------
WHY THIS MATTERS FOR CHATTERBOX-LIKE SYSTEMS
----------------------------------------------------------------------

A simplified modern voice cloning pipeline can be thought of as:

    reference speech
          |
          v
    speaker representation
          |
          +----------------+
                           |
                           v
    text/content ------> acoustic generation
                           |
                           v
                    mel/acoustic features
                           |
                           v
                       vocoder
                           |
                           v
                        waveform

This lesson implements the central acoustic representation.

We are NOT using Chatterbox itself.

We are building the concepts from scratch.

----------------------------------------------------------------------
IMPORTANT LIMITATION
----------------------------------------------------------------------

This is still an educational synthetic experiment.

The acoustic model is not yet a production TTS model.

The synthetic waveform generator gives us controlled examples where
speaker and content are deliberately separated.

Later lessons will progressively replace these components with:

    synthetic content
        ->
    real text/tokenization
        ->
    real speech data
        ->
    stronger speaker representations
        ->
    duration/alignment modeling
        ->
    autoregressive/non-autoregressive acoustic generation
        ->
    neural vocoder
        ->
    voice cloning

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

DURATION = 0.80
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

N_MELS = 80

F_MIN = 50.0
F_MAX = 7600.0

NUM_SPEAKERS = 6
NUM_CONTENTS = 12

TRAIN_SAMPLES_PER_PAIR = 4

EMBED_DIM = 32
HIDDEN_DIM = 128

BATCH_SIZE = 32
EPOCHS = 80

LEARNING_RATE = 2e-3

OUTPUT_DIR = "outputs/lesson29"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# SYNTHETIC SPEECH-LIKE WAVEFORM GENERATION
# ======================================================================

"""
We create a simplified speech-like signal.

A real voiced speech signal contains:

    fundamental frequency
    harmonics
    amplitude envelope
    phoneme-dependent spectral structure
    speaker-dependent characteristics

Our synthetic signal approximates some of these properties.

Each speaker gets different:

    fundamental frequency
    harmonic balance
    spectral tilt
    amplitude characteristics

Each content ID gets different:

    frequency pattern
    temporal modulation

Therefore:

    speaker = how the voice sounds

    content = what acoustic pattern is produced
"""


def speaker_parameters(speaker_id):
    """
    Return deterministic speaker-specific parameters.
    """

    base_f0 = 105.0 + 24.0 * speaker_id

    harmonic_strength = 0.45 + 0.07 * (speaker_id % 4)

    brightness = 0.65 + 0.06 * speaker_id

    amplitude = 0.65 + 0.04 * (speaker_id % 3)

    return {
        "f0": base_f0,
        "harmonic_strength": harmonic_strength,
        "brightness": brightness,
        "amplitude": amplitude,
    }


def content_parameters(content_id):
    """
    Return deterministic content-specific parameters.

    The content controls a slowly changing frequency pattern and
    amplitude envelope.
    """

    start_freq = 180.0 + 35.0 * (content_id % 6)

    end_freq = start_freq + 70.0 + 12.0 * (content_id % 5)

    modulation = 1.5 + 0.25 * (content_id % 4)

    return {
        "start_freq": start_freq,
        "end_freq": end_freq,
        "modulation": modulation,
    }


def generate_waveform(speaker_id, content_id, noise_level=0.002):
    """
    Generate one synthetic speech-like waveform.

    Returns:
        waveform: [NUM_SAMPLES]
    """

    speaker = speaker_parameters(speaker_id)
    content = content_parameters(content_id)

    t = torch.arange(
        NUM_SAMPLES,
        dtype=torch.float32,
    ) / SAMPLE_RATE

    # --------------------------------------------------------------
    # Content-dependent frequency trajectory
    # --------------------------------------------------------------

    progress = t / DURATION

    frequency = (
        content["start_freq"]
        + (content["end_freq"] - content["start_freq"]) * progress
    )

    frequency += (
        8.0
        * torch.sin(
            2.0
            * math.pi
            * content["modulation"]
            * t
        )
    )

    phase = 2.0 * math.pi * torch.cumsum(
        frequency,
        dim=0,
    ) / SAMPLE_RATE

    # --------------------------------------------------------------
    # Fundamental
    # --------------------------------------------------------------

    waveform = torch.sin(phase)

    # --------------------------------------------------------------
    # Speaker-specific harmonic structure
    # --------------------------------------------------------------

    for harmonic in range(2, 7):

        harmonic_amplitude = (
            speaker["harmonic_strength"]
            * speaker["brightness"] ** (harmonic - 1)
            / harmonic
        )

        waveform += (
            harmonic_amplitude
            * torch.sin(harmonic * phase)
        )

    # --------------------------------------------------------------
    # Speech-like amplitude envelope
    # --------------------------------------------------------------

    envelope = (
        0.65
        + 0.35
        * torch.sin(
            2.0
            * math.pi
            * 2.5
            * t
            + content_id * 0.4
        ) ** 2
    )

    # Add attack and decay so the waveform is not perfectly stationary.

    attack = torch.clamp(
        t / 0.08,
        0.0,
        1.0,
    )

    decay = torch.clamp(
        (DURATION - t) / 0.08,
        0.0,
        1.0,
    )

    envelope = envelope * attack * decay

    waveform = (
        speaker["amplitude"]
        * envelope
        * waveform
    )

    # --------------------------------------------------------------
    # Small noise component
    # --------------------------------------------------------------

    waveform += noise_level * torch.randn_like(waveform)

    # Normalize.

    waveform = waveform / (
        waveform.abs().max() + 1e-8
    )

    return waveform


# ======================================================================
# STEP 2 + STEP 3
# STFT + MEL FILTERBANK
# ======================================================================

"""
We deliberately implement the mel conversion ourselves.

This avoids dependency on torchaudio and therefore avoids the
libtorchcodec/FFmpeg issue encountered in previous lessons.

Pipeline:

    waveform
       |
       v
      STFT
       |
       v
    magnitude
       |
       v
   mel filterbank
       |
       v
    mel energy
       |
       v
   logarithm
       |
       v
   log-mel
"""


def hz_to_mel(freq):
    """
    Convert frequency in Hz to Mel.
    """

    return 2595.0 * torch.log10(
        1.0 + freq / 700.0
    )


def mel_to_hz(mel):
    """
    Convert Mel to Hz.
    """

    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def create_mel_filterbank(
    sample_rate,
    n_fft,
    n_mels,
    f_min,
    f_max,
):
    """
    Construct a triangular Mel filterbank.

    Returns:
        [n_mels, n_fft // 2 + 1]
    """

    num_frequency_bins = n_fft // 2 + 1

    mel_min = hz_to_mel(
        torch.tensor(f_min)
    )

    mel_max = hz_to_mel(
        torch.tensor(f_max)
    )

    mel_points = torch.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
    )

    hz_points = mel_to_hz(mel_points)

    fft_frequencies = (
        torch.arange(num_frequency_bins)
        * sample_rate
        / n_fft
    )

    filterbank = torch.zeros(
        n_mels,
        num_frequency_bins,
    )

    for m in range(1, n_mels + 1):

        left = hz_points[m - 1]
        center = hz_points[m]
        right = hz_points[m + 1]

        left_slope = (
            fft_frequencies - left
        ) / (center - left + 1e-8)

        right_slope = (
            right - fft_frequencies
        ) / (right - center + 1e-8)

        filterbank[m - 1] = torch.clamp(
            torch.minimum(
                left_slope,
                right_slope,
            ),
            min=0.0,
        )

    return filterbank


MEL_FILTERBANK = create_mel_filterbank(
    SAMPLE_RATE,
    N_FFT,
    N_MELS,
    F_MIN,
    F_MAX,
).to(DEVICE)


WINDOW = torch.hann_window(
    WIN_LENGTH,
    device=DEVICE,
)


def waveform_to_logmel(waveform):
    """
    Convert waveform to log-mel spectrogram.

    Input:
        waveform: [samples]

    Output:
        log_mel: [n_mels, time]
    """

    waveform = waveform.to(DEVICE)

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

    mel = MEL_FILTERBANK @ magnitude

    log_mel = torch.log(
        mel + 1e-5
    )

    return log_mel


# ======================================================================
# DATASET CREATION
# ======================================================================

"""
Each training example contains:

    speaker_id
    content_id
    log_mel

The acoustic target is therefore a real spectrogram.

Unlike Lesson 28, the output is now:

    [80, time]

instead of a small arbitrary acoustic vector.
"""


def build_dataset():
    speaker_ids = []
    content_ids = []
    mel_targets = []
    waveforms = []

    for speaker_id in range(NUM_SPEAKERS):

        for content_id in range(NUM_CONTENTS):

            for _ in range(
                TRAIN_SAMPLES_PER_PAIR
            ):

                waveform = generate_waveform(
                    speaker_id,
                    content_id,
                )

                log_mel = waveform_to_logmel(
                    waveform
                )

                speaker_ids.append(
                    speaker_id
                )

                content_ids.append(
                    content_id
                )

                mel_targets.append(
                    log_mel.cpu()
                )

                waveforms.append(
                    waveform
                )

    speaker_ids = torch.tensor(
        speaker_ids,
        dtype=torch.long,
    )

    content_ids = torch.tensor(
        content_ids,
        dtype=torch.long,
    )

    mel_targets = torch.stack(
        mel_targets
    )

    waveforms = torch.stack(
        waveforms
    )

    return (
        speaker_ids,
        content_ids,
        mel_targets,
        waveforms,
    )


# ======================================================================
# STEP 5
# SPEAKER ENCODER
# ======================================================================

class SpeakerEncoder(nn.Module):

    def __init__(
        self,
        num_speakers,
        embedding_dim,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_speakers,
            embedding_dim,
        )

        self.projection = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.Tanh(),
        )

    def forward(self, speaker_id):

        x = self.embedding(
            speaker_id
        )

        return self.projection(x)


# ======================================================================
# STEP 6
# CONTENT ENCODER
# ======================================================================

class ContentEncoder(nn.Module):

    def __init__(
        self,
        num_contents,
        embedding_dim,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_contents,
            embedding_dim,
        )

        self.projection = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.Tanh(),
        )

    def forward(self, content_id):

        x = self.embedding(
            content_id
        )

        return self.projection(x)


# ======================================================================
# STEP 7
# CONDITIONAL ACOUSTIC MODEL
# ======================================================================

class AcousticModel(nn.Module):

    def __init__(
        self,
        speaker_dim,
        content_dim,
        hidden_dim,
        n_mels,
        num_frames,
    ):
        super().__init__()

        input_dim = (
            speaker_dim
            + content_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                n_mels * num_frames,
            ),
        )

        self.n_mels = n_mels
        self.num_frames = num_frames

    def forward(
        self,
        speaker_embedding,
        content_embedding,
    ):

        x = torch.cat(
            [
                speaker_embedding,
                content_embedding,
            ],
            dim=-1,
        )

        x = self.network(x)

        x = x.view(
            -1,
            self.n_mels,
            self.num_frames,
        )

        return x


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 72)
    print("PHASE 1 — LESSON 29")
    print("REAL MEL-SPECTROGRAM ACOUSTIC REPRESENTATION")
    print("=" * 72)

    print()
    print(f"Device:       {DEVICE}")
    print(f"Sample rate:  {SAMPLE_RATE}")
    print(f"N_FFT:        {N_FFT}")
    print(f"Hop length:   {HOP_LENGTH}")
    print(f"Mel bins:     {N_MELS}")
    print(f"Speakers:     {NUM_SPEAKERS}")
    print(f"Contents:     {NUM_CONTENTS}")
    print()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------------

    print("Generating synthetic speech-like dataset...")

    (
        speaker_ids,
        content_ids,
        mel_targets,
        waveforms,
    ) = build_dataset()

    print(
        f"Dataset examples: {len(speaker_ids)}"
    )

    print(
        f"Waveform shape:    {tuple(waveforms.shape)}"
    )

    print(
        f"Mel shape:         {tuple(mel_targets.shape)}"
    )

    num_frames = mel_targets.shape[-1]

    print(
        f"Mel frames:        {num_frames}"
    )

    # --------------------------------------------------------------
    # Visualize waveform and mel representation
    # --------------------------------------------------------------

    example_waveform = waveforms[0]

    example_mel = mel_targets[0]

    plt.figure(figsize=(12, 4))

    time_axis = (
        torch.arange(
            len(example_waveform)
        ).numpy()
        / SAMPLE_RATE
    )

    plt.plot(
        time_axis,
        example_waveform.numpy(),
    )

    plt.title(
        "Synthetic Speech-Like Waveform"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    waveform_path = os.path.join(
        OUTPUT_DIR,
        "waveform.png",
    )

    plt.savefig(
        waveform_path,
        dpi=150,
    )

    plt.show()

    # --------------------------------------------------------------
    # Mel visualization
    # --------------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.imshow(
        example_mel.numpy(),
        origin="lower",
        aspect="auto",
    )

    plt.title(
        "Log-Mel Spectrogram"
    )

    plt.xlabel("Time Frame")
    plt.ylabel("Mel Bin")

    plt.colorbar(
        label="Log Energy"
    )

    plt.tight_layout()

    mel_path = os.path.join(
        OUTPUT_DIR,
        "log_mel.png",
    )

    plt.savefig(
        mel_path,
        dpi=150,
    )

    plt.show()

    print()
    print("Saved:")
    print(f"  {waveform_path}")
    print(f"  {mel_path}")

    # --------------------------------------------------------------
    # Move data to device
    # --------------------------------------------------------------

    speaker_ids = speaker_ids.to(DEVICE)
    content_ids = content_ids.to(DEVICE)
    mel_targets = mel_targets.to(DEVICE)

    # --------------------------------------------------------------
    # Models
    # --------------------------------------------------------------

    speaker_encoder = SpeakerEncoder(
        NUM_SPEAKERS,
        EMBED_DIM,
    ).to(DEVICE)

    content_encoder = ContentEncoder(
        NUM_CONTENTS,
        EMBED_DIM,
    ).to(DEVICE)

    acoustic_model = AcousticModel(
        EMBED_DIM,
        EMBED_DIM,
        HIDDEN_DIM,
        N_MELS,
        num_frames,
    ).to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in (
            list(speaker_encoder.parameters())
            + list(content_encoder.parameters())
            + list(acoustic_model.parameters())
        )
    )

    print()
    print(
        f"Trainable parameters: {total_parameters:,}"
    )

    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        list(speaker_encoder.parameters())
        + list(content_encoder.parameters())
        + list(acoustic_model.parameters()),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------------

    print()
    print("Training acoustic model...")
    print()

    dataset_size = len(
        speaker_ids
    )

    losses = []

    for epoch in range(1, EPOCHS + 1):

        permutation = torch.randperm(
            dataset_size,
            device=DEVICE,
        )

        epoch_loss = 0.0

        for start in range(
            0,
            dataset_size,
            BATCH_SIZE,
        ):

            indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch_speakers = speaker_ids[
                indices
            ]

            batch_contents = content_ids[
                indices
            ]

            batch_targets = mel_targets[
                indices
            ]

            speaker_embedding = (
                speaker_encoder(
                    batch_speakers
                )
            )

            content_embedding = (
                content_encoder(
                    batch_contents
                )
            )

            prediction = acoustic_model(
                speaker_embedding,
                content_embedding,
            )

            loss = F.mse_loss(
                prediction,
                batch_targets,
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(
                    speaker_encoder.parameters()
                )
                + list(
                    content_encoder.parameters()
                )
                + list(
                    acoustic_model.parameters()
                ),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_loss += (
                loss.item()
                * len(indices)
            )

        epoch_loss /= dataset_size

        losses.append(
            epoch_loss
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {epoch_loss:.6f}"
            )

    # --------------------------------------------------------------
    # Training curve
    # --------------------------------------------------------------

    plt.figure(figsize=(9, 4))

    plt.plot(
        losses
    )

    plt.title(
        "Acoustic Model Training Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

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
    # STEP 8
    # SPEAKER TRANSFER
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("SPEAKER TRANSFER EXPERIMENT")
    print("=" * 72)

    acoustic_model.eval()
    speaker_encoder.eval()
    content_encoder.eval()

    test_content = 3

    speaker_a = 0
    speaker_b = 4

    with torch.no_grad():

        content_embedding = (
            content_encoder(
                torch.tensor(
                    [test_content],
                    device=DEVICE,
                )
            )
        )

        speaker_embedding_a = (
            speaker_encoder(
                torch.tensor(
                    [speaker_a],
                    device=DEVICE,
                )
            )
        )

        speaker_embedding_b = (
            speaker_encoder(
                torch.tensor(
                    [speaker_b],
                    device=DEVICE,
                )
            )
        )

        prediction_a = acoustic_model(
            speaker_embedding_a,
            content_embedding,
        )[0]

        prediction_b = acoustic_model(
            speaker_embedding_b,
            content_embedding,
        )[0]

    print(
        f"Content: {test_content}"
    )

    print(
        f"Speaker A: {speaker_a}"
    )

    print(
        f"Speaker B: {speaker_b}"
    )

    speaker_difference = F.mse_loss(
        prediction_a,
        prediction_b,
    ).item()

    print(
        "Mel difference between speakers: "
        f"{speaker_difference:.6f}"
    )

    # --------------------------------------------------------------
    # Visualize speaker transfer
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
    )

    axes[0].imshow(
        prediction_a.cpu().numpy(),
        origin="lower",
        aspect="auto",
    )

    axes[0].set_title(
        f"Same Content {test_content} — Speaker {speaker_a}"
    )

    axes[0].set_ylabel(
        "Mel Bin"
    )

    axes[1].imshow(
        prediction_b.cpu().numpy(),
        origin="lower",
        aspect="auto",
    )

    axes[1].set_title(
        f"Same Content {test_content} — Speaker {speaker_b}"
    )

    axes[1].set_xlabel(
        "Time Frame"
    )

    axes[1].set_ylabel(
        "Mel Bin"
    )

    plt.tight_layout()

    speaker_transfer_path = os.path.join(
        OUTPUT_DIR,
        "speaker_transfer.png",
    )

    plt.savefig(
        speaker_transfer_path,
        dpi=150,
    )

    plt.show()

    # --------------------------------------------------------------
    # STEP 9
    # CONTENT TRANSFER
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("CONTENT TRANSFER EXPERIMENT")
    print("=" * 72)

    fixed_speaker = 2

    content_a = 1
    content_b = 9

    with torch.no_grad():

        speaker_embedding = (
            speaker_encoder(
                torch.tensor(
                    [fixed_speaker],
                    device=DEVICE,
                )
            )
        )

        content_embedding_a = (
            content_encoder(
                torch.tensor(
                    [content_a],
                    device=DEVICE,
                )
            )
        )

        content_embedding_b = (
            content_encoder(
                torch.tensor(
                    [content_b],
                    device=DEVICE,
                )
            )
        )

        prediction_content_a = acoustic_model(
            speaker_embedding,
            content_embedding_a,
        )[0]

        prediction_content_b = acoustic_model(
            speaker_embedding,
            content_embedding_b,
        )[0]

    content_difference = F.mse_loss(
        prediction_content_a,
        prediction_content_b,
    ).item()

    print(
        f"Speaker: {fixed_speaker}"
    )

    print(
        f"Content A: {content_a}"
    )

    print(
        f"Content B: {content_b}"
    )

    print(
        "Mel difference between contents: "
        f"{content_difference:.6f}"
    )

    # --------------------------------------------------------------
    # Visualize content transfer
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
    )

    axes[0].imshow(
        prediction_content_a.cpu().numpy(),
        origin="lower",
        aspect="auto",
    )

    axes[0].set_title(
        f"Speaker {fixed_speaker} — Content {content_a}"
    )

    axes[0].set_ylabel(
        "Mel Bin"
    )

    axes[1].imshow(
        prediction_content_b.cpu().numpy(),
        origin="lower",
        aspect="auto",
    )

    axes[1].set_title(
        f"Speaker {fixed_speaker} — Content {content_b}"
    )

    axes[1].set_xlabel(
        "Time Frame"
    )

    axes[1].set_ylabel(
        "Mel Bin"
    )

    plt.tight_layout()

    content_transfer_path = os.path.join(
        OUTPUT_DIR,
        "content_transfer.png",
    )

    plt.savefig(
        content_transfer_path,
        dpi=150,
    )

    plt.show()

    # --------------------------------------------------------------
    # STEP 10
    # SUMMARY
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("LESSON 29 COMPLETE")
    print("=" * 72)

    print()
    print("You have now implemented:")
    print()
    print("  1. Synthetic speech-like waveforms")
    print("  2. STFT")
    print("  3. Mel filterbank")
    print("  4. Log-mel spectrogram")
    print("  5. Speaker embedding")
    print("  6. Content embedding")
    print("  7. Conditional acoustic model")
    print("  8. Speaker transfer")
    print("  9. Content transfer")
    print()

    print("Core learned mapping:")
    print()
    print("    speaker + content")
    print("          |")
    print("          v")
    print("    acoustic model")
    print("          |")
    print("          v")
    print("      log-mel")
    print()

    print("Next conceptual step:")
    print()
    print("    LOG-MEL")
    print("       |")
    print("       v")
    print("    NEURAL VOCODER")
    print("       |")
    print("       v")
    print("    WAVEFORM")
    print()

    print("Output directory:")
    print(f"    {OUTPUT_DIR}")

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()