"""
File: phase1_lesson17_speaker_identity.py

PHASE 1 — LESSON 17
SPEAKER IDENTITY REPRESENTATION AND MULTI-SPEAKER TTS
=====================================================

PURPOSE
-------
Understand and implement speaker embeddings from scratch.

This experiment includes:

    1. Synthetic multi-speaker audio generation
    2. Speaker-dependent acoustic characteristics
    3. Log-Mel feature extraction
    4. CNN speaker encoder
    5. L2-normalized speaker embeddings
    6. Speaker classification training
    7. Supervised contrastive learning
    8. Cosine similarity evaluation
    9. PCA embedding visualization
   10. Similarity matrix visualization
   11. Saving and loading the trained model
   12. Extracting an embedding from a WAV recording

IMPORTANT
---------
The audio is synthetic and deliberately simplified.

This is an educational speaker-embedding experiment, NOT a
production speaker-recognition model and NOT a voice-cloning system.

A real speaker encoder must learn identity from diverse real
recordings while remaining robust to:

    - Different sentences
    - Different microphones
    - Recording environments
    - Speaking volume
    - Emotional state
    - Speaking rate
    - Phonetic content

The model in this lesson is trained from scratch.
No pretrained speaker encoder is used.

RUN
---
    python phase1_lesson17_speaker_identity.py

Optional:
    python phase1_lesson17_speaker_identity.py --epochs 30
    python phase1_lesson17_speaker_identity.py --speakers 16
    python phase1_lesson17_speaker_identity.py --utterances 20
    python phase1_lesson17_speaker_identity.py --audio my_voice.wav
    python phase1_lesson17_speaker_identity.py --no-plot

DEPENDENCIES
------------
    pip install numpy matplotlib soundfile torch

OUTPUT
------
    lesson17_speaker_encoder.pt
    lesson17_speaker_embeddings.png
    lesson17_similarity_matrix.png
    lesson17_training_curves.png


============================================================
THEORY 1 — SPEAKER IDENTITY
============================================================

An audio recording contains several types of information:

    x = f(content, speaker, prosody, channel, environment)

Where:

    content      = linguistic information
    speaker      = speaker identity
    prosody      = pitch, rhythm, stress, intonation
    channel      = microphone and recording equipment
    environment  = room acoustics and background noise

A speaker encoder attempts to extract a representation that
preserves speaker information while being less sensitive to
the other factors.

This is difficult because speaker identity and linguistic
content are mixed together in the waveform.


============================================================
THEORY 2 — SPEAKER EMBEDDING
============================================================

A speaker encoder is a neural function:

    s = E(x)

Where:

    x = audio or acoustic features
    E = speaker encoder
    s = speaker embedding

We normalize the vector:

    s_normalized = s / ||s||

For two normalized embeddings:

    cosine_similarity = s1 dot s2

Interpretation:

    Near +1:
        Similar embedding directions.

    Near 0:
        Approximately orthogonal directions.

    Near -1:
        Opposite directions.

These values are not universal speaker-identity thresholds.
A real system needs validation data and threshold calibration.


============================================================
THEORY 3 — SPEAKER CLASSIFICATION
============================================================

During training, each speaker has a class ID.

For example:

    Speaker A -> 0
    Speaker B -> 1
    Speaker C -> 2

The encoder creates an embedding:

    s = E(x)

A classification head predicts:

    p(speaker | x) = softmax(Ws + b)

Cross-entropy loss:

    L = -log(p(correct_speaker | x))

The classifier is a training mechanism.

After training, the embedding is often more useful than the
classification probabilities.

A closed-set classifier alone cannot guarantee that its
embedding generalizes to unseen speakers.


============================================================
THEORY 4 — SUPERVISED CONTRASTIVE LEARNING
============================================================

Classification teaches the network to predict speaker IDs.

Contrastive learning directly teaches relationships between
recordings.

For an anchor sample i:

    Positive samples:
        Other recordings from the same speaker.

    Negative samples:
        Recordings from different speakers.

A simplified supervised contrastive objective is:

    L_i =
        -1 / |P(i)|
        * sum over p in P(i):
            log(
                exp(sim(z_i,z_p)/temperature)
                /
                sum over a != i:
                    exp(sim(z_i,z_a)/temperature)
            )

Here:

    z_i = normalized embedding
    P(i) = positive sample indices
    temperature = scaling factor

This implementation combines speaker classification and
supervised contrastive learning.

Balanced batches are important: each training batch should
contain multiple recordings from each selected speaker.


============================================================
THEORY 5 — SPEAKER CONDITIONING IN TTS
============================================================

Once a speaker embedding is available, a TTS acoustic model
can be conditioned on it.

    text_features = TextEncoder(text)

    speaker_vector = SpeakerEncoder(reference_audio)

    mel = AcousticDecoder(
        text_features,
        speaker_vector
    )

The speaker vector can be injected by:

    1. Concatenation
    2. Addition after projection
    3. Feature-wise linear modulation (FiLM)
    4. Adaptive normalization
    5. Cross-attention

This lesson trains the speaker encoder separately.
The next stage will connect speaker information to acoustic
speech generation.


============================================================
THEORY 6 — WHY REAL VOICE CLONING IS HARDER
============================================================

A real voice-cloning system must handle:

    - Unseen speakers
    - Short reference recordings
    - Different phonetic content
    - Accent and pronunciation
    - Emotion and speaking style
    - Background noise
    - Microphone differences

The synthetic data here uses stable, deliberately distinct
speaker profiles. Real-world speaker identity is much more
subtle.

Do not interpret high synthetic validation accuracy as
evidence of real-world voice-cloning performance.
"""


import argparse
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Sampler


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64

AUDIO_SECONDS = 1.28
AUDIO_SAMPLES = int(SAMPLE_RATE * AUDIO_SECONDS)

EMBEDDING_DIM = 128

EPS = 1e-8

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


def create_mel_filterbank():

    fmin = 40.0
    fmax = SAMPLE_RATE / 2

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        N_MELS + 2
    )

    hz_points = mel_to_hz(mel_points)

    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1
    )

    filters = np.zeros(
        (N_MELS, len(frequencies)),
        dtype=np.float32
    )

    for index in range(N_MELS):

        left = hz_points[index]
        center = hz_points[index + 1]
        right = hz_points[index + 2]

        rising = (
            (frequencies - left)
            / max(center - left, EPS)
        )

        falling = (
            (right - frequencies)
            / max(right - center, EPS)
        )

        filters[index] = np.maximum(
            0.0,
            np.minimum(rising, falling)
        )

    return torch.tensor(
        filters,
        dtype=torch.float32
    )


MEL_FILTERBANK = create_mel_filterbank()


# ============================================================
# SYNTHETIC SPEAKER PROFILES
# ============================================================

def create_speaker_profiles(
    number_of_speakers,
    seed=123
):
    """
    Each synthetic speaker has a stable acoustic profile.

    Parameters:
        base_f0:
            Approximate fundamental frequency.

        formants:
            Simplified resonance frequencies.

        bandwidths:
            Width of the resonance peaks.

        spectral_tilt:
            Controls the relative strength of high harmonics.

        nasal_strength:
            Adds a simplified speaker-dependent spectral feature.

    These parameters are deliberately simplified. They are not
    a physically accurate model of human vocal production.
    """

    rng = np.random.default_rng(seed)

    profiles = []

    for speaker_id in range(number_of_speakers):

        profile = {
            "speaker_id": speaker_id,

            "base_f0": rng.uniform(95.0, 220.0),

            "formants": np.array([
                rng.uniform(450, 850),
                rng.uniform(1050, 1900),
                rng.uniform(2100, 3300),
            ]),

            "bandwidths": np.array([
                rng.uniform(90, 190),
                rng.uniform(130, 300),
                rng.uniform(180, 420),
            ]),

            "spectral_tilt": rng.uniform(0.8, 1.5),

            "nasal_strength": rng.uniform(0.0, 0.25),

            "amplitude": rng.uniform(0.6, 1.0),
        }

        profiles.append(profile)

    return profiles


# ============================================================
# SYNTHETIC AUDIO GENERATION
# ============================================================

def generate_speaker_utterance(
    profile,
    rng
):
    """
    Generate one synthetic recording for a speaker.

    The speaker profile stays relatively stable, while each
    utterance has variations in:

        - F0
        - Vibrato
        - Amplitude envelope
        - Noise
        - Timing

    This creates multiple recordings per speaker.
    """

    length = AUDIO_SAMPLES

    time_axis = (
        np.arange(length, dtype=np.float32)
        / SAMPLE_RATE
    )

    base_f0 = profile["base_f0"]

    # Utterance-level pitch variation.
    utterance_f0 = base_f0 * rng.uniform(
        0.90,
        1.10
    )

    vibrato_rate = rng.uniform(2.0, 5.0)
    vibrato_depth = rng.uniform(0.002, 0.012)

    f0_curve = (
        utterance_f0
        * (
            1.0
            + vibrato_depth
            * np.sin(
                2.0 * np.pi * vibrato_rate * time_axis
            )
        )
    )

    # Add a small changing intonation contour.
    contour_rate = rng.uniform(0.3, 1.2)

    f0_curve *= (
        1.0
        + 0.025
        * np.sin(
            2.0 * np.pi * contour_rate * time_axis
        )
    )

    phase = (
        2.0
        * np.pi
        * np.cumsum(f0_curve)
        / SAMPLE_RATE
    )

    signal = np.zeros(
        length,
        dtype=np.float32
    )

    formants = profile["formants"]
    bandwidths = profile["bandwidths"]

    max_harmonics = min(
        80,
        int((SAMPLE_RATE / 2) / max(utterance_f0, 1.0))
    )

    for harmonic in range(1, max_harmonics + 1):

        frequency = harmonic * utterance_f0

        if frequency >= SAMPLE_RATE / 2:
            break

        spectral_envelope = 0.04

        for center, bandwidth in zip(
            formants,
            bandwidths
        ):

            spectral_envelope += np.exp(
                -0.5
                * ((frequency - center) / bandwidth) ** 2
            )

        # Speaker-dependent spectral tilt.
        tilt = (
            1.0
            / (harmonic ** profile["spectral_tilt"])
        )

        amplitude = spectral_envelope * tilt

        # Small speaker-dependent harmonic variation.
        amplitude *= (
            1.0
            + 0.04
            * np.sin(
                harmonic * profile["nasal_strength"]
            )
        )

        signal += (
            amplitude
            * np.sin(harmonic * phase)
        ).astype(np.float32)

    # Utterance-specific amplitude modulation.
    modulation_rate = rng.uniform(1.0, 3.0)

    envelope = (
        0.65
        + 0.22
        * np.sin(
            2.0 * np.pi * modulation_rate * time_axis
        )
        + 0.08
        * np.sin(
            2.0 * np.pi * 0.5 * time_axis
        )
    )

    envelope = np.clip(
        envelope,
        0.2,
        1.0
    )

    signal *= envelope.astype(np.float32)

    signal *= profile["amplitude"]

    # Random recording noise.
    noise_level = rng.uniform(
        0.002,
        0.025
    )

    signal += rng.normal(
        0.0,
        noise_level,
        size=length
    ).astype(np.float32)

    # Random gain.
    signal *= rng.uniform(
        0.65,
        1.0
    )

    peak = np.max(np.abs(signal)) + EPS

    signal = signal / peak * 0.9

    return signal.astype(np.float32)


# ============================================================
# AUDIO -> LOG-MEL FEATURES
# ============================================================

def waveform_to_logmel(waveform):
    """
    waveform:
        (batch, samples)

    output:
        (batch, mel_channels, frames)
    """

    window = torch.hann_window(
        N_FFT,
        device=waveform.device
    )

    spectrum = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True
    )

    power = spectrum.abs().pow(2)

    mel_filterbank = MEL_FILTERBANK.to(
        waveform.device
    )

    mel = torch.matmul(
        mel_filterbank,
        power
    )

    return torch.log(
        mel + EPS
    )


# ============================================================
# DATASET
# ============================================================

class SpeakerDataset(Dataset):

    def __init__(
        self,
        number_of_speakers=12,
        utterances_per_speaker=16,
        seed=42
    ):

        self.profiles = create_speaker_profiles(
            number_of_speakers,
            seed=seed + 1
        )

        rng = np.random.default_rng(seed)

        waveforms = []
        labels = []

        print("\nGenerating synthetic speaker dataset...")

        for speaker_id, profile in enumerate(self.profiles):

            for _ in range(utterances_per_speaker):

                waveform = generate_speaker_utterance(
                    profile,
                    rng
                )

                waveforms.append(waveform)
                labels.append(speaker_id)

        waveforms = torch.tensor(
            np.stack(waveforms),
            dtype=torch.float32
        )

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        print("Extracting Mel features...")

        with torch.no_grad():

            logmel = waveform_to_logmel(
                waveforms
            )

        # Normalize using the entire synthetic training dataset.
        self.mel_mean = logmel.mean()
        self.mel_std = logmel.std().clamp_min(1e-5)

        logmel = (
            logmel - self.mel_mean
        ) / self.mel_std

        self.features = logmel
        self.waveforms = waveforms
        self.labels = labels

        print("Dataset ready.")
        print("Speakers:", number_of_speakers)
        print("Utterances per speaker:", utterances_per_speaker)
        print("Feature tensor:", tuple(self.features.shape))

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.features[index],
            self.labels[index]
        )


# ============================================================
# BALANCED BATCH SAMPLER
# ============================================================

class BalancedSpeakerBatchSampler(Sampler):
    """
    Create batches containing multiple speakers and multiple
    utterances per speaker.

    This is useful for supervised contrastive learning.

    Example:
        8 speakers x 4 utterances = 32 samples per batch.
    """

    def __init__(
        self,
        labels,
        speakers_per_batch=8,
        utterances_per_speaker=4
    ):

        self.labels = labels

        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker

        self.speaker_indices = {}

        for index, label in enumerate(labels.tolist()):

            self.speaker_indices.setdefault(
                label,
                []
            ).append(index)

        self.number_of_speakers = len(
            self.speaker_indices
        )

        if self.number_of_speakers < 2:

            raise ValueError(
                "At least two speakers are required."
            )

        for speaker_id, indices in self.speaker_indices.items():

            if len(indices) < utterances_per_speaker:

                raise ValueError(
                    f"Speaker {speaker_id} needs at least "
                    f"{utterances_per_speaker} utterances."
                )

    def __iter__(self):

        speaker_ids = list(
            self.speaker_indices.keys()
        )

        random.shuffle(speaker_ids)

        # Randomize the speaker order for each epoch.
        for start in range(
            0,
            len(speaker_ids),
            self.speakers_per_batch
        ):

            selected_speakers = speaker_ids[
                start:start + self.speakers_per_batch
            ]

            if len(selected_speakers) < 2:
                continue

            batch = []

            for speaker_id in selected_speakers:

                candidates = self.speaker_indices[
                    speaker_id
                ]

                selected = random.sample(
                    candidates,
                    self.utterances_per_speaker
                )

                batch.extend(selected)

            random.shuffle(batch)

            yield batch

    def __len__(self):

        return (
            self.number_of_speakers
            // self.speakers_per_batch
        )


# ============================================================
# SPEAKER ENCODER
# ============================================================

class SpeakerEncoder(nn.Module):
    """
    CNN-based speaker encoder.

    Input:
        (B, 1, mel_channels, time_frames)

    Output:
        normalized speaker embeddings (B, embedding_dim)

    The convolutional layers learn acoustic patterns.

    Adaptive pooling aggregates information across time and
    frequency before producing the speaker embedding.
    """

    def __init__(
        self,
        number_of_speakers,
        embedding_dim=128
    ):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                128,
                192,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(192),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.embedding_layer = nn.Linear(
            192,
            embedding_dim
        )

        self.classifier = nn.Linear(
            embedding_dim,
            number_of_speakers
        )

    def forward(self, mel):

        # Add the single image-like channel.
        x = mel.unsqueeze(1)

        x = self.conv(x)

        x = self.pool(x)

        x = x.flatten(start_dim=1)

        embedding = self.embedding_layer(x)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        logits = self.classifier(embedding)

        return embedding, logits


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.12
):
    """
    Encourage same-speaker embeddings to become similar.

    Encourage different-speaker embeddings to become separated.

    Each batch should contain at least two recordings per speaker.
    """

    batch_size = embeddings.size(0)

    similarity = torch.matmul(
        embeddings,
        embeddings.T
    ) / temperature

    # Remove self-comparisons from denominator.
    identity_mask = torch.eye(
        batch_size,
        device=embeddings.device,
        dtype=torch.bool
    )

    similarity = similarity.masked_fill(
        identity_mask,
        -1e9
    )

    labels = labels.view(-1, 1)

    positive_mask = torch.eq(
        labels,
        labels.T
    ).float()

    positive_mask = positive_mask.masked_fill(
        identity_mask,
        0.0
    )

    log_prob = (
        similarity
        - torch.logsumexp(
            similarity,
            dim=1,
            keepdim=True
        )
    )

    positive_count = positive_mask.sum(dim=1)

    valid = positive_count > 0

    if not valid.any():

        return embeddings.sum() * 0.0

    mean_positive_log_prob = (
        (positive_mask * log_prob).sum(dim=1)
        / positive_count.clamp_min(1.0)
    )

    loss = -mean_positive_log_prob[valid].mean()

    return loss


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model,
    dataset,
    batch_size=32
):

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    all_embeddings = []
    all_labels = []

    for mel, labels in loader:

        mel = mel.to(DEVICE)

        embeddings, _ = model(mel)

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )

    return (
        np.concatenate(all_embeddings),
        np.concatenate(all_labels)
    )


# ============================================================
# COSINE SIMILARITY MATRIX
# ============================================================

def calculate_speaker_similarity(
    embeddings,
    labels,
    number_of_speakers
):
    """
    Average utterance embeddings for each speaker, then calculate
    pairwise cosine similarity between speaker centroids.
    """

    centroids = []

    for speaker_id in range(number_of_speakers):

        selected = embeddings[
            labels == speaker_id
        ]

        centroid = selected.mean(axis=0)

        centroid /= (
            np.linalg.norm(centroid) + EPS
        )

        centroids.append(centroid)

    centroids = np.stack(centroids)

    similarity = centroids @ centroids.T

    return centroids, similarity


# ============================================================
# PCA — IMPLEMENTED WITH NUMPY
# ============================================================

def pca_2d(data):
    """
    Project embeddings into two dimensions using PCA.

    This implementation uses NumPy eigendecomposition and
    does not require scikit-learn.
    """

    data = np.asarray(
        data,
        dtype=np.float64
    )

    mean = data.mean(axis=0)

    centered = data - mean

    covariance = (
        centered.T @ centered
    ) / max(len(data) - 1, 1)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    components = eigenvectors[
        :,
        order[:2]
    ]

    projected = centered @ components

    total_variance = np.sum(
        np.maximum(eigenvalues, 0)
    )

    if total_variance > 0:

        explained = (
            np.maximum(eigenvalues[order[:2]], 0)
            / total_variance
        )

    else:

        explained = np.zeros(2)

    return projected, explained


# ============================================================
# TRAINING
# ============================================================

def train(
    number_of_speakers=12,
    utterances_per_speaker=16,
    epochs=25,
    no_plot=False
):

    dataset = SpeakerDataset(
        number_of_speakers=number_of_speakers,
        utterances_per_speaker=utterances_per_speaker
    )

    speakers_per_batch = min(
        8,
        number_of_speakers
    )

    utterances_per_batch = min(
        4,
        utterances_per_speaker
    )

    batch_sampler = BalancedSpeakerBatchSampler(
        dataset.labels,
        speakers_per_batch=speakers_per_batch,
        utterances_per_speaker=utterances_per_batch
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0
    )

    model = SpeakerEncoder(
        number_of_speakers=number_of_speakers,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    history = {
        "total": [],
        "classification": [],
        "contrastive": [],
        "accuracy": []
    }

    print("\nTraining speaker encoder.")
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss_sum = 0.0
        classification_sum = 0.0
        contrastive_sum = 0.0

        correct = 0
        total = 0
        batches = 0

        start_time = time.time()

        for mel, labels in loader:

            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            embeddings, logits = model(mel)

            loss_classification = F.cross_entropy(
                logits,
                labels
            )

            loss_contrastive = supervised_contrastive_loss(
                embeddings,
                labels
            )

            loss = (
                loss_classification
                + 0.25 * loss_contrastive
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            total_loss_sum += loss.item()

            classification_sum += (
                loss_classification.item()
            )

            contrastive_sum += (
                loss_contrastive.item()
            )

            batches += 1

        if batches == 0:

            raise RuntimeError(
                "No batches were generated. Increase the number "
                "of speakers or utterances."
            )

        mean_loss = total_loss_sum / batches

        mean_classification = (
            classification_sum / batches
        )

        mean_contrastive = (
            contrastive_sum / batches
        )

        accuracy = correct / max(total, 1)

        history["total"].append(mean_loss)

        history["classification"].append(
            mean_classification
        )

        history["contrastive"].append(
            mean_contrastive
        )

        history["accuracy"].append(
            accuracy
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Loss: {mean_loss:.4f} | "
            f"CE: {mean_classification:.4f} | "
            f"SupCon: {mean_contrastive:.4f} | "
            f"Accuracy: {accuracy * 100:.1f}% | "
            f"{elapsed:.1f}s"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "number_of_speakers": number_of_speakers,

        "embedding_dim": EMBEDDING_DIM,

        "sample_rate": SAMPLE_RATE,

        "n_fft": N_FFT,

        "hop_length": HOP_LENGTH,

        "n_mels": N_MELS,

        "mel_mean": dataset.mel_mean.cpu(),

        "mel_std": dataset.mel_std.cpu(),

        "speaker_profiles": dataset.profiles
    }

    torch.save(
        checkpoint,
        "lesson17_speaker_encoder.pt"
    )

    print(
        "\nSaved:",
        "lesson17_speaker_encoder.pt"
    )

    # ========================================================
    # EMBEDDING EVALUATION
    # ========================================================

    embeddings, labels = extract_embeddings(
        model,
        dataset
    )

    centroids, similarity = calculate_speaker_similarity(
        embeddings,
        labels,
        number_of_speakers
    )

    print("\nSpeaker similarity matrix:")
    print(np.round(similarity, 2))

    print(
        "\nMean off-diagonal speaker similarity:",
        round(
            (
                similarity.sum()
                - np.trace(similarity)
            )
            /
            (
                number_of_speakers
                * (number_of_speakers - 1)
            ),
            4
        )
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    if not no_plot:

        plot_training_history(history)

        plot_embeddings(
            embeddings,
            labels,
            number_of_speakers
        )

        plot_similarity_matrix(
            similarity
        )

    return model, dataset, embeddings, labels


# ============================================================
# VISUALIZATION — TRAINING
# ============================================================

def plot_training_history(history):

    epochs = np.arange(
        1,
        len(history["total"]) + 1
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8)
    )

    axes[0, 0].plot(
        epochs,
        history["total"]
    )

    axes[0, 0].set_title("Total Training Loss")

    axes[0, 1].plot(
        epochs,
        history["classification"]
    )

    axes[0, 1].set_title("Speaker Classification Loss")

    axes[1, 0].plot(
        epochs,
        history["contrastive"]
    )

    axes[1, 0].set_title("Supervised Contrastive Loss")

    axes[1, 1].plot(
        epochs,
        history["accuracy"]
    )

    axes[1, 1].set_title("Training Speaker Accuracy")

    for axis in axes.flatten():

        axis.set_xlabel("Epoch")
        axis.grid(True)

    plt.tight_layout()

    plt.savefig(
        "lesson17_training_curves.png",
        dpi=150
    )

    plt.show()


# ============================================================
# VISUALIZATION — PCA
# ============================================================

def plot_embeddings(
    embeddings,
    labels,
    number_of_speakers
):

    projected, explained = pca_2d(
        embeddings
    )

    plt.figure(figsize=(10, 8))

    for speaker_id in range(number_of_speakers):

        selected = labels == speaker_id

        plt.scatter(
            projected[selected, 0],
            projected[selected, 1],
            label=f"Speaker {speaker_id}",
            alpha=0.75
        )

    plt.title(
        "Speaker Embeddings — PCA Projection"
    )

    plt.xlabel(
        f"PC1 ({explained[0] * 100:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained[1] * 100:.1f}% variance)"
    )

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "lesson17_speaker_embeddings.png",
        dpi=150
    )

    plt.show()


# ============================================================
# VISUALIZATION — SIMILARITY MATRIX
# ============================================================

def plot_similarity_matrix(similarity):

    plt.figure(figsize=(9, 7))

    image = plt.imshow(
        similarity,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    plt.colorbar(
        image,
        label="Cosine similarity"
    )

    plt.title(
        "Speaker Centroid Cosine Similarity"
    )

    plt.xlabel("Speaker ID")
    plt.ylabel("Speaker ID")

    plt.xticks(
        range(len(similarity))
    )

    plt.yticks(
        range(len(similarity))
    )

    plt.tight_layout()

    plt.savefig(
        "lesson17_similarity_matrix.png",
        dpi=150
    )

    plt.show()


# ============================================================
# LOAD MODEL AND EXTRACT WAV EMBEDDING
# ============================================================

def load_encoder(checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = SpeakerEncoder(
        number_of_speakers=checkpoint[
            "number_of_speakers"
        ],
        embedding_dim=checkpoint[
            "embedding_dim"
        ]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


@torch.no_grad()
def extract_wav_embedding(
    audio_path,
    checkpoint_path="lesson17_speaker_encoder.pt"
):
    """
    Extract a speaker embedding from an external WAV file.

    IMPORTANT:
    This encoder was trained on synthetic audio. Therefore,
    its embedding for a real human recording is NOT a meaningful
    real-world speaker identity representation.

    This function demonstrates the inference pipeline only.
    """

    model, checkpoint = load_encoder(
        checkpoint_path
    )

    waveform, sample_rate = sf.read(
        audio_path,
        dtype="float32"
    )

    if waveform.ndim == 2:

        waveform = waveform.mean(axis=1)

    waveform_tensor = torch.tensor(
        waveform,
        dtype=torch.float32
    )

    if sample_rate != SAMPLE_RATE:

        waveform_tensor = torch.tensor(
            resample_waveform(
                waveform_tensor.numpy(),
                sample_rate,
                SAMPLE_RATE
            ),
            dtype=torch.float32
        )

    # Split long recordings into fixed-size segments.
    # Short recordings are padded.
    if len(waveform_tensor) < AUDIO_SAMPLES:

        waveform_tensor = F.pad(
            waveform_tensor,
            (0, AUDIO_SAMPLES - len(waveform_tensor))
        )

    else:

        waveform_tensor = waveform_tensor[
            :AUDIO_SAMPLES
        ]

    waveform_tensor = waveform_tensor.unsqueeze(0).to(
        DEVICE
    )

    logmel = waveform_to_logmel(
        waveform_tensor
    )

    mean = checkpoint["mel_mean"].to(DEVICE)
    std = checkpoint["mel_std"].to(DEVICE)

    logmel = (
        logmel - mean
    ) / std.clamp_min(1e-5)

    embedding, logits = model(logmel)

    result = embedding.squeeze(0).cpu().numpy()

    print("\nExtracted embedding:")
    print("Shape:", result.shape)
    print("L2 norm:", np.linalg.norm(result))

    print(
        "\nPredicted synthetic training speaker ID:",
        logits.argmax(dim=1).item()
    )

    print(
        "WARNING: This predicted ID has no reliable "
        "real-human speaker interpretation."
    )

    return result


# ============================================================
# SIMPLE WAV RESAMPLING
# ============================================================

def resample_waveform(
    waveform,
    original_sample_rate,
    target_sample_rate
):
    """
    Linear interpolation resampling for this demonstration.

    Production speech pipelines should use a high-quality
    band-limited resampler.
    """

    if original_sample_rate == target_sample_rate:

        return waveform.astype(np.float32)

    if len(waveform) == 0:

        raise ValueError("Input audio is empty.")

    new_length = int(
        round(
            len(waveform)
            * target_sample_rate
            / original_sample_rate
        )
    )

    old_positions = np.arange(
        len(waveform)
    )

    new_positions = np.linspace(
        0,
        len(waveform) - 1,
        new_length
    )

    return np.interp(
        new_positions,
        old_positions,
        waveform
    ).astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train a synthetic speaker encoder."
    )

    parser.add_argument(
        "--speakers",
        type=int,
        default=12
    )

    parser.add_argument(
        "--utterances",
        type=int,
        default=16
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25
    )

    parser.add_argument(
        "--audio",
        type=str,
        default=None
    )

    parser.add_argument(
        "--no-plot",
        action="store_true"
    )

    args = parser.parse_args()

    if args.speakers < 2:

        parser.error("--speakers must be at least 2")

    if args.utterances < 4:

        parser.error(
            "--utterances must be at least 4"
        )

    if args.epochs < 1:

        parser.error("--epochs must be at least 1")

    set_seed(42)

    if args.audio:

        if not Path(
            "lesson17_speaker_encoder.pt"
        ).exists():

            raise FileNotFoundError(
                "Train the model first. "
                "lesson17_speaker_encoder.pt was not found."
            )

        extract_wav_embedding(
            args.audio
        )

    else:

        train(
            number_of_speakers=args.speakers,
            utterances_per_speaker=args.utterances,
            epochs=args.epochs,
            no_plot=args.no_plot
        )


if __name__ == "__main__":

    main()