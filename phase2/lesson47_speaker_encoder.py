"""
========================================================================
VOICE CLONING COURSE
LESSON 47 — REFERENCE AUDIO SPEAKER ENCODER
========================================================================

PURPOSE
-------
Extract speaker identity embeddings directly from WAV audio.

LESSON ROADMAP
--------------

STEP 1 — WAV Dataset
    Load audio recordings organized by speaker.

STEP 2 — Audio Preprocessing
    Convert audio to mono and resample to 16 kHz.

STEP 3 — Audio Segmentation
    Extract fixed-length speech segments.

STEP 4 — STFT
    Convert waveform into frequency representations.

STEP 5 — Mel Spectrogram
    Construct a log-Mel spectrogram using a triangular filterbank.

STEP 6 — CNN Speaker Encoder
    Learn speaker-related acoustic representations.

STEP 7 — Speaker Classification
    Train the encoder to recognize training speakers.

STEP 8 — Contrastive Learning
    Encourage recordings from the same speaker to have similar
    embeddings and recordings from different speakers to differ.

STEP 9 — Speaker Verification
    Calculate cosine similarity between reference recordings.

STEP 10 — Visualization
    Plot training loss, accuracy, embeddings and similarity.

REAL-WORLD APPLICATION
----------------------
A user supplies a reference recording.

The speaker encoder extracts a 128-dimensional vector.

This vector can later condition a speech generation model.

IMPORTANT
---------
Synthetic demonstration audio is NOT real human voice data.

For actual speaker identity learning, use genuine WAV recordings.

This educational CNN is not a production-grade speaker verification
system. Real applications require larger and more diverse datasets.

REQUIREMENTS
------------
pip install torch numpy matplotlib soundfile scipy scikit-learn

RUN
---
python lesson47_speaker_encoder.py

========================================================================
"""

import os
import random
import json
import math

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import resample_poly

from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000

AUDIO_DURATION = 3.0

NUM_SAMPLES = int(
    SAMPLE_RATE * AUDIO_DURATION
)

N_FFT = 512

HOP_LENGTH = 160

WIN_LENGTH = 400

N_MELS = 80

F_MIN = 50

F_MAX = 7600

EMBED_DIM = 128

NUM_CHANNELS = 64

NUM_TRAIN_EPOCHS = 35

BATCH_SIZE = 16

LEARNING_RATE = 3e-4

CONTRASTIVE_WEIGHT = 0.2

NUM_SYNTHETIC_SPEAKERS = 4

SYNTHETIC_RECORDINGS_PER_SPEAKER = 12

DATASET_DIR = "dataset/speakers"

OUTPUT_DIR = "lesson47_outputs"

USE_SYNTHETIC_DEMO_IF_NO_DATA = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# =====================================================================
# STEP 1 — SYNTHETIC AUDIO GENERATOR
# =====================================================================

def generate_synthetic_audio(
    speaker_id,
    duration=3.0
):

    """
    Generate artificial speaker-dependent audio.

    Different speakers receive different:
        - Fundamental frequencies
        - Harmonic distributions
        - Spectral envelopes

    This is only for testing the pipeline.

    It does not simulate genuine human voice identity.
    """

    sample_rate = SAMPLE_RATE

    t = np.arange(
        int(sample_rate * duration)
    ) / sample_rate

    fundamental_frequencies = [
        110,
        145,
        185,
        225
    ]

    f0 = fundamental_frequencies[speaker_id]

    # Speaker-dependent spectral envelope.
    formants = [
        [500, 1500, 2500],
        [700, 1800, 2800],
        [400, 1200, 2200],
        [800, 2000, 3000]
    ]

    signal = np.zeros_like(t)

    for harmonic in range(1, 35):

        frequency = f0 * harmonic

        if frequency >= sample_rate / 2:
            break

        amplitude = 1.0 / harmonic

        spectral_weight = 0.0

        for formant in formants[speaker_id]:

            spectral_weight += np.exp(
                -0.5 * (
                    (frequency - formant) / 250
                ) ** 2
            )

        amplitude *= (
            0.15 + spectral_weight
        )

        phase = np.random.uniform(
            0,
            2 * np.pi
        )

        signal += amplitude * np.sin(
            2 * np.pi * frequency * t + phase
        )

    # Add slow amplitude variation.
    modulation = (
        0.75
        + 0.25 * np.sin(
            2 * np.pi * 3 * t
        )
    )

    signal *= modulation

    # Add small recording variation.
    noise = np.random.normal(
        0,
        0.015,
        len(signal)
    )

    signal += noise

    # Normalize.
    signal /= (
        np.max(np.abs(signal)) + 1e-8
    )

    return signal.astype(np.float32)


def create_synthetic_dataset():

    print("\nGenerating synthetic demonstration audio...")

    for speaker_id in range(
        NUM_SYNTHETIC_SPEAKERS
    ):

        speaker_name = f"speaker_{speaker_id}"

        speaker_dir = os.path.join(
            OUTPUT_DIR,
            "synthetic_audio",
            speaker_name
        )

        os.makedirs(
            speaker_dir,
            exist_ok=True
        )

        for recording_id in range(
            SYNTHETIC_RECORDINGS_PER_SPEAKER
        ):

            audio = generate_synthetic_audio(
                speaker_id,
                AUDIO_DURATION
            )

            path = os.path.join(
                speaker_dir,
                f"recording_{recording_id:03d}.wav"
            )

            sf.write(
                path,
                audio,
                SAMPLE_RATE
            )

    return os.path.join(
        OUTPUT_DIR,
        "synthetic_audio"
    )


# =====================================================================
# STEP 2 — WAV LOADING AND PREPROCESSING
# =====================================================================

def load_audio(path):

    audio, sample_rate = sf.read(
        path,
        dtype="float32"
    )

    # Convert stereo to mono.
    if audio.ndim == 2:

        audio = audio.mean(axis=1)

    # Resample.
    if sample_rate != SAMPLE_RATE:

        from math import gcd

        divisor = gcd(
            int(sample_rate),
            SAMPLE_RATE
        )

        up = SAMPLE_RATE // divisor

        down = sample_rate // divisor

        audio = resample_poly(
            audio,
            up,
            down
        ).astype(np.float32)

    # Remove DC offset.
    audio = audio - np.mean(audio)

    return audio.astype(np.float32)


# =====================================================================
# STEP 3 — DISCOVER SPEAKER RECORDINGS
# =====================================================================

def discover_recordings(root):

    speakers = {}

    if not os.path.isdir(root):

        return speakers

    for speaker_name in sorted(
        os.listdir(root)
    ):

        speaker_dir = os.path.join(
            root,
            speaker_name
        )

        if not os.path.isdir(speaker_dir):

            continue

        recordings = []

        for filename in sorted(
            os.listdir(speaker_dir)
        ):

            if filename.lower().endswith(".wav"):

                recordings.append(
                    os.path.join(
                        speaker_dir,
                        filename
                    )
                )

        if recordings:

            speakers[speaker_name] = recordings

    return speakers


# =====================================================================
# STEP 4 — TRAIN/VALIDATION FILE SPLIT
# =====================================================================

def split_recordings(speakers):

    train_samples = []

    val_samples = []

    speaker_to_id = {
        name: idx
        for idx, name in enumerate(
            sorted(speakers.keys())
        )
    }

    for speaker_name, recordings in speakers.items():

        if len(recordings) < 3:

            raise ValueError(
                f"{speaker_name} needs at least "
                "3 separate WAV recordings."
            )

        recordings = list(recordings)

        random.shuffle(recordings)

        val_count = max(
            1,
            int(len(recordings) * 0.2)
        )

        val_files = recordings[:val_count]

        train_files = recordings[val_count:]

        if len(train_files) < 2:

            raise ValueError(
                f"{speaker_name} needs at least "
                "2 training recordings."
            )

        speaker_id = speaker_to_id[speaker_name]

        for path in train_files:

            train_samples.append(
                (path, speaker_id)
            )

        for path in val_files:

            val_samples.append(
                (path, speaker_id)
            )

    return (
        train_samples,
        val_samples,
        speaker_to_id
    )


# =====================================================================
# STEP 5 — AUDIO DATASET
# =====================================================================

class AudioSpeakerDataset(Dataset):

    def __init__(
        self,
        samples,
        training=True
    ):

        self.samples = samples

        self.training = training

        self.audio_cache = {}

    def __len__(self):

        return len(self.samples)

    def load_cached(self, path):

        if path not in self.audio_cache:

            self.audio_cache[path] = load_audio(path)

        return self.audio_cache[path]

    def crop_audio(self, audio):

        if len(audio) < NUM_SAMPLES:

            audio = np.pad(
                audio,
                (0, NUM_SAMPLES - len(audio))
            )

        if self.training:

            max_start = len(audio) - NUM_SAMPLES

            start = random.randint(
                0,
                max_start
            )

        else:

            start = max(
                0,
                (len(audio) - NUM_SAMPLES) // 2
            )

        audio = audio[
            start:start + NUM_SAMPLES
        ]

        return audio

    def __getitem__(self, index):

        path, speaker_id = self.samples[index]

        audio = self.load_cached(path)

        audio = self.crop_audio(audio)

        # Training-only amplitude augmentation.
        if self.training:

            gain = random.uniform(
                0.8,
                1.2
            )

            audio = audio * gain

            if random.random() < 0.3:

                noise = np.random.normal(
                    0,
                    0.003,
                    len(audio)
                )

                audio = audio + noise

        audio = np.clip(
            audio,
            -1.0,
            1.0
        )

        return (
            torch.tensor(
                audio,
                dtype=torch.float32
            ),
            torch.tensor(
                speaker_id,
                dtype=torch.long
            )
        )


# =====================================================================
# STEP 6 — MEL FILTERBANK
# =====================================================================

def hz_to_mel(frequency):

    return 2595.0 * np.log10(
        1.0 + frequency / 700.0
    )


def mel_to_hz(mel):

    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def create_mel_filterbank():

    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1
    )

    mel_min = hz_to_mel(F_MIN)

    mel_max = hz_to_mel(F_MAX)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        N_MELS + 2
    )

    hz_points = mel_to_hz(
        mel_points
    )

    filters = np.zeros(
        (N_MELS, len(frequencies)),
        dtype=np.float32
    )

    for i in range(N_MELS):

        left = hz_points[i]

        center = hz_points[i + 1]

        right = hz_points[i + 2]

        rising = (
            frequencies - left
        ) / max(
            center - left,
            1e-8
        )

        falling = (
            right - frequencies
        ) / max(
            right - center,
            1e-8
        )

        filters[i] = np.maximum(
            0,
            np.minimum(
                rising,
                falling
            )
        )

    return torch.tensor(
        filters,
        dtype=torch.float32
    )


# =====================================================================
# STEP 7 — LOG-MEL AUDIO FRONTEND
# =====================================================================

class LogMelFrontend(nn.Module):

    def __init__(self):

        super().__init__()

        self.register_buffer(
            "mel_filters",
            create_mel_filterbank()
        )

        self.register_buffer(
            "window",
            torch.hann_window(WIN_LENGTH)
        )

    def forward(self, waveform):

        """
        Input:
            [B, waveform_samples]

        Output:
            [B, N_MELS, time_frames]
        """

        spectrum = torch.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            return_complex=True
        )

        magnitude = spectrum.abs()

        mel = torch.matmul(
            self.mel_filters,
            magnitude
        )

        log_mel = torch.log(
            mel.clamp(min=1e-5)
        )

        return log_mel


# =====================================================================
# STEP 8 — CNN SPEAKER ENCODER
# =====================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)


class SpeakerEncoder(nn.Module):

    def __init__(self, num_speakers):

        super().__init__()

        self.frontend = LogMelFrontend()

        self.cnn = nn.Sequential(

            ConvBlock(
                1,
                32,
                stride=2
            ),

            ConvBlock(
                32,
                64,
                stride=2
            ),

            ConvBlock(
                64,
                NUM_CHANNELS,
                stride=2
            ),

            ConvBlock(
                NUM_CHANNELS,
                NUM_CHANNELS,
                stride=2
            )

        )

        # Global average pooling.
        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.projection = nn.Sequential(

            nn.Linear(
                NUM_CHANNELS,
                EMBED_DIM
            ),

            nn.BatchNorm1d(
                EMBED_DIM
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                EMBED_DIM,
                EMBED_DIM
            )

        )

        self.classifier = nn.Linear(
            EMBED_DIM,
            num_speakers
        )

    def forward(self, waveform):

        mel = self.frontend(waveform)

        x = mel.unsqueeze(1)

        x = self.cnn(x)

        x = self.pool(x)

        x = x.flatten(1)

        embedding = self.projection(x)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1
        )

        logits = self.classifier(
            embedding
        )

        return embedding, logits, mel


# =====================================================================
# STEP 9 — CONTRASTIVE LOSS
# =====================================================================

def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.1
):

    batch_size = embeddings.shape[0]

    similarity = torch.matmul(
        embeddings,
        embeddings.T
    ) / temperature

    self_mask = torch.eye(
        batch_size,
        device=embeddings.device,
        dtype=torch.bool
    )

    similarity = similarity.masked_fill(
        self_mask,
        -1e9
    )

    labels = labels.view(-1, 1)

    positive_mask = (
        labels == labels.T
    ) & (~self_mask)

    log_prob = similarity - torch.logsumexp(
        similarity,
        dim=1,
        keepdim=True
    )

    positive_count = positive_mask.sum(
        dim=1
    )

    valid = positive_count > 0

    positive_sum = (
        log_prob * positive_mask.float()
    ).sum(dim=1)

    losses = (
        -positive_sum
        / positive_count.clamp(min=1)
    )

    if valid.any():

        return losses[valid].mean()

    return torch.tensor(
        0.0,
        device=embeddings.device
    )


# =====================================================================
# STEP 10 — VALIDATION
# =====================================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    losses = []

    predictions = []

    targets = []

    for waveform, labels in loader:

        waveform = waveform.to(DEVICE)

        labels = labels.to(DEVICE)

        embeddings, logits, mel = model(
            waveform
        )

        loss = F.cross_entropy(
            logits,
            labels
        )

        losses.append(
            loss.item()
        )

        predictions.extend(
            logits.argmax(dim=1).cpu().tolist()
        )

        targets.extend(
            labels.cpu().tolist()
        )

    accuracy = np.mean(
        np.array(predictions)
        == np.array(targets)
    )

    return (
        float(np.mean(losses)),
        float(accuracy)
    )


# =====================================================================
# STEP 11 — TRAINING
# =====================================================================

def train_model(
    model,
    train_loader,
    val_loader
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")

    patience = 8

    no_improvement = 0

    print("\nStarting speaker encoder training...\n")

    for epoch in range(NUM_TRAIN_EPOCHS):

        model.train()

        epoch_losses = []

        correct = 0

        total = 0

        for waveform, labels in train_loader:

            waveform = waveform.to(DEVICE)

            labels = labels.to(DEVICE)

            embeddings, logits, mel = model(
                waveform
            )

            classification_loss = F.cross_entropy(
                logits,
                labels
            )

            contrastive_loss = supervised_contrastive_loss(
                embeddings,
                labels
            )

            loss = (
                classification_loss
                + CONTRASTIVE_WEIGHT * contrastive_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            epoch_losses.append(
                loss.item()
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = float(
            np.mean(epoch_losses)
        )

        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader
        )

        scheduler.step(val_loss)

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print(
            f"Epoch {epoch + 1:03d}/{NUM_TRAIN_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:

            best_val_loss = val_loss

            no_improvement = 0

            torch.save(
                model.state_dict(),
                os.path.join(
                    OUTPUT_DIR,
                    "lesson47_best.pt"
                )
            )

        else:

            no_improvement += 1

        if no_improvement >= patience:

            print("\nEarly stopping activated.")

            break

    return history


# =====================================================================
# STEP 12 — SPEAKER EMBEDDING EXTRACTION
# =====================================================================

@torch.no_grad()
def extract_embeddings(
    model,
    loader
):

    model.eval()

    embeddings_all = []

    labels_all = []

    for waveform, labels in loader:

        waveform = waveform.to(DEVICE)

        embeddings, logits, mel = model(
            waveform
        )

        embeddings_all.append(
            embeddings.cpu().numpy()
        )

        labels_all.append(
            labels.numpy()
        )

    return (
        np.concatenate(embeddings_all),
        np.concatenate(labels_all)
    )


# =====================================================================
# STEP 13 — COSINE SIMILARITY
# =====================================================================

def cosine_similarity(a, b):

    a = np.asarray(a)

    b = np.asarray(b)

    return float(
        np.dot(a, b)
        / (
            np.linalg.norm(a)
            * np.linalg.norm(b)
            + 1e-8
        )
    )


def calculate_similarity(
    embeddings,
    labels
):

    intra = []

    inter = []

    for i in range(len(embeddings)):

        for j in range(i + 1, len(embeddings)):

            score = cosine_similarity(
                embeddings[i],
                embeddings[j]
            )

            if labels[i] == labels[j]:

                intra.append(score)

            else:

                inter.append(score)

    return (
        np.array(intra),
        np.array(inter)
    )


def calculate_speaker_matrix(
    embeddings,
    labels,
    num_speakers
):

    centers = []

    for speaker_id in range(num_speakers):

        speaker_embeddings = embeddings[
            labels == speaker_id
        ]

        center = speaker_embeddings.mean(
            axis=0
        )

        center /= (
            np.linalg.norm(center) + 1e-8
        )

        centers.append(center)

    centers = np.array(centers)

    return centers @ centers.T


# =====================================================================
# STEP 14 — REFERENCE AUDIO COMPARISON
# =====================================================================

@torch.no_grad()
def extract_reference_embedding(
    model,
    wav_path
):

    model.eval()

    audio = load_audio(
        wav_path
    )

    if len(audio) < NUM_SAMPLES:

        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )

    else:

        # Use center segment.
        start = (
            len(audio) - NUM_SAMPLES
        ) // 2

        audio = audio[
            start:start + NUM_SAMPLES
        ]

    waveform = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    embedding, logits, mel = model(
        waveform
    )

    return embedding[0].cpu().numpy()


def compare_reference_recordings(
    model,
    val_samples
):

    print("\n" + "=" * 72)

    print("REFERENCE AUDIO SIMILARITY TEST")

    print("=" * 72)

    references = {}

    for path, speaker_id in val_samples:

        if speaker_id not in references:

            references[speaker_id] = path

    embeddings = {}

    for speaker_id, path in references.items():

        embeddings[speaker_id] = (
            extract_reference_embedding(
                model,
                path
            )
        )

    for speaker_id, embedding in embeddings.items():

        print(
            f"\nSpeaker {speaker_id}: "
            f"{references[speaker_id]}"
        )

        scores = []

        for other_id, other_embedding in embeddings.items():

            score = cosine_similarity(
                embedding,
                other_embedding
            )

            scores.append(
                (other_id, score)
            )

        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        for other_id, score in scores:

            print(
                f"  Similarity to Speaker {other_id}: "
                f"{score:.4f}"
            )


# =====================================================================
# STEP 15 — VISUALIZATION
# =====================================================================

def plot_training(history):

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Lesson 47 — Speaker Encoder Training Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson47_training_loss.png"
        )
    )

    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        "Lesson 47 — Speaker Classification Accuracy"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson47_accuracy.png"
        )
    )

    plt.show()


def plot_embeddings(
    embeddings,
    labels
):

    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(
        embeddings
    )

    plt.figure(figsize=(9, 7))

    for speaker_id in sorted(
        np.unique(labels)
    ):

        mask = labels == speaker_id

        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=f"Speaker {speaker_id}",
            alpha=0.7
        )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title(
        "Lesson 47 — Real Audio Speaker Embeddings"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson47_embedding_pca.png"
        )
    )

    plt.show()


def plot_similarity_matrix(matrix):

    plt.figure(figsize=(7, 6))

    plt.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    plt.colorbar(
        label="Cosine Similarity"
    )

    plt.xticks(
        range(len(matrix)),
        [f"Speaker {i}" for i in range(len(matrix))]
    )

    plt.yticks(
        range(len(matrix)),
        [f"Speaker {i}" for i in range(len(matrix))]
    )

    for i in range(len(matrix)):

        for j in range(len(matrix)):

            plt.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.title(
        "Lesson 47 — Speaker Similarity Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson47_similarity_matrix.png"
        )
    )

    plt.show()


def plot_similarity_distribution(
    intra,
    inter
):

    plt.figure(figsize=(9, 5))

    plt.hist(
        intra,
        bins=30,
        alpha=0.6,
        label="Same Speaker"
    )

    plt.hist(
        inter,
        bins=30,
        alpha=0.6,
        label="Different Speakers"
    )

    plt.xlabel("Cosine Similarity")

    plt.ylabel("Frequency")

    plt.title(
        "Lesson 47 — Speaker Verification Distribution"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson47_similarity_distribution.png"
        )
    )

    plt.show()


# =====================================================================
# STEP 16 — MAIN
# =====================================================================

def main():

    print("=" * 72)

    print(
        "LESSON 47 — REFERENCE AUDIO SPEAKER ENCODER"
    )

    print("=" * 72)

    print(f"\nDevice: {DEVICE}")

    print(f"Sample rate: {SAMPLE_RATE}")

    print(f"Audio duration: {AUDIO_DURATION} seconds")

    print(f"FFT size: {N_FFT}")

    print(f"Mel bins: {N_MELS}")

    print(f"Embedding dimension: {EMBED_DIM}")

    # ---------------------------------------------------------------
    # Discover real recordings.
    # ---------------------------------------------------------------

    speakers = discover_recordings(
        DATASET_DIR
    )

    is_synthetic = False

    # ---------------------------------------------------------------
    # Synthetic fallback.
    # ---------------------------------------------------------------

    if not speakers:

        if not USE_SYNTHETIC_DEMO_IF_NO_DATA:

            raise FileNotFoundError(
                f"No WAV recordings found in {DATASET_DIR}"
            )

        synthetic_root = create_synthetic_dataset()

        speakers = discover_recordings(
            synthetic_root
        )

        is_synthetic = True

    print("\nDataset type:")

    print(
        "SYNTHETIC DEMONSTRATION"
        if is_synthetic
        else "REAL WAV RECORDINGS"
    )

    # ---------------------------------------------------------------
    # Dataset information.
    # ---------------------------------------------------------------

    print("\nSpeaker dataset:")

    for name, recordings in speakers.items():

        print(
            f"  {name}: {len(recordings)} recordings"
        )

    # ---------------------------------------------------------------
    # Split dataset.
    # ---------------------------------------------------------------

    train_samples, val_samples, speaker_to_id = (
        split_recordings(speakers)
    )

    num_speakers = len(speaker_to_id)

    print("\nSpeaker mapping:")

    for name, idx in speaker_to_id.items():

        print(
            f"  {name} -> {idx}"
        )

    print(
        f"\nTraining recordings: {len(train_samples)}"
    )

    print(
        f"Validation recordings: {len(val_samples)}"
    )

    # ---------------------------------------------------------------
    # Build datasets.
    # ---------------------------------------------------------------

    train_dataset = AudioSpeakerDataset(
        train_samples,
        training=True
    )

    val_dataset = AudioSpeakerDataset(
        val_samples,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ---------------------------------------------------------------
    # Model.
    # ---------------------------------------------------------------

    model = SpeakerEncoder(
        num_speakers
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: {parameters:,}"
    )

    # ---------------------------------------------------------------
    # Training.
    # ---------------------------------------------------------------

    history = train_model(
        model,
        train_loader,
        val_loader
    )

    # ---------------------------------------------------------------
    # Restore best checkpoint.
    # ---------------------------------------------------------------

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        "lesson47_best.pt"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    # ---------------------------------------------------------------
    # Extract embeddings.
    # ---------------------------------------------------------------

    embeddings, labels = extract_embeddings(
        model,
        val_loader
    )

    # ---------------------------------------------------------------
    # Similarity.
    # ---------------------------------------------------------------

    intra, inter = calculate_similarity(
        embeddings,
        labels
    )

    matrix = calculate_speaker_matrix(
        embeddings,
        labels,
        num_speakers
    )

    print("\n" + "=" * 72)

    print("SPEAKER SIMILARITY STATISTICS")

    print("=" * 72)

    print(
        f"\nMean intra-speaker similarity: "
        f"{intra.mean():.4f}"
    )

    print(
        f"Mean inter-speaker similarity: "
        f"{inter.mean():.4f}"
    )

    print(
        f"Similarity separation: "
        f"{intra.mean() - inter.mean():.4f}"
    )

    print("\nSpeaker similarity matrix:")

    print(
        np.array2string(
            matrix,
            precision=3
        )
    )

    # ---------------------------------------------------------------
    # Compare reference WAV recordings.
    # ---------------------------------------------------------------

    compare_reference_recordings(
        model,
        val_samples
    )

    # ---------------------------------------------------------------
    # Visualization.
    # ---------------------------------------------------------------

    plot_training(history)

    plot_embeddings(
        embeddings,
        labels
    )

    plot_similarity_matrix(
        matrix
    )

    plot_similarity_distribution(
        intra,
        inter
    )

    # ---------------------------------------------------------------
    # Save metadata.
    # ---------------------------------------------------------------

    metadata = {

        "sample_rate": SAMPLE_RATE,

        "audio_duration": AUDIO_DURATION,

        "n_fft": N_FFT,

        "n_mels": N_MELS,

        "embedding_dim": EMBED_DIM,

        "num_speakers": num_speakers,

        "speaker_to_id": speaker_to_id,

        "synthetic_demo": is_synthetic

    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "lesson47_metadata.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\n" + "=" * 72)

    print("LESSON 47 COMPLETED")

    print("=" * 72)

    print(
        f"\nSynthetic demonstration: {is_synthetic}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print("\nGenerated files:")

    print("  lesson47_best.pt")

    print("  lesson47_metadata.json")

    print("  lesson47_training_loss.png")

    print("  lesson47_accuracy.png")

    print("  lesson47_embedding_pca.png")

    print("  lesson47_similarity_matrix.png")

    print("  lesson47_similarity_distribution.png")


if __name__ == "__main__":

    main()