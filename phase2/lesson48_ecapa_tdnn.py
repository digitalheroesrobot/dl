"""
Emphasized Channel Attention, Propagation and Aggregation in TDNN-based Speaker Verification.

It is an important architecture for extracting speaker identity from speech recordings.
phase2/
├── lesson48_ecapa_tdnn.py
│
└── dataset/
    └── speakers/
        ├── speaker_0/
        │   ├── recording_001.wav
        │   ├── recording_002.wav
        │   └── ...
        │
        ├── speaker_1/
        │   ├── recording_001.wav
        │   └── ...
        │
        └── speaker_2/
            ├── recording_001.wav
            └── ...
"""
"""
========================================================================
VOICE CLONING COURSE
LESSON 48 — ECAPA-TDNN SPEAKER ENCODER FROM SCRATCH
========================================================================

PURPOSE
-------
Build an ECAPA-TDNN-inspired speaker encoder using PyTorch.

WHAT YOU LEARN
--------------

STEP 1 — WAV Dataset
    Load real speech recordings organized by speaker.

STEP 2 — Log-Mel Frontend
    Convert audio into acoustic features.

STEP 3 — TDNN
    Extract temporal speech features using 1D convolutions.

STEP 4 — Res2Net
    Learn multi-scale temporal representations.

STEP 5 — Squeeze-and-Excitation
    Learn channel attention.

STEP 6 — SERes2Net
    Combine residual learning, multi-scale features and attention.

STEP 7 — Multi-Layer Aggregation
    Combine features from multiple temporal scales.

STEP 8 — Attentive Statistics Pooling
    Extract weighted mean and standard deviation.

STEP 9 — AAM-Softmax
    Train discriminative speaker embeddings using angular margins.

STEP 10 — Speaker Verification
    Compare recordings using cosine similarity.

STEP 11 — Visualization
    Plot embeddings, losses and similarity distributions.

ARCHITECTURE
------------
Log-Mel
   |
TDNN
   |
SERes2Net x 3
   |
Multi-Layer Aggregation
   |
Attentive Statistics Pooling
   |
192-D Speaker Embedding
   |
AAM-Softmax

IMPORTANT
---------
This is an educational ECAPA-TDNN-inspired implementation.
It follows the principal architectural ideas, not every detail
of the original ECAPA-TDNN implementation.

Synthetic audio is provided for pipeline testing only.
Use real speech recordings for actual voice identity learning.

REQUIREMENTS
------------
pip install torch numpy matplotlib soundfile scipy scikit-learn

RUN
---
python lesson48_ecapa_tdnn.py

========================================================================
"""

import os
import json
import random

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

NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)

N_FFT = 512

HOP_LENGTH = 160

WIN_LENGTH = 400

N_MELS = 80

F_MIN = 50

F_MAX = 7600

CHANNELS = 128

EMBED_DIM = 192

ATTENTION_CHANNELS = 128

NUM_SPEAKERS = 4

SCALE = 8

AAM_MARGIN = 0.2

AAM_SCALE = 30.0

BATCH_SIZE = 16

EPOCHS = 35

LEARNING_RATE = 3e-4

NUM_SYNTHETIC_RECORDINGS = 20

DATASET_DIR = "dataset/speakers"

OUTPUT_DIR = "lesson48_outputs"

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
    Generate artificial speaker-dependent acoustic signals.

    Different speakers have different spectral characteristics.

    This is NOT realistic human speech.
    """

    t = np.arange(
        int(SAMPLE_RATE * duration)
    ) / SAMPLE_RATE

    base_frequencies = [
        110,
        145,
        185,
        225
    ]

    formants = [
        [500, 1500, 2500],
        [700, 1800, 2800],
        [400, 1200, 2200],
        [800, 2000, 3000]
    ]

    f0 = base_frequencies[speaker_id]

    # Vary the fundamental frequency between recordings.
    f0 *= random.uniform(0.92, 1.08)

    signal = np.zeros_like(t)

    for harmonic in range(1, 35):

        frequency = f0 * harmonic

        if frequency >= SAMPLE_RATE / 2:
            break

        amplitude = 1.0 / harmonic

        envelope = 0.0

        for formant in formants[speaker_id]:

            envelope += np.exp(
                -0.5 * (
                    (frequency - formant) / 250
                ) ** 2
            )

        amplitude *= 0.15 + envelope

        phase = random.uniform(
            0,
            2 * np.pi
        )

        signal += amplitude * np.sin(
            2 * np.pi * frequency * t + phase
        )

    # Amplitude modulation.
    modulation_frequency = random.uniform(
        2.0,
        4.0
    )

    modulation = (
        0.75
        + 0.25 * np.sin(
            2 * np.pi * modulation_frequency * t
        )
    )

    signal *= modulation

    # Add noise.
    signal += np.random.normal(
        0,
        0.015,
        len(signal)
    )

    signal /= (
        np.max(np.abs(signal)) + 1e-8
    )

    return signal.astype(np.float32)


def create_synthetic_dataset():

    print("\nGenerating synthetic audio dataset...")

    root = os.path.join(
        OUTPUT_DIR,
        "synthetic_audio"
    )

    for speaker_id in range(NUM_SPEAKERS):

        speaker_dir = os.path.join(
            root,
            f"speaker_{speaker_id}"
        )

        os.makedirs(
            speaker_dir,
            exist_ok=True
        )

        for recording_id in range(
            NUM_SYNTHETIC_RECORDINGS
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

    return root


# =====================================================================
# STEP 2 — AUDIO PREPROCESSING
# =====================================================================

def load_audio(path):

    audio, sample_rate = sf.read(
        path,
        dtype="float32"
    )

    # Stereo to mono.
    if audio.ndim == 2:

        audio = audio.mean(axis=1)

    # Resample to 16 kHz.
    if sample_rate != SAMPLE_RATE:

        from math import gcd

        divisor = gcd(
            int(sample_rate),
            SAMPLE_RATE
        )

        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            sample_rate // divisor
        )

    # Remove DC offset.
    audio = audio - np.mean(audio)

    return audio.astype(np.float32)


# =====================================================================
# STEP 3 — DISCOVER RECORDINGS
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
# STEP 4 — TRAIN / VALIDATION SPLIT
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

    for name, recordings in speakers.items():

        recordings = list(recordings)

        if len(recordings) < 3:

            raise ValueError(
                f"{name} needs at least 3 recordings."
            )

        random.shuffle(recordings)

        val_count = max(
            1,
            int(len(recordings) * 0.2)
        )

        val_files = recordings[:val_count]

        train_files = recordings[val_count:]

        if len(train_files) < 2:

            raise ValueError(
                f"{name} needs at least 2 training recordings."
            )

        speaker_id = speaker_to_id[name]

        train_samples.extend(
            [
                (path, speaker_id)
                for path in train_files
            ]
        )

        val_samples.extend(
            [
                (path, speaker_id)
                for path in val_files
            ]
        )

    return (
        train_samples,
        val_samples,
        speaker_to_id
    )


# =====================================================================
# STEP 5 — AUDIO DATASET
# =====================================================================

class AudioDataset(Dataset):

    def __init__(
        self,
        samples,
        training=True
    ):

        self.samples = samples

        self.training = training

        self.cache = {}

    def __len__(self):

        return len(self.samples)

    def get_audio(self, path):

        if path not in self.cache:

            self.cache[path] = load_audio(path)

        return self.cache[path]

    def __getitem__(self, index):

        path, speaker_id = self.samples[index]

        audio = self.get_audio(path)

        if len(audio) < NUM_SAMPLES:

            audio = np.pad(
                audio,
                (0, NUM_SAMPLES - len(audio))
            )

        max_start = len(audio) - NUM_SAMPLES

        if self.training:

            start = random.randint(
                0,
                max_start
            )

        else:

            start = max_start // 2

        audio = audio[
            start:start + NUM_SAMPLES
        ]

        if self.training:

            # Gain augmentation.
            audio = audio * random.uniform(
                0.8,
                1.2
            )

            # Noise augmentation.
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
# STEP 6 — LOG-MEL FRONTEND
# =====================================================================

def hz_to_mel(f):

    return 2595.0 * np.log10(
        1.0 + f / 700.0
    )


def mel_to_hz(m):

    return 700.0 * (
        10.0 ** (m / 2595.0) - 1.0
    )


def create_mel_filterbank():

    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1
    )

    mel_points = np.linspace(
        hz_to_mel(F_MIN),
        hz_to_mel(F_MAX),
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

        # Per-utterance feature normalization.
        mean = log_mel.mean(
            dim=-1,
            keepdim=True
        )

        std = log_mel.std(
            dim=-1,
            keepdim=True
        ).clamp(min=1e-5)

        log_mel = (
            log_mel - mean
        ) / std

        return log_mel


# =====================================================================
# STEP 7 — TDNN BLOCK
# =====================================================================

class TDNNBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        dilation=1
    ):

        super().__init__()

        padding = (
            kernel_size // 2
        ) * dilation

        self.block = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            ),

            nn.BatchNorm1d(
                out_channels
            ),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)


# =====================================================================
# STEP 8 — RES2NET TEMPORAL BLOCK
# =====================================================================

class Res2NetBlock(nn.Module):

    def __init__(
        self,
        channels,
        scale=8,
        dilation=2
    ):

        super().__init__()

        assert channels % scale == 0

        self.scale = scale

        self.width = channels // scale

        self.blocks = nn.ModuleList()

        for _ in range(scale - 1):

            self.blocks.append(

                nn.Sequential(

                    nn.Conv1d(
                        self.width,
                        self.width,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False
                    ),

                    nn.BatchNorm1d(
                        self.width
                    ),

                    nn.ReLU(inplace=True)

                )

            )

    def forward(self, x):

        chunks = torch.chunk(
            x,
            self.scale,
            dim=1
        )

        outputs = []

        outputs.append(
            chunks[0]
        )

        current = None

        for i in range(1, self.scale):

            if i == 1:

                current = chunks[i]

            else:

                current = (
                    current + chunks[i]
                )

            current = self.blocks[i - 1](
                current
            )

            outputs.append(
                current
            )

        return torch.cat(
            outputs,
            dim=1
        )


# =====================================================================
# STEP 9 — SQUEEZE AND EXCITATION
# =====================================================================

class SEBlock(nn.Module):

    def __init__(
        self,
        channels,
        reduction=8
    ):

        super().__init__()

        hidden = max(
            channels // reduction,
            8
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Conv1d(
            channels,
            hidden,
            kernel_size=1
        )

        self.fc2 = nn.Conv1d(
            hidden,
            channels,
            kernel_size=1
        )

    def forward(self, x):

        # Squeeze.
        scale = self.pool(x)

        # Excitation.
        scale = F.relu(
            self.fc1(scale)
        )

        scale = torch.sigmoid(
            self.fc2(scale)
        )

        return x * scale


# =====================================================================
# STEP 10 — SERES2NET BLOCK
# =====================================================================

class SERes2NetBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation
    ):

        super().__init__()

        self.pre = TDNNBlock(
            channels,
            channels,
            kernel_size=1
        )

        self.res2net = Res2NetBlock(
            channels,
            scale=SCALE,
            dilation=dilation
        )

        self.post = nn.Sequential(

            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm1d(
                channels
            )

        )

        self.se = SEBlock(
            channels
        )

        self.activation = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        residual = x

        x = self.pre(x)

        x = self.res2net(x)

        x = self.post(x)

        x = self.se(x)

        x = x + residual

        return self.activation(x)


# =====================================================================
# STEP 11 — ATTENTIVE STATISTICS POOLING
# =====================================================================

class AttentiveStatisticsPooling(nn.Module):

    def __init__(
        self,
        channels,
        attention_channels
    ):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Conv1d(
                channels * 3,
                attention_channels,
                kernel_size=1
            ),

            nn.Tanh(),

            nn.Conv1d(
                attention_channels,
                channels,
                kernel_size=1
            )

        )

    def forward(self, x):

        """
        Input:
            [B, C, T]

        Output:
            [B, 2*C]

        The model learns which frames are important for
        speaker identity.
        """

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        std = x.std(
            dim=-1,
            keepdim=True,
            unbiased=False
        ).clamp(min=1e-5)

        mean_context = mean.expand_as(x)

        std_context = std.expand_as(x)

        context = torch.cat(
            [
                x,
                mean_context,
                std_context
            ],
            dim=1
        )

        attention = self.attention(
            context
        )

        attention = F.softmax(
            attention,
            dim=-1
        )

        weighted_mean = torch.sum(
            x * attention,
            dim=-1
        )

        weighted_second_moment = torch.sum(
            x.square() * attention,
            dim=-1
        )

        weighted_std = torch.sqrt(
            (
                weighted_second_moment
                - weighted_mean.square()
            ).clamp(min=1e-5)
        )

        pooled = torch.cat(
            [
                weighted_mean,
                weighted_std
            ],
            dim=1
        )

        return pooled


# =====================================================================
# STEP 12 — ECAPA-TDNN SPEAKER ENCODER
# =====================================================================

class ECAPATDNN(nn.Module):

    def __init__(
        self,
        num_speakers
    ):

        super().__init__()

        self.frontend = LogMelFrontend()

        self.stem = TDNNBlock(
            N_MELS,
            CHANNELS,
            kernel_size=5
        )

        self.block1 = SERes2NetBlock(
            CHANNELS,
            dilation=2
        )

        self.block2 = SERes2NetBlock(
            CHANNELS,
            dilation=3
        )

        self.block3 = SERes2NetBlock(
            CHANNELS,
            dilation=4
        )

        # Multi-layer feature aggregation.
        self.mfa = TDNNBlock(
            CHANNELS * 3,
            CHANNELS,
            kernel_size=1
        )

        self.pooling = AttentiveStatisticsPooling(
            CHANNELS,
            ATTENTION_CHANNELS
        )

        self.pooling_bn = nn.BatchNorm1d(
            CHANNELS * 2
        )

        self.embedding_layer = nn.Linear(
            CHANNELS * 2,
            EMBED_DIM
        )

        self.embedding_bn = nn.BatchNorm1d(
            EMBED_DIM
        )

        # Speaker classifier.
        self.classifier = AAMSoftmax(
            EMBED_DIM,
            num_speakers
        )

    def forward(self, waveform):

        # Acoustic frontend.
        x = self.frontend(waveform)

        # TDNN.
        x = self.stem(x)

        # Multi-scale temporal feature extraction.
        out1 = self.block1(x)

        out2 = self.block2(out1)

        out3 = self.block3(out2)

        # Multi-layer aggregation.
        x = torch.cat(
            [
                out1,
                out2,
                out3
            ],
            dim=1
        )

        x = self.mfa(x)

        # Attentive statistics pooling.
        x = self.pooling(x)

        x = self.pooling_bn(x)

        # Speaker embedding.
        embedding = self.embedding_layer(x)

        embedding = self.embedding_bn(
            embedding
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1
        )

        return embedding


# =====================================================================
# STEP 13 — ADDITIVE ANGULAR MARGIN SOFTMAX
# =====================================================================

class AAMSoftmax(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_classes,
        margin=0.2,
        scale=30.0
    ):

        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(
                num_classes,
                embedding_dim
            )
        )

        nn.init.xavier_uniform_(
            self.weight
        )

        self.margin = margin

        self.scale = scale

    def forward(
        self,
        embeddings,
        labels
    ):

        # Normalize weights and embeddings.
        normalized_embeddings = F.normalize(
            embeddings,
            dim=1
        )

        normalized_weights = F.normalize(
            self.weight,
            dim=1
        )

        # Cosine similarity.
        cosine = F.linear(
            normalized_embeddings,
            normalized_weights
        )

        cosine = cosine.clamp(
            -1.0 + 1e-7,
            1.0 - 1e-7
        )

        sine = torch.sqrt(
            1.0 - cosine.square()
        )

        cos_m = np.cos(
            self.margin
        )

        sin_m = np.sin(
            self.margin
        )

        # cos(theta + margin).
        phi = (
            cosine * cos_m
            - sine * sin_m
        )

        one_hot = torch.zeros_like(
            cosine
        )

        one_hot.scatter_(
            1,
            labels.view(-1, 1),
            1.0
        )

        # Apply angular margin to target class only.
        output = (
            one_hot * phi
            + (1.0 - one_hot) * cosine
        )

        return output * self.scale


# =====================================================================
# STEP 14 — TRAINING AND VALIDATION
# =====================================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    losses = []

    correct = 0

    total = 0

    for waveform, labels in loader:

        waveform = waveform.to(DEVICE)

        labels = labels.to(DEVICE)

        embeddings = model(waveform)

        logits = model.classifier(
            embeddings,
            labels
        )

        loss = F.cross_entropy(
            logits,
            labels
        )

        losses.append(
            loss.item()
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        float(np.mean(losses)),
        correct / max(total, 1)
    )


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

    print("\nStarting ECAPA-TDNN training...\n")

    for epoch in range(EPOCHS):

        model.train()

        losses = []

        correct = 0

        total = 0

        for waveform, labels in train_loader:

            waveform = waveform.to(DEVICE)

            labels = labels.to(DEVICE)

            embeddings = model(waveform)

            logits = model.classifier(
                embeddings,
                labels
            )

            loss = F.cross_entropy(
                logits,
                labels
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

            losses.append(
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
            np.mean(losses)
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
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
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
                    "lesson48_best.pt"
                )
            )

        else:

            no_improvement += 1

        if no_improvement >= patience:

            print("\nEarly stopping activated.")

            break

    return history


# =====================================================================
# STEP 15 — EXTRACT SPEAKER EMBEDDINGS
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

        embeddings = model(waveform)

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
# STEP 16 — SPEAKER SIMILARITY
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


def calculate_similarity_matrix(
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
# STEP 17 — REFERENCE WAV COMPARISON
# =====================================================================

@torch.no_grad()
def extract_single_embedding(
    model,
    path
):

    model.eval()

    audio = load_audio(path)

    if len(audio) < NUM_SAMPLES:

        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )

    else:

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

    embedding = model(
        waveform
    )

    return embedding[0].cpu().numpy()


def compare_reference_recordings(
    model,
    val_samples,
    speaker_to_id
):

    references = {}

    for path, speaker_id in val_samples:

        if speaker_id not in references:

            references[speaker_id] = path

    embeddings = {}

    for speaker_id, path in references.items():

        embeddings[speaker_id] = (
            extract_single_embedding(
                model,
                path
            )
        )

    print("\nREFERENCE RECORDING COMPARISON")

    print("=" * 72)

    for speaker_id, embedding in embeddings.items():

        print(
            f"\nSpeaker {speaker_id}"
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
# STEP 18 — VISUALIZATION
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
        "Lesson 48 — ECAPA-TDNN Training Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson48_training_loss.png"
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
        "Lesson 48 — Speaker Classification Accuracy"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson48_accuracy.png"
        )
    )

    plt.show()


def plot_embeddings(
    embeddings,
    labels
):

    reduced = PCA(
        n_components=2
    ).fit_transform(
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
        "Lesson 48 — ECAPA-TDNN Speaker Embeddings"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson48_embedding_pca.png"
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

    for i in range(len(matrix)):

        for j in range(len(matrix)):

            plt.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.xticks(
        range(len(matrix)),
        [f"Speaker {i}" for i in range(len(matrix))]
    )

    plt.yticks(
        range(len(matrix)),
        [f"Speaker {i}" for i in range(len(matrix))]
    )

    plt.title(
        "Lesson 48 — Speaker Similarity Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson48_similarity_matrix.png"
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
        "Lesson 48 — Speaker Verification"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson48_similarity_distribution.png"
        )
    )

    plt.show()


# =====================================================================
# STEP 19 — MAIN
# =====================================================================

def main():

    print("=" * 72)

    print(
        "LESSON 48 — ECAPA-TDNN SPEAKER ENCODER"
    )

    print("=" * 72)

    print(f"\nDevice: {DEVICE}")

    print(f"Sample rate: {SAMPLE_RATE}")

    print(f"Audio duration: {AUDIO_DURATION}")

    print(f"Mel bins: {N_MELS}")

    print(f"Channels: {CHANNELS}")

    print(f"Embedding dimension: {EMBED_DIM}")

    print(f"Res2Net scale: {SCALE}")

    print(f"AAM margin: {AAM_MARGIN}")

    print(f"AAM scale: {AAM_SCALE}")

    # ---------------------------------------------------------------
    # Discover real recordings.
    # ---------------------------------------------------------------

    speakers = discover_recordings(
        DATASET_DIR
    )

    synthetic_demo = False

    if not speakers:

        if not USE_SYNTHETIC_DEMO_IF_NO_DATA:

            raise FileNotFoundError(
                f"No WAV recordings found in {DATASET_DIR}"
            )

        root = create_synthetic_dataset()

        speakers = discover_recordings(
            root
        )

        synthetic_demo = True

    print(
        "\nDataset type:",
        "SYNTHETIC DEMONSTRATION"
        if synthetic_demo
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
    # Split recordings.
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
    # Dataset and dataloaders.
    # ---------------------------------------------------------------

    train_dataset = AudioDataset(
        train_samples,
        training=True
    )

    val_dataset = AudioDataset(
        val_samples,
        training=False
    )

    batch_size = min(
        BATCH_SIZE,
        len(train_dataset)
    )

    if batch_size < 2:

        raise ValueError(
            "At least two training recordings are required."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # ---------------------------------------------------------------
    # Build ECAPA-TDNN.
    # ---------------------------------------------------------------

    model = ECAPATDNN(
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
    # Load best checkpoint.
    # ---------------------------------------------------------------

    model.load_state_dict(
        torch.load(
            os.path.join(
                OUTPUT_DIR,
                "lesson48_best.pt"
            ),
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
    # Similarity evaluation.
    # ---------------------------------------------------------------

    intra, inter = calculate_similarity(
        embeddings,
        labels
    )

    matrix = calculate_similarity_matrix(
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
    # Reference recording comparison.
    # ---------------------------------------------------------------

    compare_reference_recordings(
        model,
        val_samples,
        speaker_to_id
    )

    # ---------------------------------------------------------------
    # Visualizations.
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

        "architecture": "ECAPA-TDNN-inspired",

        "sample_rate": SAMPLE_RATE,

        "n_mels": N_MELS,

        "channels": CHANNELS,

        "embedding_dim": EMBED_DIM,

        "num_speakers": num_speakers,

        "speaker_to_id": speaker_to_id,

        "synthetic_demo": synthetic_demo,

        "aam_margin": AAM_MARGIN,

        "aam_scale": AAM_SCALE

    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "lesson48_metadata.json"
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

    print("LESSON 48 COMPLETED")

    print("=" * 72)

    print(
        f"\nSynthetic demonstration: {synthetic_demo}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print("\nGenerated files:")

    print("  lesson48_best.pt")

    print("  lesson48_metadata.json")

    print("  lesson48_training_loss.png")

    print("  lesson48_accuracy.png")

    print("  lesson48_embedding_pca.png")

    print("  lesson48_similarity_matrix.png")

    print("  lesson48_similarity_distribution.png")


if __name__ == "__main__":

    main()