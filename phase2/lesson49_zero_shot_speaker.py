"""
========================================================================
VOICE CLONING COURSE
LESSON 49 — ZERO-SHOT SPEAKER EMBEDDING EXTRACTION
========================================================================

PURPOSE
-------
Build a zero-shot speaker embedding and verification pipeline.

WHAT YOU LEARN
--------------

STEP 1:
    Generate or load speaker recordings.

STEP 2:
    Separate training speakers from unseen speakers.

STEP 3:
    Train an ECAPA-TDNN-inspired speaker encoder.

STEP 4:
    Extract speaker embeddings without speaker labels.

STEP 5:
    Register unseen speakers using reference audio.

STEP 6:
    Combine multiple reference embeddings.

STEP 7:
    Calculate cosine similarity.

STEP 8:
    Perform speaker verification.

STEP 9:
    Evaluate unseen-speaker identification.

STEP 10:
    Visualize speaker embeddings and similarity distributions.

IMPORTANT
---------
Synthetic audio is only for testing the software pipeline.

Synthetic speakers do not represent genuine human voice identity.

Real zero-shot generalization requires diverse real speech recordings.

The model must never receive unseen-speaker recordings during training.

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

SEED = 49

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000

DURATION = 3.0

NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 512

HOP_LENGTH = 160

WIN_LENGTH = 400

N_MELS = 80

CHANNELS = 128

EMBED_DIM = 192

SCALE = 8

NUM_TRAIN_SPEAKERS = 4

NUM_UNSEEN_SPEAKERS = 4

RECORDINGS_PER_SPEAKER = 24

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 3e-4

OUTPUT_DIR = "lesson49_outputs"

DATA_DIR = os.path.join(
    OUTPUT_DIR,
    "synthetic_dataset"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================================
# STEP 1 — SYNTHETIC SPEAKER AUDIO
# =====================================================================

def generate_speaker_audio(
    speaker_id,
    duration=3.0
):

    """
    Generate artificial speaker-dependent signals.

    Different speakers receive different acoustic characteristics.

    This is a controlled software demonstration, not natural speech.
    """

    t = np.arange(
        int(SAMPLE_RATE * duration)
    ) / SAMPLE_RATE

    # Different speaker characteristics.
    base_frequencies = [
        110, 145, 185, 225,
        130, 165, 205, 245
    ]

    formants = [
        [500, 1500, 2500],
        [700, 1800, 2800],
        [400, 1200, 2200],
        [800, 2000, 3000],
        [600, 1300, 2600],
        [450, 1700, 2900],
        [750, 1600, 2400],
        [550, 2100, 3200]
    ]

    f0 = base_frequencies[speaker_id]

    # Small recording-to-recording variation.
    f0 *= random.uniform(
        0.95,
        1.05
    )

    signal = np.zeros_like(t)

    for harmonic in range(1, 35):

        frequency = f0 * harmonic

        if frequency >= SAMPLE_RATE / 2:
            break

        spectral_weight = 0.0

        for formant in formants[speaker_id]:

            spectral_weight += np.exp(
                -0.5 * (
                    (frequency - formant) / 250
                ) ** 2
            )

        amplitude = (
            0.15 + spectral_weight
        ) / harmonic

        signal += amplitude * np.sin(
            2 * np.pi * frequency * t
        )

    # Add temporal variation.
    modulation = (
        0.8
        + 0.2 * np.sin(
            2 * np.pi * random.uniform(2, 4) * t
        )
    )

    signal *= modulation

    # Add background noise.
    noise = np.random.normal(
        0,
        0.015,
        len(signal)
    )

    signal += noise

    signal /= (
        np.max(np.abs(signal)) + 1e-8
    )

    return signal.astype(np.float32)


def create_dataset():

    print("\nGenerating demonstration dataset...")

    for speaker_id in range(
        NUM_TRAIN_SPEAKERS + NUM_UNSEEN_SPEAKERS
    ):

        if speaker_id < NUM_TRAIN_SPEAKERS:

            split = "train"

        else:

            split = "unseen"

        speaker_dir = os.path.join(
            DATA_DIR,
            split,
            f"speaker_{speaker_id}"
        )

        os.makedirs(
            speaker_dir,
            exist_ok=True
        )

        for recording_id in range(
            RECORDINGS_PER_SPEAKER
        ):

            audio = generate_speaker_audio(
                speaker_id
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


# =====================================================================
# STEP 2 — AUDIO LOADING
# =====================================================================

def load_audio(path):

    audio, sr = sf.read(
        path,
        dtype="float32"
    )

    if audio.ndim == 2:

        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:

        from math import gcd

        divisor = gcd(
            int(sr),
            SAMPLE_RATE
        )

        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            sr // divisor
        )

    audio = audio - np.mean(audio)

    return audio.astype(np.float32)


# =====================================================================
# STEP 3 — DATASET DISCOVERY
# =====================================================================

def discover_split(split):

    root = os.path.join(
        DATA_DIR,
        split
    )

    samples = []

    speaker_names = sorted(
        os.listdir(root)
    )

    speaker_to_id = {
        name: index
        for index, name in enumerate(
            speaker_names
        )
    }

    for name in speaker_names:

        speaker_dir = os.path.join(
            root,
            name
        )

        if not os.path.isdir(speaker_dir):
            continue

        for filename in sorted(
            os.listdir(speaker_dir)
        ):

            if filename.endswith(".wav"):

                samples.append(
                    (
                        os.path.join(
                            speaker_dir,
                            filename
                        ),
                        speaker_to_id[name]
                    )
                )

    return samples, speaker_to_id


# =====================================================================
# STEP 4 — DATASET
# =====================================================================

class SpeakerDataset(Dataset):

    def __init__(
        self,
        samples,
        training=False
    ):

        self.samples = samples

        self.training = training

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        path, speaker_id = self.samples[index]

        audio = load_audio(path)

        if len(audio) < NUM_SAMPLES:

            audio = np.pad(
                audio,
                (0, NUM_SAMPLES - len(audio))
            )

        if len(audio) > NUM_SAMPLES:

            start = random.randint(
                0,
                len(audio) - NUM_SAMPLES
            ) if self.training else (
                len(audio) - NUM_SAMPLES
            ) // 2

            audio = audio[
                start:start + NUM_SAMPLES
            ]

        if self.training:

            audio *= random.uniform(
                0.8,
                1.2
            )

            if random.random() < 0.3:

                audio += np.random.normal(
                    0,
                    0.003,
                    len(audio)
                )

        audio = np.clip(
            audio,
            -1,
            1
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
# STEP 5 — LOG-MEL FRONTEND
# =====================================================================

def hz_to_mel(hz):

    return 2595 * np.log10(
        1 + hz / 700
    )


def mel_to_hz(mel):

    return 700 * (
        10 ** (mel / 2595) - 1
    )


def make_mel_filter():

    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1
    )

    points = np.linspace(
        hz_to_mel(50),
        hz_to_mel(7600),
        N_MELS + 2
    )

    points = mel_to_hz(points)

    filters = np.zeros(
        (N_MELS, len(frequencies)),
        dtype=np.float32
    )

    for i in range(N_MELS):

        left, center, right = points[
            i:i + 3
        ]

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


class LogMel(nn.Module):

    def __init__(self):

        super().__init__()

        self.register_buffer(
            "filters",
            make_mel_filter()
        )

        self.register_buffer(
            "window",
            torch.hann_window(WIN_LENGTH)
        )

    def forward(self, audio):

        spec = torch.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            return_complex=True
        )

        magnitude = spec.abs()

        mel = torch.matmul(
            self.filters,
            magnitude
        )

        x = torch.log(
            mel.clamp(min=1e-5)
        )

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        std = x.std(
            dim=-1,
            keepdim=True
        ).clamp(min=1e-5)

        return (
            x - mean
        ) / std


# =====================================================================
# STEP 6 — TDNN
# =====================================================================

class TDNN(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=1,
        dilation=1
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel,
                padding=(kernel // 2) * dilation,
                dilation=dilation,
                bias=False
            ),

            nn.BatchNorm1d(
                out_channels
            ),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.net(x)


# =====================================================================
# STEP 7 — RES2NET
# =====================================================================

class Res2Net(nn.Module):

    def __init__(
        self,
        channels,
        scale=8,
        dilation=2
    ):

        super().__init__()

        self.scale = scale

        self.width = channels // scale

        self.blocks = nn.ModuleList()

        for _ in range(scale - 1):

            self.blocks.append(

                nn.Sequential(

                    nn.Conv1d(
                        self.width,
                        self.width,
                        3,
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

        outputs = [chunks[0]]

        current = None

        for i in range(1, self.scale):

            if i == 1:

                current = chunks[i]

            else:

                current = current + chunks[i]

            current = self.blocks[i - 1](
                current
            )

            outputs.append(current)

        return torch.cat(
            outputs,
            dim=1
        )


# =====================================================================
# STEP 8 — SE ATTENTION
# =====================================================================

class SEBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        hidden = max(
            channels // 8,
            8
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Conv1d(
            channels,
            hidden,
            1
        )

        self.fc2 = nn.Conv1d(
            hidden,
            channels,
            1
        )

    def forward(self, x):

        scale = self.pool(x)

        scale = F.relu(
            self.fc1(scale)
        )

        scale = torch.sigmoid(
            self.fc2(scale)
        )

        return x * scale


# =====================================================================
# STEP 9 — SE-RES2NET
# =====================================================================

class SERes2Net(nn.Module):

    def __init__(
        self,
        channels,
        dilation
    ):

        super().__init__()

        self.pre = TDNN(
            channels,
            channels
        )

        self.res2net = Res2Net(
            channels,
            dilation=dilation
        )

        self.post = nn.Sequential(

            nn.Conv1d(
                channels,
                channels,
                1,
                bias=False
            ),

            nn.BatchNorm1d(
                channels
            )

        )

        self.se = SEBlock(
            channels
        )

    def forward(self, x):

        residual = x

        x = self.pre(x)

        x = self.res2net(x)

        x = self.post(x)

        x = self.se(x)

        return F.relu(
            x + residual
        )


# =====================================================================
# STEP 10 — ATTENTIVE STATISTICS POOLING
# =====================================================================

class AttentivePooling(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Conv1d(
                channels * 3,
                128,
                1
            ),

            nn.Tanh(),

            nn.Conv1d(
                128,
                channels,
                1
            )

        )

    def forward(self, x):

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        std = x.std(
            dim=-1,
            keepdim=True,
            unbiased=False
        ).clamp(min=1e-5)

        context = torch.cat(
            [
                x,
                mean.expand_as(x),
                std.expand_as(x)
            ],
            dim=1
        )

        weights = F.softmax(
            self.attention(context),
            dim=-1
        )

        weighted_mean = (
            x * weights
        ).sum(dim=-1)

        weighted_second = (
            x.square() * weights
        ).sum(dim=-1)

        weighted_std = torch.sqrt(
            (
                weighted_second
                - weighted_mean.square()
            ).clamp(min=1e-5)
        )

        return torch.cat(
            [
                weighted_mean,
                weighted_std
            ],
            dim=1
        )


# =====================================================================
# STEP 11 — ECAPA-TDNN
# =====================================================================

class ECAPATDNN(nn.Module):

    def __init__(
        self,
        num_speakers
    ):

        super().__init__()

        self.frontend = LogMel()

        self.stem = TDNN(
            N_MELS,
            CHANNELS,
            kernel=5
        )

        self.block1 = SERes2Net(
            CHANNELS,
            2
        )

        self.block2 = SERes2Net(
            CHANNELS,
            3
        )

        self.block3 = SERes2Net(
            CHANNELS,
            4
        )

        self.mfa = TDNN(
            CHANNELS * 3,
            CHANNELS
        )

        self.pool = AttentivePooling(
            CHANNELS
        )

        self.pool_bn = nn.BatchNorm1d(
            CHANNELS * 2
        )

        self.embedding = nn.Linear(
            CHANNELS * 2,
            EMBED_DIM
        )

        self.embedding_bn = nn.BatchNorm1d(
            EMBED_DIM
        )

        self.classifier = nn.Linear(
            EMBED_DIM,
            num_speakers
        )

    def extract_embedding(self, audio):

        x = self.frontend(audio)

        x = self.stem(x)

        a = self.block1(x)

        b = self.block2(a)

        c = self.block3(b)

        x = torch.cat(
            [a, b, c],
            dim=1
        )

        x = self.mfa(x)

        x = self.pool(x)

        x = self.pool_bn(x)

        x = self.embedding(x)

        x = self.embedding_bn(x)

        return F.normalize(
            x,
            dim=-1
        )

    def forward(self, audio):

        embedding = self.extract_embedding(
            audio
        )

        logits = self.classifier(
            embedding
        )

        return logits, embedding


# =====================================================================
# STEP 12 — TRAINING
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

    history = {
        "train": [],
        "validation": []
    }

    best_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        losses = []

        for audio, labels in train_loader:

            audio = audio.to(DEVICE)

            labels = labels.to(DEVICE)

            logits, _ = model(audio)

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

        train_loss = float(
            np.mean(losses)
        )

        model.eval()

        val_losses = []

        with torch.no_grad():

            for audio, labels in val_loader:

                audio = audio.to(DEVICE)

                labels = labels.to(DEVICE)

                logits, _ = model(audio)

                loss = F.cross_entropy(
                    logits,
                    labels
                )

                val_losses.append(
                    loss.item()
                )

        val_loss = float(
            np.mean(val_losses)
        )

        history["train"].append(
            train_loss
        )

        history["validation"].append(
            val_loss
        )

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                os.path.join(
                    OUTPUT_DIR,
                    "lesson49_best.pt"
                )
            )

    return history


# =====================================================================
# STEP 13 — EXTRACT EMBEDDINGS
# =====================================================================

@torch.no_grad()
def extract_embedding(
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

    embedding = model.extract_embedding(
        waveform
    )

    return embedding[0].cpu().numpy()


# =====================================================================
# STEP 14 — MULTI-REFERENCE ENROLLMENT
# =====================================================================

def enroll_speaker(
    model,
    recording_paths
):

    """
    Create a speaker identity representation.

    Multiple reference recordings are averaged and normalized.

    This improves stability when the recordings contain
    different phonetic content or speaking conditions.
    """

    embeddings = []

    for path in recording_paths:

        embedding = extract_embedding(
            model,
            path
        )

        embeddings.append(
            embedding
        )

    embeddings = np.stack(
        embeddings
    )

    # Mean embedding.
    speaker_embedding = embeddings.mean(
        axis=0
    )

    # L2 normalization.
    speaker_embedding /= (
        np.linalg.norm(speaker_embedding) + 1e-8
    )

    return speaker_embedding


# =====================================================================
# STEP 15 — COSINE SIMILARITY
# =====================================================================

def cosine_similarity(a, b):

    return float(
        np.dot(a, b)
        / (
            np.linalg.norm(a)
            * np.linalg.norm(b)
            + 1e-8
        )
    )


# =====================================================================
# STEP 16 — ZERO-SHOT IDENTIFICATION
# =====================================================================

def evaluate_unseen_speakers(
    model,
    unseen_root
):

    print("\n" + "=" * 72)

    print("ZERO-SHOT SPEAKER IDENTIFICATION")

    print("=" * 72)

    speaker_dirs = sorted(
        os.listdir(unseen_root)
    )

    enrollment = {}

    test_samples = []

    for speaker_name in speaker_dirs:

        speaker_dir = os.path.join(
            unseen_root,
            speaker_name
        )

        recordings = sorted(
            [
                os.path.join(
                    speaker_dir,
                    f
                )
                for f in os.listdir(speaker_dir)
                if f.endswith(".wav")
            ]
        )

        if len(recordings) < 2:
            continue

        # Use first 4 recordings for enrollment.
        reference_paths = recordings[:4]

        # Remaining recordings for evaluation.
        evaluation_paths = recordings[4:]

        enrollment[speaker_name] = enroll_speaker(
            model,
            reference_paths
        )

        for path in evaluation_paths:

            test_samples.append(
                (path, speaker_name)
            )

    correct = 0

    total = 0

    for path, true_speaker in test_samples:

        test_embedding = extract_embedding(
            model,
            path
        )

        scores = {}

        for speaker_name, reference in enrollment.items():

            scores[speaker_name] = cosine_similarity(
                test_embedding,
                reference
            )

        predicted = max(
            scores,
            key=scores.get
        )

        is_correct = (
            predicted == true_speaker
        )

        correct += int(is_correct)

        total += 1

        print(
            f"True: {true_speaker:12s} | "
            f"Predicted: {predicted:12s} | "
            f"Correct: {is_correct}"
        )

    accuracy = correct / max(
        total,
        1
    )

    print("\nZero-shot identification accuracy:")

    print(
        f"{accuracy:.4f}"
    )

    return enrollment, accuracy


# =====================================================================
# STEP 17 — SAME / DIFFERENT SPEAKER VERIFICATION
# =====================================================================

def evaluate_verification(
    model,
    unseen_root,
    enrollment
):

    genuine_scores = []

    impostor_scores = []

    speaker_names = list(
        enrollment.keys()
    )

    for speaker_name in speaker_names:

        speaker_dir = os.path.join(
            unseen_root,
            speaker_name
        )

        recordings = sorted(
            [
                os.path.join(
                    speaker_dir,
                    f
                )
                for f in os.listdir(speaker_dir)
                if f.endswith(".wav")
            ]
        )

        for path in recordings[4:]:

            test_embedding = extract_embedding(
                model,
                path
            )

            for other_name, reference in enrollment.items():

                score = cosine_similarity(
                    test_embedding,
                    reference
                )

                if speaker_name == other_name:

                    genuine_scores.append(score)

                else:

                    impostor_scores.append(score)

    genuine_scores = np.array(
        genuine_scores
    )

    impostor_scores = np.array(
        impostor_scores
    )

    print("\n" + "=" * 72)

    print("ZERO-SHOT VERIFICATION STATISTICS")

    print("=" * 72)

    print(
        f"\nMean genuine similarity: "
        f"{genuine_scores.mean():.4f}"
    )

    print(
        f"Mean impostor similarity: "
        f"{impostor_scores.mean():.4f}"
    )

    print(
        f"Separation: "
        f"{genuine_scores.mean() - impostor_scores.mean():.4f}"
    )

    # Illustrative threshold only.
    # A real deployment threshold requires a representative
    # development set and operating-point calibration.

    threshold = (
        genuine_scores.mean()
        + impostor_scores.mean()
    ) / 2

    false_accept = np.mean(
        impostor_scores >= threshold
    )

    false_reject = np.mean(
        genuine_scores < threshold
    )

    print(
        f"\nIllustrative threshold: {threshold:.4f}"
    )

    print(
        f"False acceptance rate: {false_accept:.4f}"
    )

    print(
        f"False rejection rate: {false_reject:.4f}"
    )

    return (
        genuine_scores,
        impostor_scores,
        threshold
    )


# =====================================================================
# STEP 18 — VISUALIZATION
# =====================================================================

def plot_training(history):

    plt.figure(figsize=(9, 5))

    plt.plot(
        history["train"],
        label="Training"
    )

    plt.plot(
        history["validation"],
        label="Validation"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Cross-Entropy Loss")

    plt.title(
        "Lesson 49 — Training"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson49_training.png"
        )
    )

    plt.show()


def plot_unseen_embeddings(
    model,
    unseen_root
):

    all_embeddings = []

    all_labels = []

    for label, speaker_name in enumerate(
        sorted(os.listdir(unseen_root))
    ):

        speaker_dir = os.path.join(
            unseen_root,
            speaker_name
        )

        recordings = sorted(
            [
                os.path.join(
                    speaker_dir,
                    f
                )
                for f in os.listdir(speaker_dir)
                if f.endswith(".wav")
            ]
        )

        for path in recordings:

            all_embeddings.append(
                extract_embedding(
                    model,
                    path
                )
            )

            all_labels.append(label)

    all_embeddings = np.array(
        all_embeddings
    )

    all_labels = np.array(
        all_labels
    )

    reduced = PCA(
        n_components=2
    ).fit_transform(
        all_embeddings
    )

    plt.figure(figsize=(9, 7))

    for speaker_id in np.unique(
        all_labels
    ):

        mask = (
            all_labels == speaker_id
        )

        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=f"Unseen speaker {speaker_id}",
            alpha=0.7
        )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title(
        "Lesson 49 — Unseen Speaker Embeddings"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson49_unseen_embeddings.png"
        )
    )

    plt.show()


def plot_verification(
    genuine,
    impostor
):

    plt.figure(figsize=(9, 5))

    plt.hist(
        genuine,
        bins=25,
        alpha=0.6,
        label="Genuine"
    )

    plt.hist(
        impostor,
        bins=25,
        alpha=0.6,
        label="Impostor"
    )

    plt.xlabel("Cosine Similarity")

    plt.ylabel("Frequency")

    plt.title(
        "Lesson 49 — Zero-Shot Speaker Verification"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson49_verification.png"
        )
    )

    plt.show()


# =====================================================================
# STEP 19 — MAIN
# =====================================================================

def main():

    print("=" * 72)

    print(
        "LESSON 49 — ZERO-SHOT SPEAKER EMBEDDING EXTRACTION"
    )

    print("=" * 72)

    print(f"\nDevice: {DEVICE}")

    print(f"Sample rate: {SAMPLE_RATE}")

    print(f"Embedding dimension: {EMBED_DIM}")

    print(f"Training speakers: {NUM_TRAIN_SPEAKERS}")

    print(f"Unseen speakers: {NUM_UNSEEN_SPEAKERS}")

    # ---------------------------------------------------------------
    # Generate synthetic dataset.
    # ---------------------------------------------------------------

    if not os.path.exists(
        DATA_DIR
    ):

        create_dataset()

    else:

        if not os.path.exists(
            os.path.join(DATA_DIR, "train")
        ):

            create_dataset()

    # ---------------------------------------------------------------
    # Load training recordings.
    # ---------------------------------------------------------------

    train_samples, train_mapping = discover_split(
        "train"
    )

    print("\nTraining speaker mapping:")

    for name, idx in train_mapping.items():

        print(
            f"  {name} -> {idx}"
        )

    print(
        f"\nTraining recordings: {len(train_samples)}"
    )

    # ---------------------------------------------------------------
    # Create validation split by recording.
    # ---------------------------------------------------------------

    random.shuffle(
        train_samples
    )

    train_set = []

    val_set = []

    for speaker_id in range(
        NUM_TRAIN_SPEAKERS
    ):

        speaker_samples = [
            item
            for item in train_samples
            if item[1] == speaker_id
        ]

        random.shuffle(
            speaker_samples
        )

        split = max(
            1,
            int(len(speaker_samples) * 0.2)
        )

        val_set.extend(
            speaker_samples[:split]
        )

        train_set.extend(
            speaker_samples[split:]
        )

    train_dataset = SpeakerDataset(
        train_set,
        training=True
    )

    val_dataset = SpeakerDataset(
        val_set,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ---------------------------------------------------------------
    # Build model.
    # ---------------------------------------------------------------

    model = ECAPATDNN(
        NUM_TRAIN_SPEAKERS
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: {parameters:,}"
    )

    # ---------------------------------------------------------------
    # Train using training speakers only.
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
                "lesson49_best.pt"
            ),
            map_location=DEVICE,
            weights_only=True
        )
    )

    # ---------------------------------------------------------------
    # Verify no unseen speakers were used in training.
    # ---------------------------------------------------------------

    unseen_root = os.path.join(
        DATA_DIR,
        "unseen"
    )

    print("\nTraining speakers:")

    print(
        sorted(train_mapping.keys())
    )

    print("\nUnseen evaluation speakers:")

    print(
        sorted(os.listdir(unseen_root))
    )

    # ---------------------------------------------------------------
    # Zero-shot identification.
    # ---------------------------------------------------------------

    enrollment, accuracy = evaluate_unseen_speakers(
        model,
        unseen_root
    )

    # ---------------------------------------------------------------
    # Speaker verification.
    # ---------------------------------------------------------------

    genuine, impostor, threshold = evaluate_verification(
        model,
        unseen_root,
        enrollment
    )

    # ---------------------------------------------------------------
    # Visualizations.
    # ---------------------------------------------------------------

    plot_training(history)

    plot_unseen_embeddings(
        model,
        unseen_root
    )

    plot_verification(
        genuine,
        impostor
    )

    # ---------------------------------------------------------------
    # Save model and metadata.
    # ---------------------------------------------------------------

    metadata = {

        "lesson": 49,

        "architecture": "ECAPA-TDNN-inspired",

        "embedding_dimension": EMBED_DIM,

        "sample_rate": SAMPLE_RATE,

        "training_speakers": sorted(
            train_mapping.keys()
        ),

        "unseen_speakers": sorted(
            os.listdir(unseen_root)
        ),

        "zero_shot_identification_accuracy": float(
            accuracy
        ),

        "illustrative_threshold": float(
            threshold
        ),

        "synthetic_demonstration": True

    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "lesson49_metadata.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # ---------------------------------------------------------------
    # Save unseen speaker enrollment.
    # ---------------------------------------------------------------

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "lesson49_speaker_enrollment.npz"
        ),
        **enrollment
    )

    print("\n" + "=" * 72)

    print("LESSON 49 COMPLETED")

    print("=" * 72)

    print("\nOutput directory:")

    print(OUTPUT_DIR)

    print("\nGenerated files:")

    print("  lesson49_best.pt")

    print("  lesson49_metadata.json")

    print("  lesson49_speaker_enrollment.npz")

    print("  lesson49_training.png")

    print("  lesson49_unseen_embeddings.png")

    print("  lesson49_verification.png")


if __name__ == "__main__":

    main()