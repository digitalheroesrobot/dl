"""
========================================================================
PHASE 2 — LESSON 66
SPEAKER VERIFICATION FROM SCRATCH
========================================================================

PURPOSE
-------
Use a trained speaker encoder to answer:

    "Are these two audio clips from the same speaker?"

This lesson converts the learned speaker embedding into a practical
speaker verification system.

NO PRETRAINED SPEAKER MODEL
---------------------------
This lesson does NOT use:

    - pretrained speaker encoders
    - SpeechBrain
    - pyannote
    - ECAPA-TDNN
    - x-vector models
    - Wav2Vec2
    - Whisper

We use our own speaker encoder checkpoint.

COURSE PROGRESSION
------------------

Lesson 63
    Speaker-conditioned TTS
        ↓
Lesson 64
    Speaker encoder trained with classification
        ↓
Lesson 65
    Speaker metric learning
        ↓
Lesson 66
    Speaker verification
        ↓
Lesson 67+
    Reference-audio voice cloning

CORE IDEA
---------

Audio A
   ↓
Speaker Encoder
   ↓
Embedding A

Audio B
   ↓
Speaker Encoder
   ↓
Embedding B

                 cosine similarity
                         ↓
                 ┌───────────────┐
                 │ same speaker? │
                 └───────────────┘

If:

    similarity >= threshold

then:

    SAME SPEAKER

otherwise:

    DIFFERENT SPEAKER


MATHEMATICAL IDEA
-----------------

Let:

    e_a = embedding of audio A
    e_b = embedding of audio B

The cosine similarity is:

              e_a · e_b
    s = -------------------------
        ||e_a|| ||e_b||

Because our encoder produces L2-normalized embeddings:

    ||e_a|| = ||e_b|| = 1

therefore:

    s = e_a · e_b

A high value means the embeddings point in a similar direction.

A low value means they are different.

VERIFICATION VS IDENTIFICATION
-------------------------------

IDENTIFICATION asks:

    "Which speaker is this?"

Example:

    audio → speaker_007

VERIFICATION asks:

    "Is this speaker_007?"

Example:

    reference audio
          +
    query audio
          ↓
    SAME / DIFFERENT

Verification is especially important for voice cloning because a
reference recording must provide a reliable representation of the
target speaker.

THRESHOLD
---------

We need a threshold T:

    similarity >= T
        → SAME

    similarity < T
        → DIFFERENT

But the threshold is not arbitrary.

We evaluate many positive and negative pairs and find a threshold
that gives a useful tradeoff between:

    False Acceptance
        and
    False Rejection

IMPORTANT METRICS
-----------------

True Positive:

    same speaker correctly accepted

False Positive:

    different speaker incorrectly accepted

True Negative:

    different speaker correctly rejected

False Negative:

    same speaker incorrectly rejected

FAR
---
False Acceptance Rate.

    FAR = false accepts / all negative pairs

FRR
---
False Rejection Rate.

    FRR = false rejects / all positive pairs

EER
---
Equal Error Rate.

The threshold where approximately:

    FAR ≈ FRR

A lower EER generally indicates better speaker discrimination.

WHAT YOU LEARN
---------------

STEP 1
    Load the trained speaker encoder.

STEP 2
    Convert real speech into embeddings.

STEP 3
    Construct positive and negative verification pairs.

STEP 4
    Calculate cosine similarity.

STEP 5
    Sweep verification thresholds.

STEP 6
    Calculate FAR and FRR.

STEP 7
    Estimate Equal Error Rate.

STEP 8
    Select a verification threshold.

STEP 9
    Visualize similarity distributions.

STEP 10
    Test individual audio pairs.

IMPORTANT
---------
This is an educational speaker verification system.

It is NOT a production biometric authentication system.

The dataset is small and the encoder is trained from scratch.

========================================================================
"""

import os
import random
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "speaker_dataset"

OUTPUT_DIR = ROOT / "lesson66_outputs"

# We first try the Lesson 65 checkpoint because Lesson 65 is the
# metric-learning speaker encoder.
METRIC_CHECKPOINT = (
    ROOT
    / "lesson65_outputs"
    / "speaker_metric_encoder_best.pt"
)

# Fallback to Lesson 64 if the Lesson 65 checkpoint does not exist.
CLASSIFICATION_CHECKPOINT = (
    ROOT
    / "lesson64_outputs"
    / "speaker_encoder_best.pt"
)

# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------

SAMPLE_RATE = 16000

CLIP_SECONDS = 2.0

CLIP_SAMPLES = int(
    SAMPLE_RATE * CLIP_SECONDS
)

N_FFT = 1024

HOP_LENGTH = 256

WIN_LENGTH = 1024

N_MELS = 80

FMIN = 50.0

FMAX = 7600.0

# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

NUM_POSITIVE_PAIRS = 500

NUM_NEGATIVE_PAIRS = 500

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# SEED
# ============================================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DIRECTORY
# ============================================================================

def ensure_directory(path):

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# WAV LOADER
# ============================================================================

def load_wav(path):

    import wave

    with wave.open(
        str(path),
        "rb",
    ) as wav:

        channels = wav.getnchannels()

        sample_width = wav.getsampwidth()

        sample_rate = wav.getframerate()

        frames = wav.getnframes()

        raw = wav.readframes(frames)

    if sample_width == 1:

        audio = np.frombuffer(
            raw,
            dtype=np.uint8,
        ).astype(
            np.float32
        )

        audio = (
            audio - 128.0
        ) / 128.0

    elif sample_width == 2:

        audio = np.frombuffer(
            raw,
            dtype=np.int16,
        ).astype(
            np.float32
        )

        audio /= 32768.0

    elif sample_width == 3:

        raw_bytes = np.frombuffer(
            raw,
            dtype=np.uint8,
        )

        usable = (
            len(raw_bytes) // 3
        ) * 3

        raw_bytes = raw_bytes[
            :usable
        ]

        samples = raw_bytes.reshape(
            -1,
            3,
        )

        values = (
            samples[:, 0].astype(
                np.int32
            )
            |
            (
                samples[:, 1].astype(
                    np.int32
                )
                << 8
            )
            |
            (
                samples[:, 2].astype(
                    np.int32
                )
                << 16
            )
        )

        negative = (
            values & 0x800000
        ) != 0

        values[negative] -= (
            0x1000000
        )

        audio = (
            values.astype(
                np.float32
            )
            / 8388608.0
        )

    elif sample_width == 4:

        audio = np.frombuffer(
            raw,
            dtype=np.int32,
        ).astype(
            np.float32
        )

        audio /= 2147483648.0

    else:

        raise ValueError(
            f"Unsupported WAV sample width: "
            f"{sample_width}"
        )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels,
        )

        audio = audio.mean(
            axis=1
        )

    return (
        audio.astype(
            np.float32
        ),
        sample_rate,
    )


# ============================================================================
# RESAMPLING
# ============================================================================

def resample_audio(
    audio,
    source_rate,
    target_rate,
):

    if source_rate == target_rate:

        return audio.astype(
            np.float32
        )

    if len(audio) < 2:

        return audio.astype(
            np.float32
        )

    duration = (
        len(audio)
        / float(source_rate)
    )

    target_length = max(
        1,
        int(
            round(
                duration
                * target_rate
            )
        ),
    )

    old_positions = np.arange(
        len(audio),
        dtype=np.float64,
    )

    new_positions = np.linspace(
        0,
        len(audio) - 1,
        target_length,
    )

    result = np.interp(
        new_positions,
        old_positions,
        audio,
    )

    return result.astype(
        np.float32
    )


# ============================================================================
# AUDIO NORMALIZATION
# ============================================================================

def normalize_audio(audio):

    audio = audio.astype(
        np.float32
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = audio / peak

    return audio


# ============================================================================
# FIXED AUDIO LENGTH
# ============================================================================

def crop_or_pad(
    audio,
    target_length,
):

    if len(audio) == target_length:

        return audio

    if len(audio) > target_length:

        max_start = (
            len(audio)
            - target_length
        )

        start = random.randint(
            0,
            max_start,
        )

        return audio[
            start:
            start + target_length
        ]

    result = np.zeros(
        target_length,
        dtype=np.float32,
    )

    start = (
        target_length
        - len(audio)
    ) // 2

    result[
        start:
        start + len(audio)
    ] = audio

    return result


# ============================================================================
# MEL CONVERSION
# ============================================================================

def hz_to_mel(hz):

    return (
        2595.0
        * np.log10(
            1.0 + hz / 700.0
        )
    )


def mel_to_hz(mel):

    return (
        700.0
        * (
            10.0 ** (
                mel / 2595.0
            )
            - 1.0
        )
    )


def create_mel_filterbank():

    n_freqs = (
        N_FFT // 2 + 1
    )

    mel_min = hz_to_mel(
        FMIN
    )

    mel_max = hz_to_mel(
        FMAX
    )

    mel_points = np.linspace(
        mel_min,
        mel_max,
        N_MELS + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    bins = np.floor(
        (N_FFT + 1)
        * hz_points
        / SAMPLE_RATE
    ).astype(int)

    filterbank = np.zeros(
        (
            N_MELS,
            n_freqs,
        ),
        dtype=np.float32,
    )

    for m in range(N_MELS):

        left = bins[m]

        center = bins[m + 1]

        right = bins[m + 2]

        left = max(
            0,
            min(
                left,
                n_freqs - 1,
            ),
        )

        center = max(
            0,
            min(
                center,
                n_freqs - 1,
            ),
        )

        right = max(
            0,
            min(
                right,
                n_freqs - 1,
            ),
        )

        if center > left:

            filterbank[
                m,
                left:center,
            ] = (
                np.arange(
                    left,
                    center,
                )
                - left
            ) / float(
                center - left
            )

        if right > center:

            filterbank[
                m,
                center:right,
            ] = (
                right
                - np.arange(
                    center,
                    right,
                )
            ) / float(
                right - center
            )

    return filterbank


MEL_FILTERBANK = (
    create_mel_filterbank()
)


def waveform_to_log_mel(
    audio,
):

    waveform = torch.from_numpy(
        audio
    ).float()

    window = torch.hann_window(
        WIN_LENGTH
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

    magnitude = torch.abs(
        stft
    )

    power = magnitude.pow(
        2.0
    )

    mel_filter = torch.from_numpy(
        MEL_FILTERBANK
    ).float()

    mel = torch.matmul(
        mel_filter,
        power,
    )

    mel = torch.clamp(
        mel,
        min=1e-8,
    )

    log_mel = torch.log(
        mel
    )

    log_mel = log_mel.numpy()

    mean = log_mel.mean()

    std = log_mel.std()

    log_mel = (
        log_mel - mean
    ) / (
        std + 1e-5
    )

    return log_mel.astype(
        np.float32
    )


# ============================================================================
# SPEAKER ENCODER
# ============================================================================

class SpeakerEncoder(
    nn.Module
):

    def __init__(
        self,
        embedding_dim=128,
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            N_MELS,
            128,
            kernel_size=5,
            padding=2,
        )

        self.bn1 = nn.BatchNorm1d(
            128
        )

        self.conv2 = nn.Conv1d(
            128,
            128,
            kernel_size=5,
            padding=2,
        )

        self.bn2 = nn.BatchNorm1d(
            128
        )

        self.conv3 = nn.Conv1d(
            128,
            256,
            kernel_size=5,
            padding=2,
        )

        self.bn3 = nn.BatchNorm1d(
            256
        )

        self.conv4 = nn.Conv1d(
            256,
            256,
            kernel_size=3,
            padding=1,
        )

        self.bn4 = nn.BatchNorm1d(
            256
        )

        self.projection = nn.Sequential(
            nn.Linear(
                256,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.10
            ),
            nn.Linear(
                256,
                embedding_dim,
            ),
        )

    def forward(
        self,
        mel,
    ):

        x = self.conv1(
            mel
        )

        x = self.bn1(x)

        x = F.relu(x)

        x = self.conv2(x)

        x = self.bn2(x)

        x = F.relu(x)

        x = self.conv3(x)

        x = self.bn3(x)

        x = F.relu(x)

        x = self.conv4(x)

        x = self.bn4(x)

        x = F.relu(x)

        x = x.mean(
            dim=-1
        )

        embedding = self.projection(
            x
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def find_checkpoint():

    if METRIC_CHECKPOINT.exists():

        print(
            "Using Lesson 65 "
            "metric-learning checkpoint:"
        )

        print(
            f"    {METRIC_CHECKPOINT}"
        )

        return METRIC_CHECKPOINT

    if CLASSIFICATION_CHECKPOINT.exists():

        print(
            "Lesson 65 checkpoint not found."
        )

        print(
            "Falling back to Lesson 64 checkpoint:"
        )

        print(
            f"    {CLASSIFICATION_CHECKPOINT}"
        )

        return CLASSIFICATION_CHECKPOINT

    raise FileNotFoundError(
        "\nNo speaker encoder checkpoint found.\n\n"
        "Expected one of:\n"
        f"    {METRIC_CHECKPOINT}\n"
        f"    {CLASSIFICATION_CHECKPOINT}\n\n"
        "Run Lesson 64 or Lesson 65 first.\n"
    )


def load_encoder():

    checkpoint_path = (
        find_checkpoint()
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    embedding_dim = checkpoint.get(
        "embedding_dim",
        128,
    )

    encoder = SpeakerEncoder(
        embedding_dim=embedding_dim,
    ).to(DEVICE)

    state_dict = checkpoint[
        "model_state_dict"
    ]

    # ----------------------------------------------------------------------
    # Lesson 64 checkpoint contains the encoder as:
    #
    #     encoder.conv1.weight
    #
    # Lesson 65 may save the encoder differently.
    #
    # We therefore support several common checkpoint layouts.
    # ----------------------------------------------------------------------

    encoder_state = {}

    if any(
        key.startswith(
            "encoder."
        )
        for key in state_dict.keys()
    ):

        for key, value in state_dict.items():

            if key.startswith(
                "encoder."
            ):

                new_key = key[
                    len("encoder.") :
                ]

                encoder_state[
                    new_key
                ] = value

    else:

        encoder_state = state_dict

    missing, unexpected = (
        encoder.load_state_dict(
            encoder_state,
            strict=False,
        )
    )

    print()
    print(
        f"Loaded checkpoint:"
    )

    print(
        f"    {checkpoint_path}"
    )

    if missing:

        print()
        print(
            "Missing encoder parameters:"
        )

        for key in missing:

            print(
                f"    {key}"
            )

    if unexpected:

        print()
        print(
            "Unexpected parameters:"
        )

        for key in unexpected:

            print(
                f"    {key}"
            )

    encoder.eval()

    return encoder


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def discover_audio():

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            "\nSpeaker dataset does not exist:\n"
            f"    {DATASET_DIR}\n\n"
            "Run Lesson 64 first. It automatically downloads\n"
            "and prepares the sample dataset.\n"
        )

    by_speaker = defaultdict(
        list
    )

    for speaker_dir in sorted(
        DATASET_DIR.iterdir()
    ):

        if not speaker_dir.is_dir():

            continue

        wav_files = sorted(
            speaker_dir.glob(
                "*.wav"
            )
        )

        if len(wav_files) >= 2:

            by_speaker[
                speaker_dir.name
            ].extend(
                wav_files
            )

    if len(by_speaker) < 2:

        raise RuntimeError(
            "At least two speakers with "
            "two utterances each are required."
        )

    return by_speaker


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

@torch.no_grad()
def audio_to_embedding(
    encoder,
    path,
):

    audio, sample_rate = (
        load_wav(path)
    )

    audio = resample_audio(
        audio,
        sample_rate,
        SAMPLE_RATE,
    )

    audio = normalize_audio(
        audio
    )

    audio = crop_or_pad(
        audio,
        CLIP_SAMPLES,
    )

    mel = waveform_to_log_mel(
        audio
    )

    mel = torch.from_numpy(
        mel
    ).float()

    mel = mel.unsqueeze(
        0
    )

    mel = mel.to(
        DEVICE
    )

    embedding = encoder(
        mel
    )

    return embedding.squeeze(
        0
    ).cpu()


# ============================================================================
# BUILD EMBEDDING DATABASE
# ============================================================================

def build_embedding_database(
    encoder,
    by_speaker,
):

    print()
    print("=" * 72)
    print("STEP 2 — EXTRACT SPEAKER EMBEDDINGS")
    print("=" * 72)

    database = defaultdict(
        list
    )

    total = sum(
        len(files)
        for files in by_speaker.values()
    )

    processed = 0

    for speaker, files in (
        by_speaker.items()
    ):

        print()
        print(
            f"{speaker}: "
            f"{len(files)} utterances"
        )

        for path in files:

            embedding = (
                audio_to_embedding(
                    encoder,
                    path,
                )
            )

            database[speaker].append(
                {
                    "path": path,
                    "embedding": embedding,
                }
            )

            processed += 1

            print(
                f"\rProcessed "
                f"{processed}/{total}",
                end="",
                flush=True,
            )

        print()

    return database


# ============================================================================
# COSINE SIMILARITY
# ============================================================================

def cosine_similarity(
    a,
    b,
):

    a = F.normalize(
        a,
        p=2,
        dim=-1,
    )

    b = F.normalize(
        b,
        p=2,
        dim=-1,
    )

    return float(
        torch.sum(
            a * b
        )
    )


# ============================================================================
# BUILD POSITIVE PAIRS
# ============================================================================

def build_positive_pairs(
    database,
    count,
):

    speakers = [
        speaker
        for speaker, items
        in database.items()
        if len(items) >= 2
    ]

    pairs = []

    attempts = 0

    max_attempts = (
        count * 20
    )

    while (
        len(pairs) < count
        and attempts < max_attempts
    ):

        attempts += 1

        speaker = random.choice(
            speakers
        )

        items = database[
            speaker
        ]

        first, second = random.sample(
            items,
            2,
        )

        similarity = (
            cosine_similarity(
                first["embedding"],
                second["embedding"],
            )
        )

        pairs.append(
            {
                "speaker_a": speaker,
                "speaker_b": speaker,
                "path_a": first["path"],
                "path_b": second["path"],
                "similarity": similarity,
                "same": True,
            }
        )

    return pairs


# ============================================================================
# BUILD NEGATIVE PAIRS
# ============================================================================

def build_negative_pairs(
    database,
    count,
):

    speakers = list(
        database.keys()
    )

    pairs = []

    attempts = 0

    max_attempts = (
        count * 20
    )

    while (
        len(pairs) < count
        and attempts < max_attempts
    ):

        attempts += 1

        speaker_a, speaker_b = (
            random.sample(
                speakers,
                2,
            )
        )

        first = random.choice(
            database[
                speaker_a
            ]
        )

        second = random.choice(
            database[
                speaker_b
            ]
        )

        similarity = (
            cosine_similarity(
                first["embedding"],
                second["embedding"],
            )
        )

        pairs.append(
            {
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "path_a": first["path"],
                "path_b": second["path"],
                "similarity": similarity,
                "same": False,
            }
        )

    return pairs


# ============================================================================
# THRESHOLD EVALUATION
# ============================================================================

def evaluate_threshold(
    similarities,
    labels,
    threshold,
):

    similarities = np.asarray(
        similarities
    )

    labels = np.asarray(
        labels
    ).astype(bool)

    predicted_same = (
        similarities >= threshold
    )

    positive = labels

    negative = ~labels

    false_accepts = np.sum(
        predicted_same
        & negative
    )

    false_rejects = np.sum(
        (~predicted_same)
        & positive
    )

    negative_count = max(
        1,
        np.sum(negative)
    )

    positive_count = max(
        1,
        np.sum(positive)
    )

    far = (
        false_accepts
        / negative_count
    )

    frr = (
        false_rejects
        / positive_count
    )

    return (
        far,
        frr,
    )


# ============================================================================
# FIND EER
# ============================================================================

def find_eer(
    similarities,
    labels,
):

    thresholds = np.linspace(
        -1.0,
        1.0,
        2001,
    )

    best = None

    for threshold in thresholds:

        far, frr = (
            evaluate_threshold(
                similarities,
                labels,
                threshold,
            )
        )

        difference = abs(
            far - frr
        )

        if (
            best is None
            or difference
            < best["difference"]
        ):

            best = {
                "threshold": threshold,
                "far": far,
                "frr": frr,
                "eer": (
                    far + frr
                ) / 2.0,
                "difference": difference,
            }

    return best


# ============================================================================
# FIND BEST ACCURACY THRESHOLD
# ============================================================================

def find_best_accuracy_threshold(
    similarities,
    labels,
):

    thresholds = np.linspace(
        -1.0,
        1.0,
        2001,
    )

    best = None

    for threshold in thresholds:

        predictions = (
            np.asarray(
                similarities
            )
            >= threshold
        )

        accuracy = np.mean(
            predictions
            == np.asarray(labels)
        )

        if (
            best is None
            or accuracy
            > best["accuracy"]
        ):

            far, frr = (
                evaluate_threshold(
                    similarities,
                    labels,
                    threshold,
                )
            )

            best = {
                "threshold": threshold,
                "accuracy": accuracy,
                "far": far,
                "frr": frr,
            }

    return best


# ============================================================================
# PLOT SIMILARITY DISTRIBUTION
# ============================================================================

def save_similarity_distribution(
    positive,
    negative,
    threshold,
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.hist(
        positive,
        bins=40,
        alpha=0.65,
        density=True,
        label="Same Speaker",
    )

    plt.hist(
        negative,
        bins=40,
        alpha=0.65,
        density=True,
        label="Different Speaker",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"Threshold = "
            f"{threshold:.3f}"
        ),
    )

    plt.xlabel(
        "Cosine Similarity"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Lesson 66 — Speaker Verification Similarity"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "verification_similarity.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        f"Saved:"
    )

    print(
        f"    {path}"
    )


# ============================================================================
# ROC-LIKE CURVE
# ============================================================================

def save_far_frr_curve(
    similarities,
    labels,
    eer_threshold,
):

    thresholds = np.linspace(
        -1.0,
        1.0,
        500,
    )

    far_values = []

    frr_values = []

    for threshold in thresholds:

        far, frr = (
            evaluate_threshold(
                similarities,
                labels,
                threshold,
            )
        )

        far_values.append(
            far
        )

        frr_values.append(
            frr
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        thresholds,
        far_values,
        label="FAR",
    )

    plt.plot(
        thresholds,
        frr_values,
        label="FRR",
    )

    plt.axvline(
        eer_threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"EER threshold = "
            f"{eer_threshold:.3f}"
        ),
    )

    plt.xlabel(
        "Verification Threshold"
    )

    plt.ylabel(
        "Rate"
    )

    plt.title(
        "Lesson 66 — FAR / FRR"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "far_frr_curve.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved:"
    )

    print(
        f"    {path}"
    )


# ============================================================================
# SAVE VERIFICATION RESULT
# ============================================================================

def save_results(
    eer_result,
    accuracy_result,
    positive,
    negative,
):

    path = (
        OUTPUT_DIR
        / "verification_results.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "LESSON 66 — SPEAKER VERIFICATION\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            f"Positive pairs: "
            f"{len(positive)}\n"
        )

        f.write(
            f"Negative pairs: "
            f"{len(negative)}\n\n"
        )

        f.write(
            "EER operating point\n"
        )

        f.write(
            "-------------------\n"
        )

        f.write(
            f"Threshold: "
            f"{eer_result['threshold']:.6f}\n"
        )

        f.write(
            f"FAR: "
            f"{eer_result['far']:.6f}\n"
        )

        f.write(
            f"FRR: "
            f"{eer_result['frr']:.6f}\n"
        )

        f.write(
            f"Estimated EER: "
            f"{eer_result['eer']:.6f}\n\n"
        )

        f.write(
            "Best accuracy operating point\n"
        )

        f.write(
            "------------------------------\n"
        )

        f.write(
            f"Threshold: "
            f"{accuracy_result['threshold']:.6f}\n"
        )

        f.write(
            f"Accuracy: "
            f"{accuracy_result['accuracy']:.6f}\n"
        )

        f.write(
            f"FAR: "
            f"{accuracy_result['far']:.6f}\n"
        )

        f.write(
            f"FRR: "
            f"{accuracy_result['frr']:.6f}\n"
        )

    print()
    print(
        f"Results saved:"
    )

    print(
        f"    {path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    seed_everything(
        RANDOM_SEED
    )

    ensure_directory(
        OUTPUT_DIR
    )

    print("=" * 72)
    print(
        "PHASE 2 — LESSON 66"
    )

    print(
        "SPEAKER VERIFICATION FROM SCRATCH"
    )

    print("=" * 72)

    print()
    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================================
    # STEP 1
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 1 — LOAD SPEAKER ENCODER"
    )
    print("=" * 72)

    encoder = load_encoder()

    parameter_count = sum(
        parameter.numel()
        for parameter in encoder.parameters()
    )

    print()
    print(
        f"Encoder parameters: "
        f"{parameter_count:,}"
    )

    # ========================================================================
    # STEP 2
    # ========================================================================

    by_speaker = discover_audio()

    print()
    print(
        f"Speakers found: "
        f"{len(by_speaker)}"
    )

    total_audio = sum(
        len(files)
        for files in by_speaker.values()
    )

    print(
        f"Audio files: "
        f"{total_audio}"
    )

    # ========================================================================
    # STEP 3
    # ========================================================================

    database = build_embedding_database(
        encoder,
        by_speaker,
    )

    # ========================================================================
    # STEP 4
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 3 — BUILD VERIFICATION PAIRS"
    )
    print("=" * 72)

    positive_pairs = (
        build_positive_pairs(
            database,
            NUM_POSITIVE_PAIRS,
        )
    )

    negative_pairs = (
        build_negative_pairs(
            database,
            NUM_NEGATIVE_PAIRS,
        )
    )

    print()
    print(
        f"Positive pairs: "
        f"{len(positive_pairs)}"
    )

    print(
        f"Negative pairs: "
        f"{len(negative_pairs)}"
    )

    # ========================================================================
    # STEP 5
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 4 — CALCULATE SIMILARITIES"
    )
    print("=" * 72)

    positive_scores = [
        pair["similarity"]
        for pair in positive_pairs
    ]

    negative_scores = [
        pair["similarity"]
        for pair in negative_pairs
    ]

    similarities = (
        positive_scores
        + negative_scores
    )

    labels = (
        [True] * len(
            positive_scores
        )
        + [False] * len(
            negative_scores
        )
    )

    print()
    print(
        f"Mean same-speaker similarity: "
        f"{np.mean(positive_scores):.4f}"
    )

    print(
        f"Std same-speaker similarity: "
        f"{np.std(positive_scores):.4f}"
    )

    print(
        f"Mean different-speaker similarity: "
        f"{np.mean(negative_scores):.4f}"
    )

    print(
        f"Std different-speaker similarity: "
        f"{np.std(negative_scores):.4f}"
    )

    # ========================================================================
    # STEP 6
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 5 — FIND EQUAL ERROR RATE"
    )
    print("=" * 72)

    eer_result = find_eer(
        similarities,
        labels,
    )

    print()
    print(
        f"EER threshold: "
        f"{eer_result['threshold']:.4f}"
    )

    print(
        f"FAR at EER: "
        f"{eer_result['far']:.4f}"
    )

    print(
        f"FRR at EER: "
        f"{eer_result['frr']:.4f}"
    )

    print(
        f"Estimated EER: "
        f"{eer_result['eer']:.4f}"
    )

    # ========================================================================
    # STEP 7
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 6 — FIND BEST ACCURACY THRESHOLD"
    )
    print("=" * 72)

    accuracy_result = (
        find_best_accuracy_threshold(
            similarities,
            labels,
        )
    )

    print()
    print(
        f"Best threshold: "
        f"{accuracy_result['threshold']:.4f}"
    )

    print(
        f"Verification accuracy: "
        f"{accuracy_result['accuracy']:.4f}"
    )

    print(
        f"FAR: "
        f"{accuracy_result['far']:.4f}"
    )

    print(
        f"FRR: "
        f"{accuracy_result['frr']:.4f}"
    )

    # ========================================================================
    # STEP 8
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 7 — SAVE VISUALIZATIONS"
    )
    print("=" * 72)

    save_similarity_distribution(
        positive_scores,
        negative_scores,
        accuracy_result[
            "threshold"
        ],
    )

    save_far_frr_curve(
        similarities,
        labels,
        eer_result[
            "threshold"
        ],
    )

    # ========================================================================
    # STEP 9
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 8 — SAVE RESULTS"
    )
    print("=" * 72)

    save_results(
        eer_result,
        accuracy_result,
        positive_scores,
        negative_scores,
    )

    # ========================================================================
    # STEP 10
    # ========================================================================

    threshold_checkpoint = (
        OUTPUT_DIR
        / "speaker_verification_threshold.pt"
    )

    torch.save(
        {
            "threshold": (
                accuracy_result[
                    "threshold"
                ]
            ),
            "eer_threshold": (
                eer_result[
                    "threshold"
                ]
            ),
            "eer": (
                eer_result[
                    "eer"
                ]
            ),
            "far": (
                accuracy_result[
                    "far"
                ]
            ),
            "frr": (
                accuracy_result[
                    "frr"
                ]
            ),
        },
        threshold_checkpoint,
    )

    print()
    print(
        f"Threshold checkpoint saved:"
    )

    print(
        f"    {threshold_checkpoint}"
    )

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "LESSON 66 COMPLETE"
    )
    print("=" * 72)

    print()
    print(
        "Pipeline:"
    )

    print(
        "    AUDIO A"
    )

    print(
        "       ↓"
    )

    print(
        "    SPEAKER ENCODER"
    )

    print(
        "       ↓"
    )

    print(
        "    EMBEDDING A"
    )

    print(
        "       ↓"
    )

    print(
        "    COSINE SIMILARITY"
    )

    print(
        "       ↑"
    )

    print(
        "    EMBEDDING B"
    )

    print(
        "       ↑"
    )

    print(
        "    SPEAKER ENCODER"
    )

    print(
        "       ↑"
    )

    print(
        "    AUDIO B"
    )

    print()
    print(
        "Verification threshold:"
    )

    print(
        f"    {accuracy_result['threshold']:.4f}"
    )

    print()
    print(
        "Estimated EER:"
    )

    print(
        f"    {eer_result['eer']:.4f}"
    )

    print()
    print(
        "Next conceptual step:"
    )

    print(
        "    REFERENCE AUDIO"
        " → SPEAKER EMBEDDING"
        " → SPEAKER-CONDITIONED TTS"
    )

    print()
    print("=" * 72)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()