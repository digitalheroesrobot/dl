"""
========================================================================
PHASE 2 — LESSON 67
REAL SPEAKER-CONDITIONED ACOUSTIC MODEL
========================================================================

PURPOSE
-------
Move from speaker verification to REAL SPEAKER-CONDITIONED TTS.

Previous lessons:

    Lesson 63
        Speaker-conditioned TTS
        Synthetic speaker embeddings

    Lesson 64
        Speaker encoder from real audio
        Classification objective

    Lesson 65
        Speaker metric learning

    Lesson 66
        Speaker verification

NOW:

    Lesson 67
        Use a REAL reference recording to obtain a speaker embedding
        and condition an acoustic TTS model.

CORE IDEA
---------

Reference audio
       |
       v
+------------------+
| Speaker Encoder  |
+------------------+
       |
       v
Speaker Embedding
       |
       +--------------------+
       |                    |
       v                    v
     Text             Speaker Identity
       |                    |
       +---------+----------+
                 |
                 v
       Speaker-Conditioned
        Acoustic Model
                 |
                 v
           Mel Spectrogram


NO PRETRAINED MODELS
--------------------
This lesson does NOT use:

    - Whisper
    - Wav2Vec2
    - SpeechBrain
    - ECAPA
    - x-vectors
    - pretrained TTS
    - pretrained phonemizers
    - pretrained acoustic models

Everything used here is trained or implemented in this course.

WHAT YOU LEARN
--------------

STEP 1
    Load real speaker audio.

STEP 2
    Load the speaker encoder from Lesson 65/64.

STEP 3
    Extract a speaker embedding from reference audio.

STEP 4
    Build text/audio training examples.

STEP 5
    Train an acoustic model conditioned on speaker identity.

STEP 6
    Keep the speaker encoder frozen.

STEP 7
    Generate a Mel spectrogram using:

        NEW TEXT
        +
        REFERENCE SPEAKER AUDIO

STEP 8
    Compare speaker embeddings.

IMPORTANT
---------
This lesson demonstrates REAL SPEAKER CONDITIONING.

It is not yet a production-quality zero-shot voice cloning system.

========================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import re
import random
import tarfile
import hashlib
import subprocess
import math                      # FIX: required by PositionalEncoding
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "speaker_dataset"

LESSON64_DATA_DIR = ROOT / "lesson64_data"

OUTPUT_DIR = ROOT / "lesson67_outputs"

METRIC_CHECKPOINT = (
    ROOT
    / "lesson65_outputs"
    / "speaker_metric_encoder_best.pt"
)

CLASSIFICATION_CHECKPOINT = (
    ROOT
    / "lesson64_outputs"
    / "speaker_encoder_best.pt"
)

LIBRISPEECH_URL = (
    "https://www.openslr.org/resources/12/dev-clean.tar.gz"
)

LIBRISPEECH_MD5 = (
    "42e2234ba48799c1f50f24a7926300a1"
)

ARCHIVE_PATH = (
    LESSON64_DATA_DIR
    / "dev-clean.tar.gz"
)

LIBRISPEECH_ROOT = (
    LESSON64_DATA_DIR
    / "LibriSpeech"
    / "dev-clean"
)

MAX_SPEAKERS = 12

MAX_UTTERANCES_PER_SPEAKER = 20

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
# Text
# --------------------------------------------------------------------------

MAX_TEXT_LENGTH = 180

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

TEXT_EMBED_DIM = 192

TEXT_LAYERS = 3

TEXT_HEADS = 4

SPEAKER_DIM = 128

MODEL_DIM = 192

DROPOUT = 0.10

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

BATCH_SIZE = 8

EPOCHS = 30

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-4

MAX_GRAD_NORM = 1.0

VAL_RATIO = 0.15

NUM_WORKERS = 2

SEED = 42

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
# MD5
# ============================================================================

def md5sum(path):

    digest = hashlib.md5()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================================
# DATASET DOWNLOAD
# ============================================================================

def download_file(
    url,
    destination,
):

    ensure_directory(
        destination.parent
    )

    print()
    print(
        "Downloading sample dataset..."
    )

    print(
        f"URL: {url}"
    )

    try:

        import urllib.request

        urllib.request.urlretrieve(
            url,
            destination,
        )

    except Exception as exc:

        raise RuntimeError(
            "\nDataset download failed.\n"
            "Please check your network connection.\n"
            f"URL: {url}\n"
            f"Error: {exc}\n"
        )

    print(
        "Download complete."
    )


# ============================================================================
# LIBRISPEECH PREPARATION
# ============================================================================

def prepare_librispeech():

    if LIBRISPEECH_ROOT.exists():

        flac_files = list(
            LIBRISPEECH_ROOT.rglob(
                "*.flac"
            )
        )

        if len(flac_files) > 0:

            print(
                "LibriSpeech dataset already exists."
            )

            return

    ensure_directory(
        LESSON64_DATA_DIR
    )

    if not ARCHIVE_PATH.exists():

        download_file(
            LIBRISPEECH_URL,
            ARCHIVE_PATH,
        )

    print()
    print(
        "Checking dataset MD5..."
    )

    actual_md5 = md5sum(
        ARCHIVE_PATH
    )

    if actual_md5.lower() != (
        LIBRISPEECH_MD5.lower()
    ):

        raise RuntimeError(
            "\nDataset MD5 mismatch.\n"
            f"Expected: {LIBRISPEECH_MD5}\n"
            f"Actual:   {actual_md5}\n"
        )

    print(
        "MD5 verified."
    )

    print()
    print(
        "Extracting LibriSpeech..."
    )

    with tarfile.open(
        ARCHIVE_PATH,
        "r:gz",
    ) as tar:

        tar.extractall(
            LESSON64_DATA_DIR
        )

    print(
        "Extraction complete."
    )


# ============================================================================
# FFMPEG CHECK
# ============================================================================

def check_ffmpeg():

    try:

        result = subprocess.run(
            [
                "ffmpeg",
                "-version",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode == 0:

            return True

    except FileNotFoundError:

        pass

    return False


# ============================================================================
# AUDIO DATASET CREATION
# ============================================================================

def prepare_speaker_dataset():

    if DATASET_DIR.exists():

        existing = list(
            DATASET_DIR.glob(
                "speaker_*/*.wav"
            )
        )

        if len(existing) > 0:

            print()
            print(
                f"Existing speaker dataset found:"
            )

            print(
                f"    {len(existing)} WAV files"
            )

            return

    print()
    print("=" * 72)
    print(
        "PREPARING REAL SPEAKER DATASET"
    )
    print("=" * 72)

    prepare_librispeech()

    if not check_ffmpeg():

        raise RuntimeError(
            "\nffmpeg is required to convert "
            "LibriSpeech FLAC files to WAV.\n\n"
            "Install it with:\n\n"
            "    sudo apt update\n"
            "    sudo apt install ffmpeg\n"
        )

    speakers = []

    for speaker_dir in sorted(
        LIBRISPEECH_ROOT.iterdir()
    ):

        if not speaker_dir.is_dir():

            continue

        flacs = sorted(
            speaker_dir.rglob(
                "*.flac"
            )
        )

        if len(flacs) < 6:

            continue

        speakers.append(
            (
                speaker_dir.name,
                flacs,
            )
        )

        if len(speakers) >= MAX_SPEAKERS:

            break

    if len(speakers) < 2:

        raise RuntimeError(
            "Could not find enough speakers."
        )

    ensure_directory(
        DATASET_DIR
    )

    print()
    print(
        f"Selected speakers: "
        f"{len(speakers)}"
    )

    for speaker_index, (
        original_id,
        flacs,
    ) in enumerate(
        speakers
    ):

        destination = (
            DATASET_DIR
            / f"speaker_{speaker_index:03d}"
        )

        ensure_directory(
            destination
        )

        selected = flacs[
            :MAX_UTTERANCES_PER_SPEAKER
        ]

        print()
        print(
            f"Speaker {speaker_index:03d}"
        )

        print(
            f"    Original ID: {original_id}"
        )

        print(
            f"    Files: {len(selected)}"
        )

        for file_index, flac in enumerate(
            selected
        ):

            output_wav = (
                destination
                / f"{file_index:04d}.wav"
            )

            if output_wav.exists():

                continue

            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(flac),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                str(output_wav),
            ]

            result = subprocess.run(
                command
            )

            if result.returncode != 0:

                raise RuntimeError(
                    f"ffmpeg failed for {flac}"
                )

    print()
    print(
        "Speaker dataset preparation complete."
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

        raw = wav.readframes(
            frames
        )

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

    return np.interp(
        new_positions,
        old_positions,
        audio,
    ).astype(
        np.float32
    )


# ============================================================================
# FIXED AUDIO LENGTH
# ============================================================================

def crop_or_pad(
    audio,
    target_length,
    random_crop=False,
):

    if len(audio) == target_length:

        return audio

    if len(audio) > target_length:

        max_start = (
            len(audio)
            - target_length
        )

        if random_crop:

            start = random.randint(
                0,
                max_start,
            )

        else:

            start = max_start // 2

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
# MEL FILTERBANK
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

    points = np.linspace(
        mel_min,
        mel_max,
        N_MELS + 2,
    )

    hz_points = mel_to_hz(
        points
    )

    bins = np.floor(
        (N_FFT + 1)
        * hz_points
        / SAMPLE_RATE
    ).astype(int)

    bank = np.zeros(
        (
            N_MELS,
            n_freqs,
        ),
        dtype=np.float32,
    )

    for m in range(N_MELS):

        left = max(
            0,
            min(
                bins[m],
                n_freqs - 1,
            ),
        )

        center = max(
            0,
            min(
                bins[m + 1],
                n_freqs - 1,
            ),
        )

        right = max(
            0,
            min(
                bins[m + 2],
                n_freqs - 1,
            ),
        )

        if center > left:

            bank[
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

            bank[
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

    return bank


MEL_FILTERBANK = (
    create_mel_filterbank()
)


# ============================================================================
# WAVEFORM → MEL
# ============================================================================

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
            5,
            padding=2,
        )

        self.bn1 = nn.BatchNorm1d(
            128
        )

        self.conv2 = nn.Conv1d(
            128,
            128,
            5,
            padding=2,
        )

        self.bn2 = nn.BatchNorm1d(
            128
        )

        self.conv3 = nn.Conv1d(
            128,
            256,
            5,
            padding=2,
        )

        self.bn3 = nn.BatchNorm1d(
            256
        )

        self.conv4 = nn.Conv1d(
            256,
            256,
            3,
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

        x = F.relu(
            self.bn1(
                self.conv1(mel)
            )
        )

        x = F.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        x = F.relu(
            self.bn4(
                self.conv4(x)
            )
        )

        x = x.mean(
            dim=-1
        )

        x = self.projection(
            x
        )

        return F.normalize(
            x,
            p=2,
            dim=-1,
        )


# ============================================================================
# LOAD SPEAKER ENCODER
# ============================================================================

def load_speaker_encoder():

    if METRIC_CHECKPOINT.exists():

        checkpoint_path = (
            METRIC_CHECKPOINT
        )

        print(
            "Using Lesson 65 speaker encoder."
        )

    elif CLASSIFICATION_CHECKPOINT.exists():

        checkpoint_path = (
            CLASSIFICATION_CHECKPOINT
        )

        print(
            "Lesson 65 checkpoint not found."
        )

        print(
            "Using Lesson 64 speaker encoder."
        )

    else:

        raise FileNotFoundError(
            "\nNo speaker encoder checkpoint found.\n"
            "Run Lesson 64 or Lesson 65 first.\n"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    embedding_dim = checkpoint.get(
        "embedding_dim",
        SPEAKER_DIM,
    )

    encoder = SpeakerEncoder(
        embedding_dim=embedding_dim
    ).to(
        DEVICE
    )

    state = checkpoint[
        "model_state_dict"
    ]

    encoder_state = {}

    if any(
        key.startswith(
            "encoder."
        )
        for key in state.keys()
    ):

        for key, value in state.items():

            if key.startswith(
                "encoder."
            ):

                encoder_state[
                    key[len("encoder."):]
                ] = value

    else:

        encoder_state = state

    encoder.load_state_dict(
        encoder_state,
        strict=False,
    )

    encoder.eval()

    for parameter in encoder.parameters():

        parameter.requires_grad = False

    print()
    print(
        f"Speaker encoder:"
    )

    print(
        f"    {checkpoint_path}"
    )

    return encoder


# ============================================================================
# TRANSCRIPT DISCOVERY
# ============================================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z' ]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def build_transcript_index():

    transcript_map = {}

    if not LIBRISPEECH_ROOT.exists():

        return transcript_map

    for trans_file in (
        LIBRISPEECH_ROOT.rglob(
            "*.trans.txt"
        )
    ):

        try:

            with open(
                trans_file,
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:

                        continue

                    parts = line.split(
                        " ",
                        1,
                    )

                    if len(parts) != 2:

                        continue

                    utterance_id = (
                        parts[0]
                    )

                    text = normalize_text(
                        parts[1]
                    )

                    if text:

                        transcript_map[
                            utterance_id
                        ] = text

        except Exception:

            continue

    return transcript_map


# ============================================================================
# TOKENIZER
# ============================================================================

PAD_ID = 0

BOS_ID = 1

EOS_ID = 2

UNK_ID = 3


def build_vocabulary():

    symbols = list(
        "abcdefghijklmnopqrstuvwxyz"
        " '"
    )

    vocabulary = {
        "<pad>": PAD_ID,
        "<bos>": BOS_ID,
        "<eos>": EOS_ID,
        "<unk>": UNK_ID,
    }

    next_id = 4

    for symbol in symbols:

        if symbol not in vocabulary:

            vocabulary[
                symbol
            ] = next_id

            next_id += 1

    return vocabulary


VOCAB = build_vocabulary()


def encode_text(
    text,
):

    text = normalize_text(
        text
    )

    ids = [
        BOS_ID
    ]

    for char in text:

        ids.append(
            VOCAB.get(
                char,
                UNK_ID,
            )
        )

    ids.append(
        EOS_ID
    )

    return ids


# ============================================================================
# DATA RECORD
# ============================================================================

def build_records():

    transcript_map = (
        build_transcript_index()
    )

    records = []

    speaker_dirs = sorted(
        DATASET_DIR.glob(
            "speaker_*"
        )
    )

    for speaker_index, speaker_dir in enumerate(
        speaker_dirs
    ):

        wav_files = sorted(
            speaker_dir.glob(
                "*.wav"
            )
        )

        for wav_path in wav_files:

            text = None

            if LIBRISPEECH_ROOT.exists():

                original_speaker_dirs = sorted(
                    [
                        p
                        for p in LIBRISPEECH_ROOT.iterdir()
                        if p.is_dir()
                    ]
                )

                if (
                    speaker_index
                    < len(original_speaker_dirs)
                ):

                    original_dir = (
                        original_speaker_dirs[
                            speaker_index
                        ]
                    )

                    if original_dir.is_dir():

                        flacs = sorted(
                            original_dir.rglob(
                                "*.flac"
                            )
                        )

                        file_index = int(
                            wav_path.stem
                        )

                        if file_index < len(flacs):

                            utterance_id = (
                                flacs[
                                    file_index
                                ].stem
                            )

                            text = (
                                transcript_map.get(
                                    utterance_id
                                )
                            )

            if text is None:

                text = (
                    "this is a speaker "
                    f"conditioning example "
                    f"{speaker_index}"
                )

            token_ids = encode_text(
                text
            )

            if len(token_ids) > MAX_TEXT_LENGTH:

                token_ids = (
                    token_ids[
                        :MAX_TEXT_LENGTH - 1
                    ]
                    + [EOS_ID]
                )

            records.append(
                {
                    "path": wav_path,
                    "speaker": speaker_index,
                    "text": text,
                    "tokens": token_ids,
                }
            )

    return records


# ============================================================================
# SPEAKER EMBEDDING
# ============================================================================

@torch.no_grad()
def extract_reference_embedding(
    encoder,
    wav_path,
):

    audio, sample_rate = (
        load_wav(wav_path)
    )

    audio = resample_audio(
        audio,
        sample_rate,
        SAMPLE_RATE,
    )

    audio = crop_or_pad(
        audio,
        CLIP_SAMPLES,
        random_crop=False,
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = audio / peak

    mel = waveform_to_log_mel(
        audio
    )

    mel = torch.from_numpy(
        mel
    ).float()

    mel = mel.unsqueeze(
        0
    ).to(
        DEVICE
    )

    embedding = encoder(
        mel
    )

    return embedding.squeeze(
        0
    ).cpu()


# ============================================================================
# DATASET
# ============================================================================

class SpeakerConditionedDataset(
    Dataset
):

    def __init__(
        self,
        records,
        speaker_embeddings,
    ):

        self.records = records

        self.speaker_embeddings = (
            speaker_embeddings
        )

    def __len__(self):

        return len(
            self.records
        )

    def __getitem__(
        self,
        index,
    ):

        record = self.records[
            index
        ]

        audio, sample_rate = (
            load_wav(
                record["path"]
            )
        )

        audio = resample_audio(
            audio,
            sample_rate,
            SAMPLE_RATE,
        )

        audio = crop_or_pad(
            audio,
            CLIP_SAMPLES,
            random_crop=True,
        )

        peak = np.max(
            np.abs(audio)
        )

        if peak > 1e-8:

            audio = (
                audio / peak
            )

        mel = waveform_to_log_mel(
            audio
        )

        mel = torch.from_numpy(
            mel
        ).float()

        tokens = torch.tensor(
            record["tokens"],
            dtype=torch.long,
        )

        speaker_embedding = (
            self.speaker_embeddings[
                record["speaker"]
            ]
        )

        return {
            "tokens": tokens,
            "mel": mel,
            "speaker_embedding":
                speaker_embedding,
            "text": record["text"],
        }


# ============================================================================
# COLLATE
# ============================================================================

def collate_batch(
    batch,
):

    max_text = max(
        item["tokens"].shape[0]
        for item in batch
    )

    max_mel = max(
        item["mel"].shape[1]
        for item in batch
    )

    batch_size = len(
        batch
    )

    tokens = torch.full(
        (
            batch_size,
            max_text,
        ),
        PAD_ID,
        dtype=torch.long,
    )

    token_mask = torch.zeros(
        (
            batch_size,
            max_text,
        ),
        dtype=torch.bool,
    )

    mels = torch.zeros(
        (
            batch_size,
            N_MELS,
            max_mel,
        ),
        dtype=torch.float32,
    )

    mel_mask = torch.zeros(
        (
            batch_size,
            max_mel,
        ),
        dtype=torch.bool,
    )

    speaker_embeddings = []

    texts = []

    for i, item in enumerate(
        batch
    ):

        token_length = (
            item["tokens"].shape[0]
        )

        mel_length = (
            item["mel"].shape[1]
        )

        tokens[
            i,
            :token_length
        ] = item["tokens"]

        token_mask[
            i,
            :token_length
        ] = True

        mels[
            i,
            :,
            :mel_length
        ] = item["mel"]

        mel_mask[
            i,
            :mel_length
        ] = True

        speaker_embeddings.append(
            item[
                "speaker_embedding"
            ]
        )

        texts.append(
            item["text"]
        )

    speaker_embeddings = torch.stack(
        speaker_embeddings
    )

    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "mel": mels,
        "mel_mask": mel_mask,
        "speaker_embedding":
            speaker_embeddings,
        "texts": texts,
    }


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(
    nn.Module
):

    def __init__(
        self,
        dim,
        max_length=512,
    ):

        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(
            1
        )

        div_term = torch.exp(
            torch.arange(
                0,
                dim,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(
                    10000.0
                )
                / dim
            )
        )

        encoding = torch.zeros(
            max_length,
            dim,
        )

        encoding[
            :,
            0::2
        ] = torch.sin(
            position
            * div_term
        )

        encoding[
            :,
            1::2
        ] = torch.cos(
            position
            * div_term
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(
                0
            ),
        )

    def forward(
        self,
        x,
    ):

        length = x.shape[1]

        return (
            x
            + self.encoding[
                :,
                :length,
                :
            ]
        )


# ============================================================================
# SPEAKER-CONDITIONED ACOUSTIC MODEL
# ============================================================================

class RealSpeakerConditionedAcousticModel(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        speaker_dim=128,
        model_dim=192,
    ):

        super().__init__()

        self.model_dim = (
            model_dim
        )

        self.text_embedding = (
            nn.Embedding(
                vocab_size,
                model_dim,
                padding_idx=PAD_ID,
            )
        )

        self.position = (
            PositionalEncoding(
                model_dim,
                max_length=MAX_TEXT_LENGTH,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=TEXT_HEADS,
                dim_feedforward=model_dim * 4,
                dropout=DROPOUT,
                batch_first=True,
                norm_first=True,
            )
        )

        self.text_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=TEXT_LAYERS,
            )
        )

        self.speaker_projection = (
            nn.Sequential(
                nn.Linear(
                    speaker_dim,
                    model_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    model_dim,
                    model_dim,
                ),
            )
        )

        self.acoustic_decoder = (
            nn.Sequential(
                nn.Linear(
                    model_dim,
                    model_dim,
                ),
                nn.ReLU(),
                nn.LayerNorm(
                    model_dim
                ),
                nn.Linear(
                    model_dim,
                    model_dim,
                ),
                nn.ReLU(),
            )
        )

        self.mel_projection = (
            nn.Linear(
                model_dim,
                N_MELS,
            )
        )

        self.upsampler = nn.Sequential(
            nn.Conv1d(
                model_dim,
                model_dim,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                model_dim,
                model_dim,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
        )

    def encode_text(
        self,
        tokens,
        token_mask,
        speaker_embedding,
    ):

        x = self.text_embedding(
            tokens
        )

        x = self.position(
            x
        )

        speaker = (
            self.speaker_projection(
                speaker_embedding
            )
        )

        speaker = speaker.unsqueeze(
            1
        )

        # Speaker identity is injected into every text representation.
        x = x + speaker

        encoded = self.text_encoder(
            x,
            src_key_padding_mask=~token_mask,
        )

        return encoded

    def forward(
        self,
        tokens,
        token_mask,
        speaker_embedding,
        target_frames=None,
    ):

        encoded = self.encode_text(
            tokens,
            token_mask,
            speaker_embedding,
        )

        if target_frames is None:

            target_frames = (
                encoded.shape[1] * 6
            )

        x = encoded.transpose(
            1,
            2
        )

        x = F.interpolate(
            x,
            size=target_frames,
            mode="linear",
            align_corners=False,
        )

        x = self.upsampler(
            x
        )

        x = x.transpose(
            1,
            2
        )

        x = self.acoustic_decoder(
            x
        )

        mel = self.mel_projection(
            x
        )

        mel = mel.transpose(
            1,
            2
        )

        return mel


# ============================================================================
# MASKED L1 LOSS
# ============================================================================

def masked_l1_loss(
    prediction,
    target,
    mask,
):

    mask = mask.unsqueeze(
        1
    ).float()

    difference = torch.abs(
        prediction - target
    )

    difference = (
        difference * mask
    )

    denominator = (
        mask.sum()
        * prediction.shape[1]
    )

    denominator = torch.clamp(
        denominator,
        min=1.0,
    )

    return (
        difference.sum()
        / denominator
    )


# ============================================================================
# SPLIT
# ============================================================================

def split_records(
    records,
):

    by_speaker = defaultdict(
        list
    )

    for record in records:

        by_speaker[
            record["speaker"]
        ].append(
            record
        )

    train = []

    val = []

    for speaker, items in (
        by_speaker.items()
    ):

        items = list(items)

        random.shuffle(
            items
        )

        val_count = max(
            1,
            int(
                len(items)
                * VAL_RATIO
            ),
        )

        val.extend(
            items[
                :val_count
            ]
        )

        train.extend(
            items[
                val_count:
            ]
        )

    return train, val


# ============================================================================
# REFERENCE AUDIO
# ============================================================================

def select_reference_audio(
    records,
):

    by_speaker = defaultdict(
        list
    )

    for record in records:

        by_speaker[
            record["speaker"]
        ].append(
            record
        )

    speaker = sorted(
        by_speaker.keys()
    )[0]

    return by_speaker[
        speaker
    ][0]


# ============================================================================
# TRAINING PLOT
# ============================================================================

def save_training_plot(
    train_losses,
    val_losses,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        train_losses,
        label="Train",
    )

    plt.plot(
        val_losses,
        label="Validation",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Mel L1 Loss"
    )

    plt.title(
        "Lesson 67 — Speaker-Conditioned Acoustic Training"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "training_curve.png"
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
# MEL PLOT
# ============================================================================

def save_mel_plot(
    mel,
    title,
    filename,
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        mel,
        aspect="auto",
        origin="lower",
    )

    plt.xlabel(
        "Mel Frame"
    )

    plt.ylabel(
        "Mel Bin"
    )

    plt.title(
        title
    )

    plt.colorbar(
        label="Normalized log Mel"
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / filename
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
# TRAIN
# ============================================================================

def train_epoch(
    model,
    loader,
    optimizer,
):

    model.train()

    total_loss = 0.0

    batches = 0

    for batch in loader:

        tokens = batch[
            "tokens"
        ].to(
            DEVICE
        )

        token_mask = batch[
            "token_mask"
        ].to(
            DEVICE
        )

        target_mel = batch[
            "mel"
        ].to(
            DEVICE
        )

        mel_mask = batch[
            "mel_mask"
        ].to(
            DEVICE
        )

        speaker_embedding = batch[
            "speaker_embedding"
        ].to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            tokens,
            token_mask,
            speaker_embedding,
            target_frames=target_mel.shape[-1],
        )

        loss = masked_l1_loss(
            prediction,
            target_mel,
            mel_mask,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()

        total_loss += float(
            loss.item()
        )

        batches += 1

    return (
        total_loss
        / max(
            1,
            batches,
        )
    )


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def validate(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0

    batches = 0

    for batch in loader:

        tokens = batch[
            "tokens"
        ].to(
            DEVICE
        )

        token_mask = batch[
            "token_mask"
        ].to(
            DEVICE
        )

        target_mel = batch[
            "mel"
        ].to(
            DEVICE
        )

        mel_mask = batch[
            "mel_mask"
        ].to(
            DEVICE
        )

        speaker_embedding = batch[
            "speaker_embedding"
        ].to(
            DEVICE
        )

        prediction = model(
            tokens,
            token_mask,
            speaker_embedding,
            target_frames=target_mel.shape[-1],
        )

        loss = masked_l1_loss(
            prediction,
            target_mel,
            mel_mask,
        )

        total_loss += float(
            loss.item()
        )

        batches += 1

    return (
        total_loss
        / max(
            1,
            batches,
        )
    )


# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def synthesize_mel(
    model,
    speaker_encoder,
    reference_wav,
    text,
):

    speaker_embedding = (
        extract_reference_embedding(
            speaker_encoder,
            reference_wav,
        )
    )

    tokens = torch.tensor(
        encode_text(text),
        dtype=torch.long,
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    token_mask = (
        tokens != PAD_ID
    )

    speaker_embedding = (
        speaker_embedding
        .unsqueeze(
            0
        )
        .to(
            DEVICE
        )
    )

    target_frames = max(
        64,
        min(
            400,
            len(text) * 5,
        ),
    )

    prediction = model(
        tokens,
        token_mask,
        speaker_embedding,
        target_frames=target_frames,
    )

    return (
        prediction.squeeze(
            0
        ).cpu().numpy(),
        speaker_embedding.squeeze(
            0
        ).cpu().numpy(),
    )


# ============================================================================
# CHECKPOINT
# ============================================================================

def save_checkpoint(
    model,
    path,
    epoch,
    loss,
):

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "epoch":
                epoch,
            "loss":
                loss,
            "vocab":
                VOCAB,
            "sample_rate":
                SAMPLE_RATE,
            "n_mels":
                N_MELS,
            "speaker_dim":
                SPEAKER_DIM,
        },
        path,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    seed_everything(
        SEED
    )

    ensure_directory(
        OUTPUT_DIR
    )

    print("=" * 72)

    print(
        "PHASE 2 — LESSON 67"
    )

    print(
        "REAL SPEAKER-CONDITIONED ACOUSTIC MODEL"
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
        "STEP 1 — PREPARE REAL SPEAKER DATA"
    )

    print("=" * 72)

    prepare_speaker_dataset()

    # ========================================================================
    # STEP 2
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 2 — LOAD SPEAKER ENCODER"
    )

    print("=" * 72)

    speaker_encoder = (
        load_speaker_encoder()
    )

    # ========================================================================
    # STEP 3
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 3 — BUILD TEXT/AUDIO RECORDS"
    )

    print("=" * 72)

    records = build_records()

    if len(records) < 10:

        raise RuntimeError(
            "Too few training records."
        )

    print()
    print(
        f"Total records: "
        f"{len(records)}"
    )

    print(
        f"Vocabulary size: "
        f"{len(VOCAB)}"
    )

    # ========================================================================
    # STEP 4
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 4 — EXTRACT SPEAKER EMBEDDINGS"
    )

    print("=" * 72)

    speaker_reference = {}

    by_speaker = defaultdict(
        list
    )

    for record in records:

        by_speaker[
            record["speaker"]
        ].append(
            record
        )

    for speaker in sorted(
        by_speaker.keys()
    ):

        reference = (
            by_speaker[
                speaker
            ][0]
        )

        embedding = (
            extract_reference_embedding(
                speaker_encoder,
                reference["path"],
            )
        )

        speaker_reference[
            speaker
        ] = embedding

        print(
            f"Speaker {speaker:03d}: "
            f"{reference['path'].name}"
        )

    # ========================================================================
    # STEP 5
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 5 — SPLIT TRAIN / VALIDATION"
    )

    print("=" * 72)

    train_records, val_records = (
        split_records(
            records
        )
    )

    print()
    print(
        f"Training records: "
        f"{len(train_records)}"
    )

    print(
        f"Validation records: "
        f"{len(val_records)}"
    )

    # ========================================================================
    # STEP 6
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 6 — CREATE DATA LOADERS"
    )

    print("=" * 72)

    train_dataset = (
        SpeakerConditionedDataset(
            train_records,
            speaker_reference,
        )
    )

    val_dataset = (
        SpeakerConditionedDataset(
            val_records,
            speaker_reference,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    # ========================================================================
    # STEP 7
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 7 — BUILD SPEAKER-CONDITIONED ACOUSTIC MODEL"
    )

    print("=" * 72)

    model = (
        RealSpeakerConditionedAcousticModel(
            vocab_size=len(VOCAB),
            speaker_dim=SPEAKER_DIM,
            model_dim=MODEL_DIM,
        )
        .to(
            DEVICE
        )
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        f"Acoustic model parameters: "
        f"{parameter_count:,}"
    )

    # ========================================================================
    # STEP 8
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 8 — TRAIN"
    )

    print("=" * 72)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
    )

    train_losses = []

    val_losses = []

    best_val_loss = float(
        "inf"
    )

    best_path = (
        OUTPUT_DIR
        / "real_speaker_conditioned_acoustic_best.pt"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_loss = validate(
            model,
            val_loader,
        )

        scheduler.step(
            val_loss
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"LR {current_lr:.2e}"
        )

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            save_checkpoint(
                model,
                best_path,
                epoch,
                val_loss,
            )

    # ========================================================================
    # STEP 9
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 9 — SAVE TRAINING RESULTS"
    )

    print("=" * 72)

    save_training_plot(
        train_losses,
        val_losses,
    )

    final_path = (
        OUTPUT_DIR
        / "real_speaker_conditioned_acoustic_final.pt"
    )

    save_checkpoint(
        model,
        final_path,
        EPOCHS,
        val_losses[-1],
    )

    print()
    print(
        f"Best checkpoint:"
    )

    print(
        f"    {best_path}"
    )

    print(
        f"Final checkpoint:"
    )

    print(
        f"    {final_path}"
    )

    # ========================================================================
    # STEP 10
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 10 — REAL REFERENCE-AUDIO INFERENCE"
    )

    print("=" * 72)

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    reference = (
        select_reference_audio(
            records
        )
    )

    reference_path = (
        reference["path"]
    )

    print()
    print(
        "Reference speaker:"
    )

    print(
        f"    Speaker "
        f"{reference['speaker']:03d}"
    )

    print(
        f"Reference audio:"
    )

    print(
        f"    {reference_path}"
    )

    synthesis_text = (
        "this is a new sentence "
        "generated with the reference speaker"
    )

    print()
    print(
        "Text:"
    )

    print(
        f"    {synthesis_text}"
    )

    generated_mel, speaker_embedding = (
        synthesize_mel(
            model,
            speaker_encoder,
            reference_path,
            synthesis_text,
        )
    )

    # ========================================================================
    # STEP 11
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 11 — SAVE REFERENCE EMBEDDING"
    )

    print("=" * 72)

    embedding_path = (
        OUTPUT_DIR
        / "reference_speaker_embedding.pt"
    )

    torch.save(
        {
            "speaker_embedding":
                torch.tensor(
                    speaker_embedding
                ),
            "reference_audio":
                str(reference_path),
            "speaker":
                reference[
                    "speaker"
                ],
        },
        embedding_path,
    )

    print(
        f"Saved:"
    )

    print(
        f"    {embedding_path}"
    )

    # ========================================================================
    # STEP 12
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 12 — SAVE GENERATED MEL"
    )

    print("=" * 72)

    generated_mel_path = (
        OUTPUT_DIR
        / "reference_conditioned_generated_mel.npy"
    )

    np.save(
        generated_mel_path,
        generated_mel,
    )

    print(
        f"Saved:"
    )

    print(
        f"    {generated_mel_path}"
    )

    save_mel_plot(
        generated_mel,
        "Generated Mel — Real Reference Speaker",
        "reference_conditioned_generated_mel.png",
    )

    # ========================================================================
    # STEP 13
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "STEP 13 — COMPARE SAME SPEAKER REPRESENTATION"
    )

    print("=" * 72)

    same_speaker_records = [
        record
        for record in records
        if (
            record["speaker"]
            == reference[
                "speaker"
            ]
        )
    ]

    if len(
        same_speaker_records
    ) >= 2:

        second_reference = (
            same_speaker_records[1]
        )

        second_embedding = (
            extract_reference_embedding(
                speaker_encoder,
                second_reference[
                    "path"
                ],
            )
        )

        first_embedding = (
            torch.tensor(
                speaker_embedding
            )
        )

        similarity = F.cosine_similarity(
            first_embedding.unsqueeze(
                0
            ),
            second_embedding.unsqueeze(
                0
            ),
        ).item()

        print()
        print(
            "Same-speaker reference similarity:"
        )

        print(
            f"    {similarity:.4f}"
        )

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print("=" * 72)

    print(
        "LESSON 67 COMPLETE"
    )

    print("=" * 72)

    print()
    print(
        "REAL AUDIO"
    )

    print(
        "    ↓"
    )

    print(
        "SPEAKER ENCODER"
    )

    print(
        "    ↓"
    )

    print(
        "REAL SPEAKER EMBEDDING"
    )

    print(
        "    ↓"
    )

    print(
        "TEXT + SPEAKER EMBEDDING"
    )

    print(
        "    ↓"
    )

    print(
        "ACOUSTIC MODEL"
    )

    print(
        "    ↓"
    )

    print(
        "MEL SPECTROGRAM"
    )

    print()
    print(
        "This completes the real-reference speaker-conditioning milestone."
    )

    print()
    print("=" * 72)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()