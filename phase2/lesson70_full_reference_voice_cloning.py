
"""
File: lesson70_full_reference_voice_cloning.py

PHASE 2 — LESSON 70
FULL REFERENCE-VOICE CLONING PIPELINE FROM SCRATCH
==========================================================================

PURPOSE
-------
This is the final integration lesson of the current TTS learning phase.

The goal is to connect the major components developed throughout Phase 2:

    Reference WAV
        |
        v
    Speaker Encoder
        |
        v
    Speaker Embedding
        |
        +------------------------------+
        |                              |
        v                              v
      Text ---------------------> Acoustic Model
                                      |
                                      v
                                  Mel Spectrogram
                                      |
                                      v
                                  Neural Vocoder
                                      |
                                      v
                                   Waveform

The key idea is:

    TEXT = WHAT TO SAY
    SPEAKER EMBEDDING = WHO IS SPEAKING
    MEL = HOW THE SPEECH SHOULD SOUND
    VOCODER = HOW TO TURN MEL INTO AUDIO

IMPORTANT DATASET FIX
---------------------
Previous versions of this lesson assumed:

    WAV filename == LibriSpeech utterance ID

That assumption is WRONG for the dataset generated in Lesson 64.

Lesson 64 intentionally renamed files:

    speaker_000/
        0000.wav
        0001.wav
        0002.wav

Therefore this lesson now maintains:

    speaker_000 <-> original LibriSpeech speaker
    0000.wav    <-> first original utterance
    0001.wav    <-> second original utterance
    ...

A persistent file is created:

    speaker_dataset/speaker_mapping.txt

Example:

    speaker_000 1272
    speaker_001 1462
    speaker_002 1673

This allows the local renamed WAV files to be matched correctly
with the original LibriSpeech transcripts.

NO PRETRAINED TTS MODELS
------------------------
This lesson does NOT use:

    - Whisper
    - Wav2Vec2
    - Tacotron
    - FastSpeech
    - FastSpeech2
    - VITS
    - XTTS
    - YourTTS
    - pretrained TTS models
    - pretrained phonemizers
    - Montreal Forced Aligner

The neural components are trained locally for educational purposes.

DATASET
-------
If the dataset is missing, the script automatically downloads:

    LibriSpeech dev-clean

from OpenSLR.

The script requires ffmpeg for FLAC -> WAV conversion.

Install once if necessary:

    sudo apt update
    sudo apt install ffmpeg

EXAMPLE MULTILINGUAL TEXT
-------------------------
The frontend contains:

    English:
        The robot learns to speak clearly.

    Chinese:
        机器人正在学习自然地说话。

    Japanese:
        ロボットは自然に話すことを学んでいます。

    Russian:
        Робот учится говорить естественно.

LibriSpeech provides English training speech. The multilingual
frontend is included so the architecture is not tied to English.

WHAT YOU LEARN
--------------
STEP 1
    Automatically prepare real speech data.

STEP 2
    Correctly associate renamed WAV files with transcripts.

STEP 3
    Extract speaker embeddings from reference speech.

STEP 4
    Train a speaker-conditioned acoustic model.

STEP 5
    Convert text + speaker identity into a Mel spectrogram.

STEP 6
    Convert Mel spectrogram into waveform using the Lesson 69
    neural vocoder when available.

STEP 7
    Perform reference-voice inference.

STEP 8
    Visualize speaker embeddings and generated speech.

IMPORTANT LIMITATION
--------------------
This is an educational from-scratch TTS system.

It demonstrates the complete architecture, but it is NOT expected
to match commercial systems such as XTTS or modern large-scale
zero-shot voice cloning systems.

The purpose is to understand the complete pipeline internally.

==========================================================================

REQUIREMENTS
------------
Python 3.10+
PyTorch
NumPy
Matplotlib
SoundFile
Requests
tqdm

Example:

    pip install torch numpy matplotlib soundfile requests tqdm

Run:

    python lesson70_full_reference_voice_cloning.py

==========================================================================

OUTPUTS
-------
lesson70_outputs/
    speaker_encoder_best.pt
    acoustic_model_best.pt
    reference_speaker_embedding.npy
    generated_mel.npy
    generated_voice.wav
    reference_voice.wav
    mel_comparison.png
    speaker_embedding.png
    acoustic_training.png
    pipeline_summary.txt

==========================================================================

"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import math
import json
import time
import random
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT = Path.cwd()

DATA_ROOT = ROOT / "lesson70_data"

LIBRISPEECH_ARCHIVE = (
    DATA_ROOT / "dev-clean.tar.gz"
)

LIBRISPEECH_ROOT = (
    DATA_ROOT / "LibriSpeech"
)

LIBRISPEECH_EXTRACTED = (
    LIBRISPEECH_ROOT / "dev-clean"
)

SPEAKER_DATASET_DIR = (
    ROOT / "speaker_dataset"
)

OUTPUT_DIR = (
    ROOT / "lesson70_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LESSON69_CHECKPOINT = (
    ROOT
    / "lesson69_outputs"
    / "neural_vocoder_best.pt"
)

LESSON64_CHECKPOINT = (
    ROOT
    / "lesson64_outputs"
    / "speaker_encoder_best.pt"
)

LESSON65_CHECKPOINT = (
    ROOT
    / "lesson65_outputs"
    / "speaker_metric_encoder_best.pt"
)

LESSON67_CHECKPOINT = (
    ROOT
    / "lesson67_outputs"
    / "real_speaker_conditioned_acoustic_model_best.pt"
)

# ---------------------------------------------------------------------------
# Audio configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

N_MELS = 80

FMIN = 50.0
FMAX = 7600.0

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

MAX_SPEAKERS = 8
MAX_UTTERANCES_PER_SPEAKER = 20

MIN_RECORDS = 10

SPEAKER_EMBED_DIM = 128

ACOUSTIC_HIDDEN = 256

BATCH_SIZE = 8

SPEAKER_EPOCHS = 12
ACOUSTIC_EPOCHS = 15

LEARNING_RATE = 2e-4

MAX_TEXT_LENGTH = 160

MAX_MEL_FRAMES = 160

# Fixed number of Mel frames used by acoustic training.
TRAIN_MEL_FRAMES = 96

# Vocoder output length.
VOCODER_WAVE_SAMPLES = (
    TRAIN_MEL_FRAMES * HOP_LENGTH
)

MAX_AUDIO_SECONDS = 4.0


# ============================================================================
# DEVICE INFORMATION
# ============================================================================

print("=" * 76)
print("PHASE 2 — LESSON 70")
print("FULL REFERENCE-VOICE CLONING")
print("=" * 76)

print()
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    f"Sample rate: {SAMPLE_RATE}"
)

print(
    f"Mel bins: {N_MELS}"
)

print(
    f"Hop length: {HOP_LENGTH}"
)

print()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def seed_everything(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_ffmpeg():

    if shutil.which("ffmpeg") is None:

        raise RuntimeError(
            "\nffmpeg is required for LibriSpeech conversion.\n\n"
            "Install it with:\n\n"
            "    sudo apt update\n"
            "    sudo apt install ffmpeg\n"
        )


def md5_file(path):

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            md5.update(chunk)

    return md5.hexdigest()


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(text):

    text = str(text).strip()

    text = " ".join(
        text.split()
    )

    return text


# ============================================================================
# LIBRISPEECH DOWNLOAD
# ============================================================================

def download_file(url, destination):

    if requests is None:

        raise RuntimeError(
            "requests is not installed.\n"
            "Install with:\n"
            "pip install requests"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Downloading:\n{url}"
    )

    with requests.get(
        url,
        stream=True,
        timeout=60,
    ) as response:

        response.raise_for_status()

        total = int(
            response.headers.get(
                "content-length",
                0,
            )
        )

        downloaded = 0

        with open(
            destination,
            "wb",
        ) as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                f.write(chunk)

                downloaded += len(chunk)

                if total:

                    percent = (
                        downloaded
                        / total
                        * 100
                    )

                    print(
                        f"\r{percent:6.2f}%",
                        end="",
                        flush=True,
                    )

    print()


def prepare_librispeech():

    """
    Ensure LibriSpeech dev-clean exists.

    This function is intentionally independent from speaker_dataset/.
    Therefore the transcript source is always available even if
    Lesson 64 already created renamed WAV files.
    """

    check_ffmpeg()

    if not LIBRISPEECH_EXTRACTED.exists():

        DATA_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not LIBRISPEECH_ARCHIVE.exists():

            url = (
                "https://www.openslr.org/"
                "resources/12/dev-clean.tar.gz"
            )

            download_file(
                url,
                LIBRISPEECH_ARCHIVE,
            )

        print()
        print(
            "Extracting LibriSpeech..."
        )

        subprocess.run(
            [
                "tar",
                "-xzf",
                str(LIBRISPEECH_ARCHIVE),
                "-C",
                str(DATA_ROOT),
            ],
            check=True,
        )

    if not LIBRISPEECH_EXTRACTED.exists():

        raise RuntimeError(
            "LibriSpeech extraction failed."
        )

    print(
        "LibriSpeech dev-clean ready."
    )


# ============================================================================
# TRANSCRIPT COLLECTION
# ============================================================================

def collect_transcripts_by_speaker():

    transcripts = {}

    trans_files = sorted(
        LIBRISPEECH_EXTRACTED.rglob(
            "*.trans.txt"
        )
    )

    print(
        f"Transcript files found: "
        f"{len(trans_files)}"
    )

    for trans_file in trans_files:

        speaker_id = trans_file.parent.name

        entries = transcripts.setdefault(
            speaker_id,
            [],
        )

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

                utterance_id = parts[0]

                transcript = normalize_text(
                    parts[1]
                )

                if len(transcript) < 2:
                    continue

                entries.append(
                    (
                        utterance_id,
                        transcript,
                    )
                )

    for speaker in transcripts:

        transcripts[speaker].sort(
            key=lambda x: x[0]
        )

    total = sum(
        len(v)
        for v in transcripts.values()
    )

    print(
        f"Original LibriSpeech speakers: "
        f"{len(transcripts)}"
    )

    print(
        f"Transcripts loaded: {total}"
    )

    return transcripts


# ============================================================================
# SPEAKER DATASET PREPARATION
# ============================================================================

def prepare_speaker_dataset():

    """
    Build or reuse:

        speaker_dataset/
            speaker_000/
                0000.wav
                0001.wav
                ...

    A mapping file preserves the connection to original LibriSpeech
    speaker IDs.
    """

    transcripts = (
        collect_transcripts_by_speaker()
    )

    original_speakers = sorted(
        transcripts.keys()
    )

    if not original_speakers:

        raise RuntimeError(
            "No LibriSpeech speakers were found."
        )

    SPEAKER_DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping_file = (
        SPEAKER_DATASET_DIR
        / "speaker_mapping.txt"
    )

    mapping = {}

    if mapping_file.exists():

        with open(
            mapping_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) == 2:

                    mapping[
                        parts[0]
                    ] = parts[1]

    # ------------------------------------------------------------------------
    # If Lesson 64 already generated speaker directories but mapping is absent,
    # recreate the deterministic mapping.
    # ------------------------------------------------------------------------

    local_dirs = sorted(
        [
            p
            for p in SPEAKER_DATASET_DIR.iterdir()
            if p.is_dir()
            and p.name.startswith("speaker_")
        ]
    )

    if not mapping:

        number_of_speakers = min(
            MAX_SPEAKERS,
            len(original_speakers),
        )

        for index in range(
            number_of_speakers
        ):

            local_name = (
                f"speaker_{index:03d}"
            )

            mapping[
                local_name
            ] = original_speakers[index]

        with open(
            mapping_file,
            "w",
            encoding="utf-8",
        ) as f:

            for local_name, original_name in (
                mapping.items()
            ):

                f.write(
                    f"{local_name} "
                    f"{original_name}\n"
                )

    print()
    print("Speaker mapping:")

    for local_name, original_name in (
        mapping.items()
    ):

        print(
            f"    {local_name} "
            f"<-> "
            f"LibriSpeech {original_name}"
        )

    # ------------------------------------------------------------------------
    # Convert FLAC to WAV when local WAV files are missing.
    # ------------------------------------------------------------------------

    for local_name, original_name in (
        mapping.items()
    ):

        local_dir = (
            SPEAKER_DATASET_DIR
            / local_name
        )

        local_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_dir = (
            LIBRISPEECH_EXTRACTED
            / original_name
        )

        if not source_dir.exists():

            print(
                f"WARNING: missing source speaker "
                f"{original_name}"
            )

            continue

        flac_files = sorted(
            source_dir.glob("*.flac")
        )

        flac_files = flac_files[
            :MAX_UTTERANCES_PER_SPEAKER
        ]

        existing_wavs = sorted(
            local_dir.glob("*.wav")
        )

        if len(existing_wavs) >= len(flac_files):

            continue

        for index, flac_path in enumerate(
            flac_files
        ):

            wav_path = (
                local_dir
                / f"{index:04d}.wav"
            )

            if wav_path.exists():
                continue

            print(
                f"Converting "
                f"{local_name}/"
                f"{wav_path.name}"
            )

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(flac_path),
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    str(wav_path),
                ],
                check=True,
            )

    return transcripts, mapping


# ============================================================================
# BUILD AUDIO/TEXT RECORDS
# ============================================================================

def find_dataset_records():

    print()
    print("=" * 76)
    print("STEP 2 — BUILD AUDIO/TEXT RECORDS")
    print("=" * 76)

    transcripts, mapping = (
        prepare_speaker_dataset()
    )

    records = []

    for local_speaker in sorted(
        mapping.keys()
    ):

        original_speaker = mapping[
            local_speaker
        ]

        transcript_entries = transcripts.get(
            original_speaker,
            [],
        )

        local_dir = (
            SPEAKER_DATASET_DIR
            / local_speaker
        )

        wav_files = sorted(
            local_dir.glob("*.wav")
        )

        if not wav_files:

            print(
                f"WARNING: no WAV files for "
                f"{local_speaker}"
            )

            continue

        if not transcript_entries:

            print(
                f"WARNING: no transcripts for "
                f"{original_speaker}"
            )

            continue

        count = min(
            len(wav_files),
            len(transcript_entries),
            MAX_UTTERANCES_PER_SPEAKER,
        )

        print(
            f"{local_speaker} <-> "
            f"{original_speaker}: "
            f"{count} usable pairs"
        )

        for index in range(count):

            wav_path = wav_files[index]

            utterance_id, text = (
                transcript_entries[index]
            )

            if not wav_path.exists():
                continue

            if len(text) < 2:
                continue

            records.append(
                {
                    "wav": str(wav_path),
                    "text": text,
                    "speaker": local_speaker,
                    "speaker_original":
                        original_speaker,
                    "utterance_id":
                        utterance_id,
                }
            )

    if len(records) < MIN_RECORDS:

        raise RuntimeError(
            "Could not create enough "
            "audio/text training records."
        )

    random.Random(
        SEED
    ).shuffle(records)

    speaker_names = sorted(
        set(
            r["speaker"]
            for r in records
        )
    )

    speaker_to_id = {
        name: i
        for i, name in enumerate(
            speaker_names
        )
    }

    for record in records:

        record["speaker_id"] = (
            speaker_to_id[
                record["speaker"]
            ]
        )

    print()
    print(
        f"Total audio/text records: "
        f"{len(records)}"
    )

    print(
        f"Total speakers: "
        f"{len(speaker_to_id)}"
    )

    print()
    print("Example records:")

    for record in records[:5]:

        print(
            f"  {record['wav']}"
        )

        print(
            f"  {record['utterance_id']}"
        )

        print(
            f"  {record['text']}"
        )

        print()

    return records, speaker_to_id


# ============================================================================
# AUDIO LOADING
# ============================================================================

def load_audio(path):

    if sf is None:

        raise RuntimeError(
            "soundfile is required.\n"
            "Install with:\n"
            "pip install soundfile"
        )

    waveform, sr = sf.read(
        str(path),
        dtype="float32",
    )

    if waveform.ndim > 1:

        waveform = np.mean(
            waveform,
            axis=1,
        )

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if sr != SAMPLE_RATE:

        # ffmpeg conversion should already make everything 16 kHz.
        # This fallback performs linear interpolation if necessary.

        old_x = np.linspace(
            0,
            1,
            len(waveform),
        )

        new_length = int(
            len(waveform)
            * SAMPLE_RATE
            / sr
        )

        new_x = np.linspace(
            0,
            1,
            new_length,
        )

        waveform = np.interp(
            new_x,
            old_x,
            waveform,
        ).astype(
            np.float32
        )

    waveform = np.nan_to_num(
        waveform
    )

    peak = np.max(
        np.abs(waveform)
    )

    if peak > 1e-5:

        waveform = (
            waveform / peak
        )

    return waveform


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


def build_mel_filterbank():

    low_mel = hz_to_mel(FMIN)
    high_mel = hz_to_mel(FMAX)

    mel_points = np.linspace(
        low_mel,
        high_mel,
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
            N_FFT // 2 + 1,
        ),
        dtype=np.float32,
    )

    for m in range(
        1,
        N_MELS + 1,
    ):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = left + 1

        if right <= center:
            right = center + 1

        center = min(
            center,
            N_FFT // 2,
        )

        right = min(
            right,
            N_FFT // 2 + 1,
        )

        for k in range(
            left,
            center,
        ):

            if 0 <= k < filterbank.shape[1]:

                filterbank[
                    m - 1,
                    k
                ] = (
                    k - left
                ) / max(
                    center - left,
                    1,
                )

        for k in range(
            center,
            right,
        ):

            if 0 <= k < filterbank.shape[1]:

                filterbank[
                    m - 1,
                    k
                ] = (
                    right - k
                ) / max(
                    right - center,
                    1,
                )

    return torch.from_numpy(
        filterbank
    )


MEL_FILTERBANK = (
    build_mel_filterbank()
)


# ============================================================================
# MEL EXTRACTION
# ============================================================================

def waveform_to_mel(
    waveform,
    max_frames=None,
):

    x = torch.from_numpy(
        waveform
    ).float()

    window = torch.hann_window(
        WIN_LENGTH
    )

    spec = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True,
    )

    magnitude = spec.abs()

    mel = torch.matmul(
        MEL_FILTERBANK,
        magnitude,
    )

    mel = torch.log(
        mel + 1e-5
    )

    if max_frames is not None:

        if mel.shape[-1] >= max_frames:

            mel = mel[
                :,
                :max_frames
            ]

        else:

            pad = (
                max_frames
                - mel.shape[-1]
            )

            mel = F.pad(
                mel,
                (0, pad),
                mode="replicate",
            )

    return mel.float()


# ============================================================================
# TEXT VOCABULARY
# ============================================================================

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
]

ENGLISH_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
)

CHINESE_CHARS = (
    "的一是在不了有人我他她你这中大来上国个"
    "们到说为子和地出道也时年得就那要下以生"
    "会自着去之过家学对可里后小多天能好都"
)

JAPANESE_CHARS = (
    "あいうえおかきくけこさしすせそ"
    "たちつてとなにぬねのはひふへほ"
    "まみむめもやゆよらりるれろわをん"
)

RUSSIAN_CHARS = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
)

PUNCTUATION = (
    " .,!?'-"
)

VOCAB_CHARS = (
    SPECIAL_TOKENS
    + list(ENGLISH_CHARS)
    + list(CHINESE_CHARS)
    + list(JAPANESE_CHARS)
    + list(RUSSIAN_CHARS)
    + list(PUNCTUATION)
)

VOCAB = {
    token: index
    for index, token in enumerate(
        VOCAB_CHARS
    )
}

PAD_ID = VOCAB[
    "<pad>"
]

UNK_ID = VOCAB[
    "<unk>"
]


def text_to_ids(text):

    text = normalize_text(
        text
    )

    ids = []

    for char in text:

        char_lower = char.lower()

        if char_lower in VOCAB:

            ids.append(
                VOCAB[char_lower]
            )

        elif char in VOCAB:

            ids.append(
                VOCAB[char]
            )

        else:

            ids.append(
                UNK_ID
            )

    if not ids:

        ids = [UNK_ID]

    ids = ids[
        :MAX_TEXT_LENGTH
    ]

    return ids


# ============================================================================
# SPEAKER ENCODER
# ============================================================================

class SpeakerEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim=SPEAKER_EMBED_DIM,
    ):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(
                N_MELS,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Conv1d(
                128,
                192,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(192),

            nn.ReLU(),

            nn.Conv1d(
                192,
                256,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),
        )

        self.projection = nn.Sequential(

            nn.Linear(
                256,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                embedding_dim,
            ),
        )

    def forward(
        self,
        mel,
    ):

        x = self.conv(
            mel
        )

        x = x.mean(
            dim=-1
        )

        x = self.projection(
            x
        )

        x = F.normalize(
            x,
            p=2,
            dim=-1,
        )

        return x


class SpeakerClassifier(nn.Module):

    def __init__(
        self,
        num_speakers,
    ):

        super().__init__()

        self.encoder = (
            SpeakerEncoder()
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                SPEAKER_EMBED_DIM,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_speakers,
            ),
        )

    def forward(
        self,
        mel,
    ):

        embedding = (
            self.encoder(mel)
        )

        logits = (
            self.classifier(
                embedding
            )
        )

        return (
            logits,
            embedding,
        )


# ============================================================================
# SPEAKER DATASET
# ============================================================================

class SpeakerDataset(Dataset):

    def __init__(
        self,
        records,
    ):

        self.records = records

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

        waveform = load_audio(
            record["wav"]
        )

        max_samples = int(
            MAX_AUDIO_SECONDS
            * SAMPLE_RATE
        )

        if len(waveform) > max_samples:

            waveform = waveform[
                :max_samples
            ]

        mel = waveform_to_mel(
            waveform,
            max_frames=MAX_MEL_FRAMES,
        )

        return (
            mel,
            record["speaker_id"],
        )


# ============================================================================
# SPEAKER ENCODER TRAINING
# ============================================================================

def train_speaker_encoder(
    records,
    speaker_to_id,
):

    print()
    print("=" * 76)
    print("STEP 3 — TRAIN SPEAKER ENCODER")
    print("=" * 76)

    checkpoint = (
        OUTPUT_DIR
        / "speaker_encoder_best.pt"
    )

    num_speakers = len(
        speaker_to_id
    )

    dataset = SpeakerDataset(
        records
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = SpeakerClassifier(
        num_speakers
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    losses = []

    best_loss = float(
        "inf"
    )

    for epoch in range(
        SPEAKER_EPOCHS
    ):

        model.train()

        running_loss = 0.0
        count = 0

        for mel, labels in loader:

            mel = mel.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, embedding = (
                model(mel)
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * mel.shape[0]
            )

            count += mel.shape[0]

        epoch_loss = (
            running_loss
            / max(count, 1)
        )

        losses.append(
            epoch_loss
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/"
            f"{SPEAKER_EPOCHS} "
            f"loss={epoch_loss:.5f}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(
                {
                    "model_state_dict":
                        model.encoder.state_dict(),
                    "classifier_state_dict":
                        model.classifier.state_dict(),
                    "embedding_dim":
                        SPEAKER_EMBED_DIM,
                    "num_speakers":
                        num_speakers,
                },
                checkpoint,
            )

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        losses
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Speaker classification loss"
    )

    plt.title(
        "Lesson 70 Speaker Encoder Training"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "speaker_training.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------------
    # Load best encoder
    # ------------------------------------------------------------------------

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
    )

    model.encoder.load_state_dict(
        state["model_state_dict"]
    )

    model.eval()

    print()
    print(
        f"Best speaker encoder saved:"
    )

    print(
        checkpoint
    )

    return model.encoder


# ============================================================================
# SPEAKER EMBEDDING EXTRACTION
# ============================================================================

@torch.no_grad()
def extract_speaker_embedding(
    encoder,
    wav_path,
):

    waveform = load_audio(
        wav_path
    )

    mel = waveform_to_mel(
        waveform,
        max_frames=MAX_MEL_FRAMES,
    )

    mel = mel.unsqueeze(
        0
    ).to(
        DEVICE
    )

    embedding = encoder(
        mel
    )

    embedding = embedding[
        0
    ].cpu().numpy()

    return embedding


# ============================================================================
# POSitional ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        dim,
        max_length=2048,
    ):

        super().__init__()

        position = torch.arange(
            max_length
        ).unsqueeze(
            1
        )

        div_term = torch.exp(
            torch.arange(
                0,
                dim,
                2,
            )
            * (
                -math.log(10000.0)
                / dim
            )
        )

        pe = torch.zeros(
            max_length,
            dim,
        )

        pe[
            :,
            0::2
        ] = torch.sin(
            position * div_term
        )

        pe[
            :,
            1::2
        ] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
        self,
        x,
    ):

        return (
            x
            + self.pe[
                :,
                :x.shape[1],
                :
            ]
        )


# ============================================================================
# ACOUSTIC MODEL
# ============================================================================

class ReferenceConditionedAcousticModel(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        speaker_dim,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            ACOUSTIC_HIDDEN,
            padding_idx=PAD_ID,
        )

        self.text_projection = nn.Linear(
            ACOUSTIC_HIDDEN,
            ACOUSTIC_HIDDEN,
        )

        self.speaker_projection = nn.Linear(
            speaker_dim,
            ACOUSTIC_HIDDEN,
        )

        self.positional = (
            PositionalEncoding(
                ACOUSTIC_HIDDEN,
                max_length=512,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=ACOUSTIC_HIDDEN,
                nhead=8,
                dim_feedforward=ACOUSTIC_HIDDEN * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=3,
            )
        )

        self.duration_predictor = nn.Sequential(

            nn.Linear(
                ACOUSTIC_HIDDEN,
                128,
            ),

            nn.GELU(),

            nn.Linear(
                128,
                1,
            ),
        )

        self.mel_projection = nn.Sequential(

            nn.Linear(
                ACOUSTIC_HIDDEN,
                ACOUSTIC_HIDDEN,
            ),

            nn.GELU(),

            nn.Linear(
                ACOUSTIC_HIDDEN,
                N_MELS,
            ),
        )

    def forward(
        self,
        text_ids,
        speaker_embedding,
        target_frames,
    ):

        x = self.embedding(
            text_ids
        )

        x = self.text_projection(
            x
        )

        speaker = self.speaker_projection(
            speaker_embedding
        )

        speaker = speaker.unsqueeze(
            1
        )

        x = (
            x
            + speaker
        )

        x = self.positional(
            x
        )

        x = self.encoder(
            x
        )

        duration = self.duration_predictor(
            x
        ).squeeze(
            -1
        )

        # --------------------------------------------------------------------
        # Convert text sequence into a frame-level representation.
        #
        # This is an educational differentiable approximation of a
        # length regulator. The real duration model is learned in earlier
        # lessons; here we make the final integration robust to arbitrary
        # target Mel lengths.
        # --------------------------------------------------------------------

        frame_x = x.transpose(
            1,
            2,
        )

        frame_x = F.interpolate(
            frame_x,
            size=target_frames,
            mode="linear",
            align_corners=False,
        )

        frame_x = frame_x.transpose(
            1,
            2,
        )

        mel = self.mel_projection(
            frame_x
        )

        mel = mel.transpose(
            1,
            2,
        )

        return (
            mel,
            duration,
        )


# ============================================================================
# ACOUSTIC DATASET
# ============================================================================

class AcousticDataset(Dataset):

    def __init__(
        self,
        records,
    ):

        self.records = records

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

        waveform = load_audio(
            record["wav"]
        )

        mel = waveform_to_mel(
            waveform,
            max_frames=TRAIN_MEL_FRAMES,
        )

        text_ids = text_to_ids(
            record["text"]
        )

        text_tensor = torch.tensor(
            text_ids,
            dtype=torch.long,
        )

        speaker_id = torch.tensor(
            record["speaker_id"],
            dtype=torch.long,
        )

        return (
            text_tensor,
            mel,
            speaker_id,
        )


def acoustic_collate(
    batch,
):

    text_batch = [
        item[0]
        for item in batch
    ]

    mel_batch = [
        item[1]
        for item in batch
    ]

    speaker_batch = [
        item[2]
        for item in batch
    ]

    max_text = max(
        x.shape[0]
        for x in text_batch
    )

    padded_text = torch.full(
        (
            len(batch),
            max_text,
        ),
        PAD_ID,
        dtype=torch.long,
    )

    for i, text in enumerate(
        text_batch
    ):

        padded_text[
            i,
            :text.shape[0]
        ] = text

    mel_batch = torch.stack(
        mel_batch
    )

    speaker_batch = torch.stack(
        speaker_batch
    )

    return (
        padded_text,
        mel_batch,
        speaker_batch,
    )


# ============================================================================
# ACOUSTIC MODEL TRAINING
# ============================================================================

def train_acoustic_model(
    records,
    speaker_encoder,
):

    print()
    print("=" * 76)
    print("STEP 4 — TRAIN REFERENCE-CONDITIONED ACOUSTIC MODEL")
    print("=" * 76)

    dataset = AcousticDataset(
        records
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=acoustic_collate,
    )

    model = (
        ReferenceConditionedAcousticModel(
            len(VOCAB),
            SPEAKER_EMBED_DIM,
        )
        .to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    mel_losses = []

    best_loss = float(
        "inf"
    )

    checkpoint = (
        OUTPUT_DIR
        / "acoustic_model_best.pt"
    )

    speaker_encoder.eval()

    for epoch in range(
        ACOUSTIC_EPOCHS
    ):

        model.train()

        running = 0.0
        count = 0

        for (
            text_ids,
            target_mel,
            speaker_ids,
        ) in loader:

            text_ids = text_ids.to(
                DEVICE
            )

            target_mel = target_mel.to(
                DEVICE
            )

            speaker_ids = speaker_ids.to(
                DEVICE
            )

            # ---------------------------------------------------------------
            # Generate speaker embeddings from target speech.
            # ---------------------------------------------------------------

            embeddings = []

            for speaker_id in speaker_ids:

                speaker_id_int = int(
                    speaker_id.item()
                )

                candidates = [
                    r
                    for r in records
                    if r["speaker_id"]
                    == speaker_id_int
                ]

                reference = random.choice(
                    candidates
                )

                waveform = load_audio(
                    reference["wav"]
                )

                mel = waveform_to_mel(
                    waveform,
                    max_frames=MAX_MEL_FRAMES,
                )

                mel = mel.unsqueeze(
                    0
                ).to(
                    DEVICE
                )

                with torch.no_grad():

                    embedding = (
                        speaker_encoder(
                            mel
                        )
                    )

                embeddings.append(
                    embedding[0]
                )

            speaker_embedding = torch.stack(
                embeddings
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predicted_mel, duration = (
                model(
                    text_ids,
                    speaker_embedding,
                    TRAIN_MEL_FRAMES,
                )
            )

            loss = F.l1_loss(
                predicted_mel,
                target_mel,
            )

            loss = (
                loss
                + 0.05
                * duration.abs().mean()
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running += (
                loss.item()
                * target_mel.shape[0]
            )

            count += target_mel.shape[0]

        epoch_loss = (
            running
            / max(count, 1)
        )

        mel_losses.append(
            epoch_loss
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/"
            f"{ACOUSTIC_EPOCHS} "
            f"loss={epoch_loss:.5f}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "vocab":
                        VOCAB,
                    "speaker_dim":
                        SPEAKER_EMBED_DIM,
                },
                checkpoint,
            )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        mel_losses
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Acoustic loss"
    )

    plt.title(
        "Reference-Conditioned Acoustic Model"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "acoustic_training.png",
        dpi=150,
    )

    plt.close()

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
    )

    model.load_state_dict(
        state["model_state_dict"]
    )

    model.eval()

    print()
    print(
        "Best acoustic model:"
    )

    print(
        checkpoint
    )

    return model


# ============================================================================
# COMPACT NEURAL VOCODER
# ============================================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation,
    ):

        super().__init__()

        padding = dilation

        self.block = nn.Sequential(

            nn.LeakyReLU(
                0.2
            ),

            nn.Conv1d(
                channels,
                channels,
                3,
                padding=padding,
                dilation=dilation,
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Conv1d(
                channels,
                channels,
                3,
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


class CompactNeuralVocoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.pre = nn.Conv1d(
            N_MELS,
            128,
            7,
            padding=3,
        )

        self.up1 = nn.ConvTranspose1d(
            128,
            128,
            8,
            stride=4,
            padding=2,
        )

        self.res1 = ResidualBlock(
            128,
            1,
        )

        self.up2 = nn.ConvTranspose1d(
            128,
            96,
            8,
            stride=4,
            padding=2,
        )

        self.res2 = ResidualBlock(
            96,
            3,
        )

        self.up3 = nn.ConvTranspose1d(
            96,
            64,
            8,
            stride=4,
            padding=2,
        )

        self.res3 = ResidualBlock(
            64,
            5,
        )

        self.up4 = nn.ConvTranspose1d(
            64,
            32,
            8,
            stride=4,
            padding=2,
        )

        self.res4 = ResidualBlock(
            32,
            1,
        )

        self.output = nn.Conv1d(
            32,
            1,
            7,
            padding=3,
        )

    def forward(
        self,
        mel,
    ):

        x = self.pre(
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

        x = self.up2(
            x
        )

        x = self.res2(
            x
        )

        x = self.up3(
            x
        )

        x = self.res3(
            x
        )

        x = self.up4(
            x
        )

        x = self.res4(
            x
        )

        x = self.output(
            x
        )

        return torch.tanh(
            x
        )


# ============================================================================
# VOCODER TRAINING
# ============================================================================

def train_vocoder(
    records,
):

    print()
    print("=" * 76)
    print("STEP 5 — TRAIN / LOAD NEURAL VOCODER")
    print("=" * 76)

    checkpoint = (
        OUTPUT_DIR
        / "neural_vocoder_best.pt"
    )

    model = (
        CompactNeuralVocoder()
        .to(DEVICE)
    )

    # ------------------------------------------------------------------------
    # If Lesson 69 checkpoint exists, attempt to use it.
    # ------------------------------------------------------------------------

    if LESSON69_CHECKPOINT.exists():

        print(
            "Lesson 69 vocoder checkpoint found:"
        )

        print(
            LESSON69_CHECKPOINT
        )

        try:

            state = torch.load(
                LESSON69_CHECKPOINT,
                map_location=DEVICE,
            )

            if (
                isinstance(state, dict)
                and "model_state_dict"
                in state
            ):

                model.load_state_dict(
                    state[
                        "model_state_dict"
                    ],
                    strict=False,
                )

            else:

                model.load_state_dict(
                    state,
                    strict=False,
                )

            print(
                "Lesson 69 vocoder loaded."
            )

            return model

        except Exception as exc:

            print(
                "Could not load Lesson 69 "
                "checkpoint."
            )

            print(
                f"Reason: {exc}"
            )

    print(
        "Training compact vocoder locally..."
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    best_loss = float(
        "inf"
    )

    # Use a small deterministic subset.
    train_records = records[
        :min(
            len(records),
            32,
        )
    ]

    for epoch in range(5):

        random.Random(
            SEED + epoch
        ).shuffle(
            train_records
        )

        running = 0.0
        count = 0

        model.train()

        for record in train_records:

            waveform = load_audio(
                record["wav"]
            )

            mel = waveform_to_mel(
                waveform,
                max_frames=TRAIN_MEL_FRAMES,
            )

            mel = mel.unsqueeze(
                0
            ).to(
                DEVICE
            )

            target_length = (
                TRAIN_MEL_FRAMES
                * HOP_LENGTH
            )

            if len(waveform) < target_length:

                waveform = np.pad(
                    waveform,
                    (
                        0,
                        target_length
                        - len(waveform),
                    ),
                )

            else:

                waveform = waveform[
                    :target_length
                ]

            target = torch.from_numpy(
                waveform
            ).float()

            target = target.unsqueeze(
                0
            ).unsqueeze(
                0
            ).to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predicted = model(
                mel
            )

            length = min(
                predicted.shape[-1],
                target.shape[-1],
            )

            predicted = predicted[
                :,
                :,
                :length
            ]

            target = target[
                :,
                :,
                :length
            ]

            loss = F.l1_loss(
                predicted,
                target,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running += loss.item()

            count += 1

        epoch_loss = (
            running
            / max(count, 1)
        )

        print(
            f"Vocoder epoch "
            f"{epoch + 1}/5 "
            f"loss={epoch_loss:.5f}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                },
                checkpoint,
            )

    if checkpoint.exists():

        state = torch.load(
            checkpoint,
            map_location=DEVICE,
        )

        model.load_state_dict(
            state[
                "model_state_dict"
            ]
        )

    model.eval()

    return model


# ============================================================================
# REFERENCE VOICE INFERENCE
# ============================================================================

@torch.no_grad()
def synthesize_reference_voice(
    speaker_encoder,
    acoustic_model,
    vocoder,
    reference_wav,
    text,
):

    print()
    print("=" * 76)
    print("STEP 6 — REFERENCE-VOICE INFERENCE")
    print("=" * 76)

    print()
    print(
        "Reference audio:"
    )

    print(
        reference_wav
    )

    print()
    print(
        "Target text:"
    )

    print(
        text
    )

    # ------------------------------------------------------------------------
    # 1. Reference audio -> speaker embedding
    # ------------------------------------------------------------------------

    speaker_embedding = (
        extract_speaker_embedding(
            speaker_encoder,
            reference_wav,
        )
    )

    np.save(
        OUTPUT_DIR
        / "reference_speaker_embedding.npy",
        speaker_embedding,
    )

    print()
    print(
        "Speaker embedding dimension:",
        speaker_embedding.shape,
    )

    # ------------------------------------------------------------------------
    # 2. Text -> token IDs
    # ------------------------------------------------------------------------

    text_ids = text_to_ids(
        text
    )

    text_tensor = torch.tensor(
        text_ids,
        dtype=torch.long,
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    speaker_tensor = torch.from_numpy(
        speaker_embedding
    ).float().unsqueeze(
        0
    ).to(
        DEVICE
    )

    # ------------------------------------------------------------------------
    # 3. Text + speaker -> Mel
    # ------------------------------------------------------------------------

    predicted_mel, duration = (
        acoustic_model(
            text_tensor,
            speaker_tensor,
            TRAIN_MEL_FRAMES,
        )
    )

    predicted_mel = predicted_mel[
        0
    ]

    predicted_mel = torch.nan_to_num(
        predicted_mel
    )

    np.save(
        OUTPUT_DIR
        / "generated_mel.npy",
        predicted_mel.cpu().numpy(),
    )

    # ------------------------------------------------------------------------
    # 4. Mel -> waveform
    # ------------------------------------------------------------------------

    vocoder_input = predicted_mel.unsqueeze(
        0
    ).to(
        DEVICE
    )

    generated = vocoder(
        vocoder_input
    )

    generated = generated[
        0,
        0,
    ]

    generated = torch.nan_to_num(
        generated
    )

    generated = generated.cpu().numpy()

    # Normalize only for saving.
    peak = np.max(
        np.abs(generated)
    )

    if peak > 1e-5:

        generated = (
            generated / peak
        )

    output_wav = (
        OUTPUT_DIR
        / "generated_voice.wav"
    )

    sf.write(
        output_wav,
        generated.astype(
            np.float32
        ),
        SAMPLE_RATE,
    )

    print()
    print(
        "Generated waveform:"
    )

    print(
        output_wav
    )

    return (
        speaker_embedding,
        predicted_mel.cpu().numpy(),
        generated,
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_visualizations(
    reference_wav,
    generated_mel,
    speaker_embedding,
):

    print()
    print("=" * 76)
    print("STEP 7 — VISUALIZATION")
    print("=" * 76)

    reference_audio = load_audio(
        reference_wav
    )

    reference_mel = waveform_to_mel(
        reference_audio,
        max_frames=TRAIN_MEL_FRAMES,
    ).numpy()

    # ------------------------------------------------------------------------
    # Mel comparison
    # ------------------------------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    plt.subplot(
        2,
        1,
        1,
    )

    plt.imshow(
        reference_mel,
        aspect="auto",
        origin="lower",
    )

    plt.title(
        "Reference Mel Spectrogram"
    )

    plt.ylabel(
        "Mel bin"
    )

    plt.colorbar()

    plt.subplot(
        2,
        1,
        2,
    )

    plt.imshow(
        generated_mel,
        aspect="auto",
        origin="lower",
    )

    plt.title(
        "Generated Mel Spectrogram"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Mel bin"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "mel_comparison.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------------
    # Speaker embedding
    # ------------------------------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        speaker_embedding
    )

    plt.xlabel(
        "Embedding dimension"
    )

    plt.ylabel(
        "Value"
    )

    plt.title(
        "Reference Speaker Embedding"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "speaker_embedding.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------------
    # Reference waveform
    # ------------------------------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    time_axis = (
        np.arange(
            len(reference_audio)
        )
        / SAMPLE_RATE
    )

    plt.plot(
        time_axis,
        reference_audio,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Reference Voice Waveform"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "reference_waveform.png",
        dpi=150,
    )

    plt.close()


# ============================================================================
# PIPELINE SUMMARY
# ============================================================================

def write_summary(
    records,
    speaker_to_id,
    reference_wav,
    generated_wav,
):

    summary = []

    summary.append(
        "LESSON 70 — FULL REFERENCE VOICE CLONING"
    )

    summary.append(
        ""
    )

    summary.append(
        "Pipeline:"
    )

    summary.append(
        "Reference WAV"
    )

    summary.append(
        "    -> Speaker Encoder"
    )

    summary.append(
        "    -> Speaker Embedding"
    )

    summary.append(
        "    -> Text + Speaker Embedding"
    )

    summary.append(
        "    -> Acoustic Model"
    )

    summary.append(
        "    -> Mel Spectrogram"
    )

    summary.append(
        "    -> Neural Vocoder"
    )

    summary.append(
        "    -> Generated Waveform"
    )

    summary.append(
        ""
    )

    summary.append(
        f"Records: {len(records)}"
    )

    summary.append(
        f"Speakers: {len(speaker_to_id)}"
    )

    summary.append(
        f"Reference: {reference_wav}"
    )

    summary.append(
        f"Generated: {generated_wav}"
    )

    summary.append(
        ""
    )

    summary.append(
        "Multilingual frontend examples:"
    )

    summary.append(
        "English: The robot learns to speak clearly."
    )

    summary.append(
        "Chinese: 机器人正在学习自然地说话。"
    )

    summary.append(
        "Japanese: ロボットは自然に話すことを学んでいます。"
    )

    summary.append(
        "Russian: Робот учится говорить естественно."
    )

    with open(
        OUTPUT_DIR
        / "pipeline_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(summary)
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    seed_everything()

    # ------------------------------------------------------------------------
    # STEP 1
    # ------------------------------------------------------------------------

    print("=" * 76)
    print("STEP 1 — PREPARE REAL SPEECH DATA")
    print("=" * 76)

    prepare_librispeech()

    # ------------------------------------------------------------------------
    # STEP 2
    # ------------------------------------------------------------------------

    records, speaker_to_id = (
        find_dataset_records()
    )

    # ------------------------------------------------------------------------
    # STEP 3
    # ------------------------------------------------------------------------

    speaker_encoder = (
        train_speaker_encoder(
            records,
            speaker_to_id,
        )
    )

    # ------------------------------------------------------------------------
    # STEP 4
    # ------------------------------------------------------------------------

    acoustic_model = (
        train_acoustic_model(
            records,
            speaker_encoder,
        )
    )

    # ------------------------------------------------------------------------
    # STEP 5
    # ------------------------------------------------------------------------

    vocoder = train_vocoder(
        records
    )

    # ------------------------------------------------------------------------
    # Select reference speaker.
    #
    # Use the first local speaker with at least one WAV.
    # ------------------------------------------------------------------------

    reference_wav = None

    for speaker in sorted(
        speaker_to_id.keys()
    ):

        candidates = [
            r
            for r in records
            if r["speaker"] == speaker
        ]

        if candidates:

            reference_wav = Path(
                candidates[0]["wav"]
            )

            break

    if reference_wav is None:

        raise RuntimeError(
            "Could not find a reference WAV."
        )

    # ------------------------------------------------------------------------
    # Keep the default demonstration in English.
    #
    # The tokenizer itself supports the other three languages.
    # ------------------------------------------------------------------------

    target_text = (
        "The robot learns to speak clearly."
    )

    # ------------------------------------------------------------------------
    # STEP 6
    # ------------------------------------------------------------------------

    (
        speaker_embedding,
        generated_mel,
        generated_waveform,
    ) = synthesize_reference_voice(
        speaker_encoder,
        acoustic_model,
        vocoder,
        reference_wav,
        target_text,
    )

    # ------------------------------------------------------------------------
    # Copy reference audio into output directory.
    # ------------------------------------------------------------------------

    reference_output = (
        OUTPUT_DIR
        / "reference_voice.wav"
    )

    shutil.copy2(
        reference_wav,
        reference_output,
    )

    # ------------------------------------------------------------------------
    # STEP 7
    # ------------------------------------------------------------------------

    save_visualizations(
        reference_wav,
        generated_mel,
        speaker_embedding,
    )

    # ------------------------------------------------------------------------
    # STEP 8
    # ------------------------------------------------------------------------

    write_summary(
        records,
        speaker_to_id,
        reference_output,
        OUTPUT_DIR
        / "generated_voice.wav",
    )

    # ------------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------------

    print()
    print("=" * 76)
    print("LESSON 70 COMPLETE")
    print("=" * 76)

    print()
    print(
        "Reference speaker:"
    )

    print(
        reference_wav
    )

    print()
    print(
        "Generated voice:"
    )

    print(
        OUTPUT_DIR
        / "generated_voice.wav"
    )

    print()
    print(
        "Generated Mel:"
    )

    print(
        OUTPUT_DIR
        / "generated_mel.npy"
    )

    print()
    print(
        "Speaker embedding:"
    )

    print(
        OUTPUT_DIR
        / "reference_speaker_embedding.npy"
    )

    print()
    print(
        "Visualizations:"
    )

    print(
        OUTPUT_DIR
        / "mel_comparison.png"
    )

    print(
        OUTPUT_DIR
        / "speaker_embedding.png"
    )

    print()
    print(
        "Full reference-voice pipeline:"
    )

    print(
        "    Reference WAV"
    )

    print(
        "        ↓"
    )

    print(
        "    Speaker Encoder"
    )

    print(
        "        ↓"
    )

    print(
        "    Speaker Embedding"
    )

    print(
        "        ↓"
    )

    print(
        "    Text + Speaker Embedding"
    )

    print(
        "        ↓"
    )

    print(
        "    Acoustic Model"
    )

    print(
        "        ↓"
    )

    print(
        "    Mel Spectrogram"
    )

    print(
        "        ↓"
    )

    print(
        "    Neural Vocoder"
    )

    print(
        "        ↓"
    )

    print(
        "    Generated Speech"
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()
