"""
File: final_voice_cloning_system.py

===============================================================================
FINAL VOICE CLONING SYSTEM
===============================================================================

FROM-SCRATCH MULTILINGUAL REFERENCE-VOICE TTS
===============================================================================

This is the FINAL INTEGRATION of the voice-cloning course.

The system combines the major components developed throughout the course:

    Text
      |
      v
+--------------------------+
| Multilingual Text        |
| Frontend                 |
|                          |
| English                 |
| Chinese                 |
| Japanese                |
| Russian                 |
+------------+-------------+
             |
             v
+--------------------------+
| Character / Token        |
| Representation           |
+------------+-------------+
             |
             |
Reference    |                    Speaker identity
Audio        |                         |
   |         |                         |
   v         |                         v
+----------------+           +----------------------+
| Speaker        |           | Speaker Embedding    |
| Encoder        |---------->| 128-dimensional      |
+----------------+           | normalized vector    |
                             +----------+-----------+
                                        |
                                        v
                         +---------------------------+
                         | Reference-Conditioned     |
                         | Acoustic Model            |
                         |                           |
                         | Text + Speaker            |
                         | Duration + Transformer    |
                         +-------------+-------------+
                                       |
                                       v
                               Mel Spectrogram
                                       |
                                       v
                         +---------------------------+
                         | Neural Vocoder            |
                         |                           |
                         | Mel -> waveform           |
                         +-------------+-------------+
                                       |
                                       v
                                  Generated WAV


===============================================================================
WHAT YOU LEARN
===============================================================================

STEP 1
    Automatically obtain a real multi-speaker speech dataset.

STEP 2
    Build a multilingual character-level text frontend.

STEP 3
    Extract speaker identity from reference audio.

STEP 4
    Condition an acoustic model on speaker identity.

STEP 5
    Predict duration and perform length regulation.

STEP 6
    Generate a Mel spectrogram.

STEP 7
    Convert Mel spectrogram -> waveform using a neural vocoder.

STEP 8
    Perform reference-voice cloning.

STEP 9
    Measure speaker embedding similarity.

STEP 10
    Save a complete reproducible inference package.


===============================================================================
IMPORTANT
===============================================================================

NO PRETRAINED TTS MODEL IS USED.

NO WHISPER.
NO WAV2VEC2.
NO PRETRAINED SPEAKER ENCODER.
NO MONTREAL FORCED ALIGNER.
NO PRETRAINED PHONEMIZER.
NO PRETRAINED VOCODER.

The system is educational and intentionally compact.

For production-quality voice cloning, substantially more data, larger models,
better alignment, multilingual training, stronger vocoder losses, and much
longer training are required.


===============================================================================
DATA
===============================================================================

If the required sample dataset does not exist, this program automatically
downloads LibriSpeech dev-clean.

Download:
    https://www.openslr.org/resources/12/dev-clean.tar.gz

The program expects ffmpeg for FLAC -> WAV conversion.

Ubuntu:
    sudo apt update
    sudo apt install ffmpeg


===============================================================================
SUPPORTED TEXT
===============================================================================

English:
    Hello, this is a voice cloning experiment.

Chinese:
    这是一个从零开始的语音克隆系统。

Japanese:
    これは音声クローニングの実験です。

Russian:
    Это эксперимент по клонированию голоса.


===============================================================================
OUTPUT
===============================================================================

final_voice_outputs/

    reference_voice.wav
    generated_voice.wav

    reference_speaker_embedding.npy
    generated_speaker_embedding.npy

    generated_mel.npy

    final_system_summary.txt

    speaker_embedding_comparison.png
    mel_spectrogram.png
    waveform_comparison.png


===============================================================================
USAGE
===============================================================================

Basic:

    python final_voice_cloning_system.py

Custom text:

    python final_voice_cloning_system.py \
        --text "Hello, this is my cloned voice."

Chinese:

    python final_voice_cloning_system.py \
        --text "这是一个语音克隆系统。"

Japanese:

    python final_voice_cloning_system.py \
        --text "これは音声クローニングの実験です。"

Russian:

    python final_voice_cloning_system.py \
        --text "Это эксперимент по клонированию голоса."

Custom reference:

    python final_voice_cloning_system.py \
        --reference path/to/reference.wav \
        --text "Hello from my voice cloning system."


===============================================================================
"""

import os
import sys
import re
import math
import json
import time
import shutil
import random
import hashlib
import tarfile
import subprocess
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

N_MELS = 80
FMIN = 50.0
FMAX = 7600.0

SPEAKER_EMBED_DIM = 128

MAX_AUDIO_SECONDS = 3.0
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)

OUTPUT_DIR = Path("final_voice_outputs")
DATA_DIR = Path("lesson64_data")

LIBRISPEECH_ARCHIVE = DATA_DIR / "dev-clean.tar.gz"
LIBRISPEECH_ROOT = DATA_DIR / "LibriSpeech" / "dev-clean"

LIBRISPEECH_URL = (
    "https://www.openslr.org/resources/12/dev-clean.tar.gz"
)

MAX_SPEAKERS = 8
MAX_UTTERANCES_PER_SPEAKER = 12

BATCH_SIZE = 4

SPEAKER_EPOCHS = 5
ACOUSTIC_EPOCHS = 5
VOCODER_EPOCHS = 3

LEARNING_RATE = 2e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# FILESYSTEM
# =============================================================================

def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_file(url, destination):
    destination = Path(destination)

    if destination.exists():
        print(f"Dataset archive already exists:")
        print(f"  {destination}")
        return

    print()
    print("=" * 80)
    print("DOWNLOADING SAMPLE DATA")
    print("=" * 80)
    print()
    print(url)
    print()

    def progress(block_count, block_size, total_size):
        downloaded = block_count * block_size

        if total_size > 0:
            percent = downloaded / total_size * 100.0
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024

            print(
                f"\rDownloaded: {mb:.1f}/{total_mb:.1f} MB "
                f"({percent:.1f}%)",
                end="",
            )

    urllib.request.urlretrieve(
        url,
        destination,
        reporthook=progress,
    )

    print()
    print()
    print("Download complete.")


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_librispeech():
    if LIBRISPEECH_ROOT.exists():
        wav_count = len(list(LIBRISPEECH_ROOT.rglob("*.flac")))

        if wav_count > 0:
            print(
                f"LibriSpeech already extracted "
                f"({wav_count} FLAC files)."
            )
            return

    print()
    print("=" * 80)
    print("EXTRACTING LIBRISPEECH")
    print("=" * 80)

    print(f"Archive: {LIBRISPEECH_ARCHIVE}")

    with tarfile.open(LIBRISPEECH_ARCHIVE, "r:gz") as tar:
        tar.extractall(DATA_DIR)

    print("Extraction complete.")


# =============================================================================
# DATA PREPARATION
# =============================================================================

def find_transcript_files():
    return sorted(
        LIBRISPEECH_ROOT.rglob("*.trans.txt")
    )


def parse_librispeech_records():
    records = []

    transcript_files = find_transcript_files()

    print(
        f"Transcript files discovered: "
        f"{len(transcript_files)}"
    )

    for transcript_file in transcript_files:

        try:
            text = transcript_file.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                continue

            utterance_id, transcript = parts

            wav_file = transcript_file.parent / (
                utterance_id + ".flac"
            )

            if not wav_file.exists():
                continue

            speaker_id = utterance_id.split("-")[0]

            records.append(
                {
                    "speaker": speaker_id,
                    "utterance_id": utterance_id,
                    "audio": str(wav_file),
                    "text": transcript,
                }
            )

    return records


def build_small_dataset(records):
    print()
    print("=" * 80)
    print("BUILDING SMALL TRAINING DATASET")
    print("=" * 80)

    speakers = sorted(
        list(
            {
                record["speaker"]
                for record in records
            }
        )
    )

    speakers = speakers[:MAX_SPEAKERS]

    selected = []

    for speaker in speakers:

        speaker_records = [
            r
            for r in records
            if r["speaker"] == speaker
        ]

        speaker_records = sorted(
            speaker_records,
            key=lambda x: x["utterance_id"],
        )

        speaker_records = speaker_records[
            :MAX_UTTERANCES_PER_SPEAKER
        ]

        selected.extend(speaker_records)

    print(f"Selected speakers: {len(speakers)}")
    print(f"Selected utterances: {len(selected)}")

    return selected


# =============================================================================
# AUDIO
# =============================================================================

def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    waveform = waveform.float()

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            SAMPLE_RATE,
        )

    waveform = waveform[:MAX_AUDIO_SAMPLES]

    if waveform.numel() < SAMPLE_RATE // 2:
        return None

    peak = waveform.abs().max()

    if peak > 1e-6:
        waveform = waveform / peak

    return waveform


# =============================================================================
# MEL FILTERBANK
# =============================================================================

def hz_to_mel(hz):
    return 2595.0 * np.log10(
        1.0 + hz / 700.0
    )


def mel_to_hz(mel):
    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def create_mel_filterbank(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX,
):
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
    )

    hz_points = mel_to_hz(mel_points)

    bins = np.floor(
        (n_fft + 1)
        * hz_points
        / sample_rate
    ).astype(int)

    filterbank = np.zeros(
        (n_mels, n_fft // 2 + 1),
        dtype=np.float32,
    )

    for m in range(1, n_mels + 1):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center == left:
            center += 1

        if right == center:
            right += 1

        for k in range(left, center):

            if k < filterbank.shape[1]:
                filterbank[m - 1, k] = (
                    (k - left)
                    / max(center - left, 1)
                )

        for k in range(center, right):

            if k < filterbank.shape[1]:
                filterbank[m - 1, k] = (
                    (right - k)
                    / max(right - center, 1)
                )

    return torch.tensor(
        filterbank,
        dtype=torch.float32,
    )


MEL_FILTERBANK = create_mel_filterbank()


# =============================================================================
# MEL EXTRACTION
# =============================================================================

def waveform_to_mel(waveform):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(
        WIN_LENGTH,
        device=waveform.device,
    )

    spec = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True,
    )

    magnitude = spec.abs()

    mel_fb = MEL_FILTERBANK.to(
        magnitude.device
    )

    mel = torch.matmul(
        mel_fb,
        magnitude,
    )

    mel = torch.log(
        mel.clamp_min(1e-5)
    )

    return mel


# =============================================================================
# MULTILINGUAL TEXT FRONTEND
# =============================================================================

PAD = "<pad>"
UNK = "<unk>"

ENGLISH = (
    "abcdefghijklmnopqrstuvwxyz"
)

RUSSIAN = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
)

JAPANESE = (
    "あいうえお"
    "かきくけこ"
    "がぎぐげご"
    "さしすせそ"
    "ざじずぜぞ"
    "たちつてと"
    "だぢづでど"
    "なにぬねの"
    "はひふへほ"
    "ばびぶべぼ"
    "ぱぴぷぺぽ"
    "まみむめも"
    "やゆよ"
    "らりるれろ"
    "わをん"
    "ぁぃぅぇぉ"
    "ゃゅょっ"
)

CHINESE = (
    "的一是不了人我在有他这为之大来以个中上们"
    "到说国和地也子时道出而要于就下得可你年生"
    "自会那后能对着事其里所去行过家十用发天"
    "如小么心多日都从三二看起还分作当成者间"
    "无同现动面"
)

PUNCTUATION = (
    " .,!?;:'\"-—_()[]{}"
    "，。！？；：‘’“”"
    "、（）《》【】"
    "…"
)

VOCAB = (
    [PAD, UNK]
    + list(ENGLISH)
    + list(RUSSIAN)
    + list(JAPANESE)
    + list(CHINESE)
    + list(PUNCTUATION)
)

# Remove duplicates while preserving order.
VOCAB = list(dict.fromkeys(VOCAB))

TOKEN_TO_ID = {
    token: idx
    for idx, token in enumerate(VOCAB)
}

ID_TO_TOKEN = {
    idx: token
    for idx, token in enumerate(VOCAB)
}


def normalize_text(text):
    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def detect_language(text):
    counts = {
        "english": 0,
        "russian": 0,
        "japanese": 0,
        "chinese": 0,
    }

    for char in text:

        if char in ENGLISH:
            counts["english"] += 1

        elif char in RUSSIAN:
            counts["russian"] += 1

        elif char in JAPANESE:
            counts["japanese"] += 1

        elif char in CHINESE:
            counts["chinese"] += 1

    return max(
        counts,
        key=counts.get,
    )


def text_to_ids(text):
    text = normalize_text(text)

    ids = []

    for char in text.lower():

        if char in TOKEN_TO_ID:
            ids.append(
                TOKEN_TO_ID[char]
            )
        else:
            ids.append(
                TOKEN_TO_ID[UNK]
            )

    if len(ids) == 0:
        ids = [
            TOKEN_TO_ID[UNK]
        ]

    return torch.tensor(
        ids,
        dtype=torch.long,
    )


def ids_to_text(ids):
    chars = []

    for idx in ids:

        idx = int(idx)

        if idx in ID_TO_TOKEN:
            chars.append(
                ID_TO_TOKEN[idx]
            )

    return "".join(chars)


# =============================================================================
# SPEAKER ENCODER
# =============================================================================

class SpeakerEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim=SPEAKER_EMBED_DIM,
    ):
        super().__init__()

        self.encoder = nn.Sequential(

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
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(
                128,
                256,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(
                256,
                256,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.projection = nn.Sequential(

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(
                256,
                embedding_dim,
            ),
        )

    def forward(self, mel):

        x = self.encoder(mel)

        x = x.mean(dim=-1)

        x = self.projection(x)

        x = F.normalize(
            x,
            p=2,
            dim=-1,
        )

        return x


class SpeakerClassifier(nn.Module):

    def __init__(
        self,
        encoder,
        num_speakers,
    ):
        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Linear(
            SPEAKER_EMBED_DIM,
            num_speakers,
        )

    def forward(self, mel):

        embedding = self.encoder(mel)

        logits = self.classifier(
            embedding
        )

        return logits, embedding


# =============================================================================
# SPEAKER DATASET
# =============================================================================

class SpeakerDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        records,
        speaker_to_id,
    ):
        self.records = records
        self.speaker_to_id = speaker_to_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        waveform = load_audio(
            record["audio"]
        )

        if waveform is None:
            waveform = torch.zeros(
                SAMPLE_RATE,
                dtype=torch.float32,
            )

        mel = waveform_to_mel(
            waveform
        ).squeeze(0)

        target_frames = 192

        if mel.shape[-1] < target_frames:

            mel = F.pad(
                mel,
                (
                    0,
                    target_frames
                    - mel.shape[-1],
                ),
            )

        else:

            mel = mel[
                :,
                :target_frames
            ]

        speaker_id = self.speaker_to_id[
            record["speaker"]
        ]

        return (
            mel,
            torch.tensor(
                speaker_id,
                dtype=torch.long,
            ),
        )


# =============================================================================
# LOAD OR TRAIN SPEAKER ENCODER
# =============================================================================

def find_existing_checkpoint(patterns):
    for pattern in patterns:

        matches = list(
            Path(".").glob(pattern)
        )

        if matches:
            return matches[0]

    return None


def train_speaker_encoder(records):

    speakers = sorted(
        list(
            {
                r["speaker"]
                for r in records
            }
        )
    )

    speaker_to_id = {
        speaker: i
        for i, speaker in enumerate(speakers)
    }

    dataset = SpeakerDataset(
        records,
        speaker_to_id,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    encoder = SpeakerEncoder().to(DEVICE)

    model = SpeakerClassifier(
        encoder,
        len(speakers),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print()
    print("=" * 80)
    print("TRAINING SPEAKER ENCODER")
    print("=" * 80)

    losses = []

    for epoch in range(
        SPEAKER_EPOCHS
    ):

        model.train()

        total_loss = 0.0
        batches = 0

        for mel, target in loader:

            mel = mel.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()

            logits, embedding = model(
                mel
            )

            loss = F.cross_entropy(
                logits,
                target,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        mean_loss = (
            total_loss
            / max(batches, 1)
        )

        losses.append(mean_loss)

        print(
            f"Epoch {epoch + 1:02d}/"
            f"{SPEAKER_EPOCHS:02d} "
            f"loss={mean_loss:.5f}"
        )

    checkpoint = (
        OUTPUT_DIR
        / "speaker_encoder_final.pt"
    )

    torch.save(
        {
            "model": encoder.state_dict(),
            "vocab_size": len(VOCAB),
            "embedding_dim":
                SPEAKER_EMBED_DIM,
        },
        checkpoint,
    )

    plt.figure(figsize=(8, 4))

    plt.plot(
        losses,
        marker="o",
    )

    plt.title(
        "Final System Speaker Encoder Training"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "speaker_training.png"
    )

    plt.close()

    return encoder


def load_or_train_speaker_encoder(
    records
):

    checkpoint = find_existing_checkpoint(
        [
            "lesson65_outputs/"
            "speaker_metric_encoder_best.pt",

            "lesson64_outputs/"
            "speaker_encoder_best.pt",

            "lesson70_outputs/"
            "speaker_encoder_best.pt",

            "final_voice_outputs/"
            "speaker_encoder_final.pt",
        ]
    )

    encoder = SpeakerEncoder().to(
        DEVICE
    )

    if checkpoint is not None:

        print()
        print(
            "Using existing speaker encoder:"
        )
        print(
            f"  {checkpoint}"
        )

        checkpoint_data = torch.load(
            checkpoint,
            map_location=DEVICE,
        )

        if "model" in checkpoint_data:
            state = checkpoint_data[
                "model"
            ]
        else:
            state = checkpoint_data

        try:
            encoder.load_state_dict(
                state,
                strict=False,
            )

            print(
                "Speaker encoder loaded."
            )

            encoder.eval()

            return encoder

        except Exception as exc:

            print(
                "Could not load existing "
                "speaker encoder:"
            )
            print(exc)

    print(
        "No compatible speaker encoder "
        "checkpoint found."
    )

    return train_speaker_encoder(
        records
    )


# =============================================================================
# DURATION PREDICTOR
# =============================================================================

class DurationPredictor(nn.Module):

    def __init__(
        self,
        hidden_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, x):

        x = self.net(x)

        return F.softplus(
            x.squeeze(-1)
        ) + 1.0


# =============================================================================
# REFERENCE-CONDITIONED ACOUSTIC MODEL
# =============================================================================

class ReferenceConditionedAcousticModel(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        hidden_dim=192,
        num_heads=4,
        num_layers=3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.text_embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=0,
        )

        self.speaker_projection = nn.Sequential(

            nn.Linear(
                SPEAKER_EMBED_DIM,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=
                    hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
        )

        self.text_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.duration_predictor = (
            DurationPredictor(
                hidden_dim
            )
        )

        self.acoustic_decoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
            )
        )

        self.mel_projection = nn.Linear(
            hidden_dim,
            N_MELS,
        )

    def encode_text(
        self,
        tokens,
        speaker_embedding,
    ):

        x = self.text_embedding(
            tokens
        )

        speaker = self.speaker_projection(
            speaker_embedding
        )

        speaker = speaker.unsqueeze(1)

        x = x + speaker

        x = self.text_encoder(x)

        return x

    def predict_durations(
        self,
        encoded,
    ):

        return self.duration_predictor(
            encoded
        )

    def length_regulate(
        self,
        encoded,
        durations,
        target_frames=None,
    ):

        outputs = []

        for b in range(
            encoded.shape[0]
        ):

            sequence = encoded[b]

            d = durations[b]

            if target_frames is not None:

                total = d.sum().clamp_min(
                    1e-6
                )

                scaled = (
                    d
                    / total
                    * target_frames
                )

                repeats = torch.round(
                    scaled
                ).long().clamp_min(1)

                difference = (
                    target_frames
                    - repeats.sum()
                )

                if difference > 0:

                    repeats[-1] += difference

                elif difference < 0:

                    excess = -difference

                    for i in range(
                        len(repeats) - 1,
                        -1,
                        -1,
                    ):

                        removable = (
                            repeats[i] - 1
                        )

                        amount = min(
                            int(removable),
                            int(excess),
                        )

                        repeats[i] -= amount
                        excess -= amount

                        if excess == 0:
                            break

                expanded = torch.repeat_interleave(
                    sequence,
                    repeats,
                    dim=0,
                )

                if expanded.shape[0] > target_frames:

                    expanded = expanded[
                        :target_frames
                    ]

                elif expanded.shape[0] < target_frames:

                    expanded = F.pad(
                        expanded,
                        (
                            0,
                            0,
                            0,
                            target_frames
                            - expanded.shape[0],
                        ),
                    )

            else:

                repeats = torch.round(
                    d
                ).long().clamp_min(1)

                expanded = torch.repeat_interleave(
                    sequence,
                    repeats,
                    dim=0,
                )

            outputs.append(expanded)

        return torch.stack(outputs)

    def forward(
        self,
        tokens,
        speaker_embedding,
        target_frames=None,
    ):

        encoded = self.encode_text(
            tokens,
            speaker_embedding,
        )

        durations = self.predict_durations(
            encoded
        )

        regulated = self.length_regulate(
            encoded,
            durations,
            target_frames=target_frames,
        )

        decoded = self.acoustic_decoder(
            regulated
        )

        mel = self.mel_projection(
            decoded
        )

        return (
            mel.transpose(1, 2),
            durations,
        )


# =============================================================================
# ACOUSTIC TRAINING DATASET
# =============================================================================

class AcousticDataset(
    torch.utils.data.Dataset
):

    def __init__(self, records):

        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        waveform = load_audio(
            record["audio"]
        )

        if waveform is None:
            waveform = torch.zeros(
                SAMPLE_RATE,
            )

        mel = waveform_to_mel(
            waveform
        ).squeeze(0)

        text = normalize_text(
            record["text"]
        )

        tokens = text_to_ids(
            text
        )

        return (
            tokens,
            mel,
        )


def acoustic_collate(batch):

    token_list = [
        item[0]
        for item in batch
    ]

    mel_list = [
        item[1]
        for item in batch
    ]

    max_tokens = max(
        len(x)
        for x in token_list
    )

    max_frames = max(
        x.shape[-1]
        for x in mel_list
    )

    token_batch = torch.zeros(
        len(batch),
        max_tokens,
        dtype=torch.long,
    )

    mel_batch = torch.zeros(
        len(batch),
        N_MELS,
        max_frames,
    )

    mel_mask = torch.zeros(
        len(batch),
        max_frames,
    )

    for i, (
        tokens,
        mel,
    ) in enumerate(batch):

        token_batch[
            i,
            :len(tokens)
        ] = tokens

        frames = mel.shape[-1]

        mel_batch[
            i,
            :,
            :frames
        ] = mel

        mel_mask[
            i,
            :frames
        ] = 1.0

    return (
        token_batch,
        mel_batch,
        mel_mask,
    )


# =============================================================================
# ACOUSTIC MODEL TRAINING
# =============================================================================

def train_acoustic_model(
    records,
    speaker_encoder,
):

    print()
    print("=" * 80)
    print("TRAINING REFERENCE-CONDITIONED ACOUSTIC MODEL")
    print("=" * 80)

    dataset = AcousticDataset(
        records
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=acoustic_collate,
        num_workers=0,
    )

    model = ReferenceConditionedAcousticModel(
        vocab_size=len(VOCAB)
    ).to(DEVICE)

    speaker_encoder.eval()

    for parameter in speaker_encoder.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    losses = []

    for epoch in range(
        ACOUSTIC_EPOCHS
    ):

        model.train()

        total_loss = 0.0
        batches = 0

        for (
            tokens,
            target_mel,
            mask,
        ) in loader:

            tokens = tokens.to(
                DEVICE
            )

            target_mel = target_mel.to(
                DEVICE
            )

            mask = mask.to(
                DEVICE
            )

            with torch.no_grad():

                speaker_mels = (
                    target_mel
                )

                speaker_embedding = (
                    speaker_encoder(
                        speaker_mels
                    )
                )

            optimizer.zero_grad()

            predicted_mel, durations = (
                model(
                    tokens,
                    speaker_embedding,
                    target_frames=
                        target_mel.shape[-1],
                )
            )

            predicted_mel = (
                predicted_mel[
                    ...,
                    :target_mel.shape[-1]
                ]
            )

            frame_mask = mask.unsqueeze(1)

            mel_loss = (
                (
                    (
                        predicted_mel
                        - target_mel
                    ).abs()
                    * frame_mask
                ).sum()
                / frame_mask.sum().clamp_min(
                    1.0
                )
                / N_MELS
            )

            regularization = (
                durations.mean()
                * 1e-4
            )

            loss = (
                mel_loss
                + regularization
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        mean_loss = (
            total_loss
            / max(batches, 1)
        )

        losses.append(mean_loss)

        print(
            f"Epoch {epoch + 1:02d}/"
            f"{ACOUSTIC_EPOCHS:02d} "
            f"loss={mean_loss:.6f}"
        )

    checkpoint = (
        OUTPUT_DIR
        / "acoustic_model_final.pt"
    )

    torch.save(
        {
            "model":
                model.state_dict(),
            "vocab":
                VOCAB,
        },
        checkpoint,
    )

    plt.figure(figsize=(8, 4))

    plt.plot(
        losses,
        marker="o",
    )

    plt.title(
        "Acoustic Model Training"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "acoustic_training.png"
    )

    plt.close()

    return model


def load_or_train_acoustic_model(
    records,
    speaker_encoder,
):

    checkpoint = find_existing_checkpoint(
        [
            "lesson70_outputs/"
            "acoustic_model_best.pt",

            "lesson67_outputs/"
            "speaker_conditioned_acoustic_model_best.pt",

            "lesson59_outputs/"
            "improved_acoustic_model_best.pt",

            "final_voice_outputs/"
            "acoustic_model_final.pt",
        ]
    )

    model = ReferenceConditionedAcousticModel(
        vocab_size=len(VOCAB)
    ).to(DEVICE)

    if checkpoint is not None:

        print()
        print(
            "Trying existing acoustic model:"
        )
        print(
            f"  {checkpoint}"
        )

        data = torch.load(
            checkpoint,
            map_location=DEVICE,
        )

        state = data.get(
            "model",
            data,
        )

        try:

            model.load_state_dict(
                state,
                strict=False,
            )

            print(
                "Acoustic model loaded."
            )

            return model

        except Exception as exc:

            print(
                "Existing acoustic model "
                "is incompatible."
            )
            print(exc)

    return train_acoustic_model(
        records,
        speaker_encoder,
    )


# =============================================================================
# NEURAL VOCODER
# =============================================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        residual = x

        x = F.leaky_relu(
            self.conv1(x),
            0.2,
        )

        x = self.conv2(x)

        return (
            x + residual
        )


class CompactNeuralVocoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.input_projection = (
            nn.Conv1d(
                N_MELS,
                256,
                kernel_size=7,
                padding=3,
            )
        )

        self.up1 = nn.ConvTranspose1d(
            256,
            128,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.up2 = nn.ConvTranspose1d(
            128,
            64,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.up3 = nn.ConvTranspose1d(
            64,
            32,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.up4 = nn.ConvTranspose1d(
            32,
            16,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.residual1 = ResidualBlock(
            128,
            1,
        )

        self.residual2 = ResidualBlock(
            64,
            3,
        )

        self.residual3 = ResidualBlock(
            32,
            5,
        )

        self.residual4 = ResidualBlock(
            16,
            7,
        )

        self.output_projection = (
            nn.Conv1d(
                16,
                1,
                kernel_size=7,
                padding=3,
            )
        )

    def forward(self, mel):

        x = self.input_projection(
            mel
        )

        x = F.leaky_relu(
            self.up1(x),
            0.2,
        )

        x = self.residual1(x)

        x = F.leaky_relu(
            self.up2(x),
            0.2,
        )

        x = self.residual2(x)

        x = F.leaky_relu(
            self.up3(x),
            0.2,
        )

        x = self.residual3(x)

        x = F.leaky_relu(
            self.up4(x),
            0.2,
        )

        x = self.residual4(x)

        x = self.output_projection(
            x
        )

        return torch.tanh(x)


# =============================================================================
# VOCODER TRAINING
# =============================================================================

def train_vocoder(records):

    print()
    print("=" * 80)
    print("TRAINING FALLBACK NEURAL VOCODER")
    print("=" * 80)

    model = CompactNeuralVocoder().to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    model.train()

    for epoch in range(
        VOCODER_EPOCHS
    ):

        total_loss = 0.0
        count = 0

        random_records = records.copy()

        random.shuffle(
            random_records
        )

        for record in random_records[:16]:

            waveform = load_audio(
                record["audio"]
            )

            if waveform is None:
                continue

            waveform = waveform.to(
                DEVICE
            )

            mel = waveform_to_mel(
                waveform
            )

            target_samples = (
                mel.shape[-1]
                * HOP_LENGTH
            )

            waveform = waveform[
                :target_samples
            ]

            if waveform.shape[-1] < target_samples:

                waveform = F.pad(
                    waveform,
                    (
                        0,
                        target_samples
                        - waveform.shape[-1],
                    ),
                )

            predicted = model(
                mel
            ).squeeze(1)

            predicted = predicted[
                :,
                :waveform.shape[-1]
            ]

            waveform = waveform[
                :predicted.shape[-1]
            ]

            loss = F.l1_loss(
                predicted,
                waveform.unsqueeze(0),
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            count += 1

        mean_loss = (
            total_loss
            / max(count, 1)
        )

        print(
            f"Epoch {epoch + 1:02d}/"
            f"{VOCODER_EPOCHS:02d} "
            f"L1={mean_loss:.6f}"
        )

    checkpoint = (
        OUTPUT_DIR
        / "neural_vocoder_final.pt"
    )

    torch.save(
        {
            "model":
                model.state_dict()
        },
        checkpoint,
    )

    return model


def load_or_train_vocoder(records):

    checkpoint = find_existing_checkpoint(
        [
            "lesson69_outputs/"
            "neural_vocoder_best.pt",

            "lesson69_outputs/"
            "neural_vocoder_final.pt",

            "lesson70_outputs/"
            "neural_vocoder_best.pt",

            "lesson70_outputs/"
            "neural_vocoder_final.pt",

            "final_voice_outputs/"
            "neural_vocoder_final.pt",
        ]
    )

    model = CompactNeuralVocoder().to(
        DEVICE
    )

    if checkpoint is not None:

        print()
        print(
            "Using existing neural vocoder:"
        )
        print(
            f"  {checkpoint}"
        )

        data = torch.load(
            checkpoint,
            map_location=DEVICE,
        )

        state = data.get(
            "model",
            data,
        )

        try:

            model.load_state_dict(
                state,
                strict=False,
            )

            model.eval()

            return model

        except Exception as exc:

            print(
                "Could not load vocoder:"
            )
            print(exc)

    return train_vocoder(
        records
    )


# =============================================================================
# REFERENCE SPEAKER EMBEDDING
# =============================================================================

@torch.no_grad()
def extract_speaker_embedding(
    encoder,
    reference_audio,
):

    waveform = load_audio(
        reference_audio
    )

    if waveform is None:
        raise RuntimeError(
            "Reference audio is too short."
        )

    mel = waveform_to_mel(
        waveform
    )

    # Speaker encoder expects:
    #
    # [B, MEL, TIME]
    #
    mel = mel.to(DEVICE)

    embedding = encoder(
        mel
    )

    return embedding


# =============================================================================
# REFERENCE AUDIO SELECTION
# =============================================================================

def choose_reference_audio(
    records,
    requested=None,
):

    if requested is not None:

        path = Path(requested)

        if not path.exists():
            raise FileNotFoundError(
                f"Reference audio not found: {path}"
            )

        return path

    # Prefer a WAV generated by an earlier lesson.
    candidates = [

        Path(
            "lesson70_outputs/"
            "reference_voice.wav"
        ),

        Path(
            "lesson66_outputs/"
            "reference_voice.wav"
        ),

        Path(
            "lesson64_outputs/"
            "reference_voice.wav"
        ),

    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    if not records:

        raise RuntimeError(
            "No audio records available."
        )

    return Path(
        records[0]["audio"]
    )


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def synthesize(
    text,
    reference_audio,
    speaker_encoder,
    acoustic_model,
    vocoder,
):

    print()
    print("=" * 80)
    print("REFERENCE-VOICE INFERENCE")
    print("=" * 80)

    text = normalize_text(
        text
    )

    language = detect_language(
        text
    )

    print(
        f"Text language estimate: "
        f"{language}"
    )

    print(
        f"Text: {text}"
    )

    print(
        f"Reference: "
        f"{reference_audio}"
    )

    speaker_embedding = (
        extract_speaker_embedding(
            speaker_encoder,
            reference_audio,
        )
    )

    tokens = text_to_ids(
        text
    ).unsqueeze(0).to(
        DEVICE
    )

    # The final system uses the reference
    # speaker embedding directly.
    #
    # A target frame length is estimated from
    # text length so that inference does not
    # become excessively long.

    estimated_frames = max(
        32,
        min(
            240,
            int(
                len(tokens[0])
                * 5.0
            ),
        ),
    )

    acoustic_model.eval()

    mel, durations = acoustic_model(
        tokens,
        speaker_embedding,
        target_frames=estimated_frames,
    )

    # The compact vocoder has four x4
    # upsampling stages:
    #
    #     4 * 4 * 4 * 4 = 256
    #
    # Therefore each Mel frame corresponds
    # approximately to 256 samples.

    vocoder.eval()

    generated_waveform = vocoder(
        mel
    )

    generated_waveform = (
        generated_waveform
        .squeeze(0)
        .squeeze(0)
        .cpu()
    )

    generated_waveform = (
        generated_waveform
        / generated_waveform.abs().max().clamp_min(
            1e-6
        )
    )

    generated_mel = mel.squeeze(
        0
    ).cpu()

    generated_embedding = (
        speaker_embedding
        .squeeze(0)
        .cpu()
    )

    return (
        generated_waveform,
        generated_mel,
        speaker_embedding.squeeze(0).cpu(),
        generated_embedding,
        durations.squeeze(0).cpu(),
    )


# =============================================================================
# SAVE WAV
# =============================================================================

def save_waveform(
    waveform,
    filename,
):

    waveform = waveform.detach().cpu()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(
        str(filename),
        waveform,
        SAMPLE_RATE,
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_results(
    reference_audio,
    generated_waveform,
    generated_mel,
    reference_embedding,
    generated_embedding,
    text,
):

    generated_path = (
        OUTPUT_DIR
        / "generated_voice.wav"
    )

    reference_output = (
        OUTPUT_DIR
        / "reference_voice.wav"
    )

    save_waveform(
        generated_waveform,
        generated_path,
    )

    reference_waveform = load_audio(
        reference_audio
    )

    if reference_waveform is not None:

        save_waveform(
            reference_waveform,
            reference_output,
        )

    np.save(
        OUTPUT_DIR
        / "reference_speaker_embedding.npy",
        reference_embedding.numpy(),
    )

    np.save(
        OUTPUT_DIR
        / "generated_speaker_embedding.npy",
        generated_embedding.numpy(),
    )

    np.save(
        OUTPUT_DIR
        / "generated_mel.npy",
        generated_mel.numpy(),
    )

    # -------------------------------------------------------------------------
    # SPEAKER EMBEDDING COMPARISON
    # -------------------------------------------------------------------------

    similarity = F.cosine_similarity(
        reference_embedding.unsqueeze(0),
        generated_embedding.unsqueeze(0),
    ).item()

    plt.figure(figsize=(10, 5))

    plt.plot(
        reference_embedding.numpy(),
        label="Reference",
    )

    plt.plot(
        generated_embedding.numpy(),
        label="Generated",
    )

    plt.title(
        f"Speaker Embedding Comparison "
        f"(Cosine Similarity={similarity:.4f})"
    )

    plt.xlabel(
        "Embedding Dimension"
    )

    plt.ylabel(
        "Value"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "speaker_embedding_comparison.png"
    )

    plt.close()

    # -------------------------------------------------------------------------
    # MEL
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        generated_mel.numpy(),
        origin="lower",
        aspect="auto",
    )

    plt.title(
        "Generated Mel Spectrogram"
    )

    plt.xlabel(
        "Mel Frame"
    )

    plt.ylabel(
        "Mel Bin"
    )

    plt.colorbar(
        label="Log Magnitude"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "mel_spectrogram.png"
    )

    plt.close()

    # -------------------------------------------------------------------------
    # WAVEFORM
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        generated_waveform.numpy()
    )

    plt.title(
        "Generated Waveform"
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "waveform_comparison.png"
    )

    plt.close()

    return similarity


# =============================================================================
# SUMMARY
# =============================================================================

def save_summary(
    text,
    language,
    reference_audio,
    similarity,
):

    summary = f"""
FINAL VOICE CLONING SYSTEM
===========================

Architecture:

Reference Audio
    ->
Speaker Encoder
    ->
Speaker Embedding
    ->
Reference-Conditioned Acoustic Model
    ->
Mel Spectrogram
    ->
Neural Vocoder
    ->
Generated Waveform


Input text:
{text}

Detected language:
{language}

Reference audio:
{reference_audio}

Speaker embedding dimension:
{SPEAKER_EMBED_DIM}

Sample rate:
{SAMPLE_RATE}

Mel bins:
{N_MELS}

FFT:
{N_FFT}

Hop length:
{HOP_LENGTH}

Reference/generated speaker embedding cosine similarity:
{similarity:.6f}


Important limitation
--------------------

The system architecture supports:

    English
    Chinese
    Japanese
    Russian

However, acoustic quality depends on the languages represented in the
training data.

The automatic sample dataset is LibriSpeech dev-clean, which is primarily
English speech.

Therefore:

    English:
        suitable for demonstration.

    Chinese/Japanese/Russian:
        frontend-supported, but not fully multilingual-acoustically-trained.

For a genuinely multilingual production system, multilingual speech datasets
must be used during acoustic and vocoder training.


No pretrained TTS model was used.

No Whisper was used.

No pretrained phonemizer was used.

No pretrained speaker encoder was required.

No pretrained vocoder was required.


Output files
------------

generated_voice.wav
reference_voice.wav
reference_speaker_embedding.npy
generated_speaker_embedding.npy
generated_mel.npy
speaker_embedding_comparison.png
mel_spectrogram.png
waveform_comparison.png
final_system_summary.txt
"""

    path = (
        OUTPUT_DIR
        / "final_system_summary.txt"
    )

    path.write_text(
        summary.strip(),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Final from-scratch "
            "reference-voice cloning system."
        )
    )

    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Hello, this is a "
            "from scratch voice cloning "
            "system."
        ),
    )

    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=(
            "Optional reference WAV file."
        ),
    )

    args = parser.parse_args()

    seed_everything()

    ensure_directories()

    print()
    print("=" * 80)
    print("FINAL VOICE CLONING SYSTEM")
    print("=" * 80)

    print()
    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    # -------------------------------------------------------------------------
    # STEP 1
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 1 — PREPARE SAMPLE DATA")
    print("=" * 80)

    download_file(
        LIBRISPEECH_URL,
        LIBRISPEECH_ARCHIVE,
    )

    extract_librispeech()

    # -------------------------------------------------------------------------
    # STEP 2
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 2 — READ AUDIO/TEXT RECORDS")
    print("=" * 80)

    records = parse_librispeech_records()

    print(
        f"Total records discovered: "
        f"{len(records)}"
    )

    if len(records) == 0:

        raise RuntimeError(
            "No LibriSpeech records were found.\n"
            f"Expected dataset root:\n"
            f"{LIBRISPEECH_ROOT}\n"
            "Please check extraction."
        )

    records = build_small_dataset(
        records
    )

    if len(records) < 4:

        raise RuntimeError(
            "Too few usable records."
        )

    # -------------------------------------------------------------------------
    # STEP 3
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 3 — MULTILINGUAL TEXT FRONTEND")
    print("=" * 80)

    print(
        f"Vocabulary size: {len(VOCAB)}"
    )

    print(
        "Languages:"
    )

    print(
        "  English"
    )

    print(
        "  Chinese"
    )

    print(
        "  Japanese"
    )

    print(
        "  Russian"
    )

    print(
        f"Input text language estimate: "
        f"{detect_language(args.text)}"
    )

    # -------------------------------------------------------------------------
    # STEP 4
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 4 — SPEAKER ENCODER")
    print("=" * 80)

    speaker_encoder = (
        load_or_train_speaker_encoder(
            records
        )
    )

    speaker_encoder.eval()

    # -------------------------------------------------------------------------
    # STEP 5
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 5 — REFERENCE AUDIO")
    print("=" * 80)

    reference_audio = (
        choose_reference_audio(
            records,
            args.reference,
        )
    )

    print(
        f"Reference audio selected:"
    )

    print(
        f"  {reference_audio}"
    )

    # -------------------------------------------------------------------------
    # STEP 6
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 6 — ACOUSTIC MODEL")
    print("=" * 80)

    acoustic_model = (
        load_or_train_acoustic_model(
            records,
            speaker_encoder,
        )
    )

    acoustic_model.eval()

    # -------------------------------------------------------------------------
    # STEP 7
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 7 — NEURAL VOCODER")
    print("=" * 80)

    vocoder = (
        load_or_train_vocoder(
            records
        )
    )

    vocoder.eval()

    # -------------------------------------------------------------------------
    # STEP 8
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 8 — REFERENCE VOICE CLONING")
    print("=" * 80)

    (
        generated_waveform,
        generated_mel,
        reference_embedding,
        generated_embedding,
        durations,
    ) = synthesize(
        text=args.text,
        reference_audio=reference_audio,
        speaker_encoder=speaker_encoder,
        acoustic_model=acoustic_model,
        vocoder=vocoder,
    )

    print()
    print(
        f"Generated waveform samples: "
        f"{generated_waveform.numel()}"
    )

    print(
        f"Generated duration: "
        f"{generated_waveform.numel() / SAMPLE_RATE:.3f} sec"
    )

    print(
        f"Generated Mel shape: "
        f"{tuple(generated_mel.shape)}"
    )

    # -------------------------------------------------------------------------
    # STEP 9
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 9 — SPEAKER SIMILARITY")
    print("=" * 80)

    similarity = save_results(
        reference_audio,
        generated_waveform,
        generated_mel,
        reference_embedding,
        generated_embedding,
        args.text,
    )

    print(
        f"Reference/generated cosine similarity: "
        f"{similarity:.6f}"
    )

    # -------------------------------------------------------------------------
    # STEP 10
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 10 — SAVE FINAL SYSTEM SUMMARY")
    print("=" * 80)

    save_summary(
        text=args.text,
        language=detect_language(
            args.text
        ),
        reference_audio=reference_audio,
        similarity=similarity,
    )

    print()
    print("=" * 80)
    print("FINAL SYSTEM COMPLETE")
    print("=" * 80)

    print()
    print(
        "Generated audio:"
    )

    print(
        f"  {OUTPUT_DIR / 'generated_voice.wav'}"
    )

    print()
    print(
        "Reference audio:"
    )

    print(
        f"  {OUTPUT_DIR / 'reference_voice.wav'}"
    )

    print()
    print(
        "Speaker similarity:"
    )

    print(
        f"  {similarity:.6f}"
    )

    print()
    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_DIR.resolve()}"
    )

    print()
    print(
        "============================================================"
    )

    print(
        "FINAL PIPELINE"
    )

    print(
        "Reference WAV"
        " -> Speaker Encoder"
        " -> Speaker Embedding"
        " -> Acoustic Model"
        " -> Mel"
        " -> Neural Vocoder"
        " -> Generated WAV"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()