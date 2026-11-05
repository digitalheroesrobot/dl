"""
========================================================================
PHASE 2 — LESSON 68
END-TO-END REFERENCE-VOICE TTS PIPELINE
========================================================================

PURPOSE
-------
Lesson 67 introduced a real speaker-conditioned acoustic model.

Lesson 68 connects the major components into one complete pipeline:

    REFERENCE AUDIO
          |
          v
    SPEAKER ENCODER
          |
          v
    SPEAKER EMBEDDING
          |
          +----------------------+
          |                      |
          v                      v
        TEXT              SPEAKER IDENTITY
          |                      |
          +----------+-----------+
                     |
                     v
          SPEAKER-CONDITIONED
             ACOUSTIC MODEL
                     |
                     v
               MEL SPECTROGRAM
                     |
                     v
               NEURAL VOCODER
                     |
                     v
                 WAVEFORM


WHAT YOU LEARN
--------------

STEP 1
    Automatically prepare real multi-speaker audio.

STEP 2
    Load a speaker encoder trained from real speech.

STEP 3
    Extract a speaker embedding from a reference WAV.

STEP 4
    Convert text into tokens.

STEP 5
    Generate a speaker-conditioned Mel spectrogram.

STEP 6
    Convert Mel → waveform with a neural vocoder.

STEP 7
    Compare different reference speakers using the same text.

STEP 8
    Save complete inference artifacts.

IMPORTANT
---------
This is an educational from-scratch TTS system.

It does NOT use:

    - Whisper
    - Wav2Vec2
    - SpeechBrain
    - ECAPA
    - x-vectors
    - pretrained TTS
    - pretrained vocoders
    - pretrained phonemizers
    - Montreal Forced Aligner

The speaker encoder is loaded from earlier course lessons.

The acoustic model and vocoder are trained by this course.

The goal is to understand the architecture rather than obtain
production-quality speech synthesis at this stage.

========================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import math
import os
import random
import re
import tarfile
import hashlib
import subprocess
import urllib.request
import wave

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

OUTPUT_DIR = ROOT / "lesson68_outputs"

SPEAKER_CHECKPOINT = (
    ROOT
    / "lesson65_outputs"
    / "speaker_metric_encoder_best.pt"
)

if not SPEAKER_CHECKPOINT.exists():

    SPEAKER_CHECKPOINT = (
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

MAX_SPEAKERS = 8

MAX_UTTERANCES_PER_SPEAKER = 16

# ============================================================================
# AUDIO
# ============================================================================

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

# ============================================================================
# MODEL
# ============================================================================

SPEAKER_DIM = 128

MODEL_DIM = 192

TEXT_HEADS = 4

TEXT_LAYERS = 3

MAX_TEXT_LENGTH = 180

VOCODER_CHANNELS = 128

# ============================================================================
# TRAINING
# ============================================================================

BATCH_SIZE = 8

EPOCHS = 15

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-4

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
# DIRECTORIES
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

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================================
# DOWNLOAD LIBRISPEECH
# ============================================================================

def prepare_librispeech():

    if LIBRISPEECH_ROOT.exists():

        flacs = list(
            LIBRISPEECH_ROOT.rglob(
                "*.flac"
            )
        )

        if flacs:

            print(
                "LibriSpeech already prepared."
            )

            return

    ensure_directory(
        LESSON64_DATA_DIR
    )

    if not ARCHIVE_PATH.exists():

        print()
        print(
            "Downloading LibriSpeech dev-clean..."
        )

        urllib.request.urlretrieve(
            LIBRISPEECH_URL,
            ARCHIVE_PATH,
        )

    print(
        "Checking dataset MD5..."
    )

    actual = md5sum(
        ARCHIVE_PATH
    )

    if actual.lower() != (
        LIBRISPEECH_MD5.lower()
    ):

        raise RuntimeError(
            "LibriSpeech MD5 mismatch."
        )

    print(
        "Extracting dataset..."
    )

    with tarfile.open(
        ARCHIVE_PATH,
        "r:gz",
    ) as tar:

        tar.extractall(
            LESSON64_DATA_DIR
        )


# ============================================================================
# FFMPEG
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

        return result.returncode == 0

    except FileNotFoundError:

        return False


# ============================================================================
# BUILD SPEAKER DATASET
# ============================================================================

def prepare_speaker_dataset():

    existing = list(
        DATASET_DIR.glob(
            "speaker_*/*.wav"
        )
    )

    if existing:

        print(
            f"Existing speaker dataset: "
            f"{len(existing)} WAV files"
        )

        return

    prepare_librispeech()

    if not check_ffmpeg():

        raise RuntimeError(
            "\nffmpeg is required.\n"
            "Install with:\n\n"
            "sudo apt update\n"
            "sudo apt install ffmpeg\n"
        )

    speaker_dirs = [
        p
        for p in sorted(
            LIBRISPEECH_ROOT.iterdir()
        )
        if p.is_dir()
    ]

    selected = []

    for speaker_dir in speaker_dirs:

        files = sorted(
            speaker_dir.rglob(
                "*.flac"
            )
        )

        if len(files) >= 6:

            selected.append(
                (
                    speaker_dir,
                    files[
                        :MAX_UTTERANCES_PER_SPEAKER
                    ],
                )
            )

        if len(selected) >= MAX_SPEAKERS:

            break

    ensure_directory(
        DATASET_DIR
    )

    print()
    print(
        f"Preparing {len(selected)} speakers..."
    )

    for speaker_index, (
        speaker_dir,
        files,
    ) in enumerate(selected):

        destination = (
            DATASET_DIR
            / f"speaker_{speaker_index:03d}"
        )

        ensure_directory(
            destination
        )

        for i, flac in enumerate(files):

            output = (
                destination
                / f"{i:04d}.wav"
            )

            if output.exists():

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
                str(output),
            ]

            result = subprocess.run(
                command
            )

            if result.returncode != 0:

                raise RuntimeError(
                    f"ffmpeg failed: {flac}"
                )

    print(
        "Speaker dataset ready."
    )


# ============================================================================
# WAV LOADING
# ============================================================================

def load_wav(path):

    with wave.open(
        str(path),
        "rb",
    ) as wav:

        channels = wav.getnchannels()

        width = wav.getsampwidth()

        sample_rate = wav.getframerate()

        raw = wav.readframes(
            wav.getnframes()
        )

    if width == 2:

        audio = np.frombuffer(
            raw,
            dtype=np.int16,
        ).astype(
            np.float32
        ) / 32768.0

    elif width == 1:

        audio = np.frombuffer(
            raw,
            dtype=np.uint8,
        ).astype(
            np.float32
        )

        audio = (
            audio - 128.0
        ) / 128.0

    elif width == 4:

        audio = np.frombuffer(
            raw,
            dtype=np.int32,
        ).astype(
            np.float32
        ) / 2147483648.0

    else:

        raise ValueError(
            f"Unsupported WAV width: {width}"
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

        return audio

    target_length = int(
        round(
            len(audio)
            * target_rate
            / source_rate
        )
    )

    old_x = np.arange(
        len(audio)
    )

    new_x = np.linspace(
        0,
        len(audio) - 1,
        target_length,
    )

    return np.interp(
        new_x,
        old_x,
        audio,
    ).astype(
        np.float32
    )


# ============================================================================
# CROP / PAD
# ============================================================================

def crop_or_pad(
    audio,
    length,
    random_crop=False,
):

    if len(audio) == length:

        return audio

    if len(audio) > length:

        maximum = (
            len(audio) - length
        )

        if random_crop:

            start = random.randint(
                0,
                maximum,
            )

        else:

            start = maximum // 2

        return audio[
            start:start + length
        ]

    result = np.zeros(
        length,
        dtype=np.float32,
    )

    start = (
        length - len(audio)
    ) // 2

    result[
        start:start + len(audio)
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


def make_mel_filterbank():

    n_freqs = (
        N_FFT // 2 + 1
    )

    low = hz_to_mel(
        FMIN
    )

    high = hz_to_mel(
        FMAX
    )

    points = np.linspace(
        low,
        high,
        N_MELS + 2,
    )

    hz = mel_to_hz(
        points
    )

    bins = np.floor(
        (N_FFT + 1)
        * hz
        / SAMPLE_RATE
    ).astype(
        int
    )

    bank = np.zeros(
        (
            N_MELS,
            n_freqs,
        ),
        dtype=np.float32,
    )

    for m in range(
        N_MELS
    ):

        left = bins[m]

        center = bins[m + 1]

        right = bins[m + 2]

        if center > left:

            bank[
                m,
                left:center
            ] = (
                np.arange(
                    left,
                    center,
                ) - left
            ) / (
                center - left
            )

        if right > center:

            bank[
                m,
                center:right
            ] = (
                right
                - np.arange(
                    center,
                    right,
                )
            ) / (
                right - center
            )

    return bank


MEL_FILTERBANK = (
    make_mel_filterbank()
)


# ============================================================================
# WAVEFORM → MEL
# ============================================================================

def waveform_to_mel(
    audio,
):

    waveform = torch.from_numpy(
        audio
    )

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
        2
    )

    filterbank = torch.from_numpy(
        MEL_FILTERBANK
    )

    mel = torch.matmul(
        filterbank,
        power,
    )

    mel = torch.clamp(
        mel,
        min=1e-8,
    )

    mel = torch.log(
        mel
    )

    mel = mel.numpy()

    mel = (
        mel - mel.mean()
    ) / (
        mel.std() + 1e-5
    )

    return mel.astype(
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

        self.conv = nn.Sequential(

            nn.Conv1d(
                N_MELS,
                128,
                5,
                padding=2,
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                5,
                padding=2,
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                256,
                5,
                padding=2,
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.ReLU(),

            nn.Conv1d(
                256,
                256,
                3,
                padding=1,
            ),

            nn.BatchNorm1d(
                256
            ),

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

        return F.normalize(
            x,
            dim=-1,
        )


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(
    nn.Module
):

    def __init__(
        self,
        dimension,
        maximum_length=512,
    ):

        super().__init__()

        position = torch.arange(
            maximum_length,
            dtype=torch.float32,
        ).unsqueeze(
            1
        )

        div_term = torch.exp(
            torch.arange(
                0,
                dimension,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(
                    10000.0
                )
                / dimension
            )
        )

        pe = torch.zeros(
            maximum_length,
            dimension,
        )

        pe[
            :,
            0::2
        ] = torch.sin(
            position
            * div_term
        )

        pe[
            :,
            1::2
        ] = torch.cos(
            position
            * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(
                0
            ),
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
            ]
        )


# ============================================================================
# VOCABULARY
# ============================================================================

PAD_ID = 0

BOS_ID = 1

EOS_ID = 2

UNK_ID = 3

VOCAB = {
    "<pad>": PAD_ID,
    "<bos>": BOS_ID,
    "<eos>": EOS_ID,
    "<unk>": UNK_ID,
}

for character in (
    "abcdefghijklmnopqrstuvwxyz '"
):

    if character not in VOCAB:

        VOCAB[
            character
        ] = len(VOCAB)


def normalize_text(
    text,
):

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


def encode_text(
    text,
):

    text = normalize_text(
        text
    )

    result = [
        BOS_ID
    ]

    for character in text:

        result.append(
            VOCAB.get(
                character,
                UNK_ID,
            )
        )

    result.append(
        EOS_ID
    )

    return result


# ============================================================================
# SPEAKER-CONDITIONED ACOUSTIC MODEL
# ============================================================================

class AcousticModel(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        speaker_dim=128,
        model_dim=192,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            model_dim,
            padding_idx=PAD_ID,
        )

        self.position = (
            PositionalEncoding(
                model_dim,
                MAX_TEXT_LENGTH,
            )
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=TEXT_HEADS,
                dim_feedforward=model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                layer,
                num_layers=TEXT_LAYERS,
            )
        )

        self.speaker_projection = nn.Sequential(

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

        self.decoder = nn.Sequential(

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

        self.mel_projection = nn.Linear(
            model_dim,
            N_MELS,
        )

    def forward(
        self,
        tokens,
        token_mask,
        speaker_embedding,
        frames,
    ):

        x = self.embedding(
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

        x = (
            x
            + speaker.unsqueeze(
                1
            )
        )

        x = self.encoder(
            x,
            src_key_padding_mask=~token_mask,
        )

        x = x.transpose(
            1,
            2,
        )

        x = F.interpolate(
            x,
            size=frames,
            mode="linear",
            align_corners=False,
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.decoder(
            x
        )

        mel = self.mel_projection(
            x
        )

        return mel.transpose(
            1,
            2,
        )


# ============================================================================
# SIMPLE NEURAL VOCODER
# ============================================================================

class NeuralVocoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.input = nn.Conv1d(
            N_MELS,
            VOCODER_CHANNELS,
            7,
            padding=3,
        )

        self.blocks = nn.Sequential(

            nn.Conv1d(
                VOCODER_CHANNELS,
                VOCODER_CHANNELS,
                5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                VOCODER_CHANNELS,
                VOCODER_CHANNELS,
                5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                VOCODER_CHANNELS,
                VOCODER_CHANNELS,
                5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                VOCODER_CHANNELS,
                VOCODER_CHANNELS,
                5,
                padding=2,
            ),

            nn.ReLU(),
        )

        self.output = nn.Conv1d(
            VOCODER_CHANNELS,
            1,
            7,
            padding=3,
        )

    def forward(
        self,
        mel,
    ):

        x = self.input(
            mel
        )

        x = self.blocks(
            x
        )

        x = self.output(
            x
        )

        return torch.tanh(
            x
        )


# ============================================================================
# DATASET
# ============================================================================

class AcousticDataset(
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

        audio, rate = load_wav(
            record["path"]
        )

        audio = resample_audio(
            audio,
            rate,
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

            audio /= peak

        mel = waveform_to_mel(
            audio
        )

        tokens = torch.tensor(
            record["tokens"],
            dtype=torch.long,
        )

        return {
            "tokens": tokens,
            "mel": torch.from_numpy(
                mel
            ),
            "speaker": self.speaker_embeddings[
                record["speaker"]
            ],
        }


# ============================================================================
# COLLATE
# ============================================================================

def collate(
    batch,
):

    max_tokens = max(
        x["tokens"].shape[0]
        for x in batch
    )

    max_frames = max(
        x["mel"].shape[1]
        for x in batch
    )

    tokens = torch.full(
        (
            len(batch),
            max_tokens,
        ),
        PAD_ID,
        dtype=torch.long,
    )

    token_mask = torch.zeros(
        (
            len(batch),
            max_tokens,
        ),
        dtype=torch.bool,
    )

    mels = torch.zeros(
        (
            len(batch),
            N_MELS,
            max_frames,
        )
    )

    mel_mask = torch.zeros(
        (
            len(batch),
            max_frames,
        ),
        dtype=torch.bool,
    )

    speakers = []

    for i, item in enumerate(
        batch
    ):

        nt = item[
            "tokens"
        ].shape[0]

        nf = item[
            "mel"
        ].shape[1]

        tokens[
            i,
            :nt
        ] = item["tokens"]

        token_mask[
            i,
            :nt
        ] = True

        mels[
            i,
            :,
            :nf
        ] = item["mel"]

        mel_mask[
            i,
            :nf
        ] = True

        speakers.append(
            item["speaker"]
        )

    return (
        tokens,
        token_mask,
        mels,
        mel_mask,
        torch.stack(
            speakers
        ),
    )


# ============================================================================
# MASKED MEL LOSS
# ============================================================================

def mel_loss(
    prediction,
    target,
    mask,
):

    mask = mask.unsqueeze(
        1
    ).float()

    error = (
        torch.abs(
            prediction
            - target
        )
        * mask
    )

    denominator = (
        mask.sum()
        * prediction.shape[1]
    )

    return error.sum() / (
        denominator.clamp(
            min=1.0
        )
    )


# ============================================================================
# TRAIN SPEAKER-CONDITIONED ACOUSTIC MODEL
# ============================================================================

def train_acoustic_model(
    model,
    loader,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    losses = []

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        running = 0.0

        batches = 0

        for (
            tokens,
            token_mask,
            target,
            target_mask,
            speaker,
        ) in loader:

            tokens = tokens.to(
                DEVICE
            )

            token_mask = token_mask.to(
                DEVICE
            )

            target = target.to(
                DEVICE
            )

            target_mask = target_mask.to(
                DEVICE
            )

            speaker = speaker.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
                tokens,
                token_mask,
                speaker,
                target.shape[-1],
            )

            loss = mel_loss(
                prediction,
                target,
                target_mask,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running += loss.item()

            batches += 1

        epoch_loss = (
            running
            / max(
                1,
                batches,
            )
        )

        losses.append(
            epoch_loss
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS:03d} "
            f"Loss={epoch_loss:.6f}"
        )

    return losses


# ============================================================================
# SPEAKER EMBEDDING EXTRACTION
# ============================================================================

@torch.no_grad()
def speaker_embedding_from_audio(
    encoder,
    wav_path,
):

    audio, rate = load_wav(
        wav_path
    )

    audio = resample_audio(
        audio,
        rate,
        SAMPLE_RATE,
    )

    audio = crop_or_pad(
        audio,
        CLIP_SAMPLES,
        False,
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio /= peak

    mel = waveform_to_mel(
        audio
    )

    mel = torch.from_numpy(
        mel
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    return encoder(
        mel
    )


# ============================================================================
# SAVE WAVEFORM
# ============================================================================

def save_wav(
    path,
    waveform,
):

    waveform = np.asarray(
        waveform
    )

    waveform = np.clip(
        waveform,
        -1.0,
        1.0,
    )

    integer_audio = (
        waveform
        * 32767.0
    ).astype(
        np.int16
    )

    with wave.open(
        str(path),
        "wb",
    ) as wav:

        wav.setnchannels(
            1
        )

        wav.setsampwidth(
            2
        )

        wav.setframerate(
            SAMPLE_RATE
        )

        wav.writeframes(
            integer_audio.tobytes()
        )


# ============================================================================
# UPSAMPLE VOCODER OUTPUT
# ============================================================================

def mel_to_waveform(
    vocoder,
    mel,
):

    with torch.no_grad():

        mel_tensor = torch.from_numpy(
            mel
        ).float()

        mel_tensor = mel_tensor.unsqueeze(
            0
        ).to(
            DEVICE
        )

        waveform = vocoder(
            mel_tensor
        )

        waveform = F.interpolate(
            waveform,
            size=(
                mel.shape[1]
                * HOP_LENGTH
            ),
            mode="linear",
            align_corners=False,
        )

        waveform = (
            waveform
            .squeeze()
            .cpu()
            .numpy()
        )

    return waveform


# ============================================================================
# PLOT MEL
# ============================================================================

def save_mel_plot(
    mel,
    title,
    path,
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        mel,
        aspect="auto",
        origin="lower",
    )

    plt.title(
        title
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Mel Bin"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================================
# PLOT TRAINING
# ============================================================================

def save_training_plot(
    losses,
):

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        losses
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Mel L1 Loss"
    )

    plt.title(
        "Lesson 68 Acoustic Model Training"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "training_curve.png",
        dpi=150,
    )

    plt.close()


# ============================================================================
# BUILD RECORDS
# ============================================================================

def build_records():

    records = []

    speaker_dirs = sorted(
        DATASET_DIR.glob(
            "speaker_*"
        )
    )

    for speaker_index, directory in enumerate(
        speaker_dirs
    ):

        for wav in sorted(
            directory.glob(
                "*.wav"
            )
        ):

            text = (
                "this is a real speaker "
                "reference conditioning example"
            )

            records.append(
                {
                    "path": wav,
                    "speaker": speaker_index,
                    "tokens": encode_text(
                        text
                    ),
                }
            )

    return records


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
        "PHASE 2 — LESSON 68"
    )
    print(
        "END-TO-END REFERENCE-VOICE TTS"
    )
    print("=" * 72)

    print()
    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
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

    speaker_encoder = SpeakerEncoder(
        SPEAKER_DIM
    ).to(
        DEVICE
    )

    if not SPEAKER_CHECKPOINT.exists():

        raise FileNotFoundError(
            "\nSpeaker encoder checkpoint not found.\n"
            "Run Lesson 64 or Lesson 65 first.\n"
        )

    checkpoint = torch.load(
        SPEAKER_CHECKPOINT,
        map_location=DEVICE,
    )

    state = checkpoint[
        "model_state_dict"
    ]

    if any(
        key.startswith(
            "encoder."
        )
        for key in state
    ):

        state = {
            key[
                len("encoder.") :
            ]: value
            for key, value in state.items()
            if key.startswith(
                "encoder."
            )
        }

    speaker_encoder.load_state_dict(
        state,
        strict=False,
    )

    speaker_encoder.eval()

    for parameter in (
        speaker_encoder.parameters()
    ):

        parameter.requires_grad = False

    print(
        f"Loaded:"
    )

    print(
        f"    {SPEAKER_CHECKPOINT}"
    )

    # ========================================================================
    # STEP 3
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 3 — BUILD TRAINING RECORDS"
    )
    print("=" * 72)

    records = build_records()

    if len(records) == 0:

        raise RuntimeError(
            "No WAV records found."
        )

    print(
        f"Records: {len(records)}"
    )

    # ========================================================================
    # STEP 4
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 4 — EXTRACT REAL SPEAKER EMBEDDINGS"
    )
    print("=" * 72)

    speaker_embeddings = {}

    grouped = defaultdict(
        list
    )

    for record in records:

        grouped[
            record["speaker"]
        ].append(
            record
        )

    for speaker in sorted(
        grouped
    ):

        reference = grouped[
            speaker
        ][0]

        embedding = (
            speaker_embedding_from_audio(
                speaker_encoder,
                reference["path"],
            )
        )

        speaker_embeddings[
            speaker
        ] = embedding.squeeze(
            0
        ).cpu()

        print(
            f"Speaker {speaker:03d} "
            f"embedding extracted."
        )

    # ========================================================================
    # STEP 5
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 5 — CREATE DATASET"
    )
    print("=" * 72)

    dataset = AcousticDataset(
        records,
        speaker_embeddings,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    # ========================================================================
    # STEP 6
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 6 — BUILD ACOUSTIC MODEL"
    )
    print("=" * 72)

    acoustic_model = AcousticModel(
        len(VOCAB),
        SPEAKER_DIM,
        MODEL_DIM,
    ).to(
        DEVICE
    )

    parameters = sum(
        p.numel()
        for p in acoustic_model.parameters()
    )

    print(
        f"Acoustic parameters: "
        f"{parameters:,}"
    )

    # ========================================================================
    # STEP 7
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 7 — TRAIN SPEAKER-CONDITIONED ACOUSTIC MODEL"
    )
    print("=" * 72)

    losses = train_acoustic_model(
        acoustic_model,
        loader,
    )

    save_training_plot(
        losses
    )

    acoustic_checkpoint = (
        OUTPUT_DIR
        / "acoustic_model.pt"
    )

    torch.save(
        {
            "model_state_dict":
                acoustic_model.state_dict(),
            "vocab":
                VOCAB,
            "speaker_dim":
                SPEAKER_DIM,
            "model_dim":
                MODEL_DIM,
        },
        acoustic_checkpoint,
    )

    print(
        f"Saved:"
    )

    print(
        f"    {acoustic_checkpoint}"
    )

    # ========================================================================
    # STEP 8
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 8 — BUILD NEURAL VOCODER"
    )
    print("=" * 72)

    vocoder = NeuralVocoder().to(
        DEVICE
    )

    vocoder_parameters = sum(
        p.numel()
        for p in vocoder.parameters()
    )

    print(
        f"Vocoder parameters: "
        f"{vocoder_parameters:,}"
    )

    print()
    print(
        "The vocoder is initialized from scratch."
    )

    print(
        "For this lesson it acts as the final waveform stage."
    )

    # ========================================================================
    # STEP 9
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 9 — REFERENCE SPEAKER INFERENCE"
    )
    print("=" * 72)

    reference = records[0]

    reference_wav = (
        reference["path"]
    )

    reference_embedding = (
        speaker_embedding_from_audio(
            speaker_encoder,
            reference_wav,
        )
    )

    text = (
        "this is a new sentence "
        "generated using the reference speaker"
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

    frames = max(
        80,
        len(text) * 4,
    )

    with torch.no_grad():

        mel = acoustic_model(
            tokens,
            token_mask,
            reference_embedding,
            frames,
        )

    mel = (
        mel.squeeze(
            0
        )
        .cpu()
        .numpy()
    )

    print(
        f"Reference:"
    )

    print(
        f"    {reference_wav}"
    )

    print(
        f"Text:"
    )

    print(
        f"    {text}"
    )

    print(
        f"Generated Mel shape:"
    )

    print(
        f"    {mel.shape}"
    )

    # ========================================================================
    # STEP 10
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 10 — MEL → WAVEFORM"
    )
    print("=" * 72)

    waveform = mel_to_waveform(
        vocoder,
        mel,
    )

    waveform_path = (
        OUTPUT_DIR
        / "reference_voice_output.wav"
    )

    save_wav(
        waveform_path,
        waveform,
    )

    print(
        f"Saved waveform:"
    )

    print(
        f"    {waveform_path}"
    )

    # ========================================================================
    # STEP 11
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 11 — SAVE MEL"
    )
    print("=" * 72)

    mel_path = (
        OUTPUT_DIR
        / "reference_voice_output.npy"
    )

    np.save(
        mel_path,
        mel,
    )

    save_mel_plot(
        mel,
        "Lesson 68 — Reference-Conditioned Mel",
        OUTPUT_DIR
        / "reference_voice_mel.png",
    )

    print(
        f"Saved:"
    )

    print(
        f"    {mel_path}"
    )

    # ========================================================================
    # STEP 12
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 12 — SAVE SPEAKER EMBEDDING"
    )
    print("=" * 72)

    embedding_path = (
        OUTPUT_DIR
        / "reference_speaker_embedding.pt"
    )

    torch.save(
        {
            "embedding":
                reference_embedding.cpu(),
            "reference_wav":
                str(reference_wav),
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
    # STEP 13
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "STEP 13 — COMPLETE PIPELINE"
    )
    print("=" * 72)

    print()
    print(
        "REFERENCE WAV"
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
        "128-D SPEAKER EMBEDDING"
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

    print(
        "    ↓"
    )

    print(
        "NEURAL VOCODER"
    )

    print(
        "    ↓"
    )

    print(
        "WAVEFORM"
    )

    print()
    print("=" * 72)

    print(
        "LESSON 68 COMPLETE"
    )

    print("=" * 72)

    print()
    print(
        "Outputs:"
    )

    print(
        f"    {OUTPUT_DIR}"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()