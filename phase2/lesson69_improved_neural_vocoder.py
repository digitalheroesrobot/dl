"""
===============================================================================
FILE: lesson69_improved_neural_vocoder.py

PHASE 2 — LESSON 69
IMPROVED NEURAL VOCODER — MEL TO WAVEFORM FROM SCRATCH
===============================================================================

COURSE POSITION
---------------
Lesson 53  : Length Regulator
Lesson 54  : Forced Alignment
Lesson 55  : Mel Dataset Builder
Lesson 56  : Duration Prediction
Lesson 57  : Improved Duration Prediction
Lesson 58  : Advanced Length Regulator
Lesson 59  : Improved Acoustic Model
Lesson 60  : Pitch and Energy Modeling
Lesson 61  : Neural Vocoder
Lesson 62  : End-to-End TTS Integration
Lesson 63  : Speaker-Conditioned TTS
Lesson 64  : Speaker Encoder from Real Audio
Lesson 65  : Speaker Metric Learning
Lesson 66  : Reference-Audio Voice Representation
Lesson 67  : Real Speaker-Conditioned Acoustic Model
Lesson 68  : End-to-End Reference Voice TTS
Lesson 69  : IMPROVED NEURAL VOCODER  <-- THIS LESSON

===============================================================================
MAIN GOAL
===============================================================================

Learn how a neural vocoder converts:

    MEL SPECTROGRAM
          |
          v
    NEURAL GENERATOR
          |
          v
    RAW WAVEFORM

Unlike Lesson 68, this lesson actually TRAINs the vocoder.

The model is NOT pretrained.

The vocoder learns the mapping:

    Mel(t, frequency)
          ->
    audio waveform(t)

===============================================================================
WHAT YOU LEARN
===============================================================================

STEP 1
------
Automatically prepare a real speech dataset.

If speaker_dataset/ already exists:

    speaker_dataset/
        speaker_000/
        speaker_001/
        ...

it will be reused.

Otherwise the script automatically downloads:

    LibriSpeech dev-clean

from OpenSLR and converts FLAC files to WAV.

-------------------------------------------------------------------------------

STEP 2
------
Convert waveform into Mel spectrograms.

Waveform:

    x[0], x[1], ..., x[N]

becomes:

    Mel[frequency, time]

This is the same general representation used by the acoustic model.

-------------------------------------------------------------------------------

STEP 3
------
Create aligned training examples:

    Mel segment
        |
        v
    Vocoder
        |
        v
    predicted waveform

Target:

    real waveform

-------------------------------------------------------------------------------

STEP 4
------
Learn waveform reconstruction using:

    Waveform L1 loss
    Waveform MSE loss

-------------------------------------------------------------------------------

STEP 5
------
Learn spectral reconstruction using:

    Multi-Resolution STFT Loss

The same waveform can sound very different even if its raw
sample values are not extremely close.

STFT loss therefore teaches the model to reproduce:

    harmonics
    spectral envelope
    formants
    high-frequency structure
    temporal structure

-------------------------------------------------------------------------------

STEP 6
------
Use transposed convolutions to increase temporal resolution.

Example:

    Mel
    80 x 32

        |
        | x4
        v

    128 x 128

        |
        | x4
        v

    96 x 512

        |
        | x4
        v

    64 x 2048

        |
        | x4
        v

    32 x 8192

        |
        v

    waveform

Therefore:

    32 Mel frames x 256 samples/frame
        =
    8192 waveform samples

-------------------------------------------------------------------------------

STEP 7
------
Use residual blocks.

Residual learning:

    y = x + F(x)

helps the generator learn progressively finer waveform details.

-------------------------------------------------------------------------------

STEP 8
------
Validate the vocoder on speech that was not used for training.

-------------------------------------------------------------------------------

STEP 9
------
Save:

    lesson69_outputs/
        neural_vocoder_best.pt
        neural_vocoder_final.pt
        vocoder_training.png
        target_waveform.png
        reconstructed_waveform.png
        target_reconstructed_mel.png
        target.wav
        reconstructed.wav

===============================================================================
IMPORTANT DESIGN DECISION
===============================================================================

This lesson uses:

    sample rate = 16000 Hz
    Mel bins    = 80
    FFT         = 1024
    hop length = 256
    window      = 1024

Therefore:

    256 waveform samples ~= 1 Mel frame

and:

    32 Mel frames ~= 8192 waveform samples

The generator therefore upsamples:

    4 x 4 x 4 x 4 = 256

===============================================================================
NO PRETRAINED MODELS
===============================================================================

This lesson does NOT use:

    - HiFi-GAN pretrained weights
    - WaveRNN pretrained weights
    - WaveGlow
    - BigVGAN
    - MelGAN pretrained weights
    - pretrained vocoders
    - pretrained acoustic models

Everything in the vocoder is initialized randomly.

===============================================================================
REAL-WORLD EXAMPLE
===============================================================================

A TTS system may produce:

    Text
      |
      v
    Acoustic Model
      |
      v
    Mel Spectrogram
      |
      v
    Neural Vocoder
      |
      v
    Speech waveform

The vocoder is the final component that turns the acoustic
representation into something that can actually be played
through a speaker.

===============================================================================
EXPECTED LEARNING OUTCOME
===============================================================================

After this lesson you should understand:

    1. Why a vocoder is necessary.
    2. How Mel spectrograms condition a waveform generator.
    3. Why upsampling is required.
    4. Why residual blocks are useful.
    5. Why waveform loss alone is insufficient.
    6. Why STFT losses are useful.
    7. How a neural vocoder is trained.
    8. How to evaluate reconstructed speech.
    9. How the vocoder connects the acoustic model to real audio.

===============================================================================
"""

import os
import sys
import math
import wave
import tarfile
import hashlib
import random
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

N_MELS = 80
F_MIN = 50.0
F_MAX = 7600.0

SEGMENT_FRAMES = 32
SEGMENT_SAMPLES = SEGMENT_FRAMES * HOP_LENGTH

BATCH_SIZE = 8

NUM_EPOCHS = 15

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-6

NUM_WORKERS = 4

MAX_FILES = 300

MAX_SECONDS_PER_FILE = 8.0

TRAIN_RATIO = 0.9

OUTPUT_DIR = Path("lesson69_outputs")

DATA_DIR = Path("lesson69_data")

SPEAKER_DATASET_DIR = Path("speaker_dataset")

LIBRISPEECH_URL = (
    "https://www.openslr.org/resources/12/dev-clean.tar.gz"
)

LIBRISPEECH_MD5 = "42e2234ba48799c1f50f24a7926300a1"

LIBRISPEECH_ARCHIVE = DATA_DIR / "dev-clean.tar.gz"

LIBRISPEECH_EXTRACTED = DATA_DIR / "LibriSpeech" / "dev-clean"


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def check_ffmpeg():
    """
    Check whether ffmpeg is available.

    LibriSpeech contains FLAC files.
    We convert them to WAV for this course.
    """

    if shutil.which("ffmpeg") is None:

        print()
        print("=" * 72)
        print("ERROR: ffmpeg was not found.")
        print("=" * 72)
        print()
        print("Install it with:")
        print()
        print("    sudo apt update")
        print("    sudo apt install ffmpeg")
        print()

        raise RuntimeError("ffmpeg is required.")


def md5_file(path):
    """
    Calculate MD5 checksum.
    """

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            md5.update(data)

    return md5.hexdigest()


def download_file(url, destination):
    """
    Download a file with progress display.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():

        print(f"Archive already exists: {destination}")

        return

    print()
    print("Downloading:")
    print(url)
    print()

    urllib.request.urlretrieve(
        url,
        destination,
        reporthook=download_progress,
    )

    print()


def download_progress(block_num, block_size, total_size):
    """
    Simple download progress callback.
    """

    if total_size <= 0:
        return

    downloaded = block_num * block_size

    percent = min(100.0, downloaded * 100.0 / total_size)

    print(
        f"\rDownload progress: {percent:6.2f}%",
        end="",
        flush=True,
    )


def verify_md5(path, expected):
    """
    Verify downloaded archive.
    """

    print("Checking MD5 checksum...")

    actual = md5_file(path)

    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")

    if actual != expected:

        raise RuntimeError(
            "MD5 checksum mismatch. "
            "Delete the archive and run again."
        )

    print("MD5 verification passed.")


def extract_librispeech():
    """
    Extract LibriSpeech dev-clean.
    """

    target = LIBRISPEECH_EXTRACTED

    if target.exists():

        wavs = list(target.rglob("*.flac"))

        if wavs:

            print(
                f"LibriSpeech already extracted: "
                f"{len(wavs)} FLAC files"
            )

            return

    print()
    print("Extracting LibriSpeech...")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(LIBRISPEECH_ARCHIVE, "r:gz") as tar:

        tar.extractall(DATA_DIR)

    if not target.exists():

        raise RuntimeError(
            "LibriSpeech extraction failed."
        )

    print("Extraction complete.")


def convert_flac_to_wav():
    """
    Convert a subset of LibriSpeech FLAC files to:

        speaker_dataset/speaker_xxx/*.wav

    This allows the rest of the course to reuse the same
    real-speaker dataset.
    """

    check_ffmpeg()

    existing_wavs = list(SPEAKER_DATASET_DIR.rglob("*.wav"))

    if len(existing_wavs) >= 20:

        print(
            f"Existing speaker_dataset detected: "
            f"{len(existing_wavs)} WAV files"
        )

        return

    flac_files = sorted(
        LIBRISPEECH_EXTRACTED.rglob("*.flac")
    )

    if not flac_files:

        raise RuntimeError(
            "No FLAC files were found."
        )

    print()
    print(f"Found {len(flac_files)} LibriSpeech FLAC files.")
    print("Preparing WAV dataset...")
    print()

    speaker_map = {}

    for flac_path in flac_files:

        speaker_id = flac_path.parent.parent.name

        speaker_map.setdefault(
            speaker_id,
            [],
        ).append(flac_path)

    selected = []

    for speaker_id in sorted(speaker_map.keys()):

        files = speaker_map[speaker_id]

        files = files[:25]

        for flac_path in files:

            selected.append(
                (speaker_id, flac_path)
            )

            if len(selected) >= MAX_FILES:
                break

        if len(selected) >= MAX_FILES:
            break

    for index, (speaker_id, flac_path) in enumerate(selected):

        speaker_dir = (
            SPEAKER_DATASET_DIR
            / f"speaker_{speaker_id}"
        )

        speaker_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_name = (
            f"{flac_path.stem}.wav"
        )

        wav_path = speaker_dir / output_name

        if wav_path.exists():
            continue

        command = [
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
        ]

        subprocess.run(
            command,
            check=True,
        )

        print(
            f"\rConverted "
            f"{index + 1}/{len(selected)}",
            end="",
            flush=True,
        )

    print()
    print("WAV dataset preparation complete.")


def prepare_dataset():
    """
    Ensure usable real speech exists.
    """

    wavs = list(
        SPEAKER_DATASET_DIR.rglob("*.wav")
    )

    if len(wavs) >= 20:

        print(
            f"Using existing dataset: "
            f"{len(wavs)} WAV files"
        )

        return

    print()
    print("=" * 72)
    print("NO SUFFICIENT LOCAL DATASET FOUND")
    print("=" * 72)
    print()
    print("Automatically preparing LibriSpeech dev-clean.")
    print()

    download_file(
        LIBRISPEECH_URL,
        LIBRISPEECH_ARCHIVE,
    )

    verify_md5(
        LIBRISPEECH_ARCHIVE,
        LIBRISPEECH_MD5,
    )

    extract_librispeech()

    convert_flac_to_wav()


# =============================================================================
# WAV LOADING
# =============================================================================

def load_wav(path):
    """
    Load a mono 16-bit PCM WAV.

    Returns:

        waveform: float32 numpy array in [-1, 1]
        sample_rate
    """

    with wave.open(str(path), "rb") as wf:

        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()

        raw = wf.readframes(frames)

    if sample_width == 2:

        audio = np.frombuffer(
            raw,
            dtype=np.int16,
        ).astype(np.float32)

        audio /= 32768.0

    elif sample_width == 1:

        audio = np.frombuffer(
            raw,
            dtype=np.uint8,
        ).astype(np.float32)

        audio = (audio - 128.0) / 128.0

    elif sample_width == 4:

        audio = np.frombuffer(
            raw,
            dtype=np.int32,
        ).astype(np.float32)

        audio /= 2147483648.0

    else:

        raise RuntimeError(
            f"Unsupported WAV sample width: "
            f"{sample_width}"
        )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels,
        ).mean(axis=1)

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    return audio.astype(np.float32), sample_rate


def save_wav(path, audio):
    """
    Save float waveform as 16-bit PCM WAV.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    pcm = (
        audio * 32767.0
    ).astype(np.int16)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(str(path), "wb") as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)

        wf.writeframes(
            pcm.tobytes()
        )


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_numpy(audio, source_rate, target_rate):
    """
    Simple linear interpolation resampler.

    The dataset is normally already 16 kHz, but this makes
    the loader robust.
    """

    if source_rate == target_rate:
        return audio

    old_length = len(audio)

    new_length = int(
        round(
            old_length
            * target_rate
            / source_rate
        )
    )

    old_x = np.linspace(
        0.0,
        1.0,
        old_length,
    )

    new_x = np.linspace(
        0.0,
        1.0,
        new_length,
    )

    return np.interp(
        new_x,
        old_x,
        audio,
    ).astype(np.float32)


# =============================================================================
# MEL FILTERBANK
# =============================================================================

def hz_to_mel(hz):
    return 2595.0 * np.log10(
        1.0 + hz / 700.0
    )


def mel_to_hz(mel):
    return 700.0 * (
        10.0 ** (mel / 2595.0)
        - 1.0
    )


def create_mel_filterbank():
    """
    Construct a triangular Mel filterbank from scratch.
    """

    low_mel = hz_to_mel(F_MIN)
    high_mel = hz_to_mel(F_MAX)

    mel_points = np.linspace(
        low_mel,
        high_mel,
        N_MELS + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    bin_points = np.floor(
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

    for m in range(1, N_MELS + 1):

        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        if center > left:

            for k in range(
                left,
                min(center, filterbank.shape[1]),
            ):

                filterbank[m - 1, k] = (
                    k - left
                ) / (
                    center - left
                )

        if right > center:

            for k in range(
                center,
                min(right, filterbank.shape[1]),
            ):

                filterbank[m - 1, k] = (
                    right - k
                ) / (
                    right - center
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
    """
    Convert waveform to log-Mel spectrogram.

    center=True is used so the Mel sequence is stable at
    both waveform boundaries.
    """

    x = torch.tensor(
        waveform,
        dtype=torch.float32,
    )

    if x.ndim == 1:
        x = x.unsqueeze(0)

    window = torch.hann_window(
        WIN_LENGTH,
        dtype=torch.float32,
    )

    stft = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True,
    )

    magnitude = torch.abs(stft)

    mel = torch.matmul(
        MEL_FILTERBANK,
        magnitude,
    )

    mel = torch.log(
        torch.clamp(
            mel,
            min=1e-5,
        )
    )

    return mel.squeeze(0)


# =============================================================================
# DATASET
# =============================================================================

class VocoderDataset(Dataset):
    """
    Real waveform -> Mel training dataset.

    Each sample contains:

        mel       [80, 32]
        waveform  [8192]
    """

    def __init__(
        self,
        files,
        training=True,
    ):

        self.files = files
        self.training = training

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        path = self.files[index]

        waveform, sample_rate = load_wav(path)

        waveform = resample_numpy(
            waveform,
            sample_rate,
            SAMPLE_RATE,
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        required = SEGMENT_SAMPLES

        if len(waveform) < required:

            waveform = np.pad(
                waveform,
                (
                    0,
                    required - len(waveform),
                ),
            )

        max_start = (
            len(waveform)
            - required
        )

        if self.training:

            if max_start > 0:

                start = random.randint(
                    0,
                    max_start,
                )

            else:

                start = 0

        else:

            start = max_start // 2

        segment = waveform[
            start:start + required
        ]

        # -------------------------------------------------------------
        # Extract Mel from the segment itself.
        #
        # This ensures that:
        #
        #     Mel frame count
        #
        # and
        #
        #     waveform segment
        #
        # belong to the same acoustic region.
        # -------------------------------------------------------------

        mel = waveform_to_mel(
            segment
        )

        # STFT with center=True normally produces approximately
        # SEGMENT_FRAMES + a few boundary frames.
        #
        # We choose exactly SEGMENT_FRAMES frames centered around
        # the middle of the Mel sequence.

        if mel.shape[1] < SEGMENT_FRAMES:

            mel = F.pad(
                mel,
                (
                    0,
                    SEGMENT_FRAMES
                    - mel.shape[1],
                ),
            )

        if mel.shape[1] > SEGMENT_FRAMES:

            mel_start = (
                mel.shape[1]
                - SEGMENT_FRAMES
            ) // 2

            mel = mel[
                :,
                mel_start:
                mel_start + SEGMENT_FRAMES,
            ]

        waveform_tensor = torch.tensor(
            segment,
            dtype=torch.float32,
        )

        return {
            "mel": mel,
            "waveform": waveform_tensor,
            "path": str(path),
        }


# =============================================================================
# DATASET SPLIT
# =============================================================================

def collect_wav_files():
    """
    Collect WAV files and split into train/validation sets.
    """

    files = sorted(
        SPEAKER_DATASET_DIR.rglob("*.wav")
    )

    if len(files) < 10:

        raise RuntimeError(
            "Not enough WAV files found."
        )

    # Limit dataset size so the lesson remains practical.
    if len(files) > MAX_FILES:

        files = files[:MAX_FILES]

    random.Random(SEED).shuffle(files)

    split = max(
        1,
        int(len(files) * TRAIN_RATIO),
    )

    train_files = files[:split]
    val_files = files[split:]

    if len(val_files) == 0:

        val_files = train_files[-1:]

    print()
    print("=" * 72)
    print("DATASET")
    print("=" * 72)
    print(f"Total files : {len(files)}")
    print(f"Train files : {len(train_files)}")
    print(f"Val files   : {len(val_files)}")
    print()

    return train_files, val_files


# =============================================================================
# RESIDUAL BLOCK
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Residual waveform refinement block.

    y = x + F(x)

    Multiple dilation values give the block access to
    different temporal scales.
    """

    def __init__(
        self,
        channels,
        dilation=1,
    ):

        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.activation = nn.LeakyReLU(
            0.2
        )

    def forward(self, x):

        residual = x

        y = self.conv1(x)

        y = self.activation(y)

        y = self.conv2(y)

        y = self.activation(y)

        return residual + y


# =============================================================================
# NEURAL VOCODER
# =============================================================================

class ImprovedNeuralVocoder(nn.Module):
    """
    Compact Mel-conditioned neural waveform generator.

    Input:

        [B, 80, 32]

    Output:

        [B, 1, 8192]

    Upsampling:

        4 x 4 x 4 x 4 = 256

    Therefore:

        32 Mel frames * 256 samples
        =
        8192 waveform samples
    """

    def __init__(
        self,
        n_mels=N_MELS,
    ):

        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv1d(
                n_mels,
                128,
                kernel_size=7,
                padding=3,
            ),
            nn.LeakyReLU(0.2),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(
                128,
                128,
                kernel_size=8,
                stride=4,
                padding=2,
            ),
            nn.LeakyReLU(0.2),
        )

        self.res1 = nn.ModuleList(
            [
                ResidualBlock(128, 1),
                ResidualBlock(128, 3),
                ResidualBlock(128, 5),
            ]
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(
                128,
                96,
                kernel_size=8,
                stride=4,
                padding=2,
            ),
            nn.LeakyReLU(0.2),
        )

        self.res2 = nn.ModuleList(
            [
                ResidualBlock(96, 1),
                ResidualBlock(96, 3),
                ResidualBlock(96, 5),
            ]
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(
                96,
                64,
                kernel_size=8,
                stride=4,
                padding=2,
            ),
            nn.LeakyReLU(0.2),
        )

        self.res3 = nn.ModuleList(
            [
                ResidualBlock(64, 1),
                ResidualBlock(64, 3),
                ResidualBlock(64, 5),
            ]
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose1d(
                64,
                32,
                kernel_size=8,
                stride=4,
                padding=2,
            ),
            nn.LeakyReLU(0.2),
        )

        self.res4 = nn.ModuleList(
            [
                ResidualBlock(32, 1),
                ResidualBlock(32, 3),
            ]
        )

        self.post = nn.Sequential(
            nn.Conv1d(
                32,
                32,
                kernel_size=7,
                padding=3,
            ),
            nn.LeakyReLU(0.2),

            nn.Conv1d(
                32,
                1,
                kernel_size=7,
                padding=3,
            ),

            nn.Tanh(),
        )

    def apply_residuals(
        self,
        x,
        blocks,
    ):

        for block in blocks:

            x = block(x)

        return x

    def forward(self, mel):

        x = self.pre(mel)

        x = self.up1(x)
        x = self.apply_residuals(
            x,
            self.res1,
        )

        x = self.up2(x)
        x = self.apply_residuals(
            x,
            self.res2,
        )

        x = self.up3(x)
        x = self.apply_residuals(
            x,
            self.res3,
        )

        x = self.up4(x)
        x = self.apply_residuals(
            x,
            self.res4,
        )

        x = self.post(x)

        return x


# =============================================================================
# STFT LOSS
# =============================================================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    Compare predicted and target waveforms at multiple resolutions.

    Resolutions:

        256
        512
        1024

    This gives the model pressure to reproduce both:

        local waveform structure

    and:

        larger spectral structure.
    """

    def __init__(self):

        super().__init__()

        self.configurations = [
            (256, 64, 256),
            (512, 128, 512),
            (1024, 256, 1024),
        ]

    def spectral_loss(
        self,
        predicted,
        target,
        n_fft,
        hop,
        win,
    ):

        window = torch.hann_window(
            win,
            device=predicted.device,
        )

        predicted_spec = torch.stft(
            predicted,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=True,
            return_complex=True,
        )

        target_spec = torch.stft(
            target,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=True,
            return_complex=True,
        )

        predicted_mag = torch.abs(
            predicted_spec
        )

        target_mag = torch.abs(
            target_spec
        )

        magnitude_loss = F.l1_loss(
            predicted_mag,
            target_mag,
        )

        log_predicted = torch.log1p(
            predicted_mag
        )

        log_target = torch.log1p(
            target_mag
        )

        log_loss = F.l1_loss(
            log_predicted,
            log_target,
        )

        return (
            magnitude_loss
            + log_loss
        )

    def forward(
        self,
        predicted,
        target,
    ):

        total = 0.0

        predicted = predicted.squeeze(1)
        target = target.squeeze(1)

        for (
            n_fft,
            hop,
            win,
        ) in self.configurations:

            total = total + self.spectral_loss(
                predicted,
                target,
                n_fft,
                hop,
                win,
            )

        return total / len(
            self.configurations
        )


# =============================================================================
# COMPLETE LOSS
# =============================================================================

class VocoderLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.stft_loss = (
            MultiResolutionSTFTLoss()
        )

    def forward(
        self,
        predicted,
        target,
    ):

        waveform_l1 = F.l1_loss(
            predicted,
            target,
        )

        waveform_mse = F.mse_loss(
            predicted,
            target,
        )

        stft = self.stft_loss(
            predicted,
            target,
        )

        total = (
            1.0 * waveform_l1
            + 0.5 * waveform_mse
            + 1.0 * stft
        )

        return (
            total,
            waveform_l1.detach(),
            waveform_mse.detach(),
            stft.detach(),
        )


# =============================================================================
# MODEL INFORMATION
# =============================================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
):

    model.train()

    running_total = 0.0
    running_l1 = 0.0
    running_mse = 0.0
    running_stft = 0.0

    for batch_index, batch in enumerate(loader):

        mel = batch["mel"].to(
            DEVICE,
            non_blocking=True,
        )

        target = batch["waveform"].to(
            DEVICE,
            non_blocking=True,
        )

        target = target.unsqueeze(1)

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):

            predicted = model(mel)

            if predicted.shape[-1] != target.shape[-1]:

                min_length = min(
                    predicted.shape[-1],
                    target.shape[-1],
                )

                predicted = predicted[
                    ...,
                    :min_length,
                ]

                target = target[
                    ...,
                    :min_length,
                ]

            (
                loss,
                l1,
                mse,
                stft,
            ) = criterion(
                predicted,
                target,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        scaler.step(optimizer)

        scaler.update()

        running_total += loss.item()
        running_l1 += l1.item()
        running_mse += mse.item()
        running_stft += stft.item()

        if (
            batch_index % 10 == 0
            or batch_index == len(loader) - 1
        ):

            print(
                f"\r"
                f"Batch "
                f"{batch_index + 1:4d}/"
                f"{len(loader):4d} | "
                f"Loss "
                f"{loss.item():.5f} | "
                f"L1 "
                f"{l1.item():.5f} | "
                f"STFT "
                f"{stft.item():.5f}",
                end="",
                flush=True,
            )

    print()

    n = len(loader)

    return (
        running_total / n,
        running_l1 / n,
        running_mse / n,
        running_stft / n,
    )


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    total = 0.0
    l1_total = 0.0
    mse_total = 0.0
    stft_total = 0.0

    first_prediction = None
    first_target = None
    first_mel = None
    first_path = None

    for batch_index, batch in enumerate(loader):

        mel = batch["mel"].to(
            DEVICE,
            non_blocking=True,
        )

        target = batch["waveform"].to(
            DEVICE,
            non_blocking=True,
        )

        target = target.unsqueeze(1)

        predicted = model(mel)

        if predicted.shape[-1] != target.shape[-1]:

            min_length = min(
                predicted.shape[-1],
                target.shape[-1],
            )

            predicted = predicted[
                ...,
                :min_length,
            ]

            target = target[
                ...,
                :min_length,
            ]

        (
            loss,
            l1,
            mse,
            stft,
        ) = criterion(
            predicted,
            target,
        )

        total += loss.item()
        l1_total += l1.item()
        mse_total += mse.item()
        stft_total += stft.item()

        if first_prediction is None:

            first_prediction = (
                predicted[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            first_target = (
                target[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            first_mel = (
                mel[0]
                .detach()
                .cpu()
                .numpy()
            )

            first_path = batch["path"][0]

    n = len(loader)

    return (
        total / n,
        l1_total / n,
        mse_total / n,
        stft_total / n,
        first_prediction,
        first_target,
        first_mel,
        first_path,
    )


# =============================================================================
# PLOTS
# =============================================================================

def plot_training(
    history,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history["train"],
        label="Train",
    )

    plt.plot(
        history["val"],
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Lesson 69 — Neural Vocoder Training"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "vocoder_training.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Training plot saved: {path}"
    )


def plot_waveform(
    waveform,
    title,
    path,
):

    time = (
        np.arange(len(waveform))
        / SAMPLE_RATE
    )

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        time,
        waveform,
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Waveform plot saved: {path}"
    )


def plot_mel(
    mel,
    path,
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.imshow(
        mel,
        origin="lower",
        aspect="auto",
    )

    plt.xlabel("Mel frame")
    plt.ylabel("Mel bin")

    plt.title(
        "Target Mel Spectrogram"
    )

    plt.colorbar(
        label="Log magnitude"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Mel plot saved: {path}"
    )


def plot_target_reconstructed(
    target,
    predicted,
    path,
):

    n = min(
        len(target),
        len(predicted),
    )

    target = target[:n]
    predicted = predicted[:n]

    time = (
        np.arange(n)
        / SAMPLE_RATE
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        time,
        target,
        label="Target",
        alpha=0.8,
    )

    plt.plot(
        time,
        predicted,
        label="Reconstructed",
        alpha=0.8,
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.title(
        "Target vs Reconstructed Waveform"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Comparison plot saved: {path}"
    )


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    loss,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "segment_frames": SEGMENT_FRAMES,
            "segment_samples": SEGMENT_SAMPLES,
        },
        path,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 72)
    print("PHASE 2 — LESSON 69")
    print("IMPROVED NEURAL VOCODER")
    print("=" * 72)

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Sample rate: {SAMPLE_RATE}"
    )

    print(
        f"Mel bins: {N_MELS}"
    )

    print(
        f"Mel frames/segment: "
        f"{SEGMENT_FRAMES}"
    )

    print(
        f"Waveform samples/segment: "
        f"{SEGMENT_SAMPLES}"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # STEP 1
    # -------------------------------------------------------------------------

    print("=" * 72)
    print("STEP 1 — PREPARE REAL SPEECH DATA")
    print("=" * 72)

    prepare_dataset()

    # -------------------------------------------------------------------------
    # STEP 2
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 2 — COLLECT WAV FILES")
    print("=" * 72)

    train_files, val_files = (
        collect_wav_files()
    )

    # -------------------------------------------------------------------------
    # STEP 3
    # -------------------------------------------------------------------------

    print("=" * 72)
    print("STEP 3 — BUILD VOCODER DATASETS")
    print("=" * 72)

    train_dataset = VocoderDataset(
        train_files,
        training=True,
    )

    val_dataset = VocoderDataset(
        val_files,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    # -------------------------------------------------------------------------
    # STEP 4
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 4 — BUILD NEURAL VOCODER")
    print("=" * 72)

    model = ImprovedNeuralVocoder(
        n_mels=N_MELS,
    ).to(DEVICE)

    parameters = count_parameters(
        model
    )

    print(
        f"Model parameters: "
        f"{parameters:,}"
    )

    # Test dimensions before training.

    with torch.no_grad():

        test_mel = torch.randn(
            2,
            N_MELS,
            SEGMENT_FRAMES,
            device=DEVICE,
        )

        test_output = model(
            test_mel
        )

    print(
        f"Input Mel shape: "
        f"{tuple(test_mel.shape)}"
    )

    print(
        f"Output waveform shape: "
        f"{tuple(test_output.shape)}"
    )

    expected_length = (
        SEGMENT_SAMPLES
    )

    if test_output.shape[-1] != expected_length:

        raise RuntimeError(
            "Unexpected vocoder output length: "
            f"{test_output.shape[-1]}, "
            f"expected {expected_length}"
        )

    print(
        "Vocoder dimension test passed."
    )

    # -------------------------------------------------------------------------
    # STEP 5
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 5 — CREATE LOSS AND OPTIMIZER")
    print("=" * 72)

    criterion = VocoderLoss().to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.8, 0.99),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )

    # -------------------------------------------------------------------------
    # STEP 6
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 6 — TRAIN VOCODER")
    print("=" * 72)

    print()
    print(
        "Training objective:"
    )

    print(
        "    waveform L1"
    )

    print(
        "  + waveform MSE"
    )

    print(
        "  + multi-resolution STFT"
    )

    print()

    history = {
        "train": [],
        "val": [],
    }

    best_val_loss = float("inf")

    best_path = (
        OUTPUT_DIR
        / "neural_vocoder_best.pt"
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print()
        print(
            "-" * 72
        )

        print(
            f"Epoch "
            f"{epoch}/{NUM_EPOCHS}"
        )

        print(
            "-" * 72
        )

        (
            train_loss,
            train_l1,
            train_mse,
            train_stft,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
        )

        (
            val_loss,
            val_l1,
            val_mse,
            val_stft,
            _,
            _,
            _,
            _,
        ) = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        history["train"].append(
            train_loss
        )

        history["val"].append(
            val_loss
        )

        print()
        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"  L1:   {train_l1:.6f}"
        )

        print(
            f"  MSE:  {train_mse:.6f}"
        )

        print(
            f"  STFT: {train_stft:.6f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.6f}"
        )

        print(
            f"  L1:   {val_l1:.6f}"
        )

        print(
            f"  MSE:  {val_mse:.6f}"
        )

        print(
            f"  STFT: {val_stft:.6f}"
        )

        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                val_loss,
            )

            print(
                f"New best model saved: "
                f"{best_path}"
            )

    # -------------------------------------------------------------------------
    # STEP 7
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 7 — SAVE FINAL MODEL")
    print("=" * 72)

    final_path = (
        OUTPUT_DIR
        / "neural_vocoder_final.pt"
    )

    save_checkpoint(
        final_path,
        model,
        optimizer,
        NUM_EPOCHS,
        history["val"][-1],
    )

    print(
        f"Final checkpoint: {final_path}"
    )

    # -------------------------------------------------------------------------
    # STEP 8
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 8 — LOAD BEST MODEL")
    print("=" * 72)

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best monitored validation loss: "
        f"{checkpoint['loss']:.6f}"
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    # -------------------------------------------------------------------------
    # STEP 9
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 9 — RECONSTRUCT VALIDATION SPEECH")
    print("=" * 72)

    (
        val_loss,
        val_l1,
        val_mse,
        val_stft,
        predicted,
        target,
        mel,
        source_path,
    ) = validate(
        model,
        val_loader,
        criterion,
    )

    print(
        f"Final validation loss: "
        f"{val_loss:.6f}"
    )

    print(
        f"Source WAV: {source_path}"
    )

    target_wav = (
        OUTPUT_DIR
        / "target.wav"
    )

    reconstructed_wav = (
        OUTPUT_DIR
        / "reconstructed.wav"
    )

    save_wav(
        target_wav,
        target,
    )

    save_wav(
        reconstructed_wav,
        predicted,
    )

    print(
        f"Target WAV: "
        f"{target_wav}"
    )

    print(
        f"Reconstructed WAV: "
        f"{reconstructed_wav}"
    )

    # -------------------------------------------------------------------------
    # STEP 10
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 10 — VISUALIZE RESULTS")
    print("=" * 72)

    plot_training(
        history
    )

    plot_waveform(
        target,
        "Target Speech Waveform",
        OUTPUT_DIR
        / "target_waveform.png",
    )

    plot_waveform(
        predicted,
        "Reconstructed Speech Waveform",
        OUTPUT_DIR
        / "reconstructed_waveform.png",
    )

    plot_target_reconstructed(
        target,
        predicted,
        OUTPUT_DIR
        / "target_vs_reconstructed.png",
    )

    plot_mel(
        mel,
        OUTPUT_DIR
        / "target_reconstructed_mel.png",
    )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("LESSON 69 COMPLETE")
    print("=" * 72)

    print()

    print(
        "You trained a neural vocoder from scratch."
    )

    print()

    print(
        "Pipeline:"
    )

    print(
        "    Real WAV"
    )

    print(
        "       ↓"
    )

    print(
        "    Mel Spectrogram"
    )

    print(
        "       ↓"
    )

    print(
        "    Neural Vocoder"
    )

    print(
        "       ↓"
    )

    print(
        "    Reconstructed WAV"
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        f"    {best_path}"
    )

    print()

    print(
        "Final checkpoint:"
    )

    print(
        f"    {final_path}"
    )

    print()

    print(
        "Generated audio:"
    )

    print(
        f"    {target_wav}"
    )

    print(
        f"    {reconstructed_wav}"
    )

    print()

    print(
        "Plots:"
    )

    print(
        f"    {OUTPUT_DIR / 'vocoder_training.png'}"
    )

    print(
        f"    {OUTPUT_DIR / 'target_waveform.png'}"
    )

    print(
        f"    {OUTPUT_DIR / 'reconstructed_waveform.png'}"
    )

    print(
        f"    {OUTPUT_DIR / 'target_vs_reconstructed.png'}"
    )

    print(
        f"    {OUTPUT_DIR / 'target_reconstructed_mel.png'}"
    )

    print()
    print(
        "Next: Lesson 70 — Full Voice Cloning Pipeline."
    )
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()