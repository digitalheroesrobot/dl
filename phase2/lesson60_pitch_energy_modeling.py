"""
========================================================================
PHASE 2 — LESSON 60
PITCH AND ENERGY MODELING
========================================================================

PURPOSE
-------
Lesson 59 improved the acoustic model with:

    phoneme encoder
        ↓
    duration predictor
        ↓
    length regulator
        ↓
    Transformer acoustic decoder
        ↓
    coarse Mel
        ↓
    convolutional post-net
        ↓
    final Mel

Lesson 60 introduces two major FastSpeech 2 concepts:

    1. PITCH MODELING
    2. ENERGY MODELING

Speech is not determined only by phoneme identity and duration.

Two speakers can pronounce the same phoneme sequence with different:

    - pitch
    - loudness / energy
    - intonation
    - speaking style

We therefore explicitly model pitch and energy.

========================================================================
WHAT YOU LEARN
========================================================================

STEP 1 — FRAME-LEVEL PITCH
--------------------------
Estimate fundamental frequency F0 directly from the waveform.

For a periodic voiced signal:

        x(t) ≈ periodic waveform

The fundamental frequency is approximately:

        F0 = 1 / T0

where T0 is the fundamental period.

This lesson uses a simple autocorrelation-based F0 estimator.

NO PRETRAINED PITCH MODEL IS USED.

This is educational rather than production-grade pitch extraction.

-----------------------------------------------------------------------

STEP 2 — FRAME-LEVEL ENERGY
---------------------------
Calculate short-time RMS energy:

                 sqrt(mean(x^2))

for each audio frame.

Energy gives the acoustic model information about local amplitude /
loudness variation.

-----------------------------------------------------------------------

STEP 3 — PHONEME-LEVEL PITCH AND ENERGY
---------------------------------------
The acoustic model operates on phoneme representations while pitch and
energy are initially measured per audio frame.

We therefore aggregate frame-level values into phoneme-level values
using the alignment durations.

For phoneme i:

    pitch_i = mean(frame_pitch_i)

    energy_i = mean(frame_energy_i)

Unvoiced frames are excluded from the pitch average.

-----------------------------------------------------------------------

STEP 4 — VARIANCE EMBEDDING
---------------------------
Instead of simply predicting pitch and energy and ignoring them, we
convert them into embeddings.

Conceptually:

    pitch value
        ↓
    pitch embedding

    energy value
        ↓
    energy embedding

Then:

    encoder representation
          +
    pitch embedding
          +
    energy embedding

becomes the representation sent to the length regulator.

-----------------------------------------------------------------------

STEP 5 — PITCH PREDICTOR
------------------------
The network predicts:

    normalized log pitch

for every phoneme.

Pitch is represented in log space because frequency perception is
approximately logarithmic.

-----------------------------------------------------------------------

STEP 6 — ENERGY PREDICTOR
-------------------------
The network predicts normalized log energy for every phoneme.

-----------------------------------------------------------------------

STEP 7 — TEACHER-FORCED PROSODY
-------------------------------
During training:

    ground-truth duration
    ground-truth pitch
    ground-truth energy

are used for the acoustic reconstruction.

This allows us to isolate the acoustic modeling problem.

-----------------------------------------------------------------------

STEP 8 — PREDICTED PROSODY
--------------------------
At inference:

    predicted duration
    predicted pitch
    predicted energy

are used.

Therefore the complete path becomes:

    phonemes
       ↓
    encoder
       ↓
    duration predictor
       ↓
    pitch predictor
       ↓
    energy predictor
       ↓
    variance embeddings
       ↓
    length regulator
       ↓
    acoustic decoder
       ↓
    Mel spectrogram

-----------------------------------------------------------------------

STEP 9 — VISUALIZATION
----------------------
The script saves:

    pitch_energy_training.png
    target_pitch.png
    predicted_pitch.png
    target_energy.png
    predicted_energy.png
    pitch_error.png
    energy_error.png
    target_mel.png
    predicted_mel.png

-----------------------------------------------------------------------

IMPORTANT LIMITATION
--------------------
The F0 extractor here is intentionally simple and educational.

It is NOT intended to replace:

    WORLD
    CREPE
    PYIN
    Praat
    production-grade pitch trackers

The goal is to understand how pitch becomes a training signal and how
that signal is integrated into an acoustic TTS model.

========================================================================
INPUTS
========================================================================

Previous lesson outputs:

    lesson54_outputs/
        *_alignment.json

    lesson55_outputs/
        features/
            *.npz

The Mel NPZ should contain:

    log_mel
or
    normalized_log_mel

The alignment should contain:

    aligned_phonemes:
        phoneme
        start_frame
        end_frame
        duration_frames

For pitch extraction we additionally need the original waveform.

The script searches for WAV files in:

    lesson55_outputs/
    lesson55_outputs/wav/
    lesson55_outputs/audio/
    wav/
    audio/
    dataset/
    data/

If the alignment JSON contains an explicit audio path, that path is
also checked.

If a waveform cannot be found, the script can optionally generate a
synthetic educational waveform so the complete pipeline remains
runnable.

For real TTS training, use the actual recording.

========================================================================
"""

import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

OUTPUT_DIR = Path("lesson60_outputs")

ALIGNMENT_DIR = Path("lesson54_outputs")
MEL_FEATURE_DIR = Path("lesson55_outputs/features")

# Possible waveform locations.
AUDIO_SEARCH_DIRS = [
    Path("lesson55_outputs"),
    Path("lesson55_outputs/wav"),
    Path("lesson55_outputs/audio"),
    Path("wav"),
    Path("audio"),
    Path("dataset"),
    Path("data"),
]

# ---------------------------------------------------------------------------
# Acoustic model
# ---------------------------------------------------------------------------

EMBED_DIM = 128

ATTENTION_HEADS = 4

ENCODER_LAYERS = 3
DECODER_LAYERS = 3

FF_DIM = 512

DROPOUT = 0.10

POSTNET_CHANNELS = 128
POSTNET_LAYERS = 5
POSTNET_KERNEL_SIZE = 5

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

BATCH_SIZE = 1

EPOCHS = 300

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 1.0

PATIENCE = 40

# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------

DURATION_LOSS_WEIGHT = 1.0

PITCH_LOSS_WEIGHT = 1.0

ENERGY_LOSS_WEIGHT = 1.0

MEL_L1_WEIGHT = 1.0

MEL_MSE_WEIGHT = 0.5

COARSE_MEL_WEIGHT = 0.5

FINAL_MEL_WEIGHT = 1.0

RESIDUAL_WEIGHT = 0.05

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

MAX_SEQUENCE_LENGTH = 4096

MAX_DURATION = 100

MIN_DURATION = 1

MAX_SYNTHESIS_FRAMES = 2000

PAD_ID = 0

# ---------------------------------------------------------------------------
# Pitch extraction
# ---------------------------------------------------------------------------

MIN_F0 = 60.0

MAX_F0 = 500.0

# Analysis frame parameters.
# These defaults should normally agree with the Mel extraction settings.

DEFAULT_HOP_LENGTH = 256

DEFAULT_WIN_LENGTH = 1024

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def ensure_output_dir():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_int(value, default=None):

    try:
        return int(round(float(value)))
    except Exception:
        return default


def safe_float(value, default=None):

    try:

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:

        return default


# ============================================================================
# ALIGNMENT FILES
# ============================================================================

def find_alignment_files():

    if not ALIGNMENT_DIR.exists():

        raise FileNotFoundError(
            f"Alignment directory does not exist:\n"
            f"{ALIGNMENT_DIR.resolve()}\n"
            f"Run Lesson 54 first."
        )

    files = sorted(
        ALIGNMENT_DIR.glob("*_alignment.json")
    )

    if not files:

        files = sorted(
            ALIGNMENT_DIR.glob("*.json")
        )

    if not files:

        raise RuntimeError(
            f"No alignment JSON files found in "
            f"{ALIGNMENT_DIR}"
        )

    return files


# ============================================================================
# MEL FILE
# ============================================================================

def find_mel_file(stem):

    candidates = [

        MEL_FEATURE_DIR / f"{stem}.npz",

        MEL_FEATURE_DIR / f"{stem}_features.npz",

    ]

    for candidate in candidates:

        if candidate.exists():

            return candidate

    matches = list(
        MEL_FEATURE_DIR.glob(
            f"{stem}*.npz"
        )
    )

    if matches:

        return matches[0]

    return None


# ============================================================================
# AUDIO FILE SEARCH
# ============================================================================

def find_audio_file(
    stem,
    alignment_data=None,
):

    # ------------------------------------------------------------------------
    # First inspect metadata from alignment JSON.
    # ------------------------------------------------------------------------

    if isinstance(
        alignment_data,
        dict,
    ):

        possible_keys = [
            "audio_path",
            "wav_path",
            "audio_file",
            "wav_file",
            "audio",
            "wav",
        ]

        for key in possible_keys:

            value = alignment_data.get(key)

            if value is None:
                continue

            path = Path(str(value))

            if path.exists():
                return path

    # ------------------------------------------------------------------------
    # Search known directories.
    # ------------------------------------------------------------------------

    for directory in AUDIO_SEARCH_DIRS:

        if not directory.exists():
            continue

        candidates = [

            directory / f"{stem}.wav",

            directory / f"{stem}.WAV",

            directory / f"{stem}_audio.wav",

        ]

        for candidate in candidates:

            if candidate.exists():
                return candidate

    # ------------------------------------------------------------------------
    # Recursive fallback.
    # ------------------------------------------------------------------------

    for directory in AUDIO_SEARCH_DIRS:

        if not directory.exists():
            continue

        matches = list(
            directory.rglob(
                f"{stem}.wav"
            )
        )

        if matches:
            return matches[0]

    return None


# ============================================================================
# WAV LOADING
# ============================================================================

def load_wav(path):

    """
    Load PCM WAV without requiring torchaudio/librosa.

    Uses Python's standard wave module.

    Returns:

        waveform : float32 numpy array in [-1, 1]
        sample_rate : integer
    """

    import wave

    with wave.open(
        str(path),
        "rb",
    ) as wav:

        sample_rate = wav.getframerate()

        channels = wav.getnchannels()

        sample_width = wav.getsampwidth()

        n_frames = wav.getnframes()

        raw = wav.readframes(
            n_frames
        )

    if sample_width == 1:

        audio = np.frombuffer(
            raw,
            dtype=np.uint8,
        ).astype(np.float32)

        audio = (
            audio - 128.0
        ) / 128.0

    elif sample_width == 2:

        audio = np.frombuffer(
            raw,
            dtype=np.int16,
        ).astype(np.float32)

        audio /= 32768.0

    elif sample_width == 3:

        raw_bytes = np.frombuffer(
            raw,
            dtype=np.uint8,
        )

        raw_bytes = raw_bytes.reshape(
            -1,
            3,
        )

        audio = (
            raw_bytes[:, 0].astype(np.int32)
            |
            (raw_bytes[:, 1].astype(np.int32) << 8)
            |
            (raw_bytes[:, 2].astype(np.int32) << 16)
        )

        sign_mask = (
            audio & 0x800000
        ) != 0

        audio[sign_mask] -= 0x1000000

        audio = (
            audio.astype(np.float32)
            / 8388608.0
        )

    elif sample_width == 4:

        audio = np.frombuffer(
            raw,
            dtype=np.int32,
        ).astype(np.float32)

        audio /= 2147483648.0

    else:

        raise ValueError(
            f"Unsupported WAV sample width: "
            f"{sample_width} bytes"
        )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels,
        )

        audio = audio.mean(
            axis=1
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    return audio, sample_rate


# ============================================================================
# SYNTHETIC AUDIO FALLBACK
# ============================================================================

def create_synthetic_waveform(
    n_frames,
    sample_rate,
    hop_length,
    seed=SEED,
):

    """
    Educational fallback.

    Generates a voiced synthetic waveform with slowly changing pitch and
    amplitude.

    This allows the lesson to run even when the original WAV file is not
    available.

    It should NOT be interpreted as real speech data.
    """

    rng = np.random.default_rng(seed)

    n_samples = (
        n_frames * hop_length
    )

    time = np.arange(
        n_samples,
        dtype=np.float32,
    ) / sample_rate

    duration = len(time)

    f0_start = 120.0 + rng.uniform(
        -10.0,
        10.0,
    )

    f0_end = 180.0 + rng.uniform(
        -10.0,
        10.0,
    )

    f0 = np.linspace(
        f0_start,
        f0_end,
        duration,
        dtype=np.float32,
    )

    phase = (
        2.0
        * np.pi
        * np.cumsum(f0)
        / sample_rate
    )

    amplitude = (
        0.2
        + 0.1
        * np.sin(
            2.0
            * np.pi
            * 2.0
            * time
        )
    )

    waveform = (
        amplitude
        * np.sin(phase)
    )

    # Small amount of noise.
    waveform += (
        0.005
        * rng.standard_normal(
            n_samples
        )
    )

    waveform = np.clip(
        waveform,
        -1.0,
        1.0,
    )

    return waveform.astype(
        np.float32
    )


# ============================================================================
# MEL ORIENTATION
# ============================================================================

def normalize_mel_orientation(
    mel,
    npz_data,
):

    mel = np.asarray(mel)

    if mel.ndim != 2:

        raise ValueError(
            f"Mel must be 2-D, got {mel.shape}"
        )

    n_mels = None

    if "n_mels" in npz_data:

        n_mels = safe_int(
            npz_data["n_mels"]
        )

    if n_mels is not None:

        if mel.shape[0] == n_mels:

            return mel.T.astype(
                np.float32
            )

        if mel.shape[1] == n_mels:

            return mel.astype(
                np.float32
            )

        raise ValueError(
            f"Mel shape {mel.shape} "
            f"does not match n_mels={n_mels}"
        )

    common_mel_sizes = {
        40,
        64,
        80,
        100,
        128,
        160,
    }

    if mel.shape[0] in common_mel_sizes:

        return mel.T.astype(
            np.float32
        )

    if mel.shape[1] in common_mel_sizes:

        return mel.astype(
            np.float32
        )

    if mel.shape[0] < mel.shape[1]:

        return mel.T.astype(
            np.float32
        )

    return mel.astype(
        np.float32
    )


# ============================================================================
# ALIGNMENT PARSING
# ============================================================================

def parse_alignment_items(
    alignment_data,
):

    if "aligned_phonemes" in alignment_data:

        items = alignment_data[
            "aligned_phonemes"
        ]

    elif "phonemes" in alignment_data:

        items = alignment_data[
            "phonemes"
        ]

    elif isinstance(
        alignment_data,
        list,
    ):

        items = alignment_data

    else:

        raise ValueError(
            "No aligned_phonemes or phonemes "
            "field found."
        )

    parsed = []

    for index, item in enumerate(items):

        if not isinstance(
            item,
            dict,
        ):

            raise ValueError(
                f"Alignment item {index} "
                f"is not a dictionary."
            )

        phoneme = (
            item.get("phoneme")
            or item.get("token")
            or item.get("symbol")
        )

        if phoneme is None:

            raise ValueError(
                f"Alignment item {index} "
                f"has no phoneme."
            )

        phoneme = str(
            phoneme
        )

        start = safe_int(
            item.get(
                "start_frame"
            )
        )

        end = safe_int(
            item.get(
                "end_frame"
            )
        )

        duration = safe_int(
            item.get(
                "duration_frames"
            )
        )

        if start is not None and end is not None:

            if end <= start:

                raise ValueError(
                    f"Invalid interval "
                    f"[{start}, {end})"
                )

            duration = end - start

        elif duration is None:

            raise ValueError(
                f"Alignment item {index} "
                f"has no duration."
            )

        if duration < MIN_DURATION:

            raise ValueError(
                f"Invalid duration "
                f"{duration}"
            )

        parsed.append(
            {
                "phoneme": phoneme,
                "start_frame": start,
                "end_frame": end,
                "duration_frames": duration,
            }
        )

    return parsed


# ============================================================================
# SIMPLE AUTOCORRELATION F0 ESTIMATION
# ============================================================================

def estimate_f0_autocorrelation(
    waveform,
    sample_rate,
    hop_length=DEFAULT_HOP_LENGTH,
    win_length=DEFAULT_WIN_LENGTH,
    min_f0=MIN_F0,
    max_f0=MAX_F0,
):

    """
    Estimate frame-level F0 using normalized autocorrelation.

    For each frame:

        1. Apply Hann window.
        2. Remove DC.
        3. Compute autocorrelation.
        4. Search lag corresponding to min/max F0.
        5. Choose the strongest valid correlation peak.

    Pitch is returned as:

        0.0

    for unvoiced/uncertain frames.
    """

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.ndim != 1:

        waveform = waveform.reshape(
            -1
        )

    if len(waveform) == 0:

        return np.zeros(
            0,
            dtype=np.float32,
        )

    min_lag = max(
        1,
        int(
            sample_rate / max_f0
        ),
    )

    max_lag = min(
        win_length - 2,
        int(
            sample_rate / min_f0
        ),
    )

    window = np.hanning(
        win_length
    ).astype(
        np.float32
    )

    n_frames = int(
        math.ceil(
            max(
                1,
                len(waveform)
                / hop_length
            )
        )
    )

    f0_values = np.zeros(
        n_frames,
        dtype=np.float32,
    )

    for frame_index in range(
        n_frames
    ):

        start = (
            frame_index
            * hop_length
        )

        end = (
            start
            + win_length
        )

        frame = np.zeros(
            win_length,
            dtype=np.float32,
        )

        available = waveform[
            start:min(
                end,
                len(waveform),
            )
        ]

        if len(available) == 0:
            continue

        frame[
            :len(available)
        ] = available

        frame -= frame.mean()

        energy = np.mean(
            frame ** 2
        )

        if energy < 1e-7:
            continue

        frame *= window

        autocorr = np.correlate(
            frame,
            frame,
            mode="full",
        )

        autocorr = autocorr[
            win_length - 1:
        ]

        if autocorr[0] <= 1e-8:
            continue

        autocorr /= (
            autocorr[0]
            + 1e-8
        )

        search = autocorr[
            min_lag:max_lag + 1
        ]

        if len(search) == 0:
            continue

        peak_relative = int(
            np.argmax(search)
        )

        peak_lag = (
            min_lag
            + peak_relative
        )

        peak_value = float(
            search[
                peak_relative
            ]
        )

        # Conservative voicing threshold.
        if peak_value < 0.25:
            continue

        # Parabolic interpolation around the peak can improve resolution.
        if (
            peak_relative > 0
            and peak_relative < len(search) - 1
        ):

            y1 = float(
                search[
                    peak_relative - 1
                ]
            )

            y2 = peak_value

            y3 = float(
                search[
                    peak_relative + 1
                ]
            )

            denominator = (
                y1
                - 2.0 * y2
                + y3
            )

            if abs(
                denominator
            ) > 1e-8:

                offset = (
                    0.5
                    * (
                        y1 - y3
                    )
                    / denominator
                )

                peak_lag = (
                    peak_lag
                    + offset
                )

        if peak_lag <= 0:
            continue

        f0 = (
            sample_rate
            / peak_lag
        )

        if (
            min_f0
            <= f0
            <= max_f0
        ):

            f0_values[
                frame_index
            ] = f0

    return f0_values


# ============================================================================
# FRAME ENERGY
# ============================================================================

def compute_frame_energy(
    waveform,
    hop_length=DEFAULT_HOP_LENGTH,
    win_length=DEFAULT_WIN_LENGTH,
):

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    n_frames = int(
        math.ceil(
            max(
                1,
                len(waveform)
                / hop_length,
            )
        )
    )

    window = np.hanning(
        win_length
    ).astype(
        np.float32
    )

    energy = np.zeros(
        n_frames,
        dtype=np.float32,
    )

    for frame_index in range(
        n_frames
    ):

        start = (
            frame_index
            * hop_length
        )

        end = (
            start
            + win_length
        )

        frame = np.zeros(
            win_length,
            dtype=np.float32,
        )

        available = waveform[
            start:min(
                end,
                len(waveform),
            )
        ]

        if len(available) == 0:
            continue

        frame[
            :len(available)
        ] = available

        frame *= window

        rms = np.sqrt(
            np.mean(
                frame ** 2
            )
            + 1e-8
        )

        energy[
            frame_index
        ] = rms

    return energy


# ============================================================================
# LOG NORMALIZATION
# ============================================================================

def normalize_log_values(
    values,
    valid_mask,
):

    """
    Convert positive values into a normalized log representation.

    Invalid/unvoiced entries become zero.

    Returns:

        normalized
        mean
        std
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    valid = (
        valid_mask
        & np.isfinite(values)
        & (values > 0)
    )

    output = np.zeros_like(
        values,
        dtype=np.float32,
    )

    if not np.any(valid):

        return (
            output,
            0.0,
            1.0,
        )

    log_values = np.log(
        values[valid]
    )

    mean = float(
        log_values.mean()
    )

    std = float(
        log_values.std()
    )

    if std < 1e-6:
        std = 1.0

    output[valid] = (
        (
            log_values
            - mean
        )
        / std
    )

    return (
        output,
        mean,
        std,
    )


# ============================================================================
# PHONEME-LEVEL PROSODY AGGREGATION
# ============================================================================

def aggregate_frame_values_to_phonemes(
    frame_values,
    aligned_items,
    voiced_only=False,
):

    """
    Convert frame-level pitch/energy into one value per phoneme.

    Alignment intervals are assumed to be frame indices:

        [start_frame, end_frame)

    If explicit boundaries are unavailable, durations are used from
    the beginning of the sequence.
    """

    values = np.asarray(
        frame_values,
        dtype=np.float32,
    )

    phoneme_values = []

    current_frame = 0

    for item in aligned_items:

        start = item[
            "start_frame"
        ]

        end = item[
            "end_frame"
        ]

        duration = item[
            "duration_frames"
        ]

        if start is None or end is None:

            start = current_frame

            end = (
                current_frame
                + duration
            )

        start = max(
            0,
            int(start),
        )

        end = min(
            len(values),
            int(end),
        )

        segment = values[
            start:end
        ]

        if voiced_only:

            segment = segment[
                np.isfinite(segment)
                & (segment > 0)
            ]

        else:

            segment = segment[
                np.isfinite(segment)
            ]

        if len(segment) == 0:

            phoneme_values.append(
                0.0
            )

        else:

            phoneme_values.append(
                float(
                    np.mean(segment)
                )
            )

        current_frame += duration

    return np.asarray(
        phoneme_values,
        dtype=np.float32,
    )


# ============================================================================
# RECORD LOADING
# ============================================================================

def load_records():

    print()
    print("=" * 72)
    print("STEP 1 — LOAD ALIGNMENT, MEL, AND AUDIO")
    print("=" * 72)

    alignment_files = find_alignment_files()

    records = []

    for alignment_path in alignment_files:

        stem = alignment_path.stem

        if stem.endswith(
            "_alignment"
        ):

            base_stem = stem[
                :-len("_alignment")
            ]

        else:

            base_stem = stem

        mel_path = find_mel_file(
            base_stem
        )

        if mel_path is None:

            print(
                f"WARNING: no Mel file for "
                f"{base_stem}"
            )

            continue

        try:

            with open(
                alignment_path,
                "r",
                encoding="utf-8",
            ) as f:

                alignment_data = json.load(
                    f
                )

            aligned_items = (
                parse_alignment_items(
                    alignment_data
                )
            )

            # ----------------------------------------------------------------
            # Load Mel
            # ----------------------------------------------------------------

            with np.load(
                mel_path,
                allow_pickle=True,
            ) as data:

                if (
                    "normalized_log_mel"
                    in data
                ):

                    mel = data[
                        "normalized_log_mel"
                    ]

                elif "log_mel" in data:

                    mel = data[
                        "log_mel"
                    ]

                else:

                    raise ValueError(
                        "No log Mel array "
                        "found."
                    )

                mel = normalize_mel_orientation(
                    mel,
                    data,
                )

                sample_rate = (
                    safe_int(
                        data["sample_rate"]
                    )
                    if "sample_rate"
                    in data
                    else 16000
                )

                hop_length = (
                    safe_int(
                        data["hop_length"]
                    )
                    if "hop_length"
                    in data
                    else DEFAULT_HOP_LENGTH
                )

            mel_frames = mel.shape[0]

            # ----------------------------------------------------------------
            # Alignment frame interval
            # ----------------------------------------------------------------

            has_absolute = all(
                item[
                    "start_frame"
                ] is not None
                and item[
                    "end_frame"
                ] is not None
                for item in aligned_items
            )

            if has_absolute:

                previous_end = None

                for item in aligned_items:

                    start = item[
                        "start_frame"
                    ]

                    end = item[
                        "end_frame"
                    ]

                    if (
                        start < 0
                        or end > mel_frames
                    ):

                        raise ValueError(
                            f"Alignment interval "
                            f"[{start}, {end}) "
                            f"outside Mel range "
                            f"{mel_frames}"
                        )

                    if (
                        previous_end
                        is not None
                        and start
                        != previous_end
                    ):

                        raise ValueError(
                            "Alignment intervals "
                            "are not continuous."
                        )

                    previous_end = end

                first_start = (
                    aligned_items[0][
                        "start_frame"
                    ]
                )

                last_end = (
                    aligned_items[-1][
                        "end_frame"
                    ]
                )

                interval_length = (
                    last_end
                    - first_start
                )

                duration_sum = sum(
                    item[
                        "duration_frames"
                    ]
                    for item
                    in aligned_items
                )

                if (
                    interval_length
                    != duration_sum
                ):

                    raise ValueError(
                        "Alignment duration "
                        "mismatch."
                    )

                if (
                    first_start != 0
                    or mel_frames
                    != interval_length
                ):

                    print(
                        f"INFO: {base_stem}: "
                        f"cropping Mel "
                        f"[{first_start}:{last_end}] "
                        f"from {mel_frames} frames."
                    )

                    mel = mel[
                        first_start:last_end
                    ]

            else:

                duration_sum = sum(
                    item[
                        "duration_frames"
                    ]
                    for item
                    in aligned_items
                )

                if duration_sum > mel_frames:

                    raise ValueError(
                        f"Duration sum "
                        f"{duration_sum} > "
                        f"Mel frames "
                        f"{mel_frames}"
                    )

                if duration_sum != mel_frames:

                    print(
                        f"INFO: {base_stem}: "
                        f"using first "
                        f"{duration_sum} "
                        f"Mel frames."
                    )

                    mel = mel[
                        :duration_sum
                    ]

            # ----------------------------------------------------------------
            # Audio
            # ----------------------------------------------------------------

            audio_path = find_audio_file(
                base_stem,
                alignment_data,
            )

            waveform = None

            if audio_path is not None:

                waveform, audio_sr = (
                    load_wav(
                        audio_path
                    )
                )

                if audio_sr != sample_rate:

                    warnings.warn(
                        f"{base_stem}: WAV sample "
                        f"rate={audio_sr}, Mel "
                        f"sample rate={sample_rate}."
                    )

                print(
                    f"Audio: {audio_path}"
                )

            else:

                print(
                    f"WARNING: no WAV found for "
                    f"{base_stem}."
                )

                print(
                    "Using synthetic educational "
                    "waveform."
                )

                waveform = (
                    create_synthetic_waveform(
                        n_frames=mel.shape[0],
                        sample_rate=sample_rate,
                        hop_length=hop_length,
                        seed=SEED,
                    )
                )

            # ----------------------------------------------------------------
            # Pitch and energy
            # ----------------------------------------------------------------

            frame_pitch = (
                estimate_f0_autocorrelation(
                    waveform,
                    sample_rate,
                    hop_length=hop_length,
                    win_length=max(
                        DEFAULT_WIN_LENGTH,
                        hop_length * 4,
                    ),
                )
            )

            frame_energy = (
                compute_frame_energy(
                    waveform,
                    hop_length=hop_length,
                    win_length=max(
                        DEFAULT_WIN_LENGTH,
                        hop_length * 4,
                    ),
                )
            )

            # Match feature frame count to Mel.
            target_frames = mel.shape[0]

            if len(frame_pitch) < target_frames:

                frame_pitch = np.pad(
                    frame_pitch,
                    (
                        0,
                        target_frames
                        - len(frame_pitch),
                    ),
                )

            else:

                frame_pitch = frame_pitch[
                    :target_frames
                ]

            if len(frame_energy) < target_frames:

                frame_energy = np.pad(
                    frame_energy,
                    (
                        0,
                        target_frames
                        - len(frame_energy),
                    ),
                )

            else:

                frame_energy = frame_energy[
                    :target_frames
                ]

            # ----------------------------------------------------------------
            # Aggregate to phonemes
            # ----------------------------------------------------------------

            phonemes = [
                item["phoneme"]
                for item in aligned_items
            ]

            durations = [
                item["duration_frames"]
                for item in aligned_items
            ]

            pitch = (
                aggregate_frame_values_to_phonemes(
                    frame_pitch,
                    aligned_items,
                    voiced_only=True,
                )
            )

            energy = (
                aggregate_frame_values_to_phonemes(
                    frame_energy,
                    aligned_items,
                    voiced_only=False,
                )
            )

            if len(pitch) != len(
                phonemes
            ):

                raise ValueError(
                    "Pitch/phoneme count mismatch."
                )

            if len(energy) != len(
                phonemes
            ):

                raise ValueError(
                    "Energy/phoneme count mismatch."
                )

            # ----------------------------------------------------------------
            # Normalize pitch and energy
            # ----------------------------------------------------------------

            pitch_valid = (
                pitch > 0
            )

            energy_valid = (
                energy > 1e-8
            )

            normalized_pitch, pitch_mean, pitch_std = (
                normalize_log_values(
                    pitch,
                    pitch_valid,
                )
            )

            normalized_energy, energy_mean, energy_std = (
                normalize_log_values(
                    energy,
                    energy_valid,
                )
            )

            records.append(
                {
                    "id": base_stem,
                    "phonemes": phonemes,
                    "durations": durations,
                    "mel": mel.astype(
                        np.float32
                    ),
                    "pitch": normalized_pitch,
                    "energy": normalized_energy,
                    "raw_pitch": pitch,
                    "raw_energy": energy,
                    "pitch_valid": pitch_valid,
                    "energy_valid": energy_valid,
                    "pitch_mean": pitch_mean,
                    "pitch_std": pitch_std,
                    "energy_mean": energy_mean,
                    "energy_std": energy_std,
                    "sample_rate": sample_rate,
                    "hop_length": hop_length,
                }
            )

            print(
                f"Loaded: {base_stem:35s} "
                f"phonemes={len(phonemes):4d} "
                f"frames={mel.shape[0]:5d} "
                f"pitch_voiced="
                f"{pitch_valid.sum():4d}/"
                f"{len(pitch):4d}"
            )

        except Exception as exc:

            print(
                f"WARNING: skipping "
                f"{alignment_path.name}: "
                f"{exc}"
            )

    if not records:

        raise RuntimeError(
            "No valid Lesson 60 records found."
        )

    return records


# ============================================================================
# VOCABULARY
# ============================================================================

def build_vocabulary(records):

    phonemes = set()

    for record in records:

        phonemes.update(
            record["phonemes"]
        )

    vocabulary = {
        "<pad>": PAD_ID
    }

    for phoneme in sorted(
        phonemes
    ):

        if phoneme == "<pad>":
            continue

        vocabulary[
            phoneme
        ] = len(vocabulary)

    return vocabulary


# ============================================================================
# DATASET
# ============================================================================

class ProsodyDataset(Dataset):

    def __init__(
        self,
        records,
        vocabulary,
    ):

        self.records = records

        self.vocabulary = vocabulary

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

        ids = [
            self.vocabulary[
                phoneme
            ]
            for phoneme
            in record["phonemes"]
        ]

        durations = np.asarray(
            record["durations"],
            dtype=np.int64,
        )

        return {
            "id": record["id"],

            "phoneme_ids": torch.tensor(
                ids,
                dtype=torch.long,
            ),

            "durations": torch.tensor(
                durations,
                dtype=torch.long,
            ),

            "log_durations": torch.log1p(
                torch.tensor(
                    durations,
                    dtype=torch.float32,
                )
            ),

            "pitch": torch.tensor(
                record["pitch"],
                dtype=torch.float32,
            ),

            "energy": torch.tensor(
                record["energy"],
                dtype=torch.float32,
            ),

            "pitch_valid": torch.tensor(
                record["pitch_valid"],
                dtype=torch.bool,
            ),

            "energy_valid": torch.tensor(
                record["energy_valid"],
                dtype=torch.bool,
            ),

            "mel": torch.tensor(
                record["mel"],
                dtype=torch.float32,
            ),
        }


# ============================================================================
# COLLATE
# ============================================================================

def collate_batch(batch):

    batch_size = len(
        batch
    )

    max_phonemes = max(
        item["phoneme_ids"].shape[0]
        for item in batch
    )

    max_frames = max(
        item["mel"].shape[0]
        for item in batch
    )

    n_mels = batch[0][
        "mel"
    ].shape[1]

    phoneme_ids = torch.full(
        (
            batch_size,
            max_phonemes,
        ),
        PAD_ID,
        dtype=torch.long,
    )

    durations = torch.zeros(
        (
            batch_size,
            max_phonemes,
        ),
        dtype=torch.long,
    )

    log_durations = torch.zeros(
        (
            batch_size,
            max_phonemes,
        ),
        dtype=torch.float32,
    )

    pitch = torch.zeros(
        (
            batch_size,
            max_phonemes,
        ),
        dtype=torch.float32,
    )

    energy = torch.zeros(
        (
            batch_size,
            max_phonemes,
        ),
        dtype=torch.float32,
    )

    pitch_valid = torch.zeros(
        (
            batch_size,
            max_phonemes,
        ),
        dtype=torch.bool,
    )

    energy_valid = torch.zeros(
        (
            batch_size,
            max_phonemes,
        ),
        dtype=torch.bool,
    )

    mel = torch.zeros(
        (
            batch_size,
            max_frames,
            n_mels,
        ),
        dtype=torch.float32,
    )

    phoneme_padding_mask = torch.ones(
        (
            batch_size,
            max_phonemes,
        ),
        dtype=torch.bool,
    )

    mel_padding_mask = torch.ones(
        (
            batch_size,
            max_frames,
        ),
        dtype=torch.bool,
    )

    ids = []

    for i, item in enumerate(
        batch
    ):

        p_len = item[
            "phoneme_ids"
        ].shape[0]

        f_len = item[
            "mel"
        ].shape[0]

        phoneme_ids[
            i,
            :p_len
        ] = item[
            "phoneme_ids"
        ]

        durations[
            i,
            :p_len
        ] = item[
            "durations"
        ]

        log_durations[
            i,
            :p_len
        ] = item[
            "log_durations"
        ]

        pitch[
            i,
            :p_len
        ] = item[
            "pitch"
        ]

        energy[
            i,
            :p_len
        ] = item[
            "energy"
        ]

        pitch_valid[
            i,
            :p_len
        ] = item[
            "pitch_valid"
        ]

        energy_valid[
            i,
            :p_len
        ] = item[
            "energy_valid"
        ]

        mel[
            i,
            :f_len
        ] = item[
            "mel"
        ]

        phoneme_padding_mask[
            i,
            :p_len
        ] = False

        mel_padding_mask[
            i,
            :f_len
        ] = False

        ids.append(
            item["id"]
        )

    return {
        "ids": ids,
        "phoneme_ids": phoneme_ids,
        "durations": durations,
        "log_durations": log_durations,
        "pitch": pitch,
        "energy": energy,
        "pitch_valid": pitch_valid,
        "energy_valid": energy_valid,
        "mel": mel,
        "phoneme_padding_mask": phoneme_padding_mask,
        "mel_padding_mask": mel_padding_mask,
    }


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(
    nn.Module
):

    def __init__(
        self,
        embed_dim,
        max_len=4096,
    ):

        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embed_dim,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(
                    10000.0
                )
                / embed_dim
            )
        )

        pe = torch.zeros(
            max_len,
            embed_dim,
        )

        pe[:, 0::2] = torch.sin(
            position
            * div_term
        )

        pe[:, 1::2] = torch.cos(
            position
            * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(
        self,
        x,
    ):

        length = x.shape[1]

        return (
            x
            + self.pe[
                :,
                :length,
            ]
        )


# ============================================================================
# PHONEME ENCODER
# ============================================================================

class PhonemeEncoder(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        heads,
        layers,
        ff_dim,
        dropout,
        max_len,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=PAD_ID,
        )

        self.position = PositionalEncoding(
            embed_dim,
            max_len,
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                layer,
                num_layers=layers,
            )
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        ids,
        padding_mask,
    ):

        x = self.embedding(
            ids
        )

        x = self.position(
            x
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        return self.norm(
            x
        )


# ============================================================================
# GENERIC PREDICTOR
# ============================================================================

class VariancePredictor(
    nn.Module
):

    """
    Shared architecture for:

        duration
        pitch
        energy
    """

    def __init__(
        self,
        embed_dim,
        dropout=0.1,
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.output = nn.Linear(
            embed_dim,
            1,
        )

    def forward(
        self,
        x,
    ):

        x = x.transpose(
            1,
            2,
        )

        x = self.conv1(
            x
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.norm1(
            x
        )

        x = F.relu(
            x
        )

        x = self.dropout(
            x
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.conv2(
            x
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.norm2(
            x
        )

        x = F.relu(
            x
        )

        x = self.dropout(
            x
        )

        return self.output(
            x
        ).squeeze(-1)


# ============================================================================
# VARIANCE EMBEDDING
# ============================================================================

class VarianceEmbedding(
    nn.Module
):

    """
    Converts scalar pitch/energy values into learned embeddings.

    A small MLP is used instead of a discrete lookup table.

    This allows continuous values such as:

        pitch = 0.37
        energy = -0.22

    to become continuous embeddings.
    """

    def __init__(
        self,
        embed_dim,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                1,
                embed_dim,
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim,
                embed_dim,
            ),

        )

    def forward(
        self,
        values,
    ):

        return self.network(
            values.unsqueeze(-1)
        )


# ============================================================================
# LENGTH REGULATOR
# ============================================================================

class LengthRegulator(
    nn.Module
):

    def __init__(
        self,
        max_duration=MAX_DURATION,
    ):

        super().__init__()

        self.max_duration = (
            max_duration
        )

    def forward(
        self,
        x,
        durations,
    ):

        sequences = []

        for b in range(
            x.shape[0]
        ):

            pieces = []

            for t in range(
                x.shape[1]
            ):

                duration = int(
                    durations[
                        b,
                        t,
                    ].item()
                )

                if duration <= 0:
                    continue

                duration = min(
                    duration,
                    self.max_duration,
                )

                repeated = x[
                    b,
                    t,
                ].unsqueeze(0).repeat(
                    duration,
                    1,
                )

                pieces.append(
                    repeated
                )

            if pieces:

                sequence = torch.cat(
                    pieces,
                    dim=0,
                )

            else:

                sequence = x.new_zeros(
                    (
                        1,
                        x.shape[-1],
                    )
                )

            sequences.append(
                sequence
            )

        max_length = max(
            sequence.shape[0]
            for sequence
            in sequences
        )

        output = x.new_zeros(
            (
                x.shape[0],
                max_length,
                x.shape[-1],
            )
        )

        padding_mask = torch.ones(
            (
                x.shape[0],
                max_length,
            ),
            dtype=torch.bool,
            device=x.device,
        )

        for b, sequence in enumerate(
            sequences
        ):

            length = sequence.shape[0]

            output[
                b,
                :length,
            ] = sequence

            padding_mask[
                b,
                :length,
            ] = False

        return (
            output,
            padding_mask,
        )


# ============================================================================
# MEL DECODER
# ============================================================================

class MelDecoder(
    nn.Module
):

    def __init__(
        self,
        embed_dim,
        heads,
        layers,
        ff_dim,
        dropout,
        n_mels,
        max_len,
    ):

        super().__init__()

        self.position = (
            PositionalEncoding(
                embed_dim,
                max_len,
            )
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.decoder = (
            nn.TransformerEncoder(
                layer,
                num_layers=layers,
            )
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.output = nn.Linear(
            embed_dim,
            n_mels,
        )

    def forward(
        self,
        x,
        padding_mask,
    ):

        x = self.position(
            x
        )

        x = self.decoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.norm(
            x
        )

        return self.output(
            x
        )


# ============================================================================
# POST-NET
# ============================================================================

class MelPostNet(
    nn.Module
):

    def __init__(
        self,
        n_mels,
        channels=POSTNET_CHANNELS,
        layers=POSTNET_LAYERS,
        kernel_size=POSTNET_KERNEL_SIZE,
        dropout=DROPOUT,
    ):

        super().__init__()

        modules = []

        padding = (
            kernel_size // 2
        )

        modules.extend(
            [
                nn.Conv1d(
                    n_mels,
                    channels,
                    kernel_size,
                    padding=padding,
                ),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
        )

        for _ in range(
            layers - 2
        ):

            modules.extend(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=padding,
                    ),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                ]
            )

        modules.append(
            nn.Conv1d(
                channels,
                n_mels,
                kernel_size,
                padding=padding,
            )
        )

        self.network = nn.Sequential(
            *modules
        )

    def forward(
        self,
        mel,
    ):

        x = mel.transpose(
            1,
            2,
        )

        x = self.network(
            x
        )

        return x.transpose(
            1,
            2,
        )


# ============================================================================
# LESSON 60 MODEL
# ============================================================================

class PitchEnergyFastSpeech(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        n_mels,
    ):

        super().__init__()

        self.encoder = (
            PhonemeEncoder(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                heads=ATTENTION_HEADS,
                layers=ENCODER_LAYERS,
                ff_dim=FF_DIM,
                dropout=DROPOUT,
                max_len=MAX_SEQUENCE_LENGTH,
            )
        )

        # ---------------------------------------------------------------
        # Three separate predictors.
        # ---------------------------------------------------------------

        self.duration_predictor = (
            VariancePredictor(
                EMBED_DIM,
                DROPOUT,
            )
        )

        self.pitch_predictor = (
            VariancePredictor(
                EMBED_DIM,
                DROPOUT,
            )
        )

        self.energy_predictor = (
            VariancePredictor(
                EMBED_DIM,
                DROPOUT,
            )
        )

        # ---------------------------------------------------------------
        # Continuous variance embeddings.
        # ---------------------------------------------------------------

        self.pitch_embedding = (
            VarianceEmbedding(
                EMBED_DIM
            )
        )

        self.energy_embedding = (
            VarianceEmbedding(
                EMBED_DIM
            )
        )

        # ---------------------------------------------------------------
        # Duration expansion.
        # ---------------------------------------------------------------

        self.length_regulator = (
            LengthRegulator(
                MAX_DURATION
            )
        )

        # ---------------------------------------------------------------
        # Acoustic decoder.
        # ---------------------------------------------------------------

        self.decoder = (
            MelDecoder(
                embed_dim=EMBED_DIM,
                heads=ATTENTION_HEADS,
                layers=DECODER_LAYERS,
                ff_dim=FF_DIM,
                dropout=DROPOUT,
                n_mels=n_mels,
                max_len=MAX_SYNTHESIS_FRAMES,
            )
        )

        self.postnet = (
            MelPostNet(
                n_mels
            )
        )

    def predict_all_variances(
        self,
        encoded,
        phoneme_padding_mask,
    ):

        predicted_log_duration = (
            self.duration_predictor(
                encoded
            )
        )

        predicted_log_duration = (
            predicted_log_duration.clamp(
                0.0,
                math.log1p(
                    MAX_DURATION
                ),
            )
        )

        predicted_duration = (
            torch.round(
                torch.expm1(
                    predicted_log_duration
                )
            )
            .long()
            .clamp(
                MIN_DURATION,
                MAX_DURATION,
            )
        )

        predicted_duration = (
            predicted_duration.masked_fill(
                phoneme_padding_mask,
                0,
            )
        )

        predicted_pitch = (
            self.pitch_predictor(
                encoded
            )
        )

        predicted_energy = (
            self.energy_predictor(
                encoded
            )
        )

        predicted_pitch = (
            predicted_pitch.masked_fill(
                phoneme_padding_mask,
                0.0,
            )
        )

        predicted_energy = (
            predicted_energy.masked_fill(
                phoneme_padding_mask,
                0.0,
            )
        )

        return {
            "predicted_log_duration":
                predicted_log_duration,

            "predicted_duration":
                predicted_duration,

            "predicted_pitch":
                predicted_pitch,

            "predicted_energy":
                predicted_energy,
        }

    def forward(
        self,
        phoneme_ids,
        phoneme_padding_mask,
        ground_truth_durations=None,
        ground_truth_pitch=None,
        ground_truth_energy=None,
        use_predicted_variances=False,
    ):

        encoded = self.encoder(
            phoneme_ids,
            phoneme_padding_mask,
        )

        predictions = (
            self.predict_all_variances(
                encoded,
                phoneme_padding_mask,
            )
        )

        predicted_duration = (
            predictions[
                "predicted_duration"
            ]
        )

        # ---------------------------------------------------------------
        # Select durations.
        # ---------------------------------------------------------------

        if (
            ground_truth_durations
            is not None
            and not use_predicted_variances
        ):

            durations = (
                ground_truth_durations
                .clamp(
                    0,
                    MAX_DURATION,
                )
            )

            durations = durations.masked_fill(
                phoneme_padding_mask,
                0,
            )

        else:

            durations = (
                predicted_duration
            )

        # ---------------------------------------------------------------
        # Select pitch.
        # ---------------------------------------------------------------

        if (
            ground_truth_pitch
            is not None
            and not use_predicted_variances
        ):

            pitch = (
                ground_truth_pitch
            )

        else:

            pitch = (
                predictions[
                    "predicted_pitch"
                ]
            )

        # ---------------------------------------------------------------
        # Select energy.
        # ---------------------------------------------------------------

        if (
            ground_truth_energy
            is not None
            and not use_predicted_variances
        ):

            energy = (
                ground_truth_energy
            )

        else:

            energy = (
                predictions[
                    "predicted_energy"
                ]
            )

        # ---------------------------------------------------------------
        # Variance embeddings.
        # ---------------------------------------------------------------

        pitch_emb = (
            self.pitch_embedding(
                pitch
            )
        )

        energy_emb = (
            self.energy_embedding(
                energy
            )
        )

        # ---------------------------------------------------------------
        # Add prosody to encoder representation.
        # ---------------------------------------------------------------

        variance_encoded = (
            encoded
            + pitch_emb
            + energy_emb
        )

        # ---------------------------------------------------------------
        # Length regulation.
        # ---------------------------------------------------------------

        expanded, frame_padding_mask = (
            self.length_regulator(
                variance_encoded,
                durations,
            )
        )

        # ---------------------------------------------------------------
        # Mel decoder.
        # ---------------------------------------------------------------

        coarse_mel = self.decoder(
            expanded,
            frame_padding_mask,
        )

        residual = self.postnet(
            coarse_mel
        )

        final_mel = (
            coarse_mel
            + residual
        )

        # Mask padding.
        final_mel = final_mel.masked_fill(
            frame_padding_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        coarse_mel = coarse_mel.masked_fill(
            frame_padding_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        residual = residual.masked_fill(
            frame_padding_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        return {
            "encoded": encoded,

            "predicted_log_duration":
                predictions[
                    "predicted_log_duration"
                ],

            "predicted_duration":
                predicted_duration,

            "predicted_pitch":
                predictions[
                    "predicted_pitch"
                ],

            "predicted_energy":
                predictions[
                    "predicted_energy"
                ],

            "used_duration":
                durations,

            "used_pitch":
                pitch,

            "used_energy":
                energy,

            "frame_padding_mask":
                frame_padding_mask,

            "coarse_mel":
                coarse_mel,

            "residual":
                residual,

            "final_mel":
                final_mel,
        }


# ============================================================================
# MASKED LOSS
# ============================================================================

def masked_mse(
    prediction,
    target,
    mask,
):

    valid = mask

    error = (
        prediction
        - target
    ) ** 2

    error = error.masked_fill(
        ~valid,
        0.0,
    )

    denominator = (
        valid.sum()
        .clamp_min(1)
    )

    return (
        error.sum()
        / denominator
    )


def masked_l1(
    prediction,
    target,
    mask,
):

    valid = mask

    error = torch.abs(
        prediction
        - target
    )

    error = error.masked_fill(
        ~valid,
        0.0,
    )

    denominator = (
        valid.sum()
        .clamp_min(1)
    )

    return (
        error.sum()
        / denominator
    )


def mel_l1(
    prediction,
    target,
    padding_mask,
):

    valid = (
        ~padding_mask
    ).unsqueeze(-1)

    error = torch.abs(
        prediction
        - target
    )

    error = error.masked_fill(
        ~valid,
        0.0,
    )

    denominator = (
        valid.sum()
        .clamp_min(1)
        * prediction.shape[-1]
    )

    return (
        error.sum()
        / denominator
    )


def mel_mse(
    prediction,
    target,
    padding_mask,
):

    valid = (
        ~padding_mask
    ).unsqueeze(-1)

    error = (
        prediction
        - target
    ) ** 2

    error = error.masked_fill(
        ~valid,
        0.0,
    )

    denominator = (
        valid.sum()
        .clamp_min(1)
        * prediction.shape[-1]
    )

    return (
        error.sum()
        / denominator
    )


# ============================================================================
# LOSS
# ============================================================================

def compute_losses(
    outputs,
    batch,
):

    phoneme_valid = (
        ~batch[
            "phoneme_padding_mask"
        ]
    )

    pitch_valid = (
        phoneme_valid
        & batch[
            "pitch_valid"
        ]
    )

    energy_valid = (
        phoneme_valid
        & batch[
            "energy_valid"
        ]
    )

    duration_loss = masked_mse(
        outputs[
            "predicted_log_duration"
        ],
        batch[
            "log_durations"
        ],
        phoneme_valid,
    )

    pitch_loss = masked_mse(
        outputs[
            "predicted_pitch"
        ],
        batch[
            "pitch"
        ],
        pitch_valid,
    )

    energy_loss = masked_mse(
        outputs[
            "predicted_energy"
        ],
        batch[
            "energy"
        ],
        energy_valid,
    )

    coarse_l1 = mel_l1(
        outputs[
            "coarse_mel"
        ],
        batch[
            "mel"
        ],
        outputs[
            "frame_padding_mask"
        ],
    )

    coarse_mse = mel_mse(
        outputs[
            "coarse_mel"
        ],
        batch[
            "mel"
        ],
        outputs[
            "frame_padding_mask"
        ],
    )

    final_l1 = mel_l1(
        outputs[
            "final_mel"
        ],
        batch[
            "mel"
        ],
        outputs[
            "frame_padding_mask"
        ],
    )

    final_mse = mel_mse(
        outputs[
            "final_mel"
        ],
        batch[
            "mel"
        ],
        outputs[
            "frame_padding_mask"
        ],
    )

    residual_loss = mel_mse(
        outputs[
            "residual"
        ],
        torch.zeros_like(
            outputs[
                "residual"
            ]
        ),
        outputs[
            "frame_padding_mask"
        ],
    )

    total = (

        DURATION_LOSS_WEIGHT
        * duration_loss

        +

        PITCH_LOSS_WEIGHT
        * pitch_loss

        +

        ENERGY_LOSS_WEIGHT
        * energy_loss

        +

        COARSE_MEL_WEIGHT
        * (
            MEL_L1_WEIGHT
            * coarse_l1
            +
            MEL_MSE_WEIGHT
            * coarse_mse
        )

        +

        FINAL_MEL_WEIGHT
        * (
            MEL_L1_WEIGHT
            * final_l1
            +
            MEL_MSE_WEIGHT
            * final_mse
        )

        +

        RESIDUAL_WEIGHT
        * residual_loss
    )

    return {
        "total": total,
        "duration": duration_loss,
        "pitch": pitch_loss,
        "energy": energy_loss,
        "coarse_l1": coarse_l1,
        "coarse_mse": coarse_mse,
        "final_l1": final_l1,
        "final_mse": final_mse,
        "residual": residual_loss,
    }


# ============================================================================
# DATA LOADER
# ============================================================================

def create_loader(
    records,
    vocabulary,
):

    dataset = ProsodyDataset(
        records,
        vocabulary,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )

    return dataset, loader


# ============================================================================
# PARAMETER COUNT
# ============================================================================

def count_parameters(
    model,
):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model,
    loader,
    vocabulary,
    n_mels,
):

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
            patience=10,
            min_lr=1e-6,
        )
    )

    history = {
        "total": [],
        "duration": [],
        "pitch": [],
        "energy": [],
        "mel": [],
    }

    best_loss = float(
        "inf"
    )

    best_epoch = -1

    patience_counter = 0

    print()
    print("=" * 72)
    print("STEP 3 — TRAIN PITCH + ENERGY ACOUSTIC MODEL")
    print("=" * 72)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        totals = {
            "total": 0.0,
            "duration": 0.0,
            "pitch": 0.0,
            "energy": 0.0,
            "mel": 0.0,
        }

        for batch in loader:

            batch = {
                key: value.to(DEVICE)
                if torch.is_tensor(value)
                else value
                for key, value
                in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                phoneme_ids=batch[
                    "phoneme_ids"
                ],
                phoneme_padding_mask=batch[
                    "phoneme_padding_mask"
                ],
                ground_truth_durations=batch[
                    "durations"
                ],
                ground_truth_pitch=batch[
                    "pitch"
                ],
                ground_truth_energy=batch[
                    "energy"
                ],
                use_predicted_variances=False,
            )

            losses = compute_losses(
                outputs,
                batch,
            )

            losses[
                "total"
            ].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            totals[
                "total"
            ] += losses[
                "total"
            ].item()

            totals[
                "duration"
            ] += losses[
                "duration"
            ].item()

            totals[
                "pitch"
            ] += losses[
                "pitch"
            ].item()

            totals[
                "energy"
            ] += losses[
                "energy"
            ].item()

            totals[
                "mel"
            ] += losses[
                "final_l1"
            ].item()

        count = max(
            1,
            len(loader),
        )

        metrics = {
            key: value / count
            for key, value
            in totals.items()
        }

        scheduler.step(
            metrics["total"]
        )

        for key in history:

            history[
                key
            ].append(
                metrics[key]
            )

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            lr = optimizer.param_groups[
                0
            ]["lr"]

            print(
                f"Epoch {epoch:4d}/{EPOCHS} | "
                f"total={metrics['total']:.6f} | "
                f"duration={metrics['duration']:.6f} | "
                f"pitch={metrics['pitch']:.6f} | "
                f"energy={metrics['energy']:.6f} | "
                f"mel={metrics['mel']:.6f} | "
                f"lr={lr:.2e}"
            )

        if (
            metrics["total"]
            < best_loss
        ):

            best_loss = (
                metrics["total"]
            )

            best_epoch = epoch

            patience_counter = 0

            checkpoint = {

                "epoch": epoch,

                "loss": best_loss,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "vocabulary":
                    vocabulary,

                "n_mels":
                    n_mels,

                "embed_dim":
                    EMBED_DIM,

                "attention_heads":
                    ATTENTION_HEADS,

                "encoder_layers":
                    ENCODER_LAYERS,

                "decoder_layers":
                    DECODER_LAYERS,

                "ff_dim":
                    FF_DIM,

                "max_duration":
                    MAX_DURATION,
            }

            torch.save(
                checkpoint,
                OUTPUT_DIR
                / "pitch_energy_model_best.pt",
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                f"Early stopping at "
                f"epoch {epoch}."
            )

            break

    print()
    print(
        f"Best training loss: "
        f"{best_loss:.6f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "pitch_energy_model_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"Best checkpoint loaded: "
        f"{checkpoint_path}"
    )

    return history


# ============================================================================
# TRAINING PLOT
# ============================================================================

def save_training_plot(
    history,
):

    plt.figure(
        figsize=(12, 7)
    )

    epochs = np.arange(
        1,
        len(history["total"])
        + 1,
    )

    plt.plot(
        epochs,
        history["total"],
        label="Total",
    )

    plt.plot(
        epochs,
        history["duration"],
        label="Duration",
    )

    plt.plot(
        epochs,
        history["pitch"],
        label="Pitch",
    )

    plt.plot(
        epochs,
        history["energy"],
        label="Energy",
    )

    plt.plot(
        epochs,
        history["mel"],
        label="Mel L1",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Lesson 60 — Pitch and Energy Training"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "pitch_energy_training.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(
        f"Training plot saved: "
        f"{path}"
    )


# ============================================================================
# 1-D PLOT
# ============================================================================

def save_curve(
    values,
    title,
    ylabel,
    filename,
    valid_mask=None,
):

    values = np.asarray(
        values
    )

    plt.figure(
        figsize=(14, 5)
    )

    x = np.arange(
        len(values)
    )

    if valid_mask is not None:

        valid_mask = np.asarray(
            valid_mask
        )

        plot_values = values.copy()

        plot_values[
            ~valid_mask
        ] = np.nan

    else:

        plot_values = values

    plt.plot(
        x,
        plot_values,
    )

    plt.xlabel(
        "Phoneme index"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        alpha=0.3,
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

    return path


# ============================================================================
# MEL IMAGE
# ============================================================================

def save_mel_image(
    mel,
    title,
    filename,
):

    mel = np.asarray(
        mel
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.imshow(
        mel.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    plt.colorbar(
        label="Mel value"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Mel bin"
    )

    plt.title(
        title
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

    return path


# ============================================================================
# EXAMPLE EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_example(
    model,
    record,
    vocabulary,
):

    print()
    print("=" * 72)
    print("STEP 4 — PROSODY PREDICTION")
    print("=" * 72)

    model.eval()

    phoneme_ids = torch.tensor(
        [
            vocabulary[p]
            for p
            in record["phonemes"]
        ],
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    padding_mask = torch.zeros(
        (
            1,
            len(
                record[
                    "phonemes"
                ]
            ),
        ),
        dtype=torch.bool,
        device=DEVICE,
    )

    durations = torch.tensor(
        record["durations"],
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    pitch = torch.tensor(
        record["pitch"],
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    energy = torch.tensor(
        record["energy"],
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    target_mel = torch.tensor(
        record["mel"],
        dtype=torch.float32,
        device=DEVICE,
    )

    # ========================================================================
    # TEACHER-FORCED PROSODY
    # ========================================================================

    teacher_outputs = model(
        phoneme_ids=phoneme_ids,
        phoneme_padding_mask=padding_mask,
        ground_truth_durations=durations,
        ground_truth_pitch=pitch,
        ground_truth_energy=energy,
        use_predicted_variances=False,
    )

    teacher_mel = (
        teacher_outputs[
            "final_mel"
        ][0]
        .cpu()
        .numpy()
    )

    target = (
        target_mel
        .cpu()
        .numpy()
    )

    common_frames = min(
        teacher_mel.shape[0],
        target.shape[0],
    )

    teacher_mel = teacher_mel[
        :common_frames
    ]

    target = target[
        :common_frames
    ]

    teacher_mae = np.mean(
        np.abs(
            teacher_mel
            - target
        )
    )

    # ========================================================================
    # PREDICTED PROSODY
    # ========================================================================

    predicted_outputs = model(
        phoneme_ids=phoneme_ids,
        phoneme_padding_mask=padding_mask,
        ground_truth_durations=None,
        ground_truth_pitch=None,
        ground_truth_energy=None,
        use_predicted_variances=True,
    )

    predicted_pitch = (
        predicted_outputs[
            "predicted_pitch"
        ][0]
        .cpu()
        .numpy()
    )

    predicted_energy = (
        predicted_outputs[
            "predicted_energy"
        ][0]
        .cpu()
        .numpy()
    )

    predicted_duration = (
        predicted_outputs[
            "predicted_duration"
        ][0]
        .cpu()
        .numpy()
    )

    predicted_mel = (
        predicted_outputs[
            "final_mel"
        ][0]
        .cpu()
        .numpy()
    )

    # ========================================================================
    # PRINT PROSODY TABLE
    # ========================================================================

    print()
    print(
        f"{'Phoneme':12s}"
        f"{'Pitch GT':>12s}"
        f"{'Pitch Pred':>12s}"
        f"{'Energy GT':>12s}"
        f"{'Energy Pred':>12s}"
    )

    print(
        "-" * 64
    )

    for i, phoneme in enumerate(
        record["phonemes"]
    ):

        print(
            f"{phoneme:12s}"
            f"{record['pitch'][i]:12.4f}"
            f"{predicted_pitch[i]:12.4f}"
            f"{record['energy'][i]:12.4f}"
            f"{predicted_energy[i]:12.4f}"
        )

    # ========================================================================
    # PITCH / ENERGY ERRORS
    # ========================================================================

    pitch_valid = (
        np.asarray(
            record["pitch_valid"]
        )
    )

    energy_valid = (
        np.asarray(
            record["energy_valid"]
        )
    )

    if np.any(
        pitch_valid
    ):

        pitch_mae = np.mean(
            np.abs(
                predicted_pitch[
                    pitch_valid
                ]
                - record["pitch"][
                    pitch_valid
                ]
            )
        )

    else:

        pitch_mae = float(
            "nan"
        )

    if np.any(
        energy_valid
    ):

        energy_mae = np.mean(
            np.abs(
                predicted_energy[
                    energy_valid
                ]
                - record["energy"][
                    energy_valid
                ]
            )
        )

    else:

        energy_mae = float(
            "nan"
        )

    print()
    print(
        f"Pitch MAE:  "
        f"{pitch_mae:.6f}"
    )

    print(
        f"Energy MAE: "
        f"{energy_mae:.6f}"
    )

    print(
        f"Teacher-forced Mel MAE: "
        f"{teacher_mae:.6f}"
    )

    print()
    print(
        f"Actual total duration: "
        f"{sum(record['durations'])}"
    )

    print(
        f"Predicted total duration: "
        f"{int(predicted_duration.sum())}"
    )

    # ========================================================================
    # SAVE PLOTS
    # ========================================================================

    save_curve(
        record["pitch"],
        "Target Normalized Log Pitch",
        "Normalized log pitch",
        "target_pitch.png",
        pitch_valid,
    )

    save_curve(
        predicted_pitch,
        "Predicted Normalized Log Pitch",
        "Normalized log pitch",
        "predicted_pitch.png",
        pitch_valid,
    )

    save_curve(
        record["energy"],
        "Target Normalized Log Energy",
        "Normalized log energy",
        "target_energy.png",
        energy_valid,
    )

    save_curve(
        predicted_energy,
        "Predicted Normalized Log Energy",
        "Normalized log energy",
        "predicted_energy.png",
        energy_valid,
    )

    pitch_error = np.zeros_like(
        record["pitch"]
    )

    pitch_error[
        pitch_valid
    ] = np.abs(
        predicted_pitch[
            pitch_valid
        ]
        - record["pitch"][
            pitch_valid
        ]
    )

    energy_error = np.zeros_like(
        record["energy"]
    )

    energy_error[
        energy_valid
    ] = np.abs(
        predicted_energy[
            energy_valid
        ]
        - record["energy"][
            energy_valid
        ]
    )

    save_curve(
        pitch_error,
        "Pitch Prediction Error",
        "Absolute error",
        "pitch_error.png",
        pitch_valid,
    )

    save_curve(
        energy_error,
        "Energy Prediction Error",
        "Absolute error",
        "energy_error.png",
        energy_valid,
    )

    save_mel_image(
        target,
        "Target Mel Spectrogram",
        "target_mel.png",
    )

    save_mel_image(
        predicted_mel,
        "Predicted-Duration/Pitch/Energy Mel",
        "predicted_mel.png",
    )

    return {
        "pitch_mae": pitch_mae,
        "energy_mae": energy_mae,
        "mel_mae": teacher_mae,
    }


# ============================================================================
# SAVE VOCABULARY
# ============================================================================

def save_vocabulary(
    vocabulary,
):

    path = (
        OUTPUT_DIR
        / "vocabulary.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            vocabulary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Vocabulary saved: "
        f"{path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    set_seed()

    ensure_output_dir()

    print("=" * 72)
    print("PHASE 2 — LESSON 60")
    print("PITCH AND ENERGY MODELING")
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
    # STEP 1 — DATA
    # ========================================================================

    records = load_records()

    # ========================================================================
    # STEP 2 — VOCABULARY
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 2 — BUILD VOCABULARY")
    print("=" * 72)

    vocabulary = build_vocabulary(
        records
    )

    print(
        f"Phoneme vocabulary size: "
        f"{len(vocabulary)}"
    )

    save_vocabulary(
        vocabulary
    )

    # ========================================================================
    # DATASET
    # ========================================================================

    dataset, loader = create_loader(
        records,
        vocabulary,
    )

    n_mels = records[0][
        "mel"
    ].shape[1]

    print()
    print(
        f"Training samples: "
        f"{len(dataset)}"
    )

    print(
        f"Mel bins: "
        f"{n_mels}"
    )

    # ========================================================================
    # MODEL
    # ========================================================================

    print()
    print("=" * 72)
    print("MODEL CONFIGURATION")
    print("=" * 72)

    model = PitchEnergyFastSpeech(
        vocab_size=len(vocabulary),
        n_mels=n_mels,
    ).to(
        DEVICE
    )

    print(
        f"Model parameters: "
        f"{count_parameters(model):,}"
    )

    print()
    print(
        "Architecture:"
    )

    print(
        "  Phoneme Embedding"
    )

    print(
        "        ↓"
    )

    print(
        "  Transformer Encoder"
    )

    print(
        "        ├── Duration Predictor"
    )

    print(
        "        ├── Pitch Predictor"
    )

    print(
        "        └── Energy Predictor"
    )

    print(
        "                ↓"
    )

    print(
        "       Variance Embeddings"
    )

    print(
        "                ↓"
    )

    print(
        "         Length Regulator"
    )

    print(
        "                ↓"
    )

    print(
        "        Acoustic Decoder"
    )

    print(
        "                ↓"
    )

    print(
        "           Mel + PostNet"
    )

    # ========================================================================
    # TRAIN
    # ========================================================================

    print()
    print(
        "Starting training..."
    )

    history = train_model(
        model=model,
        loader=loader,
        vocabulary=vocabulary,
        n_mels=n_mels,
    )

    # ========================================================================
    # TRAINING PLOT
    # ========================================================================

    save_training_plot(
        history
    )

    # ========================================================================
    # EVALUATION
    # ========================================================================

    metrics = evaluate_example(
        model=model,
        record=records[0],
        vocabulary=vocabulary,
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 72)
    print("LESSON 60 COMPLETE")
    print("=" * 72)

    print()
    print(
        "Best checkpoint:"
    )

    print(
        OUTPUT_DIR
        / "pitch_energy_model_best.pt"
    )

    print()
    print(
        f"Pitch MAE: "
        f"{metrics['pitch_mae']:.6f}"
    )

    print(
        f"Energy MAE: "
        f"{metrics['energy_mae']:.6f}"
    )

    print(
        f"Teacher-forced Mel MAE: "
        f"{metrics['mel_mae']:.6f}"
    )

    if len(records) == 1:

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "Only one utterance is available."
        )

        print(
            "Therefore the loss demonstrates fitting "
            "the training example, not generalization."
        )

    print()
    print(
        "NEXT:"
    )

    print(
        "Lesson 61 — Neural Vocoder"
    )

    print(
        "The acoustic model now produces Mel spectrograms."
    )

    print(
        "Lesson 61 converts those Mel representations "
        "back into an actual waveform."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()