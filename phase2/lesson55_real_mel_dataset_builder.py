"""
File: lesson55_real_mel_dataset_builder.py

PHASE 2 — LESSON 55
REAL MEL DATASET BUILDER FROM SCRATCH
================================================================

PURPOSE
-------
Build a reusable speech dataset from real audio recordings
and their corresponding transcripts.

NO PRETRAINED MODELS
--------------------
No pretrained TTS models, acoustic models, phonemizers,
Whisper, Wav2Vec2, Librosa, or Torchaudio.

IMPLEMENTED
-----------
1. Dataset discovery
2. Transcript loading and cleaning
3. Audio loading and resampling
4. Stereo-to-mono conversion
5. STFT
6. Mel filterbank
7. Log-Mel spectrogram extraction
8. Feature normalization
9. NPZ feature serialization
10. JSONL and CSV manifests
11. Train/validation splitting
12. Dataset validation
13. Visualization
14. Numerical tests

IMPORTANT NORMALIZATION FIX
---------------------------
Normalization statistics are calculated in float64 to reduce
floating-point rounding errors.

The final normalized features are stored as float32.

INPUT
-----
lesson54_data/
    audio/
        sample.wav
    transcripts/
        sample.txt

OUTPUT
------
lesson55_outputs/
    features/
    manifests/
        metadata.jsonl
        metadata.csv
        train.jsonl
        validation.jsonl
    visualizations/
    dataset_summary.json

INSTALL
-------
pip install numpy scipy soundfile matplotlib

RUN
---
python lesson55_real_mel_dataset_builder.py

NEXT LESSON
-----------
Lesson 56 — Phoneme Dataset Preparation and Encoding
"""

import csv
import json
import math
import random
import re

from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.signal import resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SAMPLE_RATE = 22050

N_FFT = 1024

WIN_LENGTH = 1024

HOP_LENGTH = 256

N_MELS = 80

FMIN = 0.0

FMAX = SAMPLE_RATE / 2

EPSILON = 1e-8

VALIDATION_RATIO = 0.10

NORMALIZE_PER_UTTERANCE = True

INPUT_ROOT = Path("lesson54_data")

INPUT_AUDIO_DIR = INPUT_ROOT / "audio"

INPUT_TRANSCRIPT_DIR = INPUT_ROOT / "transcripts"

METADATA_CSV = INPUT_ROOT / "metadata.csv"

OUTPUT_ROOT = Path("lesson55_outputs")

FEATURE_DIR = OUTPUT_ROOT / "features"

MANIFEST_DIR = OUTPUT_ROOT / "manifests"

VISUALIZATION_DIR = OUTPUT_ROOT / "visualizations"

SUMMARY_PATH = OUTPUT_ROOT / "dataset_summary.json"

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
}


# ============================================================
# STEP 1 — DIRECTORY PREPARATION
# ============================================================

def prepare_directories():

    for directory in [
        FEATURE_DIR,
        MANIFEST_DIR,
        VISUALIZATION_DIR,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("\nOutput directories prepared.")

    print(f"Features:       {FEATURE_DIR}")
    print(f"Manifests:      {MANIFEST_DIR}")
    print(f"Visualizations: {VISUALIZATION_DIR}")


# ============================================================
# STEP 2 — TRANSCRIPT CLEANING
# ============================================================

def clean_transcript(text):

    if not isinstance(text, str):

        raise TypeError(
            "Transcript must be a string."
        )

    text = text.replace(
        "\ufeff",
        "",
    )

    text = text.replace(
        "\x00",
        "",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# STEP 3 — DATASET DISCOVERY
# ============================================================

def discover_audio_transcript_pairs():

    print("\nSTEP 1 — DATASET DISCOVERY")

    pairs = []

    # --------------------------------------------------------
    # CASE A — AUDIO + MATCHING TXT FILE
    # --------------------------------------------------------

    if INPUT_AUDIO_DIR.exists():

        audio_files = sorted(
            path
            for path in INPUT_AUDIO_DIR.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in AUDIO_EXTENSIONS
            )
        )

        print(
            f"Found {len(audio_files)} audio files."
        )

        for audio_path in audio_files:

            transcript_path = (
                INPUT_TRANSCRIPT_DIR
                / f"{audio_path.stem}.txt"
            )

            if not transcript_path.exists():

                print(
                    f"[SKIP] Missing transcript: "
                    f"{audio_path.name}"
                )

                continue

            text = transcript_path.read_text(
                encoding="utf-8-sig"
            )

            text = clean_transcript(
                text
            )

            if not text:

                print(
                    f"[SKIP] Empty transcript: "
                    f"{transcript_path.name}"
                )

                continue

            pairs.append({

                "utterance_id": audio_path.stem,

                "audio_path": audio_path,

                "transcript": text,

                "source": "matching_txt",
            })

    # --------------------------------------------------------
    # CASE B — LJSPEECH-STYLE METADATA
    # --------------------------------------------------------

    if METADATA_CSV.exists():

        print(
            "\nReading metadata CSV..."
        )

        metadata_audio_dir = (
            INPUT_ROOT / "wavs"
        )

        with METADATA_CSV.open(
            "r",
            encoding="utf-8-sig",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                parts = line.split("|")

                if len(parts) < 2:

                    print(
                        f"[SKIP] Invalid metadata line "
                        f"{line_number}"
                    )

                    continue

                utterance_id = parts[0].strip()

                transcript = clean_transcript(
                    parts[1]
                )

                if not transcript:
                    continue

                audio_path = (
                    metadata_audio_dir
                    / f"{utterance_id}.wav"
                )

                if not audio_path.exists():

                    print(
                        f"[SKIP] Missing audio: {audio_path}"
                    )

                    continue

                if any(
                    item["utterance_id"] == utterance_id
                    for item in pairs
                ):
                    continue

                pairs.append({

                    "utterance_id": utterance_id,

                    "audio_path": audio_path,

                    "transcript": transcript,

                    "source": "metadata_csv",
                })

    if not pairs:

        raise RuntimeError(
            "No valid audio/transcript pairs found.\n"
            "Expected audio/sample.wav and "
            "transcripts/sample.txt."
        )

    print(
        f"\nValid audio/transcript pairs: {len(pairs)}"
    )

    return pairs


# ============================================================
# STEP 4 — AUDIO LOADING AND RESAMPLING
# ============================================================

def load_and_resample_audio(audio_path):

    waveform, original_sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
    )

    if waveform.size == 0:

        raise RuntimeError(
            f"Empty audio file: {audio_path}"
        )

    if waveform.ndim == 2:

        waveform = waveform.mean(
            axis=1
        )

    if waveform.ndim != 1:

        raise RuntimeError(
            f"Unsupported audio shape: {waveform.shape}"
        )

    if original_sample_rate <= 0:

        raise RuntimeError(
            "Invalid original sample rate."
        )

    if not np.all(
        np.isfinite(waveform)
    ):

        raise RuntimeError(
            f"Invalid waveform values: {audio_path}"
        )

    if original_sample_rate != SAMPLE_RATE:

        gcd = math.gcd(
            original_sample_rate,
            SAMPLE_RATE,
        )

        up = SAMPLE_RATE // gcd

        down = original_sample_rate // gcd

        waveform = resample_poly(
            waveform,
            up,
            down,
        ).astype(np.float32)

    waveform = waveform.astype(
        np.float32
    )

    duration = (
        len(waveform) / SAMPLE_RATE
    )

    if duration <= 0:

        raise RuntimeError(
            "Audio duration is zero."
        )

    return (
        waveform,
        duration,
        original_sample_rate,
    )


# ============================================================
# STEP 5 — STFT FROM SCRATCH
# ============================================================

def calculate_stft(waveform):

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.ndim != 1:

        raise ValueError(
            "Waveform must be one-dimensional."
        )

    if len(waveform) == 0:

        raise ValueError(
            "Waveform cannot be empty."
        )

    if WIN_LENGTH > N_FFT:

        raise ValueError(
            "WIN_LENGTH cannot exceed N_FFT."
        )

    padding = N_FFT // 2

    padded = np.pad(
        waveform,
        (padding, padding),
        mode="constant",
    )

    n_frames = 1 + (
        len(padded) - N_FFT
    ) // HOP_LENGTH

    window = np.hanning(
        WIN_LENGTH
    ).astype(np.float32)

    if WIN_LENGTH < N_FFT:

        left_padding = (
            N_FFT - WIN_LENGTH
        ) // 2

        right_padding = (
            N_FFT - WIN_LENGTH - left_padding
        )

        window = np.pad(
            window,
            (
                left_padding,
                right_padding,
            ),
        )

    frames = np.zeros(
        (
            n_frames,
            N_FFT,
        ),
        dtype=np.float32,
    )

    for index in range(n_frames):

        start = (
            index * HOP_LENGTH
        )

        frame = padded[
            start:start + N_FFT
        ]

        frames[index] = (
            frame * window
        )

    spectrum = np.fft.rfft(
        frames,
        n=N_FFT,
        axis=1,
    )

    return spectrum


# ============================================================
# STEP 6 — MEL FILTERBANK
# ============================================================

def hz_to_mel(hz):

    return 2595.0 * np.log10(
        1.0 + np.asarray(hz) / 700.0
    )


def mel_to_hz(mel):

    return 700.0 * (
        10.0 ** (
            np.asarray(mel) / 2595.0
        ) - 1.0
    )


def create_mel_filterbank():

    min_mel = hz_to_mel(
        FMIN
    )

    max_mel = hz_to_mel(
        FMAX
    )

    mel_points = np.linspace(
        min_mel,
        max_mel,
        N_MELS + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    fft_frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1,
    )

    filterbank = np.zeros(
        (
            N_MELS,
            len(fft_frequencies),
        ),
        dtype=np.float32,
    )

    for mel_index in range(N_MELS):

        left = hz_points[mel_index]

        center = hz_points[mel_index + 1]

        right = hz_points[mel_index + 2]

        left_mask = (
            (fft_frequencies >= left)
            & (fft_frequencies <= center)
        )

        if center > left:

            filterbank[
                mel_index,
                left_mask,
            ] = (
                fft_frequencies[left_mask] - left
            ) / (
                center - left
            )

        right_mask = (
            (fft_frequencies >= center)
            & (fft_frequencies <= right)
        )

        if right > center:

            filterbank[
                mel_index,
                right_mask,
            ] = (
                right - fft_frequencies[right_mask]
            ) / (
                right - center
            )

    return filterbank


MEL_FILTERBANK = create_mel_filterbank()


# ============================================================
# STEP 7 — LOG-MEL EXTRACTION
# ============================================================

def extract_log_mel(waveform):

    spectrum = calculate_stft(
        waveform
    )

    power_spectrum = (
        np.abs(spectrum) ** 2
    )

    mel_spectrogram = (
        power_spectrum
        @ MEL_FILTERBANK.T
    )

    log_mel = np.log(
        mel_spectrogram + EPSILON
    )

    log_mel = log_mel.astype(
        np.float32
    )

    if not np.all(
        np.isfinite(log_mel)
    ):

        raise RuntimeError(
            "Log-Mel contains NaN or infinite values."
        )

    return log_mel


# ============================================================
# STEP 8 — FEATURE NORMALIZATION
# ============================================================

def normalize_log_mel(log_mel):
    """
    Normalize each Mel channel independently.

    IMPORTANT NUMERICAL FIX
    -----------------------
    The input may be float32.

    Computing mean and standard deviation directly in float32
    can introduce rounding errors.

    We therefore calculate statistics in float64.

    The normalized features are converted back to float32
    after normalization.

    Constant channels are handled safely using EPSILON.
    """

    features = np.asarray(
        log_mel,
        dtype=np.float64,
    )

    if features.ndim != 2:

        raise ValueError(
            "Log-Mel features must be a 2D matrix."
        )

    if features.shape[0] == 0:

        raise ValueError(
            "Cannot normalize an empty feature matrix."
        )

    if not np.all(
        np.isfinite(features)
    ):

        raise ValueError(
            "Input features contain invalid values."
        )

    mean = features.mean(
        axis=0,
        keepdims=True,
        dtype=np.float64,
    )

    std = features.std(
        axis=0,
        keepdims=True,
        dtype=np.float64,
    )

    # Prevent division by zero for constant channels.
    safe_std = np.maximum(
        std,
        EPSILON,
    )

    normalized = (
        features - mean
    ) / safe_std

    if not np.all(
        np.isfinite(normalized)
    ):

        raise RuntimeError(
            "Normalization generated invalid values."
        )

    return normalized.astype(
        np.float32
    )


# ============================================================
# STEP 9 — PROCESS ONE UTTERANCE
# ============================================================

def process_utterance(item):

    utterance_id = item["utterance_id"]

    audio_path = item["audio_path"]

    transcript = item["transcript"]

    print(
        f"\nProcessing: {utterance_id}"
    )

    waveform, duration, original_sr = (
        load_and_resample_audio(
            audio_path
        )
    )

    log_mel = extract_log_mel(
        waveform
    )

    if NORMALIZE_PER_UTTERANCE:

        normalized_log_mel = normalize_log_mel(
            log_mel
        )

    else:

        normalized_log_mel = log_mel.copy()

    feature_path = (
        FEATURE_DIR
        / f"{utterance_id}.npz"
    )

    np.savez_compressed(

        feature_path,

        log_mel=log_mel,

        normalized_log_mel=normalized_log_mel,

        sample_rate=np.int32(SAMPLE_RATE),

        original_sample_rate=np.int32(original_sr),

        hop_length=np.int32(HOP_LENGTH),

        n_fft=np.int32(N_FFT),

        n_mels=np.int32(N_MELS),

        duration=np.float32(duration),

        transcript=np.array(
            transcript
        ),
    )

    frame_count = log_mel.shape[0]

    print(
        f"  Duration: {duration:.3f} sec"
    )

    print(
        f"  Frames:   {frame_count}"
    )

    print(
        f"  Mel shape: {log_mel.shape}"
    )

    print(
        f"  Saved: {feature_path}"
    )

    return {

        "utterance_id": utterance_id,

        "audio_path": str(
            audio_path.resolve()
        ),

        "feature_path": str(
            feature_path.resolve()
        ),

        "transcript": transcript,

        "duration_seconds": float(duration),

        "original_sample_rate": int(original_sr),

        "sample_rate": SAMPLE_RATE,

        "num_samples": int(len(waveform)),

        "num_frames": int(frame_count),

        "num_mels": N_MELS,

        "hop_length": HOP_LENGTH,

        "feature_shape": list(
            log_mel.shape
        ),

        "source": item["source"],
    }


# ============================================================
# STEP 10 — MANIFEST GENERATION
# ============================================================

def save_jsonl(records, output_path):

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )


def save_csv(records, output_path):

    if not records:
        return

    fieldnames = list(
        records[0].keys()
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:

            row = record.copy()

            row["feature_shape"] = json.dumps(
                row["feature_shape"]
            )

            writer.writerow(
                row
            )


def save_manifests(records):

    print(
        "\nSTEP 9 — DATASET MANIFEST"
    )

    jsonl_path = (
        MANIFEST_DIR / "metadata.jsonl"
    )

    csv_path = (
        MANIFEST_DIR / "metadata.csv"
    )

    save_jsonl(
        records,
        jsonl_path,
    )

    save_csv(
        records,
        csv_path,
    )

    print(
        f"JSONL manifest: {jsonl_path}"
    )

    print(
        f"CSV manifest:   {csv_path}"
    )


# ============================================================
# STEP 11 — TRAIN / VALIDATION SPLIT
# ============================================================

def split_dataset(records):

    print(
        "\nSTEP 10 — TRAIN / VALIDATION SPLIT"
    )

    records = list(
        records
    )

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        records
    )

    if len(records) < 2:

        train_records = records

        validation_records = []

    else:

        validation_count = max(
            1,
            int(
                len(records) * VALIDATION_RATIO
            ),
        )

        validation_count = min(
            validation_count,
            len(records) - 1,
        )

        validation_records = records[
            :validation_count
        ]

        train_records = records[
            validation_count:
        ]

    train_path = (
        MANIFEST_DIR / "train.jsonl"
    )

    validation_path = (
        MANIFEST_DIR / "validation.jsonl"
    )

    save_jsonl(
        train_records,
        train_path,
    )

    save_jsonl(
        validation_records,
        validation_path,
    )

    print(
        f"Training samples:   {len(train_records)}"
    )

    print(
        f"Validation samples: {len(validation_records)}"
    )

    if not validation_records:

        print(
            "WARNING: Only one sample is available. "
            "A validation set cannot be created."
        )

    return train_records, validation_records


# ============================================================
# STEP 12 — DATASET VALIDATION
# ============================================================

def validate_dataset(records):

    print(
        "\nSTEP 11 — DATASET VALIDATION"
    )

    if not records:

        raise RuntimeError(
            "Cannot validate an empty dataset."
        )

    total_frames = 0

    total_duration = 0.0

    seen_ids = set()

    for record in records:

        utterance_id = record["utterance_id"]

        if utterance_id in seen_ids:

            raise RuntimeError(
                f"Duplicate utterance ID: {utterance_id}"
            )

        seen_ids.add(
            utterance_id
        )

        feature_path = Path(
            record["feature_path"]
        )

        if not feature_path.exists():

            raise RuntimeError(
                f"Missing feature file: {feature_path}"
            )

        with np.load(
            feature_path,
            allow_pickle=False,
        ) as data:

            log_mel = data["log_mel"]

            normalized = data[
                "normalized_log_mel"
            ]

            assert log_mel.ndim == 2

            assert normalized.ndim == 2

            assert log_mel.shape == normalized.shape

            assert log_mel.shape[1] == N_MELS

            assert np.all(
                np.isfinite(log_mel)
            )

            assert np.all(
                np.isfinite(normalized)
            )

            assert log_mel.shape[0] > 0

            assert len(
                str(data["transcript"].item())
            ) > 0

            total_frames += (
                log_mel.shape[0]
            )

        total_duration += (
            record["duration_seconds"]
        )

    print(
        f"Validated utterances: {len(records)}"
    )

    print(
        f"Total audio duration: {total_duration:.3f} sec"
    )

    print(
        f"Total Mel frames: {total_frames}"
    )

    print(
        "All dataset validation checks passed."
    )

    return {

        "num_utterances": len(records),

        "total_duration_seconds": total_duration,

        "total_mel_frames": total_frames,

        "num_mels": N_MELS,

        "sample_rate": SAMPLE_RATE,
    }


# ============================================================
# STEP 13 — DATASET SUMMARY
# ============================================================

def save_dataset_summary(
    records,
    train_records,
    validation_records,
    validation_results,
):

    durations = np.array(
        [
            record["duration_seconds"]
            for record in records
        ],
        dtype=np.float64,
    )

    frame_counts = np.array(
        [
            record["num_frames"]
            for record in records
        ],
        dtype=np.int64,
    )

    summary = {

        "lesson": 55,

        "dataset_name": "Real Mel Dataset",

        "num_utterances": len(records),

        "num_training_samples": len(train_records),

        "num_validation_samples": len(validation_records),

        "sample_rate": SAMPLE_RATE,

        "n_fft": N_FFT,

        "win_length": WIN_LENGTH,

        "hop_length": HOP_LENGTH,

        "n_mels": N_MELS,

        "normalization": (
            "per_utterance_float64_statistics"
            if NORMALIZE_PER_UTTERANCE
            else "none"
        ),

        "total_audio_duration_seconds": float(
            durations.sum()
        ),

        "average_audio_duration_seconds": float(
            durations.mean()
        ),

        "minimum_audio_duration_seconds": float(
            durations.min()
        ),

        "maximum_audio_duration_seconds": float(
            durations.max()
        ),

        "total_mel_frames": int(
            frame_counts.sum()
        ),

        "average_mel_frames": float(
            frame_counts.mean()
        ),

        "feature_dimension": N_MELS,

        "pretrained_models_used": False,

        "validation": validation_results,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        f"\nDataset summary saved: {SUMMARY_PATH}"
    )

    return summary


# ============================================================
# STEP 14 — VISUALIZATION
# ============================================================

def visualize_sample(record):

    print(
        "\nSTEP 12 — VISUALIZATION"
    )

    feature_path = Path(
        record["feature_path"]
    )

    audio_path = Path(
        record["audio_path"]
    )

    waveform, duration, _ = (
        load_and_resample_audio(
            audio_path
        )
    )

    with np.load(
        feature_path,
        allow_pickle=False,
    ) as data:

        log_mel = data["log_mel"]

        normalized = data[
            "normalized_log_mel"
        ]

    waveform_times = (
        np.arange(len(waveform))
        / SAMPLE_RATE
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 12),
    )

    axes[0].plot(
        waveform_times,
        waveform,
        linewidth=0.5,
    )

    axes[0].set_title(
        "Real Speech Waveform"
    )

    axes[0].set_xlabel(
        "Time (seconds)"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[0].grid(
        True,
        alpha=0.25,
    )

    axes[1].imshow(
        log_mel.T,
        origin="lower",
        aspect="auto",
        extent=[
            0,
            duration,
            0,
            N_MELS,
        ],
    )

    axes[1].set_title(
        "Log-Mel Spectrogram"
    )

    axes[1].set_xlabel(
        "Time (seconds)"
    )

    axes[1].set_ylabel(
        "Mel bins"
    )

    axes[2].imshow(
        normalized.T,
        origin="lower",
        aspect="auto",
        extent=[
            0,
            duration,
            0,
            N_MELS,
        ],
    )

    axes[2].set_title(
        "Normalized Log-Mel Spectrogram"
    )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    axes[2].set_ylabel(
        "Mel bins"
    )

    plt.suptitle(
        f"Utterance: {record['utterance_id']}\n"
        f"Transcript: {record['transcript'][:100]}",
        fontsize=11,
    )

    plt.tight_layout()

    output_path = (
        VISUALIZATION_DIR
        / f"{record['utterance_id']}_mel.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.show()

    print(
        f"Visualization saved: {output_path}"
    )


# ============================================================
# STEP 15 — NUMERICAL TESTS
# ============================================================

def run_tests():

    print(
        "\nRunning numerical tests..."
    )

    # --------------------------------------------------------
    # TEST 1 — MEL CONVERSION
    # --------------------------------------------------------

    frequencies = np.array([
        0,
        100,
        500,
        1000,
        4000,
    ])

    reconstructed = mel_to_hz(
        hz_to_mel(frequencies)
    )

    assert np.allclose(
        frequencies,
        reconstructed,
        atol=1e-5,
    )

    print(
        "  [PASS] Hz/Mel conversion"
    )

    # --------------------------------------------------------
    # TEST 2 — FILTERBANK SHAPE
    # --------------------------------------------------------

    assert MEL_FILTERBANK.shape == (
        N_MELS,
        N_FFT // 2 + 1,
    )

    assert np.all(
        MEL_FILTERBANK >= 0
    )

    print(
        "  [PASS] Mel filterbank shape"
    )

    # --------------------------------------------------------
    # TEST 3 — FILTERBANK FINITE
    # --------------------------------------------------------

    assert np.all(
        np.isfinite(MEL_FILTERBANK)
    )

    print(
        "  [PASS] Mel filterbank numerical stability"
    )

    # --------------------------------------------------------
    # TEST 4 — STFT
    # --------------------------------------------------------

    test_signal = np.sin(
        2 * np.pi * 440
        * np.arange(SAMPLE_RATE)
        / SAMPLE_RATE
    ).astype(np.float32)

    spectrum = calculate_stft(
        test_signal
    )

    assert spectrum.ndim == 2

    assert spectrum.shape[1] == (
        N_FFT // 2 + 1
    )

    print(
        "  [PASS] STFT dimensions"
    )

    # --------------------------------------------------------
    # TEST 5 — LOG-MEL SHAPE
    # --------------------------------------------------------

    log_mel = extract_log_mel(
        test_signal
    )

    assert log_mel.ndim == 2

    assert log_mel.shape[1] == N_MELS

    print(
        "  [PASS] Log-Mel dimensions"
    )

    # --------------------------------------------------------
    # TEST 6 — LOG-MEL FINITE
    # --------------------------------------------------------

    assert np.all(
        np.isfinite(log_mel)
    )

    print(
        "  [PASS] Log-Mel numerical stability"
    )

    # --------------------------------------------------------
    # TEST 7 — NORMALIZATION
    #
    # FIX:
    # Use float64 statistics and a tolerance appropriate
    # for float32 output.
    # --------------------------------------------------------

    normalized = normalize_log_mel(
        log_mel
    )

    assert normalized.shape == log_mel.shape

    assert normalized.dtype == np.float32

    assert np.all(
        np.isfinite(normalized)
    )

    normalized_mean = normalized.astype(
        np.float64
    ).mean(
        axis=0
    )

    normalized_std = normalized.astype(
        np.float64
    ).std(
        axis=0
    )

    assert np.allclose(
        normalized_mean,
        0.0,
        atol=1e-4,
    ), (
        "Normalized feature means are not sufficiently close to zero. "
        f"Maximum absolute mean: {np.max(np.abs(normalized_mean))}"
    )

    # Constant channels are allowed to have zero standard deviation.
    original_std = log_mel.astype(
        np.float64
    ).std(
        axis=0
    )

    variable_channels = (
        original_std > EPSILON
    )

    if np.any(variable_channels):

        assert np.allclose(
            normalized_std[variable_channels],
            1.0,
            atol=1e-4,
        ), (
            "Normalized standard deviations are not close to one."
        )

    print(
        "  [PASS] Feature normalization"
    )

    # --------------------------------------------------------
    # TEST 8 — CONSTANT CHANNEL NORMALIZATION
    # --------------------------------------------------------

    constant_features = np.ones(
        (20, N_MELS),
        dtype=np.float32,
    )

    constant_normalized = normalize_log_mel(
        constant_features
    )

    assert np.all(
        np.isfinite(constant_normalized)
    )

    assert np.allclose(
        constant_normalized,
        0.0,
        atol=1e-6,
    )

    print(
        "  [PASS] Constant-channel normalization"
    )

    # --------------------------------------------------------
    # TEST 9 — TRANSCRIPT CLEANING
    # --------------------------------------------------------

    dirty_text = (
        "  This   is\n\n a   test.  "
    )

    clean_text = clean_transcript(
        dirty_text
    )

    assert clean_text == (
        "This is a test."
    )

    print(
        "  [PASS] Transcript cleaning"
    )

    # --------------------------------------------------------
    # TEST 10 — FRAME RATE
    # --------------------------------------------------------

    frame_rate = (
        SAMPLE_RATE / HOP_LENGTH
    )

    assert np.isclose(
        frame_rate,
        86.1328125,
    )

    print(
        "  [PASS] Mel frame rate"
    )

    # --------------------------------------------------------
    # TEST 11 — FEATURE SAVE/LOAD
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_path = (
        OUTPUT_ROOT / "test_features.npz"
    )

    np.savez_compressed(
        test_path,
        log_mel=log_mel,
        normalized_log_mel=normalized,
    )

    with np.load(
        test_path,
        allow_pickle=False,
    ) as data:

        loaded = data["log_mel"]

        assert np.allclose(
            loaded,
            log_mel,
        )

    test_path.unlink()

    print(
        "  [PASS] Feature serialization"
    )

    print(
        "\nAll numerical tests passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    print("=" * 72)

    print(
        "PHASE 2 — LESSON 55"
    )

    print(
        "REAL MEL DATASET BUILDER"
    )

    print("=" * 72)

    print(
        f"Sample rate: {SAMPLE_RATE}"
    )

    print(
        f"FFT size:    {N_FFT}"
    )

    print(
        f"Hop length:  {HOP_LENGTH}"
    )

    print(
        f"Mel bins:    {N_MELS}"
    )

    print(
        f"Frame rate:  {SAMPLE_RATE / HOP_LENGTH:.3f} FPS"
    )

    # --------------------------------------------------------
    # STEP 1 — DIRECTORIES
    # --------------------------------------------------------

    prepare_directories()

    # --------------------------------------------------------
    # STEP 2 — NUMERICAL TESTS
    # --------------------------------------------------------

    run_tests()

    # --------------------------------------------------------
    # STEP 3 — DATASET DISCOVERY
    # --------------------------------------------------------

    pairs = discover_audio_transcript_pairs()

    # --------------------------------------------------------
    # STEP 4 — PROCESS UTTERANCES
    # --------------------------------------------------------

    print(
        "\nSTEP 2–8 — FEATURE EXTRACTION"
    )

    records = []

    failed = []

    for item in pairs:

        try:

            record = process_utterance(
                item
            )

            records.append(
                record
            )

        except Exception as error:

            print(
                f"\n[ERROR] {item['utterance_id']}: {error}"
            )

            failed.append({

                "utterance_id": item["utterance_id"],

                "error": str(error),
            })

    if not records:

        raise RuntimeError(
            "All utterances failed during processing."
        )

    # --------------------------------------------------------
    # STEP 5 — SAVE MANIFESTS
    # --------------------------------------------------------

    save_manifests(
        records
    )

    # --------------------------------------------------------
    # STEP 6 — TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    train_records, validation_records = (
        split_dataset(records)
    )

    # --------------------------------------------------------
    # STEP 7 — VALIDATE DATASET
    # --------------------------------------------------------

    validation_results = validate_dataset(
        records
    )

    # --------------------------------------------------------
    # STEP 8 — SAVE SUMMARY
    # --------------------------------------------------------

    summary = save_dataset_summary(
        records,
        train_records,
        validation_records,
        validation_results,
    )

    summary["failed_utterances"] = failed

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # STEP 9 — VISUALIZE FIRST SAMPLE
    # --------------------------------------------------------

    visualize_sample(
        records[0]
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "LESSON 55 COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        f"Processed samples:  {len(records)}"
    )

    print(
        f"Failed samples:     {len(failed)}"
    )

    print(
        f"Training samples:   {len(train_records)}"
    )

    print(
        f"Validation samples: {len(validation_records)}"
    )

    print(
        f"Total duration:     "
        f"{summary['total_audio_duration_seconds']:.3f} sec"
    )

    print(
        f"Total Mel frames:   "
        f"{summary['total_mel_frames']}"
    )

    print(
        "\nOUTPUT FILES"
    )

    print(
        f"Features:       {FEATURE_DIR}"
    )

    print(
        f"Metadata JSONL: {MANIFEST_DIR / 'metadata.jsonl'}"
    )

    print(
        f"Metadata CSV:   {MANIFEST_DIR / 'metadata.csv'}"
    )

    print(
        f"Training:       {MANIFEST_DIR / 'train.jsonl'}"
    )

    print(
        f"Validation:     {MANIFEST_DIR / 'validation.jsonl'}"
    )

    print(
        f"Summary:        {SUMMARY_PATH}"
    )

    print(
        "\nNo pretrained models were used."
    )

    print(
        "\nNEXT LESSON:"
    )

    print(
        "Lesson 56 — Phoneme Dataset Preparation and Encoding"
    )


if __name__ == "__main__":

    main()