"""
=======================================================================
FILE: phase1_lesson8_dataset_engineering.py

PHASE 1 — LESSON 8
SPEECH DATASET ENGINEERING FOR VOICE-CLONING TTS
=======================================================================

PURPOSE
-------

Build a complete dataset preparation and validation system for
a voice-cloning Text-to-Speech (TTS) project.

This program teaches the engineering principles required to construct
a high-quality speech dataset before training a neural TTS model.

The program supports:

    1. Dataset directory initialization.
    2. Dataset metadata generation.
    3. Audio file validation.
    4. Transcript validation.
    5. Speaker metadata validation.
    6. Audio quality analysis.
    7. Duplicate detection.
    8. Dataset splitting.
    9. Dataset statistics.
    10. Visualization.
    11. JSON and CSV reporting.
    12. Recording guidelines.

=======================================================================
PART 1 — WHY DATASET ENGINEERING MATTERS
=======================================================================

A neural TTS system learns a relationship between text and speech.

The training dataset consists of paired examples:

    (text, audio)

For example:

    Text:
        Hello, how are you?

    Audio:
        hello_0001.wav

The model learns:

    Text -> Speech

For voice cloning, we also need speaker identity:

    (text, audio, speaker_id)

The model learns:

    Text + Speaker Identity -> Speech

A poor dataset produces poor speech synthesis, even when the
neural network architecture is excellent.

Common dataset problems include:

    - Incorrect transcripts.
    - Incorrect speaker labels.
    - Noisy recordings.
    - Inconsistent sample rates.
    - Incorrect audio channels.
    - Clipped speech.
    - Excessive silence.
    - Duplicate recordings.
    - Missing audio files.
    - Incorrect metadata.
    - Training/test data leakage.

Dataset engineering is therefore a fundamental part of TTS development.

=======================================================================
PART 2 — DATASET DIRECTORY STRUCTURE
=======================================================================

The program creates:

voice_tts_dataset/
|
|-- metadata.csv
|
|-- speakers.csv
|
|-- dataset_config.json
|
|-- README.md
|
|-- wavs/
|   |
|   |-- speaker_001_0001.wav
|   |-- speaker_001_0002.wav
|   |-- speaker_001_0003.wav
|   |
|   |-- speaker_002_0001.wav
|   |-- speaker_002_0002.wav
|
|-- splits/
|   |
|   |-- train.csv
|   |-- validation.csv
|   |-- test.csv
|
|-- reports/
    |
    |-- validation_report.json
    |-- validation_report.csv
    |-- dataset_statistics.json
    |-- dataset_statistics.png

=======================================================================
PART 3 — METADATA FORMAT
=======================================================================

metadata.csv:

    audio_file,text,speaker_id

Example:

    speaker_001_0001.wav,Hello there.,speaker_001
    speaker_001_0002.wav,How are you?,speaker_001
    speaker_002_0001.wav,Good morning.,speaker_002

Every metadata row represents one utterance.

The audio_file field identifies the WAV recording.

The text field contains the exact spoken words.

The speaker_id field identifies the person who spoke.

IMPORTANT:

The transcript must match the actual audio.

For example:

    Transcript:
        I want to go home.

    Actual recording:
        I wanted to go home.

This is a transcript error.

A TTS model trained on incorrect pairs learns an incorrect
relationship between text and acoustic speech.

=======================================================================
PART 4 — AUDIO FUNDAMENTALS
=======================================================================

A digital audio recording is represented by samples:

    x[n]

The sample rate determines how many samples are recorded per second.

For example:

    24000 Hz = 24000 samples per second.

The duration is:

    duration = number_of_samples / sample_rate

For mono audio:

    audio.shape = (number_of_samples,)

For stereo audio:

    audio.shape = (number_of_samples, 2)

TTS datasets commonly use mono recordings.

The WAV subtype determines the stored sample representation.

For example:

    PCM_16 = 16-bit integer PCM
    PCM_24 = 24-bit integer PCM
    FLOAT  = 32-bit floating-point audio

This educational dataset configuration targets:

    Sample rate: 24000 Hz
    Channels:    1
    Format:      WAV
    Encoding:    PCM_16

These are configurable targets, not universal requirements.
Some TTS architectures use 22050 Hz, 44100 Hz, or other rates.

=======================================================================
PART 5 — AUDIO QUALITY METRICS
=======================================================================

1. PEAK AMPLITUDE

    peak = max(abs(x[n]))

Peak amplitude close to 1.0 indicates that the signal is near
the maximum representable normalized amplitude.

2. RMS AMPLITUDE

    RMS = sqrt(mean(x[n]^2))

RMS measures average signal energy.

3. CLIPPING

Clipping happens when the signal exceeds the recording system's
maximum representable amplitude.

A clipped waveform loses information.

4. SILENCE RATIO

We divide the waveform into short frames.

For each frame:

    frame_rms = sqrt(mean(frame^2))

A frame is considered silent when:

    frame_rms < silence_threshold

Silence ratio:

    silence_ratio = silent_samples / total_samples

This is a simple energy-based estimate.

It is NOT a speech activity detector.

A quiet speaker or an unvoiced consonant may also have low energy.

5. DURATION

Very short recordings may not contain enough information.

Very long recordings may be difficult to align and process.

For this educational dataset, we use:

    Minimum duration: 0.5 seconds
    Recommended maximum: 15 seconds

These values can be changed according to the TTS architecture.

=======================================================================
PART 6 — DUPLICATE DETECTION
=======================================================================

The program calculates SHA-256 hashes.

    hash = SHA256(file_bytes)

If two files have identical hashes, their file contents are identical.

IMPORTANT:

This detects byte-identical files.

It does not detect acoustically identical recordings stored using
different encodings or metadata.

Perceptual audio duplicate detection requires additional techniques.

=======================================================================
PART 7 — DATASET SPLITTING
=======================================================================

A dataset is divided into:

    Training set
    Validation set
    Test set

Example:

    Training:   80%
    Validation: 10%
    Testing:    10%

Training:
    Used to optimize neural network parameters.

Validation:
    Used to monitor training and tune hyperparameters.

Testing:
    Used for final evaluation.

IMPORTANT:

Random utterance-level splitting can cause data leakage.

For example, if two recordings from the same recording session
are nearly identical, placing one in training and another in testing
can make evaluation misleading.

This implementation groups exact normalized transcripts before
splitting, so repeated identical sentences are kept together.

However, this is NOT a complete speaker-disjoint or session-disjoint
split.

For a multi-speaker voice-cloning system, consider additional
evaluation protocols:

    - Speaker-disjoint evaluation.
    - Recording-session-disjoint evaluation.
    - Text-disjoint evaluation.
    - Unseen-speaker evaluation.

For single-speaker voice cloning, session-disjoint evaluation is
particularly useful.

=======================================================================
PART 8 — IMPORTANT LIMITATION
=======================================================================

This program validates utterance-level text/audio pairing.

It does NOT perform forced alignment.

Forced alignment estimates where phonemes or words occur in an
audio recording, given a known transcript.

That is a separate process involving acoustic models and alignment
algorithms.

=======================================================================
USAGE
=======================================================================

Initialize dataset:

    python dataset_engineering.py --init

Validate dataset:

    python dataset_engineering.py --validate

Generate dataset splits:

    python dataset_engineering.py --split

Generate statistics and plots:

    python dataset_engineering.py --plot

Run everything:

    python dataset_engineering.py --all

Display recording guidelines:

    python dataset_engineering.py --guidelines

=======================================================================
"""

import os
import csv
import json
import argparse
import hashlib
import random
import re

from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# =====================================================================
# SECTION 1 — CONFIGURATION
# =====================================================================

DATASET_DIR = Path("voice_tts_dataset")

WAV_DIR = DATASET_DIR / "wavs"
SPLIT_DIR = DATASET_DIR / "splits"
REPORT_DIR = DATASET_DIR / "reports"

METADATA_FILE = DATASET_DIR / "metadata.csv"
SPEAKERS_FILE = DATASET_DIR / "speakers.csv"
CONFIG_FILE = DATASET_DIR / "dataset_config.json"

SAMPLE_RATE = 24000
CHANNELS = 1
SUBTYPE = "PCM_16"

MIN_DURATION = 0.5
MAX_DURATION = 15.0

SILENCE_THRESHOLD = 0.005
CLIPPING_THRESHOLD = 0.999

FRAME_LENGTH_MS = 20

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav"
}


# =====================================================================
# SECTION 2 — DATASET INITIALIZATION
# =====================================================================

def initialize_dataset():

    """
    Create the complete dataset directory structure.

    Existing user recordings and metadata are preserved.

    Missing template files are created.
    """

    print("\nInitializing TTS dataset...\n")

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # metadata.csv
    # ---------------------------------------------------------------

    if not METADATA_FILE.exists():

        with open(
            METADATA_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "audio_file",
                "text",
                "speaker_id"
            ])

        print("Created metadata.csv")

    # ---------------------------------------------------------------
    # speakers.csv
    # ---------------------------------------------------------------

    if not SPEAKERS_FILE.exists():

        with open(
            SPEAKERS_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "speaker_id",
                "description",
                "language",
                "recording_notes"
            ])

        print("Created speakers.csv")

    # ---------------------------------------------------------------
    # dataset_config.json
    # ---------------------------------------------------------------

    if not CONFIG_FILE.exists():

        config = {

            "dataset_name": "voice_tts_dataset",

            "audio": {

                "sample_rate": SAMPLE_RATE,

                "channels": CHANNELS,

                "format": "WAV",

                "subtype": SUBTYPE

            },

            "validation": {

                "minimum_duration_seconds": MIN_DURATION,

                "maximum_duration_seconds": MAX_DURATION,

                "silence_threshold": SILENCE_THRESHOLD,

                "clipping_threshold": CLIPPING_THRESHOLD

            },

            "splits": {

                "train": TRAIN_RATIO,

                "validation": VALIDATION_RATIO,

                "test": TEST_RATIO

            },

            "description": (
                "Speech dataset for educational voice-cloning "
                "TTS development."
            )

        }

        with open(
            CONFIG_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                config,
                f,
                indent=4,
                ensure_ascii=False
            )

        print("Created dataset_config.json")

    # ---------------------------------------------------------------
    # README.md
    # ---------------------------------------------------------------

    readme_path = DATASET_DIR / "README.md"

    if not readme_path.exists():

        readme = """
# Voice Cloning TTS Dataset

## Directory structure

- metadata.csv: audio/transcript/speaker mapping
- speakers.csv: speaker information
- dataset_config.json: dataset configuration
- wavs/: audio recordings
- splits/: train/validation/test CSV files
- reports/: dataset validation reports

## Audio requirements

Recommended educational configuration:

- WAV
- 24 kHz
- Mono
- PCM 16-bit

## Metadata format

audio_file,text,speaker_id

Example:

speaker_001_0001.wav,Hello there.,speaker_001

## Recording rules

1. Record in a quiet environment.
2. Use consistent microphone placement.
3. Avoid clipping.
4. Speak naturally.
5. Use accurate transcripts.
6. Keep each recording associated with one speaker.
7. Do not mix speakers within an utterance.
"""

        readme_path.write_text(
            readme.strip(),
            encoding="utf-8"
        )

        print("Created README.md")

    print("\nDataset initialized successfully.")

    print(f"\nDataset directory: {DATASET_DIR.resolve()}")

    print("\nPlace your WAV recordings inside:")

    print(WAV_DIR.resolve())


# =====================================================================
# SECTION 3 — METADATA LOADING
# =====================================================================

def load_metadata():

    """
    Load metadata.csv.

    Returns a list of dictionaries:

        [
            {
                "audio_file": "...",
                "text": "...",
                "speaker_id": "..."
            }
        ]
    """

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "metadata.csv not found. Run --init first."
        )

    rows = []

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "audio_file",
            "text",
            "speaker_id"
        }

        if not reader.fieldnames:

            raise ValueError("metadata.csv is empty.")

        missing = required_columns - set(reader.fieldnames)

        if missing:

            raise ValueError(
                f"Missing metadata columns: {missing}"
            )

        for row in reader:

            rows.append({

                "audio_file": (
                    row.get("audio_file") or ""
                ).strip(),

                "text": (
                    row.get("text") or ""
                ).strip(),

                "speaker_id": (
                    row.get("speaker_id") or ""
                ).strip()

            })

    return rows


# =====================================================================
# SECTION 4 — SPEAKER METADATA VALIDATION
# =====================================================================

def load_speakers():

    """
    Read speakers.csv.

    Returns the set of declared speaker IDs.

    A speaker should be declared once in speakers.csv.
    """

    if not SPEAKERS_FILE.exists():

        return set()

    speaker_ids = set()

    with open(
        SPEAKERS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:

            return speaker_ids

        if "speaker_id" not in reader.fieldnames:

            return speaker_ids

        for row in reader:

            speaker_id = (
                row.get("speaker_id") or ""
            ).strip()

            if speaker_id:

                speaker_ids.add(speaker_id)

    return speaker_ids


# =====================================================================
# SECTION 5 — AUDIO HASHING
# =====================================================================

def calculate_sha256(file_path):

    """
    Calculate SHA-256 hash of a file.

    SHA-256 is a cryptographic hash function.

    The output is a fixed-length digest representing file contents.

    Identical files have identical hashes.

    This is useful for detecting exact duplicate recordings.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


# =====================================================================
# SECTION 6 — AUDIO STATISTICS
# =====================================================================

def calculate_audio_statistics(audio, sample_rate):

    """
    Calculate basic audio quality measurements.

    Parameters
    ----------

    audio:
        NumPy array containing audio samples.

    sample_rate:
        Samples per second.

    Returns
    -------

    Dictionary of audio statistics.
    """

    audio = np.asarray(audio)

    if audio.ndim == 1:

        mono_audio = audio

    else:

        mono_audio = np.mean(
            audio,
            axis=1
        )

    if len(mono_audio) == 0:

        raise ValueError("Audio contains no samples.")

    # ---------------------------------------------------------------
    # Duration
    # ---------------------------------------------------------------

    duration = len(mono_audio) / sample_rate

    # ---------------------------------------------------------------
    # Peak amplitude
    # ---------------------------------------------------------------

    peak = float(
        np.max(np.abs(audio))
    )

    # ---------------------------------------------------------------
    # RMS amplitude
    # ---------------------------------------------------------------

    rms = float(
        np.sqrt(np.mean(audio ** 2))
    )

    # ---------------------------------------------------------------
    # Clipping ratio
    # ---------------------------------------------------------------

    clipping_ratio = float(
        np.mean(
            np.abs(audio) >= CLIPPING_THRESHOLD
        )
    )

    # ---------------------------------------------------------------
    # Frame-based silence detection
    # ---------------------------------------------------------------

    frame_length = max(
        1,
        int(sample_rate * FRAME_LENGTH_MS / 1000)
    )

    num_frames = len(mono_audio) // frame_length

    silent_frames = 0

    for i in range(num_frames):

        start = i * frame_length

        end = start + frame_length

        frame = mono_audio[start:end]

        frame_rms = np.sqrt(
            np.mean(frame ** 2)
        )

        if frame_rms < SILENCE_THRESHOLD:

            silent_frames += 1

    if num_frames > 0:

        silence_ratio = (
            silent_frames / num_frames
        )

    else:

        silence_ratio = 0.0

    # ---------------------------------------------------------------
    # Peak in dBFS
    # ---------------------------------------------------------------

    peak_dbfs = 20 * np.log10(
        max(peak, 1e-12)
    )

    # ---------------------------------------------------------------
    # RMS in dBFS
    # ---------------------------------------------------------------

    rms_dbfs = 20 * np.log10(
        max(rms, 1e-12)
    )

    return {

        "duration": float(duration),

        "sample_rate": int(sample_rate),

        "channels": (
            1 if audio.ndim == 1 else audio.shape[1]
        ),

        "peak": peak,

        "rms": rms,

        "peak_dbfs": float(peak_dbfs),

        "rms_dbfs": float(rms_dbfs),

        "clipping_ratio": clipping_ratio,

        "silence_ratio": float(silence_ratio)

    }


# =====================================================================
# SECTION 7 — AUDIO VALIDATION
# =====================================================================

def validate_audio_file(file_path):

    """
    Validate one WAV file.

    Checks:

        1. File existence.
        2. Audio readability.
        3. Sample rate.
        4. Number of channels.
        5. Duration.
        6. Clipping.
        7. Silence ratio.
        8. Audio subtype.

    A warning does not necessarily mean the recording is unusable.

    For example, a different sample rate may be acceptable if the
    dataset is deliberately designed for another configuration.
    """

    result = {

        "audio_file": file_path.name,

        "status": "PASS",

        "errors": [],

        "warnings": [],

        "duration": None,

        "sample_rate": None,

        "channels": None,

        "subtype": None,

        "peak": None,

        "rms": None,

        "peak_dbfs": None,

        "rms_dbfs": None,

        "clipping_ratio": None,

        "silence_ratio": None,

        "sha256": None

    }

    if not file_path.exists():

        result["status"] = "FAIL"

        result["errors"].append(
            "File does not exist."
        )

        return result

    try:

        # -----------------------------------------------------------
        # Read audio metadata and samples.
        # always_2d=True ensures consistent array shape.
        # -----------------------------------------------------------

        audio, sample_rate = sf.read(
            str(file_path),
            always_2d=True,
            dtype="float32"
        )

        info = sf.info(str(file_path))

        result["sample_rate"] = int(sample_rate)

        result["channels"] = int(audio.shape[1])

        result["subtype"] = info.subtype

        # -----------------------------------------------------------
        # Sample rate validation
        # -----------------------------------------------------------

        if sample_rate != SAMPLE_RATE:

            result["warnings"].append(
                f"Expected {SAMPLE_RATE} Hz, "
                f"found {sample_rate} Hz."
            )

        # -----------------------------------------------------------
        # Channel validation
        # -----------------------------------------------------------

        if audio.shape[1] != CHANNELS:

            result["warnings"].append(
                f"Expected {CHANNELS} channel(s), "
                f"found {audio.shape[1]}."
            )

        # -----------------------------------------------------------
        # Subtype validation
        # -----------------------------------------------------------

        if info.subtype != SUBTYPE:

            result["warnings"].append(
                f"Expected {SUBTYPE}, "
                f"found {info.subtype}."
            )

        # -----------------------------------------------------------
        # Statistics
        # -----------------------------------------------------------

        stats = calculate_audio_statistics(
            audio,
            sample_rate
        )

        result.update(stats)

        # -----------------------------------------------------------
        # Duration validation
        # -----------------------------------------------------------

        if stats["duration"] < MIN_DURATION:

            result["errors"].append(
                f"Duration too short: "
                f"{stats['duration']:.2f}s."
            )

        if stats["duration"] > MAX_DURATION:

            result["warnings"].append(
                f"Long recording: "
                f"{stats['duration']:.2f}s."
            )

        # -----------------------------------------------------------
        # Clipping validation
        # -----------------------------------------------------------

        if stats["clipping_ratio"] > 0:

            result["warnings"].append(
                "Possible clipping detected."
            )

        # -----------------------------------------------------------
        # Silence validation
        # -----------------------------------------------------------

        if stats["silence_ratio"] > 0.50:

            result["warnings"].append(
                "More than 50% of frames classified as silent."
            )

        # -----------------------------------------------------------
        # Duplicate detection hash
        # -----------------------------------------------------------

        result["sha256"] = calculate_sha256(
            file_path
        )

        # -----------------------------------------------------------
        # Final status
        # -----------------------------------------------------------

        if result["errors"]:

            result["status"] = "FAIL"

        elif result["warnings"]:

            result["status"] = "WARNING"

        else:

            result["status"] = "PASS"

    except Exception as e:

        result["status"] = "FAIL"

        result["errors"].append(
            f"Audio processing error: {str(e)}"
        )

    return result


# =====================================================================
# SECTION 8 — TEXT NORMALIZATION FOR DUPLICATE GROUPING
# =====================================================================

def normalize_text(text):

    """
    Normalize transcript text for exact-text grouping.

    This is NOT a complete linguistic text normalizer.

    It performs basic operations:

        - Lowercase conversion.
        - Whitespace normalization.
        - Basic punctuation removal.

    It does not convert:

        1,000 -> one thousand

    or:

        Dr. Smith -> Doctor Smith

    Such transformations require language-aware text normalization.

    This function is only used to group identical normalized
    transcripts for dataset splitting.
    """

    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# =====================================================================
# SECTION 9 — DATASET VALIDATION
# =====================================================================

def validate_dataset():

    """
    Validate every metadata entry and associated audio file.

    Checks:

        - Missing transcript.
        - Missing speaker ID.
        - Missing WAV file.
        - Audio quality.
        - Duplicate metadata entries.
        - Duplicate audio content.
        - Undeclared speaker IDs.

    Generates:

        reports/validation_report.json

        reports/validation_report.csv
    """

    print("\nValidating dataset...\n")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = load_metadata()

    declared_speakers = load_speakers()

    report = []

    seen_audio_hashes = {}

    seen_metadata_keys = set()

    for index, row in enumerate(rows):

        audio_file = row["audio_file"]

        text = row["text"]

        speaker_id = row["speaker_id"]

        print(
            f"[{index + 1}/{len(rows)}] "
            f"Checking {audio_file}"
        )

        result = {

            "row": index + 2,

            "audio_file": audio_file,

            "text": text,

            "speaker_id": speaker_id,

            "status": "PASS",

            "errors": [],

            "warnings": []

        }

        # -----------------------------------------------------------
        # Metadata validation
        # -----------------------------------------------------------

        if not audio_file:

            result["errors"].append(
                "Missing audio filename."
            )

        if not text:

            result["errors"].append(
                "Missing transcript."
            )

        if not speaker_id:

            result["errors"].append(
                "Missing speaker ID."
            )

        # -----------------------------------------------------------
        # Duplicate metadata validation
        # -----------------------------------------------------------

        metadata_key = (
            audio_file,
            text,
            speaker_id
        )

        if metadata_key in seen_metadata_keys:

            result["errors"].append(
                "Duplicate metadata row."
            )

        seen_metadata_keys.add(
            metadata_key
        )

        # -----------------------------------------------------------
        # Speaker validation
        # -----------------------------------------------------------

        if (
            speaker_id
            and declared_speakers
            and speaker_id not in declared_speakers
        ):

            result["warnings"].append(
                "Speaker ID not declared in speakers.csv."
            )

        # -----------------------------------------------------------
        # Prevent metadata paths from escaping the WAV directory.
        # Metadata should contain filenames, not arbitrary paths.
        # -----------------------------------------------------------

        candidate = Path(audio_file)

        if (
            candidate.is_absolute()
            or candidate.name != audio_file
            or audio_file in {".", ".."}
        ):

            result["errors"].append(
                "Invalid audio filename. "
                "Use a filename located inside wavs/."
            )

            result["status"] = "FAIL"

            report.append(result)

            continue

        file_path = WAV_DIR / audio_file

        # -----------------------------------------------------------
        # Check file extension
        # -----------------------------------------------------------

        if file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:

            result["errors"].append(
                "Unsupported audio extension."
            )

        # -----------------------------------------------------------
        # Audio file validation
        # -----------------------------------------------------------

        audio_result = validate_audio_file(
            file_path
        )

        result.update({

            "duration": audio_result.get("duration"),

            "sample_rate": audio_result.get("sample_rate"),

            "channels": audio_result.get("channels"),

            "subtype": audio_result.get("subtype"),

            "peak_dbfs": audio_result.get("peak_dbfs"),

            "rms_dbfs": audio_result.get("rms_dbfs"),

            "clipping_ratio": audio_result.get(
                "clipping_ratio"
            ),

            "silence_ratio": audio_result.get(
                "silence_ratio"
            )

        })

        result["errors"].extend(
            audio_result["errors"]
        )

        result["warnings"].extend(
            audio_result["warnings"]
        )

        # -----------------------------------------------------------
        # Duplicate audio detection
        # -----------------------------------------------------------

        audio_hash = audio_result.get("sha256")

        if audio_hash:

            if audio_hash in seen_audio_hashes:

                original = seen_audio_hashes[audio_hash]

                result["warnings"].append(
                    f"Byte-identical duplicate of {original}."
                )

            else:

                seen_audio_hashes[audio_hash] = audio_file

        # -----------------------------------------------------------
        # Final status
        # -----------------------------------------------------------

        if result["errors"]:

            result["status"] = "FAIL"

        elif result["warnings"]:

            result["status"] = "WARNING"

        else:

            result["status"] = "PASS"

        report.append(result)

    # =================================================================
    # REPORT SUMMARY
    # =================================================================

    summary = {

        "total_utterances": len(report),

        "passed": sum(
            r["status"] == "PASS"
            for r in report
        ),

        "warnings": sum(
            r["status"] == "WARNING"
            for r in report
        ),

        "failed": sum(
            r["status"] == "FAIL"
            for r in report
        )

    }

    final_report = {

        "summary": summary,

        "configuration": {

            "sample_rate": SAMPLE_RATE,

            "channels": CHANNELS,

            "subtype": SUBTYPE,

            "minimum_duration": MIN_DURATION,

            "maximum_duration": MAX_DURATION

        },

        "results": report

    }

    # ---------------------------------------------------------------
    # Save JSON report
    # ---------------------------------------------------------------

    json_path = REPORT_DIR / "validation_report.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_report,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ---------------------------------------------------------------
    # Save CSV report
    # ---------------------------------------------------------------

    csv_path = REPORT_DIR / "validation_report.csv"

    fields = [

        "row",

        "audio_file",

        "speaker_id",

        "status",

        "duration",

        "sample_rate",

        "channels",

        "subtype",

        "peak_dbfs",

        "rms_dbfs",

        "clipping_ratio",

        "silence_ratio",

        "errors",

        "warnings"

    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for item in report:

            row_data = {}

            for field in fields:

                value = item.get(field, "")

                if field in {
                    "errors",
                    "warnings"
                }:

                    value = " | ".join(value)

                row_data[field] = value

            writer.writerow(row_data)

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------

    print("\n======================================")

    print("DATASET VALIDATION SUMMARY")

    print("======================================")

    print(f"Total:   {summary['total_utterances']}")

    print(f"Passed:  {summary['passed']}")

    print(f"Warnings:{summary['warnings']}")

    print(f"Failed:  {summary['failed']}")

    print("======================================")

    print(f"\nJSON report: {json_path}")

    print(f"CSV report:  {csv_path}")

    return report


# =====================================================================
# SECTION 10 — DATASET SPLITTING
# =====================================================================

def generate_dataset_splits():

    """
    Generate train, validation, and test CSV files.

    This implementation groups rows by normalized exact transcript.

    All occurrences of the same normalized transcript are assigned
    to the same split.

    This reduces exact-text leakage between splits.

    LIMITATION:

    This does not guarantee speaker-disjoint or session-disjoint
    splits.

    It is a simple baseline for learning dataset engineering.
    """

    print("\nGenerating dataset splits...\n")

    rows = load_metadata()

    if len(rows) < 3:

        raise ValueError(
            "At least three metadata entries are needed."
        )

    if not np.isclose(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO,
        1.0
    ):

        raise ValueError(
            "Dataset split ratios must sum to 1.0."
        )

    # ---------------------------------------------------------------
    # Group by normalized transcript.
    # ---------------------------------------------------------------

    groups = defaultdict(list)

    for row in rows:

        key = normalize_text(
            row["text"]
        )

        if not key:

            raise ValueError(
                f"Cannot split empty normalized transcript: "
                f"{row['audio_file']}"
            )

        groups[key].append(row)

    group_keys = list(groups.keys())

    random.Random(
        RANDOM_SEED
    ).shuffle(group_keys)

    # ---------------------------------------------------------------
    # Assign complete transcript groups to splits.
    # ---------------------------------------------------------------

    total_utterances = len(rows)

    target_train = (
        total_utterances * TRAIN_RATIO
    )

    target_validation = (
        total_utterances * VALIDATION_RATIO
    )

    train_rows = []

    validation_rows = []

    test_rows = []

    for key in group_keys:

        group = groups[key]

        if (
            len(train_rows) < target_train
            or not train_rows
        ):

            train_rows.extend(group)

        elif (
            len(validation_rows) < target_validation
            or not validation_rows
        ):

            validation_rows.extend(group)

        else:

            test_rows.extend(group)

    # ---------------------------------------------------------------
    # Prevent empty partitions when possible.
    # ---------------------------------------------------------------

    if not validation_rows or not test_rows:

        print(
            "\nWARNING: Transcript grouping produced an empty split."
        )

        print(
            "The dataset may contain too few unique transcripts."
        )

        print(
            "Consider collecting more distinct sentences."
        )

    # ---------------------------------------------------------------
    # Save splits.
    # ---------------------------------------------------------------

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    def save_split(name, data):

        path = SPLIT_DIR / f"{name}.csv"

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "audio_file",
                    "text",
                    "speaker_id"
                ]
            )

            writer.writeheader()

            writer.writerows(data)

        return path

    train_path = save_split(
        "train",
        train_rows
    )

    validation_path = save_split(
        "validation",
        validation_rows
    )

    test_path = save_split(
        "test",
        test_rows
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print("\n======================================")

    print("DATASET SPLIT SUMMARY")

    print("======================================")

    print(
        f"Total:      {total_utterances}"
    )

    print(
        f"Train:      {len(train_rows)}"
    )

    print(
        f"Validation: {len(validation_rows)}"
    )

    print(
        f"Test:       {len(test_rows)}"
    )

    print("======================================")

    print(f"\n{train_path}")

    print(validation_path)

    print(test_path)

    return {

        "train": train_rows,

        "validation": validation_rows,

        "test": test_rows

    }


# =====================================================================
# SECTION 11 — DATASET STATISTICS
# =====================================================================

def calculate_dataset_statistics():

    """
    Calculate overall dataset statistics.

    Metrics:

        - Total utterances.
        - Total duration.
        - Number of speakers.
        - Number of unique transcripts.
        - Mean duration.
        - Median duration.
        - Duration standard deviation.
        - Mean RMS.
        - Mean silence ratio.
        - Mean clipping ratio.
        - Sample rate distribution.
        - Speaker distribution.
    """

    print("\nCalculating dataset statistics...\n")

    rows = load_metadata()

    durations = []

    rms_values = []

    silence_values = []

    clipping_values = []

    sample_rates = Counter()

    speaker_counts = Counter()

    transcript_counts = Counter()

    failed_files = []

    for row in rows:

        audio_file = row["audio_file"]

        speaker_id = row["speaker_id"]

        file_path = WAV_DIR / audio_file

        result = validate_audio_file(
            file_path
        )

        if result["status"] == "FAIL":

            failed_files.append(audio_file)

            continue

        durations.append(
            result["duration"]
        )

        rms_values.append(
            result["rms"]
        )

        silence_values.append(
            result["silence_ratio"]
        )

        clipping_values.append(
            result["clipping_ratio"]
        )

        sample_rates[
            result["sample_rate"]
        ] += 1

        speaker_counts[
            speaker_id
        ] += 1

        transcript_counts[
            normalize_text(row["text"])
        ] += 1

    if not durations:

        raise ValueError(
            "No valid audio files available for statistics."
        )

    durations = np.asarray(
        durations
    )

    rms_values = np.asarray(
        rms_values
    )

    silence_values = np.asarray(
        silence_values
    )

    clipping_values = np.asarray(
        clipping_values
    )

    statistics = {

        "total_utterances": len(rows),

        "valid_utterances": len(durations),

        "failed_utterances": len(failed_files),

        "total_duration_seconds": float(
            np.sum(durations)
        ),

        "total_duration_hours": float(
            np.sum(durations) / 3600
        ),

        "mean_duration_seconds": float(
            np.mean(durations)
        ),

        "median_duration_seconds": float(
            np.median(durations)
        ),

        "duration_std_seconds": float(
            np.std(durations)
        ),

        "minimum_duration_seconds": float(
            np.min(durations)
        ),

        "maximum_duration_seconds": float(
            np.max(durations)
        ),

        "mean_rms": float(
            np.mean(rms_values)
        ),

        "mean_silence_ratio": float(
            np.mean(silence_values)
        ),

        "mean_clipping_ratio": float(
            np.mean(clipping_values)
        ),

        "number_of_speakers": len(
            speaker_counts
        ),

        "number_of_unique_transcripts": len(
            transcript_counts
        ),

        "speaker_distribution": dict(
            speaker_counts
        ),

        "sample_rate_distribution": dict(
            sample_rates
        ),

        "failed_files": failed_files

    }

    # ---------------------------------------------------------------
    # Save statistics
    # ---------------------------------------------------------------

    output_path = (
        REPORT_DIR / "dataset_statistics.json"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            statistics,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n======================================")

    print("DATASET STATISTICS")

    print("======================================")

    print(
        f"Total utterances: "
        f"{statistics['total_utterances']}"
    )

    print(
        f"Valid utterances: "
        f"{statistics['valid_utterances']}"
    )

    print(
        f"Total duration: "
        f"{statistics['total_duration_hours']:.2f} hours"
    )

    print(
        f"Mean duration: "
        f"{statistics['mean_duration_seconds']:.2f} seconds"
    )

    print(
        f"Number of speakers: "
        f"{statistics['number_of_speakers']}"
    )

    print(
        f"Unique transcripts: "
        f"{statistics['number_of_unique_transcripts']}"
    )

    print(
        f"Mean RMS: "
        f"{statistics['mean_rms']:.4f}"
    )

    print(
        f"Mean silence ratio: "
        f"{statistics['mean_silence_ratio']:.2%}"
    )

    print(
        f"Mean clipping ratio: "
        f"{statistics['mean_clipping_ratio']:.4%}"
    )

    print("======================================")

    print(f"\nSaved: {output_path}")

    return statistics


# =====================================================================
# SECTION 12 — VISUALIZATION
# =====================================================================

def visualize_dataset():

    """
    Generate dataset quality visualizations.

    Figures:

        1. Recording duration distribution.
        2. RMS distribution.
        3. Silence ratio distribution.
        4. Speaker distribution.
        5. Sample rate distribution.

    These plots help identify dataset imbalance and recording
    quality problems.
    """

    print("\nGenerating dataset visualizations...\n")

    rows = load_metadata()

    durations = []

    rms_values = []

    silence_values = []

    speaker_counts = Counter()

    sample_rate_counts = Counter()

    for row in rows:

        file_path = WAV_DIR / row["audio_file"]

        result = validate_audio_file(
            file_path
        )

        if result["status"] == "FAIL":

            continue

        durations.append(
            result["duration"]
        )

        rms_values.append(
            result["rms"]
        )

        silence_values.append(
            result["silence_ratio"]
        )

        speaker_counts[
            row["speaker_id"]
        ] += 1

        sample_rate_counts[
            result["sample_rate"]
        ] += 1

    if not durations:

        print(
            "No valid audio recordings available for visualization."
        )

        return

    # ---------------------------------------------------------------
    # 1. Duration distribution
    # ---------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        durations,
        bins=30
    )

    plt.xlabel("Duration (seconds)")

    plt.ylabel("Number of recordings")

    plt.title("Speech Recording Duration Distribution")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "duration_distribution.png",
        dpi=150
    )

    # ---------------------------------------------------------------
    # 2. RMS distribution
    # ---------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        rms_values,
        bins=30
    )

    plt.xlabel("RMS amplitude")

    plt.ylabel("Number of recordings")

    plt.title("Audio RMS Distribution")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "rms_distribution.png",
        dpi=150
    )

    # ---------------------------------------------------------------
    # 3. Silence distribution
    # ---------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        silence_values,
        bins=30
    )

    plt.xlabel("Silence ratio")

    plt.ylabel("Number of recordings")

    plt.title("Silence Ratio Distribution")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "silence_distribution.png",
        dpi=150
    )

    # ---------------------------------------------------------------
    # 4. Speaker distribution
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 5))

    speaker_names = list(
        speaker_counts.keys()
    )

    speaker_values = list(
        speaker_counts.values()
    )

    plt.bar(
        speaker_names,
        speaker_values
    )

    plt.xlabel("Speaker ID")

    plt.ylabel("Number of utterances")

    plt.title("Speaker Dataset Distribution")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "speaker_distribution.png",
        dpi=150
    )

    # ---------------------------------------------------------------
    # 5. Sample rate distribution
    # ---------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sample_rate_names = [
        str(rate)
        for rate in sample_rate_counts.keys()
    ]

    sample_rate_values = list(
        sample_rate_counts.values()
    )

    plt.bar(
        sample_rate_names,
        sample_rate_values
    )

    plt.xlabel("Sample rate (Hz)")

    plt.ylabel("Number of recordings")

    plt.title("Sample Rate Distribution")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "sample_rate_distribution.png",
        dpi=150
    )

    plt.show()

    print(
        f"\nVisualizations saved in {REPORT_DIR.resolve()}"
    )


# =====================================================================
# SECTION 13 — RECORDING GUIDELINES
# =====================================================================

def show_recording_guidelines():

    """
    Display practical instructions for collecting a TTS dataset.
    """

    guidelines = """

=======================================================================
VOICE-CLONING TTS DATA COLLECTION GUIDELINES
=======================================================================

1. RECORDING ENVIRONMENT

    - Use a quiet room.
    - Avoid fans and air conditioners where possible.
    - Avoid background music.
    - Avoid echo and strong room reverberation.
    - Maintain a consistent recording environment.

2. MICROPHONE

    - Use the same microphone throughout a recording session.
    - Maintain a consistent microphone distance.
    - Avoid microphone handling noise.
    - Avoid excessive microphone gain.

3. AUDIO CONFIGURATION

    Recommended educational configuration:

        Sample rate: 24000 Hz
        Channels:    Mono
        Format:      WAV
        Encoding:    PCM 16-bit

    Record at a higher quality if the equipment supports it.
    Resampling and format conversion can be performed later.

4. SPEAKING STYLE

    - Speak naturally.
    - Avoid shouting.
    - Avoid whispering unless specifically required.
    - Maintain a consistent speaking style.
    - Do not rush.
    - Avoid unnatural pauses.

5. TRANSCRIPTS

    - Write exactly what was spoken.
    - Preserve spoken words.
    - Do not silently correct grammatical mistakes.
    - Use a consistent punctuation convention.
    - Do not include words that were not spoken.

6. FILE NAMING

    Example:

        speaker_001_0001.wav
        speaker_001_0002.wav
        speaker_001_0003.wav

        speaker_002_0001.wav
        speaker_002_0002.wav

7. METADATA

    Example metadata.csv:

        audio_file,text,speaker_id

        speaker_001_0001.wav,Hello there.,speaker_001

        speaker_001_0002.wav,How are you?,speaker_001

        speaker_002_0001.wav,Good morning.,speaker_002

8. SPEAKER INFORMATION

    Maintain a speakers.csv file.

    Each speaker should have a stable speaker ID.

    Do not reuse the same ID for different people.

9. DATASET QUALITY

    Review:

        - Incorrect transcripts.
        - Missing recordings.
        - Duplicate recordings.
        - Excessive silence.
        - Clipping.
        - Background noise.
        - Inconsistent microphone conditions.

10. DATASET SPLITTING

    Create:

        Training set
        Validation set
        Test set

    For multi-speaker systems, plan speaker-disjoint evaluation
    when evaluating generalization to unseen speakers.

11. CONSENT AND RIGHTS

    Obtain explicit permission from each speaker to record and use
    their voice for the intended TTS and voice-cloning purposes.

    Follow applicable privacy, data protection, and voice-use laws.

=======================================================================

"""

    print(guidelines)


# =====================================================================
# SECTION 14 — COMMAND LINE INTERFACE
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Speech dataset engineering system "
            "for voice-cloning TTS."
        )

    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize dataset directory."
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset."
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="Generate dataset splits."
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate statistics and visualizations."
    )

    parser.add_argument(
        "--guidelines",
        action="store_true",
        help="Display recording guidelines."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run initialization, validation, splitting, "
             "statistics, and visualization."
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Default behavior
    # ---------------------------------------------------------------

    if not any(vars(args).values()):

        parser.print_help()

        return

    # ---------------------------------------------------------------
    # Initialize
    # ---------------------------------------------------------------

    if args.init or args.all:

        initialize_dataset()

    # ---------------------------------------------------------------
    # Recording guidelines
    # ---------------------------------------------------------------

    if args.guidelines:

        show_recording_guidelines()

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    if args.validate or args.all:

        validate_dataset()

    # ---------------------------------------------------------------
    # Split
    # ---------------------------------------------------------------

    if args.split or args.all:

        generate_dataset_splits()

    # ---------------------------------------------------------------
    # Statistics and visualization
    # ---------------------------------------------------------------

    if args.plot or args.all:

        calculate_dataset_statistics()

        visualize_dataset()

    print("\nProcessing completed.")


# =====================================================================
# SECTION 15 — PROGRAM ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()


"""
=======================================================================
END OF LESSON 8
=======================================================================

WHAT YOU HAVE BUILT
-------------------

You now have a complete educational speech dataset engineering tool.

It can:

    - Initialize a TTS dataset.
    - Read speech metadata.
    - Validate audio recordings.
    - Calculate audio quality metrics.
    - Detect exact duplicate audio files.
    - Check transcript and speaker metadata.
    - Generate basic dataset splits.
    - Calculate dataset statistics.
    - Visualize dataset characteristics.
    - Generate quality reports.

IMPORTANT NEXT STEP
-------------------

The next stage is not immediately training a large TTS model.

First, we need to understand how text and speech are connected
at a more detailed level.

In particular:

    Text
      |
      v
    Text normalization
      |
      v
    Grapheme-to-phoneme conversion
      |
      v
    Phoneme sequence
      |
      v
    Acoustic representation
      |
      v
    Speech waveform

We also need to understand forced alignment:

    Transcript + Audio
           |
           v
      Acoustic model
           |
           v
      Phoneme alignment
           |
           v
      Phoneme/word timestamps

These concepts become important when preparing data for
modern neural TTS architectures.

=======================================================================
"""