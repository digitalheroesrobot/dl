"""
File: lesson54_forced_alignment.py

PHASE 2 — LESSON 54
PHONEME-TO-AUDIO ALIGNMENT FROM SCRATCH
================================================================

PURPOSE
-------
Learn how to align a known phoneme sequence with an audio recording.

NO PRETRAINED MODELS
--------------------
This implementation does NOT use:

    - Montreal Forced Aligner
    - Whisper
    - Wav2Vec2
    - Pretrained acoustic models
    - Pretrained phonemizers
    - Pretrained TTS models

WE IMPLEMENT OURSELVES
----------------------
    1. Sample dataset downloader
    2. Audio loading and resampling
    3. Short-Time Fourier Transform (STFT)
    4. Mel filterbank
    5. Log-Mel feature extraction
    6. Handcrafted phoneme acoustic prototypes
    7. Acoustic distance calculation
    8. Segmental Dynamic Programming
    9. Phoneme boundary reconstruction
   10. Alignment visualization
   11. Alignment JSON export
   12. Numerical tests

WHAT YOU LEARN
--------------

STEP 1 — Dataset Preparation
    Download speech audio and its transcript.

STEP 2 — Audio Processing
    Load, convert to mono, and resample audio.

STEP 3 — STFT
    Convert waveform segments into frequency-domain features.

STEP 4 — Mel Filterbank
    Convert linear frequency bins into Mel frequency bins.

STEP 5 — Log-Mel Features
    Extract frame-level acoustic representations.

STEP 6 — Acoustic Prototypes
    Create simplified phoneme templates manually.

STEP 7 — Acoustic Distance
    Compare audio frames against phoneme templates.

STEP 8 — Dynamic Programming
    Find a sequence of contiguous phoneme intervals.

STEP 9 — Alignment Export
    Save phoneme boundaries and durations to JSON.

STEP 10 — Visualization
    Visualize waveform, log-Mel features, and phoneme intervals.

STEP 11 — Numerical Tests
    Verify feature extraction and alignment correctness.

IMPORTANT LIMITATION
--------------------
This is an educational baseline aligner.

The sample audio is downloaded from an external source.
The phoneme sequence is manually prepared.
The acoustic prototypes are handcrafted approximations.

This is NOT a production-quality phoneme aligner.

The next important improvement is to train our own acoustic
phoneme classifier using labeled speech data.

SAMPLE DATASET
--------------
Automatically downloads:

    sample.wav
    sample transcript

INSTALLATION
------------
    pip install numpy scipy soundfile matplotlib

RUN
---
    python lesson54_forced_alignment.py

OUTPUT
------
    lesson54_data/
        audio/sample.wav
        transcripts/sample.txt

    lesson54_outputs/
        sample_alignment.json
        sample_alignment.png

NEXT LESSON
-----------
Lesson 55 — Real Mel Dataset Builder
"""

import json
import math
import re
import urllib.request
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

HOP_LENGTH = 256

N_FFT = 1024

WIN_LENGTH = 400

N_MELS = 40

EPSILON = 1e-8

DATA_DIR = Path("lesson54_data")

AUDIO_DIR = DATA_DIR / "audio"

TRANSCRIPT_DIR = DATA_DIR / "transcripts"

OUTPUT_DIR = Path("lesson54_outputs")

AUDIO_PATH = AUDIO_DIR / "sample.wav"

TRANSCRIPT_PATH = TRANSCRIPT_DIR / "sample.txt"

JSON_PATH = OUTPUT_DIR / "sample_alignment.json"

PLOT_PATH = OUTPUT_DIR / "sample_alignment.png"

AUDIO_URL = (
    "https://www.samplefiles.online/files/"
    "sample-speech-short.wav"
)

TRANSCRIPT_URL = (
    "https://www.samplefiles.online/files/"
    "sample-speech-transcript.txt"
)

EXPECTED_TRANSCRIPT = (
    "This sample voice recording is used for testing "
    "speech recognition systems."
)


# ------------------------------------------------------------
# MANUALLY PREPARED ARPABET PHONEME SEQUENCE
# ------------------------------------------------------------

PHONEME_SEQUENCE = [
    "DH", "IH", "S",
    "S", "AE", "M", "P", "AH", "L",
    "V", "OY", "S",
    "R", "IH", "K", "AO", "R", "D", "IH", "NG",
    "IH", "Z",
    "Y", "UW", "Z", "D",
    "F", "AO", "R",
    "T", "EH", "S", "T", "IH", "NG",
    "S", "P", "IY", "CH",
    "R", "EH", "K", "AH", "G", "N", "IH", "SH", "AH", "N",
    "S", "IH", "S", "T", "AH", "M", "Z",
]


# ============================================================
# STEP 1 — DOWNLOAD SAMPLE DATASET
# ============================================================

def download_file(
    url,
    destination,
    min_bytes=1,
):
    """
    Download a file with configurable minimum size.

    WAV:
        min_bytes=1000

    TXT:
        min_bytes=1

    A temporary file prevents incomplete downloads.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reuse an existing valid-size file.
    if (
        destination.exists()
        and destination.stat().st_size >= min_bytes
    ):

        print(
            f"Already downloaded: {destination}"
        )

        return

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    print(f"\nDownloading:\n{url}")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Lesson54-Research"
        },
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:

            with open(
                temporary_path,
                "wb",
            ) as output:

                while True:

                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    output.write(chunk)

        downloaded_size = (
            temporary_path.stat().st_size
        )

        if downloaded_size < min_bytes:

            raise RuntimeError(
                f"Downloaded file is too small: "
                f"{downloaded_size} bytes. "
                f"Minimum required: {min_bytes} bytes."
            )

        temporary_path.replace(
            destination
        )

        print(
            f"Downloaded: {destination} "
            f"({downloaded_size:,} bytes)"
        )

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        raise


def prepare_sample_dataset():
    """
    Download the WAV and transcript.

    Validate the transcript before alignment.
    """

    print("\nSTEP 1 — SAMPLE DATASET")

    # --------------------------------------------------------
    # Download audio.
    # --------------------------------------------------------

    download_file(
        AUDIO_URL,
        AUDIO_PATH,
        min_bytes=1000,
    )

    # --------------------------------------------------------
    # Download transcript.
    #
    # IMPORTANT:
    # TXT files can legitimately be smaller than 1000 bytes.
    # --------------------------------------------------------

    download_file(
        TRANSCRIPT_URL,
        TRANSCRIPT_PATH,
        min_bytes=1,
    )

    # --------------------------------------------------------
    # Validate downloaded audio.
    # --------------------------------------------------------

    try:

        audio_info = sf.info(
            str(AUDIO_PATH)
        )

        if audio_info.frames <= 0:
            raise RuntimeError(
                "Downloaded WAV contains no audio frames."
            )

        print(
            f"Audio format: {audio_info.format}"
        )

        print(
            f"Audio duration: {audio_info.duration:.3f} sec"
        )

    except Exception as error:

        raise RuntimeError(
            "Downloaded audio is not a valid readable WAV file. "
            "Check the audio URL."
        ) from error

    # --------------------------------------------------------
    # Read transcript.
    # --------------------------------------------------------

    transcript_content = TRANSCRIPT_PATH.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not transcript_content:

        raise RuntimeError(
            "Downloaded transcript is empty."
        )

    # Detect HTML error pages.
    if re.match(
        r"^\s*(<!doctype\s+html|<html)",
        transcript_content,
        re.IGNORECASE,
    ):

        raise RuntimeError(
            "The transcript URL returned HTML instead of text. "
            "Check the transcript URL."
        )

    # --------------------------------------------------------
    # Extract transcript.
    #
    # Supports:
    #     "This is a sentence."
    #
    # Also supports:
    #     This is a sentence.
    # --------------------------------------------------------

    match = re.search(
        r'"([^"]+)"',
        transcript_content,
    )

    if match:

        transcript = match.group(1).strip()

    else:

        transcript = transcript_content.strip()

    # --------------------------------------------------------
    # Validate transcript.
    # --------------------------------------------------------

    if transcript.lower() != EXPECTED_TRANSCRIPT.lower():

        raise RuntimeError(
            "Downloaded transcript does not match "
            "the manually prepared phoneme sequence.\n\n"
            f"Downloaded:\n{transcript}\n\n"
            f"Expected:\n{EXPECTED_TRANSCRIPT}"
        )

    print(
        f"\nTranscript: {transcript}"
    )

    return transcript


# ============================================================
# STEP 2 — LOAD AND RESAMPLE AUDIO
# ============================================================

def load_audio(audio_path):
    """
    Load WAV, convert stereo to mono, and resample.

    Target sample rate: 22050 Hz.
    """

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    waveform, original_sample_rate = sf.read(
        audio_path,
        dtype="float32",
    )

    if waveform.ndim == 2:

        waveform = waveform.mean(
            axis=1
        )

    if len(waveform) == 0:

        raise RuntimeError(
            "Audio waveform is empty."
        )

    if original_sample_rate <= 0:

        raise ValueError(
            "Invalid original sample rate."
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

    if not np.all(
        np.isfinite(waveform)
    ):

        raise RuntimeError(
            "Audio contains NaN or infinite values."
        )

    peak = np.max(
        np.abs(waveform)
    )

    if peak < EPSILON:

        raise RuntimeError(
            "Audio waveform contains only silence."
        )

    waveform /= (
        peak + EPSILON
    )

    duration = (
        len(waveform) / SAMPLE_RATE
    )

    print("\nAudio information:")

    print(
        f"Original sample rate: {original_sample_rate}"
    )

    print(
        f"Target sample rate:   {SAMPLE_RATE}"
    )

    print(
        f"Duration:             {duration:.3f} sec"
    )

    print(
        f"Samples:              {len(waveform)}"
    )

    return waveform


# ============================================================
# STEP 3 — STFT FROM SCRATCH
# ============================================================

def stft(
    waveform,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
):
    """
    Calculate the Short-Time Fourier Transform.

    Operations:
        1. Center padding
        2. Frame extraction
        3. Hann window
        4. FFT

    Output:
        [time_frames, frequency_bins]
    """

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.ndim != 1:

        raise ValueError(
            "STFT input must be a 1D waveform."
        )

    if n_fft <= 0:
        raise ValueError("n_fft must be positive.")

    if win_length <= 0:
        raise ValueError("win_length must be positive.")

    if win_length > n_fft:

        raise ValueError(
            "win_length cannot be greater than n_fft."
        )

    if hop_length <= 0:

        raise ValueError(
            "hop_length must be positive."
        )

    if len(waveform) == 0:

        raise ValueError(
            "Waveform cannot be empty."
        )

    half_window = win_length // 2

    padded = np.pad(
        waveform,
        (half_window, half_window),
    )

    n_frames = int(
        np.ceil(
            len(waveform) / hop_length
        )
    )

    window = np.hanning(
        win_length
    ).astype(np.float32)

    output = []

    for frame_index in range(n_frames):

        start = (
            frame_index * hop_length
        )

        frame = padded[
            start:start + win_length
        ]

        if len(frame) < win_length:

            frame = np.pad(
                frame,
                (0, win_length - len(frame)),
            )

        frame = frame * window

        if win_length < n_fft:

            frame = np.pad(
                frame,
                (0, n_fft - win_length),
            )

        spectrum = np.fft.rfft(
            frame,
            n=n_fft,
        )

        output.append(
            spectrum
        )

    return np.asarray(output)


# ============================================================
# STEP 4 — MEL FILTERBANK
# ============================================================

def hz_to_mel(hz):

    hz = np.asarray(hz)

    return 2595.0 * np.log10(
        1.0 + hz / 700.0
    )


def mel_to_hz(mel):

    mel = np.asarray(mel)

    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def create_mel_filterbank():

    min_mel = hz_to_mel(
        0
    )

    max_mel = hz_to_mel(
        SAMPLE_RATE / 2
    )

    mel_points = np.linspace(
        min_mel,
        max_mel,
        N_MELS + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1,
    )

    filterbank = np.zeros(
        (
            N_MELS,
            N_FFT // 2 + 1,
        ),
        dtype=np.float32,
    )

    for m in range(N_MELS):

        left = hz_points[m]

        center = hz_points[m + 1]

        right = hz_points[m + 2]

        rising = (
            (frequencies >= left)
            & (frequencies <= center)
        )

        falling = (
            (frequencies >= center)
            & (frequencies <= right)
        )

        if center > left:

            filterbank[m, rising] = (
                frequencies[rising] - left
            ) / (
                center - left
            )

        if right > center:

            filterbank[m, falling] = (
                right - frequencies[falling]
            ) / (
                right - center
            )

    return filterbank


MEL_FILTERBANK = create_mel_filterbank()


# ============================================================
# STEP 5 — LOG-MEL FEATURE EXTRACTION
# ============================================================

def normalize_feature_vectors(features):
    """
    Normalize each frame across Mel bins.

    This emphasizes spectral shape rather than absolute volume.
    """

    mean = features.mean(
        axis=1,
        keepdims=True,
    )

    std = features.std(
        axis=1,
        keepdims=True,
    )

    return (
        features - mean
    ) / (
        std + EPSILON
    )


def extract_log_mel(waveform):

    spectrum = stft(
        waveform
    )

    power = np.abs(
        spectrum
    ) ** 2

    mel = (
        power @ MEL_FILTERBANK.T
    )

    log_mel = np.log(
        mel + EPSILON
    )

    features = normalize_feature_vectors(
        log_mel.astype(np.float32)
    )

    if not np.all(
        np.isfinite(features)
    ):

        raise RuntimeError(
            "Log-Mel features contain invalid values."
        )

    return features


# ============================================================
# STEP 6 — HANDCRAFTED PHONEME ACOUSTIC PROTOTYPES
# ============================================================

"""
These are not pretrained embeddings.

We define approximate acoustic characteristics manually.

Vowels:
    Represented by approximate formant peaks.

Fricatives:
    Represented by high-frequency noise regions.

Nasals:
    Represented by lower-frequency resonances.

Stops:
    Represented by broad transient-like spectral patterns.

Liquids and glides:
    Represented by approximate formant structures.

IMPORTANT:
    These handcrafted templates are educational approximations.
    They are not learned from real phoneme recordings.
"""

VOWEL_FORMANTS = {

    "AE": [700, 1700, 2600],

    "AH": [650, 1400, 2500],

    "AO": [570, 900, 2500],

    "EH": [550, 1850, 2600],

    "IH": [400, 2100, 2800],

    "IY": [300, 2300, 3000],

    "UW": [350, 900, 2200],

    "OY": [500, 1100, 2500],
}


FRICATIVE_CENTERS = {

    "S": 6000,

    "SH": 4000,

    "F": 5000,

    "V": 4000,

    "Z": 5000,

    "DH": 2500,
}


NASAL_CENTERS = {

    "M": 300,

    "N": 400,

    "NG": 500,
}


STOP_CENTERS = {

    "P": 2500,

    "T": 4500,

    "K": 3500,

    "D": 2000,

    "G": 1800,

    "CH": 4000,
}


LIQUID_FORMANTS = {

    "L": [400, 1100, 2800],

    "R": [450, 1300, 1700],

    "Y": [300, 2200, 3000],
}


def gaussian(x, center, width):

    return np.exp(
        -0.5 * ((x - center) / width) ** 2
    )


def mel_bin_center_frequencies():

    mel_points = np.linspace(
        hz_to_mel(0),
        hz_to_mel(SAMPLE_RATE / 2),
        N_MELS + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    return (
        hz_points[:-2]
        + hz_points[1:-1]
        + hz_points[2:]
    ) / 3.0


MEL_CENTERS = mel_bin_center_frequencies()


def build_phone_prototype(phone):
    """
    Construct one handcrafted acoustic prototype.

    Output:
        [N_MELS]
    """

    spectrum = np.zeros(
        N_MELS,
        dtype=np.float64,
    )

    if phone in VOWEL_FORMANTS:

        formants = VOWEL_FORMANTS[phone]

        for formant in formants:

            spectrum += gaussian(
                MEL_CENTERS,
                formant,
                250,
            )

    elif phone in FRICATIVE_CENTERS:

        center = FRICATIVE_CENTERS[phone]

        spectrum += gaussian(
            MEL_CENTERS,
            center,
            1800,
        )

        spectrum += 0.1

    elif phone in NASAL_CENTERS:

        center = NASAL_CENTERS[phone]

        spectrum += gaussian(
            MEL_CENTERS,
            center,
            450,
        )

        spectrum += gaussian(
            MEL_CENTERS,
            1500,
            500,
        )

    elif phone in STOP_CENTERS:

        center = STOP_CENTERS[phone]

        spectrum += gaussian(
            MEL_CENTERS,
            center,
            1200,
        )

        spectrum += 0.1

    elif phone in LIQUID_FORMANTS:

        for formant in LIQUID_FORMANTS[phone]:

            spectrum += gaussian(
                MEL_CENTERS,
                formant,
                250,
            )

    else:

        raise ValueError(
            f"Unknown phoneme: {phone}"
        )

    log_spectrum = np.log(
        spectrum + EPSILON
    )

    normalized = (
        log_spectrum - log_spectrum.mean()
    ) / (
        log_spectrum.std() + EPSILON
    )

    return normalized


def build_phoneme_templates():

    templates = {}

    for phone in sorted(
        set(PHONEME_SEQUENCE)
    ):

        templates[phone] = (
            build_phone_prototype(phone)
        )

    print(
        f"\nCreated {len(templates)} handcrafted phone templates."
    )

    return templates


# ============================================================
# STEP 7 — ACOUSTIC DISTANCE MATRIX
# ============================================================

def calculate_distance_matrix(
    audio_features,
    templates,
    phoneme_sequence,
):
    """
    Compare every audio frame against every phoneme template.

    Output:
        [number_of_phones, number_of_audio_frames]
    """

    audio_features = np.asarray(
        audio_features,
        dtype=np.float64,
    )

    if audio_features.ndim != 2:

        raise ValueError(
            "audio_features must be a 2D matrix."
        )

    n_phones = len(
        phoneme_sequence
    )

    n_frames = audio_features.shape[0]

    if n_phones == 0 or n_frames == 0:

        raise ValueError(
            "Phoneme sequence and audio features cannot be empty."
        )

    distances = np.zeros(
        (n_phones, n_frames),
        dtype=np.float64,
    )

    for phone_index, phone in enumerate(
        phoneme_sequence
    ):

        if phone not in templates:

            raise ValueError(
                f"Missing acoustic template: {phone}"
            )

        template = templates[phone]

        if template.shape[0] != audio_features.shape[1]:

            raise ValueError(
                f"Template dimension mismatch for {phone}."
            )

        difference = (
            audio_features - template[None, :]
        )

        distances[phone_index] = np.mean(
            difference ** 2,
            axis=1,
        )

    # Normalize overall scale.
    distances /= (
        np.median(distances) + EPSILON
    )

    return distances


# ============================================================
# STEP 8 — SEGMENTAL DYNAMIC PROGRAMMING
# ============================================================

def segmental_dynamic_programming(
    distances,
    min_duration=2,
    max_duration=30,
    target_duration=None,
    duration_penalty=0.08,
    require_full_coverage=False,
):
    """
    Align a phoneme sequence with audio frames.

    Every phoneme receives one contiguous segment.

    Parameters
    ----------
    distances:
        Acoustic distance matrix [n_phones, n_frames].

    min_duration:
        Minimum number of frames per phoneme.

    max_duration:
        Maximum number of frames per phoneme.

    target_duration:
        Expected phoneme duration in frames.

    duration_penalty:
        Weight applied to duration deviation.

    require_full_coverage:
        True:
            Final phoneme must end at the final audio frame.

        False:
            Final phoneme may finish before the audio ends.
            This allows trailing silence to remain unaligned.

    Returns
    -------
    alignment:
        List of phoneme intervals.

    final_cost:
        Total alignment cost.
    """

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    if distances.ndim != 2:

        raise ValueError(
            "distances must be a 2D matrix."
        )

    n_phones, n_frames = distances.shape

    if n_phones == 0 or n_frames == 0:

        raise ValueError(
            "Distance matrix cannot have an empty dimension."
        )

    if not np.all(
        np.isfinite(distances)
    ):

        raise ValueError(
            "Distance matrix contains NaN or infinite values."
        )

    if min_duration < 1:

        raise ValueError(
            "min_duration must be at least 1."
        )

    if max_duration < min_duration:

        raise ValueError(
            "max_duration must be >= min_duration."
        )

    if n_phones * min_duration > n_frames:

        raise ValueError(
            "Not enough audio frames for the requested "
            "minimum phoneme durations."
        )

    if duration_penalty < 0:

        raise ValueError(
            "duration_penalty cannot be negative."
        )

    if target_duration is None:

        target_duration = max(
            min_duration,
            n_frames / n_phones,
        )

    if target_duration <= 0:

        raise ValueError(
            "target_duration must be positive."
        )

    infinity = float("inf")

    # --------------------------------------------------------
    # Dynamic programming tables
    # --------------------------------------------------------

    dp = np.full(
        (n_phones, n_frames),
        infinity,
        dtype=np.float64,
    )

    backpointer = np.full(
        (n_phones, n_frames),
        -1,
        dtype=np.int32,
    )

    # Prefix sums allow O(1) segment-cost calculation.
    prefix = np.concatenate(
        [
            np.zeros(
                (n_phones, 1),
                dtype=np.float64,
            ),

            np.cumsum(
                distances,
                axis=1,
            ),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # INITIALIZE FIRST PHONEME
    #
    # The first phoneme must start at frame zero.
    # --------------------------------------------------------

    for end_frame in range(n_frames):

        max_d = min(
            max_duration,
            end_frame + 1,
        )

        for duration in range(
            min_duration,
            max_d + 1,
        ):

            start_frame = (
                end_frame - duration + 1
            )

            if start_frame != 0:
                continue

            acoustic_cost = (
                prefix[0, end_frame + 1]
                - prefix[0, start_frame]
            ) / duration

            duration_cost = (
                duration_penalty
                * (
                    (duration - target_duration)
                    / target_duration
                ) ** 2
            )

            dp[0, end_frame] = (
                acoustic_cost + duration_cost
            )

            backpointer[0, end_frame] = 0

    # --------------------------------------------------------
    # PROCESS REMAINING PHONEMES
    # --------------------------------------------------------

    for phone_index in range(
        1,
        n_phones,
    ):

        for end_frame in range(n_frames):

            max_d = min(
                max_duration,
                end_frame + 1,
            )

            for duration in range(
                min_duration,
                max_d + 1,
            ):

                start_frame = (
                    end_frame - duration + 1
                )

                previous_end = (
                    start_frame - 1
                )

                if previous_end < 0:
                    continue

                previous_cost = dp[
                    phone_index - 1,
                    previous_end,
                ]

                if not np.isfinite(
                    previous_cost
                ):

                    continue

                acoustic_cost = (
                    prefix[
                        phone_index,
                        end_frame + 1,
                    ]
                    - prefix[
                        phone_index,
                        start_frame,
                    ]
                ) / duration

                duration_cost = (
                    duration_penalty
                    * (
                        (duration - target_duration)
                        / target_duration
                    ) ** 2
                )

                total_cost = (
                    previous_cost
                    + acoustic_cost
                    + duration_cost
                )

                if total_cost < dp[
                    phone_index,
                    end_frame,
                ]:

                    dp[
                        phone_index,
                        end_frame,
                    ] = total_cost

                    backpointer[
                        phone_index,
                        end_frame,
                    ] = start_frame

    # --------------------------------------------------------
    # SELECT FINAL FRAME
    # --------------------------------------------------------

    if require_full_coverage:

        # Final phoneme must end at the final audio frame.

        final_frame = n_frames - 1

        final_cost = float(
            dp[-1, final_frame]
        )

    else:

        # Permit trailing silence.

        valid_costs = dp[-1]

        minimum_cost = np.min(
            valid_costs
        )

        if not np.isfinite(minimum_cost):

            raise RuntimeError(
                "No valid partial alignment path was found."
            )

        # IMPORTANT FIX:
        #
        # Several final frames can have exactly the same cost.
        #
        # Previously, np.argmin() selected the EARLIEST frame.
        # This could produce unnecessarily short final segments.
        #
        # We now prefer the LATEST frame among equal-cost paths.
        # This avoids premature termination when trailing frames
        # have equally good acoustic costs.

        candidate_frames = np.flatnonzero(
            np.isclose(
                valid_costs,
                minimum_cost,
                rtol=1e-10,
                atol=1e-12,
            )
        )

        final_frame = int(
            candidate_frames[-1]
        )

        final_cost = float(
            valid_costs[final_frame]
        )

    if not np.isfinite(final_cost):

        raise RuntimeError(
            "No valid alignment path was found. "
            "Try increasing max_duration or reducing "
            "min_duration."
        )

    # --------------------------------------------------------
    # BACKTRACKING
    # --------------------------------------------------------

    alignment = []

    current_end = final_frame

    for phone_index in reversed(
        range(n_phones)
    ):

        start_frame = int(
            backpointer[
                phone_index,
                current_end,
            ]
        )

        if start_frame < 0:

            raise RuntimeError(
                "Invalid backpointer during alignment."
            )

        # end_frame is EXCLUSIVE.
        exclusive_end = current_end + 1

        alignment.append({

            "start_frame": start_frame,

            "end_frame": exclusive_end,

            "duration_frames": (
                exclusive_end - start_frame
            ),
        })

        current_end = (
            start_frame - 1
        )

    alignment.reverse()

    # --------------------------------------------------------
    # VALIDATE RESULT
    # --------------------------------------------------------

    assert alignment[0]["start_frame"] == 0

    for index in range(
        1,
        len(alignment),
    ):

        assert (
            alignment[index]["start_frame"]
            == alignment[index - 1]["end_frame"]
        )

    if require_full_coverage:

        assert (
            alignment[-1]["end_frame"] == n_frames
        )

    assert (
        alignment[-1]["end_frame"] <= n_frames
    )

    for item in alignment:

        assert (
            item["duration_frames"] >= min_duration
        )

        assert (
            item["duration_frames"] <= max_duration
        )

    return alignment, final_cost


# ============================================================
# STEP 9 — SAVE ALIGNMENT
# ============================================================

def save_alignment(
    alignment,
    phoneme_sequence,
    output_path,
    audio_duration,
):
    """
    Convert frame boundaries into time boundaries.

    Frame intervals use exclusive end boundaries.
    """

    if len(alignment) != len(phoneme_sequence):

        raise ValueError(
            "Alignment length does not match phoneme sequence."
        )

    results = []

    for phone, item in zip(
        phoneme_sequence,
        alignment,
    ):

        start_time = (
            item["start_frame"]
            * HOP_LENGTH
            / SAMPLE_RATE
        )

        end_time = (
            item["end_frame"]
            * HOP_LENGTH
            / SAMPLE_RATE
        )

        start_time = min(
            start_time,
            audio_duration,
        )

        end_time = min(
            end_time,
            audio_duration,
        )

        results.append({

            "phoneme": phone,

            "start_frame": item["start_frame"],

            "end_frame": item["end_frame"],

            "duration_frames": item["duration_frames"],

            "start_time": start_time,

            "end_time": end_time,

            "duration_seconds": (
                end_time - start_time
            ),
        })

    output = {

        "sample_rate": SAMPLE_RATE,

        "hop_length": HOP_LENGTH,

        "frame_rate": (
            SAMPLE_RATE / HOP_LENGTH
        ),

        "audio_duration_seconds": audio_duration,

        "phoneme_sequence": phoneme_sequence,

        "alignment_type": (
            "handcrafted_template_dynamic_programming"
        ),

        "pretrained_models_used": False,

        "aligned_phonemes": results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )

    print(
        f"\nAlignment JSON saved: {output_path}"
    )

    return results


# ============================================================
# STEP 10 — PRINT RESULTS
# ============================================================

def print_alignment(results):

    print("\n" + "=" * 85)

    print(
        "PHONEME ALIGNMENT"
    )

    print("=" * 85)

    print(
        f"{'Phone':<10}"
        f"{'Start':>12}"
        f"{'End':>12}"
        f"{'Duration':>14}"
        f"{'Frames':>12}"
    )

    print("-" * 85)

    for item in results:

        print(
            f"{item['phoneme']:<10}"
            f"{item['start_time']:>12.3f}"
            f"{item['end_time']:>12.3f}"
            f"{item['duration_seconds']:>14.3f}"
            f"{item['duration_frames']:>12}"
        )

    print("-" * 85)

    print(
        f"Total aligned phones: {len(results)}"
    )


# ============================================================
# STEP 11 — VISUALIZATION
# ============================================================

def visualize(
    waveform,
    log_mel,
    results,
    output_path,
):

    duration = (
        len(waveform) / SAMPLE_RATE
    )

    times = (
        np.arange(len(waveform))
        / SAMPLE_RATE
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(17, 12),
        sharex=True,
    )

    # --------------------------------------------------------
    # WAVEFORM
    # --------------------------------------------------------

    axes[0].plot(
        times,
        waveform,
        linewidth=0.5,
    )

    axes[0].set_title(
        "Downloaded Speech Waveform"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[0].grid(
        True,
        alpha=0.25,
    )

    # --------------------------------------------------------
    # LOG-MEL
    # --------------------------------------------------------

    mel_times = (
        np.arange(log_mel.shape[0])
        * HOP_LENGTH
        / SAMPLE_RATE
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

    axes[1].set_ylabel(
        "Mel bins"
    )

    # --------------------------------------------------------
    # PHONEME ALIGNMENT
    # --------------------------------------------------------

    axes[2].plot(
        times,
        waveform,
        linewidth=0.4,
    )

    axes[2].set_title(
        "Handcrafted-Template Phoneme Alignment"
    )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    axes[2].set_ylabel(
        "Amplitude"
    )

    amplitude = max(
        float(np.max(np.abs(waveform))),
        0.1,
    )

    for item in results:

        start = item["start_time"]

        end = item["end_time"]

        midpoint = (
            start + end
        ) / 2

        axes[2].axvline(
            start,
            linestyle="--",
            linewidth=0.7,
        )

        axes[2].text(
            midpoint,
            amplitude * 0.75,
            item["phoneme"],
            ha="center",
            fontsize=7,
        )

    axes[2].grid(
        True,
        alpha=0.25,
    )

    axes[2].set_xlim(
        0,
        duration,
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.show()

    print(
        f"\nPlot saved: {output_path}"
    )


# ============================================================
# STEP 12 — NUMERICAL TESTS
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

    recovered = mel_to_hz(
        hz_to_mel(frequencies)
    )

    assert np.allclose(
        frequencies,
        recovered,
        atol=1e-5,
    )

    print(
        "  [PASS] Mel conversion"
    )

    # --------------------------------------------------------
    # TEST 2 — STFT
    # --------------------------------------------------------

    test_signal = np.sin(
        2 * np.pi * 440
        * np.arange(SAMPLE_RATE)
        / SAMPLE_RATE
    )

    spectrum = stft(
        test_signal
    )

    assert spectrum.ndim == 2

    assert spectrum.shape[1] == (
        N_FFT // 2 + 1
    )

    assert np.all(
        np.isfinite(spectrum)
    )

    print(
        "  [PASS] STFT"
    )

    # --------------------------------------------------------
    # TEST 3 — MEL FILTERBANK
    # --------------------------------------------------------

    assert MEL_FILTERBANK.shape == (
        N_MELS,
        N_FFT // 2 + 1,
    )

    assert np.all(
        MEL_FILTERBANK >= 0
    )

    assert np.all(
        np.isfinite(MEL_FILTERBANK)
    )

    print(
        "  [PASS] Mel filterbank"
    )

    # --------------------------------------------------------
    # TEST 4 — LOG-MEL FEATURES
    # --------------------------------------------------------

    features = extract_log_mel(
        test_signal
    )

    assert features.shape[1] == N_MELS

    assert np.all(
        np.isfinite(features)
    )

    print(
        "  [PASS] Log-Mel extraction"
    )

    # --------------------------------------------------------
    # TEST 5 — PHONEME TEMPLATE
    # --------------------------------------------------------

    template = build_phone_prototype(
        "IH"
    )

    assert template.shape == (
        N_MELS,
    )

    assert np.all(
        np.isfinite(template)
    )

    print(
        "  [PASS] Phoneme prototype"
    )

    # --------------------------------------------------------
    # TEST 6 — FULL-COVERAGE DYNAMIC PROGRAMMING
    # --------------------------------------------------------

    costs = np.ones(
        (2, 10),
        dtype=np.float64,
    )

    # First phoneme is correct in frames 0-4.
    costs[0, :5] = 0

    # Second phoneme is correct in frames 5-9.
    costs[1, 5:] = 0

    aligned, final_cost = (
        segmental_dynamic_programming(
            costs,
            min_duration=3,
            max_duration=7,
            target_duration=5,
            duration_penalty=0,
            require_full_coverage=True,
        )
    )

    assert aligned[0]["start_frame"] == 0

    assert aligned[0]["end_frame"] == 5

    assert aligned[1]["start_frame"] == 5

    assert aligned[1]["end_frame"] == 10

    assert aligned[-1]["end_frame"] == 10

    assert np.isfinite(
        final_cost
    )

    print(
        "  [PASS] Full-coverage DP"
    )

    # --------------------------------------------------------
    # TEST 7 — PARTIAL-COVERAGE DYNAMIC PROGRAMMING
    #
    # Frames 10 and 11 represent trailing silence.
    #
    # We want:
    #
    # Phone 1: frames 0-4
    # Phone 2: frames 5-9
    # Silence: frames 10-11
    #
    # IMPORTANT FIX:
    #
    # A positive duration penalty is used here.
    # This makes a five-frame segment preferable to a
    # shorter segment with an identical average acoustic cost.
    # --------------------------------------------------------

    costs_with_silence = np.ones(
        (2, 12),
        dtype=np.float64,
    )

    costs_with_silence[0, :5] = 0

    costs_with_silence[1, 5:10] = 0

    costs_with_silence[1, 10:] = 20

    aligned_partial, partial_cost = (
        segmental_dynamic_programming(
            costs_with_silence,
            min_duration=3,
            max_duration=7,
            target_duration=5,
            duration_penalty=0.08,
            require_full_coverage=False,
        )
    )

    assert aligned_partial[0]["start_frame"] == 0

    assert aligned_partial[0]["end_frame"] == 5

    assert aligned_partial[1]["start_frame"] == 5

    assert aligned_partial[1]["end_frame"] == 10

    # The last two frames remain unaligned.
    assert aligned_partial[-1]["end_frame"] == 10

    assert aligned_partial[-1]["end_frame"] < 12

    assert np.isfinite(
        partial_cost
    )

    print(
        "  [PASS] Partial-coverage DP"
    )

    # --------------------------------------------------------
    # TEST 8 — ALIGNMENT CONTINUITY
    # --------------------------------------------------------

    for index in range(
        1,
        len(aligned),
    ):

        assert (
            aligned[index]["start_frame"]
            == aligned[index - 1]["end_frame"]
        )

    print(
        "  [PASS] Alignment continuity"
    )

    # --------------------------------------------------------
    # TEST 9 — INVALID INPUT
    # --------------------------------------------------------

    invalid_input_detected = False

    try:

        segmental_dynamic_programming(
            np.ones((3, 4)),
            min_duration=2,
            max_duration=5,
        )

    except ValueError:

        invalid_input_detected = True

    assert invalid_input_detected

    print(
        "  [PASS] Invalid input handling"
    )

    # --------------------------------------------------------
    # TEST 10 — INVALID DURATION PENALTY
    # --------------------------------------------------------

    invalid_penalty_detected = False

    try:

        segmental_dynamic_programming(
            np.ones((2, 10)),
            duration_penalty=-0.1,
        )

    except ValueError:

        invalid_penalty_detected = True

    assert invalid_penalty_detected

    print(
        "  [PASS] Invalid duration penalty handling"
    )

    # --------------------------------------------------------
    # TEST 11 — DISTANCE MATRIX
    # --------------------------------------------------------

    templates = {
        "IH": build_phone_prototype("IH"),
        "AE": build_phone_prototype("AE"),
    }

    test_features = np.random.randn(
        20,
        N_MELS,
    )

    distances = calculate_distance_matrix(
        test_features,
        templates,
        ["IH", "AE"],
    )

    assert distances.shape == (
        2,
        20,
    )

    assert np.all(
        np.isfinite(distances)
    )

    print(
        "  [PASS] Acoustic distance matrix"
    )

    print(
        "\nAll numerical tests passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(
        SEED
    )

    print("=" * 72)

    print(
        "PHASE 2 — LESSON 54"
    )

    print(
        "PHONEME-TO-AUDIO ALIGNMENT FROM SCRATCH"
    )

    print("=" * 72)

    print(
        f"Mel sample rate: {SAMPLE_RATE}"
    )

    print(
        f"Mel hop length: {HOP_LENGTH}"
    )

    print(
        f"Mel frame rate: "
        f"{SAMPLE_RATE / HOP_LENGTH:.3f}"
    )

    # --------------------------------------------------------
    # 1. NUMERICAL TESTS
    # --------------------------------------------------------

    run_tests()

    # --------------------------------------------------------
    # 2. DOWNLOAD SAMPLE DATASET
    # --------------------------------------------------------

    transcript = prepare_sample_dataset()

    # --------------------------------------------------------
    # 3. LOAD AUDIO
    # --------------------------------------------------------

    waveform = load_audio(
        AUDIO_PATH
    )

    audio_duration = (
        len(waveform) / SAMPLE_RATE
    )

    # --------------------------------------------------------
    # 4. VALIDATE TRANSCRIPT
    # --------------------------------------------------------

    if transcript.lower() != EXPECTED_TRANSCRIPT.lower():

        raise ValueError(
            "Transcript does not match the prepared "
            "phoneme sequence."
        )

    # --------------------------------------------------------
    # 5. MANUALLY PREPARED PHONEME SEQUENCE
    # --------------------------------------------------------

    print(
        "\nPhoneme sequence:"
    )

    print(
        " ".join(PHONEME_SEQUENCE)
    )

    print(
        f"\nNumber of phones: {len(PHONEME_SEQUENCE)}"
    )

    # --------------------------------------------------------
    # 6. EXTRACT ACOUSTIC FEATURES
    # --------------------------------------------------------

    print(
        "\nExtracting log-Mel features..."
    )

    log_mel = extract_log_mel(
        waveform
    )

    print(
        f"Log-Mel shape: {log_mel.shape}"
    )

    # --------------------------------------------------------
    # 7. BUILD HANDCRAFTED TEMPLATES
    # --------------------------------------------------------

    print(
        "\nBuilding acoustic templates..."
    )

    templates = build_phoneme_templates()

    # --------------------------------------------------------
    # 8. CALCULATE ACOUSTIC DISTANCE
    # --------------------------------------------------------

    print(
        "\nCalculating acoustic distance matrix..."
    )

    distances = calculate_distance_matrix(
        log_mel,
        templates,
        PHONEME_SEQUENCE,
    )

    print(
        f"Distance matrix: {distances.shape}"
    )

    # --------------------------------------------------------
    # 9. DYNAMIC PROGRAMMING
    # --------------------------------------------------------

    print(
        "\nRunning Dynamic Programming..."
    )

    average_duration = (
        log_mel.shape[0]
        / len(PHONEME_SEQUENCE)
    )

    max_phone_duration = max(
        10,
        int(average_duration * 3),
    )

    alignment, total_cost = (
        segmental_dynamic_programming(
            distances,

            min_duration=2,

            max_duration=max_phone_duration,

            target_duration=average_duration,

            duration_penalty=0.08,

            require_full_coverage=False,
        )
    )

    print(
        f"Alignment cost: {total_cost:.4f}"
    )

    print(
        f"Aligned frames: "
        f"{alignment[-1]['end_frame']}"
        f" / {log_mel.shape[0]}"
    )

    # --------------------------------------------------------
    # 10. SAVE RESULTS
    # --------------------------------------------------------

    results = save_alignment(
        alignment,
        PHONEME_SEQUENCE,
        JSON_PATH,
        audio_duration,
    )

    # --------------------------------------------------------
    # 11. PRINT ALIGNMENT
    # --------------------------------------------------------

    print_alignment(
        results
    )

    # --------------------------------------------------------
    # 12. VISUALIZATION
    # --------------------------------------------------------

    visualize(
        waveform,
        log_mel,
        results,
        PLOT_PATH,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "LESSON 54 COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  Audio:      {AUDIO_PATH}"
    )

    print(
        f"  Transcript: {TRANSCRIPT_PATH}"
    )

    print(
        f"  JSON:       {JSON_PATH}"
    )

    print(
        f"  Plot:       {PLOT_PATH}"
    )

    print(
        "\nNo pretrained models or aligners were used."
    )

    print(
        "\nNext lesson:"
    )

    print(
        "Lesson 55 — Real Mel Dataset Builder"
    )


if __name__ == "__main__":

    main()