"""
===============================================================================
PHASE 1 — LESSON 12
PROSODY: PITCH, STRESS, INTONATION, RHYTHM AND SPEECH DURATION
===============================================================================

PURPOSE
-------

Understand and implement the prosody component of a multilingual TTS system.

This lesson covers:

    1. Fundamental frequency (F0)
    2. Pitch contours
    3. Stress and emphasis
    4. Intonation
    5. Rhythm
    6. Syllable duration
    7. Speaking rate
    8. Pauses
    9. Voiced and unvoiced regions
    10. Language-dependent prosody
    11. Prosody feature extraction
    12. Prosody-conditioned speech generation

SUPPORTED LANGUAGES
-------------------

    English     en
    Chinese     zh
    Japanese    ja
    Russian     ru

IMPORTANT DISTINCTIONS
----------------------

This is an educational prosody simulator.

It does NOT generate intelligible natural speech.

It generates simplified harmonic and noise signals to demonstrate how
prosodic parameters affect a speech-like waveform.

The implementation does not use a pretrained TTS model.

===============================================================================
THEORY 1 — WHAT IS PROSODY?
===============================================================================

Prosody describes the rhythm, pitch, timing, and emphasis patterns of speech.

A spoken sentence is not simply a sequence of phonemes.

It also contains information about:

    - Which syllables are emphasized
    - Which words receive focus
    - Whether a sentence is a question
    - Whether a speaker sounds excited or calm
    - How long syllables last
    - Where pauses occur
    - How pitch rises and falls

A simplified speech representation is:

    Speech = Linguistic content + Prosody + Speaker characteristics

For TTS:

    Text
      |
      v
    Text frontend
      |
      v
    Phoneme sequence
      |
      v
    Prosody representation
      |
      v
    Acoustic generator
      |
      v
    Vocoder
      |
      v
    Waveform

Prosody is not a cosmetic feature.

It is essential for natural speech generation.

===============================================================================
THEORY 2 — FUNDAMENTAL FREQUENCY (F0)
===============================================================================

Fundamental frequency is the rate at which the vocal folds vibrate.

If the vocal folds complete one vibration cycle every T0 seconds:

    F0 = 1 / T0

Where:

    F0 = fundamental frequency in Hz
    T0 = fundamental period in seconds

Examples:

    T0 = 0.010 seconds

    F0 = 1 / 0.010

    F0 = 100 Hz

Typical approximate pitch ranges vary considerably by speaker, age,
speaking style, and language.

For an educational simulation:

    Low pitch:       80 Hz
    Medium pitch:   150 Hz
    High pitch:     250 Hz

These are illustrative values, not universal human pitch categories.

F0 is different from formants.

F0:
    Related to vocal-fold vibration and perceived pitch.

Formants:
    Resonances of the vocal tract that strongly influence vowel quality.

A speaker can raise F0 without changing the vowel's basic formant structure.

===============================================================================
THEORY 3 — PITCH CONTOURS
===============================================================================

A pitch contour describes how F0 changes over time.

Example:

    Time:       0    1    2    3
    F0:        120  130  160  190

This is a rising pitch contour.

A falling contour:

    Time:       0    1    2    3
    F0:        190  170  140  110

Pitch contours contribute to:

    - Questions
    - Statements
    - Emphasis
    - Emotion
    - Focus
    - Linguistic tone

A simple interpolation model:

    F0(t) = interpolation of pitch targets

A more complete system may predict F0 using a neural network.

===============================================================================
THEORY 4 — INTONATION
===============================================================================

Intonation is the larger pitch pattern across a phrase or sentence.

English examples:

Statement:
    "You are ready."

Question:
    "You are ready?"

Excitement:
    "You are READY!"

Their pitch contours may differ.

A simplified English intonation model:

    Statement:
        Slight movement followed by a falling boundary.

    Yes/no question:
        Rising boundary.

    Wh-question:
        Often falling boundary.

    Emphasis:
        A local pitch movement and/or increased duration and intensity.

These are tendencies, not universal rules.

===============================================================================
THEORY 5 — STRESS AND EMPHASIS
===============================================================================

Stress is a property of syllables or words.

A stressed syllable may have:

    - Greater duration
    - Greater intensity
    - Higher or more dynamic pitch
    - Different vowel realization

Stress is not simply "high pitch".

A stressed syllable may have a falling pitch or even a relatively low F0.

For a simplified TTS prosody model:

    duration_stressed = duration_base * 1.30

    duration_unstressed = duration_base * 0.85

    intensity_stressed > intensity_unstressed

The numerical multipliers are educational parameters, not universal
phonetic laws.

English:
    Lexical stress is important.

Chinese:
    Lexical tone is essential. Stress and tone are different concepts.

Japanese:
    Pitch accent and phrase-level intonation are important.

Russian:
    Lexical stress is important and affects vowel realization.

===============================================================================
THEORY 6 — CHINESE LEXICAL TONES
===============================================================================

Mandarin Chinese uses lexical tones.

A simplified four-tone description:

    Tone 1: high and relatively level
    Tone 2: rising
    Tone 3: low/dipping in citation form
    Tone 4: falling

Neutral tone is also important.

Example Pinyin:

    mā
    má
    mǎ
    mà

The syllable segment is similar, but the tone changes its lexical meaning.

Important:

    Mandarin lexical tone is not the same as sentence intonation.

A question may have a sentence-level intonation pattern while preserving
the lexical tone information of individual syllables.

Real Mandarin prosody must also handle tone sandhi and contextual tone
realization.

This lesson implements simplified tone contours for demonstration only.

===============================================================================
THEORY 7 — JAPANESE PITCH ACCENT
===============================================================================

Japanese pitch accent involves patterns of high and low pitch.

It is not equivalent to English stress.

Different words may have different accent patterns.

Japanese prosody also involves:

    - Mora timing
    - Phrase boundaries
    - Pitch accent
    - Intonation

A mora is a timing unit used in Japanese phonology.

For example, a long vowel generally occupies more than one mora.

A Japanese TTS system should not simply apply English stress rules.

===============================================================================
THEORY 8 — RUSSIAN STRESS
===============================================================================

Russian has lexical stress.

Stress influences:

    - Vowel reduction
    - Pitch movement
    - Duration
    - Perceived prominence

Unstressed vowels may be reduced.

A Russian TTS system needs stress information before generating natural
pronunciation and timing.

===============================================================================
THEORY 9 — SYLLABLE DURATION
===============================================================================

Speech is not produced with equal duration for every phoneme or syllable.

Duration depends on:

    - Phoneme identity
    - Syllable structure
    - Stress
    - Speaking rate
    - Phrase position
    - Emphasis
    - Pauses
    - Linguistic context

A simplified duration model:

    D_i = D_base * S_i * R_i * E_i

Where:

    D_i    = predicted duration
    D_base = base duration
    S_i    = stress multiplier
    R_i    = speaking-rate multiplier
    E_i    = emphasis multiplier

Speaking-rate relationship:

    Higher speaking rate -> shorter durations.

For example:

    Normal duration = 200 ms

    Speaking rate multiplier = 0.8

    Result = 160 ms

A real TTS system must predict duration at phoneme or token level.

===============================================================================
THEORY 10 — PAUSES
===============================================================================

Pauses provide structure.

Examples:

    "Hello, John."

    "First, open the application."

    "I wanted to go, but it was raining."

Pauses may occur at:

    - Sentence boundaries
    - Clause boundaries
    - Commas
    - Discourse boundaries
    - Hesitations

A simple pause model:

    comma pause:       100-250 ms
    sentence pause:    250-500 ms

Actual pause duration depends on context and speaking style.

===============================================================================
THEORY 11 — INTENSITY AND ENERGY
===============================================================================

Speech intensity relates to signal amplitude and perceived loudness.

Short-time RMS:

    RMS = sqrt(mean(x[n]^2))

For a stressed syllable, a simplified model may increase amplitude.

However:

    Intensity is not identical to stress.

A whisper may have no periodic F0 but still contain emphasis.

A natural TTS system must model intensity and voicing separately.

===============================================================================
THEORY 12 — PROSODY CONDITIONING IN NEURAL TTS
===============================================================================

A neural TTS system may use:

    Text encoder
        |
        v
    Phoneme representations
        |
        +--------------------------+
        |                          |
        v                          v
    Duration predictor         F0 predictor
        |                          |
        v                          v
    Duration sequence          Pitch sequence
        |                          |
        +------------+-------------+
                     |
                     v
              Acoustic decoder
                     |
                     v
              Mel spectrogram
                     |
                     v
                  Vocoder
                     |
                     v
                  Speech

A neural model may predict:

    duration
    F0
    energy
    voiced/unvoiced state

These features can condition acoustic generation.

Some architectures predict prosody implicitly rather than using explicit
separate predictors.

This lesson builds an explicit educational prosody representation.

===============================================================================
THEORY 13 — F0 EXTRACTION
===============================================================================

A basic F0 estimation method uses autocorrelation.

For a signal x[n]:

    R[k] = sum(x[n] * x[n-k])

The lag corresponding to a periodicity peak estimates the fundamental
period:

    T0 = lag / sample_rate

Therefore:

    F0 = sample_rate / lag

This method has limitations:

    - Harmonic confusion
    - Octave errors
    - Background noise
    - Unvoiced sounds
    - Irregular vocal-fold vibration
    - Formant influence

The implementation below is an educational autocorrelation estimator.

It is not a production pitch tracker.

===============================================================================
THEORY 14 — WHAT WE WILL IMPLEMENT
===============================================================================

1. Create syllable-level prosody metadata.
2. Predict simplified syllable durations.
3. Generate English statement and question contours.
4. Generate simplified Mandarin lexical tone contours.
5. Generate simplified Japanese pitch patterns.
6. Generate simplified Russian stress patterns.
7. Generate a speech-like harmonic signal.
8. Add pauses.
9. Calculate short-time RMS.
10. Estimate F0 using autocorrelation.
11. Visualize pitch, duration, energy, and waveform.

===============================================================================
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Optional, Sequence


# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 24000

FRAME_SIZE = 1024

HOP_SIZE = 256

DEFAULT_F0 = 150.0

DEFAULT_DURATION = 0.22

RNG = np.random.default_rng(42)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Syllable:
    """
    Simplified syllable-level prosody information.

    label:
        Human-readable syllable label.

    stress:
        0 = unstressed
        1 = stressed

    tone:
        Mandarin tone number, if applicable.

    accent:
        Simplified Japanese pitch-accent label.

    pause_after:
        Silence duration after this syllable.

    emphasis:
        Additional emphasis multiplier.
    """

    label: str

    stress: int = 0

    tone: Optional[int] = None

    accent: Optional[str] = None

    pause_after: float = 0.0

    emphasis: float = 1.0


@dataclass
class ProsodyConfig:
    """
    Global prosody settings.
    """

    base_f0: float = DEFAULT_F0

    speaking_rate: float = 1.0

    stressed_duration_multiplier: float = 1.30

    unstressed_duration_multiplier: float = 0.85

    stressed_energy_multiplier: float = 1.25

    unstressed_energy_multiplier: float = 0.85

    minimum_duration: float = 0.07

    maximum_duration: float = 0.60


@dataclass
class SyllableResult:

    label: str

    start_time: float

    end_time: float

    duration: float

    stress: int

    f0_start: float

    f0_end: float

    energy: float

    pause_after: float


# =============================================================================
# SECTION 1 — BASIC MATHEMATICS
# =============================================================================

def semitones_to_ratio(semitones: float) -> float:
    """
    Convert semitone difference into a frequency ratio.

    Equal-tempered musical frequency relationship:

        ratio = 2 ** (semitones / 12)

    Example:

        +12 semitones = twice the frequency.
        -12 semitones = half the frequency.
    """

    return 2.0 ** (semitones / 12.0)


def shift_pitch(
    f0: float,
    semitones: float
) -> float:

    return f0 * semitones_to_ratio(semitones)


def calculate_rms(signal: np.ndarray) -> float:

    if len(signal) == 0:

        return 0.0

    return float(
        np.sqrt(
            np.mean(signal ** 2)
        )
    )


# =============================================================================
# SECTION 2 — DURATION PREDICTION
# =============================================================================

def predict_syllable_duration(
    syllable: Syllable,
    config: ProsodyConfig
) -> float:
    """
    Predict simplified syllable duration.

    Equation:

        D = D_base * stress_multiplier * emphasis / speaking_rate

    Stress changes duration.

    Speaking rate changes duration.

    Emphasis changes duration.

    The output is clamped to a reasonable educational range.

    This is a hand-designed model, not a trained duration predictor.
    """

    base = DEFAULT_DURATION

    if syllable.stress:

        stress_multiplier = (
            config.stressed_duration_multiplier
        )

    else:

        stress_multiplier = (
            config.unstressed_duration_multiplier
        )

    duration = (
        base
        * stress_multiplier
        * syllable.emphasis
        / max(config.speaking_rate, 0.1)
    )

    duration = np.clip(
        duration,
        config.minimum_duration,
        config.maximum_duration
    )

    return float(duration)


# =============================================================================
# SECTION 3 — ENGLISH INTONATION
# =============================================================================

def english_pitch_contour(
    duration: float,
    base_f0: float,
    sentence_type: str,
    stressed: bool
) -> np.ndarray:
    """
    Generate simplified English phrase-level pitch contours.

    Supported:

        statement
        question
        exclamation

    These are educational patterns, not universal English intonation rules.
    """

    n = max(
        2,
        int(duration * SAMPLE_RATE / HOP_SIZE)
    )

    t = np.linspace(
        0.0,
        1.0,
        n
    )

    if sentence_type == "statement":

        # Slight pitch movement followed by a falling boundary.
        contour = np.interp(
            t,
            [0.0, 0.35, 0.70, 1.0],
            [0.0, 1.5, 0.5, -3.0]
        )

    elif sentence_type == "question":

        # Rising phrase-final contour.
        contour = np.interp(
            t,
            [0.0, 0.35, 0.70, 1.0],
            [0.0, 0.5, 1.5, 5.0]
        )

    elif sentence_type == "exclamation":

        # Stronger pitch movement followed by a fall.
        contour = np.interp(
            t,
            [0.0, 0.25, 0.55, 0.80, 1.0],
            [0.0, 4.0, 7.0, 3.0, -1.0]
        )

    else:

        raise ValueError(
            "Unknown sentence type."
        )

    if stressed:

        contour += 2.0 * np.sin(
            np.pi * t
        )

    return base_f0 * semitones_to_ratio(contour)


# =============================================================================
# SECTION 4 — MANDARIN TONE CONTOURS
# =============================================================================

def mandarin_tone_contour(
    tone: int,
    duration: float,
    base_f0: float
) -> np.ndarray:
    """
    Generate simplified Mandarin tone contours.

    Tone 1:
        High and level.

    Tone 2:
        Rising.

    Tone 3:
        Low/dipping educational approximation.

    Tone 4:
        Falling.

    Tone 5:
        Neutral tone approximation.

    Important:

        Actual Mandarin tone realization depends on context.
        Third-tone sandhi and phrase-level effects are not implemented.
    """

    n = max(
        2,
        int(duration * SAMPLE_RATE / HOP_SIZE)
    )

    t = np.linspace(
        0.0,
        1.0,
        n
    )

    if tone == 1:

        semitones = np.full(
            n,
            3.0
        )

    elif tone == 2:

        semitones = np.linspace(
            -1.0,
            5.0,
            n
        )

    elif tone == 3:

        semitones = np.interp(
            t,
            [0.0, 0.45, 1.0],
            [-3.0, -5.0, 1.0]
        )

    elif tone == 4:

        semitones = np.linspace(
            6.0,
            -5.0,
            n
        )

    elif tone == 5:

        semitones = np.linspace(
            0.0,
            -1.0,
            n
        )

    else:

        raise ValueError(
            "Mandarin tone must be 1, 2, 3, 4, or 5."
        )

    return base_f0 * semitones_to_ratio(
        semitones
    )


# =============================================================================
# SECTION 5 — JAPANESE PITCH ACCENT
# =============================================================================

def japanese_pitch_contour(
    duration: float,
    base_f0: float,
    accent: str
) -> np.ndarray:
    """
    Simplified Japanese pitch pattern.

    Patterns:

        heiban:
            Initial low followed by high.

        atamadaka:
            Initial high followed by lower pitch.

        nakadaka:
            Initial low, high middle, then falling.

        odaka:
            High after the initial mora, with a phrase-final fall.

    These are simplified demonstrations.

    Real Japanese pitch accent requires lexical accent information and
    phrase-level modeling.
    """

    n = max(
        2,
        int(duration * SAMPLE_RATE / HOP_SIZE)
    )

    t = np.linspace(
        0.0,
        1.0,
        n
    )

    if accent == "heiban":

        semitones = np.where(
            t < 0.15,
            -2.0,
            2.0
        )

    elif accent == "atamadaka":

        semitones = np.where(
            t < 0.20,
            3.0,
            -2.0
        )

    elif accent == "nakadaka":

        semitones = np.interp(
            t,
            [0.0, 0.20, 0.65, 1.0],
            [-2.0, 2.0, 2.0, -2.0]
        )

    elif accent == "odaka":

        semitones = np.interp(
            t,
            [0.0, 0.15, 0.85, 1.0],
            [-2.0, 2.0, 2.0, -2.0]
        )

    else:

        raise ValueError(
            "Unknown Japanese accent pattern."
        )

    return base_f0 * semitones_to_ratio(
        semitones
    )


# =============================================================================
# SECTION 6 — RUSSIAN STRESS
# =============================================================================

def russian_pitch_contour(
    duration: float,
    base_f0: float,
    stressed: bool
) -> np.ndarray:
    """
    Simplified Russian syllable pitch contour.

    Stressed syllables receive a stronger pitch movement.

    This is NOT a complete Russian intonation or pronunciation model.

    In particular, this function does not model vowel reduction.
    """

    n = max(
        2,
        int(duration * SAMPLE_RATE / HOP_SIZE)
    )

    t = np.linspace(
        0.0,
        1.0,
        n
    )

    if stressed:

        semitones = np.interp(
            t,
            [0.0, 0.35, 0.70, 1.0],
            [-1.0, 3.0, 2.0, -1.0]
        )

    else:

        semitones = np.interp(
            t,
            [0.0, 1.0],
            [-1.0, -2.0]
        )

    return base_f0 * semitones_to_ratio(
        semitones
    )


# =============================================================================
# SECTION 7 — UNIFIED PROSODY GENERATOR
# =============================================================================

def generate_syllable_f0(
    syllable: Syllable,
    language: str,
    duration: float,
    config: ProsodyConfig,
    sentence_type: str = "statement"
) -> np.ndarray:
    """
    Select the appropriate language-dependent F0 model.

    This makes the prosody system modular.

    Each language has its own prosodic rules.
    """

    if language == "en":

        return english_pitch_contour(
            duration,
            config.base_f0,
            sentence_type,
            bool(syllable.stress)
        )

    if language == "zh":

        tone = syllable.tone

        if tone is None:

            tone = 1

        return mandarin_tone_contour(
            tone,
            duration,
            config.base_f0
        )

    if language == "ja":

        accent = syllable.accent

        if accent is None:

            accent = "heiban"

        return japanese_pitch_contour(
            duration,
            config.base_f0,
            accent
        )

    if language == "ru":

        return russian_pitch_contour(
            duration,
            config.base_f0,
            bool(syllable.stress)
        )

    raise ValueError(
        f"Unsupported language: {language}"
    )


# =============================================================================
# SECTION 8 — HARMONIC SIGNAL GENERATION
# =============================================================================

def generate_voiced_signal(
    f0_contour: np.ndarray,
    duration: float,
    energy: float,
    sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Generate a simplified voiced speech-like signal.

    Human voiced speech contains a fundamental frequency and harmonics.

        f0, 2*f0, 3*f0, 4*f0, ...

    We approximate this with:

        x(t) = sum(A_k * sin(2*pi*k*f0*t))

    Harmonic amplitudes decrease with harmonic number.

    IMPORTANT:

        This is not a human vocal tract model.
        It does not generate realistic vowels or intelligible speech.
    """

    num_samples = max(
        1,
        int(duration * sample_rate)
    )

    t = np.arange(
        num_samples
    ) / sample_rate

    contour_positions = np.linspace(
        0.0,
        duration,
        len(f0_contour)
    )

    f0_samples = np.interp(
        t,
        contour_positions,
        f0_contour
    )

    # Integrate instantaneous frequency into phase.
    phase = np.cumsum(
        2.0 * np.pi * f0_samples / sample_rate
    )

    signal = np.zeros(
        num_samples
    )

    for harmonic in range(1, 9):

        amplitude = 1.0 / harmonic

        signal += amplitude * np.sin(
            harmonic * phase
        )

    # Simple onset and offset envelope.
    envelope = np.ones(
        num_samples
    )

    attack = min(
        int(0.015 * sample_rate),
        num_samples
    )

    release = min(
        int(0.025 * sample_rate),
        num_samples
    )

    if attack > 1:

        envelope[:attack] = np.linspace(
            0.0,
            1.0,
            attack
        )

    if release > 1:

        envelope[-release:] = np.linspace(
            1.0,
            0.0,
            release
        )

    signal *= envelope

    signal *= energy

    return signal.astype(
        np.float32
    )


# =============================================================================
# SECTION 9 — UNVOICED NOISE GENERATION
# =============================================================================

def generate_unvoiced_signal(
    duration: float,
    energy: float,
    sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Generate simplified unvoiced noise.

    Unvoiced consonants do not have periodic vocal-fold vibration.

    Examples:

        /s/
        /f/
        /ʃ/

    This implementation uses white noise.

    Real unvoiced sounds have different spectral characteristics.
    """

    num_samples = int(
        duration * sample_rate
    )

    noise = RNG.normal(
        0.0,
        1.0,
        num_samples
    )

    envelope = np.ones(
        num_samples
    )

    attack = min(
        int(0.005 * sample_rate),
        num_samples
    )

    release = min(
        int(0.010 * sample_rate),
        num_samples
    )

    if attack > 1:

        envelope[:attack] = np.linspace(
            0.0,
            1.0,
            attack
        )

    if release > 1:

        envelope[-release:] = np.linspace(
            1.0,
            0.0,
            release
        )

    return (
        noise * envelope * energy * 0.2
    ).astype(
        np.float32
    )


# =============================================================================
# SECTION 10 — COMPLETE PROSODY SEQUENCE
# =============================================================================

def generate_prosody_sequence(
    syllables: Sequence[Syllable],
    language: str,
    config: ProsodyConfig,
    sentence_type: str = "statement"
):
    """
    Generate:

        waveform
        time axis
        syllable metadata

    Each syllable receives:

        duration
        F0 contour
        energy

    Pauses are inserted between syllables when specified.

    This is an explicit rule-based prosody simulator.
    """

    signal_parts = []

    results = []

    current_time = 0.0

    for syllable in syllables:

        duration = predict_syllable_duration(
            syllable,
            config
        )

        f0 = generate_syllable_f0(
            syllable,
            language,
            duration,
            config,
            sentence_type
        )

        if syllable.stress:

            energy = (
                config.stressed_energy_multiplier
            )

        else:

            energy = (
                config.unstressed_energy_multiplier
            )

        energy *= syllable.emphasis

        waveform = generate_voiced_signal(
            f0,
            duration,
            energy
        )

        start_time = current_time

        end_time = (
            start_time + duration
        )

        signal_parts.append(
            waveform
        )

        results.append(
            SyllableResult(
                label=syllable.label,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                stress=syllable.stress,
                f0_start=float(f0[0]),
                f0_end=float(f0[-1]),
                energy=energy,
                pause_after=syllable.pause_after
            )
        )

        current_time = end_time

        if syllable.pause_after > 0:

            pause_samples = int(
                syllable.pause_after * SAMPLE_RATE
            )

            signal_parts.append(
                np.zeros(
                    pause_samples,
                    dtype=np.float32
                )
            )

            current_time += syllable.pause_after

    if not signal_parts:

        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            results
        )

    signal = np.concatenate(
        signal_parts
    )

    time_axis = np.arange(
        len(signal)
    ) / SAMPLE_RATE

    return (
        signal,
        time_axis,
        results
    )


# =============================================================================
# SECTION 11 — SHORT-TIME RMS
# =============================================================================

def calculate_short_time_rms(
    signal: np.ndarray,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE
):
    """
    Calculate short-time RMS.

    For each frame:

        RMS = sqrt(mean(x[n]^2))

    This provides a simple energy contour.

    It can help identify:
        - Speech activity
        - Emphasis
        - Silence
        - Changes in intensity

    RMS alone cannot identify syllables or stress reliably.
    """

    values = []

    times = []

    for start in range(
        0,
        len(signal) - frame_size + 1,
        hop_size
    ):

        frame = signal[
            start:start + frame_size
        ]

        values.append(
            calculate_rms(frame)
        )

        times.append(
            (start + frame_size / 2)
            / SAMPLE_RATE
        )

    return (
        np.asarray(times),
        np.asarray(values)
    )


# =============================================================================
# SECTION 12 — EDUCATIONAL F0 ESTIMATION
# =============================================================================

def estimate_f0_autocorrelation(
    frame: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_f0: float = 60.0,
    max_f0: float = 400.0
) -> float:
    """
    Estimate F0 using autocorrelation.

    Steps:

        1. Remove DC component.
        2. Apply a Hann window.
        3. Compute autocorrelation.
        4. Search plausible pitch lags.
        5. Select the strongest autocorrelation peak.
        6. Convert lag into F0.

    Equation:

        F0 = sample_rate / lag

    Limitations:

        This simple method can produce octave errors.

        It is not reliable for unvoiced speech.

        Production pitch trackers use more advanced methods.
    """

    if len(frame) < 3:

        return 0.0

    frame = frame.astype(
        np.float64
    )

    frame = frame - np.mean(
        frame
    )

    frame *= np.hanning(
        len(frame)
    )

    if np.max(
        np.abs(frame)
    ) < 1e-5:

        return 0.0

    autocorrelation = np.correlate(
        frame,
        frame,
        mode="full"
    )

    autocorrelation = autocorrelation[
        len(autocorrelation) // 2:
    ]

    min_lag = int(
        sample_rate / max_f0
    )

    max_lag = int(
        sample_rate / min_f0
    )

    if max_lag >= len(autocorrelation):

        max_lag = len(autocorrelation) - 1

    if min_lag >= max_lag:

        return 0.0

    search_region = autocorrelation[
        min_lag:max_lag + 1
    ]

    best_lag = (
        np.argmax(search_region)
        + min_lag
    )

    if autocorrelation[0] <= 0:

        return 0.0

    confidence = (
        autocorrelation[best_lag]
        / autocorrelation[0]
    )

    if confidence < 0.25:

        return 0.0

    return float(
        sample_rate / best_lag
    )


def extract_f0_contour(
    signal: np.ndarray,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE
):
    """
    Estimate F0 frame by frame.

    Frames with insufficient periodicity are marked as zero.

    This is useful for visualizing the difference between voiced and
    unvoiced regions.
    """

    f0_values = []

    times = []

    for start in range(
        0,
        len(signal) - frame_size + 1,
        hop_size
    ):

        frame = signal[
            start:start + frame_size
        ]

        f0 = estimate_f0_autocorrelation(
            frame
        )

        f0_values.append(
            f0
        )

        times.append(
            (start + frame_size / 2)
            / SAMPLE_RATE
        )

    return (
        np.asarray(times),
        np.asarray(f0_values)
    )


# =============================================================================
# SECTION 13 — LANGUAGE DEMONSTRATIONS
# =============================================================================

def create_english_demo():
    """
    Demonstrate stress and sentence-level intonation.
    """

    return [
        Syllable("you", stress=0),

        Syllable("are", stress=0),

        Syllable(
            "READY",
            stress=1,
            emphasis=1.2
        )
    ]


def create_chinese_demo():
    """
    Demonstrate four Mandarin lexical tones.

    Each label is a syllable placeholder.

    Tone labels are the important information in this experiment.
    """

    return [
        Syllable("ma1", tone=1),

        Syllable("ma2", tone=2),

        Syllable("ma3", tone=3),

        Syllable(
            "ma4",
            tone=4,
            pause_after=0.15
        ),

        Syllable("ma5", tone=5)
    ]


def create_japanese_demo():
    """
    Demonstrate four simplified pitch-accent patterns.
    """

    return [
        Syllable(
            "heiban",
            accent="heiban"
        ),

        Syllable(
            "atamadaka",
            accent="atamadaka"
        ),

        Syllable(
            "nakadaka",
            accent="nakadaka"
        ),

        Syllable(
            "odaka",
            accent="odaka"
        )
    ]


def create_russian_demo():
    """
    Demonstrate stressed and unstressed syllables.
    """

    return [
        Syllable("pri", stress=0),

        Syllable(
            "VET",
            stress=1,
            emphasis=1.2
        ),

        Syllable("mi", stress=0),

        Syllable(
            "R",
            stress=1
        )
    ]


def get_language_demo(language: str):

    if language == "en":

        return create_english_demo()

    if language == "zh":

        return create_chinese_demo()

    if language == "ja":

        return create_japanese_demo()

    if language == "ru":

        return create_russian_demo()

    raise ValueError(
        f"Unsupported language: {language}"
    )


# =============================================================================
# SECTION 14 — VISUALIZATION
# =============================================================================

def plot_prosody(
    signal: np.ndarray,
    time_axis: np.ndarray,
    results: Sequence[SyllableResult],
    title: str
):
    """
    Display:

        1. Waveform
        2. Ground-truth generated F0 targets
        3. Estimated F0
        4. Short-time RMS
        5. Syllable durations

    Generated F0 targets are the values used to create the signal.

    Estimated F0 is obtained from the waveform using autocorrelation.

    They will not always match perfectly.
    """

    if len(signal) == 0:

        print("No signal to visualize.")

        return

    rms_times, rms_values = calculate_short_time_rms(
        signal
    )

    f0_times, f0_values = extract_f0_contour(
        signal
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 13)
    )

    fig.suptitle(
        title,
        fontsize=16
    )

    # -------------------------------------------------------------------------
    # PLOT 1 — WAVEFORM
    # -------------------------------------------------------------------------

    axes[0].plot(
        time_axis,
        signal,
        linewidth=0.5
    )

    axes[0].set_title(
        "Generated Speech-like Waveform"
    )

    axes[0].set_xlabel(
        "Time (seconds)"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[0].grid(
        alpha=0.3
    )

    # -------------------------------------------------------------------------
    # PLOT 2 — TARGET F0
    # -------------------------------------------------------------------------

    for result in results:

        midpoint = (
            result.start_time + result.end_time
        ) / 2

        axes[1].plot(
            [result.start_time, midpoint, result.end_time],
            [
                result.f0_start,
                (result.f0_start + result.f0_end) / 2,
                result.f0_end
            ],
            marker="o"
        )

        axes[1].text(
            midpoint,
            result.f0_end + 5,
            result.label,
            fontsize=8
        )

    axes[1].set_title(
        "Target F0 Contours"
    )

    axes[1].set_xlabel(
        "Time (seconds)"
    )

    axes[1].set_ylabel(
        "F0 (Hz)"
    )

    axes[1].grid(
        alpha=0.3
    )

    # -------------------------------------------------------------------------
    # PLOT 3 — ESTIMATED F0
    # -------------------------------------------------------------------------

    valid = f0_values > 0

    axes[2].plot(
        f0_times[valid],
        f0_values[valid],
        marker=".",
        linewidth=1
    )

    axes[2].set_title(
        "Estimated F0 — Autocorrelation"
    )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    axes[2].set_ylabel(
        "F0 (Hz)"
    )

    axes[2].grid(
        alpha=0.3
    )

    # -------------------------------------------------------------------------
    # PLOT 4 — ENERGY
    # -------------------------------------------------------------------------

    axes[3].plot(
        rms_times,
        rms_values
    )

    axes[3].set_title(
        "Short-Time RMS Energy"
    )

    axes[3].set_xlabel(
        "Time (seconds)"
    )

    axes[3].set_ylabel(
        "RMS"
    )

    axes[3].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

    # -------------------------------------------------------------------------
    # DURATION REPORT
    # -------------------------------------------------------------------------

    print("\nSYLLABLE DURATION REPORT")

    print("-" * 85)

    print(
        f"{'Syllable':15s}"
        f"{'Start':>10s}"
        f"{'End':>10s}"
        f"{'Duration':>12s}"
        f"{'Stress':>10s}"
    )

    for result in results:

        print(
            f"{result.label:15s}"
            f"{result.start_time:10.3f}"
            f"{result.end_time:10.3f}"
            f"{result.duration:12.3f}"
            f"{result.stress:10d}"
        )


# =============================================================================
# SECTION 15 — SPEAKING RATE EXPERIMENT
# =============================================================================

def experiment_speaking_rate():

    print("\nSPEAKING RATE EXPERIMENT")

    print("-" * 85)

    syllables = create_english_demo()

    rates = [
        0.6,
        0.8,
        1.0,
        1.2,
        1.5
    ]

    for rate in rates:

        config = ProsodyConfig(
            speaking_rate=rate
        )

        durations = [
            predict_syllable_duration(
                syllable,
                config
            )
            for syllable in syllables
        ]

        total = sum(durations)

        print(
            f"Rate: {rate:.1f} | "
            f"Total duration: {total:.3f}s | "
            f"Syllable durations: "
            f"{[round(x, 3) for x in durations]}"
        )


# =============================================================================
# SECTION 16 — PITCH EXPERIMENT
# =============================================================================

def experiment_pitch():

    print("\nPITCH EXPERIMENT")

    print("-" * 85)

    base_f0 = 150.0

    semitones = [
        -12,
        -6,
        0,
        6,
        12
    ]

    for shift in semitones:

        result = shift_pitch(
            base_f0,
            shift
        )

        print(
            f"{shift:+3d} semitones: "
            f"{result:.2f} Hz"
        )


# =============================================================================
# SECTION 17 — ENGLISH STATEMENT VS QUESTION
# =============================================================================

def experiment_english_intonation():

    syllables = create_english_demo()

    config = ProsodyConfig()

    statement_signal, statement_time, statement_results = (
        generate_prosody_sequence(
            syllables,
            "en",
            config,
            sentence_type="statement"
        )
    )

    question_signal, question_time, question_results = (
        generate_prosody_sequence(
            syllables,
            "en",
            config,
            sentence_type="question"
        )
    )

    plot_prosody(
        statement_signal,
        statement_time,
        statement_results,
        "English Statement"
    )

    plot_prosody(
        question_signal,
        question_time,
        question_results,
        "English Yes/No Question"
    )


# =============================================================================
# SECTION 18 — MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Educational multilingual TTS prosody simulator"
        )
    )

    parser.add_argument(
        "--language",
        choices=["en", "zh", "ja", "ru"],
        default="en"
    )

    parser.add_argument(
        "--sentence-type",
        choices=[
            "statement",
            "question",
            "exclamation"
        ],
        default="statement"
    )

    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--base-f0",
        type=float,
        default=150.0
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "none",
            "rate",
            "pitch",
            "intonation"
        ],
        default="none"
    )

    args = parser.parse_args()

    print("=" * 85)

    print("PHASE 1 — LESSON 12")

    print("MULTILINGUAL PROSODY SIMULATOR")

    print("=" * 85)

    if args.experiment == "rate":

        experiment_speaking_rate()

        return

    if args.experiment == "pitch":

        experiment_pitch()

        return

    if args.experiment == "intonation":

        experiment_english_intonation()

        return

    if args.speaking_rate <= 0:

        raise ValueError(
            "Speaking rate must be positive."
        )

    if args.base_f0 <= 0:

        raise ValueError(
            "Base F0 must be positive."
        )

    config = ProsodyConfig(
        base_f0=args.base_f0,
        speaking_rate=args.speaking_rate
    )

    syllables = get_language_demo(
        args.language
    )

    signal, time_axis, results = (
        generate_prosody_sequence(
            syllables,
            args.language,
            config,
            sentence_type=args.sentence_type
        )
    )

    print(
        f"\nLanguage: {args.language}"
    )

    print(
        f"Base F0: {args.base_f0} Hz"
    )

    print(
        f"Speaking rate: {args.speaking_rate}"
    )

    print(
        f"Sentence type: {args.sentence_type}"
    )

    print(
        f"Generated waveform samples: {len(signal)}"
    )

    print(
        f"Generated duration: {len(signal) / SAMPLE_RATE:.3f} seconds"
    )

    print(
        f"Waveform RMS: {calculate_rms(signal):.5f}"
    )

    plot_prosody(
        signal,
        time_axis,
        results,
        f"Prosody Simulation — {args.language}"
    )


if __name__ == "__main__":

    main()