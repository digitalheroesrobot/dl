#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 1 — DIGITAL AUDIO AND SPEECH FUNDAMENTALS
LESSON 1 — HOW A COMPUTER REPRESENTS A HUMAN VOICE
==================================================

PURPOSE
-------
This is the first practical program in the "Build Your Own Voice-Cloning
TTS System" course.

The goal is not to clone a voice yet. The goal is to understand the data
that every speech-processing and neural TTS system consumes.

This file is intentionally self-contained:
    1. It explains the important theory in comments.
    2. It loads a WAV file when a path is supplied.
    3. It creates a clearly-labelled synthetic demonstration signal when
       no WAV path is supplied.
    4. It inspects the signal numerically.
    5. It plots the waveform.
    6. It reports sample rate, channels, duration, sample count, peak,
       RMS, and approximate PCM bit depth when available.

RUN
---
Install:
    python -m pip install numpy matplotlib soundfile

Run with your own recording:
    python phase1_lesson1_digital_audio.py path/to/your_recording.wav

Run the built-in demonstration:
    python phase1_lesson1_digital_audio.py

Optional: save a normalized floating-point copy:
    python phase1_lesson1_digital_audio.py input.wav --save-copy output.wav

SUPPORTED INPUT
---------------
This first lesson intentionally focuses on WAV/PCM files. Later lessons
will cover codecs, resampling, filtering, spectrograms, and dataset formats.

IMPORTANT
---------
The synthetic demonstration is a mixture of tones, NOT human speech.
For meaningful speech inspection, use a recording of your own voice or a
recording for which you have permission to use it.

============================================================================
THEORY NOTES — READ THESE COMMENTS AS PART OF THE LESSON
============================================================================

1. WHAT IS SOUND?
-----------------
Sound is a mechanical wave: variations in pressure travel through a medium
such as air. A microphone converts these pressure variations into an
electrical signal.

A simplified sinusoidal signal is:

    x(t) = A * sin(2*pi*f*t + phi)

where:
    x(t)  = signal amplitude at time t
    A     = amplitude
    f     = frequency in Hertz (cycles per second)
    t     = time in seconds
    phi   = phase in radians

Real speech is not one sine wave. It is a complex, changing signal formed
by many frequency components. Vocal-fold vibration, vocal-tract resonances,
articulation, breath, and recording conditions all affect the waveform.

2. ANALOG-TO-DIGITAL CONVERSION (ADC)
-------------------------------------
A computer cannot directly store a continuously varying analog signal as
an infinite set of values. An ADC produces a digital representation through
two main operations:

    A. Sampling:
       Measure the analog signal at regularly spaced points in time.

    B. Quantization:
       Map each measured amplitude to one of a finite set of numerical
       levels.

Encoding then stores those numerical values as bits.

3. SAMPLE RATE
--------------
The sample rate (fs) is the number of samples recorded per second.

    N = fs * T

where:
    N  = number of samples per channel
    fs = sample rate in samples/second (Hz)
    T  = duration in seconds

Example:
    fs = 24,000 samples/second
    T  = 5 seconds
    N  = 120,000 samples per channel

The time between adjacent samples is:

    Ts = 1 / fs

At 24 kHz, Ts is approximately 41.67 microseconds.

A sample rate of 24 kHz does NOT mean that the audio contains only a
24 kHz tone. It means the signal is measured 24,000 times each second.

4. NYQUIST-SHANNON SAMPLING THEOREM — PREVIEW
---------------------------------------------
For a band-limited signal, faithful reconstruction is possible in theory
when the sampling frequency is greater than twice the highest frequency
present in the signal:

    fs > 2 * f_max

The Nyquist frequency is:

    f_Nyquist = fs / 2

Frequencies above the Nyquist limit can alias into lower frequencies if
they are not removed before sampling. Real ADC systems therefore use
anti-aliasing filters.

We will derive and simulate this in Lesson 2. Merely changing the sample
rate number in a file header is NOT proper resampling.

5. BIT DEPTH AND QUANTIZATION — PREVIEW
--------------------------------------
For a uniform b-bit representation, the number of possible quantization
codes is:

    L = 2**b

For example:
    8-bit  -> 256 codes
    16-bit -> 65,536 codes
    24-bit -> 16,777,216 codes

Signed 16-bit PCM values range from -32768 to +32767. Many Python audio
libraries can convert integer PCM into floating-point samples, commonly
scaled approximately to [-1.0, +1.0].

Floating-point audio is not the same thing as 32-bit integer PCM. A WAV
file can contain different encodings; inspect the subtype rather than
guessing from the NumPy dtype alone.

6. CHANNELS: MONO AND STEREO
----------------------------
Mono contains one channel. Stereo contains two channels, conventionally
left and right.

A mono waveform has conceptual shape:
    (num_samples,)

A stereo waveform often has shape:
    (num_samples, 2)

Some libraries can return arrays with channel-first layout instead. Always
inspect the shape and library documentation before assuming the axis order.

Do not average channels automatically if spatial information matters.
For many speech tasks, mono is used, but conversion should be deliberate.

7. DURATION
-----------
For N samples per channel at sample rate fs:

    duration_seconds = N / fs

The number of channels does not multiply the duration.

8. AMPLITUDE, PEAK, AND RMS
---------------------------
Digital sample amplitude represents signal level according to the audio
encoding and scaling convention.

Peak amplitude:
    peak = max(abs(x[n]))

Root-mean-square amplitude:
    RMS = sqrt(mean(x[n]**2))

Peak describes the largest absolute sample. RMS gives a measure of average
signal energy/amplitude over the selected segment.

For normalized floating-point audio, samples are commonly expected near
[-1, +1], although some processing pipelines can exceed this range.

9. CLIPPING — PREVIEW
---------------------
Clipping occurs when a signal exceeds the representable range of a system
and is truncated or saturated. It creates distortion. A digital peak at or
near full scale can be a warning, but it does not alone prove clipping.
We will study clipping and signal-to-noise ratio in Lesson 6.

10. WAV IS A CONTAINER
----------------------
WAV is a container format, commonly used for PCM audio. The file contains
metadata such as sample rate, channel count, and encoding information,
followed by audio data (and potentially other chunks).

"16-bit WAV" and "24 kHz WAV" describe different properties:
    16-bit  -> sample encoding precision
    24 kHz  -> sampling frequency

11. WHY THIS MATTERS FOR VOICE CLONING
-------------------------------------
A TTS training pipeline must agree on audio conventions:
    - sample rate
    - channel layout
    - amplitude scaling
    - sample encoding
    - segment duration
    - preprocessing rules

If training audio is inconsistent or interpreted incorrectly, the model
may learn from corrupted inputs or targets. This can damage intelligibility,
speaker similarity, and generated audio quality.

In later phases:
    waveform -> audio features -> neural model -> generated features
              -> vocoder -> generated waveform

The waveform is the foundation of the entire pipeline.

============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def make_demo_signal(sample_rate: int = 24_000, duration: float = 2.0) -> np.ndarray:
    """
    Create a synthetic, two-tone demonstration signal.

    This is deliberately not speech. It makes the waveform inspector usable
    even before the learner has recorded a voice sample.
    """
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate

    # Two sine waves illustrate that a signal can contain multiple
    # frequency components. A slow envelope makes the plot easier to inspect.
    envelope = 0.65 + 0.25 * np.sin(2 * np.pi * 1.0 * t)
    signal = envelope * (
        0.55 * np.sin(2 * np.pi * 220.0 * t)
        + 0.25 * np.sin(2 * np.pi * 440.0 * t)
    )

    # Keep the demonstration safely within normalized full scale.
    signal /= max(1.0, np.max(np.abs(signal)) / 0.95)
    return signal.astype(np.float32)


def load_audio(path: Path):
    """Load audio as float32 while preserving the channel dimension."""
    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    # soundfile returns shape (frames, channels) with always_2d=True.
    return audio, int(sample_rate)


def calculate_rms(audio: np.ndarray) -> np.ndarray:
    """Calculate RMS independently for each channel."""
    return np.sqrt(np.mean(np.square(audio.astype(np.float64)), axis=0))


def dbfs(value: float, floor_db: float = -120.0) -> float:
    """Convert a normalized amplitude value to dBFS."""
    if value <= 0:
        return floor_db
    return max(20.0 * np.log10(value), floor_db)


def inspect_audio(audio: np.ndarray, sample_rate: int, source_label: str,
                  subtype: str = "synthetic float32"):
    """Print important audio metadata and numerical measurements."""
    frames, channels = audio.shape
    duration = frames / sample_rate
    peak_by_channel = np.max(np.abs(audio), axis=0)
    rms_by_channel = calculate_rms(audio)

    print("\n" + "=" * 68)
    print("DIGITAL AUDIO INSPECTION REPORT")
    print("=" * 68)
    print(f"Source                 : {source_label}")
    print(f"Sample rate            : {sample_rate:,} Hz")
    print(f"Channels               : {channels}")
    print(f"Frames per channel     : {frames:,}")
    print(f"Duration               : {duration:.6f} seconds")
    print(f"Array shape            : {audio.shape}")
    print(f"NumPy dtype            : {audio.dtype}")
    print(f"WAV subtype            : {subtype}")
    print(f"Time between samples   : {1.0 / sample_rate * 1e6:.3f} microseconds")
    print(f"Nyquist frequency      : {sample_rate / 2:,.1f} Hz")

    for channel in range(channels):
        print(f"\nChannel {channel + 1}:")
        print(f"  Peak amplitude       : {peak_by_channel[channel]:.6f}")
        print(f"  Peak level           : {dbfs(float(peak_by_channel[channel])):.2f} dBFS")
        print(f"  RMS amplitude        : {rms_by_channel[channel]:.6f}")
        print(f"  RMS level            : {dbfs(float(rms_by_channel[channel])):.2f} dBFS")

    if np.any(peak_by_channel >= 0.999):
        print("\nWarning: at least one channel has samples very close to full scale.")
        print("This may warrant a clipping check in a later lesson.")

    print("\nFirst 10 frames:")
    print(audio[:10])
    print("=" * 68)


def plot_waveform(audio: np.ndarray, sample_rate: int, source_label: str,
                  max_plot_seconds: float = 0.25):
    """
    Plot a short section of the waveform and a longer overview.

    A short view makes individual sample behavior easier to see.
    The full overview shows the recording's overall amplitude pattern.
    """
    frames, channels = audio.shape
    duration = frames / sample_rate
    time = np.arange(frames) / sample_rate

    short_frames = min(frames, int(max_plot_seconds * sample_rate))
    short_time = time[:short_frames]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 8),
        constrained_layout=True,
    )

    for channel in range(channels):
        axes[0].plot(
            time,
            audio[:, channel],
            linewidth=0.65,
            label=f"Channel {channel + 1}",
        )
    axes[0].set_title(f"Full waveform — {source_label}")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    if channels > 1:
        axes[0].legend()

    for channel in range(channels):
        axes[1].plot(
            short_time,
            audio[:short_frames, channel],
            linewidth=0.9,
            label=f"Channel {channel + 1}",
        )
    axes[1].set_title(
        f"Waveform detail — first {short_frames / sample_rate:.4f} seconds"
    )
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    if channels > 1:
        axes[1].legend()

    fig.suptitle(
        f"Digital Audio Waveform | {sample_rate:,} Hz | "
        f"{channels} channel(s) | {duration:.3f} s",
        fontsize=13,
    )
    plt.show()


def save_float_copy(audio: np.ndarray, sample_rate: int, output_path: Path):
    """
    Save a float WAV copy without changing the sample rate or channel count.

    This is not resampling and not normalization. It is a simple export
    intended to demonstrate reading and writing audio.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate, subtype="FLOAT")
    print(f"\nSaved floating-point WAV copy: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1 Lesson 1: inspect a WAV recording."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        type=Path,
        help="Optional path to an input WAV file. If omitted, a synthetic demo is used.",
    )
    parser.add_argument(
        "--save-copy",
        type=Path,
        default=None,
        help="Optional output path for a floating-point WAV copy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.audio_file is None:
        sample_rate = 24_000
        audio_mono = make_demo_signal(sample_rate=sample_rate)
        audio = audio_mono[:, np.newaxis]
        source_label = "Built-in synthetic demonstration (not speech)"
        subtype = "synthetic float32"
        print("No input file supplied. Using a synthetic demonstration signal.")
        print("For speech analysis, run this program with your own WAV recording.")
    else:
        if not args.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {args.audio_file}")
        if not args.audio_file.is_file():
            raise ValueError(f"Input path is not a file: {args.audio_file}")

        audio, sample_rate = load_audio(args.audio_file)
        info = sf.info(str(args.audio_file))
        source_label = str(args.audio_file)
        subtype = info.subtype

    if audio.size == 0:
        raise ValueError("The audio file contains no samples.")
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite values.")

    inspect_audio(audio, sample_rate, source_label, subtype)
    plot_waveform(audio, sample_rate, source_label)

    if args.save_copy is not None:
        save_float_copy(audio, sample_rate, args.save_copy)

    print("\nLESSON 1 CHECKPOINT")
    print("1. Identify the sample rate and channel count.")
    print("2. Explain why duration = frames / sample_rate.")
    print("3. Explain the difference between sample rate and bit depth.")
    print("4. Explain why the demo signal is not a voice recording.")
    print("5. Run the program on a permitted recording of your own voice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)