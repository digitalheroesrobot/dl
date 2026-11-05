"""
=======================================================================
BUILD YOUR OWN VOICE-CLONING TTS SYSTEM
PHASE 1 — DIGITAL AUDIO FUNDAMENTALS

LESSON 5:
STFT, SPECTROGRAMS, MEL SCALE AND LOG-MEL SPECTROGRAMS
=======================================================================

LEARNING OBJECTIVES
-------------------

1. Understand why a normal FFT is insufficient for analyzing speech
   that changes over time.

2. Understand framing and overlapping audio windows.

3. Understand the Short-Time Fourier Transform (STFT).

4. Understand the relationship between:
       Sampling rate
       FFT size
       Window length
       Hop length
       Frequency resolution
       Time resolution

5. Understand magnitude and power spectrograms.

6. Understand the Mel frequency scale.

7. Understand triangular Mel filterbanks.

8. Understand Mel spectrogram generation.

9. Understand logarithmic compression.

10. Understand log-Mel spectrograms.

11. Visualize speech-related frequency patterns.

12. Understand why Mel spectrograms are important in neural TTS.

=======================================================================
1. WHY DO WE NEED STFT?
=======================================================================

A normal FFT analyzes an entire signal segment.

Suppose we record a one-second audio signal.

A single FFT gives us information about frequencies present in
that one-second segment.

However, speech is non-stationary.

Its frequency components change over time.

For example:

    Time 0.00 - 0.10: consonant
    Time 0.10 - 0.25: vowel
    Time 0.25 - 0.40: another consonant
    Time 0.40 - 0.60: another vowel

A single FFT combines frequency information from all these regions.

It cannot show exactly when each component occurs.

The Short-Time Fourier Transform solves this problem.

STFT:

    1. Divide audio into short frames.
    2. Apply a window function to each frame.
    3. Calculate FFT for every frame.
    4. Move the frame forward.
    5. Repeat.

The result is a time-frequency representation.

=======================================================================
2. FRAMING
=======================================================================

Suppose:

    Sampling rate = 24,000 Hz
    Frame length = 1024 samples
    Hop length = 256 samples

Frame duration:

    frame_duration = frame_length / sampling_rate

                     = 1024 / 24000

                     = 42.67 ms

Hop duration:

    hop_duration = hop_length / sampling_rate

                  = 256 / 24000

                  = 10.67 ms

Therefore, a new analysis frame begins every 10.67 ms.

Because the frame is 42.67 ms long, adjacent frames overlap.

Overlap:

    overlap = frame_length - hop_length

            = 1024 - 256

            = 768 samples

Overlap percentage:

    overlap_percent = overlap / frame_length * 100

                    = 75%

Overlapping frames provide a more continuous representation
of changing speech.

=======================================================================
3. WINDOWING
=======================================================================

Each extracted frame is multiplied by a window function.

Why?

An abrupt beginning or ending of a frame can introduce spectral
leakage.

The Hann window is commonly used:

    w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))

Windowed signal:

    x_windowed[n] = x[n] * w[n]

Windowing reduces boundary discontinuities.

However, it also affects amplitude and spectral characteristics.

=======================================================================
4. STFT
=======================================================================

The STFT is:

    X(m,k) = sum(
        x[n + mH] * w[n] * exp(-j*2*pi*k*n/N)
    )

where:

    m = frame index
    k = frequency bin
    H = hop length
    N = FFT size
    w[n] = window function

For real-valued audio, only the non-negative frequencies
are normally required.

Number of frequency bins:

    N_fft / 2 + 1

For N_fft = 1024:

    1024 / 2 + 1 = 513 bins

Frequency spacing:

    delta_f = sampling_rate / N_fft

For fs = 24,000 Hz:

    delta_f = 23.4375 Hz

=======================================================================
5. SPECTROGRAM
=======================================================================

A spectrogram visualizes signal energy over time and frequency.

Magnitude spectrogram:

    S(m,k) = abs(X(m,k))

Power spectrogram:

    P(m,k) = abs(X(m,k))^2

The horizontal axis represents time.

The vertical axis represents frequency.

The color intensity represents magnitude or power.

=======================================================================
6. MEL SCALE
=======================================================================

Human frequency perception is approximately nonlinear.

The Mel scale is a perceptual frequency scale.

Conversion:

    mel(f) = 2595 * log10(1 + f/700)

Inverse:

    f = 700 * (10^(mel/2595) - 1)

Mel filterbanks combine neighboring frequency bins into
perceptually spaced frequency bands.

The filters are commonly triangular.

The output has fewer frequency dimensions than the original FFT.

=======================================================================
7. MEL SPECTROGRAM
=======================================================================

Let:

    X = STFT
    P = abs(X)^2
    W = Mel filterbank matrix

Then:

    Mel = W @ P

The result is a time-by-Mel-frequency representation.

The exact matrix orientation depends on implementation.

=======================================================================
8. LOG-MEL SPECTROGRAM
=======================================================================

Mel energies can vary substantially.

Log compression reduces the dynamic range.

    LogMel = log(Mel + epsilon)

A decibel representation can also be used:

    Mel_dB = 10 * log10(Mel / reference)

Log-Mel features are commonly used in speech recognition
and speech representation learning.

TTS systems often use Mel spectrograms as acoustic targets.

=======================================================================
9. IMPORTANT DISTINCTION
=======================================================================

A Mel spectrogram is NOT the same as a waveform.

It is a transformed representation of audio.

A TTS acoustic model may predict a Mel spectrogram.

A neural vocoder then converts that representation into waveform audio.

Example:

    Text
      |
      v
    Text Encoder
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
    Audio Waveform

Not every TTS architecture uses Mel spectrograms.

Some systems predict waveforms, codec tokens, or other acoustic
representations directly.

=======================================================================
10. VOICE-CLONING APPLICATION
=======================================================================

Mel spectrograms preserve important time-frequency information.

They are useful for learning:

    - Vowel structure
    - Formant patterns
    - Speech timing
    - Harmonic structure
    - Acoustic transitions
    - Speaker-dependent acoustic characteristics

However, a Mel spectrogram alone does not uniquely encode speaker
identity.

Speaker identity also depends on other acoustic and linguistic
information.

=======================================================================
DEPENDENCIES
=======================================================================

    pip install numpy matplotlib scipy soundfile

=======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import argparse

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf

from scipy.signal import stft


# =====================================================================
# CONFIGURATION
# =====================================================================

EPSILON = 1e-10

DEFAULT_FS = 24000

DEFAULT_N_FFT = 1024

DEFAULT_HOP_LENGTH = 256

DEFAULT_N_MELS = 80

DEFAULT_FMIN = 0.0

DEFAULT_FMAX = 12000.0


# =====================================================================
# AUDIO GENERATION
# =====================================================================

def generate_test_audio(fs=DEFAULT_FS, duration=2.0):
    """
    Generate an educational speech-like signal.

    This is NOT real human speech.

    It combines:

        - Harmonic excitation
        - Changing frequency components
        - Amplitude envelopes
        - Noise-like components

    The purpose is to demonstrate how time-frequency analysis works.

    Real speech will be examined in Experiment 7.
    """

    number_of_samples = int(fs * duration)

    time = np.arange(number_of_samples) / fs

    signal = np.zeros(number_of_samples)

    # Segment 1: 180 Hz fundamental.

    mask1 = (time >= 0.0) & (time < 0.6)

    t1 = time[mask1]

    signal[mask1] = (
        np.sin(2 * np.pi * 180 * t1)
        + 0.5 * np.sin(2 * np.pi * 360 * t1)
        + 0.25 * np.sin(2 * np.pi * 540 * t1)
    )

    # Segment 2: 250 Hz fundamental.

    mask2 = (time >= 0.7) & (time < 1.3)

    t2 = time[mask2]

    signal[mask2] = (
        np.sin(2 * np.pi * 250 * t2)
        + 0.5 * np.sin(2 * np.pi * 500 * t2)
        + 0.25 * np.sin(2 * np.pi * 750 * t2)
    )

    # Segment 3: multiple frequency components.

    mask3 = (time >= 1.4) & (time < 2.0)

    t3 = time[mask3]

    signal[mask3] = (
        0.7 * np.sin(2 * np.pi * 400 * t3)
        + 0.4 * np.sin(2 * np.pi * 1200 * t3)
        + 0.2 * np.sin(2 * np.pi * 2400 * t3)
    )

    # Smooth amplitude envelope.

    envelope = np.ones_like(signal)

    fade_length = int(0.03 * fs)

    fade = np.linspace(
        0,
        1,
        fade_length
    )

    envelope[:fade_length] = fade

    envelope[-fade_length:] = fade[::-1]

    signal *= envelope

    # Normalize.

    signal /= np.max(
        np.abs(signal)
    ) + EPSILON

    return signal.astype(np.float32)


# =====================================================================
# AUDIO LOADING
# =====================================================================

def load_audio(filename, target_fs=None):
    """
    Load a WAV file.

    Stereo audio is converted to mono using channel averaging.

    This is acceptable for many demonstrations, but production
    dataset preprocessing should inspect channel quality and
    channel alignment before downmixing.

    If target_fs is specified, this function does NOT silently
    resample the signal.

    A sample-rate mismatch is reported instead.
    """

    audio, fs = sf.read(
        filename,
        dtype="float32",
        always_2d=True
    )

    audio = np.mean(
        audio,
        axis=1
    )

    if target_fs is not None and fs != target_fs:

        raise ValueError(
            f"Audio sample rate is {fs} Hz, "
            f"but expected {target_fs} Hz. "
            "Resample the audio explicitly before analysis."
        )

    return audio, fs


# =====================================================================
# FRAME EXTRACTION
# =====================================================================

def extract_frames(
    signal,
    frame_length,
    hop_length
):
    """
    Extract overlapping frames.

    Frame m begins at:

        start = m * hop_length

    Each frame contains frame_length samples.

    This implementation uses zero-padding at the end when necessary.

    It does not center-pad the signal at the beginning.
    """

    signal = np.asarray(
        signal,
        dtype=np.float32
    )

    if frame_length <= 0 or hop_length <= 0:

        raise ValueError(
            "Frame length and hop length must be positive."
        )

    if len(signal) == 0:

        raise ValueError(
            "Input signal is empty."
        )

    number_of_frames = max(
        1,
        int(np.ceil(
            max(0, len(signal) - frame_length)
            / hop_length
        )) + 1
    )

    required_length = (
        (number_of_frames - 1) * hop_length
        + frame_length
    )

    if len(signal) < required_length:

        signal = np.pad(
            signal,
            (0, required_length - len(signal))
        )

    frames = np.zeros(
        (number_of_frames, frame_length),
        dtype=np.float32
    )

    for index in range(number_of_frames):

        start = index * hop_length

        end = start + frame_length

        frames[index] = signal[start:end]

    return frames


# =====================================================================
# HANN WINDOW
# =====================================================================

def apply_hann_window(frames):
    """
    Apply a Hann window to each frame.

    Hann window:

        w[n] = 0.5 * (
            1 - cos(2*pi*n/(N-1))
        )
    """

    frame_length = frames.shape[1]

    window = np.hanning(
        frame_length
    ).astype(np.float32)

    windowed_frames = (
        frames * window[None, :]
    )

    return windowed_frames, window


# =====================================================================
# MEL SCALE
# =====================================================================

def hz_to_mel(frequency):
    """
    Convert frequency in Hz to Mel.

        mel = 2595 * log10(1 + f/700)
    """

    frequency = np.asarray(
        frequency,
        dtype=np.float64
    )

    return (
        2595.0
        * np.log10(1.0 + frequency / 700.0)
    )


def mel_to_hz(mel):
    """
    Convert Mel to Hz.

        f = 700 * (10^(mel/2595) - 1)
    """

    mel = np.asarray(
        mel,
        dtype=np.float64
    )

    return (
        700.0
        * (10.0 ** (mel / 2595.0) - 1.0)
    )


# =====================================================================
# MEL FILTERBANK
# =====================================================================

def create_mel_filterbank(
    fs,
    n_fft,
    n_mels=80,
    fmin=0.0,
    fmax=None
):
    """
    Create triangular Mel filters.

    Output shape:

        (n_mels, n_fft // 2 + 1)

    Each row represents one Mel filter.

    Each column represents one FFT frequency bin.

    This is a conventional triangular filterbank implementation.

    Different TTS implementations may use different filterbank
    normalization conventions.

    The filterbank here is not intended to reproduce every library's
    exact Mel feature values.
    """

    if fmax is None:

        fmax = fs / 2

    if not (0 <= fmin < fmax <= fs / 2):

        raise ValueError(
            "Require 0 <= fmin < fmax <= Nyquist frequency."
        )

    if n_mels <= 0:

        raise ValueError(
            "Number of Mel filters must be positive."
        )

    number_of_frequency_bins = (
        n_fft // 2 + 1
    )

    fft_frequencies = np.linspace(
        0,
        fs / 2,
        number_of_frequency_bins
    )

    mel_min = hz_to_mel(fmin)

    mel_max = hz_to_mel(fmax)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2
    )

    hz_points = mel_to_hz(
        mel_points
    )

    filterbank = np.zeros(
        (
            n_mels,
            number_of_frequency_bins
        ),
        dtype=np.float64
    )

    for mel_index in range(n_mels):

        left = hz_points[mel_index]

        center = hz_points[mel_index + 1]

        right = hz_points[mel_index + 2]

        rising = (
            (fft_frequencies - left)
            / max(center - left, EPSILON)
        )

        falling = (
            (right - fft_frequencies)
            / max(right - center, EPSILON)
        )

        filter_values = np.maximum(
            0.0,
            np.minimum(rising, falling)
        )

        filterbank[mel_index] = filter_values

    return filterbank, hz_points


# =====================================================================
# STFT
# =====================================================================

def calculate_stft(
    signal,
    fs,
    n_fft=1024,
    hop_length=256
):
    """
    Calculate STFT using SciPy.

    scipy.signal.stft returns:

        frequencies
        times
        complex_stft

    Boundary padding and scaling are explicitly configured.

    This implementation uses:

        Hann window
        n_fft samples per window
        hop_length through noverlap
        zero boundary extension
        padded final frame

    Note:

    For production TTS preprocessing, the exact STFT conventions
    must match the model's expected preprocessing.
    """

    if n_fft <= 0:

        raise ValueError(
            "n_fft must be positive."
        )

    if hop_length <= 0 or hop_length > n_fft:

        raise ValueError(
            "hop_length must be between 1 and n_fft."
        )

    frequencies, times, spectrum = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
        scaling="spectrum"
    )

    return frequencies, times, spectrum


# =====================================================================
# MAGNITUDE AND POWER SPECTROGRAM
# =====================================================================

def calculate_spectrogram(spectrum):
    """
    Magnitude:

        magnitude = abs(STFT)

    Power:

        power = magnitude ** 2
    """

    magnitude = np.abs(
        spectrum
    )

    power = np.square(
        magnitude
    )

    return magnitude, power


# =====================================================================
# MEL SPECTROGRAM
# =====================================================================

def calculate_mel_spectrogram(
    power_spectrogram,
    filterbank
):
    """
    Apply the Mel filterbank.

    Matrix operation:

        Mel = Filterbank @ PowerSpectrogram

    Shapes:

        Filterbank:
            (n_mels, frequency_bins)

        Power:
            (frequency_bins, time_frames)

        Result:
            (n_mels, time_frames)
    """

    mel_spectrogram = (
        filterbank @ power_spectrogram
    )

    return mel_spectrogram


# =====================================================================
# LOG-MEL
# =====================================================================

def calculate_log_mel(mel_spectrogram):
    """
    Log compression:

        LogMel = log(Mel + epsilon)

    Natural logarithm is used here.

    Some implementations use log10 or decibel conversion.

    The exact choice matters when training a neural model.
    """

    return np.log(
        mel_spectrogram + EPSILON
    )


def amplitude_to_db(amplitude):
    """
    Convert amplitude to decibels.

        dB = 20 * log10(amplitude)
    """

    return 20.0 * np.log10(
        np.maximum(
            amplitude,
            EPSILON
        )
    )


def power_to_db(power):
    """
    Convert power to decibels.

        dB = 10 * log10(power)
    """

    return 10.0 * np.log10(
        np.maximum(
            power,
            EPSILON
        )
    )


# =====================================================================
# EXPERIMENT 1
# FRAMING AND WINDOWING
# =====================================================================

def experiment_framing():

    print("\nEXPERIMENT 1: FRAMING AND WINDOWING")

    fs = DEFAULT_FS

    signal = generate_test_audio(
        fs,
        duration=0.5
    )

    frame_length = 1024

    hop_length = 256

    frames = extract_frames(
        signal,
        frame_length,
        hop_length
    )

    windowed_frames, window = apply_hann_window(
        frames
    )

    print(
        f"Sampling rate: {fs} Hz"
    )

    print(
        f"Frame length: {frame_length} samples"
    )

    print(
        f"Frame duration: "
        f"{1000 * frame_length / fs:.2f} ms"
    )

    print(
        f"Hop length: {hop_length} samples"
    )

    print(
        f"Hop duration: "
        f"{1000 * hop_length / fs:.2f} ms"
    )

    print(
        f"Overlap: "
        f"{100 * (frame_length-hop_length)/frame_length:.1f}%"
    )

    print(
        f"Number of frames: {len(frames)}"
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10)
    )

    time = np.arange(len(signal)) / fs

    axes[0].plot(
        time,
        signal
    )

    axes[0].set_title(
        "Original Audio"
    )

    axes[0].set_xlabel("Time (seconds)")

    axes[0].grid(True)

    axes[1].plot(
        np.arange(frame_length) / fs,
        frames[0]
    )

    axes[1].set_title(
        "First Extracted Frame"
    )

    axes[1].set_xlabel("Time (seconds)")

    axes[1].grid(True)

    axes[2].plot(
        np.arange(frame_length) / fs,
        frames[0],
        label="Original frame"
    )

    axes[2].plot(
        np.arange(frame_length) / fs,
        window,
        label="Hann window"
    )

    axes[2].plot(
        np.arange(frame_length) / fs,
        windowed_frames[0],
        label="Windowed frame"
    )

    axes[2].set_title(
        "Windowing Effect"
    )

    axes[2].set_xlabel("Time (seconds)")

    axes[2].legend()

    axes[2].grid(True)

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 2
# STFT
# =====================================================================

def experiment_stft():

    print("\nEXPERIMENT 2: STFT")

    fs = DEFAULT_FS

    signal = generate_test_audio(
        fs,
        duration=2.0
    )

    frequencies, times, spectrum = calculate_stft(
        signal,
        fs
    )

    print(
        f"STFT shape: {spectrum.shape}"
    )

    print(
        f"Frequency bins: {len(frequencies)}"
    )

    print(
        f"Time frames: {len(times)}"
    )

    print(
        f"Frequency resolution: "
        f"{fs / DEFAULT_N_FFT:.2f} Hz"
    )

    plt.figure(figsize=(14, 7))

    plt.imshow(
        np.abs(spectrum),
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            frequencies[0],
            frequencies[-1]
        ],
        cmap="magma"
    )

    plt.ylim(
        0,
        5000
    )

    plt.colorbar(
        label="Magnitude"
    )

    plt.title(
        "STFT Magnitude Representation"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 3
# LINEAR MAGNITUDE SPECTROGRAM
# =====================================================================

def experiment_magnitude_spectrogram():

    print("\nEXPERIMENT 3: MAGNITUDE SPECTROGRAM")

    fs = DEFAULT_FS

    signal = generate_test_audio(
        fs,
        duration=2.0
    )

    frequencies, times, spectrum = calculate_stft(
        signal,
        fs
    )

    magnitude, power = calculate_spectrogram(
        spectrum
    )

    plt.figure(figsize=(14, 7))

    plt.imshow(
        amplitude_to_db(magnitude),
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            frequencies[0],
            frequencies[-1]
        ],
        cmap="magma",
        vmin=-100,
        vmax=0
    )

    plt.ylim(
        0,
        8000
    )

    plt.colorbar(
        label="Magnitude (dB)"
    )

    plt.title(
        "Linear-Frequency Magnitude Spectrogram"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.tight_layout()

    plt.show()

    print(
        f"Magnitude shape: {magnitude.shape}"
    )

    print(
        f"Power shape: {power.shape}"
    )


# =====================================================================
# EXPERIMENT 4
# MEL FILTERBANK
# =====================================================================

def experiment_mel_filterbank():

    print("\nEXPERIMENT 4: MEL FILTERBANK")

    fs = DEFAULT_FS

    n_fft = DEFAULT_N_FFT

    n_mels = DEFAULT_N_MELS

    filterbank, hz_points = create_mel_filterbank(
        fs,
        n_fft,
        n_mels
    )

    frequencies = np.linspace(
        0,
        fs / 2,
        n_fft // 2 + 1
    )

    plt.figure(figsize=(14, 7))

    for index in range(n_mels):

        plt.plot(
            frequencies,
            filterbank[index],
            alpha=0.7
        )

    plt.title(
        "Triangular Mel Filterbank"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Filter Weight"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    print(
        f"Number of Mel filters: {n_mels}"
    )

    print(
        f"Filterbank shape: {filterbank.shape}"
    )

    print("\nMel filter center frequencies:")

    for index, frequency in enumerate(
        hz_points[1:-1]
    ):

        print(
            f"Filter {index + 1:02d}: "
            f"{frequency:.2f} Hz"
        )


# =====================================================================
# EXPERIMENT 5
# MEL AND LOG-MEL SPECTROGRAM
# =====================================================================

def experiment_log_mel():

    print("\nEXPERIMENT 5: LOG-MEL SPECTROGRAM")

    fs = DEFAULT_FS

    signal = generate_test_audio(
        fs,
        duration=2.0
    )

    frequencies, times, spectrum = calculate_stft(
        signal,
        fs
    )

    magnitude, power = calculate_spectrogram(
        spectrum
    )

    filterbank, _ = create_mel_filterbank(
        fs,
        DEFAULT_N_FFT,
        DEFAULT_N_MELS,
        DEFAULT_FMIN,
        DEFAULT_FMAX
    )

    mel = calculate_mel_spectrogram(
        power,
        filterbank
    )

    log_mel = calculate_log_mel(
        mel
    )

    mel_db = power_to_db(
        mel
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 13)
    )

    axes[0].imshow(
        magnitude,
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            frequencies[0],
            frequencies[-1]
        ]
    )

    axes[0].set_ylim(
        0,
        8000
    )

    axes[0].set_title(
        "Magnitude Spectrogram"
    )

    axes[0].set_ylabel(
        "Frequency (Hz)"
    )

    axes[1].imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            0,
            DEFAULT_N_MELS
        ]
    )

    axes[1].set_title(
        "Mel Spectrogram (dB)"
    )

    axes[1].set_ylabel(
        "Mel Filter Index"
    )

    axes[2].imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            0,
            DEFAULT_N_MELS
        ],
        cmap="magma"
    )

    axes[2].set_title(
        "Natural Log-Mel Spectrogram"
    )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    axes[2].set_ylabel(
        "Mel Filter Index"
    )

    plt.tight_layout()

    plt.show()

    print(
        f"Original STFT shape: {spectrum.shape}"
    )

    print(
        f"Mel shape: {mel.shape}"
    )

    print(
        f"Log-Mel shape: {log_mel.shape}"
    )


# =====================================================================
# EXPERIMENT 6
# FFT SIZE AND HOP LENGTH
# =====================================================================

def experiment_resolution():

    print("\nEXPERIMENT 6: FFT SIZE AND HOP LENGTH")

    fs = DEFAULT_FS

    signal = generate_test_audio(
        fs,
        duration=1.0
    )

    configurations = [
        (512, 128),
        (1024, 256),
        (2048, 512),
        (2048, 256)
    ]

    fig, axes = plt.subplots(
        len(configurations),
        1,
        figsize=(14, 14)
    )

    for index, (n_fft, hop) in enumerate(
        configurations
    ):

        frequencies, times, spectrum = calculate_stft(
            signal,
            fs,
            n_fft,
            hop
        )

        magnitude_db = amplitude_to_db(
            np.abs(spectrum)
        )

        axes[index].imshow(
            magnitude_db,
            origin="lower",
            aspect="auto",
            extent=[
                times[0],
                times[-1],
                frequencies[0],
                frequencies[-1]
            ],
            vmin=-100,
            vmax=0
        )

        axes[index].set_ylim(
            0,
            8000
        )

        axes[index].set_title(
            f"FFT={n_fft}, Hop={hop}, "
            f"Frequency Resolution={fs/n_fft:.2f} Hz"
        )

        axes[index].set_ylabel(
            "Frequency (Hz)"
        )

    axes[-1].set_xlabel(
        "Time (seconds)"
    )

    plt.tight_layout()

    plt.show()

    print("\nConfiguration summary:")

    for n_fft, hop in configurations:

        print(
            f"FFT={n_fft}, "
            f"Hop={hop}, "
            f"Frame={1000*n_fft/fs:.2f} ms, "
            f"Hop={1000*hop/fs:.2f} ms, "
            f"Frequency resolution={fs/n_fft:.2f} Hz"
        )


# =====================================================================
# EXPERIMENT 7
# ANALYZE REAL AUDIO
# =====================================================================

def experiment_real_audio(filename):

    print("\nEXPERIMENT 7: REAL AUDIO ANALYSIS")

    audio, fs = load_audio(filename)

    print(
        f"File: {filename}"
    )

    print(
        f"Sampling rate: {fs} Hz"
    )

    print(
        f"Duration: {len(audio)/fs:.2f} seconds"
    )

    print(
        f"Samples: {len(audio)}"
    )

    print(
        f"Peak amplitude: {np.max(np.abs(audio)):.6f}"
    )

    print(
        f"RMS amplitude: "
        f"{np.sqrt(np.mean(audio ** 2)):.6f}"
    )

    frequencies, times, spectrum = calculate_stft(
        audio,
        fs
    )

    magnitude, power = calculate_spectrogram(
        spectrum
    )

    fmax = min(
        12000,
        fs / 2
    )

    filterbank, _ = create_mel_filterbank(
        fs,
        DEFAULT_N_FFT,
        DEFAULT_N_MELS,
        0,
        fmax
    )

    mel = calculate_mel_spectrogram(
        power,
        filterbank
    )

    log_mel = calculate_log_mel(
        mel
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 13)
    )

    audio_time = np.arange(
        len(audio)
    ) / fs

    axes[0].plot(
        audio_time,
        audio
    )

    axes[0].set_title(
        "Real Audio Waveform"
    )

    axes[0].set_xlabel(
        "Time (seconds)"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[0].grid(True)

    axes[1].imshow(
        amplitude_to_db(magnitude),
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            frequencies[0],
            frequencies[-1]
        ],
        vmin=-100,
        vmax=0,
        cmap="magma"
    )

    axes[1].set_ylim(
        0,
        fmax
    )

    axes[1].set_title(
        "Real Audio Magnitude Spectrogram"
    )

    axes[1].set_ylabel(
        "Frequency (Hz)"
    )

    axes[2].imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            0,
            DEFAULT_N_MELS
        ],
        cmap="magma"
    )

    axes[2].set_title(
        "Real Audio Log-Mel Spectrogram"
    )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    axes[2].set_ylabel(
        "Mel Filter Index"
    )

    plt.tight_layout()

    plt.show()


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 Lesson 5: STFT and Mel Spectrogram."
        )
    )

    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        choices=range(8),
        help=(
            "0 = All synthetic experiments, "
            "1 = Framing, "
            "2 = STFT, "
            "3 = Magnitude spectrogram, "
            "4 = Mel filterbank, "
            "5 = Log-Mel, "
            "6 = Resolution, "
            "7 = Real audio"
        )
    )

    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Optional path to your own WAV recording."
    )

    args = parser.parse_args()

    experiments = {
        1: experiment_framing,
        2: experiment_stft,
        3: experiment_magnitude_spectrogram,
        4: experiment_mel_filterbank,
        5: experiment_log_mel,
        6: experiment_resolution
    }

    print("\n")
    print("=" * 75)
    print("PHASE 1 — LESSON 5")
    print("STFT, SPECTROGRAMS AND MEL REPRESENTATIONS")
    print("=" * 75)

    if args.experiment == 0:

        for experiment in experiments.values():

            experiment()

        if args.audio is not None:

            experiment_real_audio(
                args.audio
            )

    elif args.experiment == 7:

        if args.audio is None:

            raise ValueError(
                "Experiment 7 requires --audio your_recording.wav"
            )

        experiment_real_audio(
            args.audio
        )

    else:

        experiments[args.experiment]()

    print("\n")
    print("=" * 75)
    print("LESSON 5 COMPLETED")
    print("=" * 75)


if __name__ == "__main__":

    main()