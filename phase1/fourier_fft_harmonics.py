"""
=======================================================================
BUILD YOUR OWN VOICE-CLONING TTS SYSTEM
PHASE 1 — DIGITAL AUDIO FUNDAMENTALS

LESSON 4:
AUDIO WAVEFORMS, FREQUENCY, PHASE, HARMONICS AND FOURIER TRANSFORM
=======================================================================

LEARNING OBJECTIVES
-------------------

1. Understand audio waveforms in the time domain.
2. Understand frequency and phase.
3. Understand amplitude and frequency relationships.
4. Understand sinusoidal signals.
5. Understand fundamental frequency.
6. Understand harmonics.
7. Understand how complex sounds are composed of sinusoids.
8. Understand the Fourier Transform.
9. Understand the Discrete Fourier Transform (DFT).
10. Understand the Fast Fourier Transform (FFT).
11. Understand frequency resolution.
12. Understand magnitude and phase spectra.
13. Understand the relationship between time-domain and
    frequency-domain representations.
14. Understand why FFT is fundamental to speech processing.

=======================================================================
1. TIME-DOMAIN REPRESENTATION
=======================================================================

A digital audio recording is a sequence of samples:

    x[0], x[1], x[2], ..., x[N-1]

This is called a time-domain representation.

Each sample represents the amplitude of the signal at a specific time.

The corresponding sample time is:

    t[n] = n / fs

where:

    n  = sample index
    fs = sampling rate

The time-domain waveform tells us how amplitude changes over time.

However, it does not directly tell us which frequencies are present.

=======================================================================
2. SINUSOIDAL SIGNALS
=======================================================================

A pure sinusoid is:

    x(t) = A * sin(2*pi*f*t + phi)

Parameters:

    A   = amplitude
    f   = frequency in Hz
    phi = phase in radians
    t   = time

Angular frequency:

    omega = 2*pi*f

Therefore:

    x(t) = A * sin(omega*t + phi)

Amplitude controls the height of the waveform.

Frequency controls the number of cycles per second.

Phase controls the horizontal position of the waveform.

Examples:

    100 Hz:
        100 cycles per second

    440 Hz:
        440 cycles per second

    1000 Hz:
        1000 cycles per second

=======================================================================
3. PHASE
=======================================================================

Two signals can have identical amplitude and frequency but different
phases.

Example:

    x1(t) = sin(2*pi*f*t)

    x2(t) = sin(2*pi*f*t + pi/2)

The second signal is shifted by 90 degrees.

Phase is particularly important when signals are combined.

Two signals may reinforce or partially cancel one another.

This is called constructive and destructive interference.

=======================================================================
4. HARMONICS
=======================================================================

Many musical and voiced speech sounds are not pure sinusoids.

They contain a fundamental frequency and harmonic components.

If the fundamental frequency is:

    f0

Its harmonics are approximately:

    f0, 2*f0, 3*f0, 4*f0, ...

Example:

    Fundamental frequency = 120 Hz

    First harmonic  = 120 Hz
    Second harmonic = 240 Hz
    Third harmonic  = 360 Hz
    Fourth harmonic = 480 Hz

A periodic signal can be represented as a sum of sinusoidal
components.

Fourier series representation:

    x(t) = a0
         + sum(
             an*cos(2*pi*n*f0*t)
             + bn*sin(2*pi*n*f0*t)
           )

The harmonic amplitudes determine the shape and timbre of the signal.

IMPORTANT:

Fundamental frequency and harmonics are related to voiced speech,
but real speech is more complicated.

Unvoiced consonants, transient sounds, changing vocal-tract
resonances, and noise-like components cannot be fully described by
a simple stationary harmonic model.

=======================================================================
5. FOURIER TRANSFORM
=======================================================================

The Fourier Transform converts a signal from the time domain into
the frequency domain.

Conceptually:

    Time Domain -> Frequency Domain

The continuous Fourier Transform is:

    X(f) = integral[
        x(t) * exp(-j*2*pi*f*t)
    ] dt

where:

    x(t) = time-domain signal
    X(f) = complex frequency-domain representation
    j    = imaginary unit

The Fourier Transform describes the amount of each frequency
component and its phase.

The result is complex-valued.

Its magnitude:

    |X(f)|

describes frequency-component magnitude.

Its phase:

    angle(X(f))

describes frequency-component phase.

=======================================================================
6. DISCRETE FOURIER TRANSFORM (DFT)
=======================================================================

Computers work with discrete samples.

For N samples, the DFT is:

    X[k] = sum(
        x[n] * exp(-j*2*pi*k*n/N)
    )

where:

    n = time-domain sample index
    k = frequency-bin index
    N = number of samples

The corresponding frequency of bin k is:

    f[k] = k*fs/N

Frequency resolution:

    delta_f = fs/N

Since:

    N = fs*T

we can also write:

    delta_f = 1/T

where T is the duration of the analyzed signal.

Longer analysis windows provide finer frequency resolution.

=======================================================================
7. FAST FOURIER TRANSFORM (FFT)
=======================================================================

The FFT is an efficient algorithm for calculating the DFT.

A direct DFT requires approximately:

    O(N^2)

operations.

A typical FFT algorithm requires approximately:

    O(N log N)

operations.

This makes FFT practical for speech processing.

In Python:

    spectrum = np.fft.fft(signal)

The corresponding frequency bins:

    frequencies = np.fft.fftfreq(N, 1/fs)

For real-valued audio signals, we often use:

    np.fft.rfft(signal)

This calculates the non-negative frequency spectrum.

=======================================================================
8. MAGNITUDE SPECTRUM
=======================================================================

The FFT result is complex.

To obtain magnitude:

    magnitude = abs(X)

To obtain magnitude in decibels:

    magnitude_db = 20*log10(magnitude)

A small epsilon is normally added to prevent log(0).

The magnitude spectrum shows the strength of frequency components.

=======================================================================
9. WINDOWING AND SPECTRAL LEAKAGE
=======================================================================

The FFT analyzes a finite segment of a signal.

When a finite segment is extracted, its boundaries can introduce
discontinuities.

This can spread energy across neighboring frequency bins.

This phenomenon is called spectral leakage.

A window function reduces boundary discontinuities.

Common windows include:

    Rectangular
    Hann
    Hamming
    Blackman

The Hann window is:

    w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))

Windowing reduces leakage but changes spectral amplitude and
frequency resolution characteristics.

=======================================================================
10. WHY THIS MATTERS FOR VOICE CLONING
=======================================================================

Speech models frequently use frequency-domain representations.

Examples:

    - STFT
    - Magnitude spectrogram
    - Mel spectrogram
    - Log-mel spectrogram
    - Harmonic representations
    - Spectral losses

A typical TTS pipeline may look like:

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
    Waveform

A speaker encoder may also analyze spectral patterns to extract
information related to speaker identity.

The Fourier Transform is one of the mathematical foundations
underlying these techniques.

=======================================================================
EXPERIMENTS
=======================================================================

1. Visualize a sine wave in the time domain.
2. Compare signals with different frequencies.
3. Compare signals with different phases.
4. Generate a harmonic signal.
5. Observe its FFT spectrum.
6. Compare FFT magnitude and phase.
7. Observe spectral leakage.
8. Compare rectangular and Hann windows.
9. Explore frequency resolution.

=======================================================================
DEPENDENCIES
=======================================================================

    pip install numpy matplotlib scipy

=======================================================================
"""


# =====================================================================
# IMPORTS
# =====================================================================

import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import get_window


# =====================================================================
# CONFIGURATION
# =====================================================================

FS = 16000

DURATION = 0.05

EPSILON = 1e-12


# =====================================================================
# SIGNAL GENERATION
# =====================================================================

def generate_sine(
    frequency,
    fs,
    duration,
    amplitude=1.0,
    phase=0.0
):
    """
    Generate a discrete sinusoidal signal.

    Formula:

        x(t) = A*sin(2*pi*f*t + phi)
    """

    number_of_samples = int(
        fs * duration
    )

    time = (
        np.arange(number_of_samples) / fs
    )

    signal = amplitude * np.sin(
        2 * np.pi * frequency * time + phase
    )

    return time, signal


def generate_harmonic_signal(
    fundamental,
    fs,
    duration,
    number_of_harmonics=8
):
    """
    Generate a periodic signal composed of harmonics.

    Example:

        Fundamental = 120 Hz

        Components:

            120 Hz
            240 Hz
            360 Hz
            480 Hz
            ...

    Each harmonic has decreasing amplitude.

    This is a simplified educational model of a voiced periodic
    excitation signal, not a complete model of human speech.
    """

    number_of_samples = int(
        fs * duration
    )

    time = (
        np.arange(number_of_samples) / fs
    )

    signal = np.zeros(
        number_of_samples
    )

    for harmonic in range(
        1,
        number_of_harmonics + 1
    ):

        frequency = (
            fundamental * harmonic
        )

        amplitude = (
            1.0 / harmonic
        )

        signal += amplitude * np.sin(
            2 * np.pi * frequency * time
        )

    # Normalize peak amplitude.

    signal /= np.max(
        np.abs(signal)
    )

    return time, signal


# =====================================================================
# FFT ANALYSIS
# =====================================================================

def calculate_fft(signal, fs, window=None):
    """
    Calculate the one-sided FFT of a real-valued signal.

    Parameters
    ----------
    signal:
        Input signal.

    fs:
        Sampling rate.

    window:
        Optional window array.

    Returns
    -------
    frequencies:
        Non-negative frequency bins.

    magnitude:
        One-sided amplitude spectrum.

    phase:
        Phase spectrum in radians.

    spectrum:
        Complex FFT result.

    Notes
    -----
    The amplitude normalization below is designed for a
    one-sided amplitude spectrum.

    For windowed signals, the window's coherent gain is accounted
    for in the amplitude normalization.

    This is useful for interpreting sinusoidal peak amplitudes.
    It is not a general power spectral density estimator.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    number_of_samples = len(
        signal
    )

    if window is None:

        window = np.ones(
            number_of_samples
        )

    windowed_signal = (
        signal * window
    )

    spectrum = np.fft.rfft(
        windowed_signal
    )

    frequencies = np.fft.rfftfreq(
        number_of_samples,
        d=1.0 / fs
    )

    coherent_gain = np.sum(window)

    if coherent_gain == 0:

        raise ValueError(
            "Window coherent gain cannot be zero."
        )

    magnitude = (
        np.abs(spectrum) / coherent_gain
    )

    # Convert to one-sided amplitude spectrum.
    #
    # DC and Nyquist bins are not doubled.
    # Other positive-frequency bins are doubled.

    if number_of_samples % 2 == 0:

        magnitude[1:-1] *= 2

    else:

        magnitude[1:] *= 2

    phase = np.angle(
        spectrum
    )

    return (
        frequencies,
        magnitude,
        phase,
        spectrum
    )


# =====================================================================
# EXPERIMENT 1
# TIME-DOMAIN WAVEFORMS
# =====================================================================

def experiment_waveforms():

    print("\nEXPERIMENT 1: TIME-DOMAIN WAVEFORMS")

    frequencies_to_test = [
        100,
        440,
        1000,
        3000
    ]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 11)
    )

    fig.suptitle(
        "Sinusoidal Signals at Different Frequencies",
        fontsize=16
    )

    for index, frequency in enumerate(
        frequencies_to_test
    ):

        time, signal = generate_sine(
            frequency,
            FS,
            0.01
        )

        axes[index].plot(
            time,
            signal
        )

        axes[index].set_title(
            f"Frequency = {frequency} Hz"
        )

        axes[index].set_xlabel(
            "Time (seconds)"
        )

        axes[index].set_ylabel(
            "Amplitude"
        )

        axes[index].grid(True)

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 2
# PHASE DIFFERENCES
# =====================================================================

def experiment_phase():

    print("\nEXPERIMENT 2: PHASE DIFFERENCES")

    frequency = 440

    duration = 0.01

    phases = [
        0,
        np.pi / 4,
        np.pi / 2,
        np.pi
    ]

    plt.figure(figsize=(14, 6))

    for phase in phases:

        time, signal = generate_sine(
            frequency,
            FS,
            duration,
            phase=phase
        )

        plt.plot(
            time,
            signal,
            label=f"Phase = {phase:.2f} radians"
        )

    plt.title(
        "Same Frequency and Amplitude, Different Phase"
    )

    plt.xlabel("Time (seconds)")

    plt.ylabel("Amplitude")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 3
# HARMONICS
# =====================================================================

def experiment_harmonics():

    print("\nEXPERIMENT 3: HARMONICS")

    fundamental = 120

    time, signal = generate_harmonic_signal(
        fundamental,
        FS,
        0.05,
        number_of_harmonics=10
    )

    frequencies, magnitude, phase, _ = calculate_fft(
        signal,
        FS
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9)
    )

    axes[0].plot(
        time,
        signal
    )

    axes[0].set_title(
        "Time-Domain Harmonic Signal"
    )

    axes[0].set_xlabel(
        "Time (seconds)"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[0].grid(True)

    axes[1].stem(
        frequencies,
        magnitude
    )

    axes[1].set_xlim(
        0,
        1500
    )

    axes[1].set_title(
        "Frequency Spectrum: Harmonic Peaks"
    )

    axes[1].set_xlabel(
        "Frequency (Hz)"
    )

    axes[1].set_ylabel(
        "Amplitude"
    )

    axes[1].grid(True)

    plt.tight_layout()

    plt.show()

    print("\nExpected harmonic frequencies:")

    for harmonic in range(1, 11):

        print(
            f"Harmonic {harmonic}: "
            f"{fundamental * harmonic} Hz"
        )


# =====================================================================
# EXPERIMENT 4
# FOURIER TRANSFORM
# =====================================================================

def experiment_fft():

    print("\nEXPERIMENT 4: FOURIER TRANSFORM")

    fs = 16000

    duration = 1.0

    components = [
        (440, 1.0),
        (1000, 0.6),
        (2500, 0.3)
    ]

    number_of_samples = int(
        fs * duration
    )

    time = (
        np.arange(number_of_samples) / fs
    )

    signal = np.zeros(
        number_of_samples
    )

    for frequency, amplitude in components:

        signal += amplitude * np.sin(
            2 * np.pi * frequency * time
        )

    frequencies, magnitude, phase, _ = calculate_fft(
        signal,
        fs
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 12)
    )

    axes[0].plot(
        time[:500],
        signal[:500]
    )

    axes[0].set_title(
        "Time-Domain Signal"
    )

    axes[0].set_xlabel(
        "Time (seconds)"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[0].grid(True)

    axes[1].plot(
        frequencies,
        magnitude
    )

    axes[1].set_xlim(
        0,
        4000
    )

    axes[1].set_title(
        "Magnitude Spectrum"
    )

    axes[1].set_xlabel(
        "Frequency (Hz)"
    )

    axes[1].set_ylabel(
        "Magnitude"
    )

    axes[1].grid(True)

    axes[2].plot(
        frequencies,
        phase
    )

    axes[2].set_xlim(
        0,
        4000
    )

    axes[2].set_title(
        "Phase Spectrum"
    )

    axes[2].set_xlabel(
        "Frequency (Hz)"
    )

    axes[2].set_ylabel(
        "Phase (radians)"
    )

    axes[2].grid(True)

    plt.tight_layout()

    plt.show()

    print("\nExpected frequencies:")

    for frequency, amplitude in components:

        print(
            f"{frequency} Hz, amplitude = {amplitude}"
        )


# =====================================================================
# EXPERIMENT 5
# SPECTRAL LEAKAGE
# =====================================================================

def experiment_spectral_leakage():

    print("\nEXPERIMENT 5: SPECTRAL LEAKAGE")

    fs = 16000

    duration = 0.025

    frequency = 440

    time, signal = generate_sine(
        frequency,
        fs,
        duration
    )

    rectangular_window = np.ones(
        len(signal)
    )

    hann_window = get_window(
        "hann",
        len(signal),
        fftbins=False
    )

    frequencies, rectangular_magnitude, _, _ = (
        calculate_fft(
            signal,
            fs,
            rectangular_window
        )
    )

    _, hann_magnitude, _, _ = calculate_fft(
        signal,
        fs,
        hann_window
    )

    plt.figure(figsize=(14, 7))

    plt.plot(
        frequencies,
        20 * np.log10(
            rectangular_magnitude + EPSILON
        ),
        label="Rectangular window"
    )

    plt.plot(
        frequencies,
        20 * np.log10(
            hann_magnitude + EPSILON
        ),
        label="Hann window"
    )

    plt.xlim(
        0,
        2000
    )

    plt.ylim(
        -100,
        20
    )

    plt.title(
        "Spectral Leakage: Rectangular vs Hann Window"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Magnitude (dB relative to amplitude)"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 6
# FREQUENCY RESOLUTION
# =====================================================================

def experiment_frequency_resolution():

    print("\nEXPERIMENT 6: FREQUENCY RESOLUTION")

    fs = 16000

    durations = [
        0.01,
        0.025,
        0.05,
        0.1,
        0.5,
        1.0
    ]

    print(
        f"{'Duration (seconds)':<25}"
        f"{'Samples':<20}"
        f"{'Frequency Resolution'}"
    )

    print("-" * 70)

    for duration in durations:

        number_of_samples = int(
            fs * duration
        )

        frequency_resolution = (
            fs / number_of_samples
        )

        print(
            f"{duration:<25.3f}"
            f"{number_of_samples:<20}"
            f"{frequency_resolution:.2f} Hz"
        )

    resolutions = [
        1 / duration
        for duration in durations
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        durations,
        resolutions,
        "o-"
    )

    plt.xlabel(
        "Analysis Duration (seconds)"
    )

    plt.ylabel(
        "Frequency Resolution (Hz)"
    )

    plt.title(
        "Longer Analysis Windows Improve Frequency Resolution"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 7
# FREQUENCY IDENTIFICATION
# =====================================================================

def experiment_frequency_identification():

    print("\nEXPERIMENT 7: FREQUENCY IDENTIFICATION")

    fs = 16000

    duration = 1.0

    original_frequencies = [
        220,
        440,
        880,
        1760
    ]

    number_of_samples = int(
        fs * duration
    )

    time = (
        np.arange(number_of_samples) / fs
    )

    signal = np.zeros(
        number_of_samples
    )

    for frequency in original_frequencies:

        signal += np.sin(
            2 * np.pi * frequency * time
        )

    frequencies, magnitude, _, _ = calculate_fft(
        signal,
        fs
    )

    from scipy.signal import find_peaks

    peaks, _ = find_peaks(
        magnitude,
        height=0.1
    )

    peak_frequencies = frequencies[
        peaks
    ]

    print("Detected frequency peaks:")

    for frequency in peak_frequencies:

        print(
            f"{frequency:.2f} Hz"
        )

    plt.figure(figsize=(14, 6))

    plt.plot(
        frequencies,
        magnitude
    )

    plt.xlim(
        0,
        3000
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Magnitude"
    )

    plt.title(
        "Automatic Frequency Peak Detection"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 Lesson 4: Fourier Transform and Harmonics."
        )
    )

    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        choices=range(8),
        help=(
            "0 = All experiments, "
            "1 = Waveforms, "
            "2 = Phase, "
            "3 = Harmonics, "
            "4 = FFT, "
            "5 = Spectral leakage, "
            "6 = Frequency resolution, "
            "7 = Frequency identification"
        )
    )

    args = parser.parse_args()

    experiments = {
        1: experiment_waveforms,
        2: experiment_phase,
        3: experiment_harmonics,
        4: experiment_fft,
        5: experiment_spectral_leakage,
        6: experiment_frequency_resolution,
        7: experiment_frequency_identification
    }

    print("\n")
    print("=" * 70)
    print("PHASE 1 — LESSON 4")
    print("FOURIER TRANSFORM AND HARMONICS")
    print("=" * 70)

    if args.experiment == 0:

        for experiment in experiments.values():

            experiment()

    else:

        experiments[args.experiment]()

    print("\n")
    print("=" * 70)
    print("LESSON 4 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()