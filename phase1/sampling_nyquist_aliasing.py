"""
=======================================================================
COURSE: BUILD YOUR OWN VOICE-CLONING TTS SYSTEM
PHASE 1 — DIGITAL AUDIO FUNDAMENTALS

LESSON 2:
SAMPLING RATE, NYQUIST-SHANNON THEOREM AND ALIASING
=======================================================================

LEARNING OBJECTIVES
-------------------

By completing this lesson, you will understand:

1. What sampling rate means.
2. How continuous-time signals become digital signals.
3. What sampling interval means.
4. What the Nyquist frequency represents.
5. Why the Nyquist-Shannon sampling theorem matters.
6. What aliasing is and why it occurs.
7. Why insufficient sampling rates distort audio.
8. Why changing sample-rate metadata is NOT resampling.
9. How proper digital resampling works.
10. Why sampling-rate management is important for TTS and voice cloning.

=======================================================================
1. CONTINUOUS-TIME SIGNAL
=======================================================================

Real-world sound is a continuous physical phenomenon.

A microphone converts acoustic pressure variations into an electrical
signal.

Mathematically, a continuous signal can be represented as:

    x(t)

where:

    x = signal amplitude
    t = continuous time

For a pure sinusoidal signal:

    x(t) = A * sin(2*pi*f*t + phi)

where:

    A   = amplitude
    f   = frequency in Hz
    t   = time in seconds
    phi = phase in radians

Example:

    A = 1
    f = 440 Hz

This represents a 440 Hz sine wave.

A computer cannot directly store an infinitely continuous signal.
Instead, it measures the signal at discrete time intervals.

=======================================================================
2. DIGITAL SAMPLING
=======================================================================

Sampling converts a continuous signal into discrete measurements.

The sampling frequency is:

    fs = number of samples per second

Examples:

    fs = 8,000 Hz
        8,000 measurements per second

    fs = 16,000 Hz
        16,000 measurements per second

    fs = 22,050 Hz
        22,050 measurements per second

    fs = 44,100 Hz
        44,100 measurements per second

    fs = 48,000 Hz
        48,000 measurements per second

The sampling interval is:

    Ts = 1 / fs

For example:

    fs = 16,000 Hz

    Ts = 1 / 16,000

    Ts = 0.0000625 seconds

Therefore, one sample is recorded every 62.5 microseconds.

The discrete-time signal is:

    x[n] = x(n / fs)

where:

    n = discrete sample index
    fs = sampling frequency

=======================================================================
3. NYQUIST-SHANNON SAMPLING THEOREM
=======================================================================

The Nyquist-Shannon sampling theorem states that a band-limited
continuous signal can theoretically be reconstructed perfectly from
its samples when the sampling frequency is greater than twice the
highest frequency present in the signal.

Mathematically:

    fs > 2 * fmax

where:

    fs   = sampling frequency
    fmax = highest signal frequency

The Nyquist frequency is:

    f_Nyquist = fs / 2

Example:

    Sampling rate = 16,000 Hz

    Nyquist frequency = 8,000 Hz

In an ideal band-limited system, frequencies below 8 kHz can be
represented without aliasing.

IMPORTANT:

Real audio systems require anti-aliasing filters because real signals
are not perfectly band-limited.

The usable bandwidth is also affected by the filter's transition band.

=======================================================================
4. WHAT IS ALIASING?
=======================================================================

Aliasing happens when a signal contains frequencies that cannot be
correctly represented at the selected sampling rate.

For example:

    Original frequency = 7,000 Hz
    Sampling rate = 8,000 Hz

    Nyquist frequency = 4,000 Hz

Since:

    7,000 > 4,000

The signal violates the Nyquist sampling condition.

The sampled signal can appear to have a different frequency.

This false lower frequency is called an alias frequency.

For a sinusoid, its folded alias frequency can be calculated using:

    f_alias = abs(((f + fs/2) % fs) - fs/2)

This formula returns a frequency between:

    0 and fs/2

Aliasing is especially dangerous because the distorted samples may
look like a legitimate lower-frequency signal.

Once aliasing has occurred, the original frequency generally cannot
be recovered from those samples alone.

=======================================================================
5. WHY THIS MATTERS FOR VOICE CLONING
=======================================================================

Speech contains multiple frequency components.

These components carry information about:

    - Fundamental frequency (pitch)
    - Harmonics
    - Formants
    - Consonants
    - Vowels
    - Speaker identity
    - Vocal characteristics

Incorrect sampling can destroy information before the neural model
even receives the audio.

For example:

    A recording captured at 48 kHz
    Incorrectly downsampled to 16 kHz
    Without proper low-pass filtering

may contain aliasing.

This can damage the acoustic information used by:

    - Speaker encoders
    - Speech encoders
    - Mel spectrogram generators
    - Neural vocoders
    - Voice-cloning models

A good voice-cloning system must use consistent audio preprocessing.

=======================================================================
6. CHANGING METADATA IS NOT RESAMPLING
=======================================================================

Suppose an audio file contains:

    48,000 samples per second

If you simply change its sample-rate metadata to:

    16,000 samples per second

without changing the actual samples, the audio will play three times
slower and its pitch will change.

This is NOT proper resampling.

Proper resampling requires creating a new sequence of samples at the
target sampling rate.

The typical downsampling process includes:

    1. Low-pass filtering.
    2. Removing frequencies above the new Nyquist frequency.
    3. Generating samples at the new sampling interval.

This script demonstrates the difference.

=======================================================================
7. EXPERIMENTS
=======================================================================
# Experiment 1: Sampling rates
python phase1_lesson2_sampling_nyquist_aliasing.py --experiment 1

# Experiment 2: Aliasing
python phase1_lesson2_sampling_nyquist_aliasing.py --experiment 2

# Experiment 3: Nyquist condition
python phase1_lesson2_sampling_nyquist_aliasing.py --experiment 3

# Experiment 4: Proper resampling
python phase1_lesson2_sampling_nyquist_aliasing.py --experiment 4

# Experiment 5: Frequency folding
python phase1_lesson2_sampling_nyquist_aliasing.py --experiment 5

EXPERIMENT A:
    Sample a 1 kHz signal at several sampling rates.

EXPERIMENT B:
    Sample a 7 kHz signal at 8 kHz.
    Observe aliasing.

EXPERIMENT C:
    Compare a 48 kHz signal with a properly resampled 16 kHz signal.

EXPERIMENT D:
    Compare proper resampling with incorrect metadata modification.

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

from scipy.signal import resample_poly


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_DURATION = 0.005

HIGH_RESOLUTION_FS = 192000

FIGURE_SIZE = (14, 8)


# =====================================================================
# MATHEMATICAL UTILITIES
# =====================================================================

def generate_sine_wave(frequency, sampling_rate, duration, amplitude=1.0):
    """
    Generate a discrete sine wave.

    Parameters
    ----------
    frequency:
        Sine-wave frequency in Hz.

    sampling_rate:
        Number of samples per second.

    duration:
        Signal duration in seconds.

    amplitude:
        Signal amplitude.

    Returns
    -------
    time:
        Discrete time values.

    signal:
        Sampled sine-wave values.
    """

    number_of_samples = int(sampling_rate * duration)

    time = np.arange(number_of_samples) / sampling_rate

    signal = amplitude * np.sin(
        2.0 * np.pi * frequency * time
    )

    return time, signal


def generate_continuous_reference(frequency, duration):
    """
    Generate a high-resolution approximation of a continuous sine wave.

    This is only a numerical approximation of a continuous signal.

    A very high sampling rate is used to make the reference curve
    visually smooth.
    """

    time = np.linspace(
        0,
        duration,
        int(HIGH_RESOLUTION_FS * duration),
        endpoint=False
    )

    signal = np.sin(
        2.0 * np.pi * frequency * time
    )

    return time, signal


def calculate_alias_frequency(frequency, sampling_rate):
    """
    Calculate the folded alias frequency.

    The resulting frequency is between:

        0 Hz and sampling_rate / 2

    This formula assumes a sinusoidal input.
    """

    nyquist_frequency = sampling_rate / 2.0

    alias_frequency = abs(
        ((frequency + nyquist_frequency) % sampling_rate)
        - nyquist_frequency
    )

    return alias_frequency


def calculate_rms(signal):
    """
    Calculate root mean square amplitude.

        RMS = sqrt(mean(x^2))
    """

    return np.sqrt(np.mean(np.square(signal)))


# =====================================================================
# EXPERIMENT 1
# VISUALIZE SAMPLING AT DIFFERENT RATES
# =====================================================================

def experiment_sampling_rates():

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: SAMPLING RATE")
    print("=" * 70)

    frequency = 1000
    duration = 0.005

    sampling_rates = [
        2000,
        4000,
        8000,
        16000
    ]

    reference_time, reference_signal = (
        generate_continuous_reference(
            frequency,
            duration
        )
    )

    fig, axes = plt.subplots(
        len(sampling_rates),
        1,
        figsize=(FIGURE_SIZE[0], 12)
    )

    fig.suptitle(
        "Experiment 1: Sampling a 1 kHz Sine Wave",
        fontsize=16
    )

    for index, fs in enumerate(sampling_rates):

        time, signal = generate_sine_wave(
            frequency,
            fs,
            duration
        )

        nyquist = fs / 2

        axes[index].plot(
            reference_time,
            reference_signal,
            label="High-resolution reference"
        )

        axes[index].stem(
            time,
            signal,
            linefmt="C1-",
            markerfmt="C1o",
            basefmt=" "
        )

        axes[index].set_title(
            f"Sampling Rate = {fs} Hz | "
            f"Nyquist Frequency = {nyquist} Hz"
        )

        axes[index].set_xlabel("Time (seconds)")
        axes[index].set_ylabel("Amplitude")

        axes[index].grid(True)
        axes[index].legend()

    plt.tight_layout()
    plt.show()


# =====================================================================
# EXPERIMENT 2
# DEMONSTRATE ALIASING
# =====================================================================

def experiment_aliasing():

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ALIASING")
    print("=" * 70)

    original_frequency = 7000
    sampling_rate = 8000

    duration = 0.002

    nyquist = sampling_rate / 2

    alias_frequency = calculate_alias_frequency(
        original_frequency,
        sampling_rate
    )

    print(f"Original frequency : {original_frequency} Hz")
    print(f"Sampling rate      : {sampling_rate} Hz")
    print(f"Nyquist frequency  : {nyquist} Hz")
    print(f"Alias frequency    : {alias_frequency} Hz")

    reference_time, reference_signal = (
        generate_continuous_reference(
            original_frequency,
            duration
        )
    )

    sampled_time, sampled_signal = generate_sine_wave(
        original_frequency,
        sampling_rate,
        duration
    )

    alias_time, alias_signal = generate_sine_wave(
        alias_frequency,
        sampling_rate,
        duration
    )

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(
        reference_time,
        reference_signal,
        label=f"Original {original_frequency} Hz signal"
    )

    plt.plot(
        alias_time,
        alias_signal,
        "--",
        label=f"Aliased {alias_frequency} Hz signal"
    )

    plt.stem(
        sampled_time,
        sampled_signal,
        linefmt="C2-",
        markerfmt="C2o",
        basefmt=" ",
        label="Actual samples"
    )

    plt.title(
        "Aliasing: 7 kHz Signal Sampled at 8 kHz"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nInterpretation:")
    print(
        "The original 7 kHz signal is above the 4 kHz Nyquist "
        "frequency."
    )

    print(
        "The samples therefore cannot uniquely represent the "
        "original 7 kHz signal."
    )

    print(
        "They are consistent with a lower-frequency alias."
    )


# =====================================================================
# EXPERIMENT 3
# NYQUIST CONDITION
# =====================================================================

def experiment_nyquist():

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: NYQUIST CONDITION")
    print("=" * 70)

    sampling_rate = 16000

    nyquist = sampling_rate / 2

    frequencies = [
        1000,
        4000,
        7000,
        9000,
        12000
    ]

    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Nyquist frequency: {nyquist} Hz\n")

    print(
        f"{'Original Frequency':<25}"
        f"{'Alias Frequency':<20}"
        f"{'Condition'}"
    )

    print("-" * 70)

    for frequency in frequencies:

        alias = calculate_alias_frequency(
            frequency,
            sampling_rate
        )

        if frequency < nyquist:
            condition = "Below Nyquist"

        elif frequency == nyquist:
            condition = "At Nyquist boundary"

        else:
            condition = "Above Nyquist: aliasing"

        print(
            f"{frequency:<25}"
            f"{alias:<20.1f}"
            f"{condition}"
        )


# =====================================================================
# EXPERIMENT 4
# PROPER RESAMPLING
# =====================================================================

def experiment_resampling():

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: PROPER RESAMPLING")
    print("=" * 70)

    original_fs = 48000
    target_fs = 16000

    duration = 0.1

    frequency = 1000

    # Generate original signal at 48 kHz.

    original_time, original_signal = generate_sine_wave(
        frequency,
        original_fs,
        duration
    )

    # -------------------------------------------------------------
    # PROPER RESAMPLING
    # -------------------------------------------------------------

    # 48 kHz -> 16 kHz
    #
    # The conversion ratio is:
    #
    #     16000 / 48000 = 1 / 3
    #
    # resample_poly performs polyphase filtering and resampling.
    #
    # Filtering is important because frequencies above the target
    # Nyquist frequency must be removed before downsampling.

    properly_resampled = resample_poly(
        original_signal,
        up=1,
        down=3
    )

    proper_time = (
        np.arange(len(properly_resampled)) / target_fs
    )

    # -------------------------------------------------------------
    # INCORRECT METADATA MODIFICATION
    # -------------------------------------------------------------

    # This does NOT resample the signal.
    #
    # We simply pretend that the original 48 kHz samples were
    # recorded at 16 kHz.
    #
    # The number of samples stays unchanged.
    #
    # Playback duration becomes three times longer.

    wrong_metadata_time = (
        np.arange(len(original_signal)) / target_fs
    )

    wrong_metadata_duration = (
        len(original_signal) / target_fs
    )

    correct_duration = (
        len(properly_resampled) / target_fs
    )

    print(f"Original sampling rate: {original_fs} Hz")
    print(f"Target sampling rate:   {target_fs} Hz")

    print(
        f"Original sample count:  {len(original_signal)}"
    )

    print(
        f"Resampled sample count: {len(properly_resampled)}"
    )

    print(
        f"Correct duration:       {correct_duration:.4f} seconds"
    )

    print(
        f"Wrong duration:         {wrong_metadata_duration:.4f} seconds"
    )

    # -------------------------------------------------------------
    # VISUAL COMPARISON
    # -------------------------------------------------------------

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(
        original_time,
        original_signal,
        label="Original 48 kHz signal",
        alpha=0.7
    )

    plt.plot(
        proper_time,
        properly_resampled,
        "--",
        label="Properly resampled 16 kHz signal"
    )

    plt.title(
        "Proper Resampling: 48 kHz to 16 kHz"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # WRONG METADATA PLOT
    # -------------------------------------------------------------

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(
        wrong_metadata_time,
        original_signal
    )

    plt.title(
        "Incorrect: Original Samples Reinterpreted as 16 kHz"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid(True)

    plt.tight_layout()
    plt.show()


# =====================================================================
# EXPERIMENT 5
# OBSERVE ALIASING AT DIFFERENT SAMPLING RATES
# =====================================================================

def experiment_frequency_sweep():

    print("\n" + "=" * 70)
    print("EXPERIMENT 5: ALIAS FREQUENCY")
    print("=" * 70)

    sampling_rate = 16000

    frequencies = np.linspace(
        0,
        24000,
        500
    )

    alias_frequencies = [
        calculate_alias_frequency(
            frequency,
            sampling_rate
        )
        for frequency in frequencies
    ]

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(
        frequencies,
        alias_frequencies
    )

    plt.axvline(
        sampling_rate / 2,
        linestyle="--",
        label="Nyquist frequency"
    )

    plt.xlabel("Original Frequency (Hz)")
    plt.ylabel("Observed Alias Frequency (Hz)")

    plt.title(
        "Frequency Folding at a 16 kHz Sampling Rate"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# =====================================================================
# AUDIO ENGINEERING SUMMARY
# =====================================================================

def print_audio_summary():

    print("\n" + "=" * 70)
    print("AUDIO ENGINEERING SUMMARY")
    print("=" * 70)

    sampling_rates = [
        8000,
        16000,
        22050,
        24000,
        44100,
        48000
    ]

    print(
        f"{'Sampling Rate':<20}"
        f"{'Nyquist Frequency':<25}"
        f"{'Samples in 1 Second'}"
    )

    print("-" * 70)

    for fs in sampling_rates:

        nyquist = fs / 2

        print(
            f"{fs:<20}"
            f"{nyquist:<25.1f}"
            f"{fs}"
        )

    print("\nIMPORTANT TTS CONCEPTS")

    print("""
    1. Always know the actual sample rate of your audio.

    2. Do not confuse sample-rate metadata with actual resampling.

    3. Apply appropriate anti-aliasing filtering before downsampling.

    4. Use a consistent sample rate throughout a training dataset.

    5. Verify audio duration and sample count after resampling.

    6. Preserve the original recordings.

    7. Do not assume that a higher sample rate automatically
       produces better audio.

    8. The model architecture and training data determine the
       sample rate expected by a TTS model.
    """)


# =====================================================================
# MAIN PROGRAM
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 Lesson 2: Sampling, Nyquist theorem "
            "and aliasing."
        )
    )

    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help=(
            "0 = Run all experiments, "
            "1 = Sampling rates, "
            "2 = Aliasing, "
            "3 = Nyquist condition, "
            "4 = Resampling, "
            "5 = Frequency sweep"
        )
    )

    args = parser.parse_args()

    print("\n")
    print("=" * 70)
    print("PHASE 1 — LESSON 2")
    print("SAMPLING RATE, NYQUIST THEOREM AND ALIASING")
    print("=" * 70)

    if args.experiment == 0:

        experiment_sampling_rates()

        experiment_aliasing()

        experiment_nyquist()

        experiment_resampling()

        experiment_frequency_sweep()

    elif args.experiment == 1:

        experiment_sampling_rates()

    elif args.experiment == 2:

        experiment_aliasing()

    elif args.experiment == 3:

        experiment_nyquist()

    elif args.experiment == 4:

        experiment_resampling()

    elif args.experiment == 5:

        experiment_frequency_sweep()

    print_audio_summary()

    print("\nLesson 2 completed successfully.")


if __name__ == "__main__":

    main()