"""
=======================================================================
BUILD YOUR OWN VOICE-CLONING TTS SYSTEM
PHASE 1 — DIGITAL AUDIO FUNDAMENTALS

LESSON 3:
BIT DEPTH, QUANTIZATION, QUANTIZATION NOISE,
DYNAMIC RANGE AND PCM AUDIO
=======================================================================

LEARNING OBJECTIVES
-------------------

By completing this lesson, you will understand:

1. What bit depth means.
2. How binary numbers represent audio amplitude.
3. What quantization is.
4. How quantization converts continuous-valued samples into discrete
   amplitude levels.
5. How quantization noise is generated.
6. The difference between 8-bit, 16-bit and 24-bit PCM audio.
7. What dynamic range means.
8. What signal-to-quantization-noise ratio (SQNR) means.
9. How PCM audio stores digital samples.
10. Why bit depth matters when preparing datasets for voice cloning.

=======================================================================
1. DIGITAL AUDIO HAS TWO DIMENSIONS
=======================================================================

Digital audio requires two independent operations:

    1. Sampling
    2. Quantization

SAMPLING:
    Converts continuous time into discrete time.

QUANTIZATION:
    Converts continuous-valued amplitude into discrete amplitude
    levels.

Consider a continuous audio signal:

    x(t)

After sampling:

    x[n] = x(n / fs)

The sampled amplitudes can theoretically have arbitrary precision.

However, digital storage uses a finite number of bits.

Therefore, the amplitudes must be mapped to a finite set of
representable values.

This mapping is quantization.

=======================================================================
2. WHAT IS BIT DEPTH?
=======================================================================

Bit depth determines how many binary bits are used to represent
each audio sample.

If the bit depth is B, the number of possible binary combinations is:

    L = 2^B

where:

    L = number of quantization levels
    B = bit depth

Examples:

    8-bit:
        2^8 = 256 levels

    16-bit:
        2^16 = 65,536 levels

    24-bit:
        2^24 = 16,777,216 levels

More levels allow smaller amplitude differences to be represented.

For a normalized signed PCM representation, the approximate
amplitude range is:

    -1.0 to +1.0

The number of representable levels depends on the format.

For example, signed 16-bit PCM uses:

    -32768 to +32767

It has 65,536 possible integer values.

=======================================================================
3. WHAT IS QUANTIZATION?
=======================================================================

Suppose a sample has an amplitude of:

    x = 0.347281

If the audio system uses a limited number of amplitude levels,
it cannot necessarily store that exact value.

Instead, it maps the value to the nearest available quantization
level.

The quantized signal is:

    x_q[n] = Q(x[n])

The quantization error is:

    e[n] = x[n] - x_q[n]

This error is called quantization error.

In many ideal uniform quantizers, the error is bounded approximately
by:

    -Delta/2 <= e[n] <= Delta/2

where Delta is the quantization step size.

For a normalized full-scale range of approximately -1 to +1:

    Delta ~= 2 / 2^B

The exact step size depends on the quantizer convention and
endpoint representation.

=======================================================================
4. QUANTIZATION NOISE
=======================================================================

Quantization error behaves like an additional noise component.

The error is:

    e[n] = x[n] - x_q[n]

For a sufficiently busy signal and an ideal uniform quantizer,
quantization error can often be approximated as uniformly distributed.

Its approximate variance is:

    sigma_e^2 = Delta^2 / 12

This is an approximation, not a universal property of every signal.

For very low-level or periodic signals, quantization error may be
correlated with the signal.

This is why quantization noise is not always equivalent to
independent white noise.

=======================================================================
5. SIGNAL-TO-QUANTIZATION-NOISE RATIO
=======================================================================

SQNR measures the ratio between signal power and quantization
error power.

    SQNR = 10 * log10(P_signal / P_error)

For an ideal uniform quantizer receiving a full-scale sinusoid,
the commonly used approximation is:

    SQNR ~= 6.02 * B + 1.76 dB

where:

    B = bit depth

Examples:

    8-bit:
        Approximately 49.9 dB

    16-bit:
        Approximately 98.1 dB

    24-bit:
        Approximately 146.2 dB

These are theoretical values under ideal assumptions.

Real recording systems also have microphone noise, preamplifier
noise, ADC limitations and other imperfections.

=======================================================================
6. DYNAMIC RANGE
=======================================================================

Dynamic range describes the ratio between the largest usable signal
and the smallest distinguishable signal.

A larger bit depth can support a larger theoretical dynamic range.

For ideal PCM:

    Dynamic range ~= 6.02 * B dB

The actual usable dynamic range is limited by the complete recording
chain, not merely the number of bits.

=======================================================================
7. PCM AUDIO
=======================================================================

PCM means Pulse Code Modulation.

PCM stores audio as a sequence of numerical sample values.

Typical uncompressed PCM formats include:

    16-bit PCM
    24-bit PCM
    32-bit floating-point audio

A WAV file is a container format.

It can contain different audio encodings, including PCM.

Important distinction:

    WAV != necessarily 16-bit PCM

A WAV file may contain:

    PCM_16
    PCM_24
    PCM_32
    FLOAT
    DOUBLE

The encoding must be inspected rather than assumed.

=======================================================================
8. WHY THIS MATTERS FOR VOICE CLONING
=======================================================================

A voice-cloning training dataset contains audio recordings.

The recordings may have different:

    - Sample rates
    - Bit depths
    - PCM encodings
    - Loudness levels
    - Recording devices
    - Background noise levels

A dataset pipeline should inspect and standardize these properties.

For example:

    Original recording:
        48 kHz, 24-bit PCM

    Training format:
        24 kHz, 16-bit PCM

The conversion process should be deliberate.

Do not confuse:

    Bit depth conversion
    Sample-rate conversion
    Normalization
    Compression

They are different operations.

Also remember:

Converting a 16-bit recording to 24-bit does not restore information
that was lost during the original 16-bit recording.

=======================================================================
EXPERIMENTS
=======================================================================

EXPERIMENT 1:
    Quantize a sine wave at different bit depths.

EXPERIMENT 2:
    Visualize quantization error.

EXPERIMENT 3:
    Compare quantization noise across bit depths.

EXPERIMENT 4:
    Calculate theoretical SQNR.

EXPERIMENT 5:
    Inspect PCM encoding and bit depth using SoundFile.

=======================================================================
DEPENDENCIES
=======================================================================

    pip install numpy matplotlib soundfile

=======================================================================
"""


# =====================================================================
# IMPORTS
# =====================================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from pathlib import Path


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_FREQUENCY = 440
DEFAULT_DURATION = 0.02
DEFAULT_SAMPLE_RATE = 48000

BIT_DEPTHS = [2, 3, 4, 8, 16]


# =====================================================================
# SIGNAL GENERATION
# =====================================================================

def generate_sine_wave(
    frequency,
    sampling_rate,
    duration,
    amplitude=0.95
):
    """
    Generate a normalized sine wave.

    Formula:

        x(t) = A * sin(2*pi*f*t)

    Returns:
        time
        signal
    """

    number_of_samples = int(
        sampling_rate * duration
    )

    time = (
        np.arange(number_of_samples)
        / sampling_rate
    )

    signal = amplitude * np.sin(
        2 * np.pi * frequency * time
    )

    return time, signal


# =====================================================================
# QUANTIZATION
# =====================================================================

def uniform_quantize(signal, bit_depth):
    """
    Perform uniform signed PCM-like quantization.

    This function simulates signed integer PCM quantization.

    For B bits:

        Number of levels = 2^B

        Integer minimum = -2^(B-1)

        Integer maximum = 2^(B-1) - 1

    The floating-point signal is first clipped to [-1, 1].

    It is then mapped to signed integer values.

    Finally, the integer representation is converted back to
    normalized floating-point amplitude.

    This is a simplified educational quantizer.
    """

    if bit_depth < 2:
        raise ValueError(
            "Bit depth must be at least 2."
        )

    if bit_depth > 32:
        raise ValueError(
            "This educational quantizer supports up to 32 bits."
        )

    signal = np.asarray(signal, dtype=np.float64)

    signal = np.clip(
        signal,
        -1.0,
        1.0
    )

    integer_min = -(2 ** (bit_depth - 1))

    integer_max = (2 ** (bit_depth - 1)) - 1

    # Convert normalized amplitude into integer PCM values.
    #
    # Negative and positive ranges are slightly asymmetric because
    # signed two's-complement PCM has one additional negative value.

    scale = 2 ** (bit_depth - 1)

    quantized_integer = np.round(
        signal * scale
    )

    quantized_integer = np.clip(
        quantized_integer,
        integer_min,
        integer_max
    )

    quantized_signal = (
        quantized_integer / scale
    )

    return quantized_signal, quantized_integer


# =====================================================================
# QUANTIZATION ERROR
# =====================================================================

def calculate_quantization_error(
    original,
    quantized
):
    """
    Quantization error:

        e[n] = x[n] - x_q[n]
    """

    return original - quantized


# =====================================================================
# SQNR
# =====================================================================

def calculate_sqnr(
    original,
    quantized
):
    """
    Calculate empirical signal-to-quantization-noise ratio.

        SQNR = 10 * log10(P_signal / P_error)

    Important:

    This empirical result depends on the actual signal and quantizer.

    It may differ from the theoretical full-scale sine-wave SQNR.
    """

    error = original - quantized

    signal_power = np.mean(
        np.square(original)
    )

    error_power = np.mean(
        np.square(error)
    )

    if error_power == 0:

        return float("inf")

    return 10 * np.log10(
        signal_power / error_power
    )


# =====================================================================
# THEORETICAL SQNR
# =====================================================================

def theoretical_sqnr(bit_depth):
    """
    Ideal full-scale sinusoidal SQNR approximation:

        SQNR ~= 6.02 * B + 1.76 dB
    """

    return (
        6.02 * bit_depth
        + 1.76
    )


# =====================================================================
# EXPERIMENT 1
# QUANTIZATION AT DIFFERENT BIT DEPTHS
# =====================================================================

def experiment_quantization():

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: QUANTIZATION")
    print("=" * 70)

    time, original = generate_sine_wave(
        DEFAULT_FREQUENCY,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_DURATION
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 12)
    )

    fig.suptitle(
        "Quantization at Different Bit Depths",
        fontsize=16
    )

    axes[0].plot(
        time,
        original,
        label="Original signal"
    )

    axes[0].set_title(
        "Original Floating-Point Signal"
    )

    axes[0].set_ylabel("Amplitude")

    axes[0].grid(True)

    axes[0].legend()

    for bit_depth in [2, 4, 8]:

        quantized, _ = uniform_quantize(
            original,
            bit_depth
        )

        axes[1].step(
            time,
            quantized,
            where="mid",
            label=f"{bit_depth}-bit"
        )

    axes[1].plot(
        time,
        original,
        "--",
        label="Original"
    )

    axes[1].set_title(
        "Quantized Signals"
    )

    axes[1].set_ylabel("Amplitude")

    axes[1].grid(True)

    axes[1].legend()

    # Zoom into a short time region to make discrete amplitude
    # steps easier to observe.

    for bit_depth in [2, 3, 4]:

        quantized, _ = uniform_quantize(
            original,
            bit_depth
        )

        axes[2].step(
            time,
            quantized,
            where="mid",
            label=f"{bit_depth}-bit"
        )

    axes[2].set_xlim(
        0,
        0.004
    )

    axes[2].set_title(
        "Zoomed Quantization Steps"
    )

    axes[2].set_xlabel("Time (seconds)")

    axes[2].set_ylabel("Amplitude")

    axes[2].grid(True)

    axes[2].legend()

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 2
# VISUALIZE QUANTIZATION ERROR
# =====================================================================

def experiment_quantization_error():

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: QUANTIZATION ERROR")
    print("=" * 70)

    time, original = generate_sine_wave(
        DEFAULT_FREQUENCY,
        DEFAULT_SAMPLE_RATE,
        DEFAULT_DURATION
    )

    bit_depth = 4

    quantized, _ = uniform_quantize(
        original,
        bit_depth
    )

    error = calculate_quantization_error(
        original,
        quantized
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8)
    )

    axes[0].plot(
        time,
        original,
        label="Original"
    )

    axes[0].step(
        time,
        quantized,
        where="mid",
        label="Quantized"
    )

    axes[0].set_title(
        f"Original vs Quantized Signal ({bit_depth}-bit)"
    )

    axes[0].set_ylabel("Amplitude")

    axes[0].grid(True)

    axes[0].legend()

    axes[1].plot(
        time,
        error
    )

    axes[1].set_title(
        "Quantization Error"
    )

    axes[1].set_xlabel("Time (seconds)")

    axes[1].set_ylabel("Error")

    axes[1].grid(True)

    plt.tight_layout()

    plt.show()

    print(
        f"Maximum absolute quantization error: "
        f"{np.max(np.abs(error)):.8f}"
    )

    print(
        f"Mean squared quantization error: "
        f"{np.mean(error ** 2):.12f}"
    )


# =====================================================================
# EXPERIMENT 3
# COMPARE SQNR
# =====================================================================

def experiment_sqnr():

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: SQNR")
    print("=" * 70)

    time, original = generate_sine_wave(
        DEFAULT_FREQUENCY,
        DEFAULT_SAMPLE_RATE,
        1.0,
        amplitude=0.95
    )

    results = []

    for bit_depth in BIT_DEPTHS:

        quantized, _ = uniform_quantize(
            original,
            bit_depth
        )

        empirical = calculate_sqnr(
            original,
            quantized
        )

        theoretical = theoretical_sqnr(
            bit_depth
        )

        results.append(
            (
                bit_depth,
                empirical,
                theoretical
            )
        )

    print(
        f"{'Bit Depth':<15}"
        f"{'Empirical SQNR':<25}"
        f"{'Theoretical SQNR'}"
    )

    print("-" * 65)

    for bit_depth, empirical, theoretical in results:

        print(
            f"{bit_depth:<15}"
            f"{empirical:<25.2f}"
            f"{theoretical:.2f} dB"
        )

    bit_depths = [
        item[0] for item in results
    ]

    empirical_values = [
        item[1] for item in results
    ]

    theoretical_values = [
        item[2] for item in results
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        bit_depths,
        empirical_values,
        "o-",
        label="Empirical SQNR"
    )

    plt.plot(
        bit_depths,
        theoretical_values,
        "s--",
        label="Theoretical SQNR"
    )

    plt.xlabel("Bit Depth")

    plt.ylabel("SQNR (dB)")

    plt.title(
        "Bit Depth vs Signal-to-Quantization-Noise Ratio"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 4
# QUANTIZATION LEVELS
# =====================================================================

def experiment_quantization_levels():

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: NUMBER OF QUANTIZATION LEVELS")
    print("=" * 70)

    bit_depths = np.arange(
        2,
        25
    )

    levels = 2 ** bit_depths

    plt.figure(figsize=(12, 6))

    plt.semilogy(
        bit_depths,
        levels,
        "o-"
    )

    plt.xlabel("Bit Depth")

    plt.ylabel("Number of Levels (log scale)")

    plt.title(
        "Bit Depth vs Number of Quantization Levels"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    for bit_depth in [8, 16, 24, 32]:

        print(
            f"{bit_depth}-bit: "
            f"{2 ** bit_depth:,} possible levels"
        )


# =====================================================================
# EXPERIMENT 5
# CREATE AND INSPECT PCM WAV
# =====================================================================

def experiment_pcm_wav():

    print("\n" + "=" * 70)
    print("EXPERIMENT 5: PCM WAV")
    print("=" * 70)

    filename = Path(
        "lesson3_generated_16bit.wav"
    )

    sampling_rate = 48000

    time, signal = generate_sine_wave(
        440,
        sampling_rate,
        2.0,
        amplitude=0.8
    )

    # Write signed 16-bit PCM WAV.
    #
    # SoundFile handles the conversion from floating-point samples
    # to the requested PCM subtype.

    sf.write(
        filename,
        signal,
        sampling_rate,
        subtype="PCM_16"
    )

    print(f"Created WAV file: {filename}")

    # Read the WAV file back.

    loaded_audio, loaded_fs = sf.read(
        filename,
        dtype="float32"
    )

    information = sf.info(
        filename
    )

    print("\nWAV FILE INFORMATION")

    print(
        f"Sample rate: {information.samplerate} Hz"
    )

    print(
        f"Channels: {information.channels}"
    )

    print(
        f"Frames: {information.frames}"
    )

    print(
        f"Duration: {information.duration:.4f} seconds"
    )

    print(
        f"Subtype: {information.subtype}"
    )

    print(
        f"Format: {information.format}"
    )

    print(
        f"Loaded NumPy shape: {loaded_audio.shape}"
    )

    print(
        f"Loaded NumPy dtype: {loaded_audio.dtype}"
    )

    print(
        f"Peak amplitude: {np.max(np.abs(loaded_audio)):.6f}"
    )

    print(
        f"RMS amplitude: {np.sqrt(np.mean(loaded_audio ** 2)):.6f}"
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        np.arange(len(loaded_audio)) / loaded_fs,
        loaded_audio
    )

    plt.title(
        "Audio Read Back from 16-bit PCM WAV"
    )

    plt.xlabel("Time (seconds)")

    plt.ylabel("Amplitude")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# =====================================================================
# AUDIO STORAGE CALCULATOR
# =====================================================================

def calculate_audio_storage(
    sampling_rate,
    bit_depth,
    channels,
    duration_seconds
):
    """
    Calculate uncompressed PCM audio storage.

    Formula:

        Storage (bytes) =
            sampling_rate
            * bit_depth
            * channels
            * duration_seconds
            / 8

    This excludes file headers and metadata.
    """

    total_bits = (
        sampling_rate
        * bit_depth
        * channels
        * duration_seconds
    )

    total_bytes = total_bits / 8

    return total_bytes


def experiment_storage():

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: AUDIO STORAGE")
    print("=" * 70)

    duration = 60

    configurations = [
        (16000, 16, 1),
        (24000, 16, 1),
        (48000, 16, 1),
        (48000, 24, 1),
        (48000, 24, 2)
    ]

    print(
        f"{'Configuration':<35}"
        f"{'Storage (MB)'}"
    )

    print("-" * 60)

    for fs, bit_depth, channels in configurations:

        storage = calculate_audio_storage(
            fs,
            bit_depth,
            channels,
            duration
        )

        storage_mb = storage / (1024 ** 2)

        description = (
            f"{fs} Hz / {bit_depth}-bit / "
            f"{channels} channel(s)"
        )

        print(
            f"{description:<35}"
            f"{storage_mb:.2f}"
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 Lesson 3: Bit depth and quantization."
        )
    )

    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6],
        help=(
            "0 = All experiments, "
            "1 = Quantization, "
            "2 = Quantization error, "
            "3 = SQNR, "
            "4 = Quantization levels, "
            "5 = PCM WAV, "
            "6 = Audio storage"
        )
    )

    args = parser.parse_args()

    print("\n")
    print("=" * 70)
    print("PHASE 1 — LESSON 3")
    print("BIT DEPTH AND QUANTIZATION")
    print("=" * 70)

    experiments = {
        1: experiment_quantization,
        2: experiment_quantization_error,
        3: experiment_sqnr,
        4: experiment_quantization_levels,
        5: experiment_pcm_wav,
        6: experiment_storage
    }

    if args.experiment == 0:

        for experiment in experiments.values():

            experiment()

    else:

        experiments[args.experiment]()

    print("\n")
    print("=" * 70)
    print("LESSON 3 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()