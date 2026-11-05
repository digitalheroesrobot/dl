"""
File: phase1_lesson15_neural_vocoder_fundamentals.py

PHASE 1 — LESSON 15
NEURAL VOCODERS AND WAVEFORM GENERATION
============================================================

PURPOSE
-------
Understand how acoustic features become an audio waveform.

This lesson implements:

1. Synthetic speech-like waveform generation.
2. STFT.
3. Magnitude spectrogram.
4. Mel filterbank.
5. Mel spectrogram.
6. Approximate Mel-to-linear reconstruction.
7. Griffin-Lim phase reconstruction.
8. Audio export.
9. Waveform and spectrogram visualization.

IMPORTANT
---------
This lesson implements traditional signal-processing
reconstruction, not a trained neural vocoder.

The next development stage will implement a neural vocoder
using a generator network and waveform reconstruction losses.

============================================================
1. WAVEFORM REPRESENTATION
============================================================

A waveform is a discrete-time signal:

    x[n]

At a sample rate of 24,000 Hz:

    24,000 samples = 1 second

The waveform contains amplitude information over time.

============================================================
2. SHORT-TIME FOURIER TRANSFORM
============================================================

The STFT transforms short waveform segments into frequency
representations.

    X(k,m) = sum_n x[n] w[n-mH] exp(-j2*pi*k*n/N)

Where:

    N = FFT size
    H = hop length
    w = analysis window
    m = frame index
    k = frequency-bin index

The STFT output is complex:

    X = real(X) + j * imaginary(X)

Magnitude:

    M = abs(X)

Phase:

    phi = angle(X)

============================================================
3. WHY MAGNITUDE IS NOT ENOUGH
============================================================

A complex STFT contains:

    X = M * exp(j * phi)

A magnitude spectrogram contains only:

    M = abs(X)

The phase information is discarded.

Without phase, exact waveform reconstruction is generally
not possible.

However, an approximate phase can be estimated.

============================================================
4. MEL SPECTROGRAM
============================================================

The Mel scale approximates human frequency perception.

One commonly used formula is:

    mel = 2595 * log10(1 + frequency / 700)

A Mel filterbank combines linear-frequency spectral bins
into a smaller number of perceptually spaced bands.

    Mel = Filterbank * Magnitude

For this lesson:

    FFT size = 1024
    Mel bands = 80
    Sample rate = 24 kHz

============================================================
5. MEL INVERSION
============================================================

The Mel transform reduces dimensionality.

Therefore, the inverse is not exact.

If:

    Mel = F * Magnitude

We can estimate:

    Magnitude_estimated = pinv(F) * Mel

Where pinv means the Moore-Penrose pseudoinverse.

This is an approximate solution.

A production system may use non-negative least squares,
regularized inversion, or a learned inverse transform.

============================================================
6. GRIFFIN-LIM ALGORITHM
============================================================

Griffin-Lim estimates phase from a magnitude spectrogram.

Given target magnitude M:

    1. Initialize phase.
    2. Create complex STFT:

           X = M * exp(j * phase)

    3. Reconstruct waveform:

           x = inverse_STFT(X)

    4. Calculate a new STFT:

           X_new = STFT(x)

    5. Keep the new phase:

           phase = angle(X_new)

    6. Repeat.

This is an iterative consistency procedure.

The result is an approximate waveform whose STFT magnitude
approaches the target magnitude.

It is NOT a neural vocoder.

============================================================
7. NEURAL VOCODER ARCHITECTURES
============================================================

A. AUTOREGRESSIVE VOCODER

Examples:
    WaveNet

Predicts samples sequentially:

    P(x_t | x_1, ..., x_(t-1), Mel)

Advantages:
    - Powerful waveform modeling.
    - Can produce high-quality speech.

Disadvantages:
    - Sequential generation.
    - Computationally expensive.

B. GAN VOCODER

Example:
    HiFi-GAN

Generator:

    waveform = Generator(Mel)

Discriminator:
    Distinguishes real audio from generated audio.

Generator objectives may include:

    L_G =
        L_adversarial
        + lambda_fm * L_feature_matching
        + lambda_mel * L_mel

C. DIFFUSION VOCODER

Starts with noise and iteratively transforms it into
a waveform conditioned on acoustic features.

============================================================
8. TRAINING A REAL NEURAL VOCODER
============================================================

A real vocoder requires paired data:

    Mel spectrogram <-> waveform

For every training recording:

    1. Load the waveform.
    2. Compute its Mel spectrogram.
    3. Feed Mel features to the generator.
    4. Generate waveform samples.
    5. Compare generated audio with real audio.
    6. Update generator and discriminator.

A common multi-resolution STFT loss is:

    L_sc =
        || |STFT(real)| - |STFT(generated)| ||_F
        / (|| |STFT(real)| ||_F + epsilon)

A log-magnitude loss is:

    L_mag =
        mean(
            abs(
                log(|STFT(real)| + epsilon)
                -
                log(|STFT(generated)| + epsilon)
            )
        )

These losses encourage generated audio to match real
audio at multiple spectral resolutions.

============================================================
REQUIREMENTS
============================================================

    pip install numpy scipy matplotlib soundfile

RUN
---

    python phase1_lesson15_neural_vocoder_fundamentals.py

    python neural_vocoder_fundamentals.py \
        --audio my_voice.wav

    python phase1_lesson15_neural_vocoder_fundamentals.py \
        --iterations 100

    python phase1_lesson15_neural_vocoder_fundamentals.py \
        --no-plot

============================================================
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf

from scipy.signal import stft, istft, resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 24000

FFT_SIZE = 1024

HOP_LENGTH = 256

WINDOW_SIZE = 1024

NUM_MEL_BANDS = 80

MIN_FREQUENCY = 0.0

MAX_FREQUENCY = SAMPLE_RATE / 2

EPSILON = 1e-8


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path, target_sample_rate=SAMPLE_RATE):

    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True
    )

    # Convert stereo or multichannel audio to mono.
    audio = np.mean(
        audio,
        axis=1
    )

    # Resample when necessary.
    if sample_rate != target_sample_rate:

        from math import gcd

        divisor = gcd(
            sample_rate,
            target_sample_rate
        )

        up = target_sample_rate // divisor

        down = sample_rate // divisor

        audio = resample_poly(
            audio,
            up,
            down
        )

    audio = audio.astype(
        np.float32
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = audio / peak * 0.95

    return audio


# ============================================================
# SYNTHETIC SPEECH-LIKE WAVEFORM
# ============================================================

def generate_speech_like_audio(
    duration=3.0,
    sample_rate=SAMPLE_RATE,
    seed=42
):
    """
    Generate a simplified speech-like signal.

    The waveform contains:
        - Harmonic voiced excitation.
        - Slowly varying fundamental frequency.
        - Formant-like spectral resonances.
        - Amplitude modulation.
        - Short unvoiced noise regions.

    This is not natural speech.

    It is designed to demonstrate spectral analysis
    and waveform reconstruction.
    """

    rng = np.random.default_rng(seed)

    num_samples = int(
        duration * sample_rate
    )

    time = np.arange(
        num_samples
    ) / sample_rate

    # --------------------------------------------------------
    # Fundamental frequency contour
    # --------------------------------------------------------

    f0 = (
        120.0
        + 18.0 * np.sin(2 * np.pi * 0.5 * time)
        + 8.0 * np.sin(2 * np.pi * 1.2 * time)
    )

    # Integrate instantaneous frequency to obtain phase.
    phase = 2 * np.pi * np.cumsum(
        f0 / sample_rate
    )

    # --------------------------------------------------------
    # Harmonic excitation
    # --------------------------------------------------------

    voiced = np.zeros(
        num_samples,
        dtype=np.float64
    )

    for harmonic in range(1, 25):

        amplitude = 1.0 / harmonic

        voiced += amplitude * np.sin(
            harmonic * phase
        )

    # --------------------------------------------------------
    # Formant-like spectral shaping
    # --------------------------------------------------------

    # A simple combination of resonant sinusoidal components.
    #
    # This is a simplified teaching approximation.
    # It is not a full vocal-tract filter model.

    formant_signal = (
        0.9 * np.sin(phase)
        + 0.5 * np.sin(2.0 * phase)
        + 0.25 * np.sin(3.0 * phase)
    )

    # Blend the harmonic excitation and formant-like signal.
    voiced = (
        0.7 * voiced
        + 0.3 * formant_signal
    )

    # --------------------------------------------------------
    # Amplitude envelope
    # --------------------------------------------------------

    envelope = (
        0.5
        + 0.5 * np.sin(2 * np.pi * 0.7 * time) ** 2
    )

    envelope = np.clip(
        envelope,
        0.1,
        1.0
    )

    voiced *= envelope

    # --------------------------------------------------------
    # Unvoiced regions
    # --------------------------------------------------------

    unvoiced_noise = rng.normal(
        0.0,
        0.15,
        size=num_samples
    )

    unvoiced_mask = np.zeros(
        num_samples,
        dtype=bool
    )

    unvoiced_mask[
        int(0.8 * sample_rate):
        int(1.1 * sample_rate)
    ] = True

    unvoiced_mask[
        int(2.1 * sample_rate):
        int(2.3 * sample_rate)
    ] = True

    voiced[unvoiced_mask] = (
        unvoiced_noise[unvoiced_mask]
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    peak = np.max(
        np.abs(voiced)
    )

    if peak > 0:

        voiced = voiced / peak * 0.95

    return voiced.astype(
        np.float32
    )


# ============================================================
# STFT
# ============================================================

def calculate_stft(audio):

    frequencies, times, spectrum = stft(
        audio,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=WINDOW_SIZE,
        noverlap=WINDOW_SIZE - HOP_LENGTH,
        nfft=FFT_SIZE,
        boundary="zeros",
        padded=True,
        scaling="spectrum"
    )

    return (
        frequencies,
        times,
        spectrum
    )


def calculate_istft(
    spectrum,
    expected_length=None
):

    _, audio = istft(
        spectrum,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=WINDOW_SIZE,
        noverlap=WINDOW_SIZE - HOP_LENGTH,
        nfft=FFT_SIZE,
        input_onesided=True,
        boundary=True,
        scaling="spectrum"
    )

    if expected_length is not None:

        if len(audio) > expected_length:

            audio = audio[:expected_length]

        elif len(audio) < expected_length:

            audio = np.pad(
                audio,
                (0, expected_length - len(audio))
            )

    return audio.astype(
        np.float32
    )


# ============================================================
# MEL SCALE
# ============================================================

def hz_to_mel(frequency):

    return (
        2595.0
        * np.log10(
            1.0 + frequency / 700.0
        )
    )


def mel_to_hz(mel):

    return (
        700.0
        * (
            10.0 ** (mel / 2595.0)
            - 1.0
        )
    )


# ============================================================
# MEL FILTERBANK
# ============================================================

def create_mel_filterbank(
    sample_rate=SAMPLE_RATE,
    fft_size=FFT_SIZE,
    num_mel_bands=NUM_MEL_BANDS,
    min_frequency=MIN_FREQUENCY,
    max_frequency=MAX_FREQUENCY
):
    """
    Create triangular Mel filters.

    Output shape:

        [num_mel_bands, fft_size // 2 + 1]

    Each row represents one Mel filter.
    """

    min_mel = hz_to_mel(
        min_frequency
    )

    max_mel = hz_to_mel(
        max_frequency
    )

    mel_points = np.linspace(
        min_mel,
        max_mel,
        num_mel_bands + 2
    )

    hz_points = mel_to_hz(
        mel_points
    )

    fft_frequencies = np.linspace(
        0,
        sample_rate / 2,
        fft_size // 2 + 1
    )

    filterbank = np.zeros(
        (
            num_mel_bands,
            len(fft_frequencies)
        ),
        dtype=np.float64
    )

    for mel_index in range(num_mel_bands):

        left = hz_points[mel_index]

        center = hz_points[mel_index + 1]

        right = hz_points[mel_index + 2]

        rising = (
            fft_frequencies - left
        ) / max(
            center - left,
            EPSILON
        )

        falling = (
            right - fft_frequencies
        ) / max(
            right - center,
            EPSILON
        )

        filter_values = np.minimum(
            rising,
            falling
        )

        filterbank[mel_index] = np.maximum(
            filter_values,
            0.0
        )

    return filterbank


# ============================================================
# MEL SPECTROGRAM
# ============================================================

def calculate_mel_spectrogram(
    magnitude,
    filterbank
):

    mel = filterbank @ magnitude

    log_mel = np.log(
        np.maximum(
            mel,
            EPSILON
        )
    )

    return (
        mel,
        log_mel
    )


# ============================================================
# APPROXIMATE MEL INVERSION
# ============================================================

def invert_mel_spectrogram(
    mel,
    filterbank
):
    """
    Estimate a linear-frequency magnitude spectrogram.

    Forward:

        Mel = F @ Magnitude

    Approximate inverse:

        Magnitude = pinv(F) @ Mel

    The inversion is approximate because Mel filtering
    loses frequency resolution.

    A pseudoinverse may generate negative values.
    Magnitudes cannot be negative, so we clamp them to zero.
    """

    inverse_filterbank = np.linalg.pinv(
        filterbank
    )

    estimated_magnitude = (
        inverse_filterbank @ mel
    )

    estimated_magnitude = np.maximum(
        estimated_magnitude,
        0.0
    )

    return estimated_magnitude


# ============================================================
# GRIFFIN-LIM PHASE RECONSTRUCTION
# ============================================================

def griffin_lim(
    magnitude,
    iterations=60,
    seed=42,
    expected_length=None
):
    """
    Estimate a waveform from a linear-frequency
    magnitude spectrogram.

    Algorithm:

        Initialize random phase.

        Repeat:

            complex_spectrum =
                magnitude * exp(j * phase)

            waveform =
                inverse_STFT(complex_spectrum)

            estimated_spectrum =
                STFT(waveform)

            phase =
                angle(estimated_spectrum)

    The magnitude is kept fixed while phase is refined.

    This implementation uses SciPy STFT/ISTFT conventions.

    It is an educational implementation, not a production
    high-quality vocoder.
    """

    rng = np.random.default_rng(
        seed
    )

    random_phase = rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=magnitude.shape
    )

    complex_spectrum = (
        magnitude
        * np.exp(1j * random_phase)
    )

    for iteration in range(iterations):

        # ----------------------------------------------------
        # STEP 1: Inverse STFT
        # ----------------------------------------------------

        audio = calculate_istft(
            complex_spectrum,
            expected_length=expected_length
        )

        # ----------------------------------------------------
        # STEP 2: Recalculate STFT
        # ----------------------------------------------------

        _, _, estimated_spectrum = calculate_stft(
            audio
        )

        # Match dimensions if necessary.
        min_frames = min(
            estimated_spectrum.shape[1],
            magnitude.shape[1]
        )

        estimated_spectrum = estimated_spectrum[
            :,
            :min_frames
        ]

        target_magnitude = magnitude[
            :,
            :min_frames
        ]

        # ----------------------------------------------------
        # STEP 3: Update phase
        # ----------------------------------------------------

        phase = np.angle(
            estimated_spectrum
        )

        # ----------------------------------------------------
        # STEP 4: Restore target magnitude
        # ----------------------------------------------------

        complex_spectrum = (
            target_magnitude
            * np.exp(1j * phase)
        )

        if (
            iteration + 1
        ) % 10 == 0:

            print(
                f"Griffin-Lim iteration "
                f"{iteration + 1}/{iterations}"
            )

    reconstructed_audio = calculate_istft(
        complex_spectrum,
        expected_length=expected_length
    )

    peak = np.max(
        np.abs(reconstructed_audio)
    )

    if peak > 0:

        reconstructed_audio = (
            reconstructed_audio / peak * 0.95
        )

    return reconstructed_audio


# ============================================================
# SPECTRAL ERROR
# ============================================================

def calculate_spectral_error(
    reference_magnitude,
    reconstructed_magnitude
):
    """
    Calculate a simple normalized magnitude error.

    This is NOT a perceptual speech-quality metric.

    It only measures numerical spectral difference.
    """

    min_frames = min(
        reference_magnitude.shape[1],
        reconstructed_magnitude.shape[1]
    )

    reference = reference_magnitude[
        :,
        :min_frames
    ]

    reconstructed = reconstructed_magnitude[
        :,
        :min_frames
    ]

    numerator = np.linalg.norm(
        reference - reconstructed
    )

    denominator = np.linalg.norm(
        reference
    ) + EPSILON

    return float(
        numerator / denominator
    )


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_results(
    audio,
    reconstructed_audio,
    times,
    magnitude,
    mel,
    reconstructed_magnitude,
    sample_rate=SAMPLE_RATE
):

    original_time = (
        np.arange(len(audio)) / sample_rate
    )

    reconstructed_time = (
        np.arange(len(reconstructed_audio))
        / sample_rate
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 12)
    )

    fig.suptitle(
        "Vocoder Fundamentals — Griffin-Lim Reconstruction",
        fontsize=16
    )

    # --------------------------------------------------------
    # Original waveform
    # --------------------------------------------------------

    axes[0, 0].plot(
        original_time,
        audio
    )

    axes[0, 0].set_title(
        "Original waveform"
    )

    axes[0, 0].set_xlabel(
        "Time (seconds)"
    )

    axes[0, 0].set_ylabel(
        "Amplitude"
    )

    # --------------------------------------------------------
    # Reconstructed waveform
    # --------------------------------------------------------

    axes[0, 1].plot(
        reconstructed_time,
        reconstructed_audio
    )

    axes[0, 1].set_title(
        "Griffin-Lim reconstructed waveform"
    )

    axes[0, 1].set_xlabel(
        "Time (seconds)"
    )

    axes[0, 1].set_ylabel(
        "Amplitude"
    )

    # --------------------------------------------------------
    # Original magnitude spectrogram
    # --------------------------------------------------------

    axes[1, 0].imshow(
        20 * np.log10(
            np.maximum(
                magnitude,
                EPSILON
            )
        ),
        origin="lower",
        aspect="auto",
        extent=[
            times[0],
            times[-1],
            0,
            sample_rate / 2
        ]
    )

    axes[1, 0].set_title(
        "Original linear-frequency magnitude"
    )

    axes[1, 0].set_xlabel(
        "Time (seconds)"
    )

    axes[1, 0].set_ylabel(
        "Frequency (Hz)"
    )

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    axes[1, 1].imshow(
        10 * np.log10(
            np.maximum(
                mel,
                EPSILON
            )
        ),
        origin="lower",
        aspect="auto"
    )

    axes[1, 1].set_title(
        "Mel spectrogram"
    )

    axes[1, 1].set_xlabel(
        "Time frame"
    )

    axes[1, 1].set_ylabel(
        "Mel channel"
    )

    # --------------------------------------------------------
    # Reconstructed magnitude
    # --------------------------------------------------------

    axes[2, 0].imshow(
        20 * np.log10(
            np.maximum(
                reconstructed_magnitude,
                EPSILON
            )
        ),
        origin="lower",
        aspect="auto"
    )

    axes[2, 0].set_title(
        "Reconstructed magnitude spectrogram"
    )

    axes[2, 0].set_xlabel(
        "Time frame"
    )

    axes[2, 0].set_ylabel(
        "Frequency bin"
    )

    # --------------------------------------------------------
    # Waveform comparison
    # --------------------------------------------------------

    comparison_length = min(
        len(audio),
        len(reconstructed_audio)
    )

    display_length = min(
        comparison_length,
        int(0.08 * sample_rate)
    )

    comparison_time = (
        np.arange(display_length)
        / sample_rate
    )

    axes[2, 1].plot(
        comparison_time,
        audio[:display_length],
        label="Original"
    )

    axes[2, 1].plot(
        comparison_time,
        reconstructed_audio[:display_length],
        label="Reconstructed",
        alpha=0.75
    )

    axes[2, 1].set_title(
        "Waveform comparison — first 80 ms"
    )

    axes[2, 1].set_xlabel(
        "Time (seconds)"
    )

    axes[2, 1].set_ylabel(
        "Amplitude"
    )

    axes[2, 1].legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Mel spectrogram and Griffin-Lim "
            "waveform reconstruction experiment."
        )
    )

    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Optional input WAV file."
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=60
    )

    parser.add_argument(
        "--no-plot",
        action="store_true"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="griffin_lim_reconstructed.wav"
    )

    args = parser.parse_args()

    print("=" * 70)

    print("PHASE 1 — LESSON 15")

    print("VOCODER FUNDAMENTALS")

    print("=" * 70)

    print(
        f"\nSample rate: {SAMPLE_RATE} Hz"
    )

    print(
        f"FFT size: {FFT_SIZE}"
    )

    print(
        f"Hop length: {HOP_LENGTH}"
    )

    print(
        f"Mel bands: {NUM_MEL_BANDS}"
    )

    # --------------------------------------------------------
    # STEP 1: Load or generate waveform
    # --------------------------------------------------------

    if args.audio:

        if not os.path.exists(args.audio):

            raise FileNotFoundError(
                args.audio
            )

        audio = load_audio(
            args.audio
        )

        print(
            f"\nLoaded audio: {args.audio}"
        )

    else:

        audio = generate_speech_like_audio()

        print(
            "\nGenerated synthetic speech-like waveform."
        )

    print(
        f"Audio duration: {len(audio) / SAMPLE_RATE:.3f}s"
    )

    # --------------------------------------------------------
    # STEP 2: STFT
    # --------------------------------------------------------

    frequencies, times, spectrum = calculate_stft(
        audio
    )

    magnitude = np.abs(
        spectrum
    )

    print(
        f"\nSTFT shape: {spectrum.shape}"
    )

    # --------------------------------------------------------
    # STEP 3: Mel filterbank
    # --------------------------------------------------------

    filterbank = create_mel_filterbank()

    print(
        f"Mel filterbank shape: {filterbank.shape}"
    )

    # --------------------------------------------------------
    # STEP 4: Mel spectrogram
    # --------------------------------------------------------

    mel, log_mel = calculate_mel_spectrogram(
        magnitude,
        filterbank
    )

    print(
        f"Mel spectrogram shape: {mel.shape}"
    )

    # --------------------------------------------------------
    # STEP 5: Invert Mel spectrogram
    # --------------------------------------------------------

    estimated_magnitude = invert_mel_spectrogram(
        mel,
        filterbank
    )

    print(
        f"Estimated magnitude shape: "
        f"{estimated_magnitude.shape}"
    )

    # --------------------------------------------------------
    # STEP 6: Griffin-Lim reconstruction
    # --------------------------------------------------------

    print(
        "\nStarting Griffin-Lim reconstruction..."
    )

    reconstructed_audio = griffin_lim(
        estimated_magnitude,
        iterations=args.iterations,
        expected_length=len(audio)
    )

    # --------------------------------------------------------
    # STEP 7: Analyze reconstructed waveform
    # --------------------------------------------------------

    _, _, reconstructed_spectrum = calculate_stft(
        reconstructed_audio
    )

    reconstructed_magnitude = np.abs(
        reconstructed_spectrum
    )

    spectral_error = calculate_spectral_error(
        magnitude,
        reconstructed_magnitude
    )

    print("\nRESULTS")

    print("-" * 70)

    print(
        f"Original samples: {len(audio)}"
    )

    print(
        f"Reconstructed samples: {len(reconstructed_audio)}"
    )

    print(
        f"Original duration: "
        f"{len(audio) / SAMPLE_RATE:.3f}s"
    )

    print(
        f"Reconstructed duration: "
        f"{len(reconstructed_audio) / SAMPLE_RATE:.3f}s"
    )

    print(
        f"Normalized spectral error: {spectral_error:.5f}"
    )

    # --------------------------------------------------------
    # STEP 8: Export waveform
    # --------------------------------------------------------

    sf.write(
        args.output,
        reconstructed_audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    print(
        f"\nSaved reconstructed audio: {args.output}"
    )

    # --------------------------------------------------------
    # STEP 9: Visualize
    # --------------------------------------------------------

    if not args.no_plot:

        visualize_results(
            audio,
            reconstructed_audio,
            times,
            magnitude,
            mel,
            reconstructed_magnitude
        )

    print("\n" + "=" * 70)

    print("LESSON 15 COMPLETE")

    print("=" * 70)

    print(
        "\nThe reconstructed waveform is approximate."
    )

    print(
        "It is not expected to sound identical to the original."
    )

    print(
        "\nNext: learn and implement a neural vocoder architecture."
    )


if __name__ == "__main__":

    main()