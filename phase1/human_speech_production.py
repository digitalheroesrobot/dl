"""
=======================================================================
BUILD YOUR OWN VOICE-CLONING TTS SYSTEM
PHASE 1 — DIGITAL AUDIO FUNDAMENTALS

LESSON 6:
HUMAN SPEECH PRODUCTION, VOCAL TRACT, FORMANTS, PITCH AND PHONETICS
=======================================================================

LEARNING OBJECTIVES
-------------------

1. Understand the physical process of human speech production.
2. Understand airflow and vocal-fold vibration.
3. Understand the source-filter model.
4. Understand voiced and unvoiced excitation.
5. Understand fundamental frequency (F0).
6. Understand pitch period.
7. Understand harmonics.
8. Understand vocal-tract resonances.
9. Understand formants F1, F2 and F3.
10. Understand the difference between F0 and formants.
11. Understand vowel articulation.
12. Understand basic phonetic features.
13. Understand how speech sounds can be represented mathematically.
14. Understand why speech production theory matters for voice cloning.

=======================================================================
1. HUMAN SPEECH PRODUCTION
=======================================================================

Human speech is produced by coordinated biological systems.

The major components are:

    Lungs
        |
        v
    Trachea
        |
        v
    Larynx
        |
        v
    Vocal folds
        |
        v
    Pharynx
        |
        v
    Oral and nasal cavities
        |
        v
    Speech radiation

The lungs provide airflow.

The vocal folds can vibrate when airflow passes through the
appropriately configured glottis.

The tongue, lips, jaw, palate and other articulators change the
shape of the vocal tract.

This produces different speech sounds.

=======================================================================
2. SOURCE-FILTER MODEL
=======================================================================

A simplified model of speech production is:

    S(z) = G(z) * H(z)

Where:

    S(z) = speech output
    G(z) = excitation source
    H(z) = vocal-tract transfer function

In the time domain:

    s[n] = g[n] * h[n]

Here, * represents convolution.

The source provides excitation.

The filter shapes the spectrum.

For voiced speech:

    The source is approximately periodic.

For unvoiced speech:

    The source is approximately noise-like.

This model is simplified. Real speech production includes
time-varying articulation, nasal coupling, source-filter
interaction and radiation effects.

=======================================================================
3. FUNDAMENTAL FREQUENCY
=======================================================================

For a periodic signal:

    F0 = 1 / T0

Where:

    F0 = fundamental frequency in Hz
    T0 = period in seconds

Example:

    T0 = 0.005 seconds

    F0 = 1 / 0.005

    F0 = 200 Hz

The fundamental frequency is closely related to perceived pitch.

The vocal folds produce a fundamental frequency and harmonics.

    F0
    2*F0
    3*F0
    4*F0
    ...

=======================================================================
4. FORMANTS
=======================================================================

Formants are resonant frequency regions associated with the
vocal tract.

Commonly discussed formants:

    F1
    F2
    F3
    F4

F1 is generally associated with vowel height.

F2 is generally associated with tongue frontness.

F3 contributes additional information about vowel quality
and vocal-tract characteristics.

The actual formant frequencies depend on the speaker and
articulatory configuration.

=======================================================================
5. F0 IS NOT A FORMANT
=======================================================================

Fundamental frequency:

    Related to vocal-fold vibration.

Formants:

    Related to vocal-tract resonances.

A speaker can change pitch without changing the vowel.

A speaker can change the vowel without necessarily changing pitch.

These are different acoustic dimensions.

=======================================================================
6. VOICED AND UNVOICED SOUNDS
=======================================================================

Voiced sounds:

    - Vocal folds vibrate periodically.
    - Usually show harmonic structure.
    - Have an identifiable F0.

Examples:

    /a/
    /i/
    /u/
    /m/
    /n/
    /z/

Unvoiced sounds:

    - Usually have noise-like excitation.
    - Do not have a stable periodic F0.
    - Often show broadband or concentrated noise.

Examples:

    /s/
    /f/
    /h/
    /sh/

Some consonants contain both voiced and unvoiced intervals.

Speech is not simply divided into two perfectly separate classes.

=======================================================================
7. VOWELS AND CONSONANTS
=======================================================================

Vowels are produced with a relatively open vocal tract.

Their identity is strongly associated with formant patterns.

Consonants involve constrictions or closures in the vocal tract.

Examples of articulatory properties:

    Place of articulation:
        Bilabial
        Alveolar
        Velar

    Manner of articulation:
        Stop
        Fricative
        Nasal
        Approximant

    Voicing:
        Voiced
        Unvoiced

=======================================================================
8. FORMANT APPROXIMATION
=======================================================================

A simplified vocal-tract filter can be constructed from resonators.

A second-order resonator may be represented by:

    H(z) = 1 / (
        1 - 2*r*cos(w0)*z^-1 + r^2*z^-2
    )

Where:

    r  = pole radius
    w0 = resonant angular frequency

Resonant frequency:

    F = w0 * fs / (2*pi)

The pole radius controls resonance bandwidth.

For a narrow resonance:

    r is close to 1.

Multiple resonators can be cascaded to approximate a vocal-tract
filter.

This is a simplified educational model, not a complete
articulatory speech synthesizer.

=======================================================================
9. F0 ESTIMATION
=======================================================================

A simple F0 estimation method is autocorrelation.

For a signal x[n]:

    R[k] = sum(x[n] * x[n+k])

A strong autocorrelation peak at lag k indicates a possible
period.

Then:

    F0 = fs / k

Autocorrelation has limitations:

    - Octave errors
    - Background noise
    - Unvoiced speech
    - Multiple periodic components
    - Low-frequency interference

Production-quality pitch trackers use more sophisticated
algorithms.

=======================================================================
10. VOICE-CLONING CONNECTION
=======================================================================

A voice-cloning system must model several dimensions:

    Linguistic content
    Speaker characteristics
    Pitch and prosody
    Timing and duration
    Spectral characteristics
    Voice quality

The source-filter model provides a conceptual foundation.

A modern neural TTS model does not necessarily implement
an explicit source-filter architecture.

Nevertheless, understanding these physical concepts helps us
interpret what neural networks learn.

=======================================================================
DEPENDENCIES
=======================================================================

    pip install numpy scipy matplotlib soundfile

=======================================================================
python phase1_lesson6_human_speech_production.py \
    --experiment 6 \
    --audio my_voice.wav
"""

# =====================================================================
# IMPORTS
# =====================================================================

import argparse

import numpy as np

import matplotlib.pyplot as plt

import soundfile as sf

from scipy import signal as scipy_signal

from scipy.signal import find_peaks, lfilter


# =====================================================================
# CONFIGURATION
# =====================================================================

FS = 24000

DURATION = 1.0

EPSILON = 1e-10


# =====================================================================
# AUDIO NORMALIZATION
# =====================================================================

def normalize_audio(audio):
    peak = np.max(
        np.abs(audio)
    )

    if peak < EPSILON:
        return audio

    return audio / peak


# =====================================================================
# EXPERIMENTAL SOURCE GENERATION
# =====================================================================

def generate_voiced_excitation(
    fs,
    duration,
    f0=150,
    harmonics=20
):
    """
    Generate simplified periodic vocal-fold excitation.

    This is a harmonic educational signal.

    It is not a physical simulation of vocal-fold biomechanics.
    """

    number_of_samples = int(
        fs * duration
    )

    time = np.arange(
        number_of_samples
    ) / fs

    excitation = np.zeros(
        number_of_samples
    )

    for harmonic in range(
        1,
        harmonics + 1
    ):

        frequency = f0 * harmonic

        if frequency >= fs / 2:
            break

        amplitude = 1.0 / harmonic

        excitation += amplitude * np.sin(
            2 * np.pi * frequency * time
        )

    # Mild amplitude modulation to make the example less static.

    envelope = (
        0.85
        + 0.15 * np.sin(
            2 * np.pi * 2 * time
        )
    )

    excitation *= envelope

    return normalize_audio(
        excitation
    )


def generate_unvoiced_excitation(
    fs,
    duration
):
    """
    Generate noise-like excitation.

    This approximates the broadband excitation associated
    with unvoiced sounds.

    Real fricatives often contain frequency-dependent noise
    shaped by the articulation location.
    """

    number_of_samples = int(
        fs * duration
    )

    rng = np.random.default_rng(
        seed=42
    )

    noise = rng.normal(
        0,
        1,
        number_of_samples
    )

    return normalize_audio(
        noise
    )


# =====================================================================
# VOCAL-TRACT RESONATOR
# =====================================================================

def create_resonator(
    frequency,
    bandwidth,
    fs
):
    """
    Create a second-order resonator.

    Approximate transfer function:

        H(z) = 1 / (
            1 - 2*r*cos(w0)*z^-1 + r^2*z^-2
        )

    Pole radius:

        r = exp(-pi * bandwidth / fs)

    Resonance frequency:

        w0 = 2*pi*frequency/fs

    Bandwidth controls resonance width.

    Smaller bandwidth means a narrower resonance.
    """

    if frequency <= 0 or frequency >= fs / 2:

        raise ValueError(
            "Resonance frequency must be between 0 and Nyquist."
        )

    if bandwidth <= 0:

        raise ValueError(
            "Bandwidth must be positive."
        )

    radius = np.exp(
        -np.pi * bandwidth / fs
    )

    omega = (
        2 * np.pi * frequency / fs
    )

    denominator = [
        1.0,
        -2 * radius * np.cos(omega),
        radius ** 2
    ]

    numerator = [
        1.0 - 2 * radius * np.cos(omega) + radius ** 2
    ]

    return np.asarray(numerator), np.asarray(denominator)


def apply_vocal_tract_filter(
    excitation,
    formants,
    fs
):
    """
    Apply cascaded resonators.

    formants:

        [(frequency, bandwidth), ...]

    This produces a simplified source-filter model.
    """

    output = excitation.copy()

    for frequency, bandwidth in formants:

        b, a = create_resonator(
            frequency,
            bandwidth,
            fs
        )

        output = lfilter(
            b,
            a,
            output
        )

    return normalize_audio(
        output
    )


# =====================================================================
# SPECTRUM ANALYSIS
# =====================================================================

def calculate_spectrum(
    audio,
    fs
):
    """
    Calculate a windowed one-sided magnitude spectrum.
    """

    audio = np.asarray(
        audio,
        dtype=np.float64
    )

    window = np.hanning(
        len(audio)
    )

    windowed = audio * window

    spectrum = np.fft.rfft(
        windowed
    )

    frequencies = np.fft.rfftfreq(
        len(audio),
        d=1.0 / fs
    )

    magnitude = np.abs(
        spectrum
    )

    magnitude /= (
        np.sum(window) / 2 + EPSILON
    )

    return frequencies, magnitude


# =====================================================================
# AUTOCORRELATION F0 ESTIMATION
# =====================================================================

def estimate_f0_autocorrelation(
    audio,
    fs,
    min_f0=60,
    max_f0=400
):
    """
    Estimate F0 using autocorrelation.

    This educational estimator works best with a clean,
    sufficiently long voiced segment.

    It is not a production-quality pitch tracker.

    Search lag range:

        min_lag = fs / max_f0

        max_lag = fs / min_f0

    F0 estimate:

        F0 = fs / selected_lag
    """

    audio = np.asarray(
        audio,
        dtype=np.float64
    )

    audio = audio - np.mean(
        audio
    )

    audio = audio * np.hanning(
        len(audio)
    )

    autocorrelation = scipy_signal.correlate(
        audio,
        audio,
        mode="full",
        method="fft"
    )

    autocorrelation = autocorrelation[
        len(autocorrelation) // 2:
    ]

    if autocorrelation[0] <= EPSILON:

        return np.nan, autocorrelation

    autocorrelation /= (
        autocorrelation[0]
    )

    min_lag = int(
        fs / max_f0
    )

    max_lag = int(
        fs / min_f0
    )

    max_lag = min(
        max_lag,
        len(autocorrelation) - 1
    )

    if min_lag >= max_lag:

        return np.nan, autocorrelation

    search_region = autocorrelation[
        min_lag:max_lag + 1
    ]

    peaks, _ = find_peaks(
        search_region
    )

    if len(peaks) == 0:

        return np.nan, autocorrelation

    best_relative_peak = peaks[
        np.argmax(
            search_region[peaks]
        )
    ]

    best_lag = (
        min_lag + best_relative_peak
    )

    f0 = fs / best_lag

    return f0, autocorrelation


# =====================================================================
# SIMPLE FORMANT-RELATED PEAK VISUALIZATION
# =====================================================================

def detect_spectral_peaks(
    audio,
    fs,
    max_frequency=5000
):
    """
    Detect prominent spectral peaks.

    IMPORTANT:

    These are spectral peaks, not reliable formant estimates.

    Harmonics can produce spectral peaks near or around formant
    resonances.

    Accurate formant estimation requires more careful methods,
    such as LPC with appropriate preprocessing and validation.
    """

    frequencies, magnitude = calculate_spectrum(
        audio,
        fs
    )

    valid = frequencies <= max_frequency

    frequencies = frequencies[valid]

    magnitude = magnitude[valid]

    peaks, properties = find_peaks(
        magnitude,
        prominence=np.max(magnitude) * 0.01,
        distance=3
    )

    sorted_indices = np.argsort(
        magnitude[peaks]
    )[::-1]

    return (
        frequencies,
        magnitude,
        peaks[sorted_indices]
    )


# =====================================================================
# EXPERIMENT 1
# VOICED EXCITATION
# =====================================================================

def experiment_voiced():

    print("\nEXPERIMENT 1: VOICED EXCITATION")

    f0 = 150

    audio = generate_voiced_excitation(
        FS,
        DURATION,
        f0
    )

    time = np.arange(
        len(audio)
    ) / FS

    frequencies, magnitude = calculate_spectrum(
        audio,
        FS
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9)
    )

    axes[0].plot(
        time[:1000],
        audio[:1000]
    )

    axes[0].set_title(
        f"Voiced Excitation: F0 = {f0} Hz"
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
        20 * np.log10(magnitude + EPSILON)
    )

    axes[1].set_xlim(
        0,
        4000
    )

    axes[1].set_title(
        "Harmonic Spectrum"
    )

    axes[1].set_xlabel(
        "Frequency (Hz)"
    )

    axes[1].set_ylabel(
        "Magnitude (dB)"
    )

    axes[1].grid(True)

    plt.tight_layout()

    plt.show()

    print(
        f"Expected fundamental frequency: {f0} Hz"
    )


# =====================================================================
# EXPERIMENT 2
# UNVOICED EXCITATION
# =====================================================================

def experiment_unvoiced():

    print("\nEXPERIMENT 2: UNVOICED EXCITATION")

    audio = generate_unvoiced_excitation(
        FS,
        DURATION
    )

    time = np.arange(
        len(audio)
    ) / FS

    frequencies, magnitude = calculate_spectrum(
        audio,
        FS
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9)
    )

    axes[0].plot(
        time[:1000],
        audio[:1000]
    )

    axes[0].set_title(
        "Noise-Like Unvoiced Excitation"
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
        20 * np.log10(magnitude + EPSILON)
    )

    axes[1].set_xlim(
        0,
        10000
    )

    axes[1].set_title(
        "Unvoiced Excitation Spectrum"
    )

    axes[1].set_xlabel(
        "Frequency (Hz)"
    )

    axes[1].set_ylabel(
        "Magnitude (dB)"
    )

    axes[1].grid(True)

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 3
# SOURCE-FILTER MODEL
# =====================================================================

def experiment_source_filter():

    print("\nEXPERIMENT 3: SOURCE-FILTER MODEL")

    fs = FS

    duration = 1.0

    f0 = 120

    excitation = generate_voiced_excitation(
        fs,
        duration,
        f0,
        harmonics=60
    )

    # Illustrative formant frequencies.
    #
    # These are chosen for demonstration, not as universal
    # values for any particular speaker or vowel.

    formants = [
        (700, 100),
        (1200, 120),
        (2600, 180)
    ]

    filtered = apply_vocal_tract_filter(
        excitation,
        formants,
        fs
    )

    frequencies, source_magnitude = calculate_spectrum(
        excitation,
        fs
    )

    _, filtered_magnitude = calculate_spectrum(
        filtered,
        fs
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 13)
    )

    time = np.arange(
        len(excitation)
    ) / fs

    axes[0].plot(
        time[:1000],
        excitation[:1000]
    )

    axes[0].set_title(
        "Excitation Source"
    )

    axes[0].set_xlabel(
        "Time (seconds)"
    )

    axes[0].grid(True)

    axes[1].plot(
        frequencies,
        20 * np.log10(
            source_magnitude + EPSILON
        ),
        label="Source"
    )

    axes[1].plot(
        frequencies,
        20 * np.log10(
            filtered_magnitude + EPSILON
        ),
        label="After vocal-tract filter"
    )

    axes[1].set_xlim(
        0,
        5000
    )

    axes[1].set_title(
        "Source and Filtered Spectrum"
    )

    axes[1].set_xlabel(
        "Frequency (Hz)"
    )

    axes[1].set_ylabel(
        "Magnitude (dB)"
    )

    axes[1].legend()

    axes[1].grid(True)

    axes[2].plot(
        time[:1000],
        filtered[:1000]
    )

    axes[2].set_title(
        "Filtered Speech-Like Signal"
    )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    axes[2].grid(True)

    plt.tight_layout()

    plt.show()

    print("Configured resonances:")

    for index, (frequency, bandwidth) in enumerate(
        formants,
        start=1
    ):

        print(
            f"F{index}: {frequency} Hz, "
            f"bandwidth={bandwidth} Hz"
        )


# =====================================================================
# EXPERIMENT 4
# F0 VS FORMANTS
# =====================================================================

def experiment_f0_vs_formants():

    print("\nEXPERIMENT 4: F0 VS FORMANTS")

    fs = FS

    duration = 1.0

    f0_values = [
        100,
        150,
        220
    ]

    formants = [
        (700, 100),
        (1200, 120),
        (2600, 180)
    ]

    plt.figure(figsize=(14, 7))

    for f0 in f0_values:

        excitation = generate_voiced_excitation(
            fs,
            duration,
            f0,
            harmonics=70
        )

        filtered = apply_vocal_tract_filter(
            excitation,
            formants,
            fs
        )

        frequencies, magnitude = calculate_spectrum(
            filtered,
            fs
        )

        plt.plot(
            frequencies,
            20 * np.log10(magnitude + EPSILON),
            label=f"F0 = {f0} Hz"
        )

    plt.xlim(
        0,
        5000
    )

    plt.ylim(
        -80,
        5
    )

    plt.title(
        "Changing F0 While Keeping the Vocal-Tract Filter Fixed"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Magnitude (dB)"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    print(
        "The excitation harmonics change with F0."
    )

    print(
        "The configured vocal-tract resonances remain fixed."
    )

    print(
        "This demonstrates why pitch and formants are different."
    )


# =====================================================================
# EXPERIMENT 5
# AUTOCORRELATION F0 ESTIMATION
# =====================================================================

def experiment_f0_estimation():

    print("\nEXPERIMENT 5: F0 ESTIMATION")

    expected_f0 = 150

    audio = generate_voiced_excitation(
        FS,
        0.1,
        expected_f0
    )

    estimated_f0, autocorrelation = (
        estimate_f0_autocorrelation(
            audio,
            FS
        )
    )

    print(
        f"Expected F0: {expected_f0} Hz"
    )

    print(
        f"Estimated F0: {estimated_f0:.2f} Hz"
    )

    lags = np.arange(
        len(autocorrelation)
    )

    lag_time = lags / FS

    plt.figure(figsize=(14, 6))

    plt.plot(
        lag_time,
        autocorrelation
    )

    plt.xlim(
        0,
        0.03
    )

    plt.xlabel(
        "Lag (seconds)"
    )

    plt.ylabel(
        "Normalized autocorrelation"
    )

    plt.title(
        "Autocorrelation-Based Pitch Estimation"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# =====================================================================
# EXPERIMENT 6
# REAL AUDIO ANALYSIS
# =====================================================================

def experiment_real_audio(filename):

    print("\nEXPERIMENT 6: REAL AUDIO ANALYSIS")

    audio, fs = sf.read(
        filename,
        dtype="float32",
        always_2d=True
    )

    audio = np.mean(
        audio,
        axis=1
    )

    if len(audio) < int(0.04 * fs):

        raise ValueError(
            "Audio is too short for this analysis."
        )

    audio = audio - np.mean(
        audio
    )

    # Analyze the center portion of the recording.
    # The user should ideally provide a sustained vowel recording.

    segment_length = min(
        len(audio),
        int(0.08 * fs)
    )

    start = max(
        0,
        len(audio) // 2 - segment_length // 2
    )

    segment = audio[
        start:start + segment_length
    ]

    estimated_f0, autocorrelation = (
        estimate_f0_autocorrelation(
            segment,
            fs
        )
    )

    frequencies, magnitude, peaks = (
        detect_spectral_peaks(
            segment,
            fs
        )
    )

    print(
        f"Sampling rate: {fs} Hz"
    )

    print(
        f"Duration: {len(audio)/fs:.2f} seconds"
    )

    print(
        f"Estimated F0: {estimated_f0:.2f} Hz"
        if np.isfinite(estimated_f0)
        else "Estimated F0: unavailable"
    )

    print(
        "\nStrong spectral peaks "
        "(not guaranteed to be formants):"
    )

    for peak in peaks[:15]:

        print(
            f"{frequencies[peak]:.1f} Hz"
        )

    time = np.arange(
        len(audio)
    ) / fs

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10)
    )

    axes[0].plot(
        time,
        audio
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

    axes[0].grid(True)

    axes[1].plot(
        frequencies,
        20 * np.log10(
            magnitude + EPSILON
        )
    )

    axes[1].set_xlim(
        0,
        min(8000, fs / 2)
    )

    axes[1].set_title(
        "Speech Spectrum: Spectral Peaks"
    )

    axes[1].set_xlabel(
        "Frequency (Hz)"
    )

    axes[1].set_ylabel(
        "Magnitude (dB)"
    )

    axes[1].grid(True)

    plt.tight_layout()

    plt.show()


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 Lesson 6: Human Speech Production."
        )
    )

    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        choices=range(7),
        help=(
            "0 = All synthetic experiments, "
            "1 = Voiced excitation, "
            "2 = Unvoiced excitation, "
            "3 = Source-filter model, "
            "4 = F0 vs formants, "
            "5 = F0 estimation, "
            "6 = Real audio"
        )
    )

    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to a WAV recording."
    )

    args = parser.parse_args()

    experiments = {
        1: experiment_voiced,
        2: experiment_unvoiced,
        3: experiment_source_filter,
        4: experiment_f0_vs_formants,
        5: experiment_f0_estimation
    }

    print("\n")
    print("=" * 75)
    print("PHASE 1 — LESSON 6")
    print("HUMAN SPEECH PRODUCTION AND SOURCE-FILTER MODEL")
    print("=" * 75)

    if args.experiment == 0:

        for experiment in experiments.values():

            experiment()

        if args.audio:

            experiment_real_audio(
                args.audio
            )

    elif args.experiment == 6:

        if not args.audio:

            raise ValueError(
                "Please provide --audio your_voice.wav"
            )

        experiment_real_audio(
            args.audio
        )

    else:

        experiments[args.experiment]()

    print("\n")
    print("=" * 75)
    print("LESSON 6 COMPLETED")
    print("=" * 75)


if __name__ == "__main__":

    main()