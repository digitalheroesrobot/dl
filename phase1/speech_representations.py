"""
=======================================================================
FILE: phase1_lesson10_speech_representations.py

PHASE 1 — LESSON 10

SPEECH REPRESENTATIONS FOR DEEP LEARNING

FROM RAW WAVEFORM TO LOG-MEL SPECTROGRAM
=======================================================================

PURPOSE
-------

Build a complete educational speech representation pipeline.

The program converts digital speech into representations suitable
for neural-network processing.

IMPLEMENTED:

    1. Audio loading.
    2. Audio normalization.
    3. Resampling.
    4. Framing.
    5. STFT.
    6. Magnitude spectrogram.
    7. Power spectrogram.
    8. Mel filterbank.
    9. Mel spectrogram.
    10. Log-Mel spectrogram.
    11. Frequency masking.
    12. Time masking.
    13. PyTorch Dataset.
    14. PyTorch DataLoader.
    15. Educational CNN speech encoder.
    16. Feature visualization.
    17. Representation comparison.

IMPORTANT:

This is a speech representation learning lesson.

It does not yet implement a complete TTS acoustic model.

The CNN encoder is an educational feature-processing example.
It is not a pretrained speech recognition model.

=======================================================================
PART 1 — WHY SPEECH REPRESENTATION IS NECESSARY
=======================================================================

A digital waveform is:

    x[n]

A neural network can theoretically process this waveform directly.

However, a long waveform contains many samples.

For example:

    24,000 samples/second
    10 seconds

    Total samples = 240,000

A spectrogram converts the waveform into a time-frequency matrix.

This provides a more structured representation of speech.

The neural network can then learn patterns involving:

    - Pitch-related harmonic structures.
    - Formants.
    - Consonant transitions.
    - Vowel structure.
    - Energy variation.
    - Speech timing.
    - Acoustic texture.

=======================================================================
PART 2 — STFT
=======================================================================

STFT:

    X(m,k) = SUM x[n+mH]w[n]exp(-j2*pi*k*n/N)

Where:

    N = FFT size
    H = hop length
    w = window function

The output is complex-valued.

Magnitude:

    M = abs(X)

Power:

    P = abs(X)^2

=======================================================================
PART 3 — MEL FILTERBANK
=======================================================================

Human frequency perception is nonlinear.

The Mel scale approximates perceptual frequency spacing.

Mel conversion:

    mel = 2595 * log10(1 + frequency / 700)

Inverse:

    frequency = 700 * (10^(mel/2595) - 1)

The Mel filterbank is composed of triangular filters.

Each filter integrates energy over a region of frequency bins.

Mel spectrogram:

    Mel = PowerSpectrogram @ MelFilterbank.T

=======================================================================
PART 4 — LOG-MEL
=======================================================================

Log-Mel:

    LogMel = log(max(Mel, epsilon))

Why logarithms?

    - Compress dynamic range.
    - Represent low-energy information.
    - Improve numerical behavior.
    - Produce features commonly used by neural acoustic models.

Different TTS architectures use different acoustic targets.

Examples include:

    - Mel spectrograms.
    - Linear spectrograms.
    - Codec tokens.
    - Learned acoustic representations.
    - Discrete speech tokens.

Log-Mel is an important representation, not a universal requirement.

=======================================================================
PART 5 — FEATURE NORMALIZATION
=======================================================================

Neural networks often benefit from feature normalization.

For a feature matrix X:

    X_normalized = (X - mean) / standard_deviation

The mean and standard deviation should be estimated from training
data only.

Do not calculate normalization statistics using validation or test
data if you want a strict evaluation protocol.

=======================================================================
PART 6 — TIME AND FREQUENCY MASKING
=======================================================================

SpecAugment-style masking randomly hides parts of spectrograms.

Frequency masking:

    Hide selected frequency bins.

Time masking:

    Hide selected time frames.

These techniques can improve robustness for certain speech models.

However, masking is not automatically appropriate for every TTS
training target.

For TTS, careless masking can remove important acoustic information.

=======================================================================
PART 7 — TTS CONNECTION
=======================================================================

A common TTS training pair is:

    Text tokens
        |
        v
    Text encoder
        |
        v
    Acoustic decoder
        |
        v
    Predicted Mel spectrogram

The ground-truth Mel spectrogram is extracted from the actual
training recording.

A loss function compares:

    Predicted Mel vs Ground-truth Mel

For example:

    L = mean(abs(predicted_mel - target_mel))

A neural vocoder subsequently converts acoustic features into audio.

=======================================================================
USAGE
=======================================================================

Run synthetic demonstration:

    python phase1_lesson10_speech_representations.py

Process a real recording:

    python phase1_lesson10_speech_representations.py --audio my_voice.wav

Run the neural encoder demonstration:

    python phase1_lesson10_speech_representations.py --encoder

Run without visualization:

    python phase1_lesson10_speech_representations.py --no-plot

=======================================================================
"""

import argparse

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from scipy.signal import resample_poly

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# =====================================================================
# SECTION 1 — CONFIGURATION
# =====================================================================

SAMPLE_RATE = 24000

FFT_SIZE = 1024

WINDOW_SIZE = 1024

HOP_LENGTH = 256

NUM_MELS = 80

MIN_MEL_FREQ = 0.0

MAX_MEL_FREQ = 12000.0

EPSILON = 1e-8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# =====================================================================
# SECTION 2 — AUDIO LOADING
# =====================================================================

def load_audio(audio_path):

    """
    Load WAV audio.

    Converts stereo audio to mono by averaging channels.

    Returns:

        audio: float32 NumPy array
        sample_rate: integer

    IMPORTANT:

    Averaging stereo channels can cause cancellation if the channels
    contain phase-inverted signals.

    For critical datasets, inspect stereo channels before conversion.
    """

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True
    )

    audio = np.mean(
        audio,
        axis=1
    )

    if len(audio) == 0:

        raise ValueError(
            "Audio file contains no samples."
        )

    return audio, sample_rate


# =====================================================================
# SECTION 3 — AUDIO RESAMPLING
# =====================================================================

def resample_audio(
    audio,
    original_rate,
    target_rate
):

    """
    Resample audio using polyphase filtering.

    Resampling is not simply changing the sample-rate metadata.

    We must actually transform the samples.

    Example:

        44100 Hz -> 24000 Hz

    The conversion ratio is:

        24000 / 44100 = 80 / 147

    scipy.signal.resample_poly performs polyphase resampling.

    This includes filtering to reduce aliasing.
    """

    if original_rate == target_rate:

        return audio

    from math import gcd

    divisor = gcd(
        original_rate,
        target_rate
    )

    up = target_rate // divisor

    down = original_rate // divisor

    resampled = resample_poly(
        audio,
        up,
        down
    )

    return resampled.astype(
        np.float32
    )


# =====================================================================
# SECTION 4 — NORMALIZATION
# =====================================================================

def normalize_waveform(audio):

    """
    Peak normalization.

    x_normalized = x / max(abs(x))

    This is useful for demonstration.

    IMPORTANT:

    Peak normalization changes recording amplitude.

    In a real TTS dataset, consider whether preserving natural
    loudness differences is important.

    Avoid independently normalizing every recording without
    considering the training architecture.
    """

    peak = np.max(
        np.abs(audio)
    )

    if peak < EPSILON:

        raise ValueError(
            "Cannot normalize silent audio."
        )

    return (
        audio / peak
    ).astype(np.float32)


# =====================================================================
# SECTION 5 — SYNTHETIC SPEECH-LIKE SIGNAL
# =====================================================================

def generate_synthetic_speech():

    """
    Generate an educational speech-like signal.

    This is NOT natural human speech.

    It combines:

        - Harmonic excitation.
        - Time-varying amplitude.
        - Several resonant components.
        - A noise component.

    This helps visualize how time-varying spectral structures
    appear in a spectrogram.

    Natural speech has much more complex articulation and dynamics.
    """

    sr = SAMPLE_RATE

    duration = 3.0

    t = np.arange(
        int(sr * duration)
    ) / sr

    # ---------------------------------------------------------------
    # Fundamental frequency
    # ---------------------------------------------------------------

    f0 = (

        130

        + 20 * np.sin(
            2 * np.pi * 1.5 * t
        )

    )

    phase = 2 * np.pi * np.cumsum(
        f0
    ) / sr

    # ---------------------------------------------------------------
    # Harmonic source
    # ---------------------------------------------------------------

    source = np.zeros_like(t)

    for harmonic in range(1, 30):

        source += (

            np.sin(
                harmonic * phase
            )

            / harmonic

        )

    # ---------------------------------------------------------------
    # Time-varying amplitude
    # ---------------------------------------------------------------

    envelope = (

        0.5

        + 0.4 * np.sin(
            2 * np.pi * 0.5 * t
        )

    )

    envelope = np.maximum(
        envelope,
        0
    )

    # ---------------------------------------------------------------
    # Approximate time-varying formant-like components
    # ---------------------------------------------------------------

    formant_1 = np.sin(
        2 * np.pi * 700 * t
    )

    formant_2 = np.sin(
        2 * np.pi * 1200 * t
    )

    formant_3 = np.sin(
        2 * np.pi * 2500 * t
    )

    resonant_signal = (

        0.5 * formant_1

        + 0.3 * formant_2

        + 0.2 * formant_3

    )

    # ---------------------------------------------------------------
    # Add noise
    # ---------------------------------------------------------------

    noise = np.random.normal(
        0,
        0.02,
        len(t)
    )

    signal = (

        0.5 * source

        + 0.2 * resonant_signal

        + noise

    ) * envelope

    signal = normalize_waveform(
        signal
    )

    return signal.astype(
        np.float32
    ), sr


# =====================================================================
# SECTION 6 — FRAMING
# =====================================================================

def frame_signal(
    audio,
    frame_length,
    hop_length
):

    """
    Divide audio into overlapping frames.

    For example:

        Frame length = 1024
        Hop length = 256

    Each new frame starts 256 samples after the previous frame.

    Overlap:

        1024 - 256 = 768 samples

    This is 75% overlap.

    The number of complete frames is:

        1 + floor((N - frame_length) / hop_length)

    This implementation pads the final frame when necessary.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) < frame_length:

        audio = np.pad(
            audio,
            (0, frame_length - len(audio))
        )

    num_frames = int(
        np.ceil(
            (len(audio) - frame_length)
            / hop_length
        )
    ) + 1

    total_length = (

        (num_frames - 1) * hop_length

        + frame_length

    )

    padding = max(
        0,
        total_length - len(audio)
    )

    audio = np.pad(
        audio,
        (0, padding)
    )

    frames = np.zeros(

        (
            num_frames,
            frame_length
        ),

        dtype=np.float32

    )

    for i in range(num_frames):

        start = i * hop_length

        end = start + frame_length

        frames[i] = audio[
            start:end
        ]

    return frames


# =====================================================================
# SECTION 7 — STFT
# =====================================================================

def compute_stft(audio):

    """
    Compute STFT using NumPy.

    The analysis window is Hann.

    For each frame:

        X[k] = FFT(frame * Hann_window)

    We use rFFT because real-valued audio has conjugate symmetry.

    Output shape:

        (number_of_frames, FFT_SIZE // 2 + 1)

    For FFT_SIZE=1024:

        513 frequency bins.
    """

    frames = frame_signal(

        audio,

        WINDOW_SIZE,

        HOP_LENGTH

    )

    window = np.hanning(
        WINDOW_SIZE
    ).astype(np.float32)

    windowed_frames = (
        frames * window[None, :]
    )

    spectrum = np.fft.rfft(

        windowed_frames,

        n=FFT_SIZE,

        axis=1

    )

    return spectrum


# =====================================================================
# SECTION 8 — MAGNITUDE AND POWER SPECTROGRAM
# =====================================================================

def compute_spectrograms(stft):

    """
    Compute magnitude and power spectrograms.

    Magnitude:

        magnitude = abs(STFT)

    Power:

        power = magnitude ** 2
    """

    magnitude = np.abs(
        stft
    )

    power = magnitude ** 2

    return magnitude, power


# =====================================================================
# SECTION 9 — MEL CONVERSION
# =====================================================================

def hz_to_mel(frequency):

    """
    Convert frequency in Hz to Mel.

        mel = 2595 * log10(1 + hz / 700)
    """

    return (
        2595.0
        * np.log10(
            1.0 + frequency / 700.0
        )
    )


def mel_to_hz(mel):

    """
    Convert Mel to frequency in Hz.

        hz = 700 * (10 ** (mel / 2595) - 1)
    """

    return (

        700.0

        * (

            10.0 ** (mel / 2595.0)

            - 1.0

        )

    )


# =====================================================================
# SECTION 10 — MEL FILTERBANK
# =====================================================================

def create_mel_filterbank(

    sample_rate=SAMPLE_RATE,

    n_fft=FFT_SIZE,

    n_mels=NUM_MELS,

    fmin=MIN_MEL_FREQ,

    fmax=MAX_MEL_FREQ

):

    """
    Construct triangular Mel filters.

    Output:

        Shape = (n_mels, n_fft // 2 + 1)

    Each row corresponds to one Mel filter.

    Each column corresponds to one FFT frequency bin.

    A triangular filter has:

        1. Rising slope.
        2. Peak.
        3. Falling slope.

    IMPORTANT:

    Different libraries use slightly different Mel filterbank
    conventions, including:

        - HTK Mel.
        - Slaney Mel.
        - Different normalization methods.

    This implementation uses the HTK Mel conversion and
    area-normalized triangular filters.
    """

    num_frequency_bins = (
        n_fft // 2 + 1
    )

    mel_min = hz_to_mel(
        fmin
    )

    mel_max = hz_to_mel(
        fmax
    )

    mel_points = np.linspace(

        mel_min,

        mel_max,

        n_mels + 2

    )

    hz_points = mel_to_hz(
        mel_points
    )

    fft_frequencies = np.linspace(

        0,

        sample_rate / 2,

        num_frequency_bins

    )

    filterbank = np.zeros(

        (
            n_mels,
            num_frequency_bins
        ),

        dtype=np.float32

    )

    for m in range(n_mels):

        left = hz_points[m]

        center = hz_points[m + 1]

        right = hz_points[m + 2]

        # -----------------------------------------------------------
        # Rising triangular slope
        # -----------------------------------------------------------

        rising = (

            (fft_frequencies - left)

            / max(center - left, EPSILON)

        )

        # -----------------------------------------------------------
        # Falling triangular slope
        # -----------------------------------------------------------

        falling = (

            (right - fft_frequencies)

            / max(right - center, EPSILON)

        )

        filter_values = np.maximum(

            0.0,

            np.minimum(
                rising,
                falling
            )

        )

        # -----------------------------------------------------------
        # Area normalization
        # -----------------------------------------------------------

        filter_values /= max(
            right - left,
            EPSILON
        )

        filterbank[m] = filter_values

    return filterbank


# =====================================================================
# SECTION 11 — MEL SPECTROGRAM
# =====================================================================

def compute_mel_spectrogram(power):

    """
    Project power spectrogram onto Mel filters.

        Mel = Power @ Filterbank.T

    Input:

        (T, F)

    Output:

        (T, NUM_MELS)
    """

    filterbank = create_mel_filterbank()

    mel = power @ filterbank.T

    return mel, filterbank


# =====================================================================
# SECTION 12 — LOG-MEL
# =====================================================================

def compute_log_mel(mel):

    """
    Logarithmic Mel transformation.

        LogMel = log(max(Mel, epsilon))

    This avoids log(0).
    """

    return np.log(
        np.maximum(
            mel,
            EPSILON
        )
    )


# =====================================================================
# SECTION 13 — FEATURE NORMALIZATION
# =====================================================================

def normalize_features(

    features,

    mean=None,

    std=None

):

    """
    Standardize feature values.

        X_normalized = (X - mean) / std

    If mean and std are not provided, they are calculated from
    the supplied feature array.

    For model training, calculate these statistics from the
    training dataset only and reuse them for validation and testing.
    """

    if mean is None:

        mean = np.mean(
            features
        )

    if std is None:

        std = np.std(
            features
        )

    std = max(
        float(std),
        EPSILON
    )

    normalized = (
        features - mean
    ) / std

    return normalized, mean, std


# =====================================================================
# SECTION 14 — SPEC AUGMENT
# =====================================================================

def frequency_mask(

    features,

    max_width=10

):

    """
    Randomly mask consecutive Mel frequency channels.

    Input shape:

        (T, F)

    This is a simple SpecAugment-style operation.
    """

    result = features.copy()

    T, F = result.shape

    width = np.random.randint(
        0,
        min(max_width, F) + 1
    )

    if width == 0:

        return result

    start = np.random.randint(
        0,
        F - width + 1
    )

    result[:, start:start + width] = 0

    return result


def time_mask(

    features,

    max_width=30

):

    """
    Randomly mask consecutive time frames.

    Input shape:

        (T, F)
    """

    result = features.copy()

    T, F = result.shape

    width = np.random.randint(
        0,
        min(max_width, T) + 1
    )

    if width == 0:

        return result

    start = np.random.randint(
        0,
        T - width + 1
    )

    result[start:start + width, :] = 0

    return result


# =====================================================================
# SECTION 15 — COMPLETE FEATURE EXTRACTION
# =====================================================================

def extract_speech_features(audio):

    """
    Execute the complete waveform-to-feature pipeline.

    Output dictionary:

        waveform
        stft
        magnitude
        power
        mel
        log_mel
        normalized_log_mel
        filterbank
    """

    stft = compute_stft(
        audio
    )

    magnitude, power = compute_spectrograms(
        stft
    )

    mel, filterbank = compute_mel_spectrogram(
        power
    )

    log_mel = compute_log_mel(
        mel
    )

    normalized_log_mel, mean, std = normalize_features(
        log_mel
    )

    return {

        "waveform": audio,

        "stft": stft,

        "magnitude": magnitude,

        "power": power,

        "mel": mel,

        "log_mel": log_mel,

        "normalized_log_mel": normalized_log_mel,

        "filterbank": filterbank,

        "normalization_mean": mean,

        "normalization_std": std

    }


# =====================================================================
# SECTION 16 — PYTORCH DATASET
# =====================================================================

class SpeechFeatureDataset(Dataset):

    """
    Simple PyTorch dataset for speech feature tensors.

    This demonstration uses one extracted recording.

    A production implementation should load many utterances
    from the dataset metadata generated in Lesson 8.

    The dataset returns:

        feature tensor: (1, T, NUM_MELS)

    The leading dimension is the channel dimension used by CNNs.
    """

    def __init__(self, feature_array):

        self.features = torch.tensor(

            feature_array,

            dtype=torch.float32

        )

    def __len__(self):

        return 1

    def __getitem__(self, index):

        feature = self.features

        # (T, F) -> (1, T, F)

        feature = feature.unsqueeze(0)

        return feature


# =====================================================================
# SECTION 17 — COLLATE FUNCTION
# =====================================================================

def speech_collate_fn(batch):

    """
    Combine variable-length speech features.

    Different utterances have different durations.

    Therefore, they produce different numbers of frames.

    A production DataLoader must pad sequences to a common length
    and usually return their original lengths or masks.

    This educational collate function pads along the time dimension.

    Input:

        List of tensors:

            (1, T_i, F)

    Output:

        (B, 1, T_max, F)

        lengths: original frame counts
    """

    lengths = torch.tensor(

        [
            item.shape[1]
            for item in batch
        ],

        dtype=torch.long

    )

    max_length = int(
        lengths.max()
    )

    num_mels = batch[0].shape[2]

    batch_size = len(batch)

    padded = torch.zeros(

        (
            batch_size,
            1,
            max_length,
            num_mels
        ),

        dtype=torch.float32

    )

    for index, item in enumerate(batch):

        current_length = item.shape[1]

        padded[
            index,
            :,
            :current_length,
            :
        ] = item

    return padded, lengths


# =====================================================================
# SECTION 18 — EDUCATIONAL CNN SPEECH ENCODER
# =====================================================================

class SpeechCNNEncoder(nn.Module):

    """
    A small CNN that converts Mel features into learned
    frame-level representations.

    Input:

        (B, 1, T, F)

    Output:

        (B, T, D)

    This is a feature encoder demonstration.

    It is NOT a complete TTS text encoder or acoustic decoder.

    The CNN learns local time-frequency patterns.

    Convolutional filters can learn patterns associated with:

        - Harmonic structures.
        - Formant-like structures.
        - Spectral transitions.
        - Local acoustic textures.
    """

    def __init__(

        self,

        num_mels=NUM_MELS,

        hidden_dim=128

    ):

        super().__init__()

        self.conv1 = nn.Conv2d(

            1,

            32,

            kernel_size=3,

            padding=1

        )

        self.bn1 = nn.BatchNorm2d(
            32
        )

        self.conv2 = nn.Conv2d(

            32,

            64,

            kernel_size=3,

            padding=1

        )

        self.bn2 = nn.BatchNorm2d(
            64
        )

        self.activation = nn.GELU()

        # -----------------------------------------------------------
        # Frequency pooling.
        #
        # Preserve time resolution while reducing frequency dimension.
        # -----------------------------------------------------------

        self.frequency_pool = nn.AdaptiveAvgPool2d(

            (
                None,
                8
            )

        )

        self.projection = nn.Linear(

            64 * 8,

            hidden_dim

        )

    def forward(self, x):

        # Input:
        # (B, 1, T, F)

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.activation(x)

        x = self.conv2(x)

        x = self.bn2(x)

        x = self.activation(x)

        # (B, 64, T, 8)

        x = self.frequency_pool(x)

        # (B, T, 64, 8)

        x = x.permute(
            0,
            2,
            1,
            3
        )

        B, T, C, F = x.shape

        x = x.reshape(
            B,
            T,
            C * F
        )

        x = self.projection(x)

        # (B, T, hidden_dim)

        return x


# =====================================================================
# SECTION 19 — ENCODER EXPERIMENT
# =====================================================================

def run_encoder_experiment(features):

    """
    Feed log-Mel features through the educational CNN.

    Demonstrates how speech features become neural representations.
    """

    print("\n================================================")

    print("PYTORCH SPEECH ENCODER EXPERIMENT")

    print("================================================")

    dataset = SpeechFeatureDataset(
        features
    )

    loader = DataLoader(

        dataset,

        batch_size=1,

        collate_fn=speech_collate_fn

    )

    model = SpeechCNNEncoder().to(
        DEVICE
    )

    model.eval()

    with torch.no_grad():

        for batch, lengths in loader:

            batch = batch.to(
                DEVICE
            )

            output = model(
                batch
            )

            print(
                "Input tensor shape:",
                tuple(batch.shape)
            )

            print(
                "Input frame count:",
                lengths.tolist()
            )

            print(
                "Encoder output shape:",
                tuple(output.shape)
            )

            print(
                "Device:",
                DEVICE
            )

            print(
                "\nEncoder output statistics:"
            )

            print(
                "Mean:",
                output.mean().item()
            )

            print(
                "Standard deviation:",
                output.std().item()
            )

    print(
        "\nThe CNN produced a learned representation for each frame."
    )

    print(
        "No speech recognition or TTS training occurred."
    )


# =====================================================================
# SECTION 20 — VISUALIZATION
# =====================================================================

def visualize_features(

    audio,

    sample_rate,

    features

):

    """
    Visualize all major speech representations.

    Figures:

        1. Raw waveform.
        2. Magnitude spectrogram.
        3. Power spectrogram.
        4. Mel spectrogram.
        5. Log-Mel spectrogram.
        6. Normalized Log-Mel spectrogram.
        7. Mel filterbank.
    """

    waveform = features["waveform"]

    magnitude = features["magnitude"]

    power = features["power"]

    mel = features["mel"]

    log_mel = features["log_mel"]

    normalized = features["normalized_log_mel"]

    filterbank = features["filterbank"]

    # ---------------------------------------------------------------
    # Time axis
    # ---------------------------------------------------------------

    time_waveform = np.arange(
        len(waveform)
    ) / sample_rate

    time_frames = np.arange(
        magnitude.shape[0]
    ) * HOP_LENGTH / sample_rate

    frequency_axis = np.linspace(

        0,

        sample_rate / 2,

        magnitude.shape[1]

    )

    # ---------------------------------------------------------------
    # 1. Waveform
    # ---------------------------------------------------------------

    plt.figure(figsize=(13, 4))

    plt.plot(
        time_waveform,
        waveform
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Raw Speech Waveform"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    # ---------------------------------------------------------------
    # 2. Magnitude spectrogram
    # ---------------------------------------------------------------

    plt.figure(figsize=(13, 5))

    plt.imshow(

        20 * np.log10(
            magnitude.T + EPSILON
        ),

        aspect="auto",

        origin="lower",

        extent=[

            time_frames[0],

            time_frames[-1],

            frequency_axis[0],

            frequency_axis[-1]

        ]

    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.title(
        "Magnitude Spectrogram (dB)"
    )

    plt.colorbar(
        label="Magnitude (dB)"
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # 3. Power spectrogram
    # ---------------------------------------------------------------

    plt.figure(figsize=(13, 5))

    plt.imshow(

        10 * np.log10(
            power.T + EPSILON
        ),

        aspect="auto",

        origin="lower",

        extent=[

            time_frames[0],

            time_frames[-1],

            frequency_axis[0],

            frequency_axis[-1]

        ]

    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.title(
        "Power Spectrogram (dB)"
    )

    plt.colorbar(
        label="Power (dB)"
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # 4. Mel spectrogram
    # ---------------------------------------------------------------

    plt.figure(figsize=(13, 5))

    plt.imshow(

        mel.T,

        aspect="auto",

        origin="lower"

    )

    plt.xlabel(
        "Time frame"
    )

    plt.ylabel(
        "Mel channel"
    )

    plt.title(
        "Mel Spectrogram"
    )

    plt.colorbar(
        label="Mel energy"
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # 5. Log-Mel spectrogram
    # ---------------------------------------------------------------

    plt.figure(figsize=(13, 5))

    plt.imshow(

        log_mel.T,

        aspect="auto",

        origin="lower"

    )

    plt.xlabel(
        "Time frame"
    )

    plt.ylabel(
        "Mel channel"
    )

    plt.title(
        "Log-Mel Spectrogram"
    )

    plt.colorbar(
        label="Log energy"
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # 6. Normalized Log-Mel
    # ---------------------------------------------------------------

    plt.figure(figsize=(13, 5))

    plt.imshow(

        normalized.T,

        aspect="auto",

        origin="lower",

        vmin=-3,

        vmax=3

    )

    plt.xlabel(
        "Time frame"
    )

    plt.ylabel(
        "Mel channel"
    )

    plt.title(
        "Normalized Log-Mel Spectrogram"
    )

    plt.colorbar(
        label="Normalized feature"
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # 7. Mel filterbank
    # ---------------------------------------------------------------

    plt.figure(figsize=(13, 5))

    plt.imshow(

        filterbank,

        aspect="auto",

        origin="lower"

    )

    plt.xlabel(
        "FFT frequency bin"
    )

    plt.ylabel(
        "Mel filter index"
    )

    plt.title(
        "Mel Filterbank Matrix"
    )

    plt.colorbar(
        label="Filter weight"
    )

    plt.tight_layout()

    plt.show()


# =====================================================================
# SECTION 21 — AUGMENTATION VISUALIZATION
# =====================================================================

def visualize_augmentation(features):

    """
    Compare original Log-Mel features with masked features.
    """

    original = features.copy()

    augmented = frequency_mask(
        original,
        max_width=12
    )

    augmented = time_mask(
        augmented,
        max_width=30
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8)
    )

    axes[0].imshow(

        original.T,

        aspect="auto",

        origin="lower"

    )

    axes[0].set_title(
        "Original Normalized Log-Mel"
    )

    axes[1].imshow(

        augmented.T,

        aspect="auto",

        origin="lower"

    )

    axes[1].set_title(
        "Time and Frequency Masking"
    )

    axes[0].set_ylabel(
        "Mel channel"
    )

    axes[1].set_ylabel(
        "Mel channel"
    )

    axes[1].set_xlabel(
        "Time frame"
    )

    plt.tight_layout()

    plt.show()


# =====================================================================
# SECTION 22 — MAIN PIPELINE
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Speech representations for deep learning."
        )

    )

    parser.add_argument(

        "--audio",

        type=str,

        default=None,

        help="Optional WAV recording."

    )

    parser.add_argument(

        "--encoder",

        action="store_true",

        help="Run PyTorch CNN encoder experiment."

    )

    parser.add_argument(

        "--no-plot",

        action="store_true",

        help="Disable visualizations."

    )

    parser.add_argument(

        "--augment",

        action="store_true",

        help="Show time/frequency masking."

    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load or generate audio.
    # ---------------------------------------------------------------

    if args.audio:

        audio_path = Path(
            args.audio
        )

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        audio, original_rate = load_audio(
            audio_path
        )

        print(
            "Loaded real audio:",
            audio_path
        )

    else:

        audio, original_rate = generate_synthetic_speech()

        print(
            "Generated synthetic speech-like audio."
        )

    print(
        "Original sample rate:",
        original_rate
    )

    print(
        "Original duration:",
        len(audio) / original_rate,
        "seconds"
    )

    # ---------------------------------------------------------------
    # Resample.
    # ---------------------------------------------------------------

    audio = resample_audio(

        audio,

        original_rate,

        SAMPLE_RATE

    )

    print(
        "Target sample rate:",
        SAMPLE_RATE
    )

    # ---------------------------------------------------------------
    # Normalize.
    # ---------------------------------------------------------------

    audio = normalize_waveform(
        audio
    )

    # ---------------------------------------------------------------
    # Extract speech features.
    # ---------------------------------------------------------------

    features = extract_speech_features(
        audio
    )

    print("\n==============================================")

    print("SPEECH REPRESENTATION PIPELINE")

    print("==============================================")

    print(
        "Waveform:",
        features["waveform"].shape
    )

    print(
        "STFT:",
        features["stft"].shape
    )

    print(
        "Magnitude:",
        features["magnitude"].shape
    )

    print(
        "Power:",
        features["power"].shape
    )

    print(
        "Mel:",
        features["mel"].shape
    )

    print(
        "Log-Mel:",
        features["log_mel"].shape
    )

    print(
        "Normalized Log-Mel:",
        features["normalized_log_mel"].shape
    )

    print("==============================================")

    print(
        "\nFrame duration:",
        HOP_LENGTH / SAMPLE_RATE,
        "seconds"
    )

    print(
        "Frequency resolution:",
        SAMPLE_RATE / FFT_SIZE,
        "Hz"
    )

    # ---------------------------------------------------------------
    # Encoder.
    # ---------------------------------------------------------------

    if args.encoder:

        run_encoder_experiment(
            features["normalized_log_mel"]
        )

    # ---------------------------------------------------------------
    # Augmentation.
    # ---------------------------------------------------------------

    if args.augment:

        visualize_augmentation(
            features["normalized_log_mel"]
        )

    # ---------------------------------------------------------------
    # Visualizations.
    # ---------------------------------------------------------------

    if not args.no_plot:

        visualize_features(

            audio,

            SAMPLE_RATE,

            features

        )

    print("\nProcessing completed successfully.")


# =====================================================================
# SECTION 23 — ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()


"""
=======================================================================
END OF LESSON 10
=======================================================================

WHAT YOU HAVE BUILT
-------------------

You have implemented the complete feature extraction pipeline:

    Digital waveform
         |
         v
       STFT
         |
         v
    Magnitude spectrogram
         |
         v
    Power spectrogram
         |
         v
    Mel filterbank
         |
         v
    Mel spectrogram
         |
         v
      Log-Mel
         |
         v
    Normalized features
         |
         v
    PyTorch neural encoder

IMPORTANT NEXT STEP
-------------------

We now understand how to convert speech into neural-network inputs.

The next major problem is text representation.

A TTS model must understand the linguistic content that corresponds
to the acoustic representation.

We will study:

    - Graphemes
    - Phonemes
    - IPA
    - Text normalization
    - Tokenization
    - Grapheme-to-phoneme conversion
    - Language-specific pronunciation

These components form the text frontend of a TTS system.

=======================================================================
"""