
"""
File: lesson41_repair_vq_codebook.py

========================================================================
LESSON 41 — REPAIR VQ-VAE CODEBOOK COLLAPSE
========================================================================

PURPOSE
-------
Train a more stable vector-quantized autoencoder for speech features.

The previous Lesson 36 experiment showed:

    Step 1:   Codes 256/256 | Perplexity 177.65
    Step 100: Codes   1/256 | Perplexity   1.00

This is codebook collapse: almost every encoder vector selects the same
codebook entry.

This lesson introduces:

1. K-means codebook initialization.
2. Exponential moving average (EMA) codebook updates.
3. Commitment loss.
4. Dead-code detection.
5. Dead-code replacement using current encoder vectors.
6. Codebook usage and perplexity monitoring.

ARCHITECTURE
------------

WAV audio
    |
    v
STFT
    |
    v
Mel spectrogram
    |
    v
Convolutional encoder
    |
    v
Continuous latent vectors
    |
    v
Vector quantizer (EMA)
    |
    v
Discrete token IDs
    |
    v
Convolutional decoder
    |
    v
Reconstructed mel spectrogram

IMPORTANT
---------
This is a mel-spectrogram autoencoder prototype.

It does NOT yet reconstruct raw waveform audio.
It is not a production neural audio codec.

The goal is to learn stable discrete representations and monitor
codebook health before moving to a waveform-capable codec.

INPUT
-----
phase2/dataset/*.wav

The script uses SoundFile and SciPy for audio loading/resampling.
It does not use torchaudio.load() or TorchCodec.

INSTALL
-------
pip install torch numpy matplotlib soundfile scipy

RUN
---
python lesson41_repair_vq_codebook.py

For a quick experiment:
python lesson41_repair_vq_codebook.py --epochs 100

OUTPUT
------
lesson41_output/
    best_model.pt
    final_model.pt
    codebook.npy
    training_history.json
    codebook_usage.png
    training_loss.png
    example_reconstruction.png
    train_tokens/
    val_tokens/

========================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset


# ======================================================================
# CONFIGURATION
# ======================================================================

SAMPLE_RATE = 16000

N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 320

N_MELS = 80

CHUNK_FRAMES = 128

LATENT_DIM = 64
CODEBOOK_SIZE = 256

BATCH_SIZE = 4
EPOCHS = 200

LEARNING_RATE = 2e-4
COMMITMENT_WEIGHT = 0.25

EMA_DECAY = 0.95
EMA_EPSILON = 1e-5

DEAD_CODE_THRESHOLD = 0.5
DEAD_CODE_PATIENCE = 20

KMEANS_ITERATIONS = 20

SEED = 42


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# MEL FILTERBANK
# ======================================================================

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> torch.Tensor:

    minimum_mel = hz_to_mel(0)
    maximum_mel = hz_to_mel(sample_rate / 2)

    mel_points = np.linspace(
        minimum_mel,
        maximum_mel,
        n_mels + 2,
    )

    hz_points = mel_to_hz(mel_points)

    fft_frequencies = np.linspace(
        0,
        sample_rate / 2,
        n_fft // 2 + 1,
    )

    filterbank = np.zeros(
        (n_mels, len(fft_frequencies)),
        dtype=np.float32,
    )

    for mel_index in range(n_mels):

        left = hz_points[mel_index]
        center = hz_points[mel_index + 1]
        right = hz_points[mel_index + 2]

        ascending = (
            (fft_frequencies - left) /
            max(center - left, 1e-8)
        )

        descending = (
            (right - fft_frequencies) /
            max(right - center, 1e-8)
        )

        filterbank[mel_index] = np.maximum(
            0.0,
            np.minimum(ascending, descending),
        )

    return torch.from_numpy(filterbank)


# ======================================================================
# AUDIO LOADING
# ======================================================================

def load_audio(path: Path) -> torch.Tensor:

    audio, source_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    # [T, C] -> mono [T]
    audio = audio.mean(axis=1)

    if source_rate != SAMPLE_RATE:

        divisor = math.gcd(
            int(source_rate),
            SAMPLE_RATE,
        )

        up = SAMPLE_RATE // divisor
        down = int(source_rate) // divisor

        audio = resample_poly(
            audio,
            up,
            down,
        ).astype(np.float32)

    waveform = torch.from_numpy(
        np.asarray(audio, dtype=np.float32)
    )

    waveform = torch.nan_to_num(waveform)

    peak = waveform.abs().max()

    if peak > 1e-8:
        waveform = waveform / peak

    return waveform


# ======================================================================
# MEL EXTRACTION
# ======================================================================

class MelExtractor:

    def __init__(self, device: torch.device):

        self.device = device

        self.window = torch.hann_window(
            WIN_LENGTH,
            device=device,
        )

        self.filterbank = create_mel_filterbank(
            SAMPLE_RATE,
            N_FFT,
            N_MELS,
        ).to(device)

    @torch.no_grad()
    def __call__(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:

        waveform = waveform.to(self.device)

        spectrum = torch.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            return_complex=True,
        )

        power = spectrum.abs().pow(2)

        mel = torch.matmul(
            self.filterbank,
            power,
        )

        log_mel = torch.log(
            mel.clamp_min(1e-6)
        )

        # Output: [mel_bins, time_frames]
        return log_mel


# ======================================================================
# DATASET
# ======================================================================

def find_audio_files(audio_dir: Path):

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Audio directory does not exist: {audio_dir}"
        )

    extensions = {
        ".wav",
        ".flac",
    }

    files = sorted(
        path
        for path in audio_dir.iterdir()
        if path.suffix.lower() in extensions
    )

    if len(files) < 2:
        raise RuntimeError(
            "At least two WAV/FLAC files are required "
            "for a recording-level train/validation split."
        )

    return files


def split_files(files, val_ratio=0.2):

    files = list(files)

    random.shuffle(files)

    val_count = max(
        1,
        int(round(len(files) * val_ratio)),
    )

    val_count = min(
        val_count,
        len(files) - 1,
    )

    return files[val_count:], files[:val_count]


def extract_recordings(
    files,
    extractor,
):

    recordings = []

    for path in files:

        print(f"  Processing {path.name}")

        waveform = load_audio(path)

        mel = extractor(waveform)

        recordings.append(
            {
                "name": path.stem,
                "mel": mel.cpu(),
                "duration": len(waveform) / SAMPLE_RATE,
            }
        )

        print(
            f"      Duration: "
            f"{len(waveform) / SAMPLE_RATE:.2f}s"
        )

        print(
            f"      Mel shape: {tuple(mel.shape)}"
        )

    return recordings


def calculate_normalization(train_recordings):

    all_frames = torch.cat(
        [
            recording["mel"].T
            for recording in train_recordings
        ],
        dim=0,
    )

    mean = all_frames.mean(dim=0)
    std = all_frames.std(dim=0).clamp_min(1e-5)

    return mean, std


def normalize_recordings(recordings, mean, std):

    normalized = []

    for recording in recordings:

        mel = recording["mel"]

        mel = (
            mel - mean.unsqueeze(1)
        ) / std.unsqueeze(1)

        normalized.append(
            {
                "name": recording["name"],
                "mel": mel,
            }
        )

    return normalized


class MelChunkDataset(Dataset):

    def __init__(
        self,
        recordings,
        chunk_frames,
    ):

        self.chunks = []
        self.recording_indices = []

        for recording_index, recording in enumerate(recordings):

            mel = recording["mel"]

            total_frames = mel.shape[1]

            for start in range(
                0,
                total_frames,
                chunk_frames,
            ):

                chunk = mel[
                    :,
                    start:start + chunk_frames,
                ]

                if chunk.shape[1] < chunk_frames:

                    chunk = F.pad(
                        chunk,
                        (
                            0,
                            chunk_frames - chunk.shape[1],
                        ),
                    )

                self.chunks.append(chunk)
                self.recording_indices.append(recording_index)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index]


# ======================================================================
# K-MEANS INITIALIZATION
# ======================================================================

@torch.no_grad()
def kmeans_initialize(
    vectors: torch.Tensor,
    codebook_size: int,
    iterations: int,
):

    """
    Initialize codebook vectors from encoder outputs.

    Random initialization can place codebook entries far from the
    encoder's latent distribution.

    K-means places codebook vectors near actual encoder outputs.
    """

    vectors = vectors.float()

    if vectors.shape[0] < codebook_size:

        indices = torch.randint(
            0,
            vectors.shape[0],
            (codebook_size,),
            device=vectors.device,
        )

        return vectors[indices].clone()

    indices = torch.randperm(
        vectors.shape[0],
        device=vectors.device,
    )[:codebook_size]

    centers = vectors[indices].clone()

    for _ in range(iterations):

        distances = (
            vectors.pow(2).sum(dim=1, keepdim=True)
            - 2 * vectors @ centers.T
            + centers.pow(2).sum(dim=1).unsqueeze(0)
        )

        assignments = distances.argmin(dim=1)

        counts = torch.bincount(
            assignments,
            minlength=codebook_size,
        ).float()

        sums = torch.zeros_like(centers)

        sums.index_add_(
            0,
            assignments,
            vectors,
        )

        nonempty = counts > 0

        centers[nonempty] = (
            sums[nonempty] /
            counts[nonempty].unsqueeze(1)
        )

        empty = ~nonempty

        if empty.any():

            replacement_indices = torch.randint(
                0,
                vectors.shape[0],
                (int(empty.sum().item()),),
                device=vectors.device,
            )

            centers[empty] = vectors[
                replacement_indices
            ]

    return centers


# ======================================================================
# EMA VECTOR QUANTIZER
# ======================================================================

class EMAVectorQuantizer(nn.Module):

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        decay: float,
        epsilon: float,
        commitment_weight: float,
        dead_threshold: float,
        dead_patience: int,
    ):

        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        self.decay = decay
        self.epsilon = epsilon

        self.commitment_weight = commitment_weight

        self.dead_threshold = dead_threshold
        self.dead_patience = dead_patience

        self.register_buffer(
            "embedding",
            torch.randn(
                codebook_size,
                embedding_dim,
            ),
        )

        self.register_buffer(
            "ema_count",
            torch.ones(codebook_size),
        )

        self.register_buffer(
            "ema_sum",
            torch.randn(
                codebook_size,
                embedding_dim,
            ),
        )

        self.register_buffer(
            "dead_steps",
            torch.zeros(codebook_size),
        )

        self.register_buffer(
            "initialized",
            torch.tensor(False),
        )

    @torch.no_grad()
    def initialize(self, vectors):

        print("\nInitializing codebook using K-means...")

        centers = kmeans_initialize(
            vectors,
            self.codebook_size,
            KMEANS_ITERATIONS,
        )

        self.embedding.copy_(centers)

        self.ema_sum.copy_(centers)

        self.ema_count.fill_(1.0)

        self.dead_steps.zero_()

        self.initialized.fill_(True)

        print("K-means initialization complete.")

    def forward(self, z):

        """
        z: [B, D, T]

        Returns:
            quantized: [B, D, T]
            commitment_loss
            indices: [B, T]
            perplexity
            active_codes
        """

        batch_size, embedding_dim, time_steps = z.shape

        # Convert [B, D, T] -> [B*T, D]
        flat_z = z.permute(
            0,
            2,
            1,
        ).reshape(
            -1,
            embedding_dim,
        )

        if not bool(self.initialized.item()):

            self.initialize(flat_z.detach())

        # --------------------------------------------------------------
        # Nearest-neighbor lookup
        # --------------------------------------------------------------

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding.T
            + self.embedding.pow(2).sum(dim=1).unsqueeze(0)
        )

        indices = distances.argmin(dim=1)

        quantized = F.embedding(
            indices,
            self.embedding,
        )

        # --------------------------------------------------------------
        # Commitment loss
        # --------------------------------------------------------------

        commitment_loss = F.mse_loss(
            flat_z,
            quantized.detach(),
        )

        commitment_loss = (
            self.commitment_weight *
            commitment_loss
        )

        # --------------------------------------------------------------
        # EMA update
        # --------------------------------------------------------------

        if self.training:

            with torch.no_grad():

                one_hot = F.one_hot(
                    indices,
                    num_classes=self.codebook_size,
                ).float()

                batch_count = one_hot.sum(dim=0)

                batch_sum = one_hot.T @ flat_z.detach()

                self.ema_count.mul_(
                    self.decay
                ).add_(
                    batch_count,
                    alpha=1.0 - self.decay,
                )

                self.ema_sum.mul_(
                    self.decay
                ).add_(
                    batch_sum,
                    alpha=1.0 - self.decay,
                )

                # Laplace smoothing prevents division by zero.
                total_count = self.ema_count.sum()

                smoothed_count = (
                    (self.ema_count + self.epsilon)
                    /
                    (total_count + self.codebook_size * self.epsilon)
                    * total_count
                )

                updated_embedding = (
                    self.ema_sum /
                    smoothed_count.unsqueeze(1).clamp_min(
                        self.epsilon
                    )
                )

                self.embedding.copy_(
                    updated_embedding
                )

                # ------------------------------------------------------
                # Dead-code detection
                # ------------------------------------------------------

                dead = self.ema_count < self.dead_threshold

                self.dead_steps[dead] += 1
                self.dead_steps[~dead] = 0

                dead_indices = torch.where(
                    self.dead_steps >= self.dead_patience
                )[0]

                # Replace unused codes with actual encoder vectors.
                if len(dead_indices) > 0:

                    replacement_indices = torch.randint(
                        0,
                        flat_z.shape[0],
                        (len(dead_indices),),
                        device=flat_z.device,
                    )

                    replacements = flat_z[
                        replacement_indices
                    ].detach()

                    self.embedding[dead_indices] = replacements
                    self.ema_sum[dead_indices] = replacements
                    self.ema_count[dead_indices] = 1.0
                    self.dead_steps[dead_indices] = 0

        # --------------------------------------------------------------
        # Straight-through estimator
        # --------------------------------------------------------------

        quantized = F.embedding(
            indices,
            self.embedding,
        )

        # Gradient flows through the encoder as if quantization were
        # an identity function.
        quantized = flat_z + (
            quantized - flat_z
        ).detach()

        quantized = quantized.reshape(
            batch_size,
            time_steps,
            embedding_dim,
        ).permute(
            0,
            2,
            1,
        )

        # --------------------------------------------------------------
        # Codebook statistics
        # --------------------------------------------------------------

        counts = torch.bincount(
            indices,
            minlength=self.codebook_size,
        ).float()

        probabilities = counts / counts.sum().clamp_min(1.0)

        entropy = -(
            probabilities *
            torch.log(probabilities.clamp_min(1e-10))
        ).sum()

        perplexity = torch.exp(entropy)

        active_codes = (counts > 0).sum()

        return (
            quantized,
            commitment_loss,
            indices.reshape(batch_size, time_steps),
            perplexity,
            active_codes,
        )


# ======================================================================
# VQ-VAE MODEL
# ======================================================================

class SpeechMelVQVAE(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder:
        # 128 mel frames -> 64 -> 32 latent frames.

        self.encoder = nn.Sequential(

            nn.Conv1d(
                N_MELS,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv1d(
                128,
                LATENT_DIM,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

        )

        self.quantizer = EMAVectorQuantizer(
            codebook_size=CODEBOOK_SIZE,
            embedding_dim=LATENT_DIM,
            decay=EMA_DECAY,
            epsilon=EMA_EPSILON,
            commitment_weight=COMMITMENT_WEIGHT,
            dead_threshold=DEAD_CODE_THRESHOLD,
            dead_patience=DEAD_CODE_PATIENCE,
        )

        # Decoder:
        # 32 latent frames -> 64 -> 128 mel frames.

        self.decoder = nn.Sequential(

            nn.ConvTranspose1d(
                LATENT_DIM,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.GELU(),

            nn.ConvTranspose1d(
                128,
                N_MELS,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

        )

    def forward(self, mel):

        z = self.encoder(mel)

        (
            quantized,
            commitment_loss,
            indices,
            perplexity,
            active_codes,
        ) = self.quantizer(z)

        reconstruction = self.decoder(
            quantized
        )

        return (
            reconstruction,
            commitment_loss,
            indices,
            perplexity,
            active_codes,
        )

    @torch.no_grad()
    def encode(self, mel):

        z = self.encoder(mel)

        distances = (
            z.transpose(1, 2).pow(2).sum(dim=-1, keepdim=True)
            - 2 * z.transpose(1, 2) @ self.quantizer.embedding.T
            + self.quantizer.embedding.pow(2).sum(dim=1)
        )

        return distances.argmin(dim=-1)


# ======================================================================
# TRAINING AND EVALUATION
# ======================================================================

def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_commitment = 0.0

    total_perplexity = 0.0
    total_active = 0.0

    total_batches = 0

    for mel in loader:

        mel = mel.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            (
                reconstruction,
                commitment_loss,
                indices,
                perplexity,
                active_codes,
            ) = model(mel)

            reconstruction_loss = F.l1_loss(
                reconstruction,
                mel,
            )

            total_loss_value = (
                reconstruction_loss +
                commitment_loss
            )

            if training:

                total_loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        total_loss += total_loss_value.item()
        total_reconstruction += reconstruction_loss.item()
        total_commitment += commitment_loss.item()

        total_perplexity += perplexity.item()
        total_active += active_codes.item()

        total_batches += 1

    divisor = max(total_batches, 1)

    return {
        "loss": total_loss / divisor,
        "reconstruction": total_reconstruction / divisor,
        "commitment": total_commitment / divisor,
        "perplexity": total_perplexity / divisor,
        "active_codes": total_active / divisor,
    }


# ======================================================================
# TOKEN EXPORT
# ======================================================================

@torch.no_grad()
def export_tokens(
    model,
    recordings,
    mean,
    std,
    device,
    output_dir,
    split_name,
):

    model.eval()

    split_dir = output_dir / f"{split_name}_tokens"

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for recording in recordings:

        mel = recording["mel"]

        normalized = (
            mel - mean.unsqueeze(1)
        ) / std.unsqueeze(1)

        # Pad to a multiple of four for the encoder.
        original_frames = normalized.shape[1]

        padding = (
            4 - original_frames % 4
        ) % 4

        normalized = F.pad(
            normalized,
            (0, padding),
        )

        normalized = normalized.unsqueeze(0).to(device)

        tokens = model.encode(
            normalized
        )

        tokens = tokens.squeeze(0).cpu().numpy()

        output_path = (
            split_dir /
            f"{recording['name']}.npy"
        )

        np.save(
            output_path,
            tokens.astype(np.int64),
        )

        print(
            f"  {recording['name']}: "
            f"{original_frames} mel frames -> "
            f"{tokens.shape[0]} discrete tokens"
        )


# ======================================================================
# VISUALIZATION
# ======================================================================

def save_plots(
    history,
    model,
    example_batch,
    output_dir,
    device,
):

    # Training loss.
    plt.figure(figsize=(10, 5))

    plt.plot(
        history["train_loss"],
        label="Train loss",
    )

    plt.plot(
        history["val_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VQ-VAE Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "training_loss.png",
        dpi=150,
    )

    plt.close()

    # Codebook usage.
    plt.figure(figsize=(10, 5))

    plt.plot(
        history["active_codes"],
        label="Active codes",
    )

    plt.plot(
        history["perplexity"],
        label="Perplexity",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("VQ Codebook Health")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "codebook_usage.png",
        dpi=150,
    )

    plt.close()

    # Reconstruction.
    model.eval()

    with torch.no_grad():

        mel = example_batch[:1].to(device)

        (
            reconstruction,
            _,
            _,
            _,
            _,
        ) = model(mel)

    original = mel[0].cpu().numpy()
    reconstructed = reconstruction[0].cpu().numpy()

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 1, 1)

    plt.imshow(
        original,
        origin="lower",
        aspect="auto",
    )

    plt.title("Original normalized mel spectrogram")
    plt.ylabel("Mel bin")

    plt.subplot(2, 1, 2)

    plt.imshow(
        reconstructed,
        origin="lower",
        aspect="auto",
    )

    plt.title("Reconstructed normalized mel spectrogram")
    plt.xlabel("Frame")
    plt.ylabel("Mel bin")

    plt.tight_layout()

    plt.savefig(
        output_dir / "example_reconstruction.png",
        dpi=150,
    )

    plt.close()


# ======================================================================
# MAIN TRAINING
# ======================================================================

def train(args):

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("LESSON 41 — REPAIR VQ-VAE CODEBOOK COLLAPSE")
    print("=" * 72)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Mel bins: {N_MELS}")
    print(f"Codebook size: {CODEBOOK_SIZE}")
    print(f"Latent dimension: {LATENT_DIM}")
    print(f"Chunk frames: {CHUNK_FRAMES}")

    # ------------------------------------------------------------------
    # STEP 1 — AUDIO DATASET
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 1 — LOAD AUDIO")
    print("=" * 72)

    files = find_audio_files(audio_dir)

    print(f"Audio files found: {len(files)}")

    train_files, val_files = split_files(
        files,
        args.val_ratio,
    )

    print(f"Train recordings: {len(train_files)}")
    print(f"Validation recordings: {len(val_files)}")

    extractor = MelExtractor(device)

    print("\nExtracting training recordings...")

    train_recordings = extract_recordings(
        train_files,
        extractor,
    )

    print("\nExtracting validation recordings...")

    val_recordings = extract_recordings(
        val_files,
        extractor,
    )

    # ------------------------------------------------------------------
    # STEP 2 — NORMALIZATION
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 2 — NORMALIZE MEL FEATURES")
    print("=" * 72)

    mean, std = calculate_normalization(
        train_recordings
    )

    train_normalized = normalize_recordings(
        train_recordings,
        mean,
        std,
    )

    val_normalized = normalize_recordings(
        val_recordings,
        mean,
        std,
    )

    torch.save(
        {
            "mean": mean,
            "std": std,
        },
        output_dir / "mel_normalization.pt",
    )

    # ------------------------------------------------------------------
    # STEP 3 — CREATE CHUNKS
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 3 — CREATE MEL CHUNKS")
    print("=" * 72)

    train_dataset = MelChunkDataset(
        train_normalized,
        CHUNK_FRAMES,
    )

    val_dataset = MelChunkDataset(
        val_normalized,
        CHUNK_FRAMES,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            "No training or validation chunks were generated."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Train chunks: {len(train_dataset)}")
    print(f"Validation chunks: {len(val_dataset)}")

    # ------------------------------------------------------------------
    # STEP 4 — MODEL
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 4 — BUILD VQ-VAE")
    print("=" * 72)

    model = SpeechMelVQVAE().to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    # ------------------------------------------------------------------
    # STEP 5 — TRAIN
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 5 — TRAIN VQ-VAE")
    print("=" * 72)

    history = {
        "train_loss": [],
        "val_loss": [],
        "reconstruction_loss": [],
        "commitment_loss": [],
        "perplexity": [],
        "active_codes": [],
    }

    best_val_loss = float("inf")

    best_checkpoint = (
        output_dir / "best_model.pt"
    )

    for epoch in range(1, args.epochs + 1):

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            training=False,
        )

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["reconstruction_loss"].append(
            train_metrics["reconstruction"]
        )

        history["commitment_loss"].append(
            train_metrics["commitment"]
        )

        history["perplexity"].append(
            train_metrics["perplexity"]
        )

        history["active_codes"].append(
            train_metrics["active_codes"]
        )

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "config": {
                        "sample_rate": SAMPLE_RATE,
                        "n_mels": N_MELS,
                        "latent_dim": LATENT_DIM,
                        "codebook_size": CODEBOOK_SIZE,
                    },
                },
                best_checkpoint,
            )

        if (
            epoch == 1
            or epoch % args.log_interval == 0
            or epoch == args.epochs
        ):

            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"Train {train_metrics['loss']:.5f} | "
                f"Val {val_metrics['loss']:.5f} | "
                f"Recon {train_metrics['reconstruction']:.5f} | "
                f"VQ {train_metrics['commitment']:.5f} | "
                f"Codes {train_metrics['active_codes']:.0f}/"
                f"{CODEBOOK_SIZE} | "
                f"Perplexity {train_metrics['perplexity']:.2f}"
            )

    # ------------------------------------------------------------------
    # STEP 6 — RESTORE BEST CHECKPOINT
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 6 — RESTORE BEST MODEL")
    print("=" * 72)

    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.5f}")

    # ------------------------------------------------------------------
    # STEP 7 — EXPORT DISCRETE TOKENS
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 7 — EXPORT NEURAL DISCRETE TOKENS")
    print("=" * 72)

    print("\nTraining recordings:")

    export_tokens(
        model,
        train_recordings,
        mean,
        std,
        device,
        output_dir,
        "train",
    )

    print("\nValidation recordings:")

    export_tokens(
        model,
        val_recordings,
        mean,
        std,
        device,
        output_dir,
        "val",
    )

    # ------------------------------------------------------------------
    # STEP 8 — SAVE RESULTS
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 8 — SAVE RESULTS")
    print("=" * 72)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "config": {
                "sample_rate": SAMPLE_RATE,
                "n_mels": N_MELS,
                "latent_dim": LATENT_DIM,
                "codebook_size": CODEBOOK_SIZE,
            },
        },
        output_dir / "final_model.pt",
    )

    np.save(
        output_dir / "codebook.npy",
        model.quantizer.embedding.cpu().numpy(),
    )

    with open(
        output_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    example_batch = next(iter(train_loader))

    save_plots(
        history,
        model,
        example_batch,
        output_dir,
        device,
    )

    # ------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("LESSON 41 COMPLETE")
    print("=" * 72)

    print(f"Best validation loss: {best_val_loss:.5f}")

    print(
        f"Final active codes: "
        f"{history['active_codes'][-1]:.0f}/{CODEBOOK_SIZE}"
    )

    print(
        f"Final codebook perplexity: "
        f"{history['perplexity'][-1]:.2f}"
    )

    print("\nOutput directory:")

    print(output_dir.resolve())

    print("\nFiles:")
    print("  best_model.pt")
    print("  final_model.pt")
    print("  mel_normalization.pt")
    print("  codebook.npy")
    print("  training_history.json")
    print("  training_loss.png")
    print("  codebook_usage.png")
    print("  example_reconstruction.png")
    print("  train_tokens/")
    print("  val_tokens/")

    print("\nIMPORTANT:")
    print("  Check codebook_usage.png.")
    print("  A healthy codebook should use multiple codes.")
    print("  High perplexity alone does not prove good speech quality.")
    print("  This model reconstructs mel features, not waveform audio.")

    print("\nNEXT:")
    print("  Lesson 42 — Evaluate and improve neural speech representations")

    print("=" * 72)


# ======================================================================
# ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-dir",
        type=str,
        default="dataset",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="lesson41_output",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train(args)
