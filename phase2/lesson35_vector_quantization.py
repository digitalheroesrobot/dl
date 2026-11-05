
"""
======================================================================
PHASE 2 — LESSON 35
VECTOR QUANTIZATION FROM SCRATCH
======================================================================

PURPOSE
-------
Convert continuous speech latent vectors into discrete speech tokens.

Lesson 34 produced:

    waveform
        ↓
    neural speech encoder
        ↓
    continuous latent vectors

For example:

    latent =
        [
            [ 0.12, -0.41,  0.83, ... ],
            [ 0.19, -0.37,  0.71, ... ],
            [ 0.91,  0.22, -0.14, ... ],
            ...
        ]

Those vectors are continuous.

A generative TTS model such as our future T3-like model is much easier
to formulate as a sequence prediction problem if speech is represented
by discrete IDs:

    [17, 17, 241, 982, 982, 43, ...]


This lesson builds the mechanism that performs that conversion.


CORE IDEA
---------
We create a learned CODEBOOK.

Suppose:

    codebook size = 512
    latent dimension = 128

Then the codebook contains:

    512 vectors

and each vector has:

    128 dimensions


Conceptually:

    CODEBOOK

    ID 0   → [ ... 128 numbers ... ]
    ID 1   → [ ... 128 numbers ... ]
    ID 2   → [ ... 128 numbers ... ]
      ...
    ID 511 → [ ... 128 numbers ... ]


For every continuous latent vector:

    z

we find the nearest codebook vector:

    e_k

and replace:

    z

with:

    e_k


Therefore:

    continuous vector
            ↓
       nearest neighbor
            ↓
       integer index


Example:

    latent vector
        ↓
    nearest codebook vector = 173
        ↓
    speech token = 173


THIS IS VECTOR QUANTIZATION.


WHAT YOU LEARN
--------------
STEP 1
    Why continuous speech representations need discretization.

STEP 2
    Build a codebook from scratch.

STEP 3
    Calculate Euclidean distances.

STEP 4
    Select the nearest codebook vector.

STEP 5
    Convert latent vectors into integer IDs.

STEP 6
    Reconstruct the quantized continuous representation.

STEP 7
    Understand quantization error.

STEP 8
    Understand the straight-through estimator.

STEP 9
    Implement VQ loss.

STEP 10
    Measure codebook usage and perplexity.

STEP 11
    Train a small VQ module on speech-like latent data.

STEP 12
    Visualize continuous latent vectors versus quantized vectors.


IMPORTANT DISTINCTION
---------------------

Lesson 34:

    waveform
        ↓
    continuous latent

Lesson 35:

    continuous latent
        ↓
    discrete IDs

Lesson 36:

    waveform
        ↕
    encoder
        ↕
    quantizer
        ↕
    decoder

and TRAIN THE WHOLE CODEC.


THE DISCRETE SPEECH PIPELINE
----------------------------

    AUDIO
      │
      ▼
    Encoder
      │
      ▼
    continuous latent z
      │
      ▼
    Vector Quantizer
      │
      ├───────────────┐
      │               │
      ▼               ▼
    token IDs       quantized z
      │               │
      │               ▼
      │            Decoder
      │               │
      │               ▼
      │          reconstructed audio
      │
      ▼
    T3-like model


WHY INTEGER TOKENS MATTER
-------------------------

An autoregressive model can learn:

    P(token_t | text, token_1, ..., token_{t-1})


For example:

    text:
        "hello"

    target speech tokens:

        42 42 91 203 203 71 18 ...


The T3-like model can therefore learn to predict:

    next speech token

just as a language model predicts the next text token.


CODEBOOK
--------
We use:

    NUM_CODES = 512

and:

    LATENT_DIM = 128

Therefore:

    codebook shape:

        [512, 128]


QUANTIZATION
------------

Given:

    z ∈ R^128

and codebook:

    E ∈ R^(512 × 128)

we calculate:

    distance(z, E_k)

for every k.

Then:

    k* = argmin_k distance(z, E_k)


and:

    z_q = E_k*


QUANTIZATION ERROR
------------------

The quantization error is:

    ||z - z_q||²

A good codebook should make this error small.


STRAIGHT-THROUGH ESTIMATOR
--------------------------

The nearest-neighbor operation contains an argmin:

    k = argmin distance

which is not differentiable.

Therefore during training we use:

    z_q_st =
        z + (z_q - z).detach()

Forward pass:

    z_q_st ≈ z_q

Backward pass:

    gradient flows through z


This allows the encoder to learn together with the quantizer.


VQ LOSS
-------
A common VQ objective contains two important terms:

    codebook loss:

        ||sg[z] - z_q||²

    commitment loss:

        β ||z - sg[z_q]||²


where:

    sg = stop-gradient


The first term moves codebook vectors toward encoder outputs.

The second term encourages the encoder to commit to the codebook.


CODEBOOK PERPLEXITY
-------------------

We also measure how many codebook entries are actually being used.

If only one code is used:

    perplexity ≈ 1

If many codes are used uniformly:

    perplexity → NUM_CODES

This is useful because codebook collapse is a serious problem.

For example:

    512 available codes

but the model uses only:

    3 codes

Then the representation has very low information capacity.


EXPERIMENT
----------

This lesson first demonstrates VQ mathematically.

Then it trains a small VQ module on structured synthetic speech-like
latent data.

This is NOT yet a complete speech codec.

The purpose is to understand the quantizer itself.


NEXT LESSON
-----------

Lesson 36:

    TRAIN SPEECH CODEC

We will build:

    waveform
        ↓
    encoder
        ↓
    vector quantizer
        ↓
    decoder
        ↓
    reconstructed waveform

and train it end-to-end.

At that point the discrete speech tokens will no longer be arbitrary
codes. They will actually carry information needed to reconstruct
speech.
======================================================================
"""

import argparse
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

LATENT_DIM = 128

NUM_CODES = 512

BETA = 0.25

BATCH_SIZE = 64

TRAIN_STEPS = 1200

LEARNING_RATE = 2e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ======================================================================
# SYNTHETIC SPEECH LATENT DATA
# ======================================================================

def generate_speech_latents(
    batch_size,
    sequence_length,
    latent_dim=LATENT_DIM,
):
    """
    Generate structured synthetic latent vectors.

    This is NOT speech.

    We create several latent clusters representing different
    hypothetical acoustic states.

    The goal is to give the VQ module a meaningful distribution to
    learn rather than completely random Gaussian vectors.

    In a real codec:

        waveform → encoder → latent

    would produce these vectors.

    Here we directly generate synthetic latent vectors so that this
    lesson can focus on vector quantization.
    """

    device = DEVICE

    # --------------------------------------------------------------
    # Create a small set of latent prototypes.
    # --------------------------------------------------------------

    num_prototypes = 32

    prototypes = torch.randn(
        num_prototypes,
        latent_dim,
        device=device,
    )

    prototypes = F.normalize(
        prototypes,
        dim=-1,
    )

    # Scale prototypes.

    prototypes *= 2.0

    # --------------------------------------------------------------
    # Random prototype assignment.
    # --------------------------------------------------------------

    prototype_ids = torch.randint(
        0,
        num_prototypes,
        (
            batch_size,
            sequence_length,
        ),
        device=device,
    )

    latent = prototypes[
        prototype_ids
    ]

    # --------------------------------------------------------------
    # Add temporal smoothness.
    # --------------------------------------------------------------

    if sequence_length > 1:

        previous = latent[:, :-1]

        current = latent[:, 1:]

        latent[:, 1:] = (
            0.75 * current
            + 0.25 * previous
        )

    # --------------------------------------------------------------
    # Add noise.
    # --------------------------------------------------------------

    noise = (
        0.20
        * torch.randn_like(latent)
    )

    latent = latent + noise

    # --------------------------------------------------------------
    # Add a low-frequency temporal component.
    # --------------------------------------------------------------

    time = torch.linspace(
        0,
        1,
        sequence_length,
        device=device,
    )

    temporal_signal = (
        0.15
        * torch.sin(
            2.0
            * math.pi
            * 2.0
            * time
        )
    )

    temporal_signal = temporal_signal.view(
        1,
        sequence_length,
        1,
    )

    latent = latent + temporal_signal

    return latent


# ======================================================================
# VECTOR QUANTIZER
# ======================================================================

class VectorQuantizer(nn.Module):
    """
    Basic VQ module.

    Input:

        z
        [B, T, D]

    Codebook:

        [K, D]

    Output:

        quantized
        [B, T, D]

        indices
        [B, T]

        losses
        dictionary
    """

    def __init__(
        self,
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        beta=BETA,
    ):

        super().__init__()

        self.num_codes = num_codes

        self.latent_dim = latent_dim

        self.beta = beta

        # ----------------------------------------------------------
        # CODEBOOK
        # ----------------------------------------------------------

        self.codebook = nn.Embedding(
            num_codes,
            latent_dim,
        )

        # Small initialization.

        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / num_codes,
            1.0 / num_codes,
        )

    # ------------------------------------------------------------------
    # Distance calculation
    # ------------------------------------------------------------------

    def calculate_distances(self, z):

        """
        Calculate squared Euclidean distance.

        z:

            [N, D]

        codebook:

            [K, D]

        output:

            [N, K]

        Efficient identity:

            ||z - e||²
              =
            ||z||² + ||e||² - 2 z·e
        """

        codebook = (
            self.codebook.weight
        )

        z_squared = torch.sum(
            z ** 2,
            dim=1,
            keepdim=True,
        )

        codebook_squared = torch.sum(
            codebook ** 2,
            dim=1,
            keepdim=True,
        ).transpose(
            0,
            1,
        )

        dot_product = (
            z @ codebook.t()
        )

        distances = (
            z_squared
            + codebook_squared
            - 2.0 * dot_product
        )

        return distances

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z):

        batch_size = z.shape[0]

        sequence_length = z.shape[1]

        latent_dim = z.shape[2]

        # Flatten temporal positions.

        flat_z = z.reshape(
            -1,
            latent_dim,
        )

        # ----------------------------------------------------------
        # Find nearest codebook vector.
        # ----------------------------------------------------------

        distances = self.calculate_distances(
            flat_z
        )

        indices = torch.argmin(
            distances,
            dim=1,
        )

        # ----------------------------------------------------------
        # Lookup quantized vectors.
        # ----------------------------------------------------------

        quantized = self.codebook(
            indices
        )

        quantized = quantized.reshape(
            batch_size,
            sequence_length,
            latent_dim,
        )

        indices = indices.reshape(
            batch_size,
            sequence_length,
        )

        # ----------------------------------------------------------
        # VQ losses
        # ----------------------------------------------------------

        # Codebook loss:
        #
        # Move codebook vectors toward encoder outputs.
        #
        # stop-gradient on z.

        codebook_loss = F.mse_loss(
            quantized,
            z.detach(),
        )

        # Commitment loss:
        #
        # Encourage encoder outputs to stay near selected codebook
        # vectors.
        #
        # stop-gradient on quantized.

        commitment_loss = F.mse_loss(
            z,
            quantized.detach(),
        )

        vq_loss = (
            codebook_loss
            + self.beta
            * commitment_loss
        )

        # ----------------------------------------------------------
        # Straight-through estimator.
        # ----------------------------------------------------------

        quantized_st = (
            z
            + (
                quantized - z
            ).detach()
        )

        # ----------------------------------------------------------
        # Perplexity.
        # ----------------------------------------------------------

        perplexity = self.calculate_perplexity(
            indices
        )

        return (
            quantized_st,
            indices,
            vq_loss,
            codebook_loss,
            commitment_loss,
            perplexity,
        )

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def calculate_perplexity(
        self,
        indices,
    ):

        flat_indices = indices.reshape(
            -1
        )

        counts = torch.bincount(
            flat_indices,
            minlength=self.num_codes,
        ).float()

        probabilities = (
            counts
            / counts.sum().clamp_min(1.0)
        )

        nonzero = (
            probabilities > 0
        )

        entropy = -torch.sum(
            probabilities[nonzero]
            * torch.log(
                probabilities[nonzero]
                + 1e-10
            )
        )

        perplexity = torch.exp(
            entropy
        )

        return perplexity

    # ------------------------------------------------------------------
    # Code usage
    # ------------------------------------------------------------------

    def code_usage(
        self,
        indices,
    ):

        unique_codes = torch.unique(
            indices
        )

        return unique_codes.numel()


# ======================================================================
# STEP 1 — MANUAL VQ DEMONSTRATION
# ======================================================================

def manual_vq_demo():

    print("\n")
    print("=" * 72)
    print("STEP 1 — MANUAL VECTOR QUANTIZATION")
    print("=" * 72)

    codebook = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    vectors = torch.tensor(
        [
            [0.1, 0.2],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.9],
        ],
        dtype=torch.float32,
    )

    print("\nCodebook:")

    for index, vector in enumerate(
        codebook
    ):

        print(
            f"Code {index}: "
            f"{vector.numpy()}"
        )

    print("\nInput vectors:")

    for vector in vectors:

        print(
            vector.numpy()
        )

    # --------------------------------------------------------------
    # Calculate pairwise distances.
    # --------------------------------------------------------------

    distances = torch.cdist(
        vectors,
        codebook,
        p=2,
    )

    print("\nDistances:")

    print(
        distances.numpy()
    )

    indices = torch.argmin(
        distances,
        dim=1,
    )

    print(
        "\nNearest code IDs:"
    )

    print(
        indices.numpy()
    )

    quantized = codebook[
        indices
    ]

    print(
        "\nQuantized vectors:"
    )

    print(
        quantized.numpy()
    )

    errors = (
        vectors - quantized
    )

    squared_errors = torch.sum(
        errors ** 2,
        dim=1,
    )

    print(
        "\nSquared quantization errors:"
    )

    print(
        squared_errors.numpy()
    )

    print(
        "\nThis is the fundamental operation of vector quantization:"
    )

    print(
        """
continuous vector
       |
       v
distance to every codebook vector
       |
       v
nearest codebook entry
       |
       v
integer token ID
"""
    )


# ======================================================================
# STEP 2 — QUANTIZER FORWARD TEST
# ======================================================================

def quantizer_forward_test():

    print("\n")
    print("=" * 72)
    print("STEP 2 — QUANTIZER FORWARD TEST")
    print("=" * 72)

    quantizer = VectorQuantizer(
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        beta=BETA,
    ).to(DEVICE)

    latent = generate_speech_latents(
        batch_size=2,
        sequence_length=50,
    )

    with torch.no_grad():

        (
            quantized,
            indices,
            vq_loss,
            codebook_loss,
            commitment_loss,
            perplexity,
        ) = quantizer(
            latent
        )

    print(
        "\nInput latent shape:"
    )

    print(
        latent.shape
    )

    print(
        "\nQuantized shape:"
    )

    print(
        quantized.shape
    )

    print(
        "\nToken ID shape:"
    )

    print(
        indices.shape
    )

    print(
        "\nFirst speech-token sequence:"
    )

    print(
        indices[0].cpu().numpy()
    )

    print(
        "\nVQ loss:"
    )

    print(
        f"{vq_loss.item():.6f}"
    )

    print(
        "\nCodebook loss:"
    )

    print(
        f"{codebook_loss.item():.6f}"
    )

    print(
        "\nCommitment loss:"
    )

    print(
        f"{commitment_loss.item():.6f}"
    )

    print(
        "\nCodebook perplexity:"
    )

    print(
        f"{perplexity.item():.3f}"
    )

    print(
        "\nCodes used:"
    )

    print(
        quantizer.code_usage(
            indices
        )
    )

    return quantizer


# ======================================================================
# STEP 3 — TRAIN VQ MODULE
# ======================================================================

def train_quantizer(
    quantizer,
    steps=TRAIN_STEPS,
):

    print("\n")
    print("=" * 72)
    print("STEP 3 — TRAIN VECTOR QUANTIZER")
    print("=" * 72)

    optimizer = torch.optim.AdamW(
        quantizer.parameters(),
        lr=LEARNING_RATE,
    )

    losses = []

    perplexities = []

    reconstruction_errors = []

    quantizer.train()

    for step in range(1, steps + 1):

        # ----------------------------------------------------------
        # Generate structured latent vectors.
        # ----------------------------------------------------------

        latent = generate_speech_latents(
            batch_size=BATCH_SIZE,
            sequence_length=50,
        )

        # ----------------------------------------------------------
        # Forward.
        # ----------------------------------------------------------

        (
            quantized,
            indices,
            vq_loss,
            codebook_loss,
            commitment_loss,
            perplexity,
        ) = quantizer(
            latent
        )

        # ----------------------------------------------------------
        # Additional reconstruction-style objective.
        #
        # This is useful for this isolated lesson because we want
        # quantized vectors to approximate the input vectors.
        # ----------------------------------------------------------

        quantization_error = F.mse_loss(
            quantized,
            latent,
        )

        loss = (
            quantization_error
            + vq_loss
        )

        # ----------------------------------------------------------
        # Backward.
        # ----------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            quantizer.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        # ----------------------------------------------------------
        # Logging.
        # ----------------------------------------------------------

        losses.append(
            loss.item()
        )

        perplexities.append(
            perplexity.item()
        )

        reconstruction_errors.append(
            quantization_error.item()
        )

        if (
            step == 1
            or step % 100 == 0
            or step == steps
        ):

            usage = (
                quantizer.code_usage(
                    indices
                )
            )

            print(
                f"Step {step:4d}/{steps} | "
                f"Loss {loss.item():.6f} | "
                f"QuantErr {quantization_error.item():.6f} | "
                f"Perplexity {perplexity.item():.2f} | "
                f"Codes {usage:3d}"
            )

    print(
        "\nTraining complete."
    )

    return {
        "loss": losses,
        "perplexity": perplexities,
        "quantization_error": reconstruction_errors,
    }


# ======================================================================
# STEP 4 — EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate_quantizer(
    quantizer,
):

    print("\n")
    print("=" * 72)
    print("STEP 4 — EVALUATE DISCRETE SPEECH TOKENS")
    print("=" * 72)

    quantizer.eval()

    latent = generate_speech_latents(
        batch_size=4,
        sequence_length=50,
    )

    (
        quantized,
        indices,
        vq_loss,
        codebook_loss,
        commitment_loss,
        perplexity,
    ) = quantizer(
        latent
    )

    mse = F.mse_loss(
        quantized,
        latent,
    )

    usage = quantizer.code_usage(
        indices
    )

    print(
        "\nEvaluation latent shape:"
    )

    print(
        latent.shape
    )

    print(
        "\nDiscrete token shape:"
    )

    print(
        indices.shape
    )

    print(
        "\nExample speech-token sequence:"
    )

    print(
        indices[0].cpu().numpy()
    )

    print(
        "\nNumber of codes used:"
    )

    print(
        f"{usage} / {NUM_CODES}"
    )

    print(
        "\nCode usage percentage:"
    )

    print(
        f"{100.0 * usage / NUM_CODES:.2f}%"
    )

    print(
        "\nPerplexity:"
    )

    print(
        f"{perplexity.item():.3f}"
    )

    print(
        "\nQuantization MSE:"
    )

    print(
        f"{mse.item():.6f}"
    )

    return (
        latent,
        quantized,
        indices,
    )


# ======================================================================
# STEP 5 — DECODE TOKEN IDS
# ======================================================================

@torch.no_grad()
def demonstrate_token_round_trip(
    quantizer,
    latent,
):

    print("\n")
    print("=" * 72)
    print("STEP 5 — DISCRETE TOKEN ROUND TRIP")
    print("=" * 72)

    quantizer.eval()

    (
        quantized,
        indices,
        *_,
    ) = quantizer(
        latent
    )

    print(
        "\nContinuous latent:"
    )

    print(
        latent[0, 0, :8]
        .cpu()
        .numpy()
    )

    print(
        "\nDiscrete token:"
    )

    print(
        indices[0, 0].item()
    )

    print(
        "\nCodebook vector:"
    )

    code_vector = quantizer.codebook(
        indices[0, 0]
    )

    print(
        code_vector[:8]
        .cpu()
        .numpy()
    )

    print(
        """
The important information flow is:

    continuous vector
           |
           v
       nearest code
           |
           v
       integer ID
           |
           v
       codebook lookup
           |
           v
    quantized vector
"""
    )


# ======================================================================
# STEP 6 — PLOT TRAINING
# ======================================================================

def plot_training_history(
    history,
):

    steps = np.arange(
        1,
        len(history["loss"]) + 1,
    )

    # --------------------------------------------------------------
    # Loss
    # --------------------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        steps,
        history["loss"],
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Vector Quantizer Training Loss"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        "lesson35_vq_loss.png",
        dpi=150,
    )

    plt.show()

    # --------------------------------------------------------------
    # Quantization error
    # --------------------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        steps,
        history["quantization_error"],
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "MSE"
    )

    plt.title(
        "Quantization Error"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        "lesson35_quantization_error.png",
        dpi=150,
    )

    plt.show()

    # --------------------------------------------------------------
    # Perplexity
    # --------------------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        steps,
        history["perplexity"],
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Perplexity"
    )

    plt.title(
        "Codebook Perplexity"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        "lesson35_vq_perplexity.png",
        dpi=150,
    )

    plt.show()


# ======================================================================
# STEP 7 — VISUALIZE LATENT VS QUANTIZED
# ======================================================================

def plot_latent_comparison(
    latent,
    quantized,
):

    latent = latent.detach().cpu()

    quantized = quantized.detach().cpu()

    # --------------------------------------------------------------
    # Select one sample and first latent dimension.
    # --------------------------------------------------------------

    original = latent[
        0,
        :,
        0,
    ].numpy()

    reconstructed = quantized[
        0,
        :,
        0,
    ].numpy()

    time = np.arange(
        len(original)
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        time,
        original,
        label="continuous latent",
        linewidth=2,
    )

    plt.step(
        time,
        reconstructed,
        where="mid",
        label="quantized latent",
        linewidth=1.5,
    )

    plt.xlabel(
        "Speech latent frame"
    )

    plt.ylabel(
        "Latent dimension 0"
    )

    plt.title(
        "Continuous vs Quantized Speech Representation"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        "lesson35_latent_quantization.png",
        dpi=150,
    )

    plt.show()


# ======================================================================
# STEP 8 — CODEBOOK VISUALIZATION
# ======================================================================

def plot_codebook(
    quantizer,
):

    codebook = (
        quantizer.codebook.weight
        .detach()
        .cpu()
        .numpy()
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.imshow(
        codebook,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )

    plt.colorbar(
        label="Codebook value"
    )

    plt.xlabel(
        "Latent dimension"
    )

    plt.ylabel(
        "Code ID"
    )

    plt.title(
        "Learned Vector-Quantization Codebook"
    )

    plt.tight_layout()

    plt.savefig(
        "lesson35_codebook.png",
        dpi=150,
    )

    plt.show()


# ======================================================================
# STEP 9 — CODE DISTRIBUTION
# ======================================================================

@torch.no_grad()
def plot_code_distribution(
    quantizer,
):

    quantizer.eval()

    latent = generate_speech_latents(
        batch_size=32,
        sequence_length=50,
    )

    (
        _,
        indices,
        *_,
    ) = quantizer(
        latent
    )

    counts = torch.bincount(
        indices.reshape(-1),
        minlength=NUM_CODES,
    )

    counts = counts.cpu().numpy()

    used = np.where(
        counts > 0
    )[0]

    print("\n")
    print("=" * 72)
    print("CODE DISTRIBUTION")
    print("=" * 72)

    print(
        "\nUsed codes:"
    )

    print(
        len(used)
    )

    print(
        "\nMost frequently used codes:"
    )

    top_indices = np.argsort(
        counts
    )[::-1][:20]

    for index in top_indices:

        if counts[index] == 0:

            continue

        print(
            f"Code {index:4d}: "
            f"{counts[index]:6d} uses"
        )

    plt.figure(
        figsize=(14, 5)
    )

    plt.bar(
        np.arange(NUM_CODES),
        counts,
    )

    plt.xlabel(
        "Code ID"
    )

    plt.ylabel(
        "Usage count"
    )

    plt.title(
        "Speech Code Usage Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "lesson35_code_distribution.png",
        dpi=150,
    )

    plt.show()


# ======================================================================
# STEP 10 — TOKEN SEQUENCE STATISTICS
# ======================================================================

def analyze_token_sequence(
    indices,
):

    flat = indices.reshape(
        -1
    )

    unique = torch.unique(
        flat
    )

    print("\n")
    print("=" * 72)
    print("DISCRETE SPEECH TOKEN STATISTICS")
    print("=" * 72)

    print(
        "\nTotal token positions:"
    )

    print(
        flat.numel()
    )

    print(
        "\nVocabulary / codebook size:"
    )

    print(
        NUM_CODES
    )

    print(
        "\nUnique tokens observed:"
    )

    print(
        unique.numel()
    )

    print(
        "\nToken sequence example:"
    )

    print(
        indices[0]
        .cpu()
        .numpy()
    )

    print(
        """
This sequence is now compatible with an autoregressive model.

Conceptually:

    [17, 17, 18, 18, 41, 90, 90, 92, ...]

can be treated as a sequence of discrete speech symbols.
"""
    )


# ======================================================================
# STEP 11 — SANITY CHECKS
# ======================================================================

def run_sanity_checks(
    quantizer,
    latent,
    quantized,
    indices,
):

    print("\n")
    print("=" * 72)
    print("SANITY CHECKS")
    print("=" * 72)

    assert latent.ndim == 3

    assert quantized.ndim == 3

    assert indices.ndim == 2

    assert latent.shape == quantized.shape

    assert latent.shape[:2] == indices.shape

    assert latent.shape[-1] == LATENT_DIM

    assert torch.all(
        indices >= 0
    )

    assert torch.all(
        indices < NUM_CODES
    )

    assert torch.isfinite(
        quantized
    ).all()

    assert quantizer.codebook.weight.shape == (
        NUM_CODES,
        LATENT_DIM,
    )

    print(
        "PASS — latent tensor shape"
    )

    print(
        "PASS — quantized tensor shape"
    )

    print(
        "PASS — token ID shape"
    )

    print(
        "PASS — token IDs in valid range"
    )

    print(
        "PASS — codebook dimensions"
    )

    print(
        "PASS — finite quantized values"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 35 — Vector Quantization "
            "from Scratch"
        )
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=TRAIN_STEPS,
        help="Number of VQ training steps.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------

    set_seed()

    print("=" * 72)
    print("PHASE 2 — LESSON 35")
    print("VECTOR QUANTIZATION FROM SCRATCH")
    print("=" * 72)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Latent dimension: {LATENT_DIM}"
    )

    print(
        f"Codebook size: {NUM_CODES}"
    )

    print(
        f"Commitment beta: {BETA}"
    )

    # --------------------------------------------------------------
    # Manual demonstration
    # --------------------------------------------------------------

    manual_vq_demo()

    # --------------------------------------------------------------
    # Build quantizer
    # --------------------------------------------------------------

    quantizer = quantizer_forward_test()

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    history = train_quantizer(
        quantizer,
        steps=args.steps,
    )

    # --------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------

    (
        latent,
        quantized,
        indices,
    ) = evaluate_quantizer(
        quantizer
    )

    # --------------------------------------------------------------
    # Round trip
    # --------------------------------------------------------------

    demonstrate_token_round_trip(
        quantizer,
        latent,
    )

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    analyze_token_sequence(
        indices
    )

    # --------------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------------

    plot_training_history(
        history
    )

    plot_latent_comparison(
        latent,
        quantized,
    )

    plot_codebook(
        quantizer
    )

    plot_code_distribution(
        quantizer
    )

    # --------------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------------

    run_sanity_checks(
        quantizer,
        latent,
        quantized,
        indices,
    )

    # --------------------------------------------------------------
    # Final architecture
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("LESSON 35 COMPLETE")
    print("=" * 72)

    print(
        """
We have now implemented vector quantization.

The transformation is:

    CONTINUOUS SPEECH LATENT
              |
              v
       distance calculation
              |
              v
       nearest codebook vector
              |
              v
        INTEGER CODE ID
              |
              v
       quantized latent


The complete representation pipeline is now:

    WAVEFORM
       |
       v
    Speech Encoder
       |
       v
    Continuous Latent
       |
       v
    Vector Quantizer
       |
       +------------------+
       |                  |
       v                  v
    Token IDs        Quantized Latent
       |                  |
       |                  |
       |              Decoder
       |                  |
       |                  v
       |              Waveform
       |
       v
    FUTURE T3 MODEL


The most important result of this lesson is that speech can now be
represented as a sequence of discrete integers.

For example:

    [17, 17, 17, 42, 42, 109, 109, 8, 8, 71, ...]


This looks much more like a language-modeling problem.

That gives us the central bridge:

    TEXT
      |
      v
    text tokens
      |
      v
    T3
      |
      v
    speech tokens
      |
      v
    speech decoder
      |
      v
    WAVEFORM


IMPORTANT:
The quantizer from this lesson is trained only on synthetic latent
vectors. It does NOT yet produce a useful speech codec.

That is exactly what Lesson 36 fixes.


======================================================================
NEXT — LESSON 36
TRAIN SPEECH CODEC
======================================================================

We will combine:

    waveform
        ↓
    encoder
        ↓
    vector quantizer
        ↓
    decoder
        ↓
    reconstructed waveform


and train the system end-to-end.

The optimization target will force the discrete representation to
preserve enough acoustic information to reconstruct the waveform.

After Lesson 36 we will finally have a real miniature neural speech
codec:

    AUDIO
      ↔
    DISCRETE SPEECH TOKENS


Then the following lessons become much more exciting:

    37  Autoregressive speech model
    38  Text → speech tokens
    39  Speaker-conditioned T3
    40  Reference speech conditioning
    41  Sampling
    42  CFG
    43  Flow matching
    44  Conditional flow matching
    45  ODE sampling
    46  Speech token → mel
    47  Reference-conditioned S3Gen
    48  HiFi-GAN / HiFT upgrade
    49  Mini Chatterbox
    50  Zero-shot voice cloning
"""
    )


if __name__ == "__main__":
    main()
