"""
======================================================================
PHASE 1 — LESSON 27
FRESH ATTACKER / POST-HOC CONTENT-LEAKAGE PROBE
======================================================================

PURPOSE
-------
Test whether a supposedly speaker-only embedding still contains
recoverable linguistic/content information.

WHY THIS LESSON MATTERS
-----------------------
In Lesson 26, we used adversarial training:

    speaker encoder <----> adversarial content classifier

The encoder was trained to make the adversarial classifier fail.

However, that creates an important experimental question:

    "Did the encoder really remove linguistic information,
     or did it merely learn to defeat that particular classifier?"

Lesson 27 answers this by freezing the encoder and training a completely
new attacker/probe network from scratch.

The attacker has:

    - random initialization
    - no access to the original adversarial classifier
    - no influence over the speaker encoder
    - only the frozen speaker embedding

If this fresh attacker can predict linguistic information accurately,
then the embedding still contains linguistic information.

----------------------------------------------------------------------
CORE IDEA
----------------------------------------------------------------------

                    TRAINING
                    --------

speech
  |
  v
speaker encoder
  |
  +------> speaker identity
  |
  +------> adversarial content classifier
                ^
                |
          gradient reversal


                    TEST
                    ----

speech
  |
  v
FROZEN speaker encoder
  |
  v
speaker embedding
  |
  v
FRESH attacker
  |
  v
content prediction


The attacker is trained AFTER the speaker encoder is frozen.

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1
------
Create synthetic speech-like feature sequences containing two factors:

    speaker identity
    linguistic/content identity

STEP 2
------
Train a normal speaker encoder.

STEP 3
------
Train an adversarial speaker encoder using gradient reversal.

STEP 4
------
Freeze both encoders.

STEP 5
------
Generate embeddings using the frozen encoders.

STEP 6
------
Create a brand-new attacker network.

STEP 7
------
Train the attacker only from embeddings.

STEP 8
------
Measure content classification accuracy.

STEP 9
------
Compare:

    baseline embedding leakage
    adversarial embedding leakage

STEP 10
-------
Interpret the result.

----------------------------------------------------------------------
IMPORTANT INTERPRETATION
----------------------------------------------------------------------

A LOW attacker accuracy means:

    linguistic information is difficult to recover.

A HIGH attacker accuracy means:

    linguistic information remains accessible.

For a good speaker representation we ideally want:

    speaker identity     -> easy to recover
    linguistic content   -> difficult to recover

This is representation disentanglement.

----------------------------------------------------------------------
REAL-WORLD EXAMPLE
----------------------------------------------------------------------

Imagine a voice-cloning system receives:

    "The cat is sleeping."

The speaker embedding should ideally represent:

    WHO is speaking

and not:

    WHAT is being said.

If the same speaker embedding allows a fresh attacker to determine
whether the person said:

    "cat"
    "dog"
    "car"
    "hello"

then the embedding contains content information.

That could hurt generalization when the embedding is reused with
different text.

----------------------------------------------------------------------
NO REAL AUDIO REQUIRED
----------------------------------------------------------------------

This lesson deliberately uses synthetic acoustic-like feature vectors.

The purpose is to understand the representation-learning principle
before introducing a real speech dataset.

Later lessons can replace these synthetic features with:

    mel spectrograms
    HuBERT features
    wav2vec-style features
    real speech recordings

without changing the fundamental experiment.

----------------------------------------------------------------------
DEPENDENCIES
----------------------------------------------------------------------

Python >= 3.9

PyTorch
NumPy
Matplotlib

Example:

    pip install torch numpy matplotlib

----------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------

The script creates:

    outputs/lesson27/
        fresh_probe_results.png
        baseline_probe.pt
        adversarial_probe.pt

The main plot contains:

    1. baseline attacker accuracy
    2. adversarial attacker accuracy

----------------------------------------------------------------------
LESSON 27 -> LESSON 28
----------------------------------------------------------------------

Lesson 28 will move from merely measuring leakage toward improving
speaker conditioning and reference representation.

======================================================================
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs/lesson27"

# Synthetic dataset
NUM_SPEAKERS = 8
NUM_CONTENTS = 16

FEATURE_DIM = 32
EMBED_DIM = 16

SAMPLES_PER_COMBINATION = 12

TRAIN_RATIO = 0.8

BATCH_SIZE = 128

ENCODER_EPOCHS = 120
PROBE_EPOCHS = 100

ENCODER_LR = 1e-3
PROBE_LR = 1e-3

PRINT_EVERY = 10


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is useful for a learning experiment.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# SYNTHETIC SPEECH-LIKE DATA
# ======================================================================

@dataclass
class Dataset:
    x: torch.Tensor
    speaker: torch.Tensor
    content: torch.Tensor


def make_dataset() -> Dataset:
    """
    Construct synthetic acoustic-like features.

    Every sample contains two latent factors:

        speaker vector
        content vector

    The observed feature is:

        x = speaker_component
            + content_component
            + nonlinear interaction
            + noise

    This approximates the idea that acoustic observations contain
    information about both WHO is speaking and WHAT is being spoken.
    """

    rng = np.random.default_rng(SEED)

    speaker_vectors = rng.normal(
        0.0,
        1.0,
        size=(NUM_SPEAKERS, FEATURE_DIM),
    )

    content_vectors = rng.normal(
        0.0,
        1.0,
        size=(NUM_CONTENTS, FEATURE_DIM),
    )

    # Speaker-specific transformation.
    speaker_scale = rng.uniform(
        0.7,
        1.3,
        size=(NUM_SPEAKERS, FEATURE_DIM),
    )

    # Content-specific transformation.
    content_scale = rng.uniform(
        0.6,
        1.2,
        size=(NUM_CONTENTS, FEATURE_DIM),
    )

    features = []
    speakers = []
    contents = []

    for speaker_id in range(NUM_SPEAKERS):
        for content_id in range(NUM_CONTENTS):

            for _ in range(SAMPLES_PER_COMBINATION):

                speaker_part = (
                    speaker_vectors[speaker_id]
                    * speaker_scale[speaker_id]
                )

                content_part = (
                    content_vectors[content_id]
                    * content_scale[content_id]
                )

                # Nonlinear interaction makes the problem less trivial.
                interaction = (
                    0.25
                    * np.tanh(
                        speaker_part * content_part
                    )
                )

                noise = rng.normal(
                    0.0,
                    0.20,
                    size=FEATURE_DIM,
                )

                x = (
                    speaker_part
                    + 0.8 * content_part
                    + interaction
                    + noise
                )

                features.append(x)
                speakers.append(speaker_id)
                contents.append(content_id)

    x = np.asarray(features, dtype=np.float32)

    speaker = np.asarray(
        speakers,
        dtype=np.int64,
    )

    content = np.asarray(
        contents,
        dtype=np.int64,
    )

    # Shuffle.
    indices = rng.permutation(len(x))

    x = x[indices]
    speaker = speaker[indices]
    content = content[indices]

    # Normalize feature dimensions.
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6

    x = (x - mean) / std

    return Dataset(
        x=torch.tensor(x),
        speaker=torch.tensor(speaker),
        content=torch.tensor(content),
    )


# ======================================================================
# TRAIN / TEST SPLIT
# ======================================================================

def split_dataset(
    dataset: Dataset,
) -> tuple[Dataset, Dataset]:

    n = len(dataset.x)

    split = int(n * TRAIN_RATIO)

    train = Dataset(
        x=dataset.x[:split],
        speaker=dataset.speaker[:split],
        content=dataset.content[:split],
    )

    test = Dataset(
        x=dataset.x[split:],
        speaker=dataset.speaker[split:],
        content=dataset.content[split:],
    )

    return train, test


# ======================================================================
# SPEAKER ENCODER
# ======================================================================

class SpeakerEncoder(nn.Module):
    """
    Encoder that attempts to represent speaker identity.

    The same architecture is used for:

        baseline encoder
        adversarial encoder
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.network(x)

        # Normalize embedding to make it similar to common
        # speaker-embedding systems.
        z = F.normalize(z, dim=-1)

        return z


# ======================================================================
# SPEAKER CLASSIFIER
# ======================================================================

class SpeakerClassifier(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_speakers: int,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            embed_dim,
            num_speakers,
        )

    def forward(self, z):
        return self.classifier(z)


# ======================================================================
# CONTENT CLASSIFIER
# ======================================================================

class ContentClassifier(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_contents: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, num_contents),
        )

    def forward(self, z):
        return self.network(z)


# ======================================================================
# GRADIENT REVERSAL
# ======================================================================

class GradientReversalFunction(
    torch.autograd.Function
):
    """
    Forward:

        identity

    Backward:

        multiply gradient by -lambda

    Therefore the content classifier learns normally, while the encoder
    receives the opposite gradient and tries to remove content information.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        lambda_value,
    ):
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        return (
            -ctx.lambda_value * grad_output,
            None,
        )


def gradient_reverse(
    x: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:

    return GradientReversalFunction.apply(
        x,
        lambda_value,
    )


# ======================================================================
# ADVERSARIAL MODEL
# ======================================================================

class AdversarialModel(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        num_speakers: int,
        num_contents: int,
    ):
        super().__init__()

        self.encoder = SpeakerEncoder(
            feature_dim,
            embed_dim,
        )

        self.speaker_classifier = SpeakerClassifier(
            embed_dim,
            num_speakers,
        )

        self.content_classifier = ContentClassifier(
            embed_dim,
            num_contents,
        )

    def forward(
        self,
        x,
        lambda_value,
    ):

        z = self.encoder(x)

        speaker_logits = self.speaker_classifier(z)

        reversed_z = gradient_reverse(
            z,
            lambda_value,
        )

        content_logits = self.content_classifier(
            reversed_z
        )

        return (
            z,
            speaker_logits,
            content_logits,
        )


# ======================================================================
# SIMPLE DATA LOADER
# ======================================================================

def batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
):

    n = len(dataset.x)

    if shuffle:
        indices = torch.randperm(n)
    else:
        indices = torch.arange(n)

    for start in range(
        0,
        n,
        batch_size,
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        yield (
            dataset.x[batch_indices].to(DEVICE),
            dataset.speaker[batch_indices].to(DEVICE),
            dataset.content[batch_indices].to(DEVICE),
        )


# ======================================================================
# ACCURACY
# ======================================================================

@torch.no_grad()
def accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> float:

    prediction = logits.argmax(dim=-1)

    return (
        (prediction == target)
        .float()
        .mean()
        .item()
    )


# ======================================================================
# TRAIN BASELINE ENCODER
# ======================================================================

def train_baseline(
    train: Dataset,
    test: Dataset,
):

    print()
    print("=" * 72)
    print("TRAINING BASELINE SPEAKER ENCODER")
    print("=" * 72)

    encoder = SpeakerEncoder(
        FEATURE_DIM,
        EMBED_DIM,
    ).to(DEVICE)

    speaker_classifier = SpeakerClassifier(
        EMBED_DIM,
        NUM_SPEAKERS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(speaker_classifier.parameters()),
        lr=ENCODER_LR,
    )

    history = []

    for epoch in range(1, ENCODER_EPOCHS + 1):

        encoder.train()
        speaker_classifier.train()

        total_loss = 0.0
        total_accuracy = 0.0
        count = 0

        for x, speaker, _ in batches(
            train,
            BATCH_SIZE,
        ):

            z = encoder(x)

            logits = speaker_classifier(z)

            loss = F.cross_entropy(
                logits,
                speaker,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(x)

            total_loss += (
                loss.item() * batch_size
            )

            total_accuracy += (
                accuracy(logits, speaker)
                * batch_size
            )

            count += batch_size

        train_acc = total_accuracy / count

        history.append(train_acc)

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == ENCODER_EPOCHS
        ):

            test_acc = evaluate_speaker_encoder(
                encoder,
                speaker_classifier,
                test,
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train speaker acc={train_acc:.3f} | "
                f"test speaker acc={test_acc:.3f}"
            )

    return encoder


# ======================================================================
# EVALUATE SPEAKER ENCODER
# ======================================================================

@torch.no_grad()
def evaluate_speaker_encoder(
    encoder,
    classifier,
    dataset,
):

    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    for x, speaker, _ in batches(
        dataset,
        BATCH_SIZE,
        shuffle=False,
    ):

        z = encoder(x)

        logits = classifier(z)

        prediction = logits.argmax(dim=-1)

        correct += (
            prediction == speaker
        ).sum().item()

        total += len(x)

    return correct / total


# ======================================================================
# TRAIN ADVERSARIAL ENCODER
# ======================================================================

def train_adversarial(
    train: Dataset,
    test: Dataset,
):

    print()
    print("=" * 72)
    print("TRAINING ADVERSARIAL SPEAKER ENCODER")
    print("=" * 72)

    model = AdversarialModel(
        FEATURE_DIM,
        EMBED_DIM,
        NUM_SPEAKERS,
        NUM_CONTENTS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ENCODER_LR,
    )

    # Strength of the adversarial signal.
    lambda_value = 1.0

    history = []

    for epoch in range(1, ENCODER_EPOCHS + 1):

        model.train()

        total_loss = 0.0
        total_speaker_acc = 0.0
        total_content_acc = 0.0
        count = 0

        for x, speaker, content in batches(
            train,
            BATCH_SIZE,
        ):

            (
                z,
                speaker_logits,
                content_logits,
            ) = model(
                x,
                lambda_value,
            )

            speaker_loss = F.cross_entropy(
                speaker_logits,
                speaker,
            )

            content_loss = F.cross_entropy(
                content_logits,
                content,
            )

            # The GRL already reverses the content gradient
            # going into the encoder.
            loss = (
                speaker_loss
                + content_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(x)

            total_loss += (
                loss.item() * batch_size
            )

            total_speaker_acc += (
                accuracy(
                    speaker_logits,
                    speaker,
                )
                * batch_size
            )

            total_content_acc += (
                accuracy(
                    content_logits,
                    content,
                )
                * batch_size
            )

            count += batch_size

        speaker_acc = (
            total_speaker_acc / count
        )

        content_acc = (
            total_content_acc / count
        )

        history.append(
            (
                speaker_acc,
                content_acc,
            )
        )

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == ENCODER_EPOCHS
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"speaker acc={speaker_acc:.3f} | "
                f"adversarial content acc={content_acc:.3f}"
            )

    return model.encoder


# ======================================================================
# FREEZE ENCODER
# ======================================================================

def freeze_encoder(
    encoder: nn.Module,
) -> None:

    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False


# ======================================================================
# EXTRACT EMBEDDINGS
# ======================================================================

@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    dataset: Dataset,
):

    encoder.eval()

    embeddings = []
    speakers = []
    contents = []

    for x, speaker, content in batches(
        dataset,
        BATCH_SIZE,
        shuffle=False,
    ):

        z = encoder(x)

        embeddings.append(
            z.cpu()
        )

        speakers.append(
            speaker.cpu()
        )

        contents.append(
            content.cpu()
        )

    return (
        torch.cat(embeddings),
        torch.cat(speakers),
        torch.cat(contents),
    )


# ======================================================================
# FRESH ATTACKER TRAINING
# ======================================================================

def train_fresh_attacker(
    train_embeddings: torch.Tensor,
    train_content: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_content: torch.Tensor,
    label: str,
):

    print()
    print("=" * 72)
    print(f"FRESH ATTACKER — {label}")
    print("=" * 72)

    attacker = ContentClassifier(
        EMBED_DIM,
        NUM_CONTENTS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        attacker.parameters(),
        lr=PROBE_LR,
    )

    train_embeddings = train_embeddings.to(DEVICE)
    train_content = train_content.to(DEVICE)

    test_embeddings = test_embeddings.to(DEVICE)
    test_content = test_content.to(DEVICE)

    history = []

    n = len(train_embeddings)

    for epoch in range(1, PROBE_EPOCHS + 1):

        attacker.train()

        indices = torch.randperm(
            n,
            device=DEVICE,
        )

        total_correct = 0
        total = 0
        total_loss = 0.0

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            z = train_embeddings[
                batch_indices
            ]

            target = train_content[
                batch_indices
            ]

            logits = attacker(z)

            loss = F.cross_entropy(
                logits,
                target,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * len(z)
            )

            prediction = logits.argmax(
                dim=-1
            )

            total_correct += (
                prediction == target
            ).sum().item()

            total += len(z)

        train_accuracy = (
            total_correct / total
        )

        test_accuracy = evaluate_attacker(
            attacker,
            test_embeddings,
            test_content,
        )

        history.append(
            test_accuracy
        )

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == PROBE_EPOCHS
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"train acc={train_accuracy:.3f} | "
                f"test acc={test_accuracy:.3f}"
            )

    return attacker, history


# ======================================================================
# EVALUATE ATTACKER
# ======================================================================

@torch.no_grad()
def evaluate_attacker(
    attacker,
    embeddings,
    content,
):

    attacker.eval()

    logits = attacker(
        embeddings
    )

    return accuracy(
        logits,
        content,
    )


# ======================================================================
# MAIN EXPERIMENT
# ======================================================================

def main():

    set_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("=" * 72)
    print("PHASE 1 — LESSON 27")
    print("FRESH ATTACKER / POST-HOC CONTENT PROBE")
    print("=" * 72)

    print()
    print(f"Device: {DEVICE}")
    print(f"Speakers: {NUM_SPEAKERS}")
    print(f"Content classes: {NUM_CONTENTS}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Embedding dimension: {EMBED_DIM}")

    # --------------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 1 — CREATE SYNTHETIC SPEECH-LIKE DATA")
    print("=" * 72)

    dataset = make_dataset()

    train, test = split_dataset(
        dataset
    )

    print(
        f"Total samples: {len(dataset.x)}"
    )

    print(
        f"Train samples: {len(train.x)}"
    )

    print(
        f"Test samples:  {len(test.x)}"
    )

    # --------------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 2 — TRAIN BASELINE SPEAKER ENCODER")
    print("=" * 72)

    baseline_encoder = train_baseline(
        train,
        test,
    )

    # --------------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 3 — TRAIN ADVERSARIAL SPEAKER ENCODER")
    print("=" * 72)

    adversarial_encoder = train_adversarial(
        train,
        test,
    )

    # --------------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 4 — FREEZE BOTH ENCODERS")
    print("=" * 72)

    freeze_encoder(
        baseline_encoder
    )

    freeze_encoder(
        adversarial_encoder
    )

    print(
        "Both speaker encoders are now frozen."
    )

    # --------------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 5 — EXTRACT FROZEN EMBEDDINGS")
    print("=" * 72)

    (
        baseline_train_z,
        baseline_train_speaker,
        baseline_train_content,
    ) = extract_embeddings(
        baseline_encoder,
        train,
    )

    (
        baseline_test_z,
        baseline_test_speaker,
        baseline_test_content,
    ) = extract_embeddings(
        baseline_encoder,
        test,
    )

    (
        adversarial_train_z,
        adversarial_train_speaker,
        adversarial_train_content,
    ) = extract_embeddings(
        adversarial_encoder,
        train,
    )

    (
        adversarial_test_z,
        adversarial_test_speaker,
        adversarial_test_content,
    ) = extract_embeddings(
        adversarial_encoder,
        test,
    )

    print(
        "Baseline embedding shape:",
        tuple(baseline_train_z.shape),
    )

    print(
        "Adversarial embedding shape:",
        tuple(adversarial_train_z.shape),
    )

    # --------------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 6 — TRAIN BRAND-NEW ATTACKER ON BASELINE")
    print("=" * 72)

    baseline_attacker, baseline_history = (
        train_fresh_attacker(
            baseline_train_z,
            baseline_train_content,
            baseline_test_z,
            baseline_test_content,
            "BASELINE EMBEDDING",
        )
    )

    # --------------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 7 — TRAIN BRAND-NEW ATTACKER ON ADVERSARIAL")
    print("=" * 72)

    adversarial_attacker, adversarial_history = (
        train_fresh_attacker(
            adversarial_train_z,
            adversarial_train_content,
            adversarial_test_z,
            adversarial_test_content,
            "ADVERSARIAL EMBEDDING",
        )
    )

    # --------------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------------

    baseline_final = baseline_history[-1]
    adversarial_final = adversarial_history[-1]

    chance_accuracy = 1.0 / NUM_CONTENTS

    print()
    print("=" * 72)
    print("STEP 8 — FINAL POST-HOC PROBE RESULTS")
    print("=" * 72)

    print()
    print(
        f"Random-guess accuracy:       "
        f"{chance_accuracy:.3f}"
    )

    print(
        f"Baseline embedding leakage:  "
        f"{baseline_final:.3f}"
    )

    print(
        f"Adversarial embedding leakage:"
        f" {adversarial_final:.3f}"
    )

    improvement = (
        baseline_final
        - adversarial_final
    )

    print(
        f"Leakage reduction:            "
        f"{improvement:.3f}"
    )

    # --------------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 9 — SAVE PROBE MODELS")
    print("=" * 72)

    torch.save(
        baseline_attacker.state_dict(),
        os.path.join(
            OUTPUT_DIR,
            "baseline_probe.pt",
        ),
    )

    torch.save(
        adversarial_attacker.state_dict(),
        os.path.join(
            OUTPUT_DIR,
            "adversarial_probe.pt",
        ),
    )

    print(
        f"Saved models to: {OUTPUT_DIR}"
    )

    # --------------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 10 — VISUALIZE POST-HOC LEAKAGE")
    print("=" * 72)

    epochs = np.arange(
        1,
        PROBE_EPOCHS + 1,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        baseline_history,
        label="Baseline embedding",
        linewidth=2,
    )

    plt.plot(
        epochs,
        adversarial_history,
        label="Adversarial embedding",
        linewidth=2,
    )

    plt.axhline(
        chance_accuracy,
        linestyle="--",
        label="Random chance",
    )

    plt.xlabel(
        "Fresh attacker training epoch"
    )

    plt.ylabel(
        "Content classification accuracy"
    )

    plt.title(
        "Post-Hoc Linguistic Information Probe"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "fresh_probe_results.png",
    )

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.show()

    print()
    print(
        f"Saved plot to: {plot_path}"
    )

    # --------------------------------------------------------------
    # FINAL INTERPRETATION
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("LESSON 27 INTERPRETATION")
    print("=" * 72)

    if adversarial_final < baseline_final:

        print(
            "The adversarial encoder reduced the amount of "
            "content information recoverable by a fresh attacker."
        )

    else:

        print(
            "The fresh attacker still recovers substantial "
            "content information from the adversarial embedding."
        )

    if adversarial_final <= (
        chance_accuracy + 0.10
    ):

        print(
            "The adversarial embedding is close to the "
            "random-guessing level for the fresh probe."
        )

    else:

        print(
            "Content information remains detectable above "
            "the random-guessing baseline."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A low content-probe accuracy does NOT mean the "
        "embedding contains zero linguistic information."
    )

    print(
        "It means that the information is difficult for this "
        "attacker architecture to recover."
    )

    print()
    print(
        "This is why stronger probes and multiple probe "
        "architectures can be useful in future experiments."
    )

    print()
    print("=" * 72)
    print("LESSON 27 COMPLETE")
    print("=" * 72)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()