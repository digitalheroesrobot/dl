"""
======================================================================
PHASE 1 — LESSON 27.1
CONTROLLED LINGUISTIC LEAKAGE + FRESH ATTACKER
======================================================================

PURPOSE
-------
Fix the experimental weakness discovered in Lesson 27.

Lesson 27 showed that the fresh attacker could not recover content
information from either embedding. Therefore, the experiment could
not demonstrate whether adversarial training actually removed
linguistic information.

This lesson deliberately constructs a representation where:

    BASELINE:
        speaker information = strong
        content information = strong

    ADVERSARIAL:
        speaker information = strong
        content information = weak

This gives us a controlled experiment.

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1
------
Construct synthetic acoustic features containing independent speaker
and linguistic/content factors.

STEP 2
------
Train a baseline encoder that explicitly predicts BOTH:

    speaker identity
    content identity

Therefore the baseline embedding must retain both factors.

STEP 3
------
Train an adversarial encoder that:

    maximizes speaker classification
    minimizes recoverable content information

using Gradient Reversal.

STEP 4
------
Freeze both encoders.

STEP 5
------
Create a completely NEW attacker network.

The attacker:

    - has random initialization
    - was not used during encoder training
    - cannot modify the frozen encoder

STEP 6
------
Train several fresh attackers with different random seeds.

STEP 7
------
Measure:

    speaker accuracy
    content accuracy
    content leakage reduction

STEP 8
------
Visualize the difference.

----------------------------------------------------------------------
EXPECTED RESULT
----------------------------------------------------------------------

A successful experiment should approximately show:

                    Speaker       Content
                    accuracy      accuracy
------------------------------------------------
Baseline             high          high
Adversarial          high          near chance

For example:

                    Speaker       Content
------------------------------------------------
Baseline              98%           90%
Adversarial           97%           10%
Chance                12.5%          6.25%

Exact values will vary.

The important relationship is:

    adversarial content accuracy
          <
    baseline content accuracy

while:

    adversarial speaker accuracy
          remains high.

----------------------------------------------------------------------
WHY THIS IS STRONGER
----------------------------------------------------------------------

We are not asking:

    "Did the original adversary fail?"

We are asking:

    "After the encoder was frozen, can a completely new model
     recover content information from its representation?"

This is a post-hoc probing experiment.

----------------------------------------------------------------------
REAL-WORLD CONNECTION
----------------------------------------------------------------------

In a voice-cloning system, we want the speaker embedding to answer:

    WHO is speaking?

but not strongly encode:

    WHAT are they saying?

For example:

    "The robot picked up the cup."

and:

    "The robot opened the door."

should produce similar speaker representations when spoken
by the same person.

A fresh linguistic probe should therefore have difficulty
determining the sentence/content from the speaker embedding.

----------------------------------------------------------------------
DEPENDENCIES
----------------------------------------------------------------------

    pip install torch numpy matplotlib

----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

outputs/lesson27_1/

    controlled_leakage_results.png
    baseline_encoder.pt
    adversarial_encoder.pt

======================================================================
"""

from __future__ import annotations

import os
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = "outputs/lesson27_1"

# ----------------------------------------------------------------------
# Synthetic speech representation
# ----------------------------------------------------------------------

NUM_SPEAKERS = 8
NUM_CONTENTS = 16

FEATURE_DIM = 32
EMBED_DIM = 16

SAMPLES_PER_COMBINATION = 20

TRAIN_RATIO = 0.8

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

BATCH_SIZE = 128

ENCODER_EPOCHS = 120
PROBE_EPOCHS = 100

ENCODER_LR = 1e-3
PROBE_LR = 1e-3

PRINT_EVERY = 10

# ----------------------------------------------------------------------
# Multiple independent fresh attackers
# ----------------------------------------------------------------------

PROBE_SEEDS = [
    101,
    202,
    303,
    404,
    505,
]


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# DATASET
# ======================================================================

class Dataset:

    def __init__(
        self,
        x,
        speaker,
        content,
    ):

        self.x = x
        self.speaker = speaker
        self.content = content

    def __len__(self):

        return len(self.x)


# ======================================================================
# CREATE CONTROLLED DATA
# ======================================================================

def make_dataset():

    """
    Create a synthetic acoustic feature space.

    Crucially, speaker and content have comparable strength.

    We intentionally make content highly recoverable.

    This allows us to test whether adversarial training actually
    suppresses content information.
    """

    rng = np.random.default_rng(SEED)

    # --------------------------------------------------------------
    # Speaker latent vectors
    # --------------------------------------------------------------

    speaker_vectors = rng.normal(
        0.0,
        1.0,
        size=(
            NUM_SPEAKERS,
            FEATURE_DIM,
        ),
    )

    # --------------------------------------------------------------
    # Content latent vectors
    # --------------------------------------------------------------

    content_vectors = rng.normal(
        0.0,
        1.0,
        size=(
            NUM_CONTENTS,
            FEATURE_DIM,
        ),
    )

    features = []
    speakers = []
    contents = []

    for speaker_id in range(
        NUM_SPEAKERS
    ):

        for content_id in range(
            NUM_CONTENTS
        ):

            speaker = speaker_vectors[
                speaker_id
            ]

            content = content_vectors[
                content_id
            ]

            for _ in range(
                SAMPLES_PER_COMBINATION
            ):

                # --------------------------------------------------
                # Important:
                #
                # speaker and content have similar magnitude.
                #
                # Therefore BOTH are strongly encoded in x.
                # --------------------------------------------------

                noise = rng.normal(
                    0.0,
                    0.10,
                    size=FEATURE_DIM,
                )

                x = (
                    1.0 * speaker
                    + 1.0 * content
                    + noise
                )

                features.append(x)
                speakers.append(speaker_id)
                contents.append(content_id)

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    speakers = np.asarray(
        speakers,
        dtype=np.int64,
    )

    contents = np.asarray(
        contents,
        dtype=np.int64,
    )

    # --------------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------------

    indices = rng.permutation(
        len(features)
    )

    features = features[indices]
    speakers = speakers[indices]
    contents = contents[indices]

    # --------------------------------------------------------------
    # Normalize
    # --------------------------------------------------------------

    mean = features.mean(
        axis=0,
        keepdims=True,
    )

    std = features.std(
        axis=0,
        keepdims=True,
    ) + 1e-6

    features = (
        features - mean
    ) / std

    return Dataset(
        torch.tensor(features),
        torch.tensor(speakers),
        torch.tensor(contents),
    )


# ======================================================================
# TRAIN / TEST SPLIT
# ======================================================================

def split_dataset(dataset):

    n = len(dataset)

    split = int(
        n * TRAIN_RATIO
    )

    train = Dataset(
        dataset.x[:split],
        dataset.speaker[:split],
        dataset.content[:split],
    )

    test = Dataset(
        dataset.x[split:],
        dataset.speaker[split:],
        dataset.content[split:],
    )

    return train, test


# ======================================================================
# BATCH GENERATOR
# ======================================================================

def batches(
    dataset,
    batch_size,
    shuffle=True,
):

    n = len(dataset)

    if shuffle:

        indices = torch.randperm(n)

    else:

        indices = torch.arange(n)

    for start in range(
        0,
        n,
        batch_size,
    ):

        idx = indices[
            start:start + batch_size
        ]

        yield (
            dataset.x[idx].to(DEVICE),
            dataset.speaker[idx].to(DEVICE),
            dataset.content[idx].to(DEVICE),
        )


# ======================================================================
# SPEAKER ENCODER
# ======================================================================

class SpeakerEncoder(nn.Module):

    def __init__(
        self,
        feature_dim,
        embed_dim,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                feature_dim,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                embed_dim,
            ),
        )

    def forward(self, x):

        z = self.network(x)

        return F.normalize(
            z,
            dim=-1,
        )


# ======================================================================
# CLASSIFIERS
# ======================================================================

class SpeakerClassifier(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_speakers,
    ):

        super().__init__()

        self.network = nn.Linear(
            embed_dim,
            num_speakers,
        )

    def forward(self, z):

        return self.network(z)


class ContentClassifier(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_contents,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embed_dim,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                num_contents,
            ),
        )

    def forward(self, z):

        return self.network(z)


# ======================================================================
# GRADIENT REVERSAL
# ======================================================================

class GradientReversalFunction(
    torch.autograd.Function
):

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
    x,
    lambda_value,
):

    return GradientReversalFunction.apply(
        x,
        lambda_value,
    )


# ======================================================================
# BASELINE MODEL
# ======================================================================

class BaselineModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = SpeakerEncoder(
            FEATURE_DIM,
            EMBED_DIM,
        )

        self.speaker_classifier = (
            SpeakerClassifier(
                EMBED_DIM,
                NUM_SPEAKERS,
            )
        )

        self.content_classifier = (
            ContentClassifier(
                EMBED_DIM,
                NUM_CONTENTS,
            )
        )

    def forward(self, x):

        z = self.encoder(x)

        speaker_logits = (
            self.speaker_classifier(z)
        )

        content_logits = (
            self.content_classifier(z)
        )

        return (
            z,
            speaker_logits,
            content_logits,
        )


# ======================================================================
# ADVERSARIAL MODEL
# ======================================================================

class AdversarialModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = SpeakerEncoder(
            FEATURE_DIM,
            EMBED_DIM,
        )

        self.speaker_classifier = (
            SpeakerClassifier(
                EMBED_DIM,
                NUM_SPEAKERS,
            )
        )

        self.content_classifier = (
            ContentClassifier(
                EMBED_DIM,
                NUM_CONTENTS,
            )
        )

    def forward(
        self,
        x,
        lambda_value,
    ):

        z = self.encoder(x)

        speaker_logits = (
            self.speaker_classifier(z)
        )

        reversed_z = gradient_reverse(
            z,
            lambda_value,
        )

        content_logits = (
            self.content_classifier(
                reversed_z
            )
        )

        return (
            z,
            speaker_logits,
            content_logits,
        )


# ======================================================================
# ACCURACY
# ======================================================================

def accuracy(
    logits,
    target,
):

    prediction = logits.argmax(
        dim=-1
    )

    return (
        prediction == target
    ).float().mean().item()


# ======================================================================
# BASELINE TRAINING
# ======================================================================

def train_baseline(
    train,
    test,
):

    print()
    print("=" * 72)
    print("TRAINING BASELINE ENCODER")
    print("=" * 72)

    model = BaselineModel().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ENCODER_LR,
    )

    for epoch in range(
        1,
        ENCODER_EPOCHS + 1,
    ):

        model.train()

        total_speaker = 0.0
        total_content = 0.0
        total = 0

        for (
            x,
            speaker,
            content,
        ) in batches(
            train,
            BATCH_SIZE,
        ):

            (
                z,
                speaker_logits,
                content_logits,
            ) = model(x)

            speaker_loss = (
                F.cross_entropy(
                    speaker_logits,
                    speaker,
                )
            )

            content_loss = (
                F.cross_entropy(
                    content_logits,
                    content,
                )
            )

            # IMPORTANT:
            #
            # Both speaker AND content are explicitly optimized.
            #
            # Therefore the baseline representation is encouraged
            # to retain both factors.

            loss = (
                speaker_loss
                + content_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_size = len(x)

            total_speaker += (
                accuracy(
                    speaker_logits,
                    speaker,
                )
                * batch_size
            )

            total_content += (
                accuracy(
                    content_logits,
                    content,
                )
                * batch_size
            )

            total += batch_size

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == ENCODER_EPOCHS
        ):

            test_speaker, test_content = (
                evaluate_encoder(
                    model,
                    test,
                )
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train speaker={total_speaker / total:.3f} | "
                f"train content={total_content / total:.3f} | "
                f"test speaker={test_speaker:.3f} | "
                f"test content={test_content:.3f}"
            )

    return model


# ======================================================================
# ADVERSARIAL TRAINING
# ======================================================================

def train_adversarial(
    train,
    test,
):

    print()
    print("=" * 72)
    print("TRAINING ADVERSARIAL ENCODER")
    print("=" * 72)

    model = AdversarialModel().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ENCODER_LR,
    )

    # --------------------------------------------------------------
    # Adversarial strength
    # --------------------------------------------------------------

    lambda_value = 1.0

    for epoch in range(
        1,
        ENCODER_EPOCHS + 1,
    ):

        model.train()

        total_speaker = 0.0
        total_content = 0.0
        total = 0

        for (
            x,
            speaker,
            content,
        ) in batches(
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

            speaker_loss = (
                F.cross_entropy(
                    speaker_logits,
                    speaker,
                )
            )

            content_loss = (
                F.cross_entropy(
                    content_logits,
                    content,
                )
            )

            # The gradient reversal layer causes the encoder to
            # maximize content classification loss while the
            # content classifier itself minimizes it.

            loss = (
                speaker_loss
                + content_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_size = len(x)

            total_speaker += (
                accuracy(
                    speaker_logits,
                    speaker,
                )
                * batch_size
            )

            total_content += (
                accuracy(
                    content_logits,
                    content,
                )
                * batch_size
            )

            total += batch_size

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == ENCODER_EPOCHS
        ):

            (
                test_speaker,
                test_content,
            ) = evaluate_encoder(
                model,
                test,
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train speaker={total_speaker / total:.3f} | "
                f"train content={total_content / total:.3f} | "
                f"test speaker={test_speaker:.3f} | "
                f"test content={test_content:.3f}"
            )

    return model


# ======================================================================
# ENCODER EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate_encoder(
    model,
    dataset,
):

    model.eval()

    speaker_correct = 0
    content_correct = 0
    total = 0

    for (
        x,
        speaker,
        content,
    ) in batches(
        dataset,
        BATCH_SIZE,
        shuffle=False,
    ):

        if isinstance(
            model,
            AdversarialModel,
        ):

            (
                z,
                speaker_logits,
                content_logits,
            ) = model(
                x,
                1.0,
            )

        else:

            (
                z,
                speaker_logits,
                content_logits,
            ) = model(x)

        speaker_correct += (
            speaker_logits.argmax(
                dim=-1
            )
            == speaker
        ).sum().item()

        content_correct += (
            content_logits.argmax(
                dim=-1
            )
            == content
        ).sum().item()

        total += len(x)

    return (
        speaker_correct / total,
        content_correct / total,
    )


# ======================================================================
# FREEZE ENCODER
# ======================================================================

def freeze_encoder(
    encoder,
):

    encoder.eval()

    for parameter in encoder.parameters():

        parameter.requires_grad = False


# ======================================================================
# EXTRACT FROZEN EMBEDDINGS
# ======================================================================

@torch.no_grad()
def extract_embeddings(
    encoder,
    dataset,
):

    encoder.eval()

    embeddings = []
    speakers = []
    contents = []

    for (
        x,
        speaker,
        content,
    ) in batches(
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
# FRESH ATTACKER
# ======================================================================

def train_fresh_attacker(
    train_z,
    train_content,
    test_z,
    test_content,
    seed,
):

    set_seed(seed)

    attacker = ContentClassifier(
        EMBED_DIM,
        NUM_CONTENTS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        attacker.parameters(),
        lr=PROBE_LR,
    )

    train_z = train_z.to(DEVICE)
    train_content = train_content.to(
        DEVICE
    )

    test_z = test_z.to(DEVICE)
    test_content = test_content.to(
        DEVICE
    )

    best_test_accuracy = 0.0

    n = len(train_z)

    for epoch in range(
        1,
        PROBE_EPOCHS + 1,
    ):

        attacker.train()

        indices = torch.randperm(
            n,
            device=DEVICE,
        )

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):

            idx = indices[
                start:start + BATCH_SIZE
            ]

            z = train_z[idx]

            target = train_content[idx]

            logits = attacker(z)

            loss = F.cross_entropy(
                logits,
                target,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        test_accuracy = evaluate_attacker(
            attacker,
            test_z,
            test_content,
        )

        best_test_accuracy = max(
            best_test_accuracy,
            test_accuracy,
        )

    return best_test_accuracy


# ======================================================================
# ATTACKER EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate_attacker(
    attacker,
    z,
    content,
):

    attacker.eval()

    logits = attacker(z)

    return accuracy(
        logits,
        content,
    )


# ======================================================================
# RUN MULTIPLE PROBES
# ======================================================================

def run_probes(
    name,
    train_z,
    train_content,
    test_z,
    test_content,
):

    print()
    print("=" * 72)
    print(
        f"FRESH ATTACKER PROBES — {name}"
    )
    print("=" * 72)

    results = []

    for seed in PROBE_SEEDS:

        result = train_fresh_attacker(
            train_z,
            train_content,
            test_z,
            test_content,
            seed,
        )

        results.append(result)

        print(
            f"Probe seed {seed}: "
            f"best test accuracy={result:.3f}"
        )

    results = np.asarray(
        results
    )

    print()
    print(
        f"{name} mean: "
        f"{results.mean():.3f}"
    )

    print(
        f"{name} std:  "
        f"{results.std():.3f}"
    )

    return results


# ======================================================================
# MAIN
# ======================================================================

def main():

    set_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("=" * 72)
    print("PHASE 1 — LESSON 27.1")
    print("CONTROLLED LINGUISTIC LEAKAGE")
    print("=" * 72)

    print()
    print(f"Device: {DEVICE}")

    print(
        f"Speakers: {NUM_SPEAKERS}"
    )

    print(
        f"Content classes: {NUM_CONTENTS}"
    )

    print(
        f"Random content chance: "
        f"{1.0 / NUM_CONTENTS:.3f}"
    )

    # ==============================================================
    # STEP 1
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 1 — CREATE CONTROLLED DATASET")
    print("=" * 72)

    dataset = make_dataset()

    train, test = split_dataset(
        dataset
    )

    print(
        f"Total samples: {len(dataset)}"
    )

    print(
        f"Training samples: {len(train)}"
    )

    print(
        f"Test samples: {len(test)}"
    )

    # ==============================================================
    # STEP 2
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 2 — TRAIN BASELINE")
    print("=" * 72)

    baseline = train_baseline(
        train,
        test,
    )

    # ==============================================================
    # STEP 3
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 3 — TRAIN ADVERSARIAL MODEL")
    print("=" * 72)

    adversarial = train_adversarial(
        train,
        test,
    )

    # ==============================================================
    # STEP 4
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 4 — FREEZE ENCODERS")
    print("=" * 72)

    freeze_encoder(
        baseline.encoder
    )

    freeze_encoder(
        adversarial.encoder
    )

    print(
        "Baseline encoder frozen."
    )

    print(
        "Adversarial encoder frozen."
    )

    # ==============================================================
    # STEP 5
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 5 — EXTRACT EMBEDDINGS")
    print("=" * 72)

    (
        baseline_train_z,
        baseline_train_speaker,
        baseline_train_content,
    ) = extract_embeddings(
        baseline.encoder,
        train,
    )

    (
        baseline_test_z,
        baseline_test_speaker,
        baseline_test_content,
    ) = extract_embeddings(
        baseline.encoder,
        test,
    )

    (
        adversarial_train_z,
        adversarial_train_speaker,
        adversarial_train_content,
    ) = extract_embeddings(
        adversarial.encoder,
        train,
    )

    (
        adversarial_test_z,
        adversarial_test_speaker,
        adversarial_test_content,
    ) = extract_embeddings(
        adversarial.encoder,
        test,
    )

    print(
        "Baseline embedding:",
        tuple(baseline_train_z.shape),
    )

    print(
        "Adversarial embedding:",
        tuple(adversarial_train_z.shape),
    )

    # ==============================================================
    # STEP 6
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 6 — FRESH ATTACKER ON BASELINE")
    print("=" * 72)

    baseline_results = run_probes(
        "BASELINE",
        baseline_train_z,
        baseline_train_content,
        baseline_test_z,
        baseline_test_content,
    )

    # ==============================================================
    # STEP 7
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 7 — FRESH ATTACKER ON ADVERSARIAL")
    print("=" * 72)

    adversarial_results = run_probes(
        "ADVERSARIAL",
        adversarial_train_z,
        adversarial_train_content,
        adversarial_test_z,
        adversarial_test_content,
    )

    # ==============================================================
    # STEP 8
    # ==============================================================

    baseline_mean = baseline_results.mean()
    adversarial_mean = (
        adversarial_results.mean()
    )

    baseline_std = baseline_results.std()
    adversarial_std = (
        adversarial_results.std()
    )

    chance = 1.0 / NUM_CONTENTS

    absolute_reduction = (
        baseline_mean
        - adversarial_mean
    )

    relative_reduction = (
        absolute_reduction
        / max(
            baseline_mean - chance,
            1e-8,
        )
    )

    # ==============================================================
    # RESULTS
    # ==============================================================

    print()
    print("=" * 72)
    print("FINAL POST-HOC PROBE RESULTS")
    print("=" * 72)

    print()

    print(
        f"Random chance:              "
        f"{chance:.3f}"
    )

    print(
        f"Baseline probe:             "
        f"{baseline_mean:.3f} ± {baseline_std:.3f}"
    )

    print(
        f"Adversarial probe:          "
        f"{adversarial_mean:.3f} ± {adversarial_std:.3f}"
    )

    print(
        f"Absolute leakage reduction:"
        f" {absolute_reduction:.3f}"
    )

    print(
        f"Relative leakage reduction:"
        f" {relative_reduction * 100:.1f}%"
    )

    # ==============================================================
    # STEP 9 — VISUALIZATION
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 9 — VISUALIZE RESULTS")
    print("=" * 72)

    labels = [
        "Random\nchance",
        "Baseline",
        "Adversarial",
    ]

    means = [
        chance,
        baseline_mean,
        adversarial_mean,
    ]

    errors = [
        0.0,
        baseline_std,
        adversarial_std,
    ]

    plt.figure(
        figsize=(9, 6)
    )

    x = np.arange(
        len(labels)
    )

    plt.bar(
        x,
        means,
        yerr=errors,
        capsize=6,
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Fresh attacker content accuracy"
    )

    plt.title(
        "Post-Hoc Linguistic Leakage Probe"
    )

    plt.ylim(
        0,
        max(
            1.0,
            max(means) + 0.1,
        ),
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "controlled_leakage_results.png",
    )

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.show()

    print(
        f"Saved plot: {plot_path}"
    )

    # ==============================================================
    # STEP 10 — SAVE ENCODERS
    # ==============================================================

    torch.save(
        baseline.encoder.state_dict(),
        os.path.join(
            OUTPUT_DIR,
            "baseline_encoder.pt",
        ),
    )

    torch.save(
        adversarial.encoder.state_dict(),
        os.path.join(
            OUTPUT_DIR,
            "adversarial_encoder.pt",
        ),
    )

    # ==============================================================
    # FINAL INTERPRETATION
    # ==============================================================

    print()
    print("=" * 72)
    print("LESSON 27.1 INTERPRETATION")
    print("=" * 72)

    if (
        baseline_mean
        > chance + 0.20
    ):

        print(
            "PASS: The baseline embedding contains "
            "measurable content information."
        )

    else:

        print(
            "WARNING: Baseline content leakage is weaker "
            "than expected."
        )

    if (
        adversarial_mean
        < baseline_mean
    ):

        print(
            "PASS: Adversarial training reduced "
            "post-hoc recoverable content information."
        )

    else:

        print(
            "WARNING: Adversarial training did not "
            "reduce fresh-probe content accuracy."
        )

    print()

    if (
        adversarial_mean
        <= chance + 0.10
    ):

        print(
            "The adversarial embedding is close to "
            "the random-guessing level."
        )

    else:

        print(
            "Some content information remains "
            "recoverable from the adversarial embedding."
        )

    print()
    print(
        "The most important condition is:"
    )

    print(
        "speaker information remains high while "
        "content information decreases."
    )

    print()
    print("=" * 72)
    print("LESSON 27.1 COMPLETE")
    print("=" * 72)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()