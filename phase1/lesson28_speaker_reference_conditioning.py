"""
======================================================================
PHASE 1 — LESSON 28
SPEAKER REFERENCE CONDITIONING
======================================================================

PURPOSE
-------
Build the first miniature voice-cloning architecture in the course.

The goal is to explicitly separate:

    WHO is speaking
        from
    WHAT is being spoken.

This is the fundamental idea behind speaker-conditioned TTS systems.

----------------------------------------------------------------------
ARCHITECTURE
----------------------------------------------------------------------

Reference acoustic features
        |
        v
+-----------------------+
| Speaker Encoder       |
|                       |
| acoustic reference    |
+-----------+-----------+
            |
            v
      speaker embedding
            |
            |
            +-------------------+
                                |
                                v
Text/content ----------------> Generator
                                |
                                v
                         acoustic features


Mathematically:

    z_s = E_speaker(reference)

    z_t = E_text(content)

    x_hat = G(z_t, z_s)


where:

    z_s = speaker identity
    z_t = linguistic/content identity
    x_hat = generated acoustic representation


----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1
------
Create synthetic speech-like data with two independent factors:

    speaker
    content

STEP 2
------
Train a speaker encoder.

The encoder learns:

    acoustic reference -> speaker embedding

STEP 3
------
Create a content embedding.

This represents the linguistic identity of the utterance.

STEP 4
------
Build a conditional acoustic generator.

The generator receives both:

    content embedding
    speaker embedding

STEP 5
------
Train the generator.

The objective is:

    generated acoustic representation
        ≈
    target acoustic representation

STEP 6
------
Perform speaker transfer.

Keep content fixed.

Change speaker.

Expected:

    content remains similar
    speaker characteristics change

STEP 7
------
Perform content transfer.

Keep speaker fixed.

Change content.

Expected:

    speaker characteristics remain similar
    linguistic/acoustic content changes

STEP 8
------
Visualize the resulting acoustic representations.

----------------------------------------------------------------------
WHY WE USE SYNTHETIC FEATURES
----------------------------------------------------------------------

We are not training a real TTS system yet.

That would require:

    waveform/audio loading
    mel spectrograms
    text tokenization
    duration alignment
    acoustic modeling
    vocoder
    possibly phoneme/semantic representations

Instead, this lesson isolates one architectural principle:

    speaker conditioning.

Once this principle works, later lessons can replace the synthetic
components with real speech representations.

----------------------------------------------------------------------
REAL-WORLD EXAMPLE
----------------------------------------------------------------------

Suppose reference speech is:

    Speaker A:
    "Hello, how are you?"

We extract:

    speaker_embedding_A

Now we want Speaker A to say:

    "The robot picked up the cup."

The generator receives:

    text = "The robot picked up the cup."
    speaker = speaker_embedding_A

The linguistic information comes from the text.

The identity information comes from the reference speaker embedding.

----------------------------------------------------------------------
IMPORTANT LIMITATION
----------------------------------------------------------------------

This lesson does NOT produce real audio.

It produces a synthetic acoustic representation.

The purpose is to understand the conditioning mechanism before
introducing real mel spectrograms and neural vocoders.

----------------------------------------------------------------------
DEPENDENCIES
----------------------------------------------------------------------

    pip install torch numpy matplotlib

----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

outputs/lesson28/

    speaker_conditioning_results.png
    lesson28_model.pt

----------------------------------------------------------------------
NEXT
----------------------------------------------------------------------

Lesson 29 will introduce real mel-spectrogram-style representations
and move the generator closer to an actual acoustic TTS model.

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

OUTPUT_DIR = "outputs/lesson28"

# ----------------------------------------------------------------------
# Synthetic speech factors
# ----------------------------------------------------------------------

NUM_SPEAKERS = 8
NUM_CONTENTS = 20

FEATURE_DIM = 32
SPEAKER_EMBED_DIM = 16
CONTENT_EMBED_DIM = 16

SAMPLES_PER_COMBINATION = 20

# ----------------------------------------------------------------------
# Acoustic sequence
# ----------------------------------------------------------------------

ACOUSTIC_DIM = 32
ACOUSTIC_FRAMES = 16

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

BATCH_SIZE = 128

SPEAKER_EPOCHS = 100
GENERATOR_EPOCHS = 200

LEARNING_RATE = 1e-3

PRINT_EVERY = 20


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

class SpeechDataset:

    def __init__(
        self,
        reference,
        content,
        acoustic,
        speaker,
    ):

        self.reference = reference
        self.content = content
        self.acoustic = acoustic
        self.speaker = speaker

    def __len__(self):

        return len(self.reference)


# ======================================================================
# CREATE SYNTHETIC SPEECH DATA
# ======================================================================

def create_dataset():

    """
    Create a synthetic speech-like dataset.

    Every sample contains:

        reference acoustic features
        content identity
        target acoustic representation
        speaker identity

    The target acoustic representation is deliberately constructed
    from BOTH speaker and content factors.

    This gives the generator a meaningful conditioning problem.
    """

    rng = np.random.default_rng(SEED)

    # --------------------------------------------------------------
    # Speaker identity vectors
    # --------------------------------------------------------------

    speaker_vectors = rng.normal(
        0.0,
        1.0,
        size=(
            NUM_SPEAKERS,
            ACOUSTIC_DIM,
        ),
    )

    # --------------------------------------------------------------
    # Content identity vectors
    # --------------------------------------------------------------

    content_vectors = rng.normal(
        0.0,
        1.0,
        size=(
            NUM_CONTENTS,
            ACOUSTIC_DIM,
        ),
    )

    references = []
    contents = []
    acoustics = []
    speakers = []

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
                # Reference speech
                #
                # Reference should contain strong speaker
                # information and some nuisance variation.
                # --------------------------------------------------

                reference_noise = rng.normal(
                    0.0,
                    0.15,
                    size=ACOUSTIC_DIM,
                )

                reference = (
                    speaker
                    + reference_noise
                )

                # --------------------------------------------------
                # Target acoustic representation
                #
                # Both speaker and content contribute.
                # --------------------------------------------------

                target = np.zeros(
                    (
                        ACOUSTIC_FRAMES,
                        ACOUSTIC_DIM,
                    ),
                    dtype=np.float32,
                )

                for frame in range(
                    ACOUSTIC_FRAMES
                ):

                    phase = (
                        2.0
                        * np.pi
                        * frame
                        / ACOUSTIC_FRAMES
                    )

                    # Speaker characteristics remain relatively
                    # stable across the sequence.

                    speaker_component = (
                        0.8
                        * speaker
                    )

                    # Content changes across acoustic frames.

                    content_component = (
                        0.8
                        * content
                        * (
                            0.7
                            + 0.3
                            * np.sin(phase)
                        )
                    )

                    # Small temporal variation.

                    temporal = (
                        0.1
                        * np.sin(
                            phase
                            * np.arange(
                                ACOUSTIC_DIM
                            )
                        )
                    )

                    noise = rng.normal(
                        0.0,
                        0.05,
                        size=ACOUSTIC_DIM,
                    )

                    target[frame] = (
                        speaker_component
                        + content_component
                        + temporal
                        + noise
                    )

                references.append(
                    reference.astype(
                        np.float32
                    )
                )

                contents.append(
                    content_id
                )

                acoustics.append(
                    target
                )

                speakers.append(
                    speaker_id
                )

    references = np.asarray(
        references,
        dtype=np.float32,
    )

    contents = np.asarray(
        contents,
        dtype=np.int64,
    )

    acoustics = np.asarray(
        acoustics,
        dtype=np.float32,
    )

    speakers = np.asarray(
        speakers,
        dtype=np.int64,
    )

    # --------------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------------

    indices = rng.permutation(
        len(references)
    )

    references = references[indices]
    contents = contents[indices]
    acoustics = acoustics[indices]
    speakers = speakers[indices]

    # --------------------------------------------------------------
    # Normalize acoustic values
    # --------------------------------------------------------------

    acoustic_mean = (
        acoustics.mean()
    )

    acoustic_std = (
        acoustics.std()
        + 1e-6
    )

    acoustics = (
        acoustics
        - acoustic_mean
    ) / acoustic_std

    return SpeechDataset(
        torch.tensor(references),
        torch.tensor(contents),
        torch.tensor(acoustics),
        torch.tensor(speakers),
    )


# ======================================================================
# TRAIN / TEST SPLIT
# ======================================================================

def split_dataset(dataset):

    n = len(dataset)

    split = int(
        0.8 * n
    )

    train = SpeechDataset(
        dataset.reference[:split],
        dataset.content[:split],
        dataset.acoustic[:split],
        dataset.speaker[:split],
    )

    test = SpeechDataset(
        dataset.reference[split:],
        dataset.content[split:],
        dataset.acoustic[split:],
        dataset.speaker[split:],
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
            dataset.reference[idx].to(
                DEVICE
            ),

            dataset.content[idx].to(
                DEVICE
            ),

            dataset.acoustic[idx].to(
                DEVICE
            ),

            dataset.speaker[idx].to(
                DEVICE
            ),
        )


# ======================================================================
# SPEAKER ENCODER
# ======================================================================

class SpeakerEncoder(nn.Module):

    """
    Converts reference speech features into a fixed-dimensional
    speaker representation.
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                ACOUSTIC_DIM,
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
                SPEAKER_EMBED_DIM,
            ),
        )

    def forward(self, reference):

        z = self.network(
            reference
        )

        return F.normalize(
            z,
            dim=-1,
        )


# ======================================================================
# TEXT / CONTENT ENCODER
# ======================================================================

class ContentEncoder(nn.Module):

    """
    Converts a discrete linguistic/content identity into an embedding.

    In a real TTS system this would be replaced by something such as:

        text tokenizer
            ->
        token embeddings
            ->
        text/phoneme/semantic encoder
    """

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            NUM_CONTENTS,
            CONTENT_EMBED_DIM,
        )

    def forward(self, content):

        return self.embedding(
            content
        )


# ======================================================================
# SPEAKER CLASSIFIER
# ======================================================================

class SpeakerClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Linear(
            SPEAKER_EMBED_DIM,
            NUM_SPEAKERS,
        )

    def forward(self, z):

        return self.network(z)


# ======================================================================
# SPEAKER ENCODER TRAINING
# ======================================================================

def train_speaker_encoder(
    train,
    test,
):

    print()
    print("=" * 72)
    print("STEP 2 — TRAIN SPEAKER ENCODER")
    print("=" * 72)

    encoder = SpeakerEncoder().to(
        DEVICE
    )

    classifier = SpeakerClassifier().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(classifier.parameters()),
        lr=LEARNING_RATE,
    )

    for epoch in range(
        1,
        SPEAKER_EPOCHS + 1,
    ):

        encoder.train()
        classifier.train()

        total_correct = 0
        total = 0

        for (
            reference,
            content,
            acoustic,
            speaker,
        ) in batches(
            train,
            BATCH_SIZE,
        ):

            z = encoder(
                reference
            )

            logits = classifier(z)

            loss = F.cross_entropy(
                logits,
                speaker,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_correct += (
                logits.argmax(dim=-1)
                == speaker
            ).sum().item()

            total += len(
                speaker
            )

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == SPEAKER_EPOCHS
        ):

            test_accuracy = (
                evaluate_speaker_encoder(
                    encoder,
                    classifier,
                    test,
                )
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train acc={total_correct / total:.3f} | "
                f"test acc={test_accuracy:.3f}"
            )

    return encoder


# ======================================================================
# SPEAKER ENCODER EVALUATION
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

    for (
        reference,
        content,
        acoustic,
        speaker,
    ) in batches(
        dataset,
        BATCH_SIZE,
        shuffle=False,
    ):

        z = encoder(
            reference
        )

        logits = classifier(z)

        correct += (
            logits.argmax(dim=-1)
            == speaker
        ).sum().item()

        total += len(
            speaker
        )

    return correct / total


# ======================================================================
# CONDITIONAL ACOUSTIC GENERATOR
# ======================================================================

class ConditionalGenerator(
    nn.Module
):

    """
    Generates an acoustic representation from:

        speaker embedding
        content embedding

    The generator operates frame-by-frame using a shared conditioning
    representation.

    This is deliberately simple.

    Later lessons will replace this with sequence models.
    """

    def __init__(self):

        super().__init__()

        conditioning_dim = (
            SPEAKER_EMBED_DIM
            + CONTENT_EMBED_DIM
        )

        self.network = nn.Sequential(

            nn.Linear(
                conditioning_dim,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                ACOUSTIC_DIM,
            ),
        )

        # Temporal modulation.
        self.temporal = nn.Parameter(
            torch.randn(
                ACOUSTIC_FRAMES,
                ACOUSTIC_DIM,
            )
            * 0.05
        )

    def forward(
        self,
        speaker_embedding,
        content_embedding,
    ):

        conditioning = torch.cat(
            [
                speaker_embedding,
                content_embedding,
            ],
            dim=-1,
        )

        frame_base = self.network(
            conditioning
        )

        # Expand the conditioning across time.

        frame_base = frame_base.unsqueeze(
            1
        )

        frame_base = frame_base.expand(
            -1,
            ACOUSTIC_FRAMES,
            -1,
        )

        # Add learnable temporal structure.

        output = (
            frame_base
            + self.temporal.unsqueeze(0)
        )

        return output


# ======================================================================
# TRAIN GENERATOR
# ======================================================================

def train_generator(
    encoder,
    train,
    test,
):

    print()
    print("=" * 72)
    print("STEP 4 — TRAIN SPEAKER-CONDITIONED GENERATOR")
    print("=" * 72)

    # --------------------------------------------------------------
    # Freeze speaker encoder.
    #
    # The generator should learn to use the speaker representation,
    # not modify the representation itself.
    # --------------------------------------------------------------

    encoder.eval()

    for parameter in encoder.parameters():

        parameter.requires_grad = False

    # --------------------------------------------------------------
    # Content encoder
    # --------------------------------------------------------------

    content_encoder = (
        ContentEncoder()
        .to(DEVICE)
    )

    # --------------------------------------------------------------
    # Generator
    # --------------------------------------------------------------

    generator = (
        ConditionalGenerator()
        .to(DEVICE)
    )

    optimizer = torch.optim.Adam(
        list(
            content_encoder.parameters()
        )
        + list(
            generator.parameters()
        ),
        lr=LEARNING_RATE,
    )

    for epoch in range(
        1,
        GENERATOR_EPOCHS + 1,
    ):

        generator.train()
        content_encoder.train()

        total_loss = 0.0
        total = 0

        for (
            reference,
            content,
            acoustic,
            speaker,
        ) in batches(
            train,
            BATCH_SIZE,
        ):

            # --------------------------------------------------
            # Speaker representation
            # --------------------------------------------------

            with torch.no_grad():

                speaker_embedding = (
                    encoder(reference)
                )

            # --------------------------------------------------
            # Linguistic representation
            # --------------------------------------------------

            content_embedding = (
                content_encoder(content)
            )

            # --------------------------------------------------
            # Generate acoustic representation
            # --------------------------------------------------

            prediction = generator(
                speaker_embedding,
                content_embedding,
            )

            # --------------------------------------------------
            # Reconstruction loss
            # --------------------------------------------------

            loss = F.mse_loss(
                prediction,
                acoustic,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(acoustic)
            )

            total += len(acoustic)

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == GENERATOR_EPOCHS
        ):

            test_loss = (
                evaluate_generator(
                    encoder,
                    content_encoder,
                    generator,
                    test,
                )
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train loss={total_loss / total:.5f} | "
                f"test loss={test_loss:.5f}"
            )

    return (
        content_encoder,
        generator,
    )


# ======================================================================
# GENERATOR EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate_generator(
    speaker_encoder,
    content_encoder,
    generator,
    dataset,
):

    speaker_encoder.eval()
    content_encoder.eval()
    generator.eval()

    total_loss = 0.0
    total = 0

    for (
        reference,
        content,
        acoustic,
        speaker,
    ) in batches(
        dataset,
        BATCH_SIZE,
        shuffle=False,
    ):

        speaker_embedding = (
            speaker_encoder(reference)
        )

        content_embedding = (
            content_encoder(content)
        )

        prediction = generator(
            speaker_embedding,
            content_embedding,
        )

        loss = F.mse_loss(
            prediction,
            acoustic,
        )

        total_loss += (
            loss.item()
            * len(acoustic)
        )

        total += len(acoustic)

    return total_loss / total


# ======================================================================
# SINGLE SAMPLE GENERATION
# ======================================================================

@torch.no_grad()
def generate_sample(
    speaker_encoder,
    content_encoder,
    generator,
    reference,
    content_id,
):

    speaker_encoder.eval()
    content_encoder.eval()
    generator.eval()

    reference = reference.to(
        DEVICE
    ).unsqueeze(0)

    content = torch.tensor(
        [content_id],
        device=DEVICE,
    )

    speaker_embedding = (
        speaker_encoder(reference)
    )

    content_embedding = (
        content_encoder(content)
    )

    generated = generator(
        speaker_embedding,
        content_embedding,
    )

    return (
        generated[0]
        .cpu()
        .numpy()
    )


# ======================================================================
# SPEAKER SIMILARITY
# ======================================================================

def cosine_similarity(
    a,
    b,
):

    a = torch.tensor(
        a,
        dtype=torch.float32,
    )

    b = torch.tensor(
        b,
        dtype=torch.float32,
    )

    a = F.normalize(
        a.flatten(0),
        dim=0,
    )

    b = F.normalize(
        b.flatten(0),
        dim=0,
    )

    return torch.dot(
        a,
        b,
    ).item()


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
    print("PHASE 1 — LESSON 28")
    print("SPEAKER REFERENCE CONDITIONING")
    print("=" * 72)

    print()
    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Speakers: {NUM_SPEAKERS}"
    )

    print(
        f"Content classes: {NUM_CONTENTS}"
    )

    print(
        f"Acoustic frames: {ACOUSTIC_FRAMES}"
    )

    print(
        f"Acoustic dimension: {ACOUSTIC_DIM}"
    )

    # ==============================================================
    # STEP 1
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 1 — CREATE SYNTHETIC SPEECH DATA")
    print("=" * 72)

    dataset = create_dataset()

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

    speaker_encoder = (
        train_speaker_encoder(
            train,
            test,
        )
    )

    # ==============================================================
    # STEP 3
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 3 — CREATE CONTENT REPRESENTATION")
    print("=" * 72)

    print(
        "Content IDs will be converted into learned embeddings."
    )

    print(
        "In a real TTS system this becomes a text/phoneme/semantic encoder."
    )

    # ==============================================================
    # STEP 4
    # ==============================================================

    (
        content_encoder,
        generator,
    ) = train_generator(
        speaker_encoder,
        train,
        test,
    )

    # ==============================================================
    # STEP 5
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 5 — SPEAKER TRANSFER EXPERIMENT")
    print("=" * 72)

    # --------------------------------------------------------------
    # Select two speakers.
    # --------------------------------------------------------------

    speaker_a = 0
    speaker_b = 1

    # Find reference examples for both speakers.
    # We use the first matching reference.
    # --------------------------------------------------------------

    speaker_a_index = (
        train.speaker
        == speaker_a
    ).nonzero(as_tuple=True)[0][0]

    speaker_b_index = (
        train.speaker
        == speaker_b
    ).nonzero(as_tuple=True)[0][0]

    reference_a = (
        train.reference[
            speaker_a_index
        ]
    )

    reference_b = (
        train.reference[
            speaker_b_index
        ]
    )

    # Keep content fixed.
    content_id = 3

    generated_a = generate_sample(
        speaker_encoder,
        content_encoder,
        generator,
        reference_a,
        content_id,
    )

    generated_b = generate_sample(
        speaker_encoder,
        content_encoder,
        generator,
        reference_b,
        content_id,
    )

    print(
        f"Same content = {content_id}"
    )

    print(
        f"Speaker A reference -> generated representation"
    )

    print(
        f"Speaker B reference -> generated representation"
    )

    print(
        "Content is fixed while speaker conditioning changes."
    )

    # ==============================================================
    # STEP 6
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 6 — CONTENT TRANSFER EXPERIMENT")
    print("=" * 72)

    # Keep speaker fixed.
    # Change content.

    content_a = 2
    content_b = 10

    generated_content_a = (
        generate_sample(
            speaker_encoder,
            content_encoder,
            generator,
            reference_a,
            content_a,
        )
    )

    generated_content_b = (
        generate_sample(
            speaker_encoder,
            content_encoder,
            generator,
            reference_a,
            content_b,
        )
    )

    print(
        f"Same speaker = {speaker_a}"
    )

    print(
        f"Content A = {content_a}"
    )

    print(
        f"Content B = {content_b}"
    )

    print(
        "Speaker is fixed while content conditioning changes."
    )

    # ==============================================================
    # STEP 7
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 7 — REPRESENTATION SIMILARITY")
    print("=" * 72)

    speaker_transfer_similarity = (
        cosine_similarity(
            generated_a,
            generated_b,
        )
    )

    content_transfer_similarity = (
        cosine_similarity(
            generated_content_a,
            generated_content_b,
        )
    )

    print(
        f"Same content / different speaker similarity: "
        f"{speaker_transfer_similarity:.4f}"
    )

    print(
        f"Same speaker / different content similarity: "
        f"{content_transfer_similarity:.4f}"
    )

    # ==============================================================
    # STEP 8 — VISUALIZATION
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 8 — VISUALIZE GENERATED ACOUSTIC FEATURES")
    print("=" * 72)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    # --------------------------------------------------------------
    # Same content, speaker A
    # --------------------------------------------------------------

    axes[0, 0].imshow(
        generated_a.T,
        aspect="auto",
        origin="lower",
    )

    axes[0, 0].set_title(
        "Same Content — Speaker A"
    )

    axes[0, 0].set_xlabel(
        "Acoustic frame"
    )

    axes[0, 0].set_ylabel(
        "Acoustic dimension"
    )

    # --------------------------------------------------------------
    # Same content, speaker B
    # --------------------------------------------------------------

    axes[0, 1].imshow(
        generated_b.T,
        aspect="auto",
        origin="lower",
    )

    axes[0, 1].set_title(
        "Same Content — Speaker B"
    )

    axes[0, 1].set_xlabel(
        "Acoustic frame"
    )

    axes[0, 1].set_ylabel(
        "Acoustic dimension"
    )

    # --------------------------------------------------------------
    # Same speaker, content A
    # --------------------------------------------------------------

    axes[1, 0].imshow(
        generated_content_a.T,
        aspect="auto",
        origin="lower",
    )

    axes[1, 0].set_title(
        "Same Speaker — Content A"
    )

    axes[1, 0].set_xlabel(
        "Acoustic frame"
    )

    axes[1, 0].set_ylabel(
        "Acoustic dimension"
    )

    # --------------------------------------------------------------
    # Same speaker, content B
    # --------------------------------------------------------------

    axes[1, 1].imshow(
        generated_content_b.T,
        aspect="auto",
        origin="lower",
    )

    axes[1, 1].set_title(
        "Same Speaker — Content B"
    )

    axes[1, 1].set_xlabel(
        "Acoustic frame"
    )

    axes[1, 1].set_ylabel(
        "Acoustic dimension"
    )

    plt.suptitle(
        "Lesson 28 — Speaker-Conditioned Acoustic Generation"
    )

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "speaker_conditioning_results.png",
    )

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.show()

    print(
        f"Saved visualization: {plot_path}"
    )

    # ==============================================================
    # STEP 9 — SAVE MODEL
    # ==============================================================

    print()
    print("=" * 72)
    print("STEP 9 — SAVE MODEL")
    print("=" * 72)

    model_path = os.path.join(
        OUTPUT_DIR,
        "lesson28_model.pt",
    )

    torch.save(
        {
            "speaker_encoder":
                speaker_encoder.state_dict(),

            "content_encoder":
                content_encoder.state_dict(),

            "generator":
                generator.state_dict(),
        },
        model_path,
    )

    print(
        f"Saved model: {model_path}"
    )

    # ==============================================================
    # FINAL EXPLANATION
    # ==============================================================

    print()
    print("=" * 72)
    print("LESSON 28 — WHAT WE JUST BUILT")
    print("=" * 72)

    print()
    print(
        "Reference speech"
    )

    print(
        "      ↓"
    )

    print(
        "Speaker Encoder"
    )

    print(
        "      ↓"
    )

    print(
        "Speaker Embedding"
    )

    print(
        "      ↓"
    )

    print(
        "                + Text / Content Embedding"
    )

    print(
        "                ↓"
    )

    print(
        "        Conditional Generator"
    )

    print(
        "                ↓"
    )

    print(
        "        Acoustic Representation"
    )

    print()
    print(
        "This is the core conditioning idea behind "
        "speaker-conditioned TTS."
    )

    print()
    print(
        "The important conceptual separation is:"
    )

    print(
        "    speaker embedding = WHO"
    )

    print(
        "    content embedding = WHAT"
    )

    print(
        "    generator = HOW the acoustic signal is produced"
    )

    print()
    print("=" * 72)
    print("LESSON 28 COMPLETE")
    print("=" * 72)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()