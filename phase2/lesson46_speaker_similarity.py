"""
========================================================================
VOICE CLONING COURSE
LESSON 46 — SPEAKER SIMILARITY AND IDENTITY PRESERVATION
========================================================================

PURPOSE
-------
Learn to extract speaker representations and evaluate whether speech
sequences preserve speaker identity.

WHAT YOU LEARN
--------------

STEP 1 — Multi-Speaker Speech Dataset
    Create speech-token sequences for multiple speakers.

STEP 2 — Speaker Encoder
    Encode speech-token sequences into fixed-dimensional vectors.

STEP 3 — Speaker Classification
    Predict speaker identity from the extracted representation.

STEP 4 — Supervised Contrastive Learning
    Pull embeddings from the same speaker together.
    Push embeddings from different speakers apart.

STEP 5 — Cosine Similarity
    Measure similarity between two speaker representations.

STEP 6 — Identity Preservation
    Evaluate whether generated speech-token sequences preserve
    the intended speaker identity.

STEP 7 — Generalization
    Use separate training and validation sequences.

STEP 8 — Visualization
    Plot embedding distributions and similarity matrices.

REAL-WORLD APPLICATION
----------------------
A voice cloning system receives reference audio from Speaker A.

The speaker encoder extracts an identity representation.

Generated speech is then evaluated against that representation.

A high similarity score indicates that the generated speech
has a similar speaker representation.

NOTE
----
This lesson uses synthetic token data.

It demonstrates the mathematics and training architecture,
not actual human voice similarity.

Real audio-based speaker verification comes in Lesson 47.

REQUIREMENTS
------------
pip install torch numpy matplotlib scikit-learn

RUN
---
python lesson46_speaker_similarity.py

========================================================================
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 256

NUM_SPEAKERS = 4

SEQUENCE_LENGTH = 128

EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 2

BATCH_SIZE = 64

EPOCHS = 40

LEARNING_RATE = 3e-4

CONTRASTIVE_WEIGHT = 0.2

NUM_TRAIN_PER_SPEAKER = 500

NUM_VAL_PER_SPEAKER = 150

OUTPUT_DIR = "lesson46_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================================
# STEP 1 — CREATE MULTI-SPEAKER DATASET
# =====================================================================

SPEAKER_PATTERNS = [
    [10, 20, 30, 40, 50],
    [60, 70, 80, 90, 100],
    [110, 120, 130, 140, 150],
    [160, 170, 180, 190, 200],
]


def generate_sequence(speaker_id, length):

    pattern = SPEAKER_PATTERNS[speaker_id]

    tokens = []

    for _ in range(length):

        if random.random() < 0.85:

            token = random.choice(pattern)

        else:

            token = random.randint(0, VOCAB_SIZE - 1)

        tokens.append(token)

    return tokens


def create_dataset():

    train_samples = []

    val_samples = []

    for speaker_id in range(NUM_SPEAKERS):

        for _ in range(NUM_TRAIN_PER_SPEAKER):

            tokens = generate_sequence(
                speaker_id,
                SEQUENCE_LENGTH
            )

            train_samples.append(
                (tokens, speaker_id)
            )

        for _ in range(NUM_VAL_PER_SPEAKER):

            tokens = generate_sequence(
                speaker_id,
                SEQUENCE_LENGTH
            )

            val_samples.append(
                (tokens, speaker_id)
            )

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


# =====================================================================
# STEP 2 — DATASET CLASS
# =====================================================================

class SpeakerDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        tokens, speaker_id = self.samples[index]

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(speaker_id, dtype=torch.long)
        )


# =====================================================================
# STEP 3 — SPEAKER ENCODER
# =====================================================================

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_length):

        super().__init__()

        position = torch.arange(
            max_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * (-np.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(
            max_length,
            embed_dim
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class SpeakerEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM
        )

        self.position_encoding = PositionalEncoding(
            EMBED_DIM,
            SEQUENCE_LENGTH
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.projection = nn.Sequential(

            nn.Linear(
                EMBED_DIM,
                EMBED_DIM
            ),

            nn.GELU(),

            nn.Linear(
                EMBED_DIM,
                EMBED_DIM
            )

        )

        self.classifier = nn.Linear(
            EMBED_DIM,
            NUM_SPEAKERS
        )

    def forward(self, tokens):

        x = self.token_embedding(tokens)

        x = self.position_encoding(x)

        x = self.transformer(x)

        # Mean pooling converts sequence representations
        # into one fixed-dimensional speaker representation.
        pooled = x.mean(dim=1)

        embedding = self.projection(pooled)

        # L2 normalization is important for cosine similarity.
        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1
        )

        logits = self.classifier(embedding)

        return embedding, logits


# =====================================================================
# STEP 4 — SUPERVISED CONTRASTIVE LOSS
# =====================================================================

def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.1
):

    """
    Same-speaker embeddings should be close.

    Different-speaker embeddings should be separated.

    This is a supervised contrastive objective.
    """

    batch_size = embeddings.shape[0]

    similarity = torch.matmul(
        embeddings,
        embeddings.T
    ) / temperature

    # Exclude self-similarity.
    self_mask = torch.eye(
        batch_size,
        device=embeddings.device,
        dtype=torch.bool
    )

    similarity = similarity.masked_fill(
        self_mask,
        -1e9
    )

    labels = labels.view(-1, 1)

    positive_mask = (
        labels == labels.T
    ) & (~self_mask)

    log_prob = similarity - torch.logsumexp(
        similarity,
        dim=1,
        keepdim=True
    )

    positive_count = positive_mask.sum(
        dim=1
    )

    valid = positive_count > 0

    positive_log_prob = (
        log_prob * positive_mask.float()
    ).sum(dim=1)

    loss = -positive_log_prob / positive_count.clamp(
        min=1
    )

    if valid.any():

        return loss[valid].mean()

    return torch.tensor(
        0.0,
        device=embeddings.device
    )


# =====================================================================
# STEP 5 — TRAINING
# =====================================================================

def train_model(
    model,
    train_loader,
    val_loader
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "contrastive_loss": []
    }

    best_val_loss = float("inf")

    patience = 8

    epochs_without_improvement = 0

    print("\nStarting speaker encoder training...\n")

    for epoch in range(EPOCHS):

        model.train()

        train_losses = []

        train_contrastive = []

        train_predictions = []

        train_targets = []

        for tokens, labels in train_loader:

            tokens = tokens.to(DEVICE)

            labels = labels.to(DEVICE)

            embeddings, logits = model(tokens)

            classification_loss = F.cross_entropy(
                logits,
                labels
            )

            contrastive_loss = supervised_contrastive_loss(
                embeddings,
                labels
            )

            loss = (
                classification_loss
                + CONTRASTIVE_WEIGHT * contrastive_loss
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

            train_contrastive.append(
                contrastive_loss.item()
            )

            predictions = logits.argmax(dim=1)

            train_predictions.extend(
                predictions.detach().cpu().tolist()
            )

            train_targets.extend(
                labels.detach().cpu().tolist()
            )

        train_loss = float(np.mean(train_losses))

        train_accuracy = accuracy_score(
            train_targets,
            train_predictions
        )

        # Validation.
        model.eval()

        val_losses = []

        val_predictions = []

        val_targets = []

        with torch.no_grad():

            for tokens, labels in val_loader:

                tokens = tokens.to(DEVICE)

                labels = labels.to(DEVICE)

                embeddings, logits = model(tokens)

                classification_loss = F.cross_entropy(
                    logits,
                    labels
                )

                contrastive_loss = supervised_contrastive_loss(
                    embeddings,
                    labels
                )

                loss = (
                    classification_loss
                    + CONTRASTIVE_WEIGHT * contrastive_loss
                )

                val_losses.append(
                    loss.item()
                )

                predictions = logits.argmax(dim=1)

                val_predictions.extend(
                    predictions.cpu().tolist()
                )

                val_targets.extend(
                    labels.cpu().tolist()
                )

        val_loss = float(np.mean(val_losses))

        val_accuracy = accuracy_score(
            val_targets,
            val_predictions
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)

        history["val_loss"].append(val_loss)

        history["train_accuracy"].append(train_accuracy)

        history["val_accuracy"].append(val_accuracy)

        history["contrastive_loss"].append(
            float(np.mean(train_contrastive))
        )

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                os.path.join(
                    OUTPUT_DIR,
                    "lesson46_best.pt"
                )
            )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print("\nEarly stopping activated.")

            break

    return history


# =====================================================================
# STEP 6 — EXTRACT SPEAKER EMBEDDINGS
# =====================================================================

@torch.no_grad()
def extract_embeddings(model, loader):

    model.eval()

    all_embeddings = []

    all_labels = []

    for tokens, labels in loader:

        tokens = tokens.to(DEVICE)

        embeddings, logits = model(tokens)

        all_embeddings.append(
            embeddings.cpu()
        )

        all_labels.append(
            labels
        )

    return (
        torch.cat(all_embeddings).numpy(),
        torch.cat(all_labels).numpy()
    )


# =====================================================================
# STEP 7 — COSINE SIMILARITY
# =====================================================================

def cosine_similarity(a, b):

    a = np.asarray(a)

    b = np.asarray(b)

    a = a / (
        np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
    )

    b = b / (
        np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8
    )

    return np.sum(
        a * b,
        axis=-1
    )


def calculate_similarity_statistics(
    embeddings,
    labels
):

    intra_scores = []

    inter_scores = []

    for i in range(len(embeddings)):

        for j in range(i + 1, len(embeddings)):

            score = cosine_similarity(
                embeddings[i],
                embeddings[j]
            )

            if labels[i] == labels[j]:

                intra_scores.append(score)

            else:

                inter_scores.append(score)

    return (
        np.array(intra_scores),
        np.array(inter_scores)
    )


# =====================================================================
# STEP 8 — SPEAKER SIMILARITY MATRIX
# =====================================================================

def calculate_speaker_matrix(
    embeddings,
    labels
):

    speaker_means = []

    for speaker_id in range(NUM_SPEAKERS):

        speaker_embeddings = embeddings[
            labels == speaker_id
        ]

        mean_embedding = np.mean(
            speaker_embeddings,
            axis=0
        )

        mean_embedding /= (
            np.linalg.norm(mean_embedding) + 1e-8
        )

        speaker_means.append(
            mean_embedding
        )

    speaker_means = np.array(
        speaker_means
    )

    matrix = speaker_means @ speaker_means.T

    return matrix


# =====================================================================
# STEP 9 — GENERATED TOKEN IDENTITY TEST
# =====================================================================

def generate_test_sequence(speaker_id):

    """
    Synthetic generation proxy.

    Each speaker has a different token distribution.

    This is NOT autoregressive GPT generation.
    """

    return generate_sequence(
        speaker_id,
        SEQUENCE_LENGTH
    )


def evaluate_generated_identity(model):

    model.eval()

    print("\n" + "=" * 72)

    print("GENERATED SPEAKER IDENTITY EVALUATION")

    print("=" * 72)

    results = []

    with torch.no_grad():

        for speaker_id in range(NUM_SPEAKERS):

            correct = 0

            total = 100

            for _ in range(total):

                tokens = generate_test_sequence(
                    speaker_id
                )

                x = torch.tensor(
                    [tokens],
                    dtype=torch.long,
                    device=DEVICE
                )

                embedding, logits = model(x)

                predicted = logits.argmax(
                    dim=1
                ).item()

                if predicted == speaker_id:

                    correct += 1

            accuracy = correct / total

            results.append(accuracy)

            print(
                f"Speaker {speaker_id} | "
                f"Identity Accuracy: {accuracy:.4f}"
            )

    return results


# =====================================================================
# STEP 10 — VISUALIZATION
# =====================================================================

def plot_training(history):

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "Lesson 46 — Speaker Encoder Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson46_training_loss.png"
        )
    )

    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        "Speaker Classification Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson46_accuracy.png"
        )
    )

    plt.show()


def plot_embeddings(embeddings, labels):

    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(
        embeddings
    )

    plt.figure(figsize=(9, 7))

    for speaker_id in range(NUM_SPEAKERS):

        mask = labels == speaker_id

        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=f"Speaker {speaker_id}",
            alpha=0.65
        )

    plt.title(
        "Speaker Embeddings — PCA"
    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson46_embedding_pca.png"
        )
    )

    plt.show()


def plot_similarity_matrix(matrix):

    plt.figure(figsize=(7, 6))

    plt.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    plt.colorbar(
        label="Cosine Similarity"
    )

    plt.xticks(
        range(NUM_SPEAKERS),
        [f"Speaker {i}" for i in range(NUM_SPEAKERS)]
    )

    plt.yticks(
        range(NUM_SPEAKERS),
        [f"Speaker {i}" for i in range(NUM_SPEAKERS)]
    )

    plt.title(
        "Speaker-to-Speaker Similarity Matrix"
    )

    for i in range(NUM_SPEAKERS):

        for j in range(NUM_SPEAKERS):

            plt.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson46_similarity_matrix.png"
        )
    )

    plt.show()


def plot_intra_inter(intra, inter):

    plt.figure(figsize=(9, 5))

    plt.hist(
        intra,
        bins=40,
        alpha=0.6,
        label="Same Speaker"
    )

    plt.hist(
        inter,
        bins=40,
        alpha=0.6,
        label="Different Speakers"
    )

    plt.xlabel("Cosine Similarity")

    plt.ylabel("Frequency")

    plt.title(
        "Intra-Speaker vs Inter-Speaker Similarity"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson46_intra_inter_similarity.png"
        )
    )

    plt.show()


# =====================================================================
# STEP 11 — MAIN
# =====================================================================

def main():

    print("=" * 72)

    print(
        "LESSON 46 — SPEAKER SIMILARITY "
        "AND IDENTITY PRESERVATION"
    )

    print("=" * 72)

    print(f"\nDevice: {DEVICE}")

    print(f"Vocabulary: {VOCAB_SIZE}")

    print(f"Speakers: {NUM_SPEAKERS}")

    print(f"Sequence length: {SEQUENCE_LENGTH}")

    print(f"Embedding dimension: {EMBED_DIM}")

    # Dataset.
    train_samples, val_samples = create_dataset()

    print(
        f"\nTraining samples: {len(train_samples)}"
    )

    print(
        f"Validation samples: {len(val_samples)}"
    )

    train_dataset = SpeakerDataset(
        train_samples
    )

    val_dataset = SpeakerDataset(
        val_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Model.
    model = SpeakerEncoder().to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: {parameters:,}"
    )

    # Train.
    history = train_model(
        model,
        train_loader,
        val_loader
    )

    # Load best checkpoint.
    model.load_state_dict(
        torch.load(
            os.path.join(
                OUTPUT_DIR,
                "lesson46_best.pt"
            ),
            map_location=DEVICE,
            weights_only=True
        )
    )

    # Extract validation embeddings.
    embeddings, labels = extract_embeddings(
        model,
        val_loader
    )

    # Similarity statistics.
    intra, inter = calculate_similarity_statistics(
        embeddings,
        labels
    )

    print("\n" + "=" * 72)

    print("SPEAKER SIMILARITY STATISTICS")

    print("=" * 72)

    print(
        f"\nMean intra-speaker similarity: "
        f"{intra.mean():.4f}"
    )

    print(
        f"Mean inter-speaker similarity: "
        f"{inter.mean():.4f}"
    )

    print(
        f"Similarity separation: "
        f"{intra.mean() - inter.mean():.4f}"
    )

    # Speaker matrix.
    matrix = calculate_speaker_matrix(
        embeddings,
        labels
    )

    print("\nSpeaker similarity matrix:")

    print(
        np.array2string(
            matrix,
            precision=3
        )
    )

    # Generated identity evaluation.
    generated_accuracy = evaluate_generated_identity(
        model
    )

    # Visualizations.
    plot_training(history)

    plot_embeddings(
        embeddings,
        labels
    )

    plot_similarity_matrix(
        matrix
    )

    plot_intra_inter(
        intra,
        inter
    )

    # Save metrics.
    metrics = {
        "best_validation_loss": min(history["val_loss"]),
        "final_validation_accuracy": history["val_accuracy"][-1],
        "mean_intra_similarity": float(intra.mean()),
        "mean_inter_similarity": float(inter.mean()),
        "similarity_separation": float(
            intra.mean() - inter.mean()
        ),
        "generated_identity_accuracy": generated_accuracy
    }

    import json

    with open(
        os.path.join(
            OUTPUT_DIR,
            "lesson46_metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print("\n" + "=" * 72)

    print("LESSON 46 COMPLETED")

    print("=" * 72)

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\nGenerated files:")

    print("  lesson46_best.pt")

    print("  lesson46_metrics.json")

    print("  lesson46_training_loss.png")

    print("  lesson46_accuracy.png")

    print("  lesson46_embedding_pca.png")

    print("  lesson46_similarity_matrix.png")

    print("  lesson46_intra_inter_similarity.png")


if __name__ == "__main__":

    main()