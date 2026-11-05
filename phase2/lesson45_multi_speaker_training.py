"""
========================================================================
VOICE CLONING COURSE
LESSON 45 — MULTI-SPEAKER DATASET AND TRAINING
========================================================================

PURPOSE
-------
Train one autoregressive speech-token GPT model conditioned on speaker ID.

LESSON ROADMAP
--------------

STEP 1 — Multi-Speaker Dataset
    Organize speech-token sequences by speaker identity.

STEP 2 — Speaker ID Mapping
    Convert speaker names into integer IDs.

STEP 3 — Speaker Embedding
    Learn a vector representation for every speaker.

STEP 4 — Speaker-Conditioned GPT
    Add speaker information to every token representation.

STEP 5 — Autoregressive Training
    Predict the next speech token.

STEP 6 — Multi-Speaker Evaluation
    Compare generated token sequences for different speakers.

STEP 7 — Visualization
    Plot training loss and speaker embeddings.

STEP 8 — Checkpoint
    Save the trained multi-speaker model.

REAL-WORLD EXAMPLE
------------------
A voice cloning application may contain recordings from 100 speakers.

A user selects Speaker 12.

The model receives:
    - Speaker ID 12
    - Previous speech tokens

The model predicts the next speech tokens using the learned
speaker-conditioned representation.

IMPORTANT
---------
This lesson uses synthetic data by default so it can run immediately.

For real speech tokens, create a JSONL manifest:

{"speaker":"speaker_0","tokens":[12,45,67,89,23]}
{"speaker":"speaker_1","tokens":[33,55,78,91,12]}

Each sequence should contain integer speech-codec tokens.

You can also specify "token_file" instead of "tokens".
The token_file should contain a PyTorch tensor or integer list.

REQUIREMENTS
------------
pip install torch numpy matplotlib scikit-learn

RUN
---
python lesson45_multi_speaker_training.py

========================================================================
"""

import os
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 256
BLOCK_SIZE = 128

EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 3e-4

NUM_SYNTHETIC_SPEAKERS = 4
SEQUENCES_PER_SPEAKER = 100
SYNTHETIC_SEQUENCE_LENGTH = 300

MANIFEST_PATH = "dataset/multispeaker_manifest.jsonl"

OUTPUT_DIR = "lesson45_outputs"

USE_SYNTHETIC_DATA = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================================
# STEP 1 — CREATE SYNTHETIC MULTI-SPEAKER DATASET
# =====================================================================

def generate_synthetic_sequence(speaker_id, length):

    """
    Each synthetic speaker has a different token distribution.

    This is only a debugging demonstration.
    It does NOT represent real human voice identity.
    """

    speaker_patterns = [
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100],
        [110, 120, 130, 140, 150],
        [160, 170, 180, 190, 200],
    ]

    pattern = speaker_patterns[speaker_id]

    tokens = []

    current = random.choice(pattern)

    for i in range(length):

        if random.random() < 0.65:

            current = random.choice(pattern)

        else:

            current = random.randint(0, VOCAB_SIZE - 1)

        tokens.append(current)

    return tokens


def create_synthetic_dataset():

    dataset = []

    for speaker_id in range(NUM_SYNTHETIC_SPEAKERS):

        speaker_name = f"speaker_{speaker_id}"

        for _ in range(SEQUENCES_PER_SPEAKER):

            tokens = generate_synthetic_sequence(
                speaker_id,
                SYNTHETIC_SEQUENCE_LENGTH
            )

            dataset.append({
                "speaker": speaker_name,
                "tokens": tokens
            })

    return dataset


# =====================================================================
# STEP 2 — LOAD REAL OR SYNTHETIC DATASET
# =====================================================================

def load_dataset():

    if USE_SYNTHETIC_DATA:

        print("\nUsing synthetic multi-speaker dataset.")

        return create_synthetic_dataset()

    if not os.path.exists(MANIFEST_PATH):

        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    dataset = []

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:

        for line in f:

            item = json.loads(line)

            if "tokens" in item:

                tokens = item["tokens"]

            elif "token_file" in item:

                token_path = item["token_file"]

                if not os.path.isabs(token_path):
                    token_path = os.path.join(
                        os.path.dirname(MANIFEST_PATH),
                        token_path
                    )

                loaded = torch.load(
                    token_path,
                    map_location="cpu",
                    weights_only=True
                )

                if isinstance(loaded, dict):

                    if "tokens" not in loaded:
                        raise ValueError(
                            f"No 'tokens' key in {token_path}"
                        )

                    loaded = loaded["tokens"]

                tokens = torch.as_tensor(loaded).reshape(-1).tolist()

            else:

                raise ValueError(
                    "Each manifest entry needs 'tokens' "
                    "or 'token_file'."
                )

            tokens = [int(x) for x in tokens]

            if len(tokens) < 2:
                continue

            if min(tokens) < 0 or max(tokens) >= VOCAB_SIZE:
                raise ValueError(
                    f"Token outside vocabulary 0-{VOCAB_SIZE - 1}"
                )

            dataset.append({
                "speaker": str(item["speaker"]),
                "tokens": tokens
            })

    return dataset


# =====================================================================
# STEP 3 — BUILD SPEAKER ID MAPPING
# =====================================================================

def build_speaker_mapping(dataset):

    speakers = sorted(
        set(item["speaker"] for item in dataset)
    )

    speaker_to_id = {
        name: idx
        for idx, name in enumerate(speakers)
    }

    print("\nSpeaker mapping:")

    for name, idx in speaker_to_id.items():

        print(f"  {name:20s} -> {idx}")

    return speaker_to_id


# =====================================================================
# STEP 4 — SPLIT TRAINING AND VALIDATION DATA
# =====================================================================

def split_dataset(dataset):

    random.shuffle(dataset)

    split_index = int(len(dataset) * 0.9)

    train_data = dataset[:split_index]
    val_data = dataset[split_index:]

    return train_data, val_data


# =====================================================================
# STEP 5 — SPEECH TOKEN DATASET
# =====================================================================

class MultiSpeakerDataset(Dataset):

    def __init__(
        self,
        data,
        speaker_to_id,
        block_size
    ):

        self.samples = []

        self.block_size = block_size

        for item in data:

            tokens = item["tokens"]

            speaker_id = speaker_to_id[item["speaker"]]

            # Create overlapping training windows.
            stride = block_size // 2

            for start in range(
                0,
                len(tokens) - 1,
                stride
            ):

                chunk = tokens[
                    start:start + block_size + 1
                ]

                if len(chunk) < 2:
                    continue

                self.samples.append(
                    (chunk, speaker_id)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        tokens, speaker_id = self.samples[idx]

        x = tokens[:-1]
        y = tokens[1:]

        length = len(x)

        # Pad incomplete sequences.
        x = x + [0] * (self.block_size - length)

        y = y + [-100] * (self.block_size - length)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(speaker_id, dtype=torch.long)
        )


# =====================================================================
# STEP 6 — SPEAKER-CONDITIONED GPT
# =====================================================================

class CausalSelfAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True
        )

    def forward(self, x):

        length = x.shape[1]

        # Future tokens cannot be accessed.
        mask = torch.triu(
            torch.ones(
                length,
                length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=mask,
            need_weights=False
        )

        return output


class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.ln1 = nn.LayerNorm(EMBED_DIM)

        self.attention = CausalSelfAttention()

        self.ln2 = nn.LayerNorm(EMBED_DIM)

        self.mlp = nn.Sequential(

            nn.Linear(EMBED_DIM, EMBED_DIM * 4),

            nn.GELU(),

            nn.Linear(EMBED_DIM * 4, EMBED_DIM),

            nn.Dropout(DROPOUT)

        )

    def forward(self, x):

        x = x + self.attention(self.ln1(x))

        x = x + self.mlp(self.ln2(x))

        return x


class MultiSpeakerGPT(nn.Module):

    def __init__(
        self,
        num_speakers
    ):

        super().__init__()

        # Speech token embedding.
        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM
        )

        # Speaker identity embedding.
        self.speaker_embedding = nn.Embedding(
            num_speakers,
            EMBED_DIM
        )

        # Positional embedding.
        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            EMBED_DIM
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        self.final_norm = nn.LayerNorm(EMBED_DIM)

        self.output_head = nn.Linear(
            EMBED_DIM,
            VOCAB_SIZE
        )

    def forward(
        self,
        tokens,
        speaker_ids
    ):

        batch_size, seq_length = tokens.shape

        positions = torch.arange(
            seq_length,
            device=tokens.device
        )

        # Three sources of information.
        token_emb = self.token_embedding(tokens)

        position_emb = self.position_embedding(positions)

        speaker_emb = self.speaker_embedding(
            speaker_ids
        ).unsqueeze(1)

        # Speaker conditioning.
        x = (
            token_emb
            + position_emb.unsqueeze(0)
            + speaker_emb
        )

        x = self.blocks(x)

        x = self.final_norm(x)

        logits = self.output_head(x)

        return logits


# =====================================================================
# STEP 7 — EVALUATION
# =====================================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    losses = []

    for x, y, speaker_ids in loader:

        x = x.to(DEVICE)

        y = y.to(DEVICE)

        speaker_ids = speaker_ids.to(DEVICE)

        logits = model(x, speaker_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            ignore_index=-100
        )

        losses.append(loss.item())

    model.train()

    return float(np.mean(losses))


# =====================================================================
# STEP 8 — TRAINING
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

    train_history = []

    val_history = []

    best_val_loss = float("inf")

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        model.train()

        epoch_losses = []

        for x, y, speaker_ids in train_loader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            speaker_ids = speaker_ids.to(DEVICE)

            logits = model(
                x,
                speaker_ids
            )

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                ignore_index=-100
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        val_loss = evaluate(
            model,
            val_loader
        )

        train_history.append(train_loss)

        val_history.append(val_loss)

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Validation: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                os.path.join(
                    OUTPUT_DIR,
                    "lesson45_best.pt"
                )
            )

    return train_history, val_history


# =====================================================================
# STEP 9 — SPEAKER-CONDITIONED GENERATION
# =====================================================================

@torch.no_grad()
def generate_tokens(
    model,
    speaker_id,
    prompt=None,
    max_new_tokens=100,
    temperature=0.8
):

    model.eval()

    if prompt is None:

        prompt = [10]

    tokens = torch.tensor(
        [prompt],
        dtype=torch.long,
        device=DEVICE
    )

    speaker = torch.tensor(
        [speaker_id],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(max_new_tokens):

        context = tokens[:, -BLOCK_SIZE:]

        logits = model(
            context,
            speaker
        )

        next_logits = logits[:, -1, :]

        next_logits = next_logits / temperature

        probabilities = F.softmax(
            next_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )

    return tokens[0].tolist()


# =====================================================================
# STEP 10 — VISUALIZE TRAINING
# =====================================================================

def plot_training(
    train_history,
    val_history
):

    plt.figure(figsize=(10, 5))

    plt.plot(
        train_history,
        label="Training Loss"
    )

    plt.plot(
        val_history,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Cross-Entropy Loss")

    plt.title(
        "Lesson 45 — Multi-Speaker GPT Training"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson45_training_loss.png"
        )
    )

    plt.show()


# =====================================================================
# STEP 11 — VISUALIZE SPEAKER EMBEDDINGS
# =====================================================================

def plot_speaker_embeddings(model):

    embeddings = (
        model.speaker_embedding.weight
        .detach()
        .cpu()
        .numpy()
    )

    if len(embeddings) < 2:

        print("Not enough speakers for PCA visualization.")

        return

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=120
    )

    for idx, point in enumerate(reduced):

        plt.annotate(
            f"Speaker {idx}",
            (point[0], point[1])
        )

    plt.title(
        "Learned Speaker Embeddings — PCA"
    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson45_speaker_embeddings.png"
        )
    )

    plt.show()


# =====================================================================
# STEP 12 — MAIN
# =====================================================================

def main():

    print("=" * 72)

    print("LESSON 45 — MULTI-SPEAKER DATASET AND TRAINING")

    print("=" * 72)

    print(f"\nDevice: {DEVICE}")

    print(f"Vocabulary: {VOCAB_SIZE}")

    print(f"Block size: {BLOCK_SIZE}")

    print(f"Embedding dimension: {EMBED_DIM}")

    # Load data.
    dataset = load_dataset()

    print(f"\nTotal sequences: {len(dataset)}")

    # Speaker IDs.
    speaker_to_id = build_speaker_mapping(dataset)

    # Split data.
    train_data, val_data = split_dataset(dataset)

    print(f"\nTraining sequences: {len(train_data)}")

    print(f"Validation sequences: {len(val_data)}")

    # Dataset objects.
    train_dataset = MultiSpeakerDataset(
        train_data,
        speaker_to_id,
        BLOCK_SIZE
    )

    val_dataset = MultiSpeakerDataset(
        val_data,
        speaker_to_id,
        BLOCK_SIZE
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

    print(f"\nTraining chunks: {len(train_dataset)}")

    print(f"Validation chunks: {len(val_dataset)}")

    # Build model.
    model = MultiSpeakerGPT(
        num_speakers=len(speaker_to_id)
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nModel parameters: {parameters:,}")

    # Train.
    train_history, val_history = train_model(
        model,
        train_loader,
        val_loader
    )

    # Load best model.
    model.load_state_dict(
        torch.load(
            os.path.join(
                OUTPUT_DIR,
                "lesson45_best.pt"
            ),
            map_location=DEVICE,
            weights_only=True
        )
    )

    # Generate samples for each speaker.
    print("\n" + "=" * 72)

    print("SPEAKER-CONDITIONED GENERATION")

    print("=" * 72)

    for speaker_name, speaker_id in speaker_to_id.items():

        generated = generate_tokens(
            model,
            speaker_id,
            prompt=[10, 20, 30],
            max_new_tokens=50
        )

        print(f"\n{speaker_name}:")

        print("Generated tokens:", generated[:30])

    # Visualizations.
    plot_training(
        train_history,
        val_history
    )

    plot_speaker_embeddings(model)

    # Save metadata.
    metadata = {
        "vocab_size": VOCAB_SIZE,
        "block_size": BLOCK_SIZE,
        "embedding_dim": EMBED_DIM,
        "num_speakers": len(speaker_to_id),
        "speaker_to_id": speaker_to_id
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "lesson45_metadata.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\n" + "=" * 72)

    print("LESSON 45 COMPLETED")

    print("=" * 72)

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("  lesson45_best.pt")

    print("  lesson45_metadata.json")

    print("  lesson45_training_loss.png")

    print("  lesson45_speaker_embeddings.png")


if __name__ == "__main__":

    main()