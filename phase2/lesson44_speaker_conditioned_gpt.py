"""
PHASE 2 — LESSON 44
SPEAKER-CONDITIONED SPEECH TOKEN LANGUAGE MODEL
============================================================

PURPOSE
-------
Extend Lesson 43 GPT with speaker identity conditioning.

ARCHITECTURE
------------
Speech Token IDs
       |
       v
Token Embedding ------------------+
                                  |
Speaker ID                        |
       |                          |
       v                          |
Speaker Embedding                 |
       |                          |
       +---------- Add ------------+
                    |
                    v
             Position Embedding
                    |
                    v
             Transformer GPT
                    |
                    v
              Next Token
                    |
                    v
              Speech Tokens

WHAT YOU LEARN
--------------
1. Speaker embedding.
2. Speaker-conditioned token prediction.
3. Causal Transformer training.
4. Autoregressive speech-token generation.
5. Multi-speaker dataset preparation.

IMPORTANT
---------
The current dataset contains only one speaker.
The architecture supports multiple speakers, but actual
speaker differentiation requires recordings from multiple people.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# STEP 0 — CONFIGURATION
# ============================================================

SEED = 42

VOCAB_SIZE = 256
DEFAULT_NUM_SPEAKERS = 8

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1

CONTEXT_LENGTH = 128

BATCH_SIZE = 8
DEFAULT_EPOCHS = 300

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

TRAIN_DIR = Path("lesson42_output/train_tokens")
VAL_DIR = Path("lesson42_output/val_tokens")

OUTPUT_DIR = Path("lesson44_output")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# STEP 1 — LOAD TOKEN DATASET
# ============================================================

def load_token_sequences(directory):

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Token directory not found: {directory}"
        )

    files = sorted(directory.glob("*.npy"))

    if not files:
        raise RuntimeError(
            f"No .npy files found in {directory}"
        )

    sequences = []

    for file in files:

        tokens = np.load(file)
        tokens = np.asarray(tokens).reshape(-1)
        tokens = tokens.astype(np.int64)

        if len(tokens) < 2:
            print(f"Skipping short sequence: {file}")
            continue

        if tokens.min() < 0 or tokens.max() >= VOCAB_SIZE:
            raise ValueError(
                f"Invalid token ID in {file}. "
                f"Expected 0-{VOCAB_SIZE - 1}."
            )

        sequences.append({
            "name": file.stem,
            "tokens": tokens
        })

        print(
            f"Loaded {file.name}: "
            f"{len(tokens)} tokens, "
            f"{len(np.unique(tokens))} unique"
        )

    if not sequences:
        raise RuntimeError("No valid token sequences found.")

    return sequences


# ============================================================
# STEP 2 — SPEAKER ID MAPPING
# ============================================================

def load_speaker_mapping(
    train_sequences,
    val_sequences,
    num_speakers
):
    """
    Load speaker IDs from:

        lesson42_output/speaker_map.json

    If the file does not exist, all recordings are assigned
    to Speaker 0.

    Example speaker_map.json:

    {
        "sample_001": 0,
        "sample_002": 0,
        "speaker1_recording": 1,
        "speaker2_recording": 2
    }
    """

    mapping_file = Path(
        "lesson42_output/speaker_map.json"
    )

    if mapping_file.exists():

        with open(
            mapping_file,
            "r",
            encoding="utf-8"
        ) as f:

            speaker_map = json.load(f)

        print("\nLoaded speaker_map.json")

    else:

        speaker_map = {}

        for item in train_sequences + val_sequences:
            speaker_map[item["name"]] = 0

        print(
            "\nNo speaker_map.json found. "
            "Assigning all recordings to Speaker 0."
        )

    for item in train_sequences + val_sequences:

        name = item["name"]

        if name not in speaker_map:
            raise ValueError(
                f"Missing speaker ID for recording: {name}"
            )

        speaker_id = int(speaker_map[name])

        if not 0 <= speaker_id < num_speakers:
            raise ValueError(
                f"Speaker ID {speaker_id} is invalid. "
                f"Valid range: 0-{num_speakers - 1}."
            )

        item["speaker_id"] = speaker_id

    return speaker_map


# ============================================================
# STEP 3 — CREATE TRAINING WINDOWS
# ============================================================

class SpeechTokenDataset(Dataset):

    def __init__(
        self,
        sequences,
        context_length
    ):

        self.samples = []
        self.context_length = context_length

        for item in sequences:

            tokens = item["tokens"]
            speaker_id = item["speaker_id"]

            # Divide each recording into non-overlapping windows.

            for start in range(
                0,
                len(tokens) - 1,
                context_length
            ):

                chunk = tokens[
                    start:start + context_length + 1
                ]

                if len(chunk) < 2:
                    continue

                self.samples.append({
                    "tokens": chunk,
                    "speaker_id": speaker_id
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        tokens = sample["tokens"]
        speaker_id = sample["speaker_id"]

        x = tokens[:-1]
        y = tokens[1:]

        valid_length = len(x)

        x_padded = np.zeros(
            self.context_length,
            dtype=np.int64
        )

        y_padded = np.full(
            self.context_length,
            -100,
            dtype=np.int64
        )

        padding_mask = np.ones(
            self.context_length,
            dtype=bool
        )

        x_padded[:valid_length] = x
        y_padded[:valid_length] = y

        padding_mask[:valid_length] = False

        return {
            "input": torch.tensor(
                x_padded,
                dtype=torch.long
            ),

            "target": torch.tensor(
                y_padded,
                dtype=torch.long
            ),

            "speaker_id": torch.tensor(
                speaker_id,
                dtype=torch.long
            ),

            "padding_mask": torch.tensor(
                padding_mask,
                dtype=torch.bool
            )
        }


# ============================================================
# STEP 4 — SPEAKER-CONDITIONED GPT
# ============================================================

class SpeakerConditionedGPT(nn.Module):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        num_speakers=DEFAULT_NUM_SPEAKERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        context_length=CONTEXT_LENGTH,
        dropout=DROPOUT
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.num_speakers = num_speakers
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        # Speech token embedding

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Speaker identity embedding

        self.speaker_embedding = nn.Embedding(
            num_speakers,
            d_model
        )

        # Positional embedding

        self.position_embedding = nn.Embedding(
            context_length,
            d_model
        )

        # Transformer

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Next-token prediction

        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(
        self,
        input_tokens,
        speaker_ids,
        padding_mask=None
    ):

        batch_size, seq_length = input_tokens.shape

        if seq_length > self.context_length:
            raise ValueError(
                "Input sequence exceeds context length."
            )

        positions = torch.arange(
            seq_length,
            device=input_tokens.device
        )

        # Token representation

        token_embeddings = self.token_embedding(
            input_tokens
        )

        # Speaker representation

        speaker_embeddings = self.speaker_embedding(
            speaker_ids
        )

        speaker_embeddings = speaker_embeddings.unsqueeze(1)

        # Position representation

        position_embeddings = self.position_embedding(
            positions
        ).unsqueeze(0)

        # Combine token, speaker and position information

        x = (
            token_embeddings
            + speaker_embeddings
            + position_embeddings
        )

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # Causal attention mask.
        # Future tokens cannot be accessed.

        causal_mask = torch.triu(
            torch.ones(
                seq_length,
                seq_length,
                device=input_tokens.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        x = self.final_norm(x)

        logits = self.output_layer(x)

        return logits


# ============================================================
# STEP 5 — TRAINING AND VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer=None
):

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in loader:

        input_tokens = batch["input"].to(DEVICE)
        targets = batch["target"].to(DEVICE)
        speaker_ids = batch["speaker_id"].to(DEVICE)
        padding_mask = batch["padding_mask"].to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            logits = model(
                input_tokens,
                speaker_ids,
                padding_mask
            )

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                ignore_index=-100
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        valid_mask = targets != -100

        predictions = logits.argmax(dim=-1)

        correct = (
            (predictions == targets) & valid_mask
        ).sum().item()

        token_count = valid_mask.sum().item()

        total_loss += loss.item() * token_count
        total_correct += correct
        total_tokens += token_count

    average_loss = total_loss / max(total_tokens, 1)

    accuracy = total_correct / max(total_tokens, 1)

    return average_loss, accuracy


# ============================================================
# STEP 6 — AUTOREGRESSIVE TOKEN GENERATION
# ============================================================

@torch.no_grad()
def generate_tokens(
    model,
    prompt,
    speaker_id,
    max_new_tokens=128,
    temperature=1.0,
    top_k=20
):

    model.eval()

    generated = torch.tensor(
        prompt,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    speaker = torch.tensor(
        [speaker_id],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(max_new_tokens):

        current = generated[:, -model.context_length:]

        logits = model(
            current,
            speaker
        )

        next_logits = logits[:, -1, :]

        next_logits = next_logits / max(
            temperature,
            1e-5
        )

        if top_k is not None:

            values, _ = torch.topk(
                next_logits,
                min(top_k, model.vocab_size)
            )

            threshold = values[:, -1].unsqueeze(-1)

            next_logits = next_logits.masked_fill(
                next_logits < threshold,
                float("-inf")
            )

        probabilities = F.softmax(
            next_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

    return generated.squeeze(0).cpu().numpy()


# ============================================================
# STEP 7 — PLOT RESULTS
# ============================================================

def plot_training_history(history, output_dir):

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Speaker-Conditioned GPT Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)

    plt.plot(
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Next Token Prediction Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "training_curves.png",
        dpi=150
    )

    plt.close()


def plot_speaker_embeddings(model, output_dir):

    weights = (
        model.speaker_embedding.weight
        .detach()
        .cpu()
        .numpy()
    )

    plt.figure(figsize=(10, 5))

    plt.imshow(
        weights,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(label="Embedding Value")

    plt.xlabel("Embedding Dimension")
    plt.ylabel("Speaker ID")

    plt.title("Speaker Embedding Table")

    plt.tight_layout()

    plt.savefig(
        output_dir / "speaker_embeddings.png",
        dpi=150
    )

    plt.close()


# ============================================================
# STEP 8 — MAIN TRAINING PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        type=str,
        default=str(TRAIN_DIR)
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        default=str(VAL_DIR)
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR)
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        "--num_speakers",
        type=int,
        default=DEFAULT_NUM_SPEAKERS
    )

    args = parser.parse_args()

    if args.num_speakers < 1:
        raise ValueError("--num_speakers must be at least 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("PHASE 2 — LESSON 44")
    print("SPEAKER-CONDITIONED SPEECH TOKEN LANGUAGE MODEL")
    print("=" * 68)

    print(f"Device: {DEVICE}")
    print(f"Vocabulary: {VOCAB_SIZE}")
    print(f"Speakers configured: {args.num_speakers}")
    print(f"Context length: {CONTEXT_LENGTH}")
    print(f"Transformer layers: {N_LAYERS}")
    print(f"Attention heads: {N_HEADS}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading training tokens...")

    train_sequences = load_token_sequences(
        args.train_dir
    )

    print("\nLoading validation tokens...")

    val_sequences = load_token_sequences(
        args.val_dir
    )

    # --------------------------------------------------------
    # Speaker mapping
    # --------------------------------------------------------

    speaker_map = load_speaker_mapping(
        train_sequences,
        val_sequences,
        args.num_speakers
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = SpeechTokenDataset(
        train_sequences,
        CONTEXT_LENGTH
    )

    val_dataset = SpeechTokenDataset(
        val_sequences,
        CONTEXT_LENGTH
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")

    print("\nDataset summary")
    print("-" * 50)

    print(f"Training recordings: {len(train_sequences)}")
    print(f"Validation recordings: {len(val_sequences)}")
    print(f"Training windows: {len(train_dataset)}")
    print(f"Validation windows: {len(val_dataset)}")

    print("\nSpeaker assignments:")

    for name, speaker_id in speaker_map.items():
        print(f"  {name}: Speaker {speaker_id}")

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

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = SpeakerConditionedGPT(
        vocab_size=VOCAB_SIZE,
        num_speakers=args.num_speakers,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        context_length=CONTEXT_LENGTH
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nModel parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_epoch = 0

    print("\nStarting training...")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            optimizer
        )

        val_loss, val_accuracy = run_epoch(
            model,
            val_loader
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": VOCAB_SIZE,
                    "num_speakers": args.num_speakers,
                    "d_model": D_MODEL,
                    "n_heads": N_HEADS,
                    "n_layers": N_LAYERS,
                    "context_length": CONTEXT_LENGTH,
                    "speaker_map": speaker_map,
                    "epoch": epoch,
                    "val_loss": val_loss
                },
                output_dir / "best_model.pt"
            )

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"Train Loss {train_loss:.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"Train Acc {train_accuracy:.3f} | "
                f"Val Acc {val_accuracy:.3f}"
            )

    print("-" * 68)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # --------------------------------------------------------
    # Save final model and metadata
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": VOCAB_SIZE,
            "num_speakers": args.num_speakers,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "context_length": CONTEXT_LENGTH,
            "speaker_map": speaker_map
        },
        output_dir / "final_model.pt"
    )

    with open(
        output_dir / "speaker_map.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            speaker_map,
            f,
            indent=4
        )

    with open(
        output_dir / "training_history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Generate tokens using best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        output_dir / "best_model.pt",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    first_sequence = train_sequences[0]["tokens"]

    prompt_length = min(16, len(first_sequence))

    prompt = first_sequence[:prompt_length]

    speaker_id = train_sequences[0]["speaker_id"]

    print("\nGenerating speech tokens...")
    print(f"Speaker ID: {speaker_id}")
    print(f"Prompt length: {len(prompt)}")

    generated = generate_tokens(
        model,
        prompt=prompt,
        speaker_id=speaker_id,
        max_new_tokens=128,
        temperature=1.0,
        top_k=20
    )

    np.save(
        output_dir / "generated_speech_tokens.npy",
        generated
    )

    print(f"Generated sequence length: {len(generated)}")

    print(
        "Generated token preview:",
        generated[:50].tolist()
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    plot_training_history(
        history,
        output_dir
    )

    plot_speaker_embeddings(
        model,
        output_dir
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("LESSON 44 COMPLETED")
    print("=" * 68)

    print(f"Best checkpoint: {output_dir / 'best_model.pt'}")
    print(f"Final checkpoint: {output_dir / 'final_model.pt'}")
    print(f"Speaker mapping: {output_dir / 'speaker_map.json'}")

    print(
        f"Generated tokens: "
        f"{output_dir / 'generated_speech_tokens.npy'}"
    )

    print(
        f"Training curves: "
        f"{output_dir / 'training_curves.png'}"
    )

    print(
        f"Speaker embeddings: "
        f"{output_dir / 'speaker_embeddings.png'}"
    )

    print("\nIMPORTANT:")
    print(
        "Only Speaker 0 currently has training recordings."
    )

    print(
        "Additional speaker IDs need actual recordings "
        "and correct labels."
    )


if __name__ == "__main__":
    main()