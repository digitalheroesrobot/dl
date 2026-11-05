
"""
File: lesson40_train_speech_token_lm.py

========================================================================
LESSON 40 — TRAIN AN AUTOREGRESSIVE MODEL ON REAL SPEECH TOKENS
========================================================================

PURPOSE
-------
Train a causal Transformer to predict the next speech token from previous
speech tokens.

INPUT
-----
The output directory from Lesson 39:

    lesson39_tokens/
        train.npy
        val.npy
        metadata.json

Each .npy file contains fixed-length rows of discrete token IDs.

CORE IDEA
---------
Autoregressive modeling estimates:

    P(token_t | token_0, token_1, ..., token_(t-1))

The model receives a sequence shifted one position to the right and learns
to predict the next token at every position.

Example:

    Input:   [12, 45, 81, 22]
    Target:  [45, 81, 22, 93]

The Transformer uses causal attention. A token can attend to previous
tokens, but it cannot see future tokens.

LESSON STEPS
------------

STEP 1 — Load token datasets
    Read train.npy and val.npy from Lesson 39.
    Verify dimensions, token IDs, and vocabulary range.

STEP 2 — Build autoregressive windows
    Convert each recording into overlapping input/target sequences.
    This allows a small recording to provide multiple training examples.

STEP 3 — Build a causal Transformer
    Token embedding + positional embedding + Transformer encoder layers.
    Apply a causal attention mask to prevent future-token leakage.

STEP 4 — Train with next-token prediction
    Use cross-entropy loss and AdamW optimization.

STEP 5 — Evaluate
    Measure validation loss and token prediction accuracy.

STEP 6 — Generate tokens
    Give the model a short token prompt and autoregressively sample
    additional token IDs.

STEP 7 — Save artifacts
    Save the model checkpoint, training history, generated tokens,
    and plots.

IMPORTANT LIMITATIONS
---------------------
1. Lesson 39 currently uses a demonstration mel-to-token tokenizer.
   These IDs are NOT learned neural speech codec tokens.

2. Your current dataset has only one 256-token training chunk and one
   256-token validation chunk. Overlapping windows make this lesson
   executable, but they do not create additional independent speech data.

3. If train and validation contain identical recordings, validation
   results are not a meaningful measure of generalization.

4. Generated token IDs cannot yet be converted into realistic speech.
   A trained neural speech codec encoder/decoder is required.

5. This is an autoregressive token-model experiment, not a complete
   text-to-speech or voice-cloning system.

REAL-WORLD CONNECTION
---------------------
A production speech-token language model can predict acoustic tokens
conditioned on text, speaker identity, and previous acoustic tokens.
A neural codec decoder then converts the predicted tokens into audio.

This lesson trains only the autoregressive token-prediction component.

RUN
---
python lesson40_train_speech_token_lm.py

Optional:
python lesson40_train_speech_token_lm.py --epochs 100 --context-length 64

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULT_DATA_DIR = "lesson39_tokens"
DEFAULT_OUTPUT_DIR = "lesson40_output"

DEFAULT_VOCAB_SIZE = 256
DEFAULT_CONTEXT_LENGTH = 64

DEFAULT_D_MODEL = 128
DEFAULT_N_HEADS = 4
DEFAULT_N_LAYERS = 3
DEFAULT_DROPOUT = 0.10

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4

DEFAULT_GENERATION_LENGTH = 128
DEFAULT_SEED = 42


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
# DATA LOADING
# ======================================================================

def load_token_array(
    path: Path,
    vocab_size: int,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Token dataset not found: {path}")

    data = np.load(path, allow_pickle=False)

    if data.ndim != 2:
        raise ValueError(
            f"{path} must be a 2D array [recordings, sequence_length]. "
            f"Received shape: {data.shape}"
        )

    if data.size == 0:
        raise ValueError(f"{path} is empty.")

    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError(
            f"{path} must contain integer token IDs. "
            f"Received dtype: {data.dtype}"
        )

    data = data.astype(np.int64, copy=False)

    minimum = int(data.min())
    maximum = int(data.max())

    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"Token IDs in {path} must be in "
            f"[0, {vocab_size - 1}], but found [{minimum}, {maximum}]."
        )

    return data


def load_datasets(
    data_dir: Path,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:

    train_path = data_dir / "train.npy"
    val_path = data_dir / "val.npy"

    train_tokens = load_token_array(train_path, vocab_size)
    val_tokens = load_token_array(val_path, vocab_size)

    print("\nDataset information:")
    print(f"  Train shape: {train_tokens.shape}")
    print(f"  Validation shape: {val_tokens.shape}")
    print(f"  Vocabulary: {vocab_size}")
    print(f"  Train token range: {train_tokens.min()}–{train_tokens.max()}")
    print(f"  Val token range:   {val_tokens.min()}–{val_tokens.max()}")

    if np.array_equal(train_tokens, val_tokens):
        print("\nWARNING:")
        print("  Train and validation arrays are exactly identical.")
        print("  Validation metrics will not measure generalization.")
    else:
        print("\nTrain and validation arrays are not exactly identical.")

    metadata_path = data_dir / "metadata.json"

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            print("\nLesson 39 metadata loaded.")
            print(f"  Metadata keys: {list(metadata.keys())}")

        except (json.JSONDecodeError, OSError) as exc:
            print(f"\nMetadata could not be read: {exc}")

    return train_tokens, val_tokens


# ======================================================================
# AUTOREGRESSIVE WINDOW DATASET
# ======================================================================

def build_windows(
    token_sequences: np.ndarray,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    inputs = []
    targets = []

    for sequence in token_sequences:

        if len(sequence) <= context_length:
            continue

        # Every input is followed by its next-token target.
        #
        # Example:
        # Sequence: [10, 20, 30, 40, 50]
        # Context:  3
        #
        # Input:    [10, 20, 30]
        # Target:   [20, 30, 40]
        #
        # Input:    [20, 30, 40]
        # Target:   [30, 40, 50]

        for start in range(len(sequence) - context_length):

            input_sequence = sequence[
                start:start + context_length
            ]

            target_sequence = sequence[
                start + 1:start + context_length + 1
            ]

            inputs.append(input_sequence)
            targets.append(target_sequence)

    if not inputs:
        raise ValueError(
            f"No training windows could be created. "
            f"Sequence length must exceed context length "
            f"({context_length})."
        )

    x = torch.tensor(np.asarray(inputs), dtype=torch.long)
    y = torch.tensor(np.asarray(targets), dtype=torch.long)

    return x, y


# ======================================================================
# CAUSAL TRANSFORMER
# ======================================================================

class SpeechTokenTransformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            context_length,
            d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.dropout = nn.Dropout(dropout)

        self.output_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # Weight tying:
        # The output projection shares weights with the token embedding.
        self.output_head.weight = self.token_embedding.weight

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Input length {sequence_length} exceeds "
                f"context length {self.context_length}."
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        # True means that attention is blocked.
        #
        # The upper triangle prevents each position from observing
        # tokens that occur later in the sequence.

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )

        x = self.transformer(
            x,
            mask=causal_mask,
        )

        logits = self.output_head(x)

        return logits


# ======================================================================
# EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        predictions = logits.argmax(dim=-1)

        total_loss += loss.item() * y.numel()
        total_correct += (predictions == y).sum().item()
        total_tokens += y.numel()

    average_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)

    return average_loss, accuracy


# ======================================================================
# GENERATION
# ======================================================================

@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    prompt: torch.Tensor,
    generation_length: int,
    temperature: float = 1.0,
    top_k: int = 20,
) -> torch.Tensor:

    model.eval()

    generated = prompt.clone()

    for _ in range(generation_length):

        # Keep only the most recent context_length tokens.
        context = generated[:, -model.context_length:]

        logits = model(context)

        # Only the final position predicts the next token.
        next_token_logits = logits[:, -1, :]

        next_token_logits = next_token_logits / max(
            temperature,
            1e-5,
        )

        if top_k > 0:
            k = min(top_k, next_token_logits.size(-1))

            values, _ = torch.topk(
                next_token_logits,
                k=k,
                dim=-1,
            )

            threshold = values[:, -1].unsqueeze(-1)

            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < threshold,
                float("-inf"),
            )

        probabilities = F.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated


# ======================================================================
# PLOTTING
# ======================================================================

def save_training_plot(
    history: dict,
    output_dir: Path,
) -> None:

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Lesson 40 — Autoregressive Speech Token Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "training_loss.png",
        dpi=150,
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Token prediction accuracy")
    plt.title("Next-token Prediction Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "training_accuracy.png",
        dpi=150,
    )

    plt.close()


def save_generation_plot(
    generated_tokens: np.ndarray,
    output_dir: Path,
) -> None:

    plt.figure(figsize=(12, 4))

    plt.plot(
        generated_tokens,
        linewidth=1,
    )

    plt.xlabel("Token position")
    plt.ylabel("Token ID")
    plt.title("Generated Speech Token Sequence")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_dir / "generated_tokens.png",
        dpi=150,
    )

    plt.close()


# ======================================================================
# TRAINING
# ======================================================================

def train(args: argparse.Namespace) -> None:

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print("LESSON 40 — AUTOREGRESSIVE SPEECH TOKEN MODEL")
    print("=" * 72)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Data directory: {data_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")

    print(f"\nVocabulary size: {args.vocab_size}")
    print(f"Context length: {args.context_length}")
    print(f"Model dimension: {args.d_model}")
    print(f"Attention heads: {args.n_heads}")
    print(f"Transformer layers: {args.n_layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")

    # ------------------------------------------------------------------
    # STEP 1 — LOAD DATA
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 1 — LOAD LESSON 39 DATA")
    print("=" * 72)

    train_sequences, val_sequences = load_datasets(
        data_dir,
        args.vocab_size,
    )

    # ------------------------------------------------------------------
    # STEP 2 — BUILD WINDOWS
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 2 — BUILD AUTOREGRESSIVE WINDOWS")
    print("=" * 72)

    x_train, y_train = build_windows(
        train_sequences,
        args.context_length,
    )

    x_val, y_val = build_windows(
        val_sequences,
        args.context_length,
    )

    print(f"Train input windows: {x_train.shape}")
    print(f"Train target windows: {y_train.shape}")

    print(f"Validation input windows: {x_val.shape}")
    print(f"Validation target windows: {y_val.shape}")

    train_dataset = TensorDataset(
        x_train,
        y_train,
    )

    val_dataset = TensorDataset(
        x_val,
        y_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # STEP 3 — BUILD MODEL
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 3 — BUILD CAUSAL TRANSFORMER")
    print("=" * 72)

    model = SpeechTokenTransformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------
    # STEP 4 — TRAIN
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 4 — TRAIN NEXT-TOKEN PREDICTION")
    print("=" * 72)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = math.inf
    best_checkpoint = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, args.vocab_size),
                y.reshape(-1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            predictions = logits.argmax(dim=-1)

            total_loss += loss.item() * y.numel()
            total_correct += (predictions == y).sum().item()
            total_tokens += y.numel()

        train_loss = total_loss / max(total_tokens, 1)
        train_accuracy = total_correct / max(total_tokens, 1)

        # --------------------------------------------------------------
        # STEP 5 — VALIDATION
        # --------------------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "vocab_size": args.vocab_size,
                        "context_length": args.context_length,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "dropout": args.dropout,
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "history": history,
                },
                best_checkpoint,
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss {train_loss:.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"Train Acc {train_accuracy:.4f} | "
                f"Val Acc {val_accuracy:.4f}"
            )

    # ------------------------------------------------------------------
    # STEP 6 — GENERATE TOKENS
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 6 — AUTOREGRESSIVE TOKEN GENERATION")
    print("=" * 72)

    # Restore the best validation checkpoint.
    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    prompt_length = min(
        16,
        train_sequences.shape[1],
    )

    prompt = torch.tensor(
        train_sequences[0, :prompt_length],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    generated = generate_tokens(
        model=model,
        prompt=prompt,
        generation_length=args.generation_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_array = generated.squeeze(0).cpu().numpy()

    print(f"Prompt length: {prompt_length}")
    print(f"Generated length: {args.generation_length}")
    print(f"Total sequence length: {len(generated_array)}")

    print("\nFirst 40 generated token IDs:")
    print(generated_array[:40].tolist())

    np.save(
        output_dir / "generated_tokens.npy",
        generated_array,
    )

    # ------------------------------------------------------------------
    # STEP 7 — SAVE ARTIFACTS
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 7 — SAVE MODEL AND RESULTS")
    print("=" * 72)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            },
            "history": history,
            "best_val_loss": best_val_loss,
        },
        output_dir / "final_model.pt",
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

    save_training_plot(
        history,
        output_dir,
    )

    save_generation_plot(
        generated_array,
        output_dir,
    )

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("LESSON 40 COMPLETE")
    print("=" * 72)

    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print(f"\nFinal training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    print("\nOutput files:")

    for filename in [
        "best_model.pt",
        "final_model.pt",
        "generated_tokens.npy",
        "training_history.json",
        "training_loss.png",
        "training_accuracy.png",
        "generated_tokens.png",
    ]:
        print(f"  {output_dir / filename}")

    print("\nIMPORTANT:")
    print("  This model predicts demonstration token IDs.")
    print("  It does not yet generate realistic speech.")
    print("  Repair the Lesson 36 neural codec before production TTS training.")

    print("\nNEXT:")
    print("  Lesson 41 — Conditional Speech Token Modeling")
    print("=" * 72)


# ======================================================================
# ARGUMENTS
# ======================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train an autoregressive speech token Transformer."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=DEFAULT_D_MODEL,
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=DEFAULT_N_HEADS,
    )

    parser.add_argument(
        "--n-layers",
        type=int,
        default=DEFAULT_N_LAYERS,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--generation-length",
        type=int,
        default=DEFAULT_GENERATION_LENGTH,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    args = parse_args()

    if args.context_length < 2:
        raise ValueError("Context length must be at least 2.")

    if args.epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    train(args)
