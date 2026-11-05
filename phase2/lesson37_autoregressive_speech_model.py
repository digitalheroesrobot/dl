"""
========================================================================
LESSON 37 — AUTOREGRESSIVE SPEECH TOKEN MODEL
========================================================================

COURSE
------
Voice Cloning / TTS From Scratch

PREVIOUS LESSONS
----------------
Lesson 34:
    Speech Tokenizer

Lesson 35:
    Vector Quantization

Lesson 36:
    Neural Speech Codec

THIS LESSON
-----------
We now build the first generative model operating on discrete speech
tokens.

Instead of predicting waveform samples directly, we train a causal
Transformer to predict the next discrete speech token.

MATHEMATICAL OBJECTIVE
----------------------

Given a token sequence:

    s_1, s_2, ..., s_T

we model:

    P(s_1, ..., s_T)
        =
    product_t P(s_t | s_1, ..., s_{t-1})

Training minimizes:

    L = -sum_t log P(s_t | s_<t)

In practice this is implemented with cross entropy.

EXAMPLE
-------

Input:

    [12, 18, 18, 43]

Target:

    [18, 18, 43, 91]

The model therefore learns:

    12              -> 18
    12,18           -> 18
    12,18,18        -> 43
    12,18,18,43     -> 91


WHY DISCRETE TOKENS?
--------------------

Waveform modeling operates at the audio-sample level.

At 16 kHz:

    1 second = 16,000 samples

A 10-second waveform contains:

    160,000 samples

A speech tokenizer/codec can compress this into a much shorter sequence:

    waveform
        ↓
    encoder
        ↓
    latent representation
        ↓
    vector quantization
        ↓
    discrete tokens

The Transformer then operates on these tokens.

This is conceptually similar to how a language model operates on text
tokens.

ARCHITECTURE
------------

Speech tokens
      |
      v
Token Embedding
      |
      +
Positional Embedding
      |
      v
Causal Transformer
      |
      v
Linear Projection
      |
      v
Vocabulary logits
      |
      v
Next speech token


CAUSAL ATTENTION
----------------

For a sequence:

    [A B C D]

the attention pattern is:

    A . . .
    A B . .
    A B C .
    A B C D

A token cannot attend to future tokens.

This is essential for autoregressive generation.

REAL-WORLD CONNECTION
---------------------

This is the conceptual beginning of a TTS language model.

Later we will introduce:

    text representation
          +
    speaker representation
          +
    speech-token prediction
          |
          v
    speech tokens
          |
          v
    neural codec decoder
          |
          v
    waveform

Eventually this becomes:

    Text
      +
    Reference Voice
      ↓
    TTS Model
      ↓
    Speech Tokens
      ↓
    Neural Codec
      ↓
    Generated Speech


IMPORTANT
---------

This lesson can run without a real speech dataset.

When no external token dataset is supplied, we generate a structured
synthetic speech-token dataset.

The synthetic data is NOT intended to produce speech.

Its purpose is to verify that:

    1. token embedding works
    2. causal attention works
    3. next-token prediction works
    4. autoregressive generation works
    5. perplexity decreases

Later lessons will replace the synthetic token sequences with actual
speech tokens produced by the codec from Lesson 36.


WHAT YOU LEARN
--------------

STEP 1
    Represent speech as discrete token IDs.

STEP 2
    Construct input/target pairs.

STEP 3
    Implement token embeddings.

STEP 4
    Implement positional embeddings.

STEP 5
    Build a causal Transformer.

STEP 6
    Train with cross entropy.

STEP 7
    Measure accuracy and perplexity.

STEP 8
    Generate tokens autoregressively.

STEP 9
    Save the trained model.

STEP 10
    Understand the foundation of a TTS language model.


DEPENDENCIES
------------

    pip install torch numpy matplotlib


RUN
---

    python lesson37_autoregressive_speech_model.py

Optional:

    python lesson37_autoregressive_speech_model.py \
        --epochs 30 \
        --batch-size 64

Optional real token file:

    python lesson37_autoregressive_speech_model.py \
        --tokens speech_tokens.npy

The .npy file should contain integer token IDs with shape:

    [N, T]

or a one-dimensional token sequence:

    [T]


========================================================================
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ======================================================================
# 1. REPRODUCIBILITY
# ======================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ======================================================================
# 2. CONFIGURATION
# ======================================================================

VOCAB_SIZE = 256

SEQ_LEN = 64

EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 4

FF_DIM = 512

DROPOUT = 0.1

DEFAULT_EPOCHS = 30

DEFAULT_BATCH_SIZE = 64

DEFAULT_LR = 3e-4

DEFAULT_NUM_SEQUENCES = 6000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# 3. ARGUMENTS
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Lesson 37 - Autoregressive Speech Token Model"
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Optional .npy file containing integer speech tokens."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR
    )

    parser.add_argument(
        "--num-sequences",
        type=int,
        default=DEFAULT_NUM_SEQUENCES
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


# ======================================================================
# 4. SYNTHETIC SPEECH-TOKEN DATASET
# ======================================================================

class SyntheticSpeechTokenDataset:
    """
    Generate structured discrete token sequences.

    The important point is that the sequence is NOT random noise.

    Neighboring tokens have statistical relationships.

    This gives the Transformer something meaningful to learn.

    We create several latent "speech-like" patterns:

        pattern 0 -> ascending local token groups
        pattern 1 -> descending local token groups
        pattern 2 -> repeated motifs
        pattern 3 -> smooth transitions
    """

    def __init__(
        self,
        num_sequences,
        seq_len,
        vocab_size,
        seed=42
    ):
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)

        self.data = self._generate()

    def _generate(self):
        data = []

        for _ in range(self.num_sequences):

            sequence = []

            # Select a latent pattern.
            pattern = self.rng.integers(0, 4)

            # Initial token.
            token = int(
                self.rng.integers(
                    0,
                    self.vocab_size
                )
            )

            for t in range(self.seq_len):

                if pattern == 0:
                    # Mostly move upward.
                    step = int(
                        self.rng.choice(
                            [0, 1, 1, 2, 3]
                        )
                    )

                elif pattern == 1:
                    # Mostly move downward.
                    step = int(
                        self.rng.choice(
                            [0, -1, -1, -2, -3]
                        )
                    )

                elif pattern == 2:
                    # Repeating local motifs.
                    if t % 4 == 0:
                        step = int(
                            self.rng.choice(
                                [-3, -2, 2, 3]
                            )
                        )
                    else:
                        step = 0

                else:
                    # Smooth random walk.
                    step = int(
                        self.rng.choice(
                            [-1, 0, 0, 0, 1]
                        )
                    )

                token = (
                    token
                    + step
                ) % self.vocab_size

                # Small probability of a larger transition.
                if self.rng.random() < 0.03:
                    token = int(
                        self.rng.integers(
                            0,
                            self.vocab_size
                        )
                    )

                sequence.append(token)

            data.append(sequence)

        return np.asarray(
            data,
            dtype=np.int64
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# ======================================================================
# 5. LOAD REAL TOKEN DATA
# ======================================================================

def load_token_file(path, seq_len, vocab_size):
    """
    Load token sequences from a NumPy .npy file.

    Supported shapes:

        [T]
        [N, T]

    If a single long sequence is supplied, it is divided into chunks.
    """

    tokens = np.load(path)

    tokens = np.asarray(tokens)

    print()
    print("Loaded token file:")
    print("  Path:", path)
    print("  Original shape:", tokens.shape)

    if tokens.ndim == 1:

        usable_length = (
            len(tokens)
            // seq_len
        ) * seq_len

        tokens = tokens[:usable_length]

        tokens = tokens.reshape(
            -1,
            seq_len
        )

    elif tokens.ndim == 2:

        if tokens.shape[1] < seq_len:
            raise ValueError(
                f"Token sequence length {tokens.shape[1]} "
                f"is smaller than SEQ_LEN={seq_len}"
            )

        # Break longer sequences into fixed-size chunks.
        sequences = []

        for row in tokens:

            usable_length = (
                len(row)
                // seq_len
            ) * seq_len

            row = row[:usable_length]

            if usable_length > 0:

                chunks = row.reshape(
                    -1,
                    seq_len
                )

                sequences.append(chunks)

        if not sequences:
            raise ValueError(
                "No usable token sequences found."
            )

        tokens = np.concatenate(
            sequences,
            axis=0
        )

    else:

        raise ValueError(
            "Token file must have shape [T] or [N, T]."
        )

    tokens = tokens.astype(
        np.int64
    )

    if tokens.min() < 0:
        raise ValueError(
            "Token IDs must be non-negative."
        )

    if tokens.max() >= vocab_size:
        raise ValueError(
            f"Token ID {tokens.max()} is outside "
            f"vocabulary size {vocab_size}."
        )

    print("  Processed shape:", tokens.shape)
    print("  Min token:", tokens.min())
    print("  Max token:", tokens.max())

    return tokens


# ======================================================================
# 6. DATASET
# ======================================================================

class TokenDataset(torch.utils.data.Dataset):

    def __init__(self, sequences):
        self.sequences = torch.tensor(
            sequences,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        sequence = self.sequences[index]

        # Autoregressive input:
        #
        #     [A B C D]
        #
        # becomes
        #
        # input:
        #     [A B C]
        #
        # target:
        #     [B C D]

        input_tokens = sequence[:-1]

        target_tokens = sequence[1:]

        return input_tokens, target_tokens


# ======================================================================
# 7. CAUSAL SELF-ATTENTION MASK
# ======================================================================

def create_causal_mask(length, device):

    """
    Create an upper-triangular mask.

    Example for length = 4:

        [[0, -inf, -inf, -inf],
         [0,    0, -inf, -inf],
         [0,    0,    0, -inf],
         [0,    0,    0,    0]]

    This prevents attention to future positions.
    """

    mask = torch.full(
        (length, length),
        float("-inf"),
        device=device
    )

    mask = torch.triu(
        mask,
        diagonal=1
    )

    return mask


# ======================================================================
# 8. TRANSFORMER BLOCK
# ======================================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embed_dim,
                ff_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                ff_dim,
                embed_dim
            ),

            nn.Dropout(
                dropout
            )
        )

    def forward(
        self,
        x,
        causal_mask
    ):

        # Pre-normalization.
        h = self.norm1(x)

        attention_output, _ = self.attention(
            h,
            h,
            h,
            attn_mask=causal_mask,
            need_weights=False
        )

        # Residual connection.
        x = x + attention_output

        # Feed-forward block.
        h = self.norm2(x)

        x = x + self.feed_forward(h)

        return x


# ======================================================================
# 9. AUTOREGRESSIVE SPEECH TRANSFORMER
# ======================================================================

class AutoregressiveSpeechTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim,
        num_heads,
        num_layers,
        ff_dim,
        dropout
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.seq_len = seq_len

        self.embed_dim = embed_dim

        # Token embedding.
        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        # Learned positional embedding.
        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                seq_len - 1,
                embed_dim
            ) * 0.02
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            embed_dim
        )

        self.output_projection = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, tokens):

        batch_size, length = tokens.shape

        if length > self.seq_len - 1:
            raise ValueError(
                "Input sequence is too long."
            )

        # --------------------------------------------------------------
        # STEP 1 — Token embeddings
        # --------------------------------------------------------------

        x = self.token_embedding(
            tokens
        )

        # --------------------------------------------------------------
        # STEP 2 — Positional embeddings
        # --------------------------------------------------------------

        x = x + self.position_embedding[
            :, :length
        ]

        x = self.dropout(x)

        # --------------------------------------------------------------
        # STEP 3 — Causal mask
        # --------------------------------------------------------------

        causal_mask = create_causal_mask(
            length,
            tokens.device
        )

        # --------------------------------------------------------------
        # STEP 4 — Transformer blocks
        # --------------------------------------------------------------

        for block in self.blocks:

            x = block(
                x,
                causal_mask
            )

        # --------------------------------------------------------------
        # STEP 5 — Final normalization
        # --------------------------------------------------------------

        x = self.final_norm(x)

        # --------------------------------------------------------------
        # STEP 6 — Vocabulary prediction
        # --------------------------------------------------------------

        logits = self.output_projection(
            x
        )

        return logits


# ======================================================================
# 10. MODEL SUMMARY
# ======================================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ======================================================================
# 11. TRAINING
# ======================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_tokens = 0

    for input_tokens, target_tokens in loader:

        input_tokens = input_tokens.to(
            DEVICE
        )

        target_tokens = target_tokens.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_tokens
        )

        # logits:
        #
        # [batch, time, vocabulary]
        #
        # target:
        #
        # [batch, time]

        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            target_tokens.reshape(-1)
        )

        loss.backward()

        # Gradient clipping helps prevent unstable
        # updates in autoregressive models.

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * target_tokens.numel()
        )

        predictions = logits.argmax(
            dim=-1
        )

        total_correct += (
            predictions
            == target_tokens
        ).sum().item()

        total_tokens += (
            target_tokens.numel()
        )

    average_loss = (
        total_loss
        / total_tokens
    )

    accuracy = (
        total_correct
        / total_tokens
    )

    perplexity = math.exp(
        min(average_loss, 20.0)
    )

    return (
        average_loss,
        accuracy,
        perplexity
    )


# ======================================================================
# 12. VALIDATION
# ======================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_tokens = 0

    for input_tokens, target_tokens in loader:

        input_tokens = input_tokens.to(
            DEVICE
        )

        target_tokens = target_tokens.to(
            DEVICE
        )

        logits = model(
            input_tokens
        )

        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            target_tokens.reshape(-1)
        )

        total_loss += (
            loss.item()
            * target_tokens.numel()
        )

        predictions = logits.argmax(
            dim=-1
        )

        total_correct += (
            predictions
            == target_tokens
        ).sum().item()

        total_tokens += (
            target_tokens.numel()
        )

    average_loss = (
        total_loss
        / total_tokens
    )

    accuracy = (
        total_correct
        / total_tokens
    )

    perplexity = math.exp(
        min(average_loss, 20.0)
    )

    return (
        average_loss,
        accuracy,
        perplexity
    )


# ======================================================================
# 13. AUTOREGRESSIVE GENERATION
# ======================================================================

@torch.no_grad()
def generate_tokens(
    model,
    start_tokens,
    num_new_tokens,
    temperature=1.0,
    top_k=20
):

    """
    Autoregressively generate speech tokens.

    Algorithm:

        current sequence
              ↓
        Transformer
              ↓
        probability distribution
              ↓
        sample next token
              ↓
        append token
              ↓
        repeat

    """

    model.eval()

    tokens = torch.tensor(
        start_tokens,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    for _ in range(num_new_tokens):

        # Keep only the model's context window.
        context = tokens[
            :, -(model.seq_len - 1):
        ]

        logits = model(
            context
        )

        # Last position predicts next token.
        next_logits = logits[:, -1, :]

        # Temperature controls randomness.
        next_logits = (
            next_logits
            / max(temperature, 1e-5)
        )

        # Optional top-k sampling.
        if top_k is not None:

            values, indices = torch.topk(
                next_logits,
                min(
                    top_k,
                    next_logits.size(-1)
                ),
                dim=-1
            )

            filtered_logits = torch.full_like(
                next_logits,
                float("-inf")
            )

            filtered_logits.scatter_(
                1,
                indices,
                values
            )

            next_logits = filtered_logits

        probabilities = torch.softmax(
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

    return tokens.squeeze(0).cpu().numpy()


# ======================================================================
# 14. VISUALIZATION
# ======================================================================

def plot_training(
    train_losses,
    val_losses,
    train_ppls,
    val_ppls
):

    epochs = np.arange(
        1,
        len(train_losses) + 1
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Cross Entropy Loss"
    )

    plt.title(
        "Lesson 37 — Autoregressive Token Modeling"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        epochs,
        train_ppls,
        label="Train Perplexity"
    )

    plt.plot(
        epochs,
        val_ppls,
        label="Validation Perplexity"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Perplexity"
    )

    plt.title(
        "Speech Token Prediction Perplexity"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# 15. VISUALIZE TOKEN SEQUENCES
# ======================================================================

def plot_token_sequences(
    original,
    generated
):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        original,
        label="Original tokens"
    )

    plt.plot(
        generated,
        label="Generated tokens"
    )

    plt.xlabel(
        "Token position"
    )

    plt.ylabel(
        "Token ID"
    )

    plt.title(
        "Speech Token Sequence"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# 16. MAIN
# ======================================================================

def main():

    args = parse_args()

    print("=" * 72)
    print("LESSON 37 — AUTOREGRESSIVE SPEECH TOKEN MODEL")
    print("=" * 72)

    print()
    print("Device:", DEVICE)
    print("Vocabulary size:", VOCAB_SIZE)
    print("Sequence length:", SEQ_LEN)
    print("Embedding dimension:", EMBED_DIM)
    print("Attention heads:", NUM_HEADS)
    print("Transformer layers:", NUM_LAYERS)
    print("Feed-forward dimension:", FF_DIM)

    # --------------------------------------------------------------
    # STEP 1 — Load or generate token sequences
    # --------------------------------------------------------------

    if args.tokens is not None:

        sequences = load_token_file(
            args.tokens,
            SEQ_LEN,
            VOCAB_SIZE
        )

    else:

        print()
        print("No external token dataset supplied.")
        print("Generating structured synthetic speech-token data.")

        generator = SyntheticSpeechTokenDataset(
            num_sequences=args.num_sequences,
            seq_len=SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            seed=args.seed
        )

        sequences = generator.data

    print()
    print("Dataset shape:", sequences.shape)

    # --------------------------------------------------------------
    # STEP 2 — Train/validation split
    # --------------------------------------------------------------

    indices = np.arange(
        len(sequences)
    )

    np.random.shuffle(
        indices
    )

    split = int(
        0.9 * len(indices)
    )

    train_indices = indices[:split]

    val_indices = indices[split:]

    train_sequences = sequences[
        train_indices
    ]

    val_sequences = sequences[
        val_indices
    ]

    print(
        "Training sequences:",
        len(train_sequences)
    )

    print(
        "Validation sequences:",
        len(val_sequences)
    )

    # --------------------------------------------------------------
    # STEP 3 — PyTorch datasets
    # --------------------------------------------------------------

    train_dataset = TokenDataset(
        train_sequences
    )

    val_dataset = TokenDataset(
        val_sequences
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------------
    # STEP 4 — Create model
    # --------------------------------------------------------------

    model = AutoregressiveSpeechTransformer(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    print()
    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------------
    # STEP 5 — Loss and optimizer
    # --------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # Cosine learning-rate schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # --------------------------------------------------------------
    # STEP 6 — Training
    # --------------------------------------------------------------

    train_losses = []
    val_losses = []

    train_ppls = []
    val_ppls = []

    train_accs = []
    val_accs = []

    best_val_loss = float("inf")

    print()
    print("=" * 72)
    print("TRAINING")
    print("=" * 72)

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_loss, train_acc, train_ppl = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_acc, val_ppl = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step()

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        train_ppls.append(
            train_ppl
        )

        val_ppls.append(
            val_ppl
        )

        train_accs.append(
            train_acc
        )

        val_accs.append(
            val_acc
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Train Acc {train_acc * 100:6.2f}% | "
            f"Val Acc {val_acc * 100:6.2f}% | "
            f"Train PPL {train_ppl:7.2f} | "
            f"Val PPL {val_ppl:7.2f} | "
            f"LR {current_lr:.2e}"
        )

        # Save best checkpoint.
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "vocab_size":
                        VOCAB_SIZE,

                    "seq_len":
                        SEQ_LEN,

                    "embed_dim":
                        EMBED_DIM,

                    "num_heads":
                        NUM_HEADS,

                    "num_layers":
                        NUM_LAYERS,

                    "ff_dim":
                        FF_DIM,

                    "best_val_loss":
                        best_val_loss
                },
                "lesson37_best.pt"
            )

    # --------------------------------------------------------------
    # STEP 7 — Load best model
    # --------------------------------------------------------------

    checkpoint = torch.load(
        "lesson37_best.pt",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print(
        "Best model restored."
    )

    print(
        "Best validation loss:",
        f"{checkpoint['best_val_loss']:.4f}"
    )

    # --------------------------------------------------------------
    # STEP 8 — Inspect prediction
    # --------------------------------------------------------------

    model.eval()

    sample_input, sample_target = val_dataset[0]

    sample_input = sample_input.unsqueeze(
        0
    ).to(DEVICE)

    sample_target = sample_target.numpy()

    with torch.no_grad():

        logits = model(
            sample_input
        )

    predictions = logits.argmax(
        dim=-1
    ).squeeze(0).cpu().numpy()

    print()
    print("=" * 72)
    print("NEXT-TOKEN PREDICTION")
    print("=" * 72)

    print()
    print("Input tokens:")
    print(
        sample_input.squeeze(0)
        .cpu()
        .numpy()[:32]
    )

    print()
    print("Target next tokens:")
    print(
        sample_target[:32]
    )

    print()
    print("Predicted next tokens:")
    print(
        predictions[:32]
    )

    # --------------------------------------------------------------
    # STEP 9 — Autoregressive generation
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("AUTOREGRESSIVE GENERATION")
    print("=" * 72)

    start_tokens = (
        val_sequences[0][:4]
        .tolist()
    )

    generated = generate_tokens(
        model=model,
        start_tokens=start_tokens,
        num_new_tokens=60,
        temperature=0.8,
        top_k=20
    )

    print()
    print("Starting tokens:")
    print(start_tokens)

    print()
    print("Generated sequence:")
    print(generated)

    # --------------------------------------------------------------
    # STEP 10 — Visualization
    # --------------------------------------------------------------

    plot_training(
        train_losses,
        val_losses,
        train_ppls,
        val_ppls
    )

    original = val_sequences[0]

    generated_for_plot = generated[
        :len(original)
    ]

    plot_token_sequences(
        original,
        generated_for_plot
    )

    # --------------------------------------------------------------
    # STEP 11 — Final report
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("LESSON 37 COMPLETE")
    print("=" * 72)

    print()
    print("The model learned:")
    print()
    print("  1. Token embeddings")
    print("  2. Positional information")
    print("  3. Causal self-attention")
    print("  4. Next-token prediction")
    print("  5. Cross-entropy training")
    print("  6. Perplexity")
    print("  7. Autoregressive generation")
    print()

    print(
        "Checkpoint:",
        os.path.abspath(
            "lesson37_best.pt"
        )
    )

    print()
    print("IMPORTANT:")
    print(
        "The generated IDs are speech-token sequences, "
        "not audio waveforms."
    )

    print()
    print(
        "Lesson 38 will improve speech-token modeling "
        "and move closer to a real codec-token language model."
    )

    print()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()