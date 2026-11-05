"""
========================================================================
LESSON 38 — IMPROVED SPEECH TOKEN LANGUAGE MODEL
========================================================================

COURSE
------
Voice Cloning / TTS From Scratch

LESSON 37
---------
We built a basic autoregressive Transformer that learned:

    P(token_t | token_1 ... token_{t-1})

LESSON 38
---------
We improve the training and generation pipeline.

MAIN IDEAS
----------

1. Better speech-token statistics

2. Label smoothing

3. Transformer pre-normalization

4. Learning-rate warmup

5. Gradient clipping

6. Temperature sampling

7. Top-k sampling

8. Top-p / nucleus sampling

9. Generation diagnostics

10. Checkpoint save/reload


CORE EQUATION
-------------

The autoregressive objective is:

    P(x_1,...,x_T)
      =
    product_t P(x_t | x_<t)

The negative log likelihood is:

    L
      =
    -sum_t log P(x_t | x_<t)

With label smoothing:

    y_smooth
      =
    (1-epsilon)y
      +
    epsilon / V

where:

    epsilon = smoothing coefficient
    V       = vocabulary size


WHY LABEL SMOOTHING?
--------------------

Without smoothing, the model can become excessively confident.

For a target class:

    target = 1

and all other classes:

    target = 0

Label smoothing changes this into approximately:

    target = 1 - epsilon
    others = epsilon / (V-1)

This can improve generalization and produce less pathological
probability distributions.


SAMPLING
--------

During generation, the model produces:

    logits

Convert them into probabilities:

    p_i = softmax(logits_i / temperature)

Temperature:

    T < 1
        more deterministic

    T = 1
        normal sampling

    T > 1
        more random


TOP-K
-----

Keep only the K most likely tokens.

Example:

    vocabulary = 256
    top_k = 20

Only the 20 highest-probability tokens remain candidates.


TOP-P
-----

Nucleus sampling keeps the smallest set of tokens whose cumulative
probability exceeds p.

Example:

    top_p = 0.9

The model keeps enough tokens to represent 90% of the probability mass.


IMPORTANT
---------

This lesson still uses a synthetic structured token dataset by default.

That is intentional.

We first verify the language-model machinery.

Later we replace this dataset with real codec tokens generated from
speech.


REAL TTS PIPELINE
-----------------

                 AUDIO
                   |
                   v
            Speech Encoder
                   |
                   v
            Vector Quantizer
                   |
                   v
             Speech Tokens
                   |
                   v
        Autoregressive Model
                   |
                   v
          Generated Tokens
                   |
                   v
            Neural Decoder
                   |
                   v
                AUDIO


WHAT YOU LEARN
--------------

STEP 1
    Analyze token distributions.

STEP 2
    Train a stronger autoregressive model.

STEP 3
    Understand label smoothing.

STEP 4
    Implement learning-rate warmup.

STEP 5
    Use gradient clipping.

STEP 6
    Understand temperature.

STEP 7
    Understand top-k.

STEP 8
    Understand top-p.

STEP 9
    Generate token sequences.

STEP 10
    Save and reload a trained language model.

========================================================================
"""

import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

SEQ_LEN = 96

EMBED_DIM = 192

NUM_HEADS = 6

NUM_LAYERS = 6

FF_DIM = 768

DROPOUT = 0.10

LABEL_SMOOTHING = 0.05

WARMUP_STEPS = 500

DEFAULT_EPOCHS = 30

DEFAULT_BATCH_SIZE = 64

DEFAULT_LR = 3e-4

DEFAULT_NUM_SEQUENCES = 8000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# 3. ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 38 - Improved Speech Token "
            "Language Model"
        )
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Optional .npy speech-token file."
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
        default=SEED
    )

    return parser.parse_args()


# ======================================================================
# 4. SYNTHETIC SPEECH TOKEN DATASET
# ======================================================================

class SpeechTokenDatasetGenerator:

    """
    Generate structured token sequences.

    We intentionally create temporal correlations.

    Real speech tokens also contain temporal structure:

        phonetic continuity
        acoustic continuity
        rhythm
        repeated patterns
        local transitions

    The synthetic dataset allows us to verify the language model
    independently from the codec.
    """

    def __init__(
        self,
        num_sequences,
        seq_len,
        vocab_size,
        seed
    ):

        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.rng = np.random.default_rng(
            seed
        )

    def generate(self):

        sequences = []

        for _ in range(
            self.num_sequences
        ):

            sequence = []

            # Latent mode.
            mode = int(
                self.rng.integers(
                    0,
                    5
                )
            )

            token = int(
                self.rng.integers(
                    0,
                    self.vocab_size
                )
            )

            motif = self.rng.integers(
                0,
                self.vocab_size,
                size=8
            )

            for t in range(
                self.seq_len
            ):

                if mode == 0:

                    step = int(
                        self.rng.choice(
                            [-1, 0, 0, 1, 1, 2]
                        )
                    )

                elif mode == 1:

                    step = int(
                        self.rng.choice(
                            [-2, -1, -1, 0, 0, 1]
                        )
                    )

                elif mode == 2:

                    token = int(
                        motif[t % len(motif)]
                    )

                    step = 0

                elif mode == 3:

                    step = int(
                        self.rng.choice(
                            [-3, -2, -1, 0, 1, 2, 3]
                        )
                    )

                else:

                    step = int(
                        self.rng.choice(
                            [-1, 0, 1]
                        )
                    )

                    if t % 12 == 0:

                        step += int(
                            self.rng.choice(
                                [-8, 8]
                            )
                        )

                token = (
                    token + step
                ) % self.vocab_size

                # Occasional long-range transition.
                if self.rng.random() < 0.02:

                    token = int(
                        self.rng.integers(
                            0,
                            self.vocab_size
                        )
                    )

                sequence.append(
                    token
                )

            sequences.append(
                sequence
            )

        return np.asarray(
            sequences,
            dtype=np.int64
        )


# ======================================================================
# 5. LOAD REAL TOKEN DATA
# ======================================================================

def load_real_tokens(
    path,
    seq_len,
    vocab_size
):

    tokens = np.load(path)

    tokens = np.asarray(
        tokens
    )

    print()
    print("Loading token dataset")
    print("Path:", path)
    print("Original shape:", tokens.shape)

    if tokens.ndim == 1:

        usable = (
            len(tokens)
            // seq_len
        ) * seq_len

        tokens = tokens[:usable]

        tokens = tokens.reshape(
            -1,
            seq_len
        )

    elif tokens.ndim == 2:

        chunks = []

        for row in tokens:

            usable = (
                len(row)
                // seq_len
            ) * seq_len

            if usable == 0:
                continue

            row = row[:usable]

            chunks.append(
                row.reshape(
                    -1,
                    seq_len
                )
            )

        if len(chunks) == 0:

            raise ValueError(
                "No usable token sequences."
            )

        tokens = np.concatenate(
            chunks,
            axis=0
        )

    else:

        raise ValueError(
            "Expected token array with "
            "shape [T] or [N,T]."
        )

    tokens = tokens.astype(
        np.int64
    )

    if tokens.min() < 0:

        raise ValueError(
            "Negative token IDs detected."
        )

    if tokens.max() >= vocab_size:

        raise ValueError(
            f"Token ID {tokens.max()} exceeds "
            f"vocabulary size {vocab_size}."
        )

    print(
        "Processed shape:",
        tokens.shape
    )

    return tokens


# ======================================================================
# 6. TOKEN DATASET
# ======================================================================

class TokenDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        sequences
    ):

        self.tokens = torch.tensor(
            sequences,
            dtype=torch.long
        )

    def __len__(self):

        return len(
            self.tokens
        )

    def __getitem__(
        self,
        index
    ):

        sequence = self.tokens[index]

        return (
            sequence[:-1],
            sequence[1:]
        )


# ======================================================================
# 7. TOKEN STATISTICS
# ======================================================================

def print_token_statistics(
    sequences,
    vocab_size
):

    flattened = sequences.reshape(-1)

    counts = np.bincount(
        flattened,
        minlength=vocab_size
    )

    probabilities = (
        counts
        / counts.sum()
    )

    nonzero = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        nonzero
        * np.log2(nonzero)
    )

    print()
    print("=" * 72)
    print("TOKEN STATISTICS")
    print("=" * 72)

    print(
        "Total tokens:",
        len(flattened)
    )

    print(
        "Unique tokens:",
        np.count_nonzero(counts)
    )

    print(
        "Token entropy:",
        f"{entropy:.3f} bits"
    )

    print(
        "Most frequent token:",
        int(np.argmax(counts))
    )

    print(
        "Most frequent count:",
        int(counts.max())
    )

    plt.figure(
        figsize=(11, 4)
    )

    plt.bar(
        np.arange(vocab_size),
        counts
    )

    plt.xlabel(
        "Token ID"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Speech Token Distribution"
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# 8. CAUSAL MASK
# ======================================================================

def causal_mask(
    length,
    device
):

    mask = torch.triu(
        torch.ones(
            length,
            length,
            device=device,
            dtype=torch.bool
        ),
        diagonal=1
    )

    return mask


# ======================================================================
# 9. TRANSFORMER BLOCK
# ======================================================================

class TransformerBlock(
    nn.Module
):

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

        self.ffn = nn.Sequential(

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
        mask
    ):

        h = self.norm1(
            x
        )

        attention_output, _ = (
            self.attention(
                h,
                h,
                h,
                attn_mask=mask,
                need_weights=False
            )
        )

        x = (
            x
            + attention_output
        )

        h = self.norm2(
            x
        )

        x = (
            x
            + self.ffn(h)
        )

        return x


# ======================================================================
# 10. IMPROVED SPEECH TOKEN TRANSFORMER
# ======================================================================

class SpeechTokenTransformer(
    nn.Module
):

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

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

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
                    embed_dim,
                    num_heads,
                    ff_dim,
                    dropout
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.final_norm = nn.LayerNorm(
            embed_dim
        )

        self.output = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False
        )

        # Weight tying.
        #
        # The output projection and token embedding
        # use the same matrix.
        #
        # This is common in language models.

        self.output.weight = (
            self.token_embedding.weight
        )

    def forward(
        self,
        tokens
    ):

        batch_size, length = (
            tokens.shape
        )

        if length > self.seq_len - 1:

            raise ValueError(
                "Sequence exceeds model context."
            )

        x = self.token_embedding(
            tokens
        )

        x = (
            x
            + self.position_embedding[
                :, :length
            ]
        )

        x = self.dropout(
            x
        )

        mask = causal_mask(
            length,
            tokens.device
        )

        for block in self.blocks:

            x = block(
                x,
                mask
            )

        x = self.final_norm(
            x
        )

        logits = self.output(
            x
        )

        # Weight tying scales logits significantly.
        # A standard initialization correction is useful.
        logits = (
            logits
            / math.sqrt(self.embed_dim)
        )

        return logits


# ======================================================================
# 11. LABEL-SMOOTHING CROSS ENTROPY
# ======================================================================

def smoothed_cross_entropy(
    logits,
    targets,
    smoothing
):

    """
    Manual label-smoothed cross entropy.

    logits:
        [N, V]

    targets:
        [N]
    """

    log_probs = F.log_softmax(
        logits,
        dim=-1
    )

    n_classes = (
        logits.size(-1)
    )

    with torch.no_grad():

        true_distribution = torch.full_like(
            log_probs,
            smoothing
            / (n_classes - 1)
        )

        true_distribution.scatter_(
            1,
            targets.unsqueeze(1),
            1.0 - smoothing
        )

    loss = (
        -true_distribution
        * log_probs
    ).sum(
        dim=-1
    ).mean()

    return loss


# ======================================================================
# 12. LEARNING RATE WARMUP
# ======================================================================

def update_learning_rate(
    optimizer,
    base_lr,
    step,
    warmup_steps
):

    if step < warmup_steps:

        scale = (
            (step + 1)
            / warmup_steps
        )

    else:

        scale = 1.0

    lr = (
        base_lr
        * scale
    )

    for group in optimizer.param_groups:

        group["lr"] = lr

    return lr


# ======================================================================
# 13. TRAINING
# ======================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    global_step,
    base_lr
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_tokens = 0

    for input_tokens, targets in loader:

        input_tokens = (
            input_tokens.to(
                DEVICE,
                non_blocking=True
            )
        )

        targets = (
            targets.to(
                DEVICE,
                non_blocking=True
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_tokens
        )

        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            targets.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        global_step += 1

        lr = update_learning_rate(
            optimizer,
            base_lr,
            global_step,
            WARMUP_STEPS
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * targets.numel()
        )

        predictions = logits.argmax(
            dim=-1
        )

        total_correct += (
            predictions
            == targets
        ).sum().item()

        total_tokens += (
            targets.numel()
        )

    loss = (
        total_loss
        / total_tokens
    )

    accuracy = (
        total_correct
        / total_tokens
    )

    perplexity = math.exp(
        min(loss, 20.0)
    )

    return (
        loss,
        accuracy,
        perplexity,
        global_step,
        lr
    )


# ======================================================================
# 14. EVALUATION
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

    for input_tokens, targets in loader:

        input_tokens = (
            input_tokens.to(
                DEVICE
            )
        )

        targets = (
            targets.to(
                DEVICE
            )
        )

        logits = model(
            input_tokens
        )

        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            targets.reshape(-1)
        )

        total_loss += (
            loss.item()
            * targets.numel()
        )

        predictions = logits.argmax(
            dim=-1
        )

        total_correct += (
            predictions
            == targets
        ).sum().item()

        total_tokens += (
            targets.numel()
        )

    loss = (
        total_loss
        / total_tokens
    )

    accuracy = (
        total_correct
        / total_tokens
    )

    perplexity = math.exp(
        min(loss, 20.0)
    )

    return (
        loss,
        accuracy,
        perplexity
    )


# ======================================================================
# 15. TOP-K + TOP-P SAMPLING
# ======================================================================

def sample_next_token(
    logits,
    temperature=1.0,
    top_k=20,
    top_p=0.9
):

    logits = (
        logits
        / max(
            temperature,
            1e-5
        )
    )

    # --------------------------------------------------------------
    # TOP-K
    # --------------------------------------------------------------

    if top_k is not None:

        top_k = min(
            top_k,
            logits.size(-1)
        )

        values, indices = torch.topk(
            logits,
            top_k,
            dim=-1
        )

        filtered = torch.full_like(
            logits,
            float("-inf")
        )

        filtered.scatter_(
            1,
            indices,
            values
        )

        logits = filtered

    # --------------------------------------------------------------
    # TOP-P
    # --------------------------------------------------------------

    if top_p is not None:

        sorted_logits, sorted_indices = (
            torch.sort(
                logits,
                descending=True,
                dim=-1
            )
        )

        sorted_probs = torch.softmax(
            sorted_logits,
            dim=-1
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1
        )

        remove = (
            cumulative_probs
            > top_p
        )

        # Always retain at least one token.
        remove[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove,
            float("-inf")
        )

        logits = torch.full_like(
            logits,
            float("-inf")
        )

        logits.scatter_(
            1,
            sorted_indices,
            sorted_logits
        )

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return next_token


# ======================================================================
# 16. AUTOREGRESSIVE GENERATION
# ======================================================================

@torch.no_grad()
def generate(
    model,
    start_tokens,
    num_new_tokens,
    temperature,
    top_k,
    top_p
):

    model.eval()

    tokens = torch.tensor(
        start_tokens,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    for _ in range(
        num_new_tokens
    ):

        context = tokens[
            :,
            -(model.seq_len - 1):
        ]

        logits = model(
            context
        )

        next_logits = logits[
            :, -1, :
        ]

        next_token = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        tokens = torch.cat(
            [
                tokens,
                next_token
            ],
            dim=1
        )

    return tokens.squeeze(
        0
    ).cpu().numpy()


# ======================================================================
# 17. GENERATION ENTROPY
# ======================================================================

@torch.no_grad()
def measure_generation_entropy(
    model,
    tokens
):

    model.eval()

    x = torch.tensor(
        tokens[:-1],
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    logits = model(
        x
    )

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    entropy = -(
        probabilities
        * torch.log(
            probabilities + 1e-9
        )
    ).sum(
        dim=-1
    )

    return (
        entropy.mean().item()
    )


# ======================================================================
# 18. PLOTS
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
        label="Train loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Lesson 38 — Token Language Model Loss"
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
        label="Train perplexity"
    )

    plt.plot(
        epochs,
        val_ppls,
        label="Validation perplexity"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Perplexity"
    )

    plt.title(
        "Lesson 38 — Token Language Model Perplexity"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# 19. MAIN
# ======================================================================

def main():

    args = parse_args()

    print("=" * 72)
    print("LESSON 38 — IMPROVED SPEECH TOKEN LANGUAGE MODEL")
    print("=" * 72)

    print()
    print("Device:", DEVICE)
    print("Vocabulary:", VOCAB_SIZE)
    print("Context length:", SEQ_LEN)
    print("Embedding dimension:", EMBED_DIM)
    print("Heads:", NUM_HEADS)
    print("Layers:", NUM_LAYERS)
    print("FF dimension:", FF_DIM)
    print("Label smoothing:", LABEL_SMOOTHING)

    # --------------------------------------------------------------
    # STEP 1 — DATA
    # --------------------------------------------------------------

    if args.tokens is not None:

        sequences = load_real_tokens(
            args.tokens,
            SEQ_LEN,
            VOCAB_SIZE
        )

    else:

        print()
        print(
            "No real token file supplied."
        )

        print(
            "Generating structured synthetic "
            "speech-token sequences."
        )

        generator = SpeechTokenDatasetGenerator(
            num_sequences=args.num_sequences,
            seq_len=SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            seed=args.seed
        )

        sequences = generator.generate()

    print(
        "Dataset:",
        sequences.shape
    )

    # --------------------------------------------------------------
    # STEP 2 — TOKEN STATISTICS
    # --------------------------------------------------------------

    print_token_statistics(
        sequences,
        VOCAB_SIZE
    )

    # --------------------------------------------------------------
    # STEP 3 — TRAIN/VALIDATION SPLIT
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

    train_sequences = sequences[
        indices[:split]
    ]

    val_sequences = sequences[
        indices[split:]
    ]

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
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # --------------------------------------------------------------
    # STEP 4 — MODEL
    # --------------------------------------------------------------

    model = SpeechTokenTransformer(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Trainable parameters:",
        f"{parameters:,}"
    )

    # --------------------------------------------------------------
    # STEP 5 — OPTIMIZER
    # --------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    criterion = lambda logits, targets: (
        smoothed_cross_entropy(
            logits,
            targets,
            LABEL_SMOOTHING
        )
    )

    # --------------------------------------------------------------
    # STEP 6 — TRAIN
    # --------------------------------------------------------------

    train_losses = []
    val_losses = []

    train_ppls = []
    val_ppls = []

    global_step = 0

    best_val_loss = float(
        "inf"
    )

    print()
    print("=" * 72)
    print("TRAINING")
    print("=" * 72)

    for epoch in range(
        1,
        args.epochs + 1
    ):

        (
            train_loss,
            train_acc,
            train_ppl,
            global_step,
            lr
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            global_step,
            args.lr
        )

        (
            val_loss,
            val_acc,
            val_ppl
        ) = evaluate(
            model,
            val_loader,
            criterion
        )

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

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Train Acc {train_acc*100:6.2f}% | "
            f"Val Acc {val_acc*100:6.2f}% | "
            f"Train PPL {train_ppl:7.2f} | "
            f"Val PPL {val_ppl:7.2f} | "
            f"LR {lr:.2e}"
        )

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
                "lesson38_best.pt"
            )

    # --------------------------------------------------------------
    # STEP 7 — RESTORE BEST MODEL
    # --------------------------------------------------------------

    checkpoint = torch.load(
        "lesson38_best.pt",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print()
    print(
        "Best checkpoint restored."
    )

    print(
        "Best validation loss:",
        f"{checkpoint['best_val_loss']:.4f}"
    )

    # --------------------------------------------------------------
    # STEP 8 — AUTOREGRESSIVE GENERATION
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("AUTOREGRESSIVE GENERATION")
    print("=" * 72)

    start_tokens = (
        val_sequences[0][:8]
        .tolist()
    )

    settings = [
        {
            "temperature": 0.6,
            "top_k": 10,
            "top_p": 0.90
        },
        {
            "temperature": 0.8,
            "top_k": 20,
            "top_p": 0.90
        },
        {
            "temperature": 1.0,
            "top_k": 40,
            "top_p": 0.95
        },
        {
            "temperature": 1.2,
            "top_k": 50,
            "top_p": 0.95
        }
    ]

    generated_sequences = []

    for setting in settings:

        generated = generate(
            model=model,
            start_tokens=start_tokens,
            num_new_tokens=80,
            temperature=setting[
                "temperature"
            ],
            top_k=setting["top_k"],
            top_p=setting["top_p"]
        )

        generated_sequences.append(
            generated
        )

        entropy = (
            measure_generation_entropy(
                model,
                generated
            )
        )

        print()
        print(
            "Temperature:",
            setting["temperature"]
        )

        print(
            "Top-k:",
            setting["top_k"]
        )

        print(
            "Top-p:",
            setting["top_p"]
        )

        print(
            "Generation entropy:",
            f"{entropy:.4f}"
        )

        print(
            "Generated:",
            generated[:40]
        )

    # --------------------------------------------------------------
    # STEP 9 — VISUALIZE GENERATION
    # --------------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        generated_sequences[0],
        label="T=0.6"
    )

    plt.plot(
        generated_sequences[1],
        label="T=0.8"
    )

    plt.plot(
        generated_sequences[2],
        label="T=1.0"
    )

    plt.plot(
        generated_sequences[3],
        label="T=1.2"
    )

    plt.xlabel(
        "Token position"
    )

    plt.ylabel(
        "Token ID"
    )

    plt.title(
        "Effect of Sampling Temperature"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------------
    # STEP 10 — TRAINING CURVES
    # --------------------------------------------------------------

    plot_training(
        train_losses,
        val_losses,
        train_ppls,
        val_ppls
    )

    # --------------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("LESSON 38 COMPLETE")
    print("=" * 72)

    print()
    print("You implemented:")

    print(
        "  [1] Token statistics"
    )

    print(
        "  [2] Label smoothing"
    )

    print(
        "  [3] Transformer language model"
    )

    print(
        "  [4] Learning-rate warmup"
    )

    print(
        "  [5] Gradient clipping"
    )

    print(
        "  [6] Temperature sampling"
    )

    print(
        "  [7] Top-k sampling"
    )

    print(
        "  [8] Top-p sampling"
    )

    print(
        "  [9] Autoregressive generation"
    )

    print(
        "  [10] Checkpointing"
    )

    print()

    print(
        "Checkpoint:",
        os.path.abspath(
            "lesson38_best.pt"
        )
    )

    print()
    print(
        "The model still generates discrete speech tokens."
    )

    print(
        "It does NOT directly generate waveform audio yet."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "Lesson 39 — Real Speech Token Dataset Pipeline"
    )

    print()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()