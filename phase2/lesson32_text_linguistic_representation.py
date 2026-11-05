"""
======================================================================
PHASE 2 — LESSON 32
TEXT LINGUISTIC REPRESENTATION
======================================================================

PURPOSE
-------
Start Phase 2 of the Chatterbox-like TTS course.

The goal of this lesson is to understand how text is converted into
a neural representation that can later condition a speech-token
generation model.

We are NOT building the T3 model yet.

We are building the foundation that T3 will consume.

CORE IDEA
---------
A generative TTS system must transform:

    TEXT
      |
      v
    linguistic representation
      |
      v
    speech representation
      |
      v
    waveform

The first question is therefore:

    "How should text be represented inside a neural network?"

We will investigate:

    1. Text normalization
    2. Character-level representation
    3. Word-level representation
    4. Token IDs
    5. Embeddings
    6. Positional embeddings
    7. Transformer contextualization

IMPORTANT ARCHITECTURAL DISTINCTION
-----------------------------------
Our eventual Chatterbox-like system will approximately become:

    TEXT
      |
      v
    TEXT REPRESENTATION
      |
      v
    T3-LIKE AUTOREGRESSIVE MODEL
      |
      v
    DISCRETE SPEECH TOKENS
      |
      v
    S3GEN-LIKE GENERATIVE DECODER
      |
      v
    MEL / ACOUSTIC REPRESENTATION
      |
      v
    NEURAL VOCODER
      |
      v
    WAVEFORM


WHAT YOU LEARN
--------------
STEP 1
    Normalize text into a predictable form.

STEP 2
    Understand different linguistic granularities.

STEP 3
    Convert discrete symbols into trainable vectors.

STEP 4
    Add positional information.

STEP 5
    Use self-attention to create contextual representations.

STEP 6
    Visualize the resulting text representation.

WHY THIS MATTERS FOR VOICE CLONING
----------------------------------
Later we want to separate:

    WHAT IS SAID
        =
    linguistic/content information

from:

    HOW IT SOUNDS
        =
    speaker/acoustic information

This lesson builds the first side of that separation.

The future architecture will look like:

                         TEXT
                          |
                          v
                 linguistic representation
                          |
                          |
REFERENCE AUDIO           |
      |                   |
      v                   v
speaker representation -> T3
                          |
                          v
                   speech tokens
                          |
                          v
                    S3GEN-like
                          |
                          v
                       vocoder
                          |
                          v
                       AUDIO


MULTILINGUAL EXAMPLES
---------------------
We deliberately use:

    English
    Chinese
    Japanese
    Russian

because the final system should not be designed around English-only
assumptions.

No external tokenizer or pretrained language model is required.

DEPENDENCIES
------------
    Python 3
    PyTorch
    NumPy
    Matplotlib

RUN
---
    python lesson32_text_linguistic_representation.py


NEXT LESSON
-----------
Lesson 33 will build a real text tokenizer from scratch:

    text
      |
      v
    vocabulary
      |
      v
    token IDs
      |
      v
    padding / masks
      |
      v
    batches

Then we can use those token IDs as the input to the models developed
in this lesson.
======================================================================
"""

import math
import random
import re

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_LENGTH = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# STEP 1 — TEXT NORMALIZATION
# ======================================================================

def normalize_text(text: str) -> str:
    """
    Basic normalization.

    This is intentionally simple.

    We are NOT trying to build a production tokenizer here.

    The purpose is to understand that raw text usually needs
    preprocessing before entering a neural model.
    """

    text = text.strip()

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    # Lowercase only Latin characters.
    #
    # Chinese/Japanese/Russian characters remain unchanged.
    text = text.lower()

    return text


# ======================================================================
# STEP 2 — LINGUISTIC UNIT ANALYSIS
# ======================================================================

def character_units(text: str):
    """
    Character-level representation.

    This keeps each Unicode character as a unit.

    Example:

        hello

    becomes approximately:

        h e l l o
    """

    return list(text)


def simple_word_units(text: str):
    """
    Extremely simple word-like segmentation.

    This is NOT a production multilingual tokenizer.

    It is only used to demonstrate the conceptual difference between
    character-level and word-level representations.

    For whitespace-delimited languages:

        "hello world"

    becomes:

        ["hello", "world"]

    For Chinese/Japanese text without spaces, characters remain useful
    as fallback units.
    """

    text = text.strip()

    if " " in text:
        return text.split()

    return list(text)


# ======================================================================
# STEP 3 — TEMPORARY SYMBOL REPRESENTATION
# ======================================================================

def unicode_symbol_id(symbol: str) -> int:
    """
    Convert a Unicode character into a deterministic integer.

    IMPORTANT:

    This is NOT our final tokenizer.

    Lesson 33 will build a proper vocabulary/token-ID system.

    Here we simply need a numerical representation so that we can
    demonstrate embeddings.
    """

    if not symbol:
        return 0

    return ord(symbol[0])


def symbols_to_ids(symbols):
    return [unicode_symbol_id(symbol) for symbol in symbols]


# ======================================================================
# STEP 4 — EMBEDDING REPRESENTATION
# ======================================================================

class SymbolEmbedding(nn.Module):
    """
    Demonstration embedding layer.

    Because Unicode code points are sparse and very large, we use a
    simple projection table indexed through modulo arithmetic.

    Again:

        This is pedagogical.

    Lesson 33 will replace this with a proper learned vocabulary.
    """

    def __init__(self, table_size=4096, embedding_dim=EMBED_DIM):
        super().__init__()

        self.table_size = table_size

        self.embedding = nn.Embedding(
            num_embeddings=table_size,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids):
        token_ids = token_ids % self.table_size

        return self.embedding(token_ids)


# ======================================================================
# STEP 5 — POSITIONAL ENCODING
# ======================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Transformer self-attention does not inherently know sequence order.

    Therefore:

        token embedding

    is combined with:

        positional information
    """

    def __init__(self, embedding_dim, max_length=MAX_LENGTH):

        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / embedding_dim)
        )

        pe = torch.zeros(
            max_length,
            embedding_dim,
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        sequence_length = x.size(1)

        return x + self.pe[:, :sequence_length]


# ======================================================================
# STEP 6 — CONTEXTUAL TEXT ENCODER
# ======================================================================

class TextRepresentationModel(nn.Module):
    """
    Small Transformer encoder.

    Pipeline:

        token IDs
            |
            v
        embedding
            |
            v
        positional encoding
            |
            v
        Transformer encoder
            |
            v
        contextual representation
    """

    def __init__(
        self,
        embedding_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_length=MAX_LENGTH,
    ):

        super().__init__()

        self.embedding = SymbolEmbedding(
            embedding_dim=embedding_dim
        )

        self.position = SinusoidalPositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_length,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, token_ids):

        x = self.embedding(token_ids)

        embedding_output = x.clone()

        x = self.position(x)

        positional_output = x.clone()

        contextual_output = self.encoder(x)

        return (
            embedding_output,
            positional_output,
            contextual_output,
        )


# ======================================================================
# HELPER — PAD SEQUENCES
# ======================================================================

def make_batch(texts):

    normalized = [
        normalize_text(text)
        for text in texts
    ]

    units = [
        character_units(text)
        for text in normalized
    ]

    ids = [
        symbols_to_ids(sequence)
        for sequence in units
    ]

    max_length = max(len(sequence) for sequence in ids)

    max_length = min(max_length, MAX_LENGTH)

    batch = []

    attention_mask = []

    for sequence in ids:

        sequence = sequence[:max_length]

        padded = sequence + [0] * (max_length - len(sequence))

        mask = (
            [1] * len(sequence)
            + [0] * (max_length - len(sequence))
        )

        batch.append(padded)
        attention_mask.append(mask)

    batch = torch.tensor(
        batch,
        dtype=torch.long,
        device=DEVICE,
    )

    attention_mask = torch.tensor(
        attention_mask,
        dtype=torch.bool,
        device=DEVICE,
    )

    return normalized, units, batch, attention_mask


# ======================================================================
# VISUALIZATION 1 — REPRESENTATION HEATMAP
# ======================================================================

def plot_representation(
    representation,
    tokens,
    title,
    filename,
):

    representation = representation.detach().cpu().numpy()

    # Use only the first sample.
    matrix = representation[0]

    valid_length = len(tokens)

    matrix = matrix[:valid_length]

    plt.figure(figsize=(14, 6))

    plt.imshow(
        matrix.T,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(label="activation")

    plt.xticks(
        range(valid_length),
        tokens,
        rotation=45,
        ha="right",
    )

    plt.xlabel("Text position")

    plt.ylabel("Embedding dimension")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150,
    )

    plt.show()


# ======================================================================
# VISUALIZATION 2 — TOKEN EMBEDDING SIMILARITY
# ======================================================================

def plot_embedding_similarity(
    embedding_output,
    tokens,
    filename,
):

    x = embedding_output[0].detach().cpu()

    valid_length = len(tokens)

    x = x[:valid_length]

    x = torch.nn.functional.normalize(
        x,
        dim=-1,
    )

    similarity = x @ x.T

    similarity = similarity.numpy()

    plt.figure(figsize=(8, 7))

    plt.imshow(
        similarity,
        aspect="equal",
        interpolation="nearest",
    )

    plt.colorbar(label="cosine similarity")

    plt.xticks(
        range(valid_length),
        tokens,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(valid_length),
        tokens,
    )

    plt.xlabel("Token position")

    plt.ylabel("Token position")

    plt.title("Token Embedding Cosine Similarity")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150,
    )

    plt.show()


# ======================================================================
# STEP 7 — CONTEXTUAL REPRESENTATION EXPERIMENT
# ======================================================================

def contextual_difference_experiment(model):

    """
    Demonstrate that contextual representations can change depending
    on surrounding text.

    We compare the same word in two different sentences.

    Example:

        "the bank approved the loan"

        "we sat beside the bank"

    The word "bank" appears in different linguistic contexts.

    A Transformer can theoretically produce different contextual
    representations for the same token position.

    Our tiny untrained model is only demonstrating the mechanism,
    not semantic language understanding.
    """

    sentences = [
        "the bank approved the loan",
        "we sat beside the bank",
    ]

    normalized, units, batch, mask = make_batch(sentences)

    with torch.no_grad():

        embedding_output, _, contextual_output = model(batch)

    print("\nContextual representation experiment")
    print("------------------------------------")

    for i, sentence in enumerate(normalized):

        tokens = units[i]

        bank_positions = [
            j
            for j, token in enumerate(tokens)
            if token == "b"
        ]

        print(f"\nSentence {i + 1}: {sentence}")

        print(
            "Character positions containing 'b':",
            bank_positions,
        )

    return contextual_output


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 72)
    print("PHASE 2 — LESSON 32")
    print("TEXT LINGUISTIC REPRESENTATION")
    print("=" * 72)

    set_seed()

    print(f"\nDevice: {DEVICE}")
    print(f"Embedding dimension: {EMBED_DIM}")
    print(f"Transformer layers: {NUM_LAYERS}")
    print(f"Attention heads: {NUM_HEADS}")

    # --------------------------------------------------------------
    # Example multilingual text
    # --------------------------------------------------------------

    texts = [
        "Hello, this is a voice cloning experiment.",
        "人工智能正在改变机器人技术。",
        "音声合成とロボット技術を研究しています。",
        "Искусственный интеллект меняет робототехнику.",
    ]

    print("\n")
    print("-" * 72)
    print("STEP 1 — TEXT NORMALIZATION")
    print("-" * 72)

    for text in texts:

        normalized = normalize_text(text)

        print(f"\nOriginal   : {text}")
        print(f"Normalized : {normalized}")

    # --------------------------------------------------------------
    # Character representation
    # --------------------------------------------------------------

    print("\n")
    print("-" * 72)
    print("STEP 2 — CHARACTER REPRESENTATION")
    print("-" * 72)

    example = normalize_text(texts[0])

    chars = character_units(example)

    print("\nText:")
    print(example)

    print("\nCharacter units:")
    print(chars)

    print("\nNumber of characters:", len(chars))

    # --------------------------------------------------------------
    # Word-like representation
    # --------------------------------------------------------------

    print("\n")
    print("-" * 72)
    print("WORD-LIKE REPRESENTATION")
    print("-" * 72)

    for text in texts:

        normalized = normalize_text(text)

        units = simple_word_units(normalized)

        print(f"\nText: {normalized}")
        print("Units:", units)

    # --------------------------------------------------------------
    # Character IDs
    # --------------------------------------------------------------

    print("\n")
    print("-" * 72)
    print("STEP 3 — NUMERICAL REPRESENTATION")
    print("-" * 72)

    ids = symbols_to_ids(chars)

    print("\nFirst 30 characters:")
    print(chars[:30])

    print("\nFirst 30 Unicode IDs:")
    print(ids[:30])

    # --------------------------------------------------------------
    # Build model
    # --------------------------------------------------------------

    print("\n")
    print("-" * 72)
    print("STEP 4 — EMBEDDINGS")
    print("-" * 72)

    model = TextRepresentationModel().to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nText representation model parameters: "
        f"{parameter_count:,}"
    )

    normalized, units, batch, mask = make_batch(texts)

    print("\nBatch shape:")
    print(batch.shape)

    print("\nBatch dtype:")
    print(batch.dtype)

    print("\nAttention mask shape:")
    print(mask.shape)

    with torch.no_grad():

        (
            embedding_output,
            positional_output,
            contextual_output,
        ) = model(batch)

    print("\nEmbedding representation:")
    print(embedding_output.shape)

    print("\nAfter positional encoding:")
    print(positional_output.shape)

    print("\nContextual representation:")
    print(contextual_output.shape)

    # --------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------

    print("\n")
    print("-" * 72)
    print("STEP 5 — VISUALIZATION")
    print("-" * 72)

    first_tokens = units[0]

    plot_representation(
        embedding_output,
        first_tokens,
        "Raw Token Embedding Representation",
        "lesson32_embedding_representation.png",
    )

    plot_representation(
        positional_output,
        first_tokens,
        "Embedding + Positional Encoding",
        "lesson32_positional_representation.png",
    )

    plot_representation(
        contextual_output,
        first_tokens,
        "Transformer Contextual Representation",
        "lesson32_contextual_representation.png",
    )

    plot_embedding_similarity(
        embedding_output,
        first_tokens,
        "lesson32_embedding_similarity.png",
    )

    # --------------------------------------------------------------
    # Context experiment
    # --------------------------------------------------------------

    print("\n")
    print("-" * 72)
    print("STEP 6 — CONTEXTUAL REPRESENTATION")
    print("-" * 72)

    contextual_difference_experiment(model)

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("LESSON 32 COMPLETE")
    print("=" * 72)

    print(
        """
We built the following pipeline:

    text
      |
      v
    normalized text
      |
      v
    character units
      |
      v
    numerical representation
      |
      v
    embedding
      |
      v
    positional encoding
      |
      v
    Transformer encoder
      |
      v
    contextual text representation

This representation will eventually become the conditioning signal
for our T3-like speech-token generator.

IMPORTANT:

The Unicode-ID mechanism used here is deliberately temporary.

Lesson 33 will replace it with a real tokenizer:

    text
      |
      v
    vocabulary
      |
      v
    token IDs
      |
      v
    padding + masks
      |
      v
    neural representation

Then Lesson 34 will move to the other side of the problem:

    waveform
      |
      v
    speech encoder
      |
      v
    discrete speech tokens

At that point we will have both sides:

    TEXT  -> text tokens
    AUDIO -> speech tokens

and we can begin learning how a T3-like model maps one to the other.
"""
    )


if __name__ == "__main__":
    main()