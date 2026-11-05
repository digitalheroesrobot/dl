"""
========================================================================
PHASE 1 — LESSON 32
TEXT / LINGUISTIC REPRESENTATION FROM SCRATCH
========================================================================

PURPOSE
-------
Move from an isolated neural vocoder toward the linguistic side of TTS.

In Lesson 31 we had:

    MEL SPECTROGRAM
          |
          v
    NEURAL VOCODER
          |
          v
       WAVEFORM

Now we begin building the front half:

    TEXT
      |
      v
    TOKENIZATION
      |
      v
    TOKEN EMBEDDINGS
      |
      v
    TRANSFORMER ENCODER
      |
      v
    LINGUISTIC REPRESENTATION
      |
      v
    ACOUSTIC PREDICTOR
      |
      v
    MEL SPECTROGRAM


WHAT YOU LEARN
--------------

STEP 1
------
Understand why raw text cannot be directly given to a neural network.

Example:

    "Hello"

must become something like:

    [token_1, token_2, token_3, ...]

STEP 2
------
Build a simple multilingual character tokenizer.

The lesson includes:

    English
    Chinese
    Japanese
    Russian

No language-specific pretrained tokenizer is used.

STEP 3
------
Convert token IDs into dense vectors:

    token ID
       |
       v
    Embedding
       |
       v
    vector representation

STEP 4
------
Add positional information.

Transformers do not inherently know that:

    token A comes before token B.

Therefore we add positional encoding.

STEP 5
------
Build a Transformer encoder from scratch using PyTorch's
TransformerEncoder building block.

STEP 6
------
Predict acoustic information from the linguistic representation.

The acoustic predictor estimates:

    duration
    pitch
    energy
    mel spectrogram

This is intentionally simplified.

STEP 7
------
Train the complete text-to-acoustic model.

The important conceptual separation is:

    TEXT
      |
      v
    LINGUISTIC ENCODER
      |
      v
    ACOUSTIC MODEL
      |
      v
    MEL

and then later:

    MEL
      |
      v
    VOCODER
      |
      v
    WAVEFORM


REAL-WORLD EXAMPLE
------------------
A modern TTS system often has some variation of:

    text
      ↓
    text normalization
      ↓
    tokenizer / phonemizer
      ↓
    linguistic representation
      ↓
    acoustic model
      ↓
    mel spectrogram
      ↓
    neural vocoder
      ↓
    speech

This lesson implements a small educational version of that pipeline.


IMPORTANT LIMITATION
--------------------
This is NOT yet a high-quality TTS system.

We are intentionally learning the architecture one component at a time.

The synthetic acoustic target is deterministic enough that the network
can learn the relationship between text tokens and acoustic features.


MULTILINGUAL EXAMPLES
---------------------
English:

    Hello world

Chinese:

    你好世界

Japanese:

    こんにちは世界

Russian:

    Привет мир


OUTPUT
------
outputs/lesson32/

    linguistic_model.pt
    lesson32_linguistic_representation.png


NEXT LESSON
-----------
Lesson 33 will introduce:

    TEXT
      ↓
    linguistic representation
      +
    SPEAKER EMBEDDING
      ↓
    speaker-conditioned acoustic representation

This is where the architecture begins moving toward
MULTI-SPEAKER VOICE CLONING.

========================================================================
"""

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path(
    "outputs/lesson32"
)

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "linguistic_model.pt"
)

# Text representation
VOCAB_SIZE = 256
EMBED_DIM = 128

# Transformer
NUM_HEADS = 4
NUM_LAYERS = 3
FF_DIM = 256
DROPOUT = 0.1

# Acoustic representation
N_MELS = 80
MAX_TEXT_LENGTH = 32

# Synthetic dataset
NUM_SAMPLES = 512
BATCH_SIZE = 16

# Training
EPOCHS = 30
LEARNING_RATE = 2e-4

# Synthetic acoustic target
ACOUSTIC_FRAMES = 32


# ============================================================================
# RANDOM SEED
# ============================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(
    nn.Module
):
    """
    Standard sinusoidal positional encoding.

    For position p and dimension i:

        PE(p, 2i)
            = sin(p / 10000^(2i/d))

        PE(p, 2i+1)
            = cos(p / 10000^(2i/d))

    The network therefore receives information about token position.
    """

    def __init__(
        self,
        embed_dim,
        max_length=MAX_TEXT_LENGTH,
    ):

        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embed_dim,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / embed_dim
            )
        )

        pe = torch.zeros(
            max_length,
            embed_dim,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):

        length = x.shape[1]

        return (
            x
            + self.pe[
                :,
                :length,
            ]
        )


# ============================================================================
# SIMPLE MULTILINGUAL TOKENIZER
# ============================================================================

class CharacterTokenizer:
    """
    Educational character tokenizer.

    Special tokens:

        0 = PAD
        1 = BOS
        2 = EOS
        3 = UNK

    Remaining IDs correspond to Unicode characters.

    We deliberately use a character-level tokenizer so that the same
    implementation can process English, Chinese, Japanese and Russian.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(
        self,
        texts,
        vocab_size=VOCAB_SIZE,
    ):

        characters = set()

        for text in texts:

            characters.update(
                list(text)
            )

        characters = sorted(
            characters
        )

        available = (
            vocab_size - 4
        )

        characters = characters[
            :available
        ]

        self.id_to_char = {
            self.PAD: "<PAD>",
            self.BOS: "<BOS>",
            self.EOS: "<EOS>",
            self.UNK: "<UNK>",
        }

        self.char_to_id = {
            "<PAD>": self.PAD,
            "<BOS>": self.BOS,
            "<EOS>": self.EOS,
            "<UNK>": self.UNK,
        }

        next_id = 4

        for char in characters:

            self.char_to_id[
                char
            ] = next_id

            self.id_to_char[
                next_id
            ] = char

            next_id += 1

        self.vocab_size = next_id

    def encode(
        self,
        text,
        max_length=MAX_TEXT_LENGTH,
    ):

        ids = [
            self.BOS
        ]

        for char in text:

            ids.append(
                self.char_to_id.get(
                    char,
                    self.UNK,
                )
            )

        ids.append(
            self.EOS
        )

        ids = ids[
            :max_length
        ]

        if len(ids) < max_length:

            ids += [
                self.PAD
            ] * (
                max_length
                - len(ids)
            )

        return torch.tensor(
            ids,
            dtype=torch.long,
        )

    def decode(self, ids):

        chars = []

        for token_id in ids:

            token_id = int(
                token_id
            )

            if token_id in (
                self.PAD,
                self.BOS,
                self.EOS,
            ):
                continue

            chars.append(
                self.id_to_char.get(
                    token_id,
                    "<UNK>",
                )
            )

        return "".join(chars)


# ============================================================================
# TRAINING TEXT
# ============================================================================

def build_text_corpus():

    return [

        # English
        "Hello world",
        "Good morning",
        "How are you",
        "The robot is moving",
        "Pick up the object",
        "Please open the door",
        "This is a speech system",

        # Chinese
        "你好世界",
        "早上好",
        "你好吗",
        "机器人正在移动",
        "拿起这个物体",
        "请打开门",

        # Japanese
        "こんにちは世界",
        "おはようございます",
        "元気ですか",
        "ロボットが動いています",
        "物体を持ち上げてください",
        "ドアを開けてください",

        # Russian
        "Привет мир",
        "Доброе утро",
        "Как дела",
        "Робот движется",
        "Возьми предмет",
        "Открой дверь",

    ]


# ============================================================================
# SYNTHETIC ACOUSTIC TARGET
# ============================================================================

def text_to_synthetic_acoustic(
    token_ids,
    n_mels=N_MELS,
    frames=ACOUSTIC_FRAMES,
):
    """
    Create a deterministic synthetic acoustic target from token IDs.

    This is NOT a speech synthesizer.

    It simply creates a structured learning problem so that we can
    verify that the linguistic encoder can learn a relationship between
    text and acoustic features.
    """

    token_ids = token_ids.float()

    valid = token_ids[
        token_ids > 3
    ]

    if valid.numel() == 0:

        valid = torch.ones(
            1,
            dtype=torch.float32,
        )

    # Average token identity.
    token_mean = (
        valid.mean()
        / VOCAB_SIZE
    )

    # Token variance.
    token_std = (
        valid.std()
        if valid.numel() > 1
        else torch.tensor(
            0.0
        )
    )

    token_std = (
        token_std
        / VOCAB_SIZE
    )

    time = torch.linspace(
        0.0,
        1.0,
        frames,
    )

    mel_position = torch.linspace(
        0.0,
        1.0,
        n_mels,
    ).unsqueeze(1)

    time_grid = time.unsqueeze(0)

    # Global spectral structure.
    base = torch.sin(
        2.0
        * math.pi
        * (
            1.0
            + 8.0 * token_mean
        )
        * time_grid
    )

    spectral = torch.exp(
        -(
            mel_position
            - (
                0.25
                + 0.4 * token_mean
            )
        ).pow(2)
        / 0.025
    )

    # Temporal structure.
    modulation = (
        0.5
        + 0.5
        * torch.sin(
            2.0
            * math.pi
            * (
                1.0
                + 5.0 * token_std
            )
            * time_grid
        )
    )

    mel = (
        spectral
        * (
            0.5
            + 0.5 * base
        )
        * modulation
    )

    # Add a small deterministic harmonic component.
    harmonic = (
        0.2
        * torch.sin(
            2.0
            * math.pi
            * (
                mel_position * 3.0
                + token_mean * 2.0
            )
        )
    )

    mel = mel + harmonic

    # Log-like compression.
    mel = torch.log1p(
        torch.relu(mel)
        * 10.0
    )

    # Normalize.
    mel = (
        mel - mel.mean()
    ) / (
        mel.std()
        + 1e-6
    )

    return mel


# ============================================================================
# DATASET
# ============================================================================

class TextAcousticDataset(
    Dataset
):

    def __init__(
        self,
        texts,
        tokenizer,
        num_samples,
    ):

        self.samples = []

        for i in range(
            num_samples
        ):

            text = random.choice(
                texts
            )

            token_ids = (
                tokenizer.encode(
                    text
                )
            )

            acoustic = (
                text_to_synthetic_acoustic(
                    token_ids
                )
            )

            self.samples.append(
                (
                    token_ids,
                    acoustic,
                )
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        return self.samples[
            index
        ]


# ============================================================================
# TEXT / LINGUISTIC ENCODER
# ============================================================================

class LinguisticEncoder(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.positional_encoding = (
            PositionalEncoding(
                embed_dim
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        token_ids,
    ):

        padding_mask = (
            token_ids == 0
        )

        x = self.embedding(
            token_ids
        )

        x = (
            self.positional_encoding(
                x
            )
        )

        x = self.transformer(
            x,
            src_key_padding_mask=(
                padding_mask
            ),
        )

        x = self.norm(x)

        return x


# ============================================================================
# ACOUSTIC PREDICTOR
# ============================================================================

class AcousticPredictor(
    nn.Module
):

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        n_mels=N_MELS,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim,
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim,
                embed_dim,
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim,
                n_mels,
            ),
        )

    def forward(
        self,
        linguistic_features,
        token_ids,
    ):
        """
        Convert token-level linguistic features into a fixed-length
        acoustic representation.

        We use masked mean pooling here.

        Real TTS systems generally need much more sophisticated
        alignment / duration mechanisms.
        """

        mask = (
            token_ids != 0
        ).float()

        mask = mask.unsqueeze(
            -1
        )

        summed = (
            linguistic_features
            * mask
        ).sum(
            dim=1
        )

        denominator = (
            mask.sum(
                dim=1
            )
            .clamp_min(1.0)
        )

        pooled = (
            summed
            / denominator
        )

        mel_frame = (
            self.network(
                pooled
            )
        )

        # Expand the acoustic vector across time.
        mel = (
            mel_frame
            .unsqueeze(-1)
            .repeat(
                1,
                1,
                ACOUSTIC_FRAMES,
            )
        )

        return mel


# ============================================================================
# COMPLETE TEXT-TO-ACOUSTIC MODEL
# ============================================================================

class TextToAcousticModel(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
    ):

        super().__init__()

        self.encoder = (
            LinguisticEncoder(
                vocab_size
            )
        )

        self.acoustic_predictor = (
            AcousticPredictor()
        )

    def forward(
        self,
        token_ids,
    ):

        linguistic = (
            self.encoder(
                token_ids
            )
        )

        mel = (
            self.acoustic_predictor(
                linguistic,
                token_ids,
            )
        )

        return (
            mel,
            linguistic,
        )


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
):

    model.train()

    total_loss = 0.0

    for (
        token_ids,
        target_mel,
    ) in loader:

        token_ids = token_ids.to(
            DEVICE
        )

        target_mel = target_mel.to(
            DEVICE
        )

        predicted_mel, _ = (
            model(
                token_ids
            )
        )

        loss = F.mse_loss(
            predicted_mel,
            target_mel,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

    return (
        total_loss
        / len(loader)
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(
    model,
    tokenizer,
    texts,
):

    model.eval()

    examples = [
        texts[0],
        texts[7],
        texts[13],
        texts[19],
    ]

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(14, 14),
    )

    for row, text in enumerate(
        examples
    ):

        token_ids = (
            tokenizer.encode(
                text
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            predicted_mel, linguistic = (
                model(
                    token_ids
                )
            )

        mel = (
            predicted_mel[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        linguistic = (
            linguistic[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        axes[row, 0].imshow(
            mel,
            aspect="auto",
            origin="lower",
        )

        axes[row, 0].set_title(
            f"Predicted Acoustic Representation\n{text}"
        )

        axes[row, 0].set_xlabel(
            "Time Frames"
        )

        axes[row, 0].set_ylabel(
            "Mel Bin"
        )

        # Linguistic embedding heatmap.
        axes[row, 1].imshow(
            linguistic.T,
            aspect="auto",
            origin="lower",
        )

        axes[row, 1].set_title(
            f"Linguistic Representation\n{text}"
        )

        axes[row, 1].set_xlabel(
            "Token Position"
        )

        axes[row, 1].set_ylabel(
            "Embedding Dimension"
        )

    plt.suptitle(
        "Lesson 32 — Text to Linguistic Representation",
        fontsize=16,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "lesson32_linguistic_representation.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print()
    print(
        f"Saved visualization:"
    )

    print(
        output_path
    )

    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "PHASE 1 — LESSON 32"
    )
    print(
        "TEXT / LINGUISTIC REPRESENTATION"
    )
    print("=" * 72)

    print(
        f"Device: {DEVICE}"
    )

    # ------------------------------------------------------------------------
    # TEXT CORPUS
    # ------------------------------------------------------------------------

    texts = build_text_corpus()

    print()
    print(
        "Multilingual examples:"
    )

    for text in texts[:4]:

        print(
            f"  {text}"
        )

    print(
        f"Total text templates: "
        f"{len(texts)}"
    )

    # ------------------------------------------------------------------------
    # TOKENIZER
    # ------------------------------------------------------------------------

    tokenizer = (
        CharacterTokenizer(
            texts
        )
    )

    print()
    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print()
    print(
        "Tokenizer examples:"
    )

    examples = [
        "Hello world",
        "你好世界",
        "こんにちは世界",
        "Привет мир",
    ]

    for text in examples:

        ids = tokenizer.encode(
            text
        )

        decoded = (
            tokenizer.decode(
                ids
            )
        )

        print()
        print(
            f"Text:    {text}"
        )

        print(
            f"Token IDs: "
            f"{ids.tolist()}"
        )

        print(
            f"Decoded: {decoded}"
        )

    # ------------------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------------------

    dataset = (
        TextAcousticDataset(
            texts,
            tokenizer,
            NUM_SAMPLES,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print()
    print(
        f"Dataset size: "
        f"{len(dataset)}"
    )

    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    model = (
        TextToAcousticModel(
            tokenizer.vocab_size
        )
        .to(DEVICE)
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # ------------------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # ------------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------------

    print()
    print(
        "Starting training..."
    )
    print()

    losses = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        loss = train_one_epoch(
            model,
            loader,
            optimizer,
        )

        losses.append(
            loss
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS} | "
            f"Loss: {loss:.6f}"
        )

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "vocab_size":
                tokenizer.vocab_size,

            "embed_dim":
                EMBED_DIM,

            "n_mels":
                N_MELS,

            "max_text_length":
                MAX_TEXT_LENGTH,
        },
        CHECKPOINT_PATH,
    )

    print()
    print(
        f"Saved checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    # ------------------------------------------------------------------------
    # LOSS PLOT
    # ------------------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        losses
    )

    plt.title(
        "Lesson 32 Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "MSE Loss"
    )

    plt.grid(
        alpha=0.3
    )

    loss_path = (
        OUTPUT_DIR
        / "training_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"Saved loss plot:"
    )

    print(
        loss_path
    )

    plt.show()

    # ------------------------------------------------------------------------
    # FINAL VISUALIZATION
    # ------------------------------------------------------------------------

    plot_results(
        model,
        tokenizer,
        texts,
    )

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "LESSON 32 COMPLETE"
    )
    print("=" * 72)

    print(
        """
The architecture you built is:

    TEXT
      |
      v
    CHARACTER TOKENIZER
      |
      v
    TOKEN IDs
      |
      v
    EMBEDDING
      |
      v
    POSITIONAL ENCODING
      |
      v
    TRANSFORMER ENCODER
      |
      v
    LINGUISTIC REPRESENTATION
      |
      v
    ACOUSTIC PREDICTOR
      |
      v
    MEL REPRESENTATION

The important conceptual separation is:

    Linguistic information
            |
            v
       Text Encoder
            |
            v
    Acoustic representation
            |
            v
         Vocoder
            |
            v
         Waveform


WHY THIS MATTERS FOR VOICE CLONING
----------------------------------
The text encoder represents WHAT should be spoken.

The speaker representation represents WHO should speak.

The acoustic model combines:

    WHAT
     +
    WHO
     |
     v
    speech characteristics

Lesson 33 will introduce this speaker-conditioning mechanism.

NEXT:

    Lesson 33 — Speaker-Conditioned Text-to-Acoustic Modeling

We will combine:

    TEXT
      +
    SPEAKER EMBEDDING
      |
      v
    CONDITIONED ACOUSTIC MODEL
      |
      v
    MEL SPECTROGRAM

This is the key architectural step toward
multi-speaker TTS and voice cloning.
"""
    )


if __name__ == "__main__":
    main()