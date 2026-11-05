"""
========================================================================
VOICE CLONING COURSE
LESSON 51 — MULTILINGUAL TEXT ENCODER AND PHONEME PROCESSING
========================================================================

PURPOSE
-------
Build a multilingual text frontend and Transformer-based phoneme
encoder for an educational text-to-speech system.

SUPPORTED LANGUAGES
-------------------
English
Chinese
Japanese
Russian

WHAT YOU LEARN
--------------

STEP 1:
    Normalize multilingual text.

STEP 2:
    Build language-specific phoneme representations.

STEP 3:
    Construct a shared phoneme vocabulary.

STEP 4:
    Convert phonemes into discrete token IDs.

STEP 5:
    Add language embeddings.

STEP 6:
    Add positional embeddings.

STEP 7:
    Implement Transformer encoder layers.

STEP 8:
    Generate contextual linguistic representations.

STEP 9:
    Train a multilingual encoder using a synthetic learning task.

STEP 10:
    Extract and visualize linguistic embeddings.

REAL-WORLD APPLICATION
----------------------
A production voice-cloning model needs a reliable text frontend.
This module provides the linguistic input representation that can
later condition a speaker-conditioned speech-token generator.

IMPORTANT LIMITATION
--------------------
The pronunciation rules in this lesson are deliberately simplified.

They are NOT production-grade G2P systems.

For real applications, replace them with:
    English: phonemizer / CMU Pronouncing Dictionary
    Chinese: pypinyin with tone information
    Japanese: a pronunciation frontend such as pyopenjtalk
    Russian: a language-specific phonological G2P frontend

The synthetic training objective teaches the architecture, not
real-world pronunciation accuracy.

========================================================================
"""

import os
import re
import math
import random
import unicodedata

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 51

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANGUAGES = {
    "en": 0,
    "zh": 1,
    "ja": 2,
    "ru": 3
}

LANGUAGE_NAMES = {
    0: "English",
    1: "Chinese",
    2: "Japanese",
    3: "Russian"
}

PAD_ID = 0
UNK_ID = 1

MAX_SEQUENCE_LENGTH = 64

EMBED_DIM = 256

NUM_HEADS = 8

NUM_LAYERS = 3

FFN_DIM = 512

DROPOUT = 0.1

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 3e-4

NUM_TRAIN_SAMPLES = 2400

NUM_VAL_SAMPLES = 400

OUTPUT_DIR = "lesson51_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================================
# STEP 1 — TEXT NORMALIZATION
# =====================================================================

def normalize_text(text, language):

    """
    Normalize text while preserving language-specific characters.
    """

    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    if language == "en":

        text = text.lower()

    elif language == "ru":

        text = text.lower()

    return text


# =====================================================================
# STEP 2 — SIMPLIFIED MULTILINGUAL PHONEME FRONTEND
# =====================================================================

ENGLISH_MAP = {

    "a": ["ae"],
    "b": ["b"],
    "c": ["k"],
    "d": ["d"],
    "e": ["eh"],
    "f": ["f"],
    "g": ["g"],
    "h": ["h"],
    "i": ["ih"],
    "j": ["jh"],
    "k": ["k"],
    "l": ["l"],
    "m": ["m"],
    "n": ["n"],
    "o": ["ow"],
    "p": ["p"],
    "q": ["k"],
    "r": ["r"],
    "s": ["s"],
    "t": ["t"],
    "u": ["uh"],
    "v": ["v"],
    "w": ["w"],
    "x": ["k", "s"],
    "y": ["y"],
    "z": ["z"]

}


CHINESE_MAP = {

    "你": ["n", "i3"],
    "好": ["h", "ao3"],
    "我": ["w", "o3"],
    "是": ["sh", "i4"],
    "中": ["zh", "ong1"],
    "国": ["g", "uo2"],
    "人": ["r", "en2"],
    "们": ["m", "en"],
    "的": ["d", "e5"],
    "谢": ["x", "ie4"],
    "再": ["z", "ai4"],
    "见": ["j", "ian4"],
    "早": ["z", "ao3"],
    "安": ["", "an1"],
    "天": ["t", "ian1"],
    "气": ["q", "i4"]

}


JAPANESE_MAP = {

    "こ": ["k", "o"],
    "ん": ["N"],
    "に": ["n", "i"],
    "ち": ["ch", "i"],
    "は": ["w", "a"],
    "わ": ["w", "a"],
    "た": ["t", "a"],
    "し": ["sh", "i"],
    "で": ["d", "e"],
    "す": ["s", "u"],
    "あ": ["a"],
    "り": ["r", "i"],
    "が": ["g", "a"],
    "と": ["t", "o"],
    "う": ["u"],
    "ま": ["m", "a"],
    "せ": ["s", "e"],
    "お": ["o"],
    "き": ["k", "i"],
    "な": ["n", "a"]

}


RUSSIAN_MAP = {

    "а": ["a"],
    "б": ["b"],
    "в": ["v"],
    "г": ["g"],
    "д": ["d"],
    "е": ["ye"],
    "ё": ["yo"],
    "ж": ["zh"],
    "з": ["z"],
    "и": ["i"],
    "й": ["y"],
    "к": ["k"],
    "л": ["l"],
    "м": ["m"],
    "н": ["n"],
    "о": ["o"],
    "п": ["p"],
    "р": ["r"],
    "с": ["s"],
    "т": ["t"],
    "у": ["u"],
    "ф": ["f"],
    "х": ["kh"],
    "ц": ["ts"],
    "ч": ["ch"],
    "ш": ["sh"],
    "щ": ["shch"],
    "ъ": ["hard_sign"],
    "ы": ["y"],
    "ь": ["soft_sign"],
    "э": ["e"],
    "ю": ["yu"],
    "я": ["ya"]

}


PHONEME_MAPS = {
    "en": ENGLISH_MAP,
    "zh": CHINESE_MAP,
    "ja": JAPANESE_MAP,
    "ru": RUSSIAN_MAP
}


def text_to_phonemes(text, language):

    """
    Simplified character-to-phoneme conversion.

    Spaces and unsupported punctuation are skipped.
    Unknown characters are represented by <unk_char>.
    """

    text = normalize_text(
        text,
        language
    )

    mapping = PHONEME_MAPS[language]

    phonemes = []

    for char in text:

        if char.isspace():
            continue

        if char in mapping:

            phonemes.extend(
                mapping[char]
            )

        elif char.isdigit():

            phonemes.append(
                f"digit_{char}"
            )

        elif char in ".,!?;:":

            phonemes.append(
                "silence"
            )

        else:

            phonemes.append(
                "<unk_char>"
            )

    if not phonemes:

        phonemes = [
            "<unk_char>"
        ]

    return phonemes


# =====================================================================
# STEP 3 — SAMPLE MULTILINGUAL TEXT
# =====================================================================

SAMPLE_TEXTS = [

    (
        "en",
        "Hello, how are you?"
    ),

    (
        "en",
        "Welcome to our voice cloning system."
    ),

    (
        "zh",
        "你好，中国人。"
    ),

    (
        "zh",
        "谢谢，再见。"
    ),

    (
        "ja",
        "こんにちは。"
    ),

    (
        "ja",
        "ありがとうございます。"
    ),

    (
        "ru",
        "Привет."
    ),

    (
        "ru",
        "Добро пожаловать."
    )

]


def print_frontend_examples():

    print("\nMULTILINGUAL PHONEME FRONTEND")

    print("-" * 72)

    for language, text in SAMPLE_TEXTS:

        phonemes = text_to_phonemes(
            text,
            language
        )

        print(
            f"\nLanguage: {LANGUAGE_NAMES[LANGUAGES[language]]}"
        )

        print(
            f"Text: {text}"
        )

        print(
            f"Phonemes: {' '.join(phonemes)}"
        )


# =====================================================================
# STEP 4 — BUILD SHARED PHONEME VOCABULARY
# =====================================================================

class PhonemeVocabulary:

    def __init__(self):

        self.token_to_id = {
            "<pad>": PAD_ID,
            "<unk>": UNK_ID
        }

        self.id_to_token = {
            PAD_ID: "<pad>",
            UNK_ID: "<unk>"
        }

    def build(self, samples):

        for language, text in samples:

            phonemes = text_to_phonemes(
                text,
                language
            )

            for phoneme in phonemes:

                if phoneme not in self.token_to_id:

                    token_id = len(
                        self.token_to_id
                    )

                    self.token_to_id[
                        phoneme
                    ] = token_id

                    self.id_to_token[
                        token_id
                    ] = phoneme

    def encode(self, phonemes):

        return [
            self.token_to_id.get(
                token,
                UNK_ID
            )
            for token in phonemes
        ]

    def decode(self, ids):

        return [
            self.id_to_token.get(
                int(token_id),
                "<unk>"
            )
            for token_id in ids
        ]

    def __len__(self):

        return len(
            self.token_to_id
        )


# =====================================================================
# STEP 5 — SYNTHETIC MULTILINGUAL TRAINING DATA
# =====================================================================

class MultilingualDataset(Dataset):

    """
    Synthetic dataset for learning a language-conditioned
    phoneme representation.

    Training target:
        Reconstruct the phoneme token sequence.

    This is a representation-learning demonstration,
    not a real speech-alignment dataset.
    """

    def __init__(
        self,
        num_samples,
        vocabulary
    ):

        self.samples = []

        for _ in range(num_samples):

            language = random.choice(
                list(LANGUAGES.keys())
            )

            source_language, text = random.choice(
                [
                    item
                    for item in SAMPLE_TEXTS
                    if item[0] == language
                ]
            )

            phonemes = text_to_phonemes(
                text,
                language
            )

            ids = vocabulary.encode(
                phonemes
            )

            ids = ids[:MAX_SEQUENCE_LENGTH]

            attention_mask = [
                1
            ] * len(ids)

            padding = (
                MAX_SEQUENCE_LENGTH - len(ids)
            )

            ids += [
                PAD_ID
            ] * padding

            attention_mask += [
                0
            ] * padding

            self.samples.append(
                (
                    ids,
                    attention_mask,
                    LANGUAGES[language]
                )
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(self, index):

        ids, mask, language = self.samples[index]

        return (

            torch.tensor(
                ids,
                dtype=torch.long
            ),

            torch.tensor(
                mask,
                dtype=torch.long
            ),

            torch.tensor(
                language,
                dtype=torch.long
            )

        )


# =====================================================================
# STEP 6 — MULTILINGUAL PHONEME ENCODER
# =====================================================================

class MultilingualPhonemeEncoder(nn.Module):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()

        self.phoneme_embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,
            padding_idx=PAD_ID
        )

        self.language_embedding = nn.Embedding(
            len(LANGUAGES),
            EMBED_DIM
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                MAX_SEQUENCE_LENGTH,
                EMBED_DIM
            ) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=EMBED_DIM,

            nhead=NUM_HEADS,

            dim_feedforward=FFN_DIM,

            dropout=DROPOUT,

            batch_first=True,

            norm_first=True,

            activation="gelu"

        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=NUM_LAYERS

        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

        self.output = nn.Linear(
            EMBED_DIM,
            vocab_size
        )

    def forward(
        self,
        phoneme_ids,
        language_ids,
        attention_mask
    ):

        batch, length = phoneme_ids.shape

        x = self.phoneme_embedding(
            phoneme_ids
        )

        language_features = self.language_embedding(
            language_ids
        ).unsqueeze(1)

        x = (
            x
            + language_features
            + self.position_embedding[:, :length]
        )

        padding_mask = (
            attention_mask == 0
        )

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.norm(x)

        logits = self.output(x)

        # Masked mean pooling.
        mask = attention_mask.unsqueeze(
            -1
        ).float()

        pooled = (
            x * mask
        ).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1.0)

        return logits, x, pooled


# =====================================================================
# STEP 7 — TRAINING
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

    history = {
        "train": [],
        "validation": []
    }

    best_validation_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        train_losses = []

        for phonemes, mask, language in train_loader:

            phonemes = phonemes.to(DEVICE)

            mask = mask.to(DEVICE)

            language = language.to(DEVICE)

            logits, _, _ = model(
                phonemes,
                language,
                mask
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                phonemes.reshape(-1),
                ignore_index=PAD_ID
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(train_losses)
        )

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for phonemes, mask, language in val_loader:

                phonemes = phonemes.to(DEVICE)

                mask = mask.to(DEVICE)

                language = language.to(DEVICE)

                logits, _, _ = model(
                    phonemes,
                    language,
                    mask
                )

                loss = F.cross_entropy(
                    logits.reshape(
                        -1,
                        logits.size(-1)
                    ),
                    phonemes.reshape(-1),
                    ignore_index=PAD_ID
                )

                val_losses.append(
                    loss.item()
                )

        val_loss = float(
            np.mean(val_losses)
        )

        history["train"].append(
            train_loss
        )

        history["validation"].append(
            val_loss
        )

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Validation: {val_loss:.4f}"
        )

        if val_loss < best_validation_loss:

            best_validation_loss = val_loss

            torch.save(
                model.state_dict(),
                os.path.join(
                    OUTPUT_DIR,
                    "lesson51_best.pt"
                )
            )

    return history


# =====================================================================
# STEP 8 — PHONEME ENCODING INFERENCE
# =====================================================================

@torch.no_grad()
def encode_text(
    model,
    vocabulary,
    text,
    language
):

    model.eval()

    phonemes = text_to_phonemes(
        text,
        language
    )

    ids = vocabulary.encode(
        phonemes
    )

    ids = ids[:MAX_SEQUENCE_LENGTH]

    length = len(ids)

    padding = (
        MAX_SEQUENCE_LENGTH - length
    )

    ids += [
        PAD_ID
    ] * padding

    mask = (
        [1] * length
        + [0] * padding
    )

    phoneme_tensor = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    mask_tensor = torch.tensor(
        [mask],
        dtype=torch.long,
        device=DEVICE
    )

    language_tensor = torch.tensor(
        [LANGUAGES[language]],
        dtype=torch.long,
        device=DEVICE
    )

    logits, features, pooled = model(
        phoneme_tensor,
        language_tensor,
        mask_tensor
    )

    return {

        "phonemes": phonemes,

        "token_ids": ids[:length],

        "features": features[
            0,
            :length
        ].cpu(),

        "sentence_embedding": pooled[
            0
        ].cpu()

    }


# =====================================================================
# STEP 9 — VISUALIZATION
# =====================================================================

def plot_training(history):

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        history["train"],
        label="Training"
    )

    plt.plot(
        history["validation"],
        label="Validation"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Cross Entropy")

    plt.title(
        "Lesson 51 — Multilingual Phoneme Encoder Training"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson51_training.png"
        )
    )

    plt.show()


def plot_embeddings(embeddings, labels):

    """
    PCA projection of sentence embeddings.
    """

    from sklearn.decomposition import PCA

    matrix = np.stack(
        embeddings
    )

    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(
        matrix
    )

    plt.figure(
        figsize=(9, 7)
    )

    for language_id in range(
        len(LANGUAGES)
    ):

        indices = [
            i
            for i, label in enumerate(labels)
            if label == language_id
        ]

        plt.scatter(
            reduced[indices, 0],
            reduced[indices, 1],
            label=LANGUAGE_NAMES[language_id],
            s=90
        )

    plt.xlabel("PCA Component 1")

    plt.ylabel("PCA Component 2")

    plt.title(
        "Lesson 51 — Multilingual Sentence Embeddings"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "lesson51_embeddings.png"
        )
    )

    plt.show()


# =====================================================================
# STEP 10 — MAIN
# =====================================================================

def main():

    print("=" * 72)

    print(
        "LESSON 51 — MULTILINGUAL PHONEME ENCODER"
    )

    print("=" * 72)

    print(
        f"\nDevice: {DEVICE}"
    )

    # ---------------------------------------------------------------
    # Inspect frontend
    # ---------------------------------------------------------------

    print_frontend_examples()

    # ---------------------------------------------------------------
    # Vocabulary
    # ---------------------------------------------------------------

    vocabulary = PhonemeVocabulary()

    vocabulary.build(
        SAMPLE_TEXTS
    )

    print(
        f"\nPhoneme vocabulary size: {len(vocabulary)}"
    )

    print(
        "\nVocabulary:"
    )

    print(
        vocabulary.token_to_id
    )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    train_dataset = MultilingualDataset(
        NUM_TRAIN_SAMPLES,
        vocabulary
    )

    val_dataset = MultilingualDataset(
        NUM_VAL_SAMPLES,
        vocabulary
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"\nTraining samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = MultilingualPhonemeEncoder(
        len(vocabulary)
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: {parameters:,}"
    )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    print(
        "\nStarting training...\n"
    )

    history = train_model(
        model,
        train_loader,
        val_loader
    )

    # ---------------------------------------------------------------
    # Load best model
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        os.path.join(
            OUTPUT_DIR,
            "lesson51_best.pt"
        ),
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    # ---------------------------------------------------------------
    # Multilingual inference
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "MULTILINGUAL TEXT ENCODING"
    )

    print(
        "=" * 72
    )

    embeddings = []

    labels = []

    for language, text in SAMPLE_TEXTS:

        result = encode_text(
            model,
            vocabulary,
            text,
            language
        )

        embedding = result[
            "sentence_embedding"
        ]

        embeddings.append(
            embedding.numpy()
        )

        labels.append(
            LANGUAGES[language]
        )

        print(
            f"\nLanguage: {language}"
        )

        print(
            f"Text: {text}"
        )

        print(
            f"Phonemes: {' '.join(result['phonemes'])}"
        )

        print(
            f"Token IDs: {result['token_ids']}"
        )

        print(
            f"Embedding shape: {tuple(embedding.shape)}"
        )

    # ---------------------------------------------------------------
    # Save model and vocabulary
    # ---------------------------------------------------------------

    torch.save(
        model.state_dict(),
        os.path.join(
            OUTPUT_DIR,
            "lesson51_final.pt"
        )
    )

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "lesson51_embeddings.npz"
        ),
        embeddings=np.array(embeddings),
        labels=np.array(labels)
    )

    # ---------------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------------

    plot_training(
        history
    )

    plot_embeddings(
        embeddings,
        labels
    )

    # ---------------------------------------------------------------
    # Completion
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "LESSON 51 COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        "\nOutput directory:",
        OUTPUT_DIR
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  lesson51_best.pt"
    )

    print(
        "  lesson51_final.pt"
    )

    print(
        "  lesson51_training.png"
    )

    print(
        "  lesson51_embeddings.png"
    )

    print(
        "  lesson51_embeddings.npz"
    )


if __name__ == "__main__":

    main()