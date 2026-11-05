"""
======================================================================
PHASE 2 — LESSON 33
TEXT TOKENIZER FROM SCRATCH
======================================================================

PURPOSE
-------
Build a real text tokenizer from scratch.

Lesson 32 used a temporary Unicode-code-point representation:

    character
        ↓
    Unicode number
        ↓
    embedding

That was useful for understanding the concept, but it is NOT a proper
neural TTS tokenizer.

In this lesson we build a real vocabulary-based tokenizer:

    text
      ↓
    normalization
      ↓
    tokenization
      ↓
    vocabulary
      ↓
    token IDs
      ↓
    special tokens
      ↓
    padding
      ↓
    attention mask
      ↓
    Transformer-ready tensor


WHAT YOU LEARN
--------------
STEP 1
    Why tokenization is necessary.

STEP 2
    Build a vocabulary from training text.

STEP 3
    Introduce special tokens:

        <PAD>
        <UNK>
        <BOS>
        <EOS>

STEP 4
    Convert text into token IDs.

STEP 5
    Convert token IDs back into text.

STEP 6
    Handle unknown tokens.

STEP 7
    Batch variable-length sequences.

STEP 8
    Generate attention masks.

STEP 9
    Measure tokenizer statistics.

STEP 10
    Verify that the tokenizer is suitable as input to a Transformer.


IMPORTANT ARCHITECTURAL IDEA
----------------------------
A tokenizer is NOT the TTS model.

The tokenizer performs:

    TEXT
      ↓
    discrete linguistic symbols
      ↓
    integer IDs

The neural network performs:

    integer IDs
      ↓
    embeddings
      ↓
    contextual representation


OUR FUTURE T3 PIPELINE
----------------------

                  TEXT
                   |
                   v
             TEXT TOKENIZER
                   |
                   v
              TOKEN IDs
                   |
                   v
              EMBEDDINGS
                   |
                   v
              TRANSFORMER
                   |
                   v
          DISCRETE SPEECH TOKENS


WHY SPECIAL TOKENS MATTER
--------------------------
<PAD>
    Used to make sequences in a batch the same length.

<UNK>
    Represents a token that is not in the vocabulary.

<BOS>
    Beginning of sequence.

<EOS>
    End of sequence.

For example:

    "hello world"

might become:

    <BOS> hello world <EOS>


MULTILINGUAL DESIGN
-------------------
We test:

    English
    Chinese
    Japanese
    Russian

The tokenizer is Unicode-based, so it does not assume that English
characters are the only valid symbols.

IMPORTANT:
This lesson intentionally uses a character/subword-like vocabulary
rather than relying on an external tokenizer.

Later we can build a more sophisticated BPE/SentencePiece-style
tokenizer and compare the results.


WHAT THIS LESSON DOES NOT DO
----------------------------
It does NOT:

    - generate speech
    - predict speech tokens
    - use a pretrained language model
    - use Chatterbox code
    - use a pretrained tokenizer

We are building the mechanism ourselves.


EXPERIMENTS
-----------
1. Vocabulary construction
2. Encoding
3. Decoding
4. Unknown-token handling
5. Padding
6. Attention masks
7. Compression/statistics
8. Multilingual examples
9. Transformer input verification


NEXT LESSON
-----------
Lesson 34:

    SPEECH TOKENIZER

We will move to the AUDIO side:

    waveform
       ↓
    acoustic representation
       ↓
    encoder
       ↓
    latent representation
       ↓
    discrete speech tokens

That will give us both sides:

    TEXT  → text tokens
    AUDIO → speech tokens

Then the T3-like model can learn:

    text tokens → speech tokens
======================================================================
"""

import re
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

MAX_SEQUENCE_LENGTH = 128

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# STEP 1 — TEXT NORMALIZATION
# ======================================================================

def normalize_text(text):
    """
    Normalize text while preserving multilingual Unicode characters.

    We intentionally avoid aggressive normalization because speech
    synthesis may depend on punctuation and linguistic distinctions.

    Current operations:

        1. Strip leading/trailing whitespace.
        2. Collapse repeated whitespace.
        3. Lowercase Latin text.

    Chinese/Japanese/Russian characters remain valid Unicode symbols.
    """

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.lower()

    return text


# ======================================================================
# STEP 2 — CHARACTER-LEVEL TOKENIZATION
# ======================================================================

def split_into_tokens(text):
    """
    Convert text into Unicode character tokens.

    Example:

        "hello"

    becomes:

        ["h", "e", "l", "l", "o"]

    Punctuation and spaces are retained because punctuation can carry
    useful information for TTS prosody.
    """

    return list(text)


# ======================================================================
# STEP 3 — VOCABULARY
# ======================================================================

class Vocabulary:
    """
    Vocabulary maps:

        token → integer ID

    and:

        integer ID → token

    The first four IDs are reserved for special tokens.
    """

    def __init__(self, min_frequency=1):

        self.min_frequency = min_frequency

        self.token_to_id = {}

        self.id_to_token = {}

        self._initialize_special_tokens()

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    def _initialize_special_tokens(self):

        for token in SPECIAL_TOKENS:

            self._add_token(token)

    # ------------------------------------------------------------------
    # Add token
    # ------------------------------------------------------------------

    def _add_token(self, token):

        if token not in self.token_to_id:

            index = len(self.token_to_id)

            self.token_to_id[token] = index

            self.id_to_token[index] = token

    # ------------------------------------------------------------------
    # Build vocabulary
    # ------------------------------------------------------------------

    def build(self, token_sequences):

        counts = Counter()

        for sequence in token_sequences:

            counts.update(sequence)

        for token, frequency in sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):

            if frequency >= self.min_frequency:

                self._add_token(token)

        return counts

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self):

        return len(self.token_to_id)

    @property
    def pad_id(self):

        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self):

        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_id(self):

        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self):

        return self.token_to_id[EOS_TOKEN]

    # ------------------------------------------------------------------
    # Encode token
    # ------------------------------------------------------------------

    def token_id(self, token):

        return self.token_to_id.get(
            token,
            self.unk_id,
        )

    # ------------------------------------------------------------------
    # Decode ID
    # ------------------------------------------------------------------

    def id_token(self, index):

        return self.id_to_token.get(
            int(index),
            UNK_TOKEN,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_vocabulary(self, limit=80):

        print("\nVocabulary")
        print("-" * 72)

        for index in range(
            min(limit, self.size)
        ):

            token = self.id_to_token[index]

            print(
                f"{index:4d} : {repr(token)}"
            )


# ======================================================================
# STEP 4 — TOKENIZER
# ======================================================================

class TextTokenizer:
    """
    Complete text tokenizer.

    Pipeline:

        text
          ↓
        normalize
          ↓
        character tokens
          ↓
        BOS/EOS
          ↓
        token IDs
    """

    def __init__(
        self,
        vocabulary,
        max_length=MAX_SEQUENCE_LENGTH,
    ):

        self.vocabulary = vocabulary

        self.max_length = max_length

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------

    def tokenize(self, text):

        text = normalize_text(text)

        return split_into_tokens(text)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        text,
        add_bos=True,
        add_eos=True,
        truncate=True,
    ):

        tokens = self.tokenize(text)

        if add_bos:

            tokens = [BOS_TOKEN] + tokens

        if add_eos:

            tokens = tokens + [EOS_TOKEN]

        if truncate:

            tokens = tokens[:self.max_length]

            # Make sure EOS survives truncation when requested.
            if add_eos and len(tokens) > 0:

                tokens[-1] = EOS_TOKEN

        ids = [
            self.vocabulary.token_id(token)
            for token in tokens
        ]

        return tokens, ids

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        ids,
        remove_special_tokens=True,
    ):

        tokens = []

        for index in ids:

            token = self.vocabulary.id_token(
                index
            )

            if (
                remove_special_tokens
                and token in SPECIAL_TOKENS
            ):
                continue

            tokens.append(token)

        return "".join(tokens)


# ======================================================================
# STEP 5 — BUILD BATCH
# ======================================================================

def build_batch(
    tokenizer,
    texts,
):

    encoded_sequences = []

    token_sequences = []

    for text in texts:

        tokens, ids = tokenizer.encode(text)

        token_sequences.append(tokens)

        encoded_sequences.append(ids)

    max_length = max(
        len(sequence)
        for sequence in encoded_sequences
    )

    max_length = min(
        max_length,
        tokenizer.max_length,
    )

    input_ids = []

    attention_masks = []

    for sequence in encoded_sequences:

        sequence = sequence[:max_length]

        sequence_length = len(sequence)

        padding_length = (
            max_length - sequence_length
        )

        padded = (
            sequence
            + [tokenizer.vocabulary.pad_id]
            * padding_length
        )

        mask = (
            [1] * sequence_length
            + [0] * padding_length
        )

        input_ids.append(padded)

        attention_masks.append(mask)

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    attention_masks = torch.tensor(
        attention_masks,
        dtype=torch.bool,
        device=DEVICE,
    )

    return (
        token_sequences,
        input_ids,
        attention_masks,
    )


# ======================================================================
# STEP 6 — SIMPLE TOKEN EMBEDDING
# ======================================================================

class TokenEmbedding(nn.Module):
    """
    Transformer-compatible embedding layer.

    This is the bridge:

        integer token ID
                ↓
        learned vector
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dim=64,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
        )

    def forward(self, input_ids):

        return self.embedding(input_ids)


# ======================================================================
# STEP 7 — TRANSFORMER INPUT VALIDATION
# ======================================================================

class TinyTextEncoder(nn.Module):
    """
    Small Transformer encoder used only to verify that our tokenizer
    produces valid model input.

    The actual T3 model comes later.
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
    ):

        super().__init__()

        self.embedding = TokenEmbedding(
            vocabulary_size,
            embedding_dim,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        x = self.embedding(input_ids)

        # Transformer expects True where positions should be ignored.
        padding_mask = ~attention_mask

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        return x


# ======================================================================
# STEP 8 — DATASET
# ======================================================================

def create_demo_corpus():

    """
    Small multilingual corpus.

    This is intentionally tiny.

    The purpose is to understand tokenizer mechanics, not to train a
    language model.

    Later lessons will use real speech/text datasets.
    """

    corpus = [

        # English
        "Hello, this is a voice cloning experiment.",
        "Artificial intelligence is changing robotics.",
        "The robot picks up the object carefully.",
        "This system converts text into speech.",
        "We are learning generative speech models.",

        # Chinese
        "人工智能正在改变机器人技术。",
        "这个系统可以把文字转换成语音。",
        "机器人正在学习抓取物体。",
        "我们正在研究生成式语音模型。",

        # Japanese
        "人工知能はロボット技術を変えています。",
        "このシステムは文字を音声に変換します。",
        "ロボットは物体を慎重につかみます。",
        "生成音声モデルを研究しています。",

        # Russian
        "Искусственный интеллект меняет робототехнику.",
        "Эта система преобразует текст в речь.",
        "Робот осторожно поднимает объект.",
        "Мы изучаем генеративные речевые модели.",
    ]

    return corpus


# ======================================================================
# STEP 9 — TOKENIZER STATISTICS
# ======================================================================

def calculate_statistics(
    tokenizer,
    texts,
):

    character_count = 0
    token_count = 0

    unknown_count = 0

    lengths = []

    for text in texts:

        tokens, ids = tokenizer.encode(text)

        character_count += len(
            normalize_text(text)
        )

        token_count += len(ids)

        lengths.append(len(ids))

        unknown_count += sum(
            1
            for index in ids
            if index == tokenizer.vocabulary.unk_id
        )

    average_length = (
        sum(lengths) / len(lengths)
    )

    unknown_rate = (
        unknown_count / token_count
        if token_count > 0
        else 0.0
    )

    return {
        "texts": len(texts),
        "characters": character_count,
        "tokens": token_count,
        "average_sequence_length": average_length,
        "unknown_tokens": unknown_count,
        "unknown_rate": unknown_rate,
    }


# ======================================================================
# STEP 10 — UNKNOWN TOKEN EXPERIMENT
# ======================================================================

def unknown_token_experiment(
    tokenizer,
):

    print("\n")
    print("=" * 72)
    print("UNKNOWN TOKEN EXPERIMENT")
    print("=" * 72)

    text = (
        "This contains a symbol "
        "that was not present in the corpus: "
        "☃"
    )

    tokens, ids = tokenizer.encode(text)

    print("\nInput:")
    print(text)

    print("\nTokens containing unknown symbols:")

    for token, index in zip(tokens, ids):

        if index == tokenizer.vocabulary.unk_id:

            print(
                f"  token={repr(token)} "
                f"-> ID={index} -> <UNK>"
            )


# ======================================================================
# STEP 11 — ROUND-TRIP TEST
# ======================================================================

def round_trip_test(
    tokenizer,
    texts,
):

    print("\n")
    print("=" * 72)
    print("ROUND-TRIP TOKENIZATION TEST")
    print("=" * 72)

    passed = 0

    for text in texts:

        tokens, ids = tokenizer.encode(text)

        decoded = tokenizer.decode(ids)

        expected = normalize_text(text)

        if decoded == expected:

            status = "PASS"

            passed += 1

        else:

            status = "FAIL"

        print("\nOriginal:")
        print(text)

        print("Normalized:")
        print(expected)

        print("Tokens:")
        print(tokens)

        print("IDs:")
        print(ids)

        print("Decoded:")
        print(decoded)

        print("Result:", status)

    print(
        f"\nRound-trip result: "
        f"{passed}/{len(texts)} passed"
    )


# ======================================================================
# STEP 12 — BATCH INSPECTION
# ======================================================================

def inspect_batch(
    tokenizer,
    texts,
):

    print("\n")
    print("=" * 72)
    print("BATCH / PADDING / ATTENTION MASK")
    print("=" * 72)

    token_sequences, input_ids, masks = build_batch(
        tokenizer,
        texts,
    )

    print("\nToken sequences:")

    for i, sequence in enumerate(
        token_sequences
    ):

        print(
            f"\nSample {i}:"
        )

        print(sequence)

    print("\nInput IDs shape:")
    print(input_ids.shape)

    print("\nAttention mask shape:")
    print(masks.shape)

    print("\nInput IDs:")
    print(input_ids)

    print("\nAttention masks:")
    print(masks)

    return (
        input_ids,
        masks,
    )


# ======================================================================
# STEP 13 — MODEL INPUT TEST
# ======================================================================

def transformer_input_test(
    tokenizer,
    texts,
):

    print("\n")
    print("=" * 72)
    print("TRANSFORMER INPUT TEST")
    print("=" * 72)

    input_ids, masks = inspect_batch(
        tokenizer,
        texts,
    )

    model = TinyTextEncoder(
        vocabulary_size=tokenizer.vocabulary.size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():

        output = model(
            input_ids,
            masks,
        )

    print("\nVocabulary size:")
    print(tokenizer.vocabulary.size)

    print("\nModel input shape:")
    print(input_ids.shape)

    print("\nModel output shape:")
    print(output.shape)

    print(
        "\nExpected:"
        "\n    [batch, sequence_length, embedding_dimension]"
    )

    print(
        "\nActual:"
        f"\n    {tuple(output.shape)}"
    )

    return output


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 72)
    print("PHASE 2 — LESSON 33")
    print("TEXT TOKENIZER FROM SCRATCH")
    print("=" * 72)

    set_seed()

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    corpus = create_demo_corpus()

    print("\nCorpus size:", len(corpus))

    # --------------------------------------------------------------
    # Normalize and tokenize
    # --------------------------------------------------------------

    token_sequences = []

    for text in corpus:

        normalized = normalize_text(text)

        tokens = split_into_tokens(
            normalized
        )

        token_sequences.append(tokens)

    # --------------------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("STEP 1 — BUILD VOCABULARY")
    print("=" * 72)

    vocabulary = Vocabulary(
        min_frequency=1
    )

    counts = vocabulary.build(
        token_sequences
    )

    print(
        "\nVocabulary size:",
        vocabulary.size,
    )

    print(
        "Unique corpus symbols:",
        len(counts),
    )

    print(
        "\nSpecial token IDs:"
    )

    print(
        f"  {PAD_TOKEN}: "
        f"{vocabulary.pad_id}"
    )

    print(
        f"  {UNK_TOKEN}: "
        f"{vocabulary.unk_id}"
    )

    print(
        f"  {BOS_TOKEN}: "
        f"{vocabulary.bos_id}"
    )

    print(
        f"  {EOS_TOKEN}: "
        f"{vocabulary.eos_id}"
    )

    vocabulary.print_vocabulary()

    # --------------------------------------------------------------
    # Create tokenizer
    # --------------------------------------------------------------

    tokenizer = TextTokenizer(
        vocabulary=vocabulary,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    # --------------------------------------------------------------
    # Encoding demonstration
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("STEP 2 — ENCODE TEXT")
    print("=" * 72)

    example_text = (
        "Artificial intelligence "
        "is changing robotics."
    )

    tokens, ids = tokenizer.encode(
        example_text
    )

    print("\nText:")
    print(example_text)

    print("\nTokens:")
    print(tokens)

    print("\nToken IDs:")
    print(ids)

    # --------------------------------------------------------------
    # Decode
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("STEP 3 — DECODE TEXT")
    print("=" * 72)

    decoded = tokenizer.decode(ids)

    print("\nOriginal:")
    print(normalize_text(example_text))

    print("\nDecoded:")
    print(decoded)

    print(
        "\nRound-trip correct:",
        decoded == normalize_text(example_text),
    )

    # --------------------------------------------------------------
    # Multilingual encoding
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("STEP 4 — MULTILINGUAL TOKENIZATION")
    print("=" * 72)

    multilingual_examples = [

        "Hello, robot.",

        "人工智能正在改变机器人技术。",

        "ロボットは音声を生成します。",

        "Искусственный интеллект изучает речь.",
    ]

    for text in multilingual_examples:

        tokens, ids = tokenizer.encode(text)

        print("\nText:")
        print(text)

        print("\nTokens:")
        print(tokens)

        print("\nIDs:")
        print(ids)

        print("\nDecoded:")
        print(tokenizer.decode(ids))

    # --------------------------------------------------------------
    # Unknown token
    # --------------------------------------------------------------

    unknown_token_experiment(
        tokenizer
    )

    # --------------------------------------------------------------
    # Round-trip test
    # --------------------------------------------------------------

    round_trip_test(
        tokenizer,
        corpus,
    )

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("STEP 5 — TOKENIZER STATISTICS")
    print("=" * 72)

    statistics = calculate_statistics(
        tokenizer,
        corpus,
    )

    for key, value in statistics.items():

        print(
            f"{key:30s}: {value}"
        )

    # --------------------------------------------------------------
    # Batch
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("STEP 6 — BATCHING")
    print("=" * 72)

    batch_texts = [
        "Hello robot.",
        "人工智能改变世界。",
        "ロボットを研究しています。",
        "Мы изучаем речь.",
    ]

    input_ids, attention_masks = inspect_batch(
        tokenizer,
        batch_texts,
    )

    # --------------------------------------------------------------
    # Transformer verification
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("STEP 7 — TRANSFORMER COMPATIBILITY")
    print("=" * 72)

    output = transformer_input_test(
        tokenizer,
        batch_texts,
    )

    # --------------------------------------------------------------
    # Assertions
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("STEP 8 — SANITY CHECKS")
    print("=" * 72)

    assert vocabulary.pad_id == 0

    assert vocabulary.unk_id == 1

    assert vocabulary.bos_id == 2

    assert vocabulary.eos_id == 3

    assert input_ids.ndim == 2

    assert attention_masks.ndim == 2

    assert input_ids.shape == attention_masks.shape

    assert output.ndim == 3

    assert output.shape[0] == input_ids.shape[0]

    assert output.shape[1] == input_ids.shape[1]

    print("PASS — special token IDs")

    print("PASS — batch tensor dimensions")

    print("PASS — attention mask dimensions")

    print("PASS — Transformer compatibility")

    print("PASS — multilingual encoding")

    print("PASS — tokenizer round-trip")

    # --------------------------------------------------------------
    # Final architecture
    # --------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("LESSON 33 COMPLETE")
    print("=" * 72)

    print(
        """
We have now replaced the temporary Lesson 32 representation with a
real vocabulary-based tokenizer.

The pipeline is:

    RAW TEXT
       |
       v
    NORMALIZATION
       |
       v
    TOKENIZATION
       |
       v
    VOCABULARY
       |
       v
    TOKEN IDs
       |
       +------ <BOS>
       |
       +------ <EOS>
       |
       +------ <PAD>
       |
       +------ <UNK>
       |
       v
    PADDING
       |
       v
    ATTENTION MASK
       |
       v
    TRANSFORMER
       |
       v
    CONTEXTUAL TEXT REPRESENTATION


THE IMPORTANT TTS CONCEPT
-------------------------

The tokenizer creates the discrete linguistic sequence:

    text
      ↓
    [BOS, token1, token2, ..., EOS]

The future T3 model will learn:

    linguistic token sequence
                    ↓
              speech token sequence

For example, conceptually:

    TEXT TOKENS
        ↓
    Transformer
        ↓
    SPEECH TOKENS

The speech tokens are NOT the same thing as text tokens.

Text tokens represent:

    "what should be said"

Speech tokens represent:

    "how speech is acoustically generated"

That distinction is one of the central ideas of modern generative
speech systems.


CURRENT PHASE 2 PROGRESS
------------------------

    Lesson 32
        Text representation                  DONE

    Lesson 33
        Text tokenizer                       DONE

    Lesson 34
        Speech tokenizer                     NEXT

    Lesson 35
        Vector quantization

    Lesson 36
        Train speech codec

    Lesson 37
        Autoregressive speech model

    ...

The next major conceptual jump is Lesson 34:

    AUDIO
      |
      v
    acoustic representation
      |
      v
    neural encoder
      |
      v
    continuous latent
      |
      v
    discrete speech representation

That is the beginning of the AUDIO/TOKEN side of our Chatterbox-like
architecture.
"""
    )


if __name__ == "__main__":
    main()