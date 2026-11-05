"""
========================================================================
PHASE 1 — LESSON 7
PHONETICS, IPA, G2P, TEXT NORMALIZATION AND TOKENIZATION
========================================================================

PURPOSE
-------
Build a small educational TTS text frontend from scratch.

This program demonstrates how raw written text becomes numerical
representations that a neural TTS model can consume.

PIPELINE
--------
Raw text
    |
    v
Text normalization
    |
    v
Word tokenization
    |
    +-----------------------------+
    |                             |
    v                             v
Character tokenizer         Toy G2P converter
    |                             |
    v                             v
Character IDs               Phoneme sequence
                                  |
                                  v
                            Phoneme tokenizer
                                  |
                                  v
                             Phoneme IDs
                                  |
                                  v
                         Padding / batching
                                  |
                                  v
                         TTS model input


========================================================================
THEORY 1 — GRAPHEMES, PHONEMES AND PHONES
========================================================================

A GRAPHEME is a written symbol or a group of written symbols.

Examples:
    a
    b
    sh
    ch

A PHONEME is an abstract sound unit that can distinguish meaning.

Examples:
    /p/
    /b/
    /t/

A PHONE is the actual physical realization of a speech sound.

A phoneme can have multiple phonetic realizations called ALLOPHONES.

For example, English /p/ can be aspirated in "pin" and less aspirated
after /s/ in "spin".

This is why spelling, phonemic transcription and physical speech
are not interchangeable concepts.


========================================================================
THEORY 2 — GRAPHEME-TO-PHONEME CONVERSION
========================================================================

G2P converts written symbols into a pronunciation representation.

Mathematically:

    G2P: G* -> P*

Where:
    G = grapheme vocabulary
    P = phoneme vocabulary
    * = a sequence of symbols

A neural G2P system can be described as:

    P(phoneme_sequence | grapheme_sequence)

A training dataset may contain pairs such as:

    "cat"       -> K AE T
    "dog"       -> D AO G
    "hello"     -> HH AH L OW

A neural model learns this mapping from many examples.

This educational program uses a small dictionary instead of a
trained G2P model. The dictionary is intentionally limited so that
the conversion process remains transparent.


========================================================================
THEORY 3 — TEXT NORMALIZATION
========================================================================

Written text often contains forms that should not be spoken literally.

Examples:

    "25"       -> "twenty-five"
    "50%"      -> "fifty percent"
    "Dr."      -> "Doctor"

Text normalization is context-sensitive.

For example:

    "2026"

Could be spoken as:
    "twenty twenty-six"
    "two thousand twenty-six"

A production TTS frontend needs rules, contextual information,
language-specific processing and sometimes machine-learning models.


========================================================================
THEORY 4 — TOKENIZATION
========================================================================

Neural networks process numerical tensors, not raw text strings.

A tokenizer maps symbolic units into integer IDs.

Character-level example:

    Vocabulary:
        a -> 0
        b -> 1
        c -> 2
        ...

    "cat" -> [2, 0, 19]     # Illustrative IDs only.

Phoneme-level example:

    Vocabulary:
        K  -> 0
        AE -> 1
        T  -> 2

    "K AE T" -> [0, 1, 2]

Special tokens are commonly used:

    <PAD> = padding
    <UNK> = unknown symbol
    <BOS> = beginning of sequence
    <EOS> = end of sequence

The actual IDs depend on the vocabulary construction order.


========================================================================
THEORY 5 — PADDING
========================================================================

Neural networks commonly process multiple sequences in a batch.

But sequences may have different lengths.

Example:

    Sequence A: [2, 4, 7]
    Sequence B: [3, 8]

After padding:

    Sequence A: [2, 4, 7]
    Sequence B: [3, 8, 0]

The padding ID is not a real linguistic symbol.

A model must use an appropriate padding mask so that padding does
not contribute incorrectly to attention or loss calculations.


========================================================================
IMPORTANT LIMITATIONS
========================================================================

This is an educational frontend, NOT a production multilingual G2P.

It does not provide:

    - Complete English pronunciation coverage
    - Context-sensitive pronunciation prediction
    - Full IPA conversion
    - Accurate syllabification
    - Korean phonological rule processing
    - Japanese kanji reading disambiguation
    - Prosody prediction
    - Neural G2P training

The dictionary uses ARPABET-like phoneme symbols for simplicity.

A real system should select a phoneme inventory and pronunciation
standard deliberately.

For multilingual TTS, language identification and language-specific
normalization/G2P are important architectural components.


========================================================================
DEPENDENCIES
========================================================================

    pip install numpy matplotlib

RUN
---

    python phase1_lesson7_text_frontend.py

Custom text:

    python phase1_lesson7_text_frontend.py --text "I have 2 apples."

Run selected experiment:

    python phase1_lesson7_text_frontend.py --experiment 1
    python phase1_lesson7_text_frontend.py --experiment 2
    python phase1_lesson7_text_frontend.py --experiment 3
    python phase1_lesson7_text_frontend.py --experiment 4

========================================================================
"""

from __future__ import annotations

import argparse
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURATION
# =====================================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
WORD_BOUNDARY = "<WB>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    WORD_BOUNDARY,
]

DEFAULT_TEXT = "I have 2 apples. Hello computer!"

# This is a deliberately small educational pronunciation dictionary.
#
# The phonemes are ARPABET-like symbols, not IPA.
#
# Each word maps to a list of phoneme tokens.
#
# For example:
#
#   "cat" -> ["K", "AE", "T"]
#
# A real dictionary would contain many more entries and may include
# multiple pronunciations for the same word.

PRONUNCIATION_DICTIONARY: Dict[str, List[str]] = {
    "i": ["AY"],
    "have": ["HH", "AE", "V"],
    "two": ["T", "UW"],
    "apples": ["AE", "P", "AH", "L", "Z"],
    "cat": ["K", "AE", "T"],
    "dog": ["D", "AO", "G"],
    "see": ["S", "IY"],
    "hello": ["HH", "AH", "L", "OW"],
    "computer": ["K", "AH", "M", "P", "Y", "UW", "T", "ER"],
    "though": ["DH", "OW"],
    "through": ["TH", "R", "UW"],
    "tough": ["T", "AH", "F"],
    "thought": ["TH", "AO", "T"],
    "doctor": ["D", "AA", "K", "T", "ER"],
    "smith": ["S", "M", "IH", "TH"],
    "twenty": ["T", "W", "EH", "N", "T", "IY"],
    "five": ["F", "AY", "V"],
    "fifty": ["F", "IH", "F", "T", "IY"],
    "percent": ["P", "ER", "S", "EH", "N", "T"],
    "dollars": ["D", "AA", "L", "ER", "Z"],
    "and": ["AE", "N", "D"],
    "cents": ["S", "EH", "N", "T", "S"],
    "one": ["W", "AH", "N"],
    "three": ["TH", "R", "IY"],
    "four": ["F", "AO", "R"],
    "six": ["S", "IH", "K", "S"],
    "seven": ["S", "EH", "V", "AH", "N"],
    "eight": ["EY", "T"],
    "nine": ["N", "AY", "N"],
    "zero": ["Z", "IH", "R", "OW"],
    "ten": ["T", "EH", "N"],
    "hundred": ["HH", "AH", "N", "D", "R", "AH", "D"],
    "thousand": ["TH", "AW", "Z", "AH", "N", "D"],
}


# =====================================================================
# THEORY IMPLEMENTATION 1 — NUMBER NORMALIZATION
# =====================================================================

ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

TEENS = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def integer_to_words(number: int) -> str:
    """
    Convert an integer from 0 to 999,999 into English words.

    This is a simple educational implementation.

    Real-world number normalization requires context:
        - dates
        - years
        - phone numbers
        - currency
        - decimal values
        - measurement units
    """

    if number < 0:
        return "minus " + integer_to_words(abs(number))

    if number > 999_999:
        raise ValueError("This educational converter supports up to 999999.")

    if number < 10:
        return ONES[number]

    if number < 20:
        return TEENS[number]

    if number < 100:
        tens_value = (number // 10) * 10
        remainder = number % 10

        if remainder == 0:
            return TENS[tens_value]

        return TENS[tens_value] + "-" + ONES[remainder]

    if number < 1000:
        hundreds = number // 100
        remainder = number % 100

        result = ONES[hundreds] + " hundred"

        if remainder:
            result += " " + integer_to_words(remainder)

        return result

    thousands = number // 1000
    remainder = number % 1000

    result = integer_to_words(thousands) + " thousand"

    if remainder:
        result += " " + integer_to_words(remainder)

    return result


def normalize_numbers(text: str) -> str:
    """
    Convert simple integers and percentages into spoken English.

    Supported examples:

        "I have 2 apples."
            -> "I have two apples."

        "50%"
            -> "fifty percent"

    Decimal numbers, currency expressions and dates are intentionally
    left for future lessons.
    """

    # First normalize percentages.
    def percentage_replacement(match: re.Match) -> str:
        value = int(match.group(1))
        return integer_to_words(value) + " percent"

    text = re.sub(
        r"\b(\d+)\s*%",
        percentage_replacement,
        text,
    )

    # Then normalize standalone integers.
    def integer_replacement(match: re.Match) -> str:
        value = int(match.group(0))

        try:
            return integer_to_words(value)
        except ValueError:
            return match.group(0)

    text = re.sub(
        r"\b\d+\b",
        integer_replacement,
        text,
    )

    return text


# =====================================================================
# THEORY IMPLEMENTATION 2 — TEXT NORMALIZATION
# =====================================================================

ABBREVIATIONS = {
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "Ms.": "Miss",
    "Prof.": "Professor",
}


def normalize_text(text: str) -> str:
    """
    Apply a simple text-normalization pipeline.

    Steps:
        1. Replace common abbreviations.
        2. Expand percentages and integers.
        3. Normalize repeated whitespace.
        4. Strip leading/trailing whitespace.

    This is NOT a full text normalization engine.
    """

    normalized = text

    for abbreviation, expansion in ABBREVIATIONS.items():
        normalized = normalized.replace(
            abbreviation,
            expansion,
        )

    normalized = normalize_numbers(normalized)

    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


# =====================================================================
# THEORY IMPLEMENTATION 3 — WORD TOKENIZATION
# =====================================================================

def tokenize_words(text: str) -> List[str]:
    """
    Split text into words and punctuation.

    This is a simple English-oriented tokenizer.

    Example:

        "Hello, world!"

    Returns:

        ["Hello", ",", "world", "!"]

    It is not suitable as a universal multilingual tokenizer.
    """

    return re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]",
        text,
    )


# =====================================================================
# THEORY IMPLEMENTATION 4 — CHARACTER TOKENIZATION
# =====================================================================

def build_character_vocabulary(
    texts: Sequence[str],
) -> Dict[str, int]:
    """
    Build a character vocabulary from a collection of texts.

    The vocabulary is constructed from observed characters.

    Special tokens are included first.

    In a production model, vocabulary construction must be performed
    consistently and saved alongside the trained model.
    """

    characters = set()

    for text in texts:
        characters.update(text.lower())

    vocabulary = {}

    for token in SPECIAL_TOKENS:
        vocabulary[token] = len(vocabulary)

    for character in sorted(characters):
        if character not in vocabulary:
            vocabulary[character] = len(vocabulary)

    return vocabulary


def encode_characters(
    text: str,
    vocabulary: Dict[str, int],
) -> List[int]:
    """
    Convert text characters into numerical IDs.

    Unknown characters are mapped to <UNK>.
    """

    unknown_id = vocabulary[UNK_TOKEN]

    return [
        vocabulary.get(character, unknown_id)
        for character in text.lower()
    ]


def decode_characters(
    token_ids: Sequence[int],
    vocabulary: Dict[str, int],
) -> str:
    """
    Convert character IDs back into text.

    Special tokens are omitted.
    """

    reverse_vocabulary = {
        index: token
        for token, index in vocabulary.items()
    }

    output = []

    for token_id in token_ids:
        token = reverse_vocabulary.get(token_id, UNK_TOKEN)

        if token in SPECIAL_TOKENS:
            continue

        output.append(token)

    return "".join(output)


# =====================================================================
# THEORY IMPLEMENTATION 5 — TOY GRAPHEME-TO-PHONEME CONVERSION
# =====================================================================

@dataclass
class G2PResult:
    words: List[str]
    phoneme_tokens: List[str]
    unknown_words: List[str]


def toy_g2p(
    text: str,
    dictionary: Dict[str, List[str]],
) -> G2PResult:
    """
    Convert words into phoneme tokens using a small dictionary.

    Word boundaries are represented by <WB>.

    Punctuation is preserved as a separate token.

    Unknown words are reported. They are not assigned fabricated
    pronunciations.

    Example:

        "cat dog"

        -> ["K", "AE", "T", "<WB>", "D", "AO", "G"]

    This implementation uses dictionary lookup only.
    """

    words = tokenize_words(text)

    phoneme_tokens = []
    unknown_words = []

    punctuation = set(string.punctuation)

    for word in words:

        if word in punctuation:
            phoneme_tokens.append(word)
            continue

        normalized_word = word.lower()

        if normalized_word in dictionary:
            pronunciation = dictionary[normalized_word]

            phoneme_tokens.extend(pronunciation)

        else:
            unknown_words.append(word)

            # Preserve the word boundary, but do not invent
            # a pronunciation.
            phoneme_tokens.append(UNK_TOKEN)

        phoneme_tokens.append(WORD_BOUNDARY)

    # Remove the final word-boundary marker.
    if phoneme_tokens and phoneme_tokens[-1] == WORD_BOUNDARY:
        phoneme_tokens.pop()

    return G2PResult(
        words=words,
        phoneme_tokens=phoneme_tokens,
        unknown_words=unknown_words,
    )


# =====================================================================
# THEORY IMPLEMENTATION 6 — PHONEME VOCABULARY
# =====================================================================

def build_phoneme_vocabulary(
    phoneme_sequences: Sequence[Sequence[str]],
) -> Dict[str, int]:
    """
    Build a numerical vocabulary for phoneme tokens.

    Unlike character tokenization, phoneme tokenization treats each
    phoneme symbol as one unit.

    For example:

        ["K", "AE", "T"]

    becomes three IDs, not a sequence of individual letters.
    """

    observed_tokens = set()

    for sequence in phoneme_sequences:
        observed_tokens.update(sequence)

    vocabulary = {}

    for token in SPECIAL_TOKENS:
        vocabulary[token] = len(vocabulary)

    for token in sorted(observed_tokens):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def encode_phonemes(
    phoneme_tokens: Sequence[str],
    vocabulary: Dict[str, int],
) -> List[int]:
    """
    Convert phoneme symbols into numerical IDs.
    """

    unknown_id = vocabulary[UNK_TOKEN]

    return [
        vocabulary.get(token, unknown_id)
        for token in phoneme_tokens
    ]


def decode_phonemes(
    token_ids: Sequence[int],
    vocabulary: Dict[str, int],
) -> List[str]:
    """
    Convert phoneme IDs back into symbolic tokens.
    """

    reverse_vocabulary = {
        index: token
        for token, index in vocabulary.items()
    }

    output = []

    for token_id in token_ids:
        token = reverse_vocabulary.get(token_id, UNK_TOKEN)

        if token in SPECIAL_TOKENS:
            continue

        output.append(token)

    return output


# =====================================================================
# THEORY IMPLEMENTATION 7 — PADDING
# =====================================================================

def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad variable-length sequences into a rectangular NumPy array.

    Returns:
        padded:
            Integer matrix with shape [batch_size, max_length].

        mask:
            Binary matrix:
                1 = real token
                0 = padding

    The mask is useful when implementing attention-based models.
    """

    if not sequences:
        raise ValueError("At least one sequence is required.")

    max_length = max(len(sequence) for sequence in sequences)

    batch_size = len(sequences)

    padded = np.full(
        (batch_size, max_length),
        fill_value=pad_id,
        dtype=np.int64,
    )

    mask = np.zeros(
        (batch_size, max_length),
        dtype=np.int64,
    )

    for index, sequence in enumerate(sequences):

        length = len(sequence)

        if length == 0:
            continue

        padded[index, :length] = sequence
        mask[index, :length] = 1

    return padded, mask


# =====================================================================
# EXPERIMENT 1 — TEXT NORMALIZATION
# =====================================================================

def experiment_1_normalization(text: str) -> str:

    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — TEXT NORMALIZATION")
    print("=" * 70)

    examples = [
        "I have 2 apples.",
        "The price is 50%.",
        "Dr. Smith has 3 dogs.",
        "There are 125 books.",
    ]

    for example in examples:

        normalized = normalize_text(example)

        print(f"\nOriginal:   {example}")
        print(f"Normalized: {normalized}")

    print("\nCustom input:")
    print("Original:  ", text)
    print("Normalized:", normalize_text(text))

    return normalize_text(text)


# =====================================================================
# EXPERIMENT 2 — CHARACTER TOKENIZATION
# =====================================================================

def experiment_2_character_tokenization(text: str):

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — CHARACTER TOKENIZATION")
    print("=" * 70)

    normalized = normalize_text(text)

    vocabulary = build_character_vocabulary(
        [normalized]
    )

    token_ids = encode_characters(
        normalized,
        vocabulary,
    )

    reconstructed = decode_characters(
        token_ids,
        vocabulary,
    )

    print("\nNormalized text:")
    print(normalized)

    print("\nCharacter vocabulary:")
    for token, token_id in vocabulary.items():
        print(f"{repr(token):12s} -> {token_id}")

    print("\nCharacter sequence:")
    print(list(normalized.lower()))

    print("\nToken IDs:")
    print(token_ids)

    print("\nDecoded text:")
    print(reconstructed)

    print("\nSequence length:", len(token_ids))
    print("Vocabulary size:", len(vocabulary))

    return token_ids, vocabulary


# =====================================================================
# EXPERIMENT 3 — G2P
# =====================================================================

def experiment_3_g2p(text: str):

    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — GRAPHEME-TO-PHONEME")
    print("=" * 70)

    normalized = normalize_text(text)

    result = toy_g2p(
        normalized,
        PRONUNCIATION_DICTIONARY,
    )

    print("\nInput:")
    print(normalized)

    print("\nWord tokens:")
    print(result.words)

    print("\nPhoneme tokens:")
    print(result.phoneme_tokens)

    print("\nPhoneme sequence:")
    print(" ".join(result.phoneme_tokens))

    if result.unknown_words:

        print("\nUnknown words:")
        print(result.unknown_words)

        print(
            "\nThese words are missing from the educational "
            "pronunciation dictionary."
        )

    else:
        print("\nAll words were found in the dictionary.")

    return result


# =====================================================================
# EXPERIMENT 4 — PHONEME TOKENIZATION AND PADDING
# =====================================================================

def experiment_4_phoneme_tokenization(text: str):

    print("\n" + "=" * 70)
    print("EXPERIMENT 4 — PHONEME TOKENIZATION AND PADDING")
    print("=" * 70)

    examples = [
        "cat",
        "dog",
        "hello",
        "computer",
    ]

    sequences = []

    for example in examples:

        result = toy_g2p(
            example,
            PRONUNCIATION_DICTIONARY,
        )

        sequences.append(result.phoneme_tokens)

    vocabulary = build_phoneme_vocabulary(
        sequences
    )

    encoded_sequences = [
        encode_phonemes(sequence, vocabulary)
        for sequence in sequences
    ]

    print("\nPhoneme vocabulary:")

    for token, token_id in vocabulary.items():
        print(f"{token:12s} -> {token_id}")

    print("\nEncoded sequences:")

    for example, phonemes, encoded in zip(
        examples,
        sequences,
        encoded_sequences,
    ):

        print(f"\nText:     {example}")
        print(f"Phonemes: {phonemes}")
        print(f"IDs:      {encoded}")

    padded, mask = pad_sequences(
        encoded_sequences,
        vocabulary[PAD_TOKEN],
    )

    print("\nPadded batch:")
    print(padded)

    print("\nPadding mask:")
    print(mask)

    print("\nBatch shape:")
    print(padded.shape)

    print("\nInterpretation:")
    print("Rows    = individual text samples")
    print("Columns = token positions")
    print("Mask 1  = valid token")
    print("Mask 0  = padding")

    return padded, mask


# =====================================================================
# VISUALIZATION 1 — CHARACTER TOKEN IDs
# =====================================================================

def plot_character_encoding(text: str):

    normalized = normalize_text(text)

    vocabulary = build_character_vocabulary(
        [normalized]
    )

    token_ids = encode_characters(
        normalized,
        vocabulary,
    )

    characters = list(normalized.lower())

    if not characters:
        print("No characters available for visualization.")
        return

    figure_width = max(10, len(characters) * 0.45)

    fig, ax = plt.subplots(
        figsize=(figure_width, 4)
    )

    x = np.arange(len(characters))

    ax.bar(
        x,
        token_ids,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        characters,
        rotation=45,
        ha="right",
    )

    ax.set_xlabel("Character position")
    ax.set_ylabel("Token ID")
    ax.set_title("Character-Level Text Encoding")

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()


# =====================================================================
# VISUALIZATION 2 — PHONEME SEQUENCE
# =====================================================================

def plot_phoneme_sequence(text: str):

    normalized = normalize_text(text)

    result = toy_g2p(
        normalized,
        PRONUNCIATION_DICTIONARY,
    )

    tokens = result.phoneme_tokens

    if not tokens:
        print("No phonemes available for visualization.")
        return

    vocabulary = build_phoneme_vocabulary(
        [tokens]
    )

    token_ids = encode_phonemes(
        tokens,
        vocabulary,
    )

    fig, ax = plt.subplots(
        figsize=(max(9, len(tokens) * 0.65), 4)
    )

    x = np.arange(len(tokens))

    ax.bar(
        x,
        token_ids,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        tokens,
        rotation=45,
        ha="right",
    )

    ax.set_xlabel("Phoneme position")
    ax.set_ylabel("Token ID")
    ax.set_title("Phoneme-Level Text Encoding")

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()


# =====================================================================
# VISUALIZATION 3 — CHARACTER VS PHONEME SEQUENCE LENGTH
# =====================================================================

def compare_sequence_lengths():

    examples = [
        "cat",
        "dog",
        "hello",
        "computer",
        "though",
        "through",
        "tough",
        "thought",
    ]

    character_lengths = []
    phoneme_lengths = []

    for word in examples:

        character_lengths.append(
            len(word)
        )

        result = toy_g2p(
            word,
            PRONUNCIATION_DICTIONARY,
        )

        phoneme_lengths.append(
            len(result.phoneme_tokens)
        )

    x = np.arange(len(examples))
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        x - width / 2,
        character_lengths,
        width,
        label="Characters",
    )

    ax.bar(
        x + width / 2,
        phoneme_lengths,
        width,
        label="Phonemes",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(examples)

    ax.set_ylabel("Sequence length")
    ax.set_title("Character Sequence vs Phoneme Sequence")

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()


# =====================================================================
# MAIN PROGRAM
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 Lesson 7: TTS Text Frontend"
        )
    )

    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_TEXT,
        help="Input text to process.",
    )

    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=0,
        help=(
            "Select experiment. "
            "0 runs all experiments."
        ),
    )

    args = parser.parse_args()

    text = args.text

    print("\n" + "=" * 70)
    print("BUILD YOUR OWN VOICE-CLONING TTS SYSTEM")
    print("PHASE 1 — LESSON 7")
    print("=" * 70)

    print("\nOriginal input:")
    print(text)

    if args.experiment in (0, 1):
        experiment_1_normalization(text)

    if args.experiment in (0, 2):
        experiment_2_character_tokenization(text)

    if args.experiment in (0, 3):
        experiment_3_g2p(text)

    if args.experiment in (0, 4):
        experiment_4_phoneme_tokenization(text)

    if args.experiment in (0, 5):

        print("\n" + "=" * 70)
        print("EXPERIMENT 5 — VISUAL COMPARISON")
        print("=" * 70)

        plot_character_encoding(text)

        plot_phoneme_sequence(text)

        compare_sequence_lengths()

    if args.experiment == 0:

        print("\n" + "=" * 70)
        print("ALL EXPERIMENTS COMPLETED")
        print("=" * 70)

        print(
            "\nYou have implemented a basic TTS text frontend."
        )

        print(
            "\nNext, we will study how speech datasets are "
            "represented, aligned and prepared for neural TTS."
        )


if __name__ == "__main__":
    main()