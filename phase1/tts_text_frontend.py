"""
===============================================================================
PHASE 1 — LESSON 11
MULTILINGUAL TEXT FRONTEND FOR A TEXT-TO-SPEECH SYSTEM
===============================================================================

SUPPORTED LANGUAGES
-------------------
1. English       (en)
2. Chinese       (zh) - Simplified Chinese examples
3. Japanese      (ja)
4. Russian       (ru)

PURPOSE
-------
Build a multilingual text frontend that converts written text into numerical
representations suitable for a neural TTS model.

The frontend performs:

    Raw text
       |
       v
    Unicode normalization
       |
       v
    Language-specific text normalization
       |
       v
    Language-aware grapheme tokenization
       |
       +--------------------------+
       |                          |
       v                          v
    Grapheme tokens           Subword tokens
       |                          |
       v                          v
    Token IDs                 Token IDs
       |                          |
       +------------+-------------+
                    |
                    v
             Model-ready input

An additional English educational G2P demonstration converts selected English
words into IPA-like phoneme tokens.

IMPORTANT
---------
This is an educational implementation.

It does NOT implement complete production-quality pronunciation conversion
for all four languages.

The language-specific capabilities are:

English:
    - Unicode normalization
    - Basic number and currency normalization
    - Abbreviation expansion
    - Grapheme tokenization
    - Small dictionary-based IPA demonstration

Chinese:
    - Unicode normalization
    - Simplified Chinese character tokenization
    - Chinese punctuation handling
    - Toy subword tokenization
    - No Pinyin or tone prediction

Japanese:
    - Unicode normalization
    - Hiragana/Katakana/Kanji grapheme tokenization
    - Japanese punctuation handling
    - Toy subword tokenization
    - No morphological analysis or reading conversion

Russian:
    - Unicode normalization
    - Cyrillic grapheme tokenization
    - Russian punctuation handling
    - Toy subword tokenization
    - No Russian G2P model

No pretrained TTS model is used.
No external dataset is required.

REQUIREMENTS
------------
Python 3.9+

Core implementation:
    Python standard library only.

Optional visualization:
    pip install matplotlib

RUN
---
English:
    python phase1_lesson11_multilingual_tts_frontend.py --language en

Chinese:
    python phase1_lesson11_multilingual_tts_frontend.py --language zh

Japanese:
    python phase1_lesson11_multilingual_tts_frontend.py --language ja

Russian:
    python phase1_lesson11_multilingual_tts_frontend.py --language ru

Grapheme mode:
    python phase1_lesson11_multilingual_tts_frontend.py --language en --mode grapheme

Subword mode:
    python phase1_lesson11_multilingual_tts_frontend.py --language zh --mode subword

English phoneme mode:
    python phase1_lesson11_multilingual_tts_frontend.py --language en --mode phoneme

Run experiments:
    python phase1_lesson11_multilingual_tts_frontend.py --experiments

Show visualization:
    python phase1_lesson11_multilingual_tts_frontend.py --plot

===============================================================================
THEORY 1 — GRAPHEME, PHONEME, PHONE, AND SYLLABLE
===============================================================================

A grapheme is a written unit in a writing system.

Examples:

English:
    c, a, t

Chinese:
    你, 好, 世, 界

Japanese:
    あ, か, さ, ア, カ, サ, 日, 本

Russian:
    п, р, и, в, е, т

A phoneme is an abstract sound unit that can distinguish meaning in a language.

Examples in English:
    /p/ versus /b/

    "pat" and "bat" have different meanings because /p/ and /b/ contrast.

A phone is an actual speech sound produced by a speaker.

Phonemes are conventionally written between slashes:
    /t/

Phones are conventionally written between square brackets:
    [t], [tʰ], [ɾ]

An allophone is a context-dependent realization of a phoneme.

A syllable is a rhythmic unit of speech, often organized around a vowel or
syllabic nucleus.

A word may contain:
    - one or more graphemes
    - one or more phonemes
    - one or more syllables

These units are not interchangeable.

===============================================================================
THEORY 2 — WHY WRITTEN TEXT IS NOT PRONUNCIATION
===============================================================================

Different languages have different relationships between spelling and sound.

ENGLISH
-------
English spelling is historically complex.

Examples:
    though
    through
    tough
    thought

Similar letter sequences can have different pronunciations.

CHINESE
-------
Chinese characters generally represent morphemes and syllables.

Mandarin pronunciation is commonly represented using Pinyin.

Example:
    你好

    Pinyin:
        nǐ hǎo

The tone marks are important.

Mandarin has lexical tones. Changing the tone can change the meaning of a
syllable.

A Chinese TTS frontend must handle:
    - character-to-Pinyin conversion
    - polyphonic characters
    - contextual pronunciation
    - tone information
    - word segmentation

JAPANESE
--------
Japanese writing combines:
    - Hiragana
    - Katakana
    - Kanji

Kanji may have multiple readings depending on context.

Example:
    今日

Its common reading is:
    きょう

A character-by-character conversion is not sufficient.

A Japanese TTS frontend needs:
    - morphological analysis
    - reading generation
    - pronunciation conversion
    - pitch-accent or prosody information where appropriate

RUSSIAN
-------
Russian uses the Cyrillic alphabet.

Written Russian has a more regular spelling-to-sound relationship than
English, but pronunciation is still context-dependent.

Examples:
    stress affects vowel realization
    unstressed vowels may be reduced
    consonants may be palatalized
    some consonants undergo voicing or devoicing changes

A Russian TTS frontend should handle:
    - stress placement
    - vowel reduction
    - consonant palatalization
    - contextual pronunciation

Therefore:

    Text characters != phonemes

A production multilingual TTS system needs language-aware pronunciation
processing.

===============================================================================
THEORY 3 — INTERNATIONAL PHONETIC ALPHABET (IPA)
===============================================================================

The International Phonetic Alphabet represents speech sounds.

Examples:
    /iː/  as in "see" in many English accents
    /æ/   as in "cat"
    /ʃ/   as in "ship"
    /θ/   as in "think"

IPA can represent sounds across languages.

However, a pronunciation convention must be defined consistently.

The pronunciation of a word may vary by:
    - language
    - dialect
    - accent
    - context
    - speaking style

For multilingual TTS, phoneme tokens may be shared across languages, but
language-specific pronunciation rules are still necessary.

===============================================================================
THEORY 4 — UNICODE AND MULTILINGUAL TEXT
===============================================================================

A computer stores text as Unicode code points.

Examples:

English:
    A

Chinese:
    你

Japanese:
    あ

Russian:
    Я

Some visible characters can be represented by multiple Unicode code points.

Unicode normalization helps standardize equivalent text representations.

NFC normalization:
    unicodedata.normalize("NFC", text)

However, Unicode normalization does not:
    - convert text into pronunciation
    - resolve ambiguous kanji readings
    - generate Chinese Pinyin
    - determine Russian stress

Unicode normalization is only one stage of the text frontend.

===============================================================================
THEORY 5 — TEXT NORMALIZATION
===============================================================================

Text normalization converts written forms into forms suitable for speech.

Examples:

English:
    "I have 2 apples."
        ->
    "I have two apples."

    "$12.50"
        ->
    "twelve dollars and fifty cents"

Chinese:
    Number and date readings depend on context and usage.

Japanese:
    Kanji readings depend on word context.

Russian:
    Number inflection depends on grammatical context.

A generic number-to-word converter cannot correctly normalize all four
languages.

This lesson implements basic English number normalization and Unicode
normalization for all languages.

===============================================================================
THEORY 6 — TOKENIZATION
===============================================================================

A neural model usually processes token IDs, not raw strings.

Three common approaches:

1. Grapheme tokenization

    "cat" -> ["c", "a", "t"]

    "你好" -> ["你", "好"]

    "こんにちは" -> individual written units

    "привет" -> ["п", "р", "и", "в", "е", "т"]

2. Phoneme tokenization

    "cat" -> ["k", "æ", "t"]

    Chinese:
        characters -> Pinyin syllables with tone information

    Japanese:
        text -> pronunciation units

    Russian:
        text -> phoneme sequence with stress and contextual rules

3. Subword tokenization

    A word or text sequence is split into frequently occurring pieces.

    Examples:
        speech -> ["spe", "ech"]
        привет -> ["при", "вет"]

The actual subword pieces depend on the vocabulary.

The toy tokenizer in this lesson is not BPE or SentencePiece.

===============================================================================
THEORY 7 — TOKEN IDS AND EMBEDDINGS
===============================================================================

A neural network cannot directly consume a string such as "hello".

We create a vocabulary:

    "<pad>" -> 0
    "<unk>" -> 1
    "<bos>" -> 2
    "<eos>" -> 3
    "h"     -> 4
    "e"     -> 5

Then:

    "hello" -> [4, 5, ...]

An embedding layer maps each token ID to a learned vector:

    E: token_id -> R^d

For a sequence of N tokens:

    Token IDs:
        [t1, t2, ..., tN]

    Embeddings:
        [e1, e2, ..., eN]

The embedding vectors are learned during neural model training.

This lesson creates token IDs but does not train embeddings.

===============================================================================
THEORY 8 — MULTILINGUAL TTS ARCHITECTURE
===============================================================================

A multilingual voice-cloning TTS model may receive:

    Text
      |
      v
    Language identification
      |
      v
    Language-specific normalization
      |
      v
    Language-specific pronunciation frontend
      |
      v
    Token IDs
      |
      +-----------------------+
      |                       |
      v                       v
    Text encoder          Speaker encoder
      |                       |
      +-----------+-----------+
                  |
                  v
          Speech generator
                  |
                  v
           Neural vocoder
                  |
                  v
           Generated speech

The language frontend determines what should be spoken.

The speaker representation controls the voice identity.

A multilingual model must learn both language-specific pronunciation and
speaker characteristics.

===============================================================================
"""

import argparse
import json
import re
import unicodedata

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

SUPPORTED_LANGUAGES = ["en", "zh", "ja", "ru"]

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>"
]


# =============================================================================
# LANGUAGE EXAMPLES
# =============================================================================

DEMO_TEXTS = {

    "en": [
        "Hello world!",
        "The robot can generate speech.",
        "I have 2 apples.",
        "The price is $12.50.",
        "Dr. Smith works on 2026-11-04.",
        "Though the task is tough, we can think it through."
    ],

    "zh": [
        "你好，世界！",
        "人工智能正在改变语音合成技术。",
        "我喜欢学习机器学习。",
        "机器人可以说话。"
    ],

    "ja": [
        "こんにちは。",
        "音声合成を勉強します。",
        "人工知能は音声技術を変えています。",
        "東京へ行きます。"
    ],

    "ru": [
        "Привет, мир!",
        "Робот может генерировать речь.",
        "Я изучаю искусственный интеллект.",
        "Синтез речи использует нейронные сети."
    ]
}


# =============================================================================
# THEORY IMPLEMENTATION 1 — UNICODE NORMALIZATION
# =============================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize text using Unicode NFC.

    NFC standardizes canonically equivalent Unicode sequences.

    This is important for multilingual text processing.

    It does not perform pronunciation conversion.
    """

    return unicodedata.normalize("NFC", text)


# =============================================================================
# THEORY IMPLEMENTATION 2 — ENGLISH NUMBER NORMALIZATION
# =============================================================================

ONES = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
]

TEENS = [
    "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
]

TENS = [
    "", "", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety"
]


def integer_to_english_words(number: int) -> str:
    """
    Convert integers from 0 to 999,999 into English words.

    Educational implementation only.
    """

    if number < 0:
        return "minus " + integer_to_english_words(abs(number))

    if number < 10:
        return ONES[number]

    if number < 20:
        return TEENS[number - 10]

    if number < 100:

        tens = TENS[number // 10]
        remainder = number % 10

        if remainder == 0:
            return tens

        return tens + " " + ONES[remainder]

    if number < 1000:

        hundreds = number // 100
        remainder = number % 100

        result = ONES[hundreds] + " hundred"

        if remainder:
            result += " " + integer_to_english_words(remainder)

        return result

    if number < 1_000_000:

        thousands = number // 1000
        remainder = number % 1000

        result = integer_to_english_words(thousands) + " thousand"

        if remainder:
            result += " " + integer_to_english_words(remainder)

        return result

    raise ValueError(
        "Supported range is below 1,000,000."
    )


def normalize_english_numbers(text: str) -> str:
    """
    Demonstration English text normalization.

    Supports:
        - Basic integers
        - USD amounts
        - A few abbreviations
        - ISO-style dates

    This is not a production-grade normalizer.
    """

    # ISO date: YYYY-MM-DD
    def replace_date(match):

        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))

        return (
            integer_to_english_words(month)
            + " "
            + integer_to_english_words(day)
            + ", "
            + integer_to_english_words(year)
        )

    text = re.sub(
        r"\b(\d{4})-(\d{2})-(\d{2})\b",
        replace_date,
        text
    )

    # Currency: $12.50
    def replace_currency(match):

        raw = match.group(1).replace(",", "")

        amount = float(raw)

        dollars = int(amount)

        cents = round((amount - dollars) * 100)

        if cents == 100:
            dollars += 1
            cents = 0

        if dollars == 1:
            dollar_text = "one dollar"
        else:
            dollar_text = (
                integer_to_english_words(dollars)
                + " dollars"
            )

        if cents == 0:
            return dollar_text

        if cents == 1:
            cent_text = "one cent"
        else:
            cent_text = (
                integer_to_english_words(cents)
                + " cents"
            )

        return dollar_text + " and " + cent_text

    text = re.sub(
        r"\$(\d[\d,]*(?:\.\d{1,2})?)",
        replace_currency,
        text
    )

    # Abbreviations.
    abbreviations = {
        r"\bDr\.": "Doctor",
        r"\bMr\.": "Mister",
        r"\bMrs\.": "Misses",
        r"\bSt\.": "Street",
        r"\bProf\.": "Professor"
    }

    for pattern, replacement in abbreviations.items():

        text = re.sub(
            pattern,
            replacement,
            text
        )

    # Remaining simple integers.
    def replace_integer(match):

        value = int(match.group(0))

        try:
            return integer_to_english_words(value)

        except ValueError:
            return match.group(0)

    text = re.sub(
        r"(?<![\w.])\d{1,6}(?![\w.])",
        replace_integer,
        text
    )

    return text


# =============================================================================
# THEORY IMPLEMENTATION 3 — LANGUAGE-AWARE NORMALIZATION
# =============================================================================

def normalize_text(
    text: str,
    language: str
) -> str:
    """
    Main multilingual normalization interface.

    All languages:
        Unicode NFC
        Whitespace normalization

    English:
        Basic number, currency, date, abbreviation expansion

    Chinese:
        Unicode normalization only

    Japanese:
        Unicode normalization only

    Russian:
        Unicode normalization only

    Production normalization for Chinese, Japanese, and Russian must be
    expanded with their own language-specific rules.
    """

    if language not in SUPPORTED_LANGUAGES:

        raise ValueError(
            f"Unsupported language: {language}"
        )

    text = normalize_unicode(text)

    if language == "en":

        text = normalize_english_numbers(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =============================================================================
# THEORY IMPLEMENTATION 4 — GRAPHEME EXTRACTION
# =============================================================================

def split_grapheme_like_units(
    text: str
) -> List[str]:
    """
    Split text into approximate grapheme units.

    Python strings iterate over Unicode code points, which are not always
    identical to user-perceived characters.

    This function attaches combining marks to the preceding code point.

    It is an educational approximation, not a complete Unicode grapheme
    cluster algorithm.

    A production system should use Unicode extended grapheme segmentation.
    """

    units = []

    for char in text:

        if unicodedata.combining(char) and units:

            units[-1] += char

        else:

            units.append(char)

    return units


# =============================================================================
# THEORY IMPLEMENTATION 5 — LANGUAGE-AWARE GRAPHEME TOKENIZATION
# =============================================================================

def tokenize_english(text: str) -> List[str]:
    """
    English grapheme tokenization.

    Spaces are represented by <space>.
    Punctuation remains a token.
    """

    return tokenize_graphemes(text)


def tokenize_chinese(text: str) -> List[str]:
    """
    Chinese character-level tokenization.

    Chinese does not normally use spaces between every word.

    This educational implementation therefore uses character-level tokens.

    Example:
        你好世界

        -> ["你", "好", "世", "界"]

    IMPORTANT:
        Chinese character tokenization is not Chinese word segmentation.

    Production systems may use:
        - word segmentation
        - character-level tokenization
        - Pinyin tokenization
        - subword tokenization
    """

    return tokenize_graphemes(text)


def tokenize_japanese(text: str) -> List[str]:
    """
    Japanese grapheme-level tokenization.

    Preserves:
        - Hiragana
        - Katakana
        - Kanji
        - punctuation

    This does not perform Japanese morphological analysis.

    Example:
        こんにちは

        -> individual written units

    Production Japanese TTS generally needs morphological analysis and reading
    generation.
    """

    return tokenize_graphemes(text)


def tokenize_russian(text: str) -> List[str]:
    """
    Russian Cyrillic grapheme tokenization.

    Example:
        привет

        -> ["п", "р", "и", "в", "е", "т"]

    This does not determine Russian stress or phoneme-level pronunciation.
    """

    return tokenize_graphemes(text)


def tokenize_graphemes(text: str) -> List[str]:
    """
    Generic multilingual grapheme-like tokenizer.

    Spaces become <space>.
    Other characters are preserved.
    """

    result = []

    for unit in split_grapheme_like_units(text):

        if unit.isspace():

            result.append("<space>")

        else:

            result.append(unit)

    return result


def language_grapheme_tokenize(
    text: str,
    language: str
) -> List[str]:

    functions = {
        "en": tokenize_english,
        "zh": tokenize_chinese,
        "ja": tokenize_japanese,
        "ru": tokenize_russian
    }

    if language not in functions:

        raise ValueError(
            f"Unsupported language: {language}"
        )

    return functions[language](text)


# =============================================================================
# THEORY IMPLEMENTATION 6 — EDUCATIONAL ENGLISH G2P
# =============================================================================

ENGLISH_PRONUNCIATION_DICTIONARY = {

    "cat": ["k", "æ", "t"],

    "dog": ["d", "ɔ", "ɡ"],

    "hello": ["h", "ə", "ˈl", "oʊ"],

    "world": ["w", "ɝ", "l", "d"],

    "speech": ["s", "p", "iː", "tʃ"],

    "voice": ["v", "ɔɪ", "s"],

    "text": ["t", "ɛ", "k", "s", "t"],

    "robot": ["ˈr", "oʊ", "b", "ɑ", "t"],

    "through": ["θ", "r", "uː"],

    "though": ["ð", "oʊ"],

    "tough": ["t", "ʌ", "f"],

    "thought": ["θ", "ɔː", "t"]
}


def tokenize_words_and_punctuation(
    text: str
) -> List[str]:

    return re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE
    )


def simple_english_g2p(
    text: str
) -> List[str]:
    """
    Small dictionary-based English pronunciation demonstration.

    Known words:
        Return the educational IPA-like sequence.

    Unknown words:
        Use grapheme fallback.

    The fallback is not pronunciation.

    It is only a mechanism to preserve unknown words in the token sequence.
    """

    tokens = tokenize_words_and_punctuation(text)

    phonemes = []

    for token in tokens:

        lower = token.lower()

        if lower in ENGLISH_PRONUNCIATION_DICTIONARY:

            phonemes.extend(
                ENGLISH_PRONUNCIATION_DICTIONARY[lower]
            )

        elif re.fullmatch(r"\w+", token, flags=re.UNICODE):

            phonemes.append("<grapheme_fallback>")

            phonemes.extend(
                list(lower)
            )

        else:

            phonemes.append(token)

    return phonemes


# =============================================================================
# THEORY IMPLEMENTATION 7 — TOY SUBWORD TOKENIZATION
# =============================================================================

def train_toy_subword_vocabulary(
    texts: Sequence[str],
    min_frequency: int = 2,
    max_piece_length: int = 4
) -> List[str]:
    """
    Build a frequency-based toy subword vocabulary.

    This is NOT BPE or SentencePiece.

    It collects character n-grams from the training texts.

    It can operate on:
        - Latin text
        - Chinese text
        - Japanese text
        - Cyrillic text

    However, the resulting units are not necessarily linguistic words,
    morphemes, syllables, or phonemes.
    """

    counts = Counter()

    for text in texts:

        units = split_grapheme_like_units(
            text.lower()
        )

        # Do not create n-grams across spaces.
        words = []

        current = []

        for unit in units:

            if unit.isspace():

                if current:
                    words.append(current)
                    current = []

            else:

                current.append(unit)

        if current:
            words.append(current)

        for word in words:

            for length in range(
                1,
                min(max_piece_length, len(word)) + 1
            ):

                for start in range(
                    0,
                    len(word) - length + 1
                ):

                    piece = "".join(
                        word[start:start + length]
                    )

                    counts[piece] += 1

    pieces = [
        piece
        for piece, count in counts.items()
        if count >= min_frequency
    ]

    pieces.sort(
        key=lambda piece: (
            -len(piece),
            -counts[piece],
            piece
        )
    )

    return pieces


def greedy_subword_tokenize(
    text: str,
    vocabulary: Sequence[str]
) -> List[str]:
    """
    Greedy longest-match subword tokenizer.

    This is a simple educational algorithm.

    At each position:
        1. Find the longest vocabulary piece matching the text.
        2. If no piece matches, emit one grapheme unit.
        3. Continue until the text is consumed.

    Spaces are represented by <space>.

    This is not the BPE training algorithm.
    """

    vocab = set(vocabulary)

    units = split_grapheme_like_units(
        text.lower()
    )

    result = []

    position = 0

    while position < len(units):

        if units[position].isspace():

            result.append("<space>")

            position += 1

            continue

        match = None

        max_length = len(units) - position

        for length in range(
            max_length,
            0,
            -1
        ):

            candidate = "".join(
                units[position:position + length]
            )

            if candidate in vocab:

                match = candidate

                break

        if match is None:

            match = units[position]

        result.append(match)

        position += len(
            split_grapheme_like_units(match)
        )

    return result


# =============================================================================
# THEORY IMPLEMENTATION 8 — VOCABULARY AND TOKEN IDS
# =============================================================================

@dataclass
class Vocabulary:
    """
    Converts text tokens into integer IDs.

    Special tokens:

        <pad> = padding
        <unk> = unknown token
        <bos> = beginning of sequence
        <eos> = end of sequence

    A separate vocabulary is created for each language and representation
    configuration in this lesson.
    """

    token_to_id: Dict[str, int]

    @classmethod
    def build(
        cls,
        token_sequences: Sequence[Sequence[str]]
    ):

        tokens = set(SPECIAL_TOKENS)

        for sequence in token_sequences:

            tokens.update(sequence)

        ordered = list(SPECIAL_TOKENS)

        remaining = sorted(
            tokens - set(SPECIAL_TOKENS)
        )

        ordered.extend(remaining)

        mapping = {
            token: index
            for index, token in enumerate(ordered)
        }

        return cls(mapping)

    @property
    def id_to_token(self):

        return {
            index: token
            for token, index in self.token_to_id.items()
        }

    def encode(
        self,
        tokens: Sequence[str],
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[int]:

        unknown_id = self.token_to_id["<unk>"]

        ids = [
            self.token_to_id.get(
                token,
                unknown_id
            )
            for token in tokens
        ]

        if add_bos:

            ids.insert(
                0,
                self.token_to_id["<bos>"]
            )

        if add_eos:

            ids.append(
                self.token_to_id["<eos>"]
            )

        return ids

    def decode(
        self,
        ids: Sequence[int]
    ) -> List[str]:

        reverse = self.id_to_token

        return [
            reverse.get(
                index,
                "<unk>"
            )
            for index in ids
        ]

    def save(
        self,
        path: Path
    ):

        path.write_text(
            json.dumps(
                self.token_to_id,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: Path
    ):

        mapping = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return cls(mapping)


# =============================================================================
# THEORY IMPLEMENTATION 9 — BATCH PADDING
# =============================================================================

def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int = 0
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Pad sequences to equal length.

    Returns:

        padded_sequences
        attention_masks

    Attention mask:

        1 = real token
        0 = padding

    Example:

        Sequence A: [2, 5, 6]
        Sequence B: [2, 7]

        Padded:

            [2, 5, 6]
            [2, 7, 0]

        Masks:

            [1, 1, 1]
            [1, 1, 0]
    """

    if not sequences:

        return [], []

    max_length = max(
        len(sequence)
        for sequence in sequences
    )

    padded = []

    masks = []

    for sequence in sequences:

        padding_length = (
            max_length - len(sequence)
        )

        padded.append(
            list(sequence)
            + [pad_id] * padding_length
        )

        masks.append(
            [1] * len(sequence)
            + [0] * padding_length
        )

    return padded, masks


# =============================================================================
# THEORY IMPLEMENTATION 10 — LANGUAGE-SPECIFIC EXAMPLES
# =============================================================================

def print_language_examples():

    print("\n" + "=" * 85)

    print("MULTILINGUAL TEXT REPRESENTATION")

    print("=" * 85)

    examples = {

        "English": "Hello world",

        "Chinese": "你好世界",

        "Japanese": "こんにちは世界",

        "Russian": "Привет мир"
    }

    language_codes = {
        "English": "en",
        "Chinese": "zh",
        "Japanese": "ja",
        "Russian": "ru"
    }

    for name, text in examples.items():

        language = language_codes[name]

        tokens = language_grapheme_tokenize(
            text,
            language
        )

        print(f"\n{name}")

        print("Text:  ", text)

        print("Tokens:", tokens)


# =============================================================================
# THEORY IMPLEMENTATION 11 — FRONTEND PIPELINE
# =============================================================================

def build_representation(
    text: str,
    language: str,
    mode: str,
    subword_vocabulary: Sequence[str]
) -> List[str]:
    """
    Build the selected representation.

    Modes:

        grapheme:
            Supported for all four languages.

        subword:
            Supported for all four languages.

        phoneme:
            Educational English dictionary only.

    Chinese, Japanese, and Russian phoneme processing will be added in
    subsequent pronunciation-model lessons.
    """

    if mode == "grapheme":

        return language_grapheme_tokenize(
            text,
            language
        )

    if mode == "subword":

        return greedy_subword_tokenize(
            text,
            subword_vocabulary
        )

    if mode == "phoneme":

        if language == "en":

            return simple_english_g2p(text)

        raise NotImplementedError(
            f"Phoneme conversion for language '{language}' "
            "is not implemented in this lesson. "
            "Use grapheme or subword mode."
        )

    raise ValueError(
        "Supported modes: grapheme, subword, phoneme"
    )


# =============================================================================
# OPTIONAL VISUALIZATION
# =============================================================================

def plot_token_ids(
    tokens: Sequence[str],
    ids: Sequence[int]
):

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "Matplotlib is not installed. "
            "Run: pip install matplotlib"
        )

        return

    positions = list(
        range(len(ids))
    )

    plt.figure(
        figsize=(max(10, len(ids) * 0.45), 5)
    )

    plt.bar(
        positions,
        ids
    )

    plt.xticks(
        positions,
        tokens,
        rotation=60,
        ha="right"
    )

    plt.xlabel("Token position")

    plt.ylabel("Token ID")

    plt.title(
        "Multilingual TTS Frontend — Token IDs"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_demo(
    language: str,
    mode: str,
    show_plot: bool
):

    print("\n" + "=" * 85)

    print("MULTILINGUAL TTS TEXT FRONTEND — LESSON 11")

    print("=" * 85)

    print(f"Language: {language}")

    print(f"Mode:     {mode}")

    texts = DEMO_TEXTS[language]

    # -------------------------------------------------------------------------
    # STEP 1 — INPUT TEXT
    # -------------------------------------------------------------------------

    print("\n1. INPUT TEXT")

    print("-" * 85)

    for index, text in enumerate(
        texts,
        start=1
    ):

        print(
            f"{index}. {text}"
        )

    # -------------------------------------------------------------------------
    # STEP 2 — NORMALIZATION
    # -------------------------------------------------------------------------

    print("\n2. TEXT NORMALIZATION")

    print("-" * 85)

    normalized_texts = []

    for text in texts:

        normalized = normalize_text(
            text,
            language
        )

        normalized_texts.append(
            normalized
        )

        print("Original:   ", text)

        print("Normalized: ", normalized)

        print()

    # -------------------------------------------------------------------------
    # STEP 3 — SUBWORD VOCABULARY
    # -------------------------------------------------------------------------

    print("\n3. SUBWORD VOCABULARY")

    print("-" * 85)

    subword_vocabulary = train_toy_subword_vocabulary(
        normalized_texts,
        min_frequency=2,
        max_piece_length=4
    )

    print(
        "Number of subword pieces:",
        len(subword_vocabulary)
    )

    print(
        "First 60 pieces:",
        subword_vocabulary[:60]
    )

    # -------------------------------------------------------------------------
    # STEP 4 — TOKEN REPRESENTATION
    # -------------------------------------------------------------------------

    print("\n4. TOKEN REPRESENTATION")

    print("-" * 85)

    token_sequences = []

    for text in normalized_texts:

        tokens = build_representation(
            text=text,
            language=language,
            mode=mode,
            subword_vocabulary=subword_vocabulary
        )

        token_sequences.append(tokens)

        print("Text:  ", text)

        print("Tokens:", tokens)

        print()

    # -------------------------------------------------------------------------
    # STEP 5 — VOCABULARY
    # -------------------------------------------------------------------------

    print("\n5. VOCABULARY CONSTRUCTION")

    print("-" * 85)

    vocabulary = Vocabulary.build(
        token_sequences
    )

    print(
        "Vocabulary size:",
        len(vocabulary.token_to_id)
    )

    for token, token_id in list(
        vocabulary.token_to_id.items()
    )[:100]:

        print(
            f"{token!r:25s} -> {token_id}"
        )

    # -------------------------------------------------------------------------
    # STEP 6 — TOKEN IDS
    # -------------------------------------------------------------------------

    print("\n6. TEXT TO TOKEN IDS")

    print("-" * 85)

    encoded_sequences = []

    for tokens in token_sequences:

        ids = vocabulary.encode(tokens)

        encoded_sequences.append(ids)

        print("Tokens:", tokens)

        print("IDs:   ", ids)

        print()

    # -------------------------------------------------------------------------
    # STEP 7 — BATCH PADDING
    # -------------------------------------------------------------------------

    print("\n7. BATCH PADDING")

    print("-" * 85)

    padded, masks = pad_sequences(
        encoded_sequences,
        pad_id=vocabulary.token_to_id["<pad>"]
    )

    for index, (sequence, mask) in enumerate(
        zip(padded, masks)
    ):

        print(f"Sample {index + 1}")

        print("Input IDs:     ", sequence)

        print("Attention mask:", mask)

    # -------------------------------------------------------------------------
    # STEP 8 — DECODE
    # -------------------------------------------------------------------------

    print("\n8. DECODE CHECK")

    print("-" * 85)

    if encoded_sequences:

        ids = encoded_sequences[0]

        decoded = vocabulary.decode(ids)

        print("Encoded IDs:")

        print(ids)

        print("Decoded tokens:")

        print(decoded)

    # -------------------------------------------------------------------------
    # STEP 9 — SAVE VOCABULARY
    # -------------------------------------------------------------------------

    print("\n9. SAVE VOCABULARY")

    print("-" * 85)

    output_path = Path(
        f"lesson11_{language}_{mode}_vocabulary.json"
    )

    vocabulary.save(output_path)

    print(
        "Saved:",
        output_path.resolve()
    )

    # -------------------------------------------------------------------------
    # STEP 10 — VISUALIZATION
    # -------------------------------------------------------------------------

    if show_plot and token_sequences:

        print("\n10. TOKEN VISUALIZATION")

        plot_token_ids(
            token_sequences[0],
            vocabulary.encode(token_sequences[0])
        )

    print("\nDEMONSTRATION COMPLETE.")


# =============================================================================
# ADDITIONAL EXPERIMENTS
# =============================================================================

def run_experiments():

    print("\n" + "=" * 85)

    print("MULTILINGUAL EXPERIMENTS")

    print("=" * 85)

    # -------------------------------------------------------------------------
    # EXPERIMENT 1 — UNICODE
    # -------------------------------------------------------------------------

    print("\nEXPERIMENT 1 — UNICODE NORMALIZATION")

    examples = [
        "Hello",
        "你好",
        "こんにちは",
        "Привет"
    ]

    for text in examples:

        normalized = normalize_unicode(text)

        print("Original: ", text)

        print("Normalized:", normalized)

        print(
            "Code points:",
            [hex(ord(char)) for char in normalized]
        )

        print()

    # -------------------------------------------------------------------------
    # EXPERIMENT 2 — FOUR LANGUAGE TOKENIZATION
    # -------------------------------------------------------------------------

    print("\nEXPERIMENT 2 — FOUR LANGUAGE TOKENIZATION")

    language_examples = [
        ("en", "Hello world"),
        ("zh", "你好世界"),
        ("ja", "こんにちは世界"),
        ("ru", "Привет мир")
    ]

    for language, text in language_examples:

        tokens = language_grapheme_tokenize(
            text,
            language
        )

        print(f"\nLanguage: {language}")

        print("Text:  ", text)

        print("Tokens:", tokens)

    # -------------------------------------------------------------------------
    # EXPERIMENT 3 — ENGLISH NUMBER NORMALIZATION
    # -------------------------------------------------------------------------

    print("\nEXPERIMENT 3 — ENGLISH NUMBER NORMALIZATION")

    samples = [
        "I have 2 apples.",
        "The price is $12.50.",
        "The date is 2026-11-04.",
        "Dr. Smith has 3 robots."
    ]

    for sample in samples:

        print("Input: ", sample)

        print(
            "Output:",
            normalize_text(sample, "en")
        )

        print()

    # -------------------------------------------------------------------------
    # EXPERIMENT 4 — ENGLISH GRAPHEME VS PHONEME
    # -------------------------------------------------------------------------

    print("\nEXPERIMENT 4 — GRAPHEME VS PHONEME")

    word = "cat"

    graphemes = tokenize_graphemes(word)

    phonemes = simple_english_g2p(word)

    print("Word:      ", word)

    print("Graphemes: ", graphemes)

    print("Phonemes:  ", phonemes)

    # -------------------------------------------------------------------------
    # EXPERIMENT 5 — SUBWORD VOCABULARY
    # -------------------------------------------------------------------------

    print("\nEXPERIMENT 5 — MULTILINGUAL SUBWORD VOCABULARY")

    multilingual_texts = [
        "hello world",
        "你好世界",
        "こんにちは世界",
        "привет мир"
    ]

    subwords = train_toy_subword_vocabulary(
        multilingual_texts,
        min_frequency=1,
        max_piece_length=3
    )

    print("Subword pieces:")

    print(subwords[:100])

    # -------------------------------------------------------------------------
    # EXPERIMENT 6 — TOKEN ID
    # -------------------------------------------------------------------------

    print("\nEXPERIMENT 6 — TOKEN ID MAPPING")

    examples = [
        ["h", "e", "l", "l", "o"],
        ["你", "好"],
        ["こ", "ん", "に", "ち", "は"],
        ["п", "р", "и", "в", "е", "т"]
    ]

    vocabulary = Vocabulary.build(examples)

    for sequence in examples:

        print(
            sequence,
            "->",
            vocabulary.encode(sequence)
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Multilingual educational TTS text frontend"
        )
    )

    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default="en"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "grapheme",
            "subword",
            "phoneme"
        ],
        default="grapheme"
    )

    parser.add_argument(
        "--plot",
        action="store_true"
    )

    parser.add_argument(
        "--experiments",
        action="store_true"
    )

    args = parser.parse_args()

    print_language_examples()

    run_demo(
        language=args.language,
        mode=args.mode,
        show_plot=args.plot
    )

    if args.experiments:

        run_experiments()


if __name__ == "__main__":

    main()