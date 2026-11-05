"""
LESSON 51 — MODEL TESTING

Tests:
1. Load trained checkpoint.
2. Test English, Chinese, Japanese and Russian input.
3. Inspect phoneme sequences.
4. Check embedding dimensions and numerical stability.
5. Measure phoneme reconstruction accuracy.
6. Compare sentence embedding similarities.
"""

import torch
import torch.nn.functional as F
import numpy as np

from lesson51_multilingual_phoneme_encoder import (
    DEVICE,
    LANGUAGES,
    SAMPLE_TEXTS,
    MAX_SEQUENCE_LENGTH,
    OUTPUT_DIR,
    PhonemeVocabulary,
    MultilingualPhonemeEncoder,
    text_to_phonemes,
    encode_text,
)

# ---------------------------------------------------------
# 1. Load vocabulary
# ---------------------------------------------------------

vocab = PhonemeVocabulary()

vocab.build(SAMPLE_TEXTS)

# ---------------------------------------------------------
# 2. Load trained model
# ---------------------------------------------------------

model = MultilingualPhonemeEncoder(
    len(vocab)
).to(DEVICE)

checkpoint = torch.load(
    f"{OUTPUT_DIR}/lesson51_best.pt",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.eval()

print("=" * 65)
print("LESSON 51 — MODEL TEST")
print("=" * 65)

print(f"Device: {DEVICE}")
print(f"Vocabulary size: {len(vocab)}")

# ---------------------------------------------------------
# 3. Test multilingual inputs
# ---------------------------------------------------------

test_sentences = [
    ("en", "Hello"),
    ("en", "Welcome"),
    ("zh", "你好"),
    ("zh", "谢谢"),
    ("ja", "こんにちは"),
    ("ja", "ありがとう"),
    ("ru", "Привет"),
    ("ru", "Спасибо"),
]

embeddings = []

for language, text in test_sentences:

    result = encode_text(
        model,
        vocab,
        text,
        language
    )

    embedding = result["sentence_embedding"]

    embeddings.append(embedding.numpy())

    print("\n" + "-" * 65)

    print(f"Language: {language}")
    print(f"Text: {text}")

    print(
        "Phonemes:",
        " ".join(result["phonemes"])
    )

    print(
        "Token IDs:",
        result["token_ids"]
    )

    print(
        "Feature shape:",
        tuple(result["features"].shape)
    )

    print(
        "Sentence embedding shape:",
        tuple(embedding.shape)
    )

    print(
        "Embedding norm:",
        f"{embedding.norm().item():.4f}"
    )

    assert embedding.shape[0] == 256

    assert torch.isfinite(embedding).all()

# ---------------------------------------------------------
# 4. Test phoneme reconstruction accuracy
# ---------------------------------------------------------

correct = 0
total = 0

with torch.no_grad():

    for language, text in test_sentences:

        phonemes = text_to_phonemes(
            text,
            language
        )

        ids = vocab.encode(phonemes)

        ids = ids[:MAX_SEQUENCE_LENGTH]

        length = len(ids)

        padded_ids = (
            ids
            + [0] * (MAX_SEQUENCE_LENGTH - length)
        )

        mask = (
            [1] * length
            + [0] * (MAX_SEQUENCE_LENGTH - length)
        )

        x = torch.tensor(
            [padded_ids],
            dtype=torch.long,
            device=DEVICE
        )

        m = torch.tensor(
            [mask],
            dtype=torch.long,
            device=DEVICE
        )

        language_id = torch.tensor(
            [LANGUAGES[language]],
            dtype=torch.long,
            device=DEVICE
        )

        logits, _, _ = model(
            x,
            language_id,
            m
        )

        predictions = logits.argmax(dim=-1)[0, :length]

        targets = x[0, :length]

        correct += (
            predictions == targets
        ).sum().item()

        total += length

accuracy = 100.0 * correct / max(total, 1)

print("\n" + "=" * 65)

print("PHONEME RECONSTRUCTION")

print("=" * 65)

print(f"Correct tokens: {correct}/{total}")

print(f"Accuracy: {accuracy:.2f}%")

# ---------------------------------------------------------
# 5. Sentence embedding similarity
# ---------------------------------------------------------

matrix = np.stack(embeddings)

norms = np.linalg.norm(
    matrix,
    axis=1,
    keepdims=True
)

normalized = matrix / np.maximum(
    norms,
    1e-8
)

similarity = normalized @ normalized.T

print("\n" + "=" * 65)

print("SENTENCE EMBEDDING COSINE SIMILARITY")

print("=" * 65)

for i, (language, text) in enumerate(test_sentences):

    print(
        f"{i}: [{language}] {text}"
    )

print("\nSimilarity matrix:")

print(
    np.round(similarity, 3)
)

assert np.isfinite(similarity).all()

print("\nALL BASIC TESTS COMPLETED.")