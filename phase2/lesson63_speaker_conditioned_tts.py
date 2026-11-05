"""
========================================================================
PHASE 2 — LESSON 63
SPEAKER-CONDITIONED TTS AND VOICE IDENTITY FROM SCRATCH
========================================================================

PURPOSE
-------
Add SPEAKER IDENTITY to the TTS architecture.

Until Lesson 62, the system was approximately:

    Text
      ↓
    Token representation
      ↓
    Duration
      ↓
    Acoustic model
      ↓
    Mel
      ↓
    Vocoder
      ↓
    Waveform

The problem is:

    WHO is speaking?

A text-to-speech model without speaker conditioning cannot explicitly
represent multiple speaker identities.

This lesson introduces:

    SPEAKER EMBEDDING

The new architecture becomes:

    Text
      ↓
    Token Encoder
      ↓
    Text Representation
                         \
                          → Speaker-conditioned representation
                         /
    Speaker Reference
      ↓
    Speaker Encoder
      ↓
    Speaker Embedding

                         ↓
                    Acoustic Model
                         ↓
                        Mel
                         ↓
                       Vocoder
                         ↓
                      Waveform


------------------------------------------------------------------------
NO PRETRAINED MODELS
------------------------------------------------------------------------

This lesson does NOT use:

    - ECAPA-TDNN
    - x-vector models
    - speaker verification models
    - WavLM
    - Wav2Vec2
    - SpeechBrain
    - Resemblyzer
    - pretrained TTS
    - pretrained speaker encoders

The speaker encoder is implemented from scratch.

------------------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------------------

STEP 1
------
Understand why speaker conditioning is necessary.

STEP 2
------
Build a speaker encoder that maps acoustic information into a compact
speaker embedding.

STEP 3
------
Normalize the speaker embedding.

STEP 4
------
Inject the speaker embedding into the acoustic model.

STEP 5
------
Demonstrate that identical text can produce different speaker-
conditioned acoustic representations.

STEP 6
------
Understand the difference between:

    speaker identity
    linguistic content
    acoustic realization

------------------------------------------------------------------------
IMPORTANT CONCEPT
------------------------------------------------------------------------

TTS must separate:

    WHAT is being said

from:

    WHO is saying it

For example:

    "Hello"

Speaker A:

    deep voice
    slower speaking style
    different spectral characteristics

Speaker B:

    higher voice
    different timbre
    different spectral characteristics

The TEXT stays the same.

The SPEAKER EMBEDDING changes.

------------------------------------------------------------------------
VOICE CLONING CONNECTION
------------------------------------------------------------------------

This is the conceptual foundation of voice cloning:

    reference speech
          ↓
    speaker encoder
          ↓
    speaker embedding
          ↓
    TTS acoustic model
          ↓
    synthesized speech

A real voice-cloning system needs much more than this educational
demonstration, including:

    - robust speaker embeddings
    - large multi-speaker datasets
    - speaker disentanglement
    - prosody modeling
    - pitch modeling
    - energy modeling
    - duration modeling
    - high-quality neural vocoding
    - speaker similarity evaluation

------------------------------------------------------------------------
MULTILINGUAL EXAMPLES
------------------------------------------------------------------------

English:
    Hello, this is my voice.

Chinese:
    这是我的声音。

Japanese:
    これは私の声です。

Russian:
    Это мой голос.

------------------------------------------------------------------------
REAL-WORLD EXAMPLE
------------------------------------------------------------------------

Suppose we have:

    Text:
        "Hello, how are you?"

and two reference speakers:

    Speaker A → reference.wav
    Speaker B → reference.wav

The text encoder receives the SAME text.

The speaker encoder receives DIFFERENT reference speech.

Therefore:

    Text representation:
        SAME

    Speaker embedding:
        DIFFERENT

    Acoustic representation:
        DIFFERENT

This is the basic mechanism behind multi-speaker TTS.

------------------------------------------------------------------------
"""

import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path(
    "lesson63_outputs"
)

N_MELS = 80

EMBED_DIM = 128

HIDDEN_DIM = 128

SPEAKER_DIM = 64

NUM_SPEAKERS = 4

NUM_REFERENCE_FRAMES = 80


# ============================================================================
# SEED
# ============================================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# STEP 1 — TOKENIZER
# ============================================================================

class SimpleTokenizer:

    def __init__(self):

        symbols = [
            "<pad>",
            "<unk>",
            "<sil>",
        ]

        symbols += list(
            "abcdefghijklmnopqrstuvwxyz"
        )

        symbols += list(
            "0123456789"
        )

        symbols += list(
            " .,!?;:'-"
        )

        symbols += list(
            "这是一个从零开始构建的语音合成系统。"
        )

        symbols += list(
            "これはゼロから構築した音声合成システムです。"
        )

        symbols += list(
            "этосистема"
        )

        unique = []

        for symbol in symbols:

            if symbol not in unique:
                unique.append(symbol)

        self.symbols = unique

        self.symbol_to_id = {
            symbol: i
            for i, symbol in enumerate(
                self.symbols
            )
        }

        self.id_to_symbol = {
            i: symbol
            for i, symbol in enumerate(
                self.symbols
            )
        }

        self.unk_id = self.symbol_to_id[
            "<unk>"
        ]

    def encode(self, text):

        text = text.strip().lower()

        ids = []

        for char in text:

            ids.append(
                self.symbol_to_id.get(
                    char,
                    self.unk_id,
                )
            )

        if len(ids) == 0:

            ids = [
                self.unk_id
            ]

        return ids

    @property
    def vocab_size(self):

        return len(self.symbols)


# ============================================================================
# STEP 2 — TEXT ENCODER
# ============================================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        dim,
        max_length=10000,
    ):

        super().__init__()

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                dim,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / dim
            )
        )

        pe = torch.zeros(
            max_length,
            dim,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x):

        return (
            x
            + self.pe[
                :,
                :x.shape[1],
                :,
            ]
        )


class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        self.input_projection = nn.Linear(
            embed_dim,
            hidden_dim,
        )

        self.position = PositionalEncoding(
            hidden_dim
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=3,
        )

    def forward(self, token_ids):

        x = self.embedding(
            token_ids
        )

        x = self.input_projection(
            x
        )

        x = self.position(
            x
        )

        x = self.encoder(
            x
        )

        return x


# ============================================================================
# STEP 3 — SPEAKER ENCODER
# ============================================================================

class SpeakerEncoder(nn.Module):
    """
    Convert a Mel-like acoustic representation into a speaker embedding.

    Input:

        [B, n_mels, frames]

    Output:

        [B, speaker_dim]

    The temporal dimension is pooled because speaker identity should
    not depend strongly on the exact position of individual phonemes.
    """

    def __init__(
        self,
        n_mels=N_MELS,
        speaker_dim=SPEAKER_DIM,
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            n_mels,
            128,
            kernel_size=5,
            padding=2,
        )

        self.conv2 = nn.Conv1d(
            128,
            128,
            kernel_size=5,
            padding=2,
        )

        self.conv3 = nn.Conv1d(
            128,
            256,
            kernel_size=5,
            padding=2,
        )

        self.norm1 = nn.BatchNorm1d(
            128
        )

        self.norm2 = nn.BatchNorm1d(
            128
        )

        self.norm3 = nn.BatchNorm1d(
            256
        )

        self.projection = nn.Sequential(

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                speaker_dim,
            ),
        )

    def forward(self, mel):

        x = self.conv1(
            mel
        )

        x = self.norm1(
            x
        )

        x = F.relu(
            x
        )

        x = self.conv2(
            x
        )

        x = self.norm2(
            x
        )

        x = F.relu(
            x
        )

        x = self.conv3(
            x
        )

        x = self.norm3(
            x
        )

        x = F.relu(
            x
        )

        # Global temporal pooling.
        x = torch.mean(
            x,
            dim=-1,
        )

        embedding = self.projection(
            x
        )

        # L2 normalization.
        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding


# ============================================================================
# SPEAKER CONDITIONING
# ============================================================================

class SpeakerConditioning(nn.Module):

    def __init__(
        self,
        speaker_dim=SPEAKER_DIM,
        hidden_dim=HIDDEN_DIM,
    ):

        super().__init__()

        self.projection = nn.Sequential(

            nn.Linear(
                speaker_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

    def forward(
        self,
        text_features,
        speaker_embedding,
    ):

        speaker = self.projection(
            speaker_embedding
        )

        # [B, hidden]
        #
        # becomes
        #
        # [B, 1, hidden]

        speaker = speaker.unsqueeze(
            1
        )

        # Broadcast over all text frames/tokens.
        speaker = speaker.expand(
            -1,
            text_features.shape[1],
            -1,
        )

        conditioned = (
            text_features
            + speaker
        )

        return conditioned


# ============================================================================
# DURATION PREDICTOR
# ============================================================================

class DurationPredictor(nn.Module):

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, x):

        return self.net(
            x
        ).squeeze(-1)


# ============================================================================
# LENGTH REGULATOR
# ============================================================================

class LengthRegulator(nn.Module):

    def forward(
        self,
        x,
        durations,
    ):

        batch_outputs = []

        for b in range(
            x.shape[0]
        ):

            frames = []

            for t in range(
                x.shape[1]
            ):

                duration = int(
                    durations[
                        b,
                        t,
                    ].item()
                )

                duration = max(
                    1,
                    min(
                        duration,
                        20,
                    ),
                )

                frames.append(
                    x[
                        b,
                        t,
                    ].unsqueeze(0).repeat(
                        duration,
                        1,
                    )
                )

            if len(frames) == 0:

                output = x[
                    b
                ].new_zeros(
                    (
                        1,
                        x.shape[-1],
                    )
                )

            else:

                output = torch.cat(
                    frames,
                    dim=0,
                )

            batch_outputs.append(
                output
            )

        max_length = max(
            x.shape[0]
            for x in batch_outputs
        )

        padded = []

        for x in batch_outputs:

            if x.shape[0] < max_length:

                padding = x.new_zeros(
                    (
                        max_length - x.shape[0],
                        x.shape[1],
                    )
                )

                x = torch.cat(
                    [
                        x,
                        padding,
                    ],
                    dim=0,
                )

            padded.append(
                x
            )

        return torch.stack(
            padded,
            dim=0,
        )


# ============================================================================
# ACOUSTIC MODEL
# ============================================================================

class SpeakerConditionedAcousticModel(nn.Module):

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        n_mels=N_MELS,
    ):

        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=3,
        )

        self.output = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                n_mels,
            ),
        )

    def forward(self, x):

        x = self.encoder(
            x
        )

        mel = self.output(
            x
        )

        return mel


# ============================================================================
# COMPLETE SPEAKER-CONDITIONED TTS
# ============================================================================

class SpeakerConditionedTTS(nn.Module):

    def __init__(
        self,
        vocab_size,
    ):

        super().__init__()

        self.text_encoder = TextEncoder(
            vocab_size
        )

        self.speaker_encoder = SpeakerEncoder(
            N_MELS,
            SPEAKER_DIM,
        )

        self.speaker_conditioning = SpeakerConditioning(
            SPEAKER_DIM,
            HIDDEN_DIM,
        )

        self.duration_predictor = DurationPredictor(
            HIDDEN_DIM
        )

        self.length_regulator = LengthRegulator()

        self.acoustic_model = SpeakerConditionedAcousticModel(
            HIDDEN_DIM,
            N_MELS,
        )

    def forward(
        self,
        token_ids,
        speaker_mel,
        durations,
    ):

        # ------------------------------------------------------------
        # Text representation
        # ------------------------------------------------------------

        text_features = self.text_encoder(
            token_ids
        )

        # ------------------------------------------------------------
        # Speaker representation
        # ------------------------------------------------------------

        speaker_embedding = self.speaker_encoder(
            speaker_mel
        )

        # ------------------------------------------------------------
        # Speaker conditioning
        # ------------------------------------------------------------

        conditioned = self.speaker_conditioning(
            text_features,
            speaker_embedding,
        )

        # ------------------------------------------------------------
        # Duration
        # ------------------------------------------------------------

        predicted_log_duration = (
            self.duration_predictor(
                conditioned
            )
        )

        # ------------------------------------------------------------
        # Length regulation
        # ------------------------------------------------------------

        frame_features = self.length_regulator(
            conditioned,
            durations,
        )

        # ------------------------------------------------------------
        # Acoustic model
        # ------------------------------------------------------------

        mel = self.acoustic_model(
            frame_features
        )

        return {
            "speaker_embedding": speaker_embedding,
            "text_features": text_features,
            "conditioned_features": conditioned,
            "predicted_log_duration": predicted_log_duration,
            "frame_features": frame_features,
            "mel": mel,
        }


# ============================================================================
# DEMO DURATIONS
# ============================================================================

def make_demo_durations(
    token_ids,
):

    durations = []

    for token_id in token_ids:

        # Deterministic but varied durations.
        duration = (
            3
            + (
                int(token_id)
                % 4
            )
        )

        durations.append(
            duration
        )

    return torch.tensor(
        durations,
        dtype=torch.long,
    )


# ============================================================================
# CREATE SYNTHETIC SPEAKER REFERENCES
# ============================================================================

def create_demo_speaker_mel(
    speaker_id,
    frames=NUM_REFERENCE_FRAMES,
):

    """
    Create deterministic synthetic acoustic references.

    These are NOT real speaker recordings.

    They are used only to demonstrate the mechanics of speaker
    conditioning without requiring a pretrained speaker encoder.
    """

    time = np.linspace(
        0,
        1,
        frames,
        dtype=np.float32,
    )

    mel = np.zeros(
        (
            N_MELS,
            frames,
        ),
        dtype=np.float32,
    )

    # Speaker-dependent spectral envelope.
    center = (
        20
        + speaker_id * 8
    )

    width = (
        10
        + speaker_id * 2
    )

    frequency_bins = np.arange(
        N_MELS,
        dtype=np.float32,
    )

    envelope = np.exp(
        -(
            (
                frequency_bins
                - center
            )
            ** 2
        )
        / (
            2 * width ** 2
        )
    )

    # Time-varying component.
    temporal = (
        0.5
        + 0.5
        * np.sin(
            2
            * np.pi
            * (
                speaker_id + 1
            )
            * time
        )
    )

    mel += (
        envelope[:, None]
        * temporal[None, :]
    )

    # Speaker-specific energy.
    mel *= (
        0.8
        + 0.1 * speaker_id
    )

    # Small deterministic harmonic pattern.
    mel += (
        0.03
        * np.sin(
            (
                frequency_bins[:, None]
            )
            * (
                speaker_id + 1
            )
            * 0.3
            + time[None, :]
            * 5.0
        )
    )

    return torch.from_numpy(
        mel
    ).float()


# ============================================================================
# COSINE SIMILARITY
# ============================================================================

def cosine_similarity(
    a,
    b,
):

    return F.cosine_similarity(
        a,
        b,
        dim=-1,
    )


# ============================================================================
# EMBEDDING VISUALIZATION
# ============================================================================

def save_embedding_plot(
    embeddings,
    labels,
):

    embeddings = embeddings.detach().cpu()

    # We only need two dimensions for visualization.
    #
    # PCA is implemented directly here.

    centered = (
        embeddings
        - embeddings.mean(
            dim=0,
            keepdim=True,
        )
    )

    covariance = (
        centered.T
        @ centered
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(
        covariance
    )

    order = torch.argsort(
        eigenvalues,
        descending=True,
    )

    components = eigenvectors[
        :,
        order[:2],
    ]

    projected = (
        centered
        @ components
    )

    projected = projected.numpy()

    plt.figure(
        figsize=(8, 6)
    )

    for i, label in enumerate(
        labels
    ):

        plt.scatter(
            projected[i, 0],
            projected[i, 1],
            s=100,
        )

        plt.text(
            projected[i, 0],
            projected[i, 1],
            f" Speaker {label}",
        )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        "Speaker Embedding Space"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "speaker_embedding_space.png",
        dpi=150,
    )

    plt.close()


# ============================================================================
# MEL DIFFERENCE VISUALIZATION
# ============================================================================

def save_mel_difference(
    mel_a,
    mel_b,
):

    a = mel_a.detach().cpu().numpy()

    b = mel_b.detach().cpu().numpy()

    if a.ndim == 3:
        a = a[0]

    if b.ndim == 3:
        b = b[0]

    min_frames = min(
        a.shape[0],
        b.shape[0],
    )

    a = a[:min_frames]
    b = b[:min_frames]

    difference = (
        a - b
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        difference.T,
        aspect="auto",
        origin="lower",
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Mel Bin"
    )

    plt.title(
        "Speaker-Conditioned Mel Difference"
    )

    plt.colorbar(
        label="Difference"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "speaker_conditioned_mel_difference.png",
        dpi=150,
    )

    plt.close()


# ============================================================================
# PARAMETER COUNT
# ============================================================================

def count_parameters(
    model,
):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("PHASE 2 — LESSON 63")
    print("SPEAKER-CONDITIONED TTS")
    print("=" * 72)

    print()
    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================================
    # STEP 1 — TOKENIZER
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 1 — TEXT REPRESENTATION")
    print("=" * 72)

    tokenizer = SimpleTokenizer()

    text = (
        "Hello, this is my voice."
    )

    token_ids = tokenizer.encode(
        text
    )

    token_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
    ).unsqueeze(0).to(
        DEVICE
    )

    print(
        f"Text: {text}"
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"Token count: "
        f"{len(token_ids)}"
    )

    # ========================================================================
    # STEP 2 — BUILD MODEL
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 2 — BUILD SPEAKER-CONDITIONED TTS")
    print("=" * 72)

    model = SpeakerConditionedTTS(
        tokenizer.vocab_size
    ).to(
        DEVICE
    )

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ========================================================================
    # STEP 3 — CREATE SPEAKER REFERENCES
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 3 — SPEAKER REFERENCES")
    print("=" * 72)

    speaker_references = []

    for speaker_id in range(
        NUM_SPEAKERS
    ):

        reference = create_demo_speaker_mel(
            speaker_id
        )

        reference = reference.unsqueeze(
            0
        ).to(
            DEVICE
        )

        speaker_references.append(
            reference
        )

        print(
            f"Speaker {speaker_id}: "
            f"reference shape = "
            f"{tuple(reference.shape)}"
        )

    # ========================================================================
    # STEP 4 — EXTRACT SPEAKER EMBEDDINGS
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 4 — SPEAKER ENCODER")
    print("=" * 72)

    model.eval()

    embeddings = []

    with torch.no_grad():

        for speaker_id, reference in enumerate(
            speaker_references
        ):

            embedding = model.speaker_encoder(
                reference
            )

            embeddings.append(
                embedding[0]
            )

            print(
                f"Speaker {speaker_id}: "
                f"embedding shape = "
                f"{tuple(embedding.shape)}"
            )

            print(
                f"  embedding norm = "
                f"{embedding.norm().item():.6f}"
            )

    embeddings = torch.stack(
        embeddings,
        dim=0,
    )

    # ========================================================================
    # STEP 5 — SPEAKER SIMILARITY
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 5 — SPEAKER EMBEDDING SIMILARITY")
    print("=" * 72)

    similarity_matrix = torch.zeros(
        NUM_SPEAKERS,
        NUM_SPEAKERS,
    )

    for i in range(
        NUM_SPEAKERS
    ):

        for j in range(
            NUM_SPEAKERS
        ):

            similarity_matrix[
                i,
                j,
            ] = cosine_similarity(
                embeddings[i:i + 1],
                embeddings[j:j + 1],
            )[0]

    print()
    print(
        "Cosine similarity matrix:"
    )

    print(
        similarity_matrix.cpu().numpy()
    )

    save_embedding_plot(
        embeddings,
        list(
            range(NUM_SPEAKERS)
        ),
    )

    # ========================================================================
    # STEP 6 — SAME TEXT, DIFFERENT SPEAKERS
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 6 — SAME TEXT + DIFFERENT SPEAKER")
    print("=" * 72)

    durations = make_demo_durations(
        token_ids
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    outputs = []

    with torch.no_grad():

        for speaker_id in range(
            NUM_SPEAKERS
        ):

            output = model(
                token_tensor,
                speaker_references[
                    speaker_id
                ],
                durations,
            )

            outputs.append(
                output
            )

            print()
            print(
                f"Speaker {speaker_id}"
            )

            print(
                f"  Speaker embedding: "
                f"{tuple(output['speaker_embedding'].shape)}"
            )

            print(
                f"  Text representation: "
                f"{tuple(output['text_features'].shape)}"
            )

            print(
                f"  Conditioned representation: "
                f"{tuple(output['conditioned_features'].shape)}"
            )

            print(
                f"  Mel: "
                f"{tuple(output['mel'].shape)}"
            )

    # ========================================================================
    # STEP 7 — COMPARE SPEAKER-CONDITIONED MELS
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 7 — SPEAKER EFFECT ON ACOUSTIC OUTPUT")
    print("=" * 72)

    mel_a = outputs[0][
        "mel"
    ]

    mel_b = outputs[1][
        "mel"
    ]

    mel_difference = torch.mean(
        torch.abs(
            mel_a - mel_b
        )
    )

    print(
        f"Mean absolute Mel difference "
        f"between Speaker 0 and Speaker 1: "
        f"{mel_difference.item():.6f}"
    )

    save_mel_difference(
        mel_a,
        mel_b,
    )

    # ========================================================================
    # STEP 8 — SPEAKER CONDITIONING TEST
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 8 — SPEAKER CONDITIONING TEST")
    print("=" * 72)

    print()
    print(
        "Same text:"
    )

    print(
        f"    {text}"
    )

    print()
    print(
        "Speaker reference changes:"
    )

    print(
        "    Speaker 0 → Speaker embedding A"
    )

    print(
        "    Speaker 1 → Speaker embedding B"
    )

    print()
    print(
        "Therefore the acoustic representations differ."
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    checkpoint = (
        OUTPUT_DIR
        / "speaker_conditioned_tts.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "speaker_dim": SPEAKER_DIM,
            "n_mels": N_MELS,
        },
        checkpoint,
    )

    print()
    print(
        f"Checkpoint saved:"
    )

    print(
        f"  {checkpoint}"
    )

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print("=" * 72)
    print("LESSON 63 COMPLETE")
    print("=" * 72)

    print()
    print(
        "The architecture is now:"
    )

    print(
        """
Text
  ↓
Text Encoder
  ↓
                         Speaker Reference
                                ↓
                         Speaker Encoder
                                ↓
                         Speaker Embedding
                                ↓
Text Representation ─────→ Conditioning
                                ↓
                       Duration Prediction
                                ↓
                         Length Regulation
                                ↓
                      Acoustic Transformer
                                ↓
                              Mel
"""
    )

    print(
        "Core concept:"
    )

    print(
        "    TEXT = what is spoken"
    )

    print(
        "    SPEAKER EMBEDDING = who is speaking"
    )

    print()
    print(
        "This is the foundation for multi-speaker TTS"
    )

    print(
        "and eventually zero-shot voice cloning."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The speaker references in this lesson are synthetic "
        "demonstration features, not real voices."
    )

    print(
        "The next stage should train the speaker representation "
        "from REAL multi-speaker audio."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()