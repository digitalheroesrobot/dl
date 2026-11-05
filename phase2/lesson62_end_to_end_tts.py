"""
========================================================================
PHASE 2 — LESSON 62
END-TO-END TTS INTEGRATION FROM SCRATCH
========================================================================

PURPOSE
-------
Integrate the major components developed throughout Phase 2 into one
complete text-to-speech pipeline.

This lesson is about SYSTEM INTEGRATION.

The individual components are:

    text processing
        ↓
    phoneme representation
        ↓
    duration prediction
        ↓
    length regulation
        ↓
    acoustic model
        ↓
    Mel-spectrogram
        ↓
    neural vocoder
        ↓
    waveform

NO PRETRAINED MODELS
--------------------
This lesson does NOT use:

    - Whisper
    - Tacotron
    - FastSpeech
    - FastSpeech 2 pretrained checkpoints
    - VITS
    - HiFi-GAN
    - WaveRNN
    - pretrained phonemizers
    - pretrained acoustic models
    - pretrained vocoders

Everything demonstrated here is implemented with PyTorch/Python.

------------------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------------------

STEP 1 — TEXT PROCESSING
------------------------
Convert input text into a deterministic token sequence.

STEP 2 — TOKEN EMBEDDING
------------------------
Convert tokens into vectors.

STEP 3 — DURATION PREDICTION
----------------------------
Predict how many acoustic frames each token should occupy.

STEP 4 — LENGTH REGULATION
--------------------------
Expand token-level representations into frame-level representations.

STEP 5 — ACOUSTIC MODEL
-----------------------
Predict a Mel-spectrogram from the frame-level representation.

STEP 6 — NEURAL VOCODER
----------------------
Convert the predicted Mel-spectrogram into a waveform.

STEP 7 — COMPLETE PIPELINE
--------------------------
Execute:

    text → tokens → durations → frames → Mel → waveform

------------------------------------------------------------------------
IMPORTANT EDUCATIONAL LIMITATION
------------------------------------------------------------------------

This lesson demonstrates the COMPLETE DATA FLOW.

It is intentionally not presented as a production-quality TTS system.

A real TTS system requires:

    - much larger datasets
    - robust multilingual text normalization
    - real phoneme conversion
    - many speakers
    - speaker conditioning
    - better duration supervision
    - pitch/energy conditioning
    - stronger acoustic models
    - better vocoders
    - validation/test splits
    - extensive audio evaluation

The purpose here is to understand how the components fit together.

------------------------------------------------------------------------
MULTILINGUAL EXAMPLES
------------------------------------------------------------------------

English:
    Hello, this is a neural text to speech system.

Chinese:
    这是一个从零开始构建的语音合成系统。

Japanese:
    これはゼロから構築した音声合成システムです。

Russian:
    Это система синтеза речи, созданная с нуля.

------------------------------------------------------------------------
REAL-WORLD EXAMPLE
------------------------------------------------------------------------

A voice assistant performs approximately:

    User:
        "Turn on the lights."

    ↓

    Text normalization

    ↓

    phoneme/token sequence

    ↓

    duration prediction

    ↓

    acoustic representation

    ↓

    Mel-spectrogram

    ↓

    neural vocoder

    ↓

    waveform

    ↓

    speaker hears:

        "Turn on the lights."

The important engineering insight is that TTS is not one model.

It is a PIPELINE of representations.

------------------------------------------------------------------------
"""

import math
import random
import re
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
    "lesson62_outputs"
)

SAMPLE_RATE = 16000

HOP_LENGTH = 256

N_MELS = 80

EMBED_DIM = 128

HIDDEN_DIM = 128

MAX_DURATION = 20


# ============================================================================
# RANDOM SEED
# ============================================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# STEP 1 — SIMPLE MULTILINGUAL TEXT TOKENIZATION
# ============================================================================

class SimpleTokenizer:
    """
    Educational tokenizer.

    This is deliberately simple.

    It does NOT claim to be a real phonemizer.

    Instead, it provides a deterministic token representation that lets
    us demonstrate the complete TTS architecture.

    In later stages, this can be replaced by a real multilingual
    phoneme representation built from scratch.
    """

    def __init__(self):

        symbols = [
            "<pad>",
            "<unk>",
            "<sil>",
        ]

        # ASCII letters.
        symbols += list(
            "abcdefghijklmnopqrstuvwxyz"
        )

        # Digits.
        symbols += list(
            "0123456789"
        )

        # Punctuation.
        symbols += list(
            " .,!?;:'-"
        )

        # Chinese characters used by the educational examples.
        symbols += list(
            "这是一个从零开始构建的语音合成系统。"
        )

        # Japanese characters.
        symbols += list(
            "これゼロ音声合成です。"
        )

        # Russian characters.
        symbols += list(
            "этосистема"
        )

        # Remove duplicates while preserving order.
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

        self.pad_id = self.symbol_to_id[
            "<pad>"
        ]

        self.unk_id = self.symbol_to_id[
            "<unk>"
        ]

        self.sil_id = self.symbol_to_id[
            "<sil>"
        ]

    def normalize(self, text):

        text = text.strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    def encode(self, text):

        text = self.normalize(text)

        ids = []

        for char in text.lower():

            if char in self.symbol_to_id:

                ids.append(
                    self.symbol_to_id[char]
                )

            else:

                ids.append(
                    self.unk_id
                )

        if len(ids) == 0:

            ids = [
                self.sil_id
            ]

        return ids

    def decode(self, ids):

        return "".join(
            self.id_to_symbol.get(
                int(i),
                "<unk>",
            )
            for i in ids
        )

    @property
    def vocab_size(self):

        return len(self.symbols)


# ============================================================================
# STEP 2 — TOKEN EMBEDDING
# ============================================================================

class TokenEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

    def forward(self, token_ids):

        return self.embedding(
            token_ids
        )


# ============================================================================
# STEP 3 — DURATION PREDICTOR
# ============================================================================

class DurationPredictor(nn.Module):
    """
    Predict log-duration for every token.

    Input:

        [B, T, D]

    Output:

        [B, T]

    The logarithmic representation is useful because speech durations
    are positive and usually have a strongly skewed distribution.
    """

    def __init__(
        self,
        hidden_dim,
    ):

        super().__init__()

        self.network = nn.Sequential(

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

        output = self.network(x)

        output = output.squeeze(-1)

        return output


# ============================================================================
# STEP 4 — LENGTH REGULATOR
# ============================================================================

class LengthRegulator(nn.Module):
    """
    Convert token-level features into frame-level features.

    Example:

        token features:

            A B C

        durations:

            2 3 1

        becomes:

            A A B B B C

    """

    def forward(
        self,
        x,
        durations,
    ):

        outputs = []

        batch_size = x.shape[0]

        for b in range(batch_size):

            expanded = []

            for t in range(
                x.shape[1]
            ):

                duration = int(
                    durations[b, t].item()
                )

                duration = max(
                    1,
                    duration,
                )

                duration = min(
                    MAX_DURATION,
                    duration,
                )

                repeated = x[
                    b,
                    t,
                ].unsqueeze(0).repeat(
                    duration,
                    1,
                )

                expanded.append(
                    repeated
                )

            if len(expanded) == 0:

                expanded_tensor = x[
                    b
                ].new_zeros(
                    (1, x.shape[-1])
                )

            else:

                expanded_tensor = torch.cat(
                    expanded,
                    dim=0,
                )

            outputs.append(
                expanded_tensor
            )

        max_frames = max(
            item.shape[0]
            for item in outputs
        )

        padded = []

        for item in outputs:

            if item.shape[0] < max_frames:

                padding = item.new_zeros(
                    (
                        max_frames - item.shape[0],
                        item.shape[1],
                    )
                )

                item = torch.cat(
                    [
                        item,
                        padding,
                    ],
                    dim=0,
                )

            padded.append(item)

        return torch.stack(
            padded,
            dim=0,
        )


# ============================================================================
# POSITIONAL ENCODING
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

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):

        length = x.shape[1]

        return x + self.pe[
            :,
            :length,
            :,
        ]


# ============================================================================
# STEP 5 — ACOUSTIC MODEL
# ============================================================================

class AcousticModel(nn.Module):
    """
    Small educational acoustic model.

    Frame-level hidden representation:

        [B, frames, hidden]

    becomes:

        [B, frames, n_mels]
    """

    def __init__(
        self,
        hidden_dim,
        n_mels,
    ):

        super().__init__()

        self.position = PositionalEncoding(
            hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
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

        x = self.position(x)

        x = self.encoder(x)

        mel = self.output(x)

        return mel


# ============================================================================
# STEP 6 — EDUCATIONAL NEURAL VOCODER
# ============================================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            3,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            3,
            padding=1,
        )

    def forward(self, x):

        residual = x

        x = self.conv1(x)

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.conv2(x)

        x = x + residual

        x = F.leaky_relu(
            x,
            0.2,
        )

        return x


class SimpleNeuralVocoder(nn.Module):
    """
    Small demonstration vocoder.

    Mel:

        [B, n_mels, frames]

    →

        waveform

    Four ×4 upsampling stages give approximately:

        4 × 4 × 4 × 4 = 256

    samples per Mel frame.
    """

    def __init__(
        self,
        n_mels,
    ):

        super().__init__()

        self.input = nn.Conv1d(
            n_mels,
            128,
            7,
            padding=3,
        )

        self.up1 = nn.ConvTranspose1d(
            128,
            96,
            8,
            stride=4,
            padding=2,
        )

        self.up2 = nn.ConvTranspose1d(
            96,
            64,
            8,
            stride=4,
            padding=2,
        )

        self.up3 = nn.ConvTranspose1d(
            64,
            48,
            8,
            stride=4,
            padding=2,
        )

        self.up4 = nn.ConvTranspose1d(
            48,
            32,
            8,
            stride=4,
            padding=2,
        )

        self.residual = ResidualBlock(
            32
        )

        self.output = nn.Conv1d(
            32,
            1,
            7,
            padding=3,
        )

    def forward(
        self,
        mel,
    ):

        x = self.input(mel)

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.up1(x)

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.up2(x)

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.up3(x)

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.up4(x)

        x = F.leaky_relu(
            x,
            0.2,
        )

        x = self.residual(x)

        x = self.output(x)

        x = torch.tanh(x)

        return x


# ============================================================================
# COMPLETE TTS MODEL
# ============================================================================

class EndToEndTTS(nn.Module):
    """
    Complete educational TTS architecture.

        tokens
          ↓
        embedding
          ↓
        duration predictor
          ↓
        length regulator
          ↓
        acoustic model
          ↓
        Mel
          ↓
        vocoder
          ↓
        waveform
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_mels=N_MELS,
    ):

        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size,
            embedding_dim,
        )

        self.input_projection = nn.Linear(
            embedding_dim,
            hidden_dim,
        )

        self.duration_predictor = DurationPredictor(
            hidden_dim
        )

        self.length_regulator = LengthRegulator()

        self.acoustic_model = AcousticModel(
            hidden_dim,
            n_mels,
        )

        self.vocoder = SimpleNeuralVocoder(
            n_mels
        )

    def predict_mel(
        self,
        token_ids,
        durations,
    ):

        x = self.embedding(
            token_ids
        )

        x = self.input_projection(
            x
        )

        log_durations = self.duration_predictor(
            x
        )

        frame_features = self.length_regulator(
            x,
            durations,
        )

        mel = self.acoustic_model(
            frame_features
        )

        return (
            mel,
            log_durations,
            frame_features,
        )

    def forward(
        self,
        token_ids,
        durations,
    ):

        mel, log_durations, frame_features = self.predict_mel(
            token_ids,
            durations,
        )

        mel_for_vocoder = mel.transpose(
            1,
            2,
        )

        waveform = self.vocoder(
            mel_for_vocoder
        )

        return {
            "mel": mel,
            "log_durations": log_durations,
            "frame_features": frame_features,
            "waveform": waveform,
        }


# ============================================================================
# DEMONSTRATION DURATION HEURISTIC
# ============================================================================

def estimate_demo_durations(
    token_ids,
    tokenizer,
):
    """
    Generate deterministic demonstration durations.

    This is NOT a trained duration predictor.

    It lets us demonstrate the complete pipeline before the full
    integrated system is trained.

    Later, replace this with:

        predicted_duration = duration_predictor(...)
    """

    durations = []

    for token_id in token_ids:

        symbol = tokenizer.id_to_symbol[
            int(token_id)
        ]

        if symbol in " .,!?;:'-":

            duration = 2

        elif symbol in "aeiou":

            duration = 5

        elif symbol in "0123456789":

            duration = 5

        else:

            duration = 4

        durations.append(
            duration
        )

    return torch.tensor(
        durations,
        dtype=torch.long,
    )


# ============================================================================
# PARAMETER COUNT
# ============================================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================================
# MEL VISUALIZATION
# ============================================================================

def save_mel_plot(
    mel,
    filename,
    title,
):

    mel = mel.detach().cpu().numpy()

    if mel.ndim == 3:

        mel = mel[0]

    if mel.shape[0] != N_MELS:

        mel = mel.T

    plt.figure(
        figsize=(12, 5)
    )

    plt.imshow(
        mel.T,
        aspect="auto",
        origin="lower",
    )

    plt.xlabel(
        "Time Frame"
    )

    plt.ylabel(
        "Mel Bin"
    )

    plt.title(
        title
    )

    plt.colorbar(
        label="Value"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150,
    )

    plt.close()


# ============================================================================
# WAVEFORM VISUALIZATION
# ============================================================================

def save_waveform_plot(
    waveform,
    filename,
    title,
):

    waveform = waveform.detach().cpu().numpy()

    waveform = waveform.reshape(-1)

    time = np.arange(
        len(waveform)
    ) / SAMPLE_RATE

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        time,
        waveform,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150,
    )

    plt.close()


# ============================================================================
# ARCHITECTURE PRINT
# ============================================================================

def print_architecture():

    print()
    print("=" * 72)
    print("COMPLETE TTS ARCHITECTURE")
    print("=" * 72)

    print(
        """
Text
 ↓
Text normalization
 ↓
Tokenization
 ↓
Token IDs
 ↓
Embedding
 ↓
Duration Predictor
 ↓
Length Regulator
 ↓
Frame-level representation
 ↓
Acoustic Transformer
 ↓
Mel-spectrogram
 ↓
Neural Vocoder
 ↓
Waveform
"""
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
    print("PHASE 2 — LESSON 62")
    print("END-TO-END TTS INTEGRATION FROM SCRATCH")
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
    print("STEP 1 — TEXT PROCESSING")
    print("=" * 72)

    tokenizer = SimpleTokenizer()

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    examples = [
        "Hello, this is a neural text to speech system.",
        "这是一个从零开始构建的语音合成系统。",
        "これはゼロから構築した音声合成システムです。",
        "Это система синтеза речи, созданная с нуля.",
    ]

    for text in examples:

        ids = tokenizer.encode(
            text
        )

        print()
        print(
            f"Text: {text}"
        )

        print(
            f"Tokens: {len(ids)}"
        )

        print(
            f"IDs: {ids[:30]}"
            + (
                " ..."
                if len(ids) > 30
                else ""
            )
        )

    # ========================================================================
    # SELECT DEMO TEXT
    # ========================================================================

    text = examples[0]

    token_ids = tokenizer.encode(
        text
    )

    token_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
    ).unsqueeze(0).to(
        DEVICE
    )

    print()
    print(
        f"Selected text:"
    )

    print(
        f"  {text}"
    )

    print(
        f"Token count: "
        f"{len(token_ids)}"
    )

    # ========================================================================
    # STEP 2 — MODEL
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 2 — BUILD COMPLETE TTS MODEL")
    print("=" * 72)

    model = EndToEndTTS(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_mels=N_MELS,
    ).to(
        DEVICE
    )

    print(
        f"Total parameters: "
        f"{count_parameters(model):,}"
    )

    print_architecture()

    # ========================================================================
    # STEP 3 — DURATION
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 3 — DURATION + LENGTH REGULATION")
    print("=" * 72)

    demo_durations = estimate_demo_durations(
        token_ids,
        tokenizer,
    )

    demo_durations = demo_durations.unsqueeze(
        0
    ).to(
        DEVICE
    )

    print(
        f"Number of tokens: "
        f"{len(token_ids)}"
    )

    print(
        f"Total acoustic frames: "
        f"{demo_durations.sum().item()}"
    )

    print()
    print(
        "First token durations:"
    )

    for i in range(
        min(
            20,
            len(token_ids),
        )
    ):

        symbol = tokenizer.id_to_symbol[
            token_ids[i]
        ]

        duration = demo_durations[
            0,
            i,
        ].item()

        print(
            f"  {i:3d} "
            f"'{symbol}' "
            f"→ {duration} frames"
        )

    # ========================================================================
    # STEP 4 — ACOUSTIC MODEL
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 4 — PREDICT MEL-SPECTROGRAM")
    print("=" * 72)

    model.eval()

    with torch.no_grad():

        output = model(
            token_tensor,
            demo_durations,
        )

    mel = output["mel"]

    frame_features = output[
        "frame_features"
    ]

    predicted_log_durations = output[
        "log_durations"
    ]

    waveform = output[
        "waveform"
    ]

    print(
        f"Frame representation shape: "
        f"{tuple(frame_features.shape)}"
    )

    print(
        f"Mel shape: "
        f"{tuple(mel.shape)}"
    )

    print(
        f"Predicted log-duration shape: "
        f"{tuple(predicted_log_durations.shape)}"
    )

    # Acoustic model output is:
    #
    #     [B, frames, n_mels]
    #
    # Save visualization.

    save_mel_plot(
        mel,
        "predicted_mel.png",
        "Lesson 62 — Predicted Mel-Spectrogram",
    )

    # ========================================================================
    # STEP 5 — VOCODER
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 5 — MEL → WAVEFORM")
    print("=" * 72)

    print(
        f"Waveform shape: "
        f"{tuple(waveform.shape)}"
    )

    waveform_length = waveform.shape[-1]

    duration_seconds = (
        waveform_length
        / SAMPLE_RATE
    )

    print(
        f"Generated waveform samples: "
        f"{waveform_length}"
    )

    print(
        f"Approximate duration: "
        f"{duration_seconds:.3f} sec"
    )

    save_waveform_plot(
        waveform,
        "generated_waveform.png",
        "Lesson 62 — Generated Waveform",
    )

    # ========================================================================
    # STEP 6 — SAVE AUDIO
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 6 — SAVE OUTPUT")
    print("=" * 72)

    waveform_np = waveform[
        0,
        0,
    ].detach().cpu().numpy()

    waveform_np = np.clip(
        waveform_np,
        -1.0,
        1.0,
    )

    # Write WAV using Python's wave module.

    import wave

    pcm = (
        waveform_np * 32767.0
    ).astype(
        np.int16
    )

    output_wav = (
        OUTPUT_DIR
        / "generated_speech.wav"
    )

    with wave.open(
        str(output_wav),
        "wb",
    ) as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(
            SAMPLE_RATE
        )
        wf.writeframes(
            pcm.tobytes()
        )

    print(
        f"Generated audio:"
    )

    print(
        f"  {output_wav}"
    )

    # ========================================================================
    # STEP 7 — PIPELINE SUMMARY
    # ========================================================================

    print()
    print("=" * 72)
    print("STEP 7 — COMPLETE PIPELINE SUMMARY")
    print("=" * 72)

    print(
        f"""
INPUT TEXT
----------
{text}

        ↓

TOKENIZATION
-------------
{len(token_ids)} tokens

        ↓

EMBEDDING
---------
{tuple(
    model.embedding(
        token_tensor
    ).shape
)}

        ↓

DURATION PREDICTION
-------------------
{tuple(
    predicted_log_durations.shape
)}

        ↓

LENGTH REGULATION
-----------------
{tuple(
    frame_features.shape
)}

        ↓

ACOUSTIC MODEL
--------------
{tuple(
    mel.shape
)}

        ↓

NEURAL VOCODER
--------------
{tuple(
    waveform.shape
)}

        ↓

WAVEFORM
--------
{waveform_length} samples

        ↓

AUDIO
-----
{output_wav}
"""
    )

    # ========================================================================
    # SAVE MODEL
    # ========================================================================

    checkpoint = (
        OUTPUT_DIR
        / "lesson62_end_to_end_tts.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "symbols": tokenizer.symbols,
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
        },
        checkpoint,
    )

    print(
        f"Checkpoint:"
    )

    print(
        f"  {checkpoint}"
    )

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print("=" * 72)
    print("LESSON 62 COMPLETE")
    print("=" * 72)

    print()
    print(
        "You have now connected:"
    )

    print(
        "    Text"
    )

    print(
        "      ↓"
    )

    print(
        "    Tokens"
    )

    print(
        "      ↓"
    )

    print(
        "    Duration"
    )

    print(
        "      ↓"
    )

    print(
        "    Length Regulation"
    )

    print(
        "      ↓"
    )

    print(
        "    Acoustic Model"
    )

    print(
        "      ↓"
    )

    print(
        "    Mel"
    )

    print(
        "      ↓"
    )

    print(
        "    Neural Vocoder"
    )

    print(
        "      ↓"
    )

    print(
        "    Waveform"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The current model is an architectural integration demo."
    )

    print(
        "The weights are randomly initialized, so the generated audio"
    )

    print(
        "will NOT yet be intelligible speech."
    )

    print()
    print(
        "The next lessons should focus on turning this integrated"
    )

    print(
        "pipeline into a properly trained TTS system."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()