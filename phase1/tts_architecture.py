"""
File: phase1_lesson14_tts_architecture.py

PHASE 1 — LESSON 14
FOUNDATIONS OF TEXT-TO-SPEECH ARCHITECTURE
============================================================

PURPOSE
-------
Build a miniature non-autoregressive TTS architecture using PyTorch.

IMPLEMENTED COMPONENTS
----------------------
1. Phoneme embedding
2. Transformer text encoder
3. Phoneme duration predictor
4. Length regulator
5. Acoustic decoder
6. Mel spectrogram prediction
7. Synthetic training dataset
8. Training loop
9. Inference demonstration
10. Visualization

IMPORTANT
---------
This is an educational TTS architecture.

The training data contains synthetic acoustic targets generated
from phoneme IDs. It is NOT real human speech.

Consequently:
    - The model learns the architecture.
    - The model learns a synthetic phoneme-to-feature mapping.
    - It does not learn real pronunciation.
    - It does not produce natural human speech.
    - It does not include a neural vocoder.

The next development stages will replace synthetic targets
with real speech recordings and real Mel spectrograms.

============================================================
1. TTS ARCHITECTURE
============================================================

Text
  |
  v
Phoneme IDs
  |
  v
Embedding
  |
  v
Transformer Encoder
  |
  +--------------------------+
  |                          |
  v                          v
Duration Predictor      Acoustic Information
  |                          |
  v                          |
Predicted Durations          |
  |                          |
  +------------+-------------+
               |
               v
        Length Regulator
               |
               v
       Frame-Level Features
               |
               v
        Acoustic Decoder
               |
               v
         Mel Spectrogram
               |
               v
         Neural Vocoder
               |
               v
           Waveform

============================================================
2. TENSOR DIMENSIONS
============================================================

Let:

    B = batch size
    N = number of phonemes
    D = hidden dimension
    T = number of acoustic frames
    M = number of Mel channels

Phoneme IDs:

    [B, N]

Embedding:

    [B, N, D]

Text encoder:

    [B, N, D]

Duration predictor:

    [B, N]

Length regulator:

    [B, T, D]

Acoustic decoder:

    [B, T, M]

============================================================
3. DURATION PREDICTION
============================================================

The duration predictor estimates:

    d_i = number of acoustic frames for phoneme i

We predict log durations:

    log(1 + d_i)

The training loss is:

    L_duration =
        MSE(
            predicted_log_duration,
            target_log_duration
        )

During inference:

    predicted_duration =
        round(exp(predicted_log_duration) - 1)

The result is clamped to a minimum of one frame.

============================================================
4. LENGTH REGULATOR
============================================================

For each phoneme representation h_i:

    repeat h_i, d_i times

For example:

    phonemes:  A  B  C

    durations: 2  3  1

    expanded:
       A A B B B C

The number of output frames is:

    T = sum(d_i)

This operation connects phoneme-level representations
to frame-level acoustic representations.

============================================================
5. SYNTHETIC TRAINING TARGETS
============================================================

For this lesson, each phoneme ID has a synthetic Mel template.

The target Mel spectrogram is constructed by repeating
the template according to a randomly generated duration.

This allows us to train and test the architecture without
collecting speech recordings yet.

It is NOT a realistic speech-generation objective.

============================================================
REQUIREMENTS
============================================================

    pip install numpy matplotlib torch

RUN
---

    python phase1_lesson14_tts_architecture.py

============================================================
"""

import argparse
import math
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOCAB_SIZE = 32

HIDDEN_DIM = 128

NUM_HEADS = 4

NUM_ENCODER_LAYERS = 2

MEL_CHANNELS = 80

MIN_DURATION = 2

MAX_DURATION = 8

NUM_TRAINING_SAMPLES = 100

NUM_EPOCHS = 50

LEARNING_RATE = 0.001


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# SYNTHETIC PHONEME VOCABULARY
# ============================================================

PHONEME_NAMES = [
    "<pad>",
    "a", "e", "i", "o", "u",
    "h", "l", "m", "n", "p",
    "r", "s", "t", "w", "d",
    "k", "g", "f", "v", "b",
    "ə", "ɜ", "ʊ", "ʃ", "z",
    "θ", "ŋ", "j", "x", "q", "ɑ"
]

assert len(PHONEME_NAMES) == VOCAB_SIZE


# Example input:
#
#     h e l o
#
# The actual text frontend and pronunciation conversion
# are not implemented in this lesson.
#
# We directly provide phoneme IDs.


def encode_phonemes(phone_names):

    phone_to_id = {
        phone: index
        for index, phone in enumerate(PHONEME_NAMES)
    }

    result = []

    for phone in phone_names:

        if phone not in phone_to_id:

            raise ValueError(
                f"Unknown phoneme: {phone}"
            )

        result.append(
            phone_to_id[phone]
        )

    return result


# ============================================================
# SYNTHETIC MEL TEMPLATE GENERATOR
# ============================================================

def create_synthetic_mel_templates(
    vocab_size,
    mel_channels,
    seed=SEED
):
    """
    Create a fixed synthetic acoustic template for each phoneme.

    Shape:

        [vocab_size, mel_channels]

    Each phoneme has a different synthetic feature vector.

    These templates are NOT derived from actual speech.

    They simply provide a learnable mapping for testing
    the neural architecture.
    """

    generator = torch.Generator()

    generator.manual_seed(seed)

    templates = torch.randn(
        vocab_size,
        mel_channels,
        generator=generator
    )

    templates = templates * 0.5

    # Padding template is zero.
    templates[0] = 0.0

    return templates


# ============================================================
# SYNTHETIC DATASET
# ============================================================

class SyntheticTTSDataset:

    def __init__(
        self,
        num_samples,
        templates,
        min_length=4,
        max_length=8,
    ):

        self.samples = []

        vocab_size = templates.shape[0]

        for _ in range(num_samples):

            sequence_length = random.randint(
                min_length,
                max_length
            )

            # IDs 1..VOCAB_SIZE-1.
            phonemes = torch.randint(
                low=1,
                high=vocab_size,
                size=(sequence_length,)
            )

            durations = torch.randint(
                low=MIN_DURATION,
                high=MAX_DURATION + 1,
                size=(sequence_length,)
            )

            mel_frames = []

            for phone_id, duration in zip(
                phonemes,
                durations
            ):

                template = templates[phone_id]

                # Repeat the phoneme's synthetic acoustic vector.
                repeated = template.unsqueeze(0).repeat(
                    int(duration),
                    1
                )

                # Add small noise to the synthetic target.
                noise = torch.randn_like(repeated) * 0.015

                repeated = repeated + noise

                mel_frames.append(repeated)

            mel_target = torch.cat(
                mel_frames,
                dim=0
            )

            self.samples.append({
                "phonemes": phonemes,
                "durations": durations.float(),
                "mel": mel_target.float(),
            })

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        return self.samples[index]


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Transformer encoders need positional information because
    self-attention alone does not inherently represent sequence
    order.

    PE(pos, 2i) =
        sin(pos / 10000^(2i/D))

    PE(pos, 2i+1) =
        cos(pos / 10000^(2i/D))
    """

    def __init__(
        self,
        dimension,
        max_length=2000
    ):

        super().__init__()

        position = torch.arange(
            max_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                dimension,
                2
            ) * (-math.log(10000.0) / dimension)
        )

        encoding = torch.zeros(
            max_length,
            dimension
        )

        encoding[:, 0::2] = torch.sin(
            position * div_term
        )

        encoding[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0)
        )

    def forward(self, x):

        sequence_length = x.size(1)

        return (
            x + self.encoding[:, :sequence_length]
        )


# ============================================================
# TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):
    """
    Input:
        phoneme IDs [B, N]

    Output:
        hidden representations [B, N, D]

    Components:
        - Embedding
        - Positional encoding
        - Transformer encoder
    """

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=0
        )

        self.position = PositionalEncoding(
            hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, phoneme_ids):

        x = self.embedding(
            phoneme_ids
        )

        x = self.position(x)

        padding_mask = (
            phoneme_ids == 0
        )

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        return x


# ============================================================
# DURATION PREDICTOR
# ============================================================

class DurationPredictor(nn.Module):
    """
    Predict log(1 + duration) for every phoneme.

    Input:
        [B, N, D]

    Output:
        [B, N]

    This educational predictor uses a small MLP.

    More advanced TTS systems may use convolutional blocks.
    """

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim // 2,
                1
            )
        )

    def forward(self, hidden):

        log_duration = self.network(
            hidden
        ).squeeze(-1)

        return log_duration


# ============================================================
# LENGTH REGULATOR
# ============================================================

class LengthRegulator(nn.Module):
    """
    Expand phoneme representations into frame-level vectors.

    Example:

        phonemes = [A, B, C]

        durations = [2, 3, 1]

        output = [A, A, B, B, B, C]

    Input:
        hidden [B, N, D]
        durations [B, N]

    Output:
        List of tensors, one per batch item.

    We use a list because each item may have a different
    number of acoustic frames.
    """

    def forward(
        self,
        hidden,
        durations
    ):

        outputs = []

        batch_size = hidden.size(0)

        for batch_index in range(batch_size):

            expanded = []

            sequence_length = hidden.size(1)

            for phone_index in range(sequence_length):

                duration = int(
                    durations[
                        batch_index,
                        phone_index
                    ].item()
                )

                if duration <= 0:
                    continue

                phone_vector = hidden[
                    batch_index,
                    phone_index
                ]

                repeated = phone_vector.unsqueeze(0).repeat(
                    duration,
                    1
                )

                expanded.append(repeated)

            if len(expanded) == 0:

                raise RuntimeError(
                    "Length regulator produced an empty sequence."
                )

            outputs.append(
                torch.cat(
                    expanded,
                    dim=0
                )
            )

        return outputs


# ============================================================
# ACOUSTIC DECODER
# ============================================================

class AcousticDecoder(nn.Module):
    """
    Convert frame-level hidden representations into Mel features.

    Input:
        [B, T, D]

    Output:
        [B, T, MEL_CHANNELS]

    The decoder operates independently on each frame in this
    miniature implementation.

    A production acoustic decoder should model temporal
    dependencies between neighboring acoustic frames.
    """

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        mel_channels=MEL_CHANNELS
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                mel_channels
            )
        )

    def forward(self, frame_hidden):

        return self.network(
            frame_hidden
        )


# ============================================================
# COMPLETE MINIATURE TTS MODEL
# ============================================================

class MiniTTS(nn.Module):
    """
    Non-autoregressive TTS architecture.

    Training:
        Ground-truth durations are supplied to the length regulator.

    Inference:
        Predicted durations are used.

    This is teacher forcing for the duration expansion stage:
    the acoustic decoder receives correctly expanded training
    sequences instead of noisy predicted durations.
    """

    def __init__(self):

        super().__init__()

        self.encoder = TextEncoder()

        self.duration_predictor = DurationPredictor()

        self.length_regulator = LengthRegulator()

        self.decoder = AcousticDecoder()

    def encode(self, phoneme_ids):

        hidden = self.encoder(
            phoneme_ids
        )

        return hidden

    def predict_durations(self, hidden):

        log_durations = self.duration_predictor(
            hidden
        )

        predicted_durations = torch.round(
            torch.expm1(
                torch.clamp(
                    log_durations,
                    min=0.0,
                    max=math.log(100.0)
                )
            )
        )

        predicted_durations = predicted_durations.clamp(
            min=1,
            max=30
        )

        return (
            log_durations,
            predicted_durations.long()
        )

    def forward(
        self,
        phoneme_ids,
        target_durations
    ):

        hidden = self.encode(
            phoneme_ids
        )

        predicted_log_durations = (
            self.duration_predictor(hidden)
        )

        expanded = self.length_regulator(
            hidden,
            target_durations
        )

        mel_predictions = []

        for frame_hidden in expanded:

            mel = self.decoder(
                frame_hidden.unsqueeze(0)
            ).squeeze(0)

            mel_predictions.append(mel)

        return (
            predicted_log_durations,
            mel_predictions
        )

    @torch.no_grad()
    def inference(self, phoneme_ids):

        self.eval()

        hidden = self.encode(
            phoneme_ids
        )

        log_durations, predicted_durations = (
            self.predict_durations(hidden)
        )

        expanded = self.length_regulator(
            hidden,
            predicted_durations
        )

        mel_outputs = []

        for frame_hidden in expanded:

            mel = self.decoder(
                frame_hidden.unsqueeze(0)
            ).squeeze(0)

            mel_outputs.append(mel)

        return {
            "predicted_log_durations": log_durations,
            "predicted_durations": predicted_durations,
            "mel": mel_outputs[0],
        }


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    dataset,
    epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE
):

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "total": [],
        "duration": [],
        "mel": [],
    }

    print("\nStarting synthetic TTS training...")

    for epoch in range(epochs):

        random_indices = list(
            range(len(dataset))
        )

        random.shuffle(
            random_indices
        )

        total_loss_value = 0.0

        duration_loss_value = 0.0

        mel_loss_value = 0.0

        for index in random_indices:

            sample = dataset[index]

            phonemes = sample["phonemes"].unsqueeze(0).to(
                DEVICE
            )

            durations = sample["durations"].unsqueeze(0).to(
                DEVICE
            )

            target_mel = sample["mel"].to(
                DEVICE
            )

            optimizer.zero_grad()

            (
                predicted_log_durations,
                predicted_mel
            ) = model(
                phonemes,
                durations.long()
            )

            # ------------------------------------------------
            # DURATION LOSS
            # ------------------------------------------------

            target_log_durations = torch.log1p(
                durations
            )

            duration_loss = F.mse_loss(
                predicted_log_durations,
                target_log_durations
            )

            # ------------------------------------------------
            # MEL RECONSTRUCTION LOSS
            # ------------------------------------------------

            predicted_mel = predicted_mel[0]

            if predicted_mel.shape != target_mel.shape:

                raise RuntimeError(
                    "Predicted and target Mel shapes differ."
                )

            mel_loss = F.mse_loss(
                predicted_mel,
                target_mel
            )

            # ------------------------------------------------
            # TOTAL LOSS
            # ------------------------------------------------

            total_loss = (
                duration_loss
                + mel_loss
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss_value += total_loss.item()

            duration_loss_value += duration_loss.item()

            mel_loss_value += mel_loss.item()

        number_of_samples = len(dataset)

        total_loss_value /= number_of_samples

        duration_loss_value /= number_of_samples

        mel_loss_value /= number_of_samples

        history["total"].append(
            total_loss_value
        )

        history["duration"].append(
            duration_loss_value
        )

        history["mel"].append(
            mel_loss_value
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:

            print(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"Total: {total_loss_value:.5f} | "
                f"Duration: {duration_loss_value:.5f} | "
                f"Mel: {mel_loss_value:.5f}"
            )

    return history


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    model,
    phone_names
):

    phoneme_ids = encode_phonemes(
        phone_names
    )

    input_tensor = torch.tensor(
        phoneme_ids,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    output = model.inference(
        input_tensor
    )

    predicted_durations = (
        output["predicted_durations"]
        .squeeze(0)
        .cpu()
        .numpy()
    )

    predicted_mel = (
        output["mel"]
        .cpu()
        .numpy()
    )

    return (
        phoneme_ids,
        predicted_durations,
        predicted_mel
    )


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_results(
    history,
    phone_names,
    target_durations,
    predicted_durations,
    mel
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10)
    )

    # --------------------------------------------------------
    # 1. Training loss
    # --------------------------------------------------------

    axes[0, 0].plot(
        history["total"],
        label="Total loss"
    )

    axes[0, 0].plot(
        history["duration"],
        label="Duration loss"
    )

    axes[0, 0].plot(
        history["mel"],
        label="Mel loss"
    )

    axes[0, 0].set_title(
        "Training losses"
    )

    axes[0, 0].set_xlabel(
        "Epoch"
    )

    axes[0, 0].set_ylabel(
        "Loss"
    )

    axes[0, 0].legend()

    axes[0, 0].grid(True)

    # --------------------------------------------------------
    # 2. Duration comparison
    # --------------------------------------------------------

    x = np.arange(
        len(phone_names)
    )

    width = 0.35

    axes[0, 1].bar(
        x - width / 2,
        target_durations,
        width,
        label="Synthetic target"
    )

    axes[0, 1].bar(
        x + width / 2,
        predicted_durations,
        width,
        label="Model prediction"
    )

    axes[0, 1].set_xticks(x)

    axes[0, 1].set_xticklabels(
        phone_names
    )

    axes[0, 1].set_title(
        "Phoneme duration prediction"
    )

    axes[0, 1].set_xlabel(
        "Phoneme"
    )

    axes[0, 1].set_ylabel(
        "Frames"
    )

    axes[0, 1].legend()

    axes[0, 1].grid(
        axis="y"
    )

    # --------------------------------------------------------
    # 3. Predicted Mel spectrogram
    # --------------------------------------------------------

    axes[1, 0].imshow(
        mel.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )

    axes[1, 0].set_title(
        "Predicted synthetic Mel features"
    )

    axes[1, 0].set_xlabel(
        "Acoustic frame"
    )

    axes[1, 0].set_ylabel(
        "Mel channel"
    )

    # --------------------------------------------------------
    # 4. Predicted duration timeline
    # --------------------------------------------------------

    current_frame = 0

    for phone, duration in zip(
        phone_names,
        predicted_durations
    ):

        axes[1, 1].barh(
            y=0,
            width=int(duration),
            left=current_frame,
            height=0.4
        )

        center = (
            current_frame + int(duration) / 2
        )

        axes[1, 1].text(
            center,
            0,
            phone,
            ha="center",
            va="center"
        )

        current_frame += int(duration)

    axes[1, 1].set_title(
        "Predicted phoneme timeline"
    )

    axes[1, 1].set_xlabel(
        "Acoustic frame"
    )

    axes[1, 1].set_yticks([])

    axes[1, 1].grid(
        axis="x"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=NUM_TRAINING_SAMPLES
    )

    parser.add_argument(
        "--no-plot",
        action="store_true"
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 70)

    print("PHASE 1 — LESSON 14")

    print("MINIATURE NEURAL TTS ARCHITECTURE")

    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    print(f"Vocabulary size: {VOCAB_SIZE}")

    print(f"Hidden dimension: {HIDDEN_DIM}")

    print(f"Mel channels: {MEL_CHANNELS}")

    # --------------------------------------------------------
    # STEP 1: Create synthetic Mel templates
    # --------------------------------------------------------

    templates = create_synthetic_mel_templates(
        VOCAB_SIZE,
        MEL_CHANNELS
    )

    # --------------------------------------------------------
    # STEP 2: Create synthetic dataset
    # --------------------------------------------------------

    dataset = SyntheticTTSDataset(
        num_samples=args.samples,
        templates=templates
    )

    print(
        f"\nTraining samples: {len(dataset)}"
    )

    # --------------------------------------------------------
    # STEP 3: Build model
    # --------------------------------------------------------

    model = MiniTTS().to(
        DEVICE
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: {total_parameters:,}"
    )

    # --------------------------------------------------------
    # STEP 4: Train
    # --------------------------------------------------------

    history = train_model(
        model,
        dataset,
        epochs=args.epochs
    )

    # --------------------------------------------------------
    # STEP 5: Inference
    # --------------------------------------------------------

    # Simplified phoneme sequence for "hello".
    #
    # This is supplied directly as phonemes.
    # A real TTS system needs a text frontend.

    phone_names = [
        "h",
        "e",
        "l",
        "o"
    ]

    (
        phoneme_ids,
        predicted_durations,
        predicted_mel
    ) = run_inference(
        model,
        phone_names
    )

    print("\nINFERENCE RESULTS")

    print("-" * 70)

    print("Phonemes:", phone_names)

    print("Phoneme IDs:", phoneme_ids)

    print(
        "Predicted durations:",
        predicted_durations
    )

    print(
        "Total predicted frames:",
        int(np.sum(predicted_durations))
    )

    print(
        "Predicted Mel shape:",
        predicted_mel.shape
    )

    print(
        "Predicted duration:",
        np.sum(predicted_durations) * 0.01,
        "seconds"
    )

    # --------------------------------------------------------
    # STEP 6: Visualization
    # --------------------------------------------------------

    if not args.no_plot:

        # For comparison only, use an example duration target.
        # This is not a measured alignment from real speech.

        target_durations = np.array(
            [4, 5, 3, 8]
        )

        visualize_results(
            history,
            phone_names,
            target_durations,
            predicted_durations,
            predicted_mel
        )

    print("\n" + "=" * 70)

    print("LESSON 14 COMPLETE")

    print("=" * 70)

    print(
        "\nImportant: the generated features are synthetic."
    )

    print(
        "The next stage will introduce real acoustic features "
        "and neural vocoder fundamentals."
    )


if __name__ == "__main__":

    main()