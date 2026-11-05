"""
File: phase2/lesson53_length_regulator.py

======================================================================
PHASE 2 — LESSON 53
LENGTH REGULATOR FROM SCRATCH
======================================================================

PURPOSE
-------
In Lesson 52, we implemented a Transformer-based phoneme encoder
and a duration predictor.

However, the encoder produces one representation per phoneme,
while an acoustic model needs one representation per mel frame.

The Length Regulator solves this problem.

It expands phoneme-level hidden representations into frame-level
representations using phoneme durations.

ARCHITECTURE
------------

Phoneme IDs
    |
    v
Transformer Phoneme Encoder
    |
    v
Phoneme Hidden States
    |
    +----------------------------+
    |                            |
    v                            v
Duration Predictor          Length Regulator
    |                            ^
    v                            |
Predicted Durations -------------+
                                 |
                                 v
                       Frame-Level Hidden States
                                 |
                                 v
                       Acoustic Decoder
                                 |
                                 v
                         Mel Spectrogram


WHAT YOU LEARN
--------------

STEP 1 — Length Regulator
    Expand each phoneme representation according to its duration.

STEP 2 — Variable-Length Batching
    Handle different phoneme and frame sequence lengths.

STEP 3 — Duration Integration
    Connect the duration predictor from Lesson 52.

STEP 4 — Acoustic Decoder
    Transform frame-level hidden states into mel features.

STEP 5 — Training
    Train duration prediction and frame-level acoustic reconstruction.

STEP 6 — Inference
    Generate frame-level representations using predicted durations.

STEP 7 — Numerical Verification
    Verify repeat operations, zero-duration handling, padding,
    and output sequence lengths.

REAL-WORLD APPLICATION
----------------------

A voice-cloning system receives:

    "Hello, how are you?"

After phonemization:

    HH EH L OW ...

The duration predictor estimates how long each phoneme should last.

The Length Regulator expands these representations into a
sequence suitable for mel-spectrogram generation.

This is a core component of FastSpeech-style non-autoregressive TTS.

IMPORTANT
---------

This lesson uses synthetic training data to understand the architecture.

The next stage will introduce real phoneme-to-audio alignment
and real duration supervision.

The synthetic acoustic targets are NOT real speech recordings.
======================================================================
"""

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ================================================================
# 0. CONFIGURATION
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VOCAB_SIZE = 80

PAD_ID = 0

D_MODEL = 128

N_HEADS = 4

NUM_ENCODER_LAYERS = 2

MEL_DIM = 80

BATCH_SIZE = 32

TRAIN_SAMPLES = 800

VAL_SAMPLES = 150

EPOCHS = 15

LEARNING_RATE = 1e-3

MAX_PHONEME_LENGTH = 32

MAX_DURATION = 8

MAX_INFER_DURATION = 12

OUTPUT_DIR = "lesson53_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("PHASE 2 — LESSON 53: LENGTH REGULATOR")
print("=" * 72)

print("Device:", DEVICE)
print("Vocabulary:", VOCAB_SIZE)
print("Hidden dimension:", D_MODEL)
print("Mel dimension:", MEL_DIM)


# ================================================================
# STEP 1 — NUMERICAL LENGTH REGULATOR
# ================================================================

class LengthRegulator(nn.Module):

    """
    Expand phoneme-level representations into frame-level sequences.

    Input:
        hidden_states: [B, N, D]
        durations:     [B, N]

    Output:
        expanded:      [B, T, D]
        lengths:       [B]

    Each phoneme representation is repeated according to
    its corresponding duration.

    Example:

        Input hidden states:

            A B C D

        Durations:

            2 3 1 2

        Output:

            A A B B B C D D

    A duration of zero removes that phoneme from the expanded
    sequence.

    Padding is added only after expansion so that all samples
    in a batch have the same frame length.
    """

    def forward(self, hidden_states, durations):

        batch_size, num_phonemes, hidden_dim = hidden_states.shape

        expanded_sequences = []

        lengths = []

        for batch_idx in range(batch_size):

            phoneme_states = hidden_states[batch_idx]

            phoneme_durations = durations[batch_idx]

            # Repeat each phoneme representation.
            expanded = torch.repeat_interleave(
                phoneme_states,
                phoneme_durations,
                dim=0
            )

            expanded_sequences.append(expanded)

            lengths.append(expanded.size(0))

        # Maximum expanded length in this batch.
        max_length = max(max(lengths), 1)

        output = hidden_states.new_zeros(
            batch_size,
            max_length,
            hidden_dim
        )

        for batch_idx, expanded in enumerate(expanded_sequences):

            length = expanded.size(0)

            if length > 0:

                output[batch_idx, :length] = expanded

        lengths = torch.tensor(
            lengths,
            dtype=torch.long,
            device=hidden_states.device
        )

        return output, lengths


# ================================================================
# STEP 2 — NUMERICAL UNIT TESTS
# ================================================================

def test_length_regulator():

    print("\nSTEP 2 — LENGTH REGULATOR TESTS")

    regulator = LengthRegulator()

    # ------------------------------------------------------------
    # TEST 1: Basic expansion
    # ------------------------------------------------------------

    hidden = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0]]]
    )

    durations = torch.tensor(
        [[2, 3, 1, 2]]
    )

    expanded, lengths = regulator(hidden, durations)

    expected = torch.tensor(
        [1., 1., 2., 2., 2., 3., 4., 4.]
    )

    assert torch.equal(
        expanded[0, :8, 0],
        expected
    )

    assert lengths.item() == 8

    print("PASS: Basic duration expansion")

    # ------------------------------------------------------------
    # TEST 2: Zero duration
    # ------------------------------------------------------------

    durations = torch.tensor(
        [[2, 0, 3, 1]]
    )

    expanded, lengths = regulator(hidden, durations)

    expected = torch.tensor(
        [1., 1., 3., 3., 3., 4.]
    )

    assert torch.equal(
        expanded[0, :6, 0],
        expected
    )

    assert lengths.item() == 6

    print("PASS: Zero-duration handling")

    # ------------------------------------------------------------
    # TEST 3: Variable-length batch
    # ------------------------------------------------------------

    hidden = torch.tensor(
        [
            [[1.], [2.], [3.]],
            [[4.], [5.], [6.]]
        ]
    )

    durations = torch.tensor(
        [
            [2, 2, 2],
            [1, 1, 0]
        ]
    )

    expanded, lengths = regulator(hidden, durations)

    assert expanded.shape == (2, 6, 1)

    assert lengths.tolist() == [6, 2]

    assert torch.all(expanded[1, 2:] == 0)

    print("PASS: Variable-length batch and padding")

    # ------------------------------------------------------------
    # TEST 4: Gradient propagation
    # ------------------------------------------------------------

    hidden = torch.randn(
        2, 4, 8,
        requires_grad=True
    )

    durations = torch.tensor(
        [
            [2, 1, 3, 1],
            [1, 2, 1, 2]
        ]
    )

    expanded, lengths = regulator(hidden, durations)

    loss = expanded.sum()

    loss.backward()

    assert hidden.grad is not None

    print("PASS: Gradient propagation")

    print("All numerical tests passed!\n")


test_length_regulator()


# ================================================================
# STEP 3 — SYNTHETIC DATASET
# ================================================================

class SyntheticTTSDataset(Dataset):

    """
    Generate synthetic phoneme sequences and duration labels.

    The duration labels simulate phoneme alignment.

    Every phoneme receives a duration between 1 and MAX_DURATION.

    A fixed teacher table provides synthetic acoustic targets.

    Each phoneme's acoustic feature is repeated according
    to its duration.

    This allows us to train the Length Regulator and acoustic
    decoder without downloading a real speech dataset.
    """

    def __init__(
        self,
        num_samples,
        teacher_table
    ):

        self.num_samples = num_samples

        self.teacher_table = teacher_table

        self.samples = []

        for _ in range(num_samples):

            sequence_length = random.randint(
                6,
                MAX_PHONEME_LENGTH
            )

            phoneme_ids = torch.randint(
                2,
                VOCAB_SIZE,
                (sequence_length,)
            )

            durations = torch.randint(
                1,
                MAX_DURATION + 1,
                (sequence_length,)
            )

            # Simulate occasional short phonemes.
            short_mask = torch.rand(sequence_length) < 0.15

            durations[short_mask] = 1

            self.samples.append(
                (phoneme_ids, durations)
            )

    def __len__(self):

        return self.num_samples

    def __getitem__(self, idx):

        phoneme_ids, durations = self.samples[idx]

        # Generate synthetic frame-level mel targets.
        phoneme_features = self.teacher_table[phoneme_ids]

        mel_target = torch.repeat_interleave(
            phoneme_features,
            durations,
            dim=0
        )

        return {
            "phoneme_ids": phoneme_ids,
            "durations": durations,
            "mel_target": mel_target
        }


# ------------------------------------------------
# Collate variable-length samples
# ------------------------------------------------

def collate_fn(batch):

    batch_size = len(batch)

    max_phonemes = max(
        item["phoneme_ids"].size(0)
        for item in batch
    )

    max_frames = max(
        item["mel_target"].size(0)
        for item in batch
    )

    phoneme_ids = torch.full(
        (batch_size, max_phonemes),
        PAD_ID,
        dtype=torch.long
    )

    durations = torch.zeros(
        batch_size,
        max_phonemes,
        dtype=torch.long
    )

    mel_targets = torch.zeros(
        batch_size,
        max_frames,
        MEL_DIM
    )

    phoneme_lengths = []

    frame_lengths = []

    for i, item in enumerate(batch):

        n = item["phoneme_ids"].size(0)

        t = item["mel_target"].size(0)

        phoneme_ids[i, :n] = item["phoneme_ids"]

        durations[i, :n] = item["durations"]

        mel_targets[i, :t] = item["mel_target"]

        phoneme_lengths.append(n)

        frame_lengths.append(t)

    return {
        "phoneme_ids": phoneme_ids,
        "durations": durations,
        "mel_target": mel_targets,
        "phoneme_lengths": torch.tensor(phoneme_lengths),
        "frame_lengths": torch.tensor(frame_lengths)
    }


# ================================================================
# STEP 4 — TRANSFORMER PHONEME ENCODER
# ================================================================

class PhonemeEncoder(nn.Module):

    """
    Convert phoneme IDs into contextual hidden representations.

    Input:
        [B, N]

    Output:
        [B, N, D_MODEL]
    """

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL,
            padding_idx=PAD_ID
        )

        self.position_embedding = nn.Embedding(
            256,
            D_MODEL
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_ENCODER_LAYERS
        )

        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, phoneme_ids):

        batch_size, seq_length = phoneme_ids.shape

        positions = torch.arange(
            seq_length,
            device=phoneme_ids.device
        ).unsqueeze(0)

        x = self.embedding(phoneme_ids)

        x = x + self.position_embedding(positions)

        padding_mask = phoneme_ids.eq(PAD_ID)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        x = self.norm(x)

        return x


# ================================================================
# STEP 5 — DURATION PREDICTOR
# ================================================================

class DurationPredictor(nn.Module):

    """
    Predict log(1 + duration) for every phoneme.

    Input:
        [B, N, D_MODEL]

    Output:
        [B, N]

    We predict log durations instead of raw durations because
    phoneme durations have a positive, skewed distribution.

    During inference:

        duration = expm1(predicted_log_duration)

    This follows the approach introduced in Lesson 52.
    """

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            D_MODEL,
            D_MODEL,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            D_MODEL,
            D_MODEL,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.LayerNorm(D_MODEL)

        self.norm2 = nn.LayerNorm(D_MODEL)

        self.linear = nn.Linear(
            D_MODEL,
            1
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        # [B, N, D] -> [B, D, N]
        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = F.relu(x)

        x = x.transpose(1, 2)

        x = self.norm1(x)

        x = self.dropout(x)

        x = x.transpose(1, 2)

        x = self.conv2(x)

        x = F.relu(x)

        x = x.transpose(1, 2)

        x = self.norm2(x)

        x = self.dropout(x)

        output = self.linear(x).squeeze(-1)

        return output


# ================================================================
# STEP 6 — FRAME-LEVEL ACOUSTIC DECODER
# ================================================================

class AcousticDecoder(nn.Module):

    """
    Convert frame-level hidden representations into mel features.

    Input:
        [B, T, D_MODEL]

    Output:
        [B, T, MEL_DIM]

    This is a simplified acoustic decoder.

    A production TTS system would use a more sophisticated
    decoder to predict realistic mel spectrograms.
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                D_MODEL,
                D_MODEL * 2
            ),

            nn.GELU(),

            nn.LayerNorm(
                D_MODEL * 2
            ),

            nn.Linear(
                D_MODEL * 2,
                D_MODEL
            ),

            nn.GELU(),

            nn.Linear(
                D_MODEL,
                MEL_DIM
            )
        )

    def forward(self, x):

        return self.network(x)


# ================================================================
# STEP 7 — COMPLETE TTS MODEL
# ================================================================

class LengthRegulatorTTS(nn.Module):

    """
    Integrate:

        1. Transformer phoneme encoder
        2. Duration predictor
        3. Length regulator
        4. Acoustic decoder

    Two operation modes:

        TRAINING:
            Use ground-truth durations.

        INFERENCE:
            Use predicted durations.
    """

    def __init__(self):

        super().__init__()

        self.encoder = PhonemeEncoder()

        self.duration_predictor = DurationPredictor()

        self.length_regulator = LengthRegulator()

        self.acoustic_decoder = AcousticDecoder()

    def forward(
        self,
        phoneme_ids,
        durations=None,
        use_predicted_durations=False
    ):

        # --------------------------------------------------------
        # 1. Phoneme encoding
        # --------------------------------------------------------

        hidden_states = self.encoder(phoneme_ids)

        # --------------------------------------------------------
        # 2. Duration prediction
        # --------------------------------------------------------

        predicted_log_durations = self.duration_predictor(
            hidden_states
        )

        # --------------------------------------------------------
        # 3. Choose durations
        # --------------------------------------------------------

        if use_predicted_durations:

            predicted_durations = torch.expm1(
                predicted_log_durations
            )

            predicted_durations = torch.round(
                predicted_durations
            )

            predicted_durations = predicted_durations.clamp(
                min=1,
                max=MAX_INFER_DURATION
            ).long()

            predicted_durations = predicted_durations.masked_fill(
                phoneme_ids.eq(PAD_ID),
                0
            )

            used_durations = predicted_durations

        else:

            if durations is None:

                raise ValueError(
                    "Ground-truth durations are required during training."
                )

            used_durations = durations

        # --------------------------------------------------------
        # 4. Length regulation
        # --------------------------------------------------------

        expanded_states, frame_lengths = self.length_regulator(
            hidden_states,
            used_durations
        )

        # --------------------------------------------------------
        # 5. Acoustic decoding
        # --------------------------------------------------------

        mel_prediction = self.acoustic_decoder(
            expanded_states
        )

        return {
            "predicted_log_durations": predicted_log_durations,
            "used_durations": used_durations,
            "expanded_states": expanded_states,
            "frame_lengths": frame_lengths,
            "mel_prediction": mel_prediction
        }


# ================================================================
# STEP 8 — LOSS FUNCTIONS
# ================================================================

def duration_loss_function(
    predicted_log_durations,
    target_durations,
    phoneme_ids
):

    """
    Masked log-duration MSE.

    Padding positions must not contribute to the loss.
    """

    target_log_durations = torch.log1p(
        target_durations.float()
    )

    mask = phoneme_ids.ne(PAD_ID)

    loss = F.mse_loss(
        predicted_log_durations[mask],
        target_log_durations[mask]
    )

    return loss


def acoustic_loss_function(
    predicted_mel,
    target_mel,
    frame_lengths
):

    """
    Masked L1 acoustic reconstruction loss.
    """

    batch_size, max_frames, mel_dim = predicted_mel.shape

    positions = torch.arange(
        max_frames,
        device=predicted_mel.device
    ).unsqueeze(0)

    mask = positions < frame_lengths.unsqueeze(1)

    mask = mask.unsqueeze(-1).expand(
        batch_size,
        max_frames,
        mel_dim
    )

    loss = F.l1_loss(
        predicted_mel[mask],
        target_mel[mask]
    )

    return loss


# ================================================================
# STEP 9 — CREATE DATASETS
# ================================================================

print("\nCreating synthetic datasets...")

# Fixed synthetic acoustic teacher.
generator = torch.Generator().manual_seed(SEED)

teacher_table = torch.randn(
    VOCAB_SIZE,
    MEL_DIM,
    generator=generator
)

teacher_table[PAD_ID] = 0

train_dataset = SyntheticTTSDataset(
    TRAIN_SAMPLES,
    teacher_table
)

val_dataset = SyntheticTTSDataset(
    VAL_SAMPLES,
    teacher_table
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print("Training samples:", len(train_dataset))

print("Validation samples:", len(val_dataset))


# ================================================================
# STEP 10 — INITIALIZE MODEL
# ================================================================

model = LengthRegulatorTTS().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)

print(
    "\nModel parameters:",
    sum(p.numel() for p in model.parameters())
)


# ================================================================
# STEP 11 — TRAINING LOOP
# ================================================================

history = {
    "train_total": [],
    "train_duration": [],
    "train_acoustic": [],
    "val_total": [],
    "val_duration": [],
    "val_acoustic": []
}


def train_one_epoch():

    model.train()

    total_loss_sum = 0

    duration_loss_sum = 0

    acoustic_loss_sum = 0

    for batch in train_loader:

        phoneme_ids = batch["phoneme_ids"].to(DEVICE)

        durations = batch["durations"].to(DEVICE)

        mel_target = batch["mel_target"].to(DEVICE)

        frame_lengths = batch["frame_lengths"].to(DEVICE)

        optimizer.zero_grad()

        # Training uses ground-truth durations for regulation.
        outputs = model(
            phoneme_ids,
            durations=durations,
            use_predicted_durations=False
        )

        duration_loss = duration_loss_function(
            outputs["predicted_log_durations"],
            durations,
            phoneme_ids
        )

        acoustic_loss = acoustic_loss_function(
            outputs["mel_prediction"],
            mel_target,
            frame_lengths
        )

        total_loss = duration_loss + acoustic_loss

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss_sum += total_loss.item()

        duration_loss_sum += duration_loss.item()

        acoustic_loss_sum += acoustic_loss.item()

    num_batches = len(train_loader)

    return (
        total_loss_sum / num_batches,
        duration_loss_sum / num_batches,
        acoustic_loss_sum / num_batches
    )


# ================================================================
# STEP 12 — VALIDATION
# ================================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss_sum = 0

    duration_loss_sum = 0

    acoustic_loss_sum = 0

    for batch in val_loader:

        phoneme_ids = batch["phoneme_ids"].to(DEVICE)

        durations = batch["durations"].to(DEVICE)

        mel_target = batch["mel_target"].to(DEVICE)

        frame_lengths = batch["frame_lengths"].to(DEVICE)

        outputs = model(
            phoneme_ids,
            durations=durations,
            use_predicted_durations=False
        )

        duration_loss = duration_loss_function(
            outputs["predicted_log_durations"],
            durations,
            phoneme_ids
        )

        acoustic_loss = acoustic_loss_function(
            outputs["mel_prediction"],
            mel_target,
            frame_lengths
        )

        total_loss = duration_loss + acoustic_loss

        total_loss_sum += total_loss.item()

        duration_loss_sum += duration_loss.item()

        acoustic_loss_sum += acoustic_loss.item()

    num_batches = len(val_loader)

    return (
        total_loss_sum / num_batches,
        duration_loss_sum / num_batches,
        acoustic_loss_sum / num_batches
    )


# ================================================================
# STEP 13 — MAIN TRAINING
# ================================================================

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    train_total, train_dur, train_acoustic = train_one_epoch()

    val_total, val_dur, val_acoustic = validate()

    history["train_total"].append(train_total)

    history["train_duration"].append(train_dur)

    history["train_acoustic"].append(train_acoustic)

    history["val_total"].append(val_total)

    history["val_duration"].append(val_dur)

    history["val_acoustic"].append(val_acoustic)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train: {train_total:.5f} | "
        f"Val: {val_total:.5f} | "
        f"Duration: {val_dur:.5f} | "
        f"Acoustic: {val_acoustic:.5f}"
    )


# ================================================================
# STEP 14 — INFERENCE
# ================================================================

@torch.no_grad()
def inference(phoneme_ids):

    """
    Generate frame-level representations from phoneme IDs.

    Unlike training, inference does not require ground-truth
    durations.
    """

    model.eval()

    phoneme_ids = phoneme_ids.unsqueeze(0).to(DEVICE)

    outputs = model(
        phoneme_ids,
        use_predicted_durations=True
    )

    predicted_durations = outputs["used_durations"]

    expanded_states = outputs["expanded_states"]

    mel_prediction = outputs["mel_prediction"]

    frame_length = outputs["frame_lengths"].item()

    return {
        "durations": predicted_durations[0].cpu(),
        "expanded_states": expanded_states[0, :frame_length].cpu(),
        "mel": mel_prediction[0, :frame_length].cpu(),
        "frame_length": frame_length
    }


# Select one validation sample.
test_sample = val_dataset[0]

test_phoneme_ids = test_sample["phoneme_ids"]

ground_truth_durations = test_sample["durations"]

result = inference(test_phoneme_ids)

predicted_durations = result["durations"]

print("\n" + "=" * 72)

print("INFERENCE RESULTS")

print("=" * 72)

print("\nPhoneme IDs:")

print(test_phoneme_ids.tolist())

print("\nGround-truth durations:")

print(ground_truth_durations.tolist())

print("\nPredicted durations:")

print(predicted_durations.tolist())

print("\nGround-truth frame count:")

print(ground_truth_durations.sum().item())

print("\nPredicted frame count:")

print(result["frame_length"])

print("\nExpanded hidden representation:")

print(result["expanded_states"].shape)

print("\nPredicted mel spectrogram:")

print(result["mel"].shape)


# ================================================================
# STEP 15 — VISUALIZATION
# ================================================================

def visualize_training():

    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    plt.plot(
        epochs,
        history["train_total"],
        label="Train Total"
    )

    plt.plot(
        epochs,
        history["val_total"],
        label="Validation Total"
    )

    plt.title("Lesson 53 — Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "training_loss.png")
    )

    plt.show()


def visualize_duration_prediction():

    n = len(ground_truth_durations)

    x = np.arange(n)

    plt.figure(figsize=(13, 5))

    plt.bar(
        x - 0.2,
        ground_truth_durations.numpy(),
        width=0.4,
        label="Ground Truth"
    )

    plt.bar(
        x + 0.2,
        predicted_durations.numpy(),
        width=0.4,
        label="Predicted"
    )

    plt.title("Phoneme Duration Prediction")

    plt.xlabel("Phoneme Index")

    plt.ylabel("Duration (Frames)")

    plt.legend()

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "duration_comparison.png")
    )

    plt.show()


def visualize_length_regulation():

    """
    Visualize the phoneme expansion process.

    Each color represents a different phoneme ID.
    """

    phoneme_ids = test_phoneme_ids.numpy()

    durations = predicted_durations.numpy()

    frame_phoneme_ids = np.repeat(
        phoneme_ids,
        durations
    )

    plt.figure(figsize=(15, 4))

    plt.imshow(
        frame_phoneme_ids.reshape(1, -1),
        aspect="auto",
        interpolation="nearest"
    )

    plt.title(
        "Length Regulator — Phoneme-to-Frame Expansion"
    )

    plt.xlabel("Acoustic Frame")

    plt.ylabel("Expanded Sequence")

    plt.colorbar(
        label="Phoneme ID"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "length_regulation.png")
    )

    plt.show()


def visualize_mel():

    mel = result["mel"].numpy()

    plt.figure(figsize=(14, 5))

    plt.imshow(
        mel.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )

    plt.title(
        "Synthetic Mel Spectrogram Prediction"
    )

    plt.xlabel("Time Frame")

    plt.ylabel("Mel Frequency Bin")

    plt.colorbar(
        label="Feature Value"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "predicted_mel.png")
    )

    plt.show()


# ================================================================
# STEP 16 — SAVE MODEL
# ================================================================

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "lesson53_length_regulator.pt"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "mel_dim": MEL_DIM
        },
        "history": history
    },
    checkpoint_path
)

print("\nModel saved:", checkpoint_path)


# ================================================================
# STEP 17 — RUN VISUALIZATIONS
# ================================================================

visualize_training()

visualize_duration_prediction()

visualize_length_regulation()

visualize_mel()


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 72)

print("LESSON 53 COMPLETED")

print("=" * 72)

print("1. Implemented Length Regulator.")

print("2. Expanded phoneme representations into frame-level sequences.")

print("3. Implemented variable-length batch processing.")

print("4. Integrated the Transformer phoneme encoder.")

print("5. Integrated the Lesson 52 duration predictor.")

print("6. Implemented a frame-level acoustic decoder.")

print("7. Trained and validated the complete architecture.")

print("8. Generated predicted frame-level representations.")

print("\nOutput directory:", OUTPUT_DIR)

print("\nNEXT LESSON:")

print("LESSON 54 — REAL PHONEME-TO-AUDIO ALIGNMENT")

print("=" * 72)