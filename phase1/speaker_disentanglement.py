"""
=======================================================================
PHASE 1 — LESSON 26
SPEAKER DISENTANGLEMENT & CONTENT LEAKAGE DETECTION
=======================================================================

FILE
----
speaker_disentanglement.py

PURPOSE
-------
Teach the model to separate:

    SPEAKER IDENTITY
        from
    LINGUISTIC CONTENT

This is one of the most important ideas in voice cloning.

A speaker encoder should answer:

    "WHO is speaking?"

It should NOT strongly encode:

    "WHAT did the person say?"

This lesson demonstrates the problem experimentally and introduces
adversarial training using a Gradient Reversal Layer (GRL).

=======================================================================
WHAT YOU LEARN
=======================================================================

STEP 1 — SPEAKER EMBEDDING
--------------------------

A speaker encoder transforms an audio representation into:

    audio
      |
      v
    speaker encoder
      |
      v
    speaker embedding

The embedding should represent characteristics that remain relatively
stable across different utterances from the same speaker.

Examples:

    vocal tract characteristics
    pitch tendencies
    timbre
    resonance
    spectral characteristics

-----------------------------------------------------------------------

STEP 2 — CONTENT LEAKAGE
------------------------

Suppose a speaker says:

    "Hello"

and another recording says:

    "The robot is moving."

If the speaker embedding contains too much information about the
actual words, then a classifier can recover the content.

That is called:

    CONTENT LEAKAGE

Conceptually:

    audio
      |
      v
    speaker encoder
      |
      v
    speaker embedding
      |
      v
    content classifier
      |
      v
    "What was spoken?"

A strong content classifier means the speaker representation contains
information that ideally belongs to the linguistic/content pathway.

-----------------------------------------------------------------------

STEP 3 — GRADIENT REVERSAL
--------------------------

We introduce a Gradient Reversal Layer.

Forward:

    GRL(x) = x

Backward:

    dL/dx = -lambda * dL/dx

Therefore:

    content classifier
        tries to predict content

while:

    speaker encoder
        tries to prevent content prediction.

This creates an adversarial objective.

-----------------------------------------------------------------------

STEP 4 — TWO OBJECTIVES
-----------------------

The speaker encoder receives two competing learning signals.

Speaker objective:

    recognize the correct speaker.

Content adversarial objective:

    remove information that allows content prediction.

Therefore:

    Speaker embedding
          |
          +----> speaker classifier
          |          |
          |          +----> high accuracy
          |
          +----> GRL ----> content classifier
                             |
                             +----> low accuracy

-----------------------------------------------------------------------

STEP 5 — CONTENT ENCODER
------------------------

The linguistic information should have another path.

We therefore create:

    tokens
      |
      v
    content encoder
      |
      v
    content representation

The acoustic decoder receives:

    speaker representation
             +
    content representation
             |
             v
       acoustic decoder
             |
             v
       mel spectrogram

This is conceptually similar to the separation used in many modern
speech generation systems.

-----------------------------------------------------------------------

STEP 6 — LEAKAGE MEASUREMENT
----------------------------

We explicitly measure:

    speaker accuracy
    content accuracy

A useful disentangled representation should approximately produce:

    speaker accuracy → high

    content accuracy → near chance

For N content classes:

    chance accuracy = 1 / N

Example:

    8 content classes

    chance = 12.5%

If the content classifier reaches:

    80%

then substantial content information remains in the speaker
embedding.

-----------------------------------------------------------------------

REAL-WORLD EXAMPLE
-----------------------------------------------------------------------

Imagine one person provides:

    "Hello, how are you?"
    "The weather is nice today."
    "The robot picked up the cup."
    "Artificial intelligence is useful."

The speaker embedding should remain approximately:

    Speaker A
       ↓
    same identity

while the content representation changes:

    Hello...
       ↓
    Content A

    Weather...
       ↓
    Content B

    Robot...
       ↓
    Content C

    AI...
       ↓
    Content D

This separation is critical for voice cloning.

=======================================================================
MATHEMATICAL OBJECTIVE
=======================================================================

Let:

    z_s = speaker encoder(audio)

    z_c = content encoder(text)

Speaker classifier:

    P(speaker | z_s)

Content classifier:

    P(content | z_s)

With gradient reversal, the speaker encoder effectively minimizes:

    L_speaker - lambda * L_content

while the content classifier minimizes:

    L_content

The complete system therefore attempts to:

    maximize speaker information

while:

    minimizing recoverable content information.

The decoder learns:

    mel = Decoder(z_s, z_c)

=======================================================================
LESSON 26 EXPERIMENT
=======================================================================

We train TWO models:

    MODEL A
        Normal speaker encoder

    MODEL B
        Speaker encoder + Gradient Reversal

Then compare:

    speaker classification accuracy
    content classification accuracy

Expected result:

    MODEL A
        speaker accuracy = high
        content leakage = noticeable

    MODEL B
        speaker accuracy = high
        content leakage = reduced

This is the important experiment.

=======================================================================
REQUIREMENTS
=======================================================================

Python 3.10+

    pip install torch numpy matplotlib

No torchaudio required.

WAV output uses Python's standard-library wave module.

=======================================================================
OUTPUT FILES
=======================================================================

    lesson26_baseline_disentanglement.pt
    lesson26_adversarial_disentanglement.pt

    lesson26_training_curves.png
    lesson26_accuracy_comparison.png
    lesson26_embedding_projection.png

    lesson26_reference.wav

=======================================================================
NEXT LESSON
=======================================================================

PHASE 1 — LESSON 27

    Content Leakage Detection
    with a Dedicated Post-Hoc Probe

Then we will move toward:

    voice conditioning
        ↓
    flow matching
        ↓
    expressive generation
        ↓
    streaming TTS
        ↓
    complete Chatterbox-like architecture

=======================================================================
"""

from __future__ import annotations

import math
import random
import wave
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class Config:

    # --------------------------------------------------------------
    # Audio
    # --------------------------------------------------------------

    sample_rate: int = 16000

    n_fft: int = 512

    hop_length: int = 256

    win_length: int = 512

    n_mels: int = 80

    mel_frames: int = 32

    waveform_length: int = 8192

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    num_speakers: int = 8

    num_contents: int = 8

    samples_per_speaker_content: int = 6

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    speaker_embedding_dim: int = 64

    content_embedding_dim: int = 64

    hidden_dim: int = 128

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    batch_size: int = 32

    epochs: int = 15

    learning_rate: float = 2e-3

    gradient_reversal_strength: float = 1.0

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    seed: int = 42

    # --------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


CFG = Config()


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================================
# MEL FILTERBANK
# =====================================================================

def hz_to_mel(freq):

    return 2595.0 * math.log10(
        1.0 + freq / 700.0
    )


def mel_to_hz(mel):

    return 700.0 * (
        10.0 ** (mel / 2595.0)
        - 1.0
    )


def create_mel_filterbank():

    mel_min = hz_to_mel(20.0)

    mel_max = hz_to_mel(
        CFG.sample_rate / 2.0
    )

    mel_points = torch.linspace(
        mel_min,
        mel_max,
        CFG.n_mels + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    bins = torch.floor(
        (CFG.n_fft + 1)
        * hz_points
        / CFG.sample_rate
    ).long()

    filterbank = torch.zeros(
        CFG.n_mels,
        CFG.n_fft // 2 + 1,
    )

    for m in range(
        1,
        CFG.n_mels + 1,
    ):

        left = int(
            bins[m - 1]
        )

        center = int(
            bins[m]
        )

        right = int(
            bins[m + 1]
        )

        if center > left:

            filterbank[
                m - 1,
                left:center,
            ] = torch.linspace(
                0.0,
                1.0,
                center - left,
            )

        if right > center:

            filterbank[
                m - 1,
                center:right,
            ] = torch.linspace(
                1.0,
                0.0,
                right - center,
            )

    return filterbank


MEL_FILTERBANK = create_mel_filterbank()


# =====================================================================
# WAVEFORM TO MEL
# =====================================================================

def waveform_to_mel(
    waveform: torch.Tensor,
):

    window = torch.hann_window(
        CFG.win_length,
        device=waveform.device,
    )

    spec = torch.stft(
        waveform,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        win_length=CFG.win_length,
        window=window,
        return_complex=True,
        center=True,
    )

    magnitude = spec.abs()

    mel_filter = (
        MEL_FILTERBANK
        .to(waveform.device)
    )

    mel = torch.matmul(
        mel_filter,
        magnitude,
    )

    mel = torch.log(
        mel.clamp_min(1e-5)
    )

    mel = mel.unsqueeze(0)

    mel = F.interpolate(
        mel,
        size=CFG.mel_frames,
        mode="linear",
        align_corners=False,
    )

    return mel.squeeze(0)


# =====================================================================
# SYNTHETIC SPEAKER PROFILE
# =====================================================================

class SpeakerProfile:

    def __init__(
        self,
        speaker_id: int,
    ):

        rng = np.random.default_rng(
            CFG.seed
            + speaker_id * 1000
        )

        self.base_f0 = float(
            rng.uniform(
                100.0,
                220.0,
            )
        )

        self.harmonic_tilt = float(
            rng.uniform(
                0.20,
                0.60,
            )
        )

        self.brightness = float(
            rng.uniform(
                0.7,
                1.4,
            )
        )

        self.resonance = float(
            rng.uniform(
                0.8,
                1.3,
            )
        )


# =====================================================================
# CONTENT PROFILE
# =====================================================================

class ContentProfile:

    def __init__(
        self,
        content_id: int,
    ):

        rng = np.random.default_rng(
            50000
            + content_id * 100
        )

        self.modulation_frequency = float(
            rng.uniform(
                2.0,
                7.0,
            )
        )

        self.energy_pattern = float(
            rng.uniform(
                0.5,
                1.5,
            )
        )

        self.phrase_frequency = float(
            rng.uniform(
                1.0,
                3.0,
            )
        )


# =====================================================================
# SYNTHETIC SPEECH
# =====================================================================

def generate_speech(
    speaker: SpeakerProfile,
    content: ContentProfile,
    seed: int,
):

    rng = np.random.default_rng(
        seed
    )

    n = CFG.waveform_length

    t = np.arange(n) / CFG.sample_rate

    # --------------------------------------------------------------
    # Speaker-dependent pitch
    # --------------------------------------------------------------

    pitch_curve = (
        speaker.base_f0
        * (
            1.0
            + 0.025
            * np.sin(
                2.0
                * np.pi
                * 1.5
                * t
            )
        )
    )

    phase = np.cumsum(
        2.0
        * np.pi
        * pitch_curve
        / CFG.sample_rate
    )

    waveform = np.zeros(
        n,
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # Speaker-dependent harmonic structure
    # --------------------------------------------------------------

    for harmonic in range(
        1,
        9,
    ):

        amplitude = (
            1.0
            / (
                harmonic
                ** speaker.harmonic_tilt
            )
        )

        amplitude *= (
            speaker.brightness
            ** (
                -0.12
                * harmonic
            )
        )

        waveform += (
            amplitude
            * np.sin(
                harmonic * phase
            )
        )

    # --------------------------------------------------------------
    # CONTENT-dependent modulation
    #
    # This intentionally creates information that belongs to CONTENT.
    # The disentanglement system must learn not to encode this strongly
    # inside the speaker embedding.
    # --------------------------------------------------------------

    content_modulation = (
        1.0
        + 0.18
        * np.sin(
            2.0
            * np.pi
            * content.modulation_frequency
            * t
        )
    )

    waveform *= content_modulation

    # --------------------------------------------------------------
    # Content-dependent envelope
    # --------------------------------------------------------------

    envelope = (
        0.72
        + 0.28
        * (
            0.5
            + 0.5
            * np.sin(
                2.0
                * np.pi
                * content.phrase_frequency
                * t
            )
        )
    )

    envelope *= (
        0.9
        + 0.1
        * content.energy_pattern
    )

    waveform *= envelope

    # --------------------------------------------------------------
    # Fade in / fade out
    # --------------------------------------------------------------

    fade = int(
        0.05
        * CFG.sample_rate
    )

    waveform[:fade] *= np.linspace(
        0.0,
        1.0,
        fade,
    )

    waveform[-fade:] *= np.linspace(
        1.0,
        0.0,
        fade,
    )

    # --------------------------------------------------------------
    # Small recording noise
    # --------------------------------------------------------------

    waveform += rng.normal(
        0.0,
        0.006,
        n,
    )

    waveform /= (
        np.max(
            np.abs(waveform)
        )
        + 1e-8
    )

    return torch.tensor(
        waveform,
        dtype=torch.float32,
    )


# =====================================================================
# DATASET
# =====================================================================

class DisentanglementDataset(
    Dataset
):

    def __init__(
        self,
        samples_per_combination,
    ):

        self.samples = []

        for speaker_id in range(
            CFG.num_speakers
        ):

            speaker = SpeakerProfile(
                speaker_id
            )

            for content_id in range(
                CFG.num_contents
            ):

                content = ContentProfile(
                    content_id
                )

                for sample_id in range(
                    samples_per_combination
                ):

                    waveform = generate_speech(
                        speaker,
                        content,
                        seed=(
                            speaker_id
                            * 100000
                            + content_id
                            * 1000
                            + sample_id
                        ),
                    )

                    mel = waveform_to_mel(
                        waveform
                    )

                    self.samples.append(
                        {
                            "mel": mel,
                            "waveform": waveform,
                            "speaker_id": speaker_id,
                            "content_id": content_id,
                        }
                    )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        item = self.samples[
            index
        ]

        return {
            "mel":
                item["mel"],

            "waveform":
                item["waveform"],

            "speaker_id":
                torch.tensor(
                    item["speaker_id"],
                    dtype=torch.long,
                ),

            "content_id":
                torch.tensor(
                    item["content_id"],
                    dtype=torch.long,
                ),
        }


# =====================================================================
# GRADIENT REVERSAL
# =====================================================================

class GradientReversalFunction(
    torch.autograd.Function
):

    @staticmethod
    def forward(
        ctx,
        x,
        lambda_value,
    ):

        ctx.lambda_value = (
            lambda_value
        )

        return x.view_as(x)

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):

        return (
            -ctx.lambda_value
            * grad_output,
            None,
        )


class GradientReversalLayer(
    nn.Module
):

    def __init__(
        self,
        lambda_value=1.0,
    ):

        super().__init__()

        self.lambda_value = (
            lambda_value
        )

    def forward(
        self,
        x,
    ):

        return GradientReversalFunction.apply(
            x,
            self.lambda_value,
        )


# =====================================================================
# SPEAKER ENCODER
# =====================================================================

class SpeakerEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                CFG.n_mels,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.projection = nn.Linear(
            128,
            CFG.speaker_embedding_dim,
        )

    def forward(
        self,
        mel,
    ):

        x = self.network(
            mel
        )

        x = x.squeeze(-1)

        x = self.projection(
            x
        )

        x = F.normalize(
            x,
            dim=-1,
        )

        return x


# =====================================================================
# CONTENT ENCODER
# =====================================================================

class ContentEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                CFG.n_mels,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.projection = nn.Linear(
            128,
            CFG.content_embedding_dim,
        )

    def forward(
        self,
        mel,
    ):

        x = self.network(
            mel
        )

        x = x.squeeze(-1)

        x = self.projection(
            x
        )

        return x


# =====================================================================
# CLASSIFIER
# =====================================================================

class Classifier(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        num_classes,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                CFG.hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                CFG.hidden_dim,
                CFG.hidden_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                CFG.hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# =====================================================================
# ACOUSTIC RECONSTRUCTION HEAD
# =====================================================================

class AcousticDecoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        combined_dim = (
            CFG.speaker_embedding_dim
            + CFG.content_embedding_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                combined_dim,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                512,
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                CFG.n_mels
                * CFG.mel_frames,
            ),
        )

    def forward(
        self,
        speaker_embedding,
        content_embedding,
    ):

        x = torch.cat(
            [
                speaker_embedding,
                content_embedding,
            ],
            dim=-1,
        )

        x = self.network(
            x
        )

        x = x.view(
            -1,
            CFG.n_mels,
            CFG.mel_frames,
        )

        return x


# =====================================================================
# COMPLETE MODEL
# =====================================================================

class DisentanglementModel(
    nn.Module
):

    def __init__(
        self,
        adversarial=False,
    ):

        super().__init__()

        self.adversarial = (
            adversarial
        )

        self.speaker_encoder = (
            SpeakerEncoder()
        )

        self.content_encoder = (
            ContentEncoder()
        )

        self.speaker_classifier = (
            Classifier(
                CFG.speaker_embedding_dim,
                CFG.num_speakers,
            )
        )

        self.content_classifier = (
            Classifier(
                CFG.speaker_embedding_dim,
                CFG.num_contents,
            )
        )

        self.content_path_classifier = (
            Classifier(
                CFG.content_embedding_dim,
                CFG.num_contents,
            )
        )

        self.decoder = (
            AcousticDecoder()
        )

        self.gradient_reversal = (
            GradientReversalLayer(
                CFG.gradient_reversal_strength
            )
        )

    def forward(
        self,
        mel,
    ):

        speaker_embedding = (
            self.speaker_encoder(
                mel
            )
        )

        content_embedding = (
            self.content_encoder(
                mel
            )
        )

        speaker_logits = (
            self.speaker_classifier(
                speaker_embedding
            )
        )

        if self.adversarial:

            content_probe_input = (
                self.gradient_reversal(
                    speaker_embedding
                )
            )

        else:

            content_probe_input = (
                speaker_embedding
            )

        content_from_speaker_logits = (
            self.content_classifier(
                content_probe_input
            )
        )

        content_logits = (
            self.content_path_classifier(
                content_embedding
            )
        )

        reconstructed_mel = (
            self.decoder(
                speaker_embedding,
                content_embedding,
            )
        )

        return {
            "speaker_embedding":
                speaker_embedding,

            "content_embedding":
                content_embedding,

            "speaker_logits":
                speaker_logits,

            "content_from_speaker_logits":
                content_from_speaker_logits,

            "content_logits":
                content_logits,

            "reconstructed_mel":
                reconstructed_mel,
        }


# =====================================================================
# TRAINING
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    total_speaker = 0.0

    total_content = 0.0

    total_content_path = 0.0

    total_reconstruction = 0.0

    for batch in loader:

        mel = batch[
            "mel"
        ].to(device)

        speaker_id = batch[
            "speaker_id"
        ].to(device)

        content_id = batch[
            "content_id"
        ].to(device)

        optimizer.zero_grad()

        output = model(
            mel
        )

        speaker_loss = (
            F.cross_entropy(
                output[
                    "speaker_logits"
                ],
                speaker_id,
            )
        )

        content_leakage_loss = (
            F.cross_entropy(
                output[
                    "content_from_speaker_logits"
                ],
                content_id,
            )
        )

        content_path_loss = (
            F.cross_entropy(
                output[
                    "content_logits"
                ],
                content_id,
            )
        )

        reconstruction_loss = (
            F.l1_loss(
                output[
                    "reconstructed_mel"
                ],
                mel,
            )
        )

        # ----------------------------------------------------------
        # Important:
        #
        # The GRL automatically reverses the gradient from
        # content_leakage_loss before it reaches the speaker encoder.
        #
        # Therefore the speaker encoder learns:
        #
        #     good speaker representation
        #
        # while simultaneously reducing:
        #
        #     recoverable content information
        # ----------------------------------------------------------

        loss = (
            speaker_loss
            + content_leakage_loss
            + content_path_loss
            + reconstruction_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

        total_speaker += (
            speaker_loss.item()
        )

        total_content += (
            content_leakage_loss.item()
        )

        total_content_path += (
            content_path_loss.item()
        )

        total_reconstruction += (
            reconstruction_loss.item()
        )

    n = len(loader)

    return {
        "loss":
            total_loss / n,

        "speaker":
            total_speaker / n,

        "content":
            total_content / n,

        "content_path":
            total_content_path / n,

        "reconstruction":
            total_reconstruction / n,
    }


# =====================================================================
# ACCURACY
# =====================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    speaker_correct = 0

    content_correct = 0

    total = 0

    content_path_correct = 0

    all_speaker_embeddings = []

    all_speaker_ids = []

    all_content_ids = []

    for batch in loader:

        mel = batch[
            "mel"
        ].to(device)

        speaker_id = batch[
            "speaker_id"
        ].to(device)

        content_id = batch[
            "content_id"
        ].to(device)

        output = model(
            mel
        )

        speaker_prediction = (
            output[
                "speaker_logits"
            ].argmax(
                dim=-1
            )
        )

        content_prediction = (
            output[
                "content_from_speaker_logits"
            ].argmax(
                dim=-1
            )
        )

        content_path_prediction = (
            output[
                "content_logits"
            ].argmax(
                dim=-1
            )
        )

        speaker_correct += (
            (
                speaker_prediction
                == speaker_id
            )
            .sum()
            .item()
        )

        content_correct += (
            (
                content_prediction
                == content_id
            )
            .sum()
            .item()
        )

        content_path_correct += (
            (
                content_path_prediction
                == content_id
            )
            .sum()
            .item()
        )

        total += (
            speaker_id.shape[0]
        )

        all_speaker_embeddings.append(
            output[
                "speaker_embedding"
            ].cpu()
        )

        all_speaker_ids.append(
            speaker_id.cpu()
        )

        all_content_ids.append(
            content_id.cpu()
        )

    return {
        "speaker_accuracy":
            speaker_correct / total,

        "content_accuracy":
            content_correct / total,

        "content_path_accuracy":
            content_path_correct / total,

        "embeddings":
            torch.cat(
                all_speaker_embeddings
            ),

        "speaker_ids":
            torch.cat(
                all_speaker_ids
            ),

        "content_ids":
            torch.cat(
                all_content_ids
            ),
    }


# =====================================================================
# EMBEDDING VISUALIZATION
# =====================================================================

def project_embeddings(
    embeddings,
):

    # --------------------------------------------------------------
    # PCA implemented directly with torch.
    #
    # This avoids requiring scikit-learn.
    # --------------------------------------------------------------

    x = embeddings.float()

    x = x - x.mean(
        dim=0,
        keepdim=True,
    )

    _, _, v = torch.pca_lowrank(
        x,
        q=2,
    )

    projection = (
        x @ v[:, :2]
    )

    return projection.numpy()


def plot_embeddings(
    evaluation,
    path,
):

    projection = project_embeddings(
        evaluation["embeddings"]
    )

    speaker_ids = (
        evaluation[
            "speaker_ids"
        ].numpy()
    )

    content_ids = (
        evaluation[
            "content_ids"
        ].numpy()
    )

    plt.figure(
        figsize=(10, 7)
    )

    scatter = plt.scatter(
        projection[:, 0],
        projection[:, 1],
        c=speaker_ids,
        cmap="tab10",
        alpha=0.75,
        s=35,
    )

    plt.colorbar(
        scatter,
        label="Speaker ID",
    )

    # --------------------------------------------------------------
    # Content labels are shown as text.
    #
    # This lets us visually inspect whether recordings with the same
    # content form suspicious clusters inside speaker space.
    # --------------------------------------------------------------

    for i in range(
        min(
            len(projection),
            100,
        )
    ):

        if i % 4 == 0:

            plt.text(
                projection[i, 0],
                projection[i, 1],
                str(content_ids[i]),
                fontsize=7,
                alpha=0.6,
            )

    plt.xlabel(
        "Embedding dimension 1"
    )

    plt.ylabel(
        "Embedding dimension 2"
    )

    plt.title(
        "Speaker Embedding Projection"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# =====================================================================
# SAVE WAV
# =====================================================================

def save_waveform(
    path,
    waveform,
    sample_rate,
):

    waveform = (
        waveform
        .detach()
        .cpu()
        .float()
        .squeeze()
    )

    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    waveform = waveform.clamp(
        -1.0,
        1.0,
    )

    pcm = (
        waveform.numpy()
        * 32767.0
    ).round().astype(
        np.int16
    )

    with wave.open(
        path,
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            1
        )

        wav_file.setsampwidth(
            2
        )

        wav_file.setframerate(
            sample_rate
        )

        wav_file.writeframes(
            pcm.tobytes()
        )


# =====================================================================
# TRAIN ONE COMPLETE MODEL
# =====================================================================

def train_model(
    adversarial,
    train_loader,
    validation_loader,
    device,
):

    model = (
        DisentanglementModel(
            adversarial=adversarial
        )
        .to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=1e-4,
    )

    history = {
        "loss": [],
        "speaker": [],
        "content": [],
        "content_path": [],
        "reconstruction": [],
        "speaker_accuracy": [],
        "content_accuracy": [],
    }

    for epoch in range(
        1,
        CFG.epochs + 1,
    ):

        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        evaluation = evaluate(
            model,
            validation_loader,
            device,
        )

        history["loss"].append(
            metrics["loss"]
        )

        history["speaker"].append(
            metrics["speaker"]
        )

        history["content"].append(
            metrics["content"]
        )

        history["content_path"].append(
            metrics["content_path"]
        )

        history["reconstruction"].append(
            metrics["reconstruction"]
        )

        history[
            "speaker_accuracy"
        ].append(
            evaluation[
                "speaker_accuracy"
            ]
        )

        history[
            "content_accuracy"
        ].append(
            evaluation[
                "content_accuracy"
            ]
        )

        name = (
            "ADVERSARIAL"
            if adversarial
            else "BASELINE"
        )

        print(
            f"[{name}] "
            f"Epoch "
            f"{epoch:02d}/{CFG.epochs} | "
            f"Loss "
            f"{metrics['loss']:.4f} | "
            f"Speaker Acc "
            f"{evaluation['speaker_accuracy']:.3f} | "
            f"Content Leakage "
            f"{evaluation['content_accuracy']:.3f}"
        )

    return (
        model,
        history,
        evaluation,
    )


# =====================================================================
# PLOT TRAINING CURVES
# =====================================================================

def plot_training_curves(
    baseline_history,
    adversarial_history,
):

    epochs = np.arange(
        1,
        CFG.epochs + 1,
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        epochs,
        baseline_history[
            "speaker_accuracy"
        ],
        marker="o",
        label="Baseline speaker accuracy",
    )

    plt.plot(
        epochs,
        adversarial_history[
            "speaker_accuracy"
        ],
        marker="o",
        label="Adversarial speaker accuracy",
    )

    plt.plot(
        epochs,
        baseline_history[
            "content_accuracy"
        ],
        marker="s",
        label="Baseline content leakage",
    )

    plt.plot(
        epochs,
        adversarial_history[
            "content_accuracy"
        ],
        marker="s",
        label="Adversarial content leakage",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Speaker Disentanglement Training"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "lesson26_training_curves.png",
        dpi=150,
    )

    plt.close()


# =====================================================================
# PLOT FINAL COMPARISON
# =====================================================================

def plot_accuracy_comparison(
    baseline_eval,
    adversarial_eval,
):

    labels = [
        "Speaker accuracy",
        "Content leakage",
    ]

    baseline_values = [
        baseline_eval[
            "speaker_accuracy"
        ],
        baseline_eval[
            "content_accuracy"
        ],
    ]

    adversarial_values = [
        adversarial_eval[
            "speaker_accuracy"
        ],
        adversarial_eval[
            "content_accuracy"
        ],
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Baseline",
    )

    plt.bar(
        x + width / 2,
        adversarial_values,
        width,
        label="Adversarial",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.title(
        "Disentanglement Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        "lesson26_accuracy_comparison.png",
        dpi=150,
    )

    plt.close()


# =====================================================================
# MAIN
# =====================================================================

def main():

    set_seed(
        CFG.seed
    )

    print(
        "=" * 72
    )

    print(
        "PHASE 1 — LESSON 26"
    )

    print(
        "SPEAKER DISENTANGLEMENT"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {CFG.device}"
    )

    print(
        f"Speakers: {CFG.num_speakers}"
    )

    print(
        f"Content classes: {CFG.num_contents}"
    )

    print(
        f"Chance content accuracy: "
        f"{1.0 / CFG.num_contents:.3f}"
    )

    print(
        "=" * 72
    )

    # ================================================================
    # DATASET
    # ================================================================

    print(
        "\nCreating synthetic speech dataset..."
    )

    dataset = DisentanglementDataset(
        CFG.samples_per_speaker_content
    )

    # --------------------------------------------------------------
    # Deterministic train / validation split
    # --------------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        CFG.seed
    )

    train_size = int(
        0.80
        * len(dataset)
    )

    validation_size = (
        len(dataset)
        - train_size
    )

    train_dataset, validation_dataset = (
        torch.utils.data.random_split(
            dataset,
            [
                train_size,
                validation_size,
            ],
            generator=generator,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Total samples: {len(dataset)}"
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    # ================================================================
    # BASELINE
    # ================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "MODEL A — BASELINE"
    )

    print(
        "=" * 72
    )

    baseline_model, baseline_history, baseline_eval = (
        train_model(
            adversarial=False,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=CFG.device,
        )
    )

    # ================================================================
    # ADVERSARIAL MODEL
    # ================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "MODEL B — ADVERSARIAL DISENTANGLEMENT"
    )

    print(
        "=" * 72
    )

    adversarial_model, adversarial_history, adversarial_eval = (
        train_model(
            adversarial=True,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=CFG.device,
        )
    )

    # ================================================================
    # SAVE MODELS
    # ================================================================

    torch.save(
        baseline_model.state_dict(),
        "lesson26_baseline_disentanglement.pt",
    )

    torch.save(
        adversarial_model.state_dict(),
        "lesson26_adversarial_disentanglement.pt",
    )

    # ================================================================
    # VISUALIZATIONS
    # ================================================================

    plot_training_curves(
        baseline_history,
        adversarial_history,
    )

    plot_accuracy_comparison(
        baseline_eval,
        adversarial_eval,
    )

    plot_embeddings(
        adversarial_eval,
        "lesson26_embedding_projection.png",
    )

    # ================================================================
    # SAVE SAMPLE AUDIO
    # ================================================================

    sample_speaker = SpeakerProfile(
        0
    )

    sample_content = ContentProfile(
        0
    )

    sample_waveform = generate_speech(
        sample_speaker,
        sample_content,
        seed=12345,
    )

    save_waveform(
        "lesson26_reference.wav",
        sample_waveform,
        CFG.sample_rate,
    )

    # ================================================================
    # FINAL RESULTS
    # ================================================================

    baseline_speaker = (
        baseline_eval[
            "speaker_accuracy"
        ]
    )

    baseline_content = (
        baseline_eval[
            "content_accuracy"
        ]
    )

    adversarial_speaker = (
        adversarial_eval[
            "speaker_accuracy"
        ]
    )

    adversarial_content = (
        adversarial_eval[
            "content_accuracy"
        ]
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        "\nBASELINE"
    )

    print(
        f"Speaker accuracy: "
        f"{baseline_speaker:.3f}"
    )

    print(
        f"Content leakage: "
        f"{baseline_content:.3f}"
    )

    print(
        "\nADVERSARIAL"
    )

    print(
        f"Speaker accuracy: "
        f"{adversarial_speaker:.3f}"
    )

    print(
        f"Content leakage: "
        f"{adversarial_content:.3f}"
    )

    print(
        "\nCHANCE CONTENT ACCURACY"
    )

    print(
        f"{1.0 / CFG.num_contents:.3f}"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 72
    )

    print(
        """
The desired behavior is:

    HIGH speaker accuracy
    LOW content leakage

The adversarial model uses Gradient Reversal to make the
speaker embedding less useful for predicting linguistic content.

This demonstrates the basic principle of speaker/content
disentanglement.
"""
    )

    print(
        "Generated files:"
    )

    print(
        "  lesson26_baseline_disentanglement.pt"
    )

    print(
        "  lesson26_adversarial_disentanglement.pt"
    )

    print(
        "  lesson26_training_curves.png"
    )

    print(
        "  lesson26_accuracy_comparison.png"
    )

    print(
        "  lesson26_embedding_projection.png"
    )

    print(
        "  lesson26_reference.wav"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "LESSON 26 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "\nNext:"
    )

    print(
        "LESSON 27 — POST-HOC CONTENT LEAKAGE PROBE"
    )

    print(
        "\nWe will freeze the speaker encoder and train "
        "a completely separate attacker classifier."
    )

    print(
        "\nThat gives us a much stronger test of whether "
        "content information is actually present in the "
        "speaker embedding."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()