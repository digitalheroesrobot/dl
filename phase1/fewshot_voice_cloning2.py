"""
=======================================================================
PHASE 1 — LESSON 25
FEW-SHOT VOICE CLONING WITH MULTIPLE REFERENCE UTTERANCES
=======================================================================

FILE
----
fewshot_voice_cloning.py

PURPOSE
-------
Extend the single-reference voice cloning system from Lesson 24 into a
more robust few-shot voice cloning system.

Instead of extracting speaker identity from one reference utterance,
we provide multiple reference utterances and learn how to combine them.

This lesson is educational and uses synthetic speech data.

It is NOT intended to reproduce Chatterbox internally.
Instead, it teaches the important ideas needed to understand modern
few-shot / zero-shot voice cloning systems.

=======================================================================
WHAT YOU LEARN
=======================================================================

STEP 1 — MULTIPLE SPEAKER REFERENCES
-------------------------------------
A real speaker may provide several reference recordings.

Instead of:

    reference_audio
          |
    speaker_encoder
          |
    speaker_embedding

we use:

    reference_1 ----\
    reference_2 -----\
    reference_3 ------> speaker encoder
    reference_4 -----/
    reference_5 ----/

Each recording produces a speaker embedding.

-----------------------------------------------------------------------

STEP 2 — REFERENCE ATTENTION
----------------------------

Not every reference recording is equally useful.

For example:

    Reference 1 → clean
    Reference 2 → noisy
    Reference 3 → emotional
    Reference 4 → very short
    Reference 5 → clean

We learn a score for every reference:

    s_i = attention_score(e_i)

and convert scores into weights:

                    exp(s_i)
    alpha_i = -----------------------
              sum_j exp(s_j)

The final speaker representation becomes:

    e_speaker = sum_i alpha_i e_i

This allows the model to emphasize useful references.

-----------------------------------------------------------------------

STEP 3 — REFERENCE DROPOUT
--------------------------

During training, some references are randomly removed.

Example:

    [R1 R2 R3 R4 R5]

becomes:

    [R1 X  R3 X  R5]

This prevents the model from becoming dependent on a particular
reference recording.

It also teaches the model to work with different numbers of
references.

-----------------------------------------------------------------------

STEP 4 — NOISY REFERENCE AUGMENTATION
--------------------------------------

Real recordings contain:

    microphone noise
    room noise
    gain differences
    spectral distortion

Therefore, some training references are intentionally corrupted.

The model should learn:

    speaker identity
        NOT
    recording identity

-----------------------------------------------------------------------

STEP 5 — SPEAKER / CONTENT DISENTANGLEMENT
-------------------------------------------

The reference encoder should primarily represent:

    WHO is speaking

while the text encoder represents:

    WHAT is being spoken

The acoustic decoder combines them:

    speaker_embedding + text_features
                       |
                       v
                  acoustic model
                       |
                       v
                   mel-spectrogram

-----------------------------------------------------------------------

STEP 6 — FEW-SHOT EVALUATION
----------------------------

We evaluate cloning with:

    1 reference
    2 references
    3 references
    5 references

and compare:

    speaker cosine similarity
    mel reconstruction error

This demonstrates why multiple references can improve robustness.

=======================================================================
MATHEMATICAL MODEL
=======================================================================

For K reference utterances:

    r_1, r_2, ..., r_K

Speaker encoder:

    e_i = E(r_i)

Reference attention:

    q_i = f(e_i)

    alpha_i = softmax(q_i)

Aggregated speaker representation:

    e = sum_i alpha_i e_i

Text representation:

    h_text = T(x)

Acoustic decoder:

    mel_hat = D(h_text, e)

Training objective:

    L =
        L_mel
        + lambda_spk L_speaker
        + lambda_consistency L_consistency

where:

    L_mel
        = reconstruction error

    L_speaker
        = speaker classification loss

    L_consistency
        = encourages different reference subsets
          to produce similar speaker representations

=======================================================================
REAL-WORLD APPLICATION
=======================================================================

This technique is useful for:

    - personalized TTS
    - audiobook narration
    - virtual assistants
    - character voices
    - accessibility systems
    - conversational AI
    - voice personalization
    - few-shot voice cloning

Example:

    User provides 5 short recordings.

    The system extracts:

        speaker identity

    rather than simply copying:

        acoustic characteristics of one recording.

=======================================================================
LESSON 25 ROADMAP
=======================================================================

    1. Generate synthetic multi-reference speech
    2. Extract reference embeddings
    3. Add reference attention
    4. Add reference dropout
    5. Add noisy-reference augmentation
    6. Train few-shot cloning model
    7. Compare 1/2/3/5 reference inference
    8. Measure speaker similarity
    9. Visualize attention weights
   10. Generate WAV files

=======================================================================
REQUIREMENTS
=======================================================================

Python 3.10+

    pip install torch numpy matplotlib

No torchaudio is required.

WAV files are written using Python's standard-library `wave` module.

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

    # 32 * 256 = 8192 samples
    waveform_length: int = 8192
    mel_frames: int = 32

    # --------------------------------------------------------------
    # Synthetic speakers
    # --------------------------------------------------------------

    num_speakers: int = 16
    references_per_speaker: int = 5
    targets_per_speaker: int = 3

    # --------------------------------------------------------------
    # Text
    # --------------------------------------------------------------

    vocab_size: int = 64
    token_length: int = 12

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model_dim: int = 128
    speaker_embedding_dim: int = 64
    num_speakers_classifier: int = 16

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    train_samples_per_speaker: int = 3
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 2e-3

    speaker_loss_weight: float = 0.20
    consistency_loss_weight: float = 0.30

    reference_dropout_probability: float = 0.25
    noisy_reference_probability: float = 0.40

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    seed: int = 42

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# TEXT EXAMPLES
# =====================================================================

TEXT_EXAMPLES = [
    "Hello, this is a voice cloning demonstration.",
    "Artificial intelligence can learn patterns from speech.",
    "The robot is sorting different objects on the table.",
    "Today we are studying neural speech synthesis.",
    "Voice technology can make computers easier to use.",
    "This system learns who is speaking from several examples.",
    "The model combines language and speaker information.",
    "Speech synthesis is becoming increasingly expressive.",
    "机器学习可以从语音数据中学习。",
    "人工智能正在改变语音技术。",
    "ロボットはテーブルの上の物体を分類します。",
    "音声合成技術は急速に進歩しています。",
    "Искусственный интеллект может изучать речь.",
    "Синтез речи становится все более естественным.",
]


# =====================================================================
# MEL FILTERBANK
# =====================================================================

def hz_to_mel(freq):
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> torch.Tensor:

    if f_max is None:
        f_max = sample_rate / 2.0

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = torch.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
    )

    hz_points = mel_to_hz(mel_points)

    bins = torch.floor(
        (n_fft + 1) * hz_points / sample_rate
    ).long()

    filterbank = torch.zeros(
        n_mels,
        n_fft // 2 + 1,
    )

    for m in range(1, n_mels + 1):

        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])

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


MEL_FILTERBANK = create_mel_filterbank(
    CFG.sample_rate,
    CFG.n_fft,
    CFG.n_mels,
)


# =====================================================================
# WAVEFORM → MEL
# =====================================================================

def waveform_to_mel(
    waveform: torch.Tensor,
) -> torch.Tensor:

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

    mel_filter = MEL_FILTERBANK.to(
        waveform.device
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

    def __init__(self, speaker_id: int):

        rng = np.random.default_rng(
            CFG.seed + speaker_id * 100
        )

        self.base_f0 = float(
            rng.uniform(100.0, 220.0)
        )

        self.harmonic_tilt = float(
            rng.uniform(0.15, 0.55)
        )

        self.brightness = float(
            rng.uniform(0.5, 1.5)
        )

        self.resonance = float(
            rng.uniform(0.8, 1.3)
        )

        self.style = float(
            rng.uniform(0.7, 1.3)
        )


# =====================================================================
# SYNTHETIC SPEECH GENERATOR
# =====================================================================

def generate_synthetic_speech(
    speaker: SpeakerProfile,
    seed: int,
) -> torch.Tensor:

    rng = np.random.default_rng(seed)

    n = CFG.waveform_length

    t = np.arange(n) / CFG.sample_rate

    # --------------------------------------------------------------
    # Utterance variation
    # --------------------------------------------------------------

    pitch_variation = rng.uniform(
        0.92,
        1.08,
    )

    f0 = (
        speaker.base_f0
        * pitch_variation
    )

    # Slow pitch movement
    f0_curve = f0 * (
        1.0
        + 0.025
        * np.sin(
            2.0
            * np.pi
            * rng.uniform(1.0, 3.0)
            * t
        )
    )

    phase = np.cumsum(
        2.0
        * np.pi
        * f0_curve
        / CFG.sample_rate
    )

    waveform = np.zeros(n)

    # --------------------------------------------------------------
    # Harmonic speech-like structure
    # --------------------------------------------------------------

    for harmonic in range(1, 9):

        amplitude = (
            1.0
            / (harmonic ** speaker.harmonic_tilt)
        )

        amplitude *= speaker.brightness ** (
            -0.15 * harmonic
        )

        waveform += amplitude * np.sin(
            harmonic * phase
        )

    # --------------------------------------------------------------
    # Formant-like modulation
    # --------------------------------------------------------------

    formant_1 = 2.0 + 0.4 * speaker.resonance
    formant_2 = 5.0 + 0.8 * speaker.resonance

    modulation = (
        1.0
        + 0.20
        * np.sin(
            2.0
            * np.pi
            * formant_1
            * t
        )
        + 0.10
        * np.sin(
            2.0
            * np.pi
            * formant_2
            * t
        )
    )

    waveform *= modulation

    # --------------------------------------------------------------
    # Speech envelope
    # --------------------------------------------------------------

    envelope = np.ones(n)

    attack = int(0.06 * CFG.sample_rate)
    release = int(0.08 * CFG.sample_rate)

    envelope[:attack] = np.linspace(
        0.0,
        1.0,
        attack,
    )

    envelope[-release:] = np.linspace(
        1.0,
        0.0,
        release,
    )

    # Slow syllabic amplitude movement
    syllable = (
        0.72
        + 0.28
        * (
            0.5
            + 0.5
            * np.sin(
                2.0
                * np.pi
                * speaker.style
                * 4.0
                * t
            )
        )
    )

    waveform *= envelope
    waveform *= syllable

    # --------------------------------------------------------------
    # Small noise floor
    # --------------------------------------------------------------

    noise = rng.normal(
        0.0,
        0.008,
        n,
    )

    waveform += noise

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
# REFERENCE AUGMENTATION
# =====================================================================

def augment_reference(
    waveform: torch.Tensor,
    seed: int,
) -> torch.Tensor:

    rng = np.random.default_rng(seed)

    x = waveform.clone()

    # --------------------------------------------------------------
    # Gain variation
    # --------------------------------------------------------------

    gain = rng.uniform(
        0.75,
        1.20,
    )

    x *= gain

    # --------------------------------------------------------------
    # Additive noise
    # --------------------------------------------------------------

    noise_level = rng.uniform(
        0.0,
        0.025,
    )

    noise = torch.randn_like(x) * noise_level

    x = x + noise

    # --------------------------------------------------------------
    # Mild amplitude compression
    # --------------------------------------------------------------

    if rng.random() < 0.5:

        x = torch.tanh(
            1.5 * x
        )

    x = x.clamp(
        -1.0,
        1.0,
    )

    return x


# =====================================================================
# TOKEN GENERATION
# =====================================================================

def generate_tokens(
    speaker_id: int,
    sample_index: int,
) -> torch.Tensor:

    rng = np.random.default_rng(
        CFG.seed
        + speaker_id * 1000
        + sample_index
    )

    tokens = rng.integers(
        1,
        CFG.vocab_size,
        size=CFG.token_length,
    )

    return torch.tensor(
        tokens,
        dtype=torch.long,
    )


# =====================================================================
# DATASET
# =====================================================================

class FewShotVoiceDataset(Dataset):

    def __init__(
        self,
        speaker_ids,
        training: bool = True,
    ):

        self.samples = []

        for speaker_id in speaker_ids:

            profile = SpeakerProfile(
                speaker_id
            )

            # ------------------------------------------------------
            # Build several reference utterances for each speaker.
            # ------------------------------------------------------

            reference_waveforms = []

            for ref_id in range(
                CFG.references_per_speaker
            ):

                waveform = generate_synthetic_speech(
                    profile,
                    seed=(
                        10000
                        + speaker_id * 100
                        + ref_id
                    ),
                )

                reference_waveforms.append(
                    waveform
                )

            # ------------------------------------------------------
            # Build target utterances.
            # ------------------------------------------------------

            num_targets = (
                CFG.train_samples_per_speaker
                if training
                else CFG.targets_per_speaker
            )

            for target_id in range(
                num_targets
            ):

                target_waveform = (
                    generate_synthetic_speech(
                        profile,
                        seed=(
                            50000
                            + speaker_id * 100
                            + target_id
                        ),
                    )
                )

                target_mel = waveform_to_mel(
                    target_waveform
                )

                target_tokens = generate_tokens(
                    speaker_id,
                    target_id,
                )

                self.samples.append(
                    {
                        "speaker_id": speaker_id,
                        "reference_waveforms":
                            reference_waveforms,
                        "target_waveform":
                            target_waveform,
                        "target_mel":
                            target_mel,
                        "tokens":
                            target_tokens,
                    }
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        references = []

        for ref_id, waveform in enumerate(
            sample["reference_waveforms"]
        ):

            reference = waveform.clone()

            if random.random() < (
                CFG.noisy_reference_probability
            ):

                reference = augment_reference(
                    reference,
                    seed=(
                        index * 100
                        + ref_id
                    ),
                )

            mel = waveform_to_mel(
                reference
            )

            references.append(mel)

        reference_mels = torch.stack(
            references
        )

        return {
            "reference_mels":
                reference_mels,

            "tokens":
                sample["tokens"],

            "target_mel":
                sample["target_mel"],

            "target_waveform":
                sample["target_waveform"],

            "speaker_id":
                torch.tensor(
                    sample["speaker_id"],
                    dtype=torch.long,
                ),
        }


# =====================================================================
# SPEAKER ENCODER
# =====================================================================

class SpeakerEncoder(nn.Module):

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
        )

        self.projection = nn.Linear(
            128,
            CFG.speaker_embedding_dim,
        )

    def forward(self, mel):

        x = self.network(mel)

        x = x.mean(dim=-1)

        x = self.projection(x)

        x = F.normalize(
            x,
            dim=-1,
        )

        return x


# =====================================================================
# REFERENCE ATTENTION
# =====================================================================

class ReferenceAttention(nn.Module):

    def __init__(self):

        super().__init__()

        d = CFG.speaker_embedding_dim

        self.score = nn.Sequential(

            nn.Linear(
                d,
                d,
            ),

            nn.Tanh(),

            nn.Linear(
                d,
                1,
            ),
        )

    def forward(
        self,
        embeddings,
        reference_mask=None,
    ):

        # embeddings:
        #
        # [batch, references, embedding_dim]

        scores = self.score(
            embeddings
        ).squeeze(-1)

        if reference_mask is not None:

            scores = scores.masked_fill(
                ~reference_mask,
                -1e9,
            )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        aggregated = (
            embeddings
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        aggregated = F.normalize(
            aggregated,
            dim=-1,
        )

        return aggregated, weights


# =====================================================================
# TEXT ENCODER
# =====================================================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            CFG.vocab_size,
            CFG.model_dim,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=CFG.model_dim,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

    def forward(self, tokens):

        x = self.embedding(
            tokens
        )

        x = self.encoder(x)

        return x


# =====================================================================
# ACOUSTIC DECODER
# =====================================================================

class AcousticDecoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.speaker_projection = nn.Linear(
            CFG.speaker_embedding_dim,
            CFG.model_dim,
        )

        self.network = nn.Sequential(

            nn.Conv1d(
                CFG.model_dim,
                256,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                256,
                256,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                256,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                CFG.n_mels,
                kernel_size=5,
                padding=2,
            ),
        )

    def forward(
        self,
        text_features,
        speaker_embedding,
    ):

        speaker = self.speaker_projection(
            speaker_embedding
        )

        speaker = speaker.unsqueeze(1)

        x = text_features + speaker

        x = x.transpose(
            1,
            2,
        )

        x = F.interpolate(
            x,
            size=CFG.mel_frames,
            mode="linear",
            align_corners=False,
        )

        x = self.network(x)

        return x


# =====================================================================
# COMPLETE FEW-SHOT MODEL
# =====================================================================

class FewShotVoiceCloningModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.speaker_encoder = (
            SpeakerEncoder()
        )

        self.reference_attention = (
            ReferenceAttention()
        )

        self.text_encoder = (
            TextEncoder()
        )

        self.decoder = (
            AcousticDecoder()
        )

        self.speaker_classifier = nn.Linear(
            CFG.speaker_embedding_dim,
            CFG.num_speakers_classifier,
        )

    def encode_references(
        self,
        reference_mels,
        reference_mask=None,
    ):

        batch_size, num_refs, _, _ = (
            reference_mels.shape
        )

        flat = reference_mels.reshape(
            batch_size * num_refs,
            CFG.n_mels,
            CFG.mel_frames,
        )

        embeddings = (
            self.speaker_encoder(flat)
        )

        embeddings = embeddings.reshape(
            batch_size,
            num_refs,
            CFG.speaker_embedding_dim,
        )

        speaker_embedding, weights = (
            self.reference_attention(
                embeddings,
                reference_mask,
            )
        )

        return (
            speaker_embedding,
            weights,
            embeddings,
        )

    def forward(
        self,
        reference_mels,
        tokens,
        reference_mask=None,
    ):

        speaker_embedding, weights, embeddings = (
            self.encode_references(
                reference_mels,
                reference_mask,
            )
        )

        text_features = self.text_encoder(
            tokens
        )

        mel = self.decoder(
            text_features,
            speaker_embedding,
        )

        speaker_logits = (
            self.speaker_classifier(
                speaker_embedding
            )
        )

        return {
            "mel":
                mel,

            "speaker_embedding":
                speaker_embedding,

            "speaker_logits":
                speaker_logits,

            "attention_weights":
                weights,

            "reference_embeddings":
                embeddings,
        }


# =====================================================================
# REFERENCE DROPOUT
# =====================================================================

def create_reference_mask(
    batch_size: int,
    num_references: int,
    device: torch.device,
) -> torch.Tensor:

    mask = torch.ones(
        batch_size,
        num_references,
        dtype=torch.bool,
        device=device,
    )

    for b in range(batch_size):

        if random.random() < (
            CFG.reference_dropout_probability
        ):

            num_to_drop = random.randint(
                1,
                max(
                    1,
                    num_references - 1,
                ),
            )

            indices = random.sample(
                range(num_references),
                num_to_drop,
            )

            for index in indices:
                mask[b, index] = False

    return mask


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
    total_mel = 0.0
    total_speaker = 0.0
    total_consistency = 0.0

    for batch in loader:

        reference_mels = batch[
            "reference_mels"
        ].to(device)

        tokens = batch[
            "tokens"
        ].to(device)

        target_mel = batch[
            "target_mel"
        ].to(device)

        speaker_id = batch[
            "speaker_id"
        ].to(device)

        optimizer.zero_grad()

        # ----------------------------------------------------------
        # Random reference dropout
        # ----------------------------------------------------------

        reference_mask = create_reference_mask(
            reference_mels.shape[0],
            reference_mels.shape[1],
            device,
        )

        output = model(
            reference_mels,
            tokens,
            reference_mask,
        )

        predicted_mel = output["mel"]

        speaker_logits = (
            output["speaker_logits"]
        )

        speaker_embedding = (
            output["speaker_embedding"]
        )

        # ----------------------------------------------------------
        # Mel reconstruction
        # ----------------------------------------------------------

        mel_loss = F.l1_loss(
            predicted_mel,
            target_mel,
        )

        # ----------------------------------------------------------
        # Speaker classification
        # ----------------------------------------------------------

        speaker_loss = F.cross_entropy(
            speaker_logits,
            speaker_id,
        )

        # ----------------------------------------------------------
        # Reference consistency
        #
        # Compare the aggregated embedding against the average
        # embedding from all references.
        # ----------------------------------------------------------

        reference_embeddings = (
            output["reference_embeddings"]
        )

        reference_mean = F.normalize(
            reference_embeddings.mean(dim=1),
            dim=-1,
        )

        consistency_loss = (
            1.0
            - F.cosine_similarity(
                speaker_embedding,
                reference_mean,
                dim=-1,
            ).mean()
        )

        loss = (
            mel_loss
            + CFG.speaker_loss_weight
            * speaker_loss
            + CFG.consistency_loss_weight
            * consistency_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        total_mel += mel_loss.item()
        total_speaker += speaker_loss.item()
        total_consistency += (
            consistency_loss.item()
        )

    n = len(loader)

    return {
        "loss": total_loss / n,
        "mel": total_mel / n,
        "speaker": total_speaker / n,
        "consistency":
            total_consistency / n,
    }


# =====================================================================
# SPEAKER SIMILARITY
# =====================================================================

@torch.no_grad()
def evaluate_reference_count(
    model,
    dataset,
    device,
    reference_count: int,
):

    model.eval()

    similarities = []
    mel_errors = []

    attention_examples = []

    for index in range(
        min(
            len(dataset),
            32,
        )
    ):

        sample = dataset[index]

        reference_mels = sample[
            "reference_mels"
        ].unsqueeze(0).to(device)

        tokens = sample[
            "tokens"
        ].unsqueeze(0).to(device)

        target_mel = sample[
            "target_mel"
        ].unsqueeze(0).to(device)

        # ----------------------------------------------------------
        # Select only the requested number of references.
        # ----------------------------------------------------------

        reference_mels = (
            reference_mels[
                :,
                :reference_count,
            ]
        )

        output = model(
            reference_mels,
            tokens,
        )

        generated_embedding = (
            output["speaker_embedding"]
        )

        # Use the first reference as an identity anchor.
        anchor_embedding = (
            output["reference_embeddings"][
                :,
                0,
            ]
        )

        similarity = F.cosine_similarity(
            generated_embedding,
            anchor_embedding,
            dim=-1,
        ).item()

        mel_error = F.l1_loss(
            output["mel"],
            target_mel,
        ).item()

        similarities.append(
            similarity
        )

        mel_errors.append(
            mel_error
        )

        if len(attention_examples) < 5:
            attention_examples.append(
                output[
                    "attention_weights"
                ][0].cpu().numpy()
            )

    return {
        "similarity":
            float(np.mean(similarities)),

        "mel_error":
            float(np.mean(mel_errors)),

        "attention":
            attention_examples,
    }


# =====================================================================
# WAV SAVING
# =====================================================================

def save_waveform(
    path: str,
    waveform: torch.Tensor,
    sample_rate: int,
):

    waveform = (
        waveform.detach()
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

        wav_file.setnchannels(1)

        wav_file.setsampwidth(2)

        wav_file.setframerate(
            sample_rate
        )

        wav_file.writeframes(
            pcm.tobytes()
        )


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_attention(
    attention,
    path="lesson25_reference_attention.png",
):

    plt.figure(
        figsize=(8, 4)
    )

    mean_attention = np.mean(
        np.stack(attention),
        axis=0,
    )

    plt.bar(
        np.arange(
            len(mean_attention)
        ) + 1,
        mean_attention,
    )

    plt.xlabel(
        "Reference utterance"
    )

    plt.ylabel(
        "Attention weight"
    )

    plt.title(
        "Learned Reference Attention"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def plot_reference_comparison(
    results,
    path="lesson25_fewshot_comparison.png",
):

    counts = sorted(
        results.keys()
    )

    similarities = [
        results[c]["similarity"]
        for c in counts
    ]

    mel_errors = [
        results[c]["mel_error"]
        for c in counts
    ]

    fig, ax1 = plt.subplots(
        figsize=(8, 5)
    )

    ax1.plot(
        counts,
        similarities,
        marker="o",
        linewidth=2,
        label="Speaker similarity",
    )

    ax1.set_xlabel(
        "Number of reference utterances"
    )

    ax1.set_ylabel(
        "Cosine speaker similarity"
    )

    ax1.set_xticks(counts)

    ax2 = ax1.twinx()

    ax2.plot(
        counts,
        mel_errors,
        marker="s",
        linewidth=2,
        label="Mel error",
    )

    ax2.set_ylabel(
        "Mel reconstruction error"
    )

    plt.title(
        "Few-Shot Voice Cloning Performance"
    )

    fig.tight_layout()

    plt.savefig(
        path,
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
        "PHASE 1 — LESSON 25"
    )

    print(
        "FEW-SHOT VOICE CLONING"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {CFG.device}"
    )

    print(
        f"Sample rate: {CFG.sample_rate}"
    )

    print(
        f"Mel dimensions: "
        f"{CFG.n_mels} x {CFG.mel_frames}"
    )

    print(
        f"References per speaker: "
        f"{CFG.references_per_speaker}"
    )

    print(
        "=" * 72
    )

    # ================================================================
    # DATA
    # ================================================================

    print(
        "\nCreating synthetic few-shot dataset..."
    )

    train_speakers = list(
        range(0, 12)
    )

    validation_speakers = list(
        range(12, 16)
    )

    train_dataset = FewShotVoiceDataset(
        train_speakers,
        training=True,
    )

    validation_dataset = FewShotVoiceDataset(
        validation_speakers,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,
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
    # MODEL
    # ================================================================

    model = (
        FewShotVoiceCloningModel()
        .to(CFG.device)
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=1e-4,
    )

    # ================================================================
    # TRAINING
    # ================================================================

    history = {
        "loss": [],
        "mel": [],
        "speaker": [],
        "consistency": [],
    }

    print(
        "\nStarting few-shot training..."
    )

    for epoch in range(
        1,
        CFG.epochs + 1,
    ):

        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            CFG.device,
        )

        history["loss"].append(
            metrics["loss"]
        )

        history["mel"].append(
            metrics["mel"]
        )

        history["speaker"].append(
            metrics["speaker"]
        )

        history["consistency"].append(
            metrics["consistency"]
        )

        print(
            f"Epoch {epoch:02d}/{CFG.epochs} | "
            f"Loss {metrics['loss']:.4f} | "
            f"Mel {metrics['mel']:.4f} | "
            f"Speaker {metrics['speaker']:.4f} | "
            f"Consistency {metrics['consistency']:.4f}"
        )

    # ================================================================
    # FEW-SHOT EVALUATION
    # ================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FEW-SHOT EVALUATION"
    )

    print(
        "=" * 72
    )

    results = {}

    for reference_count in [
        1,
        2,
        3,
        5,
    ]:

        result = evaluate_reference_count(
            model,
            validation_dataset,
            CFG.device,
            reference_count,
        )

        results[
            reference_count
        ] = result

        print(
            f"{reference_count} reference(s) | "
            f"speaker similarity = "
            f"{result['similarity']:.4f} | "
            f"mel error = "
            f"{result['mel_error']:.4f}"
        )

    # ================================================================
    # ATTENTION VISUALIZATION
    # ================================================================

    plot_attention(
        results[5]["attention"]
    )

    plot_reference_comparison(
        results
    )

    # ================================================================
    # TRAINING CURVES
    # ================================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        history["loss"],
        label="Total loss",
    )

    plt.plot(
        history["mel"],
        label="Mel loss",
    )

    plt.plot(
        history["speaker"],
        label="Speaker loss",
    )

    plt.plot(
        history["consistency"],
        label="Consistency loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Lesson 25 Training Curves"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "lesson25_training_curves.png",
        dpi=150,
    )

    plt.close()

    # ================================================================
    # SAVE MODEL
    # ================================================================

    torch.save(
        model.state_dict(),
        "lesson25_fewshot_voice_cloning.pt",
    )

    # ================================================================
    # SAVE EXAMPLE REFERENCE WAV
    # ================================================================

    sample = validation_dataset[0]

    # Recreate a clean reference waveform
    speaker_id = validation_speakers[0]

    profile = SpeakerProfile(
        speaker_id
    )

    reference_waveform = (
        generate_synthetic_speech(
            profile,
            seed=10000
            + speaker_id * 100,
        )
    )

    target_waveform = (
        generate_synthetic_speech(
            profile,
            seed=50000
            + speaker_id * 100,
        )
    )

    save_waveform(
        "lesson25_reference.wav",
        reference_waveform,
        CFG.sample_rate,
    )

    save_waveform(
        "lesson25_target.wav",
        target_waveform,
        CFG.sample_rate,
    )

    # ================================================================
    # PRINT FINAL REPORT
    # ================================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "LESSON 25 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  lesson25_fewshot_voice_cloning.pt"
    )

    print(
        "  lesson25_reference.wav"
    )

    print(
        "  lesson25_target.wav"
    )

    print(
        "  lesson25_training_curves.png"
    )

    print(
        "  lesson25_reference_attention.png"
    )

    print(
        "  lesson25_fewshot_comparison.png"
    )

    print(
        "\nKey lesson:"
    )

    print(
        "Multiple reference recordings can be combined "
        "into a more robust speaker representation."
    )

    print(
        "\nReference attention allows the model to "
        "learn which reference utterances are most useful."
    )

    print(
        "\nReference dropout and noisy-reference "
        "augmentation make the model less dependent "
        "on one perfect recording."
    )

    print(
        "\nNext lesson:"
    )

    print(
        "PHASE 1 — LESSON 26"
    )

    print(
        "Speaker Disentanglement and "
        "Content Leakage Detection"
    )

    print(
        "=" * 72
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()