"""
=======================================================================
FILE: phase1_lesson9_forced_alignment.py

PHASE 1 — LESSON 9
CTC FORCED ALIGNMENT AND PHONEME/WORD TIMING
=======================================================================

PURPOSE
-------

Understand and implement forced alignment using:

    1. CTC (Connectionist Temporal Classification)
    2. Log probabilities
    3. Expanded target sequences
    4. Dynamic programming
    5. Viterbi decoding
    6. Backtracking
    7. Token timestamps
    8. Word timestamps
    9. Alignment visualization

IMPORTANT:

This is an educational implementation of CTC Viterbi forced alignment.

It implements the alignment algorithm itself.

It does NOT implement or train an acoustic neural network.

For real audio, a trained acoustic model must first generate:

    log_probs.shape = (number_of_frames, vocabulary_size)

The resulting frame-level probabilities are passed to this aligner.

=======================================================================
PART 1 — FORCED ALIGNMENT
=======================================================================

Forced alignment estimates where known text units occur in an audio
recording.

Inputs:

    Audio recording
    Known transcript

Output:

    Token start times
    Token end times
    Word start times
    Word end times

For example:

    Transcript:
        HELLO WORLD

Possible alignment:

    H       0.00 - 0.08
    E       0.08 - 0.16
    L       0.16 - 0.28
    L       0.28 - 0.36
    O       0.36 - 0.52
    W       0.60 - 0.72
    O       0.72 - 0.84
    R       0.84 - 0.92
    L       0.92 - 1.00
    D       1.00 - 1.12

The timestamps are estimates based on the acoustic model.

=======================================================================
PART 2 — CTC
=======================================================================

CTC is useful when input and output sequences have different lengths.

Suppose the target is:

    H E L L O

A possible frame-level path is:

    _ H H _ E E L L _ L O O _

The CTC collapse operation:

    1. Merge consecutive repetitions.
    2. Remove blank tokens.

Result:

    H E L L O

IMPORTANT:

For repeated target tokens such as L L, a blank must separate the
two token emissions in a valid CTC path.

For example:

    L L

collapses to:

    L

But:

    L _ L

collapses to:

    L L

=======================================================================
PART 3 — CTC TARGET EXPANSION
=======================================================================

Given:

    H E L L O

We construct:

    _ H _ E _ L _ L _ O _

The blank token is inserted:

    Before the first target token.
    Between every pair of target tokens.
    After the final target token.

For N target tokens:

    Number of expanded states = 2N + 1

=======================================================================
PART 4 — DYNAMIC PROGRAMMING
=======================================================================

Let:

    T = number of acoustic frames

    S = number of expanded CTC states

    p[t, k] = probability of token k at frame t

    z[s] = token associated with expanded state s

Define:

    V[t, s]

as the maximum log probability of a valid alignment path ending
at frame t and expanded state s.

The recurrence is:

    V[t,s] = log p[t,z[s]] + max(
        V[t-1,s],
        V[t-1,s-1],
        V[t-1,s-2]
    )

The transition from s-2 to s is permitted only when:

    1. The destination state is not blank.
    2. The destination token differs from the token two states
       earlier.

This prevents invalid CTC paths.

We store the best predecessor for each state.

After processing all frames, we backtrack through the table.

=======================================================================
PART 5 — LOG PROBABILITIES
=======================================================================

Neural networks usually produce logits.

A softmax converts logits into probabilities:

    p(k) = exp(logit[k]) / sum_j exp(logit[j])

However, multiplying many probabilities can cause numerical
underflow.

Therefore, we use logarithms:

    log P(path) = sum_t log p[t, token_t]

Log probabilities turn multiplication into addition.

This is numerically more stable.

=======================================================================
PART 6 — FRAME-TO-TIME CONVERSION
=======================================================================

Suppose:

    sample_rate = 24000 Hz

    hop_length = 240 samples

Then:

    frame_duration = hop_length / sample_rate

                  = 240 / 24000

                  = 0.01 seconds

Therefore, each acoustic frame represents 10 milliseconds.

For frame index i:

    time = i * hop_length / sample_rate

The exact timestamp convention depends on the acoustic feature
extractor and whether timestamps refer to frame starts, centers,
or ends.

This educational program uses frame boundaries.

=======================================================================
PART 7 — WORD TIMESTAMPS
=======================================================================

The aligner initially produces token timestamps.

To obtain word timestamps, we need to know which tokens belong
to each word.

For example:

    H E L L O | W O R L D

Token-to-word mapping:

    H -> HELLO
    E -> HELLO
    L -> HELLO
    L -> HELLO
    O -> HELLO

    W -> WORLD
    O -> WORLD
    R -> WORLD
    L -> WORLD
    D -> WORLD

Word start time:

    Minimum start time of its tokens.

Word end time:

    Maximum end time of its tokens.

This educational example uses character tokens.

A production TTS frontend will normally use phonemes or another
model-specific tokenization scheme.

=======================================================================
PART 8 — LIMITATIONS
=======================================================================

This program demonstrates alignment using synthetic acoustic
probabilities.

It does not estimate acoustic probabilities from real audio.

Real forced alignment requires:

    1. Audio loading.
    2. Feature extraction.
    3. A trained acoustic model.
    4. Correct vocabulary mapping.
    5. Correct transcript tokenization.
    6. CTC or another alignment algorithm.

A real acoustic model may use:

    - CNN layers
    - RNN layers
    - LSTM layers
    - Transformer encoders
    - Conformer encoders

The aligner is only one component of a larger speech system.

=======================================================================
USAGE
=======================================================================

Run the complete demonstration:

    python phase1_lesson9_forced_alignment.py

Run a specific experiment:

    python phase1_lesson9_forced_alignment.py --experiment 1

    python phase1_lesson9_forced_alignment.py --experiment 2

    python phase1_lesson9_forced_alignment.py --experiment 3

    python phase1_lesson9_forced_alignment.py --experiment 4

=======================================================================
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# =====================================================================
# SECTION 1 — CONFIGURATION
# =====================================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

SAMPLE_RATE = 24000

HOP_LENGTH = 240

FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE

BLANK_TOKEN = "<blank>"

VOCABULARY = [
    BLANK_TOKEN,
    "H",
    "E",
    "L",
    "O",
    "W",
    "R",
    "D"
]

TOKEN_TO_ID = {
    token: index
    for index, token in enumerate(VOCABULARY)
}

ID_TO_TOKEN = {
    index: token
    for token, index in TOKEN_TO_ID.items()
}


# =====================================================================
# SECTION 2 — NUMERICAL UTILITIES
# =====================================================================

def logsumexp(values):

    """
    Numerically stable log-sum-exp.

    logsumexp(x) = log(sum(exp(x)))

    We subtract the maximum before exponentiation to avoid
    numerical overflow.
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    maximum = np.max(values)

    if np.isneginf(maximum):

        return -np.inf

    return maximum + np.log(
        np.sum(
            np.exp(values - maximum)
        )
    )


def softmax(logits, axis=-1):

    """
    Convert logits into normalized probabilities.

    softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    """

    logits = np.asarray(
        logits,
        dtype=np.float64
    )

    shifted = (
        logits - np.max(
            logits,
            axis=axis,
            keepdims=True
        )
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=axis,
        keepdims=True
    )


def log_softmax(logits, axis=-1):

    """
    Convert logits into log probabilities.

    This is preferable when calculating long sequence probabilities.
    """

    logits = np.asarray(
        logits,
        dtype=np.float64
    )

    maximum = np.max(
        logits,
        axis=axis,
        keepdims=True
    )

    shifted = logits - maximum

    log_normalizer = np.log(
        np.sum(
            np.exp(shifted),
            axis=axis,
            keepdims=True
        )
    )

    return shifted - log_normalizer


# =====================================================================
# SECTION 3 — CTC COLLAPSE
# =====================================================================

def ctc_collapse(path, blank_id=0):

    """
    Apply the CTC collapse operation.

    Operation order:

        1. Merge consecutive repeated tokens.
        2. Remove blank tokens.

    Example:

        Input:
            [0, 1, 1, 0, 2, 2, 3, 0]

        Output:
            [1, 2, 3]

    IMPORTANT:

    Blanks separate repeated labels.

        [3, 0, 3]

    becomes:

        [3, 3]

    It must NOT be treated as one repeated run.
    """

    output = []

    previous = None

    for token in path:

        token = int(token)

        if token != previous:

            if token != blank_id:

                output.append(token)

        previous = token

    return output


# =====================================================================
# SECTION 4 — TARGET EXPANSION
# =====================================================================

def expand_ctc_target(
    target,
    blank_id=0
):

    """
    Insert blank states around and between target tokens.

    Target:

        H E L L O

    Expanded target:

        blank H blank E blank L blank L blank O blank

    For N tokens:

        S = 2N + 1

    """

    target = list(target)

    expanded = [blank_id]

    for token in target:

        expanded.append(int(token))

        expanded.append(blank_id)

    return expanded


# =====================================================================
# SECTION 5 — ALIGNMENT DATA STRUCTURES
# =====================================================================

@dataclass
class TokenAlignment:

    token: str

    token_id: int

    start_frame: int

    end_frame: int

    start_time: float

    end_time: float

    confidence: float


@dataclass
class AlignmentResult:

    target_tokens: List[int]

    expanded_target: List[int]

    frame_path: List[int]

    token_alignments: List[TokenAlignment]

    total_log_probability: float

    success: bool

    message: str


# =====================================================================
# SECTION 6 — CTC VITERBI FORCED ALIGNMENT
# =====================================================================

def ctc_viterbi_align(
    log_probs,
    target,
    vocabulary,
    blank_id=0,
    frame_duration=0.01
):

    """
    Perform CTC Viterbi forced alignment.

    Parameters
    ----------

    log_probs:

        Shape:

            (T, V)

        T = number of acoustic frames.

        V = vocabulary size.

        Each row contains log probabilities over vocabulary tokens.

    target:

        List of target token IDs.

    vocabulary:

        List mapping token IDs to token strings.

    blank_id:

        ID of the CTC blank token.

    frame_duration:

        Duration represented by one acoustic frame.

    Returns
    -------

    AlignmentResult

    Algorithm
    ---------

    1. Expand target with blank tokens.
    2. Initialize dynamic programming table.
    3. Evaluate valid CTC transitions.
    4. Store predecessor states.
    5. Select the best terminal state.
    6. Backtrack to obtain frame-level alignment.
    7. Convert frame path into token timestamps.

    """

    log_probs = np.asarray(
        log_probs,
        dtype=np.float64
    )

    target = list(target)

    if log_probs.ndim != 2:

        raise ValueError(
            "log_probs must have shape (T, V)."
        )

    T, V = log_probs.shape

    if V != len(vocabulary):

        raise ValueError(
            "Vocabulary size does not match log_probs."
        )

    if len(target) == 0:

        raise ValueError(
            "Target sequence cannot be empty."
        )

    if any(
        token < 0 or token >= V
        for token in target
    ):

        raise ValueError(
            "Target contains invalid token IDs."
        )

    if blank_id in target:

        raise ValueError(
            "Target must not contain the CTC blank token."
        )

    expanded = expand_ctc_target(
        target,
        blank_id
    )

    S = len(expanded)

    # ---------------------------------------------------------------
    # Check whether enough frames exist.
    #
    # Repeated consecutive target tokens require at least one
    # separating blank frame in a valid CTC path.
    # ---------------------------------------------------------------

    minimum_frames = len(target)

    for i in range(1, len(target)):

        if target[i] == target[i - 1]:

            minimum_frames += 1

    if T < minimum_frames:

        raise ValueError(
            f"Not enough frames. Need at least "
            f"{minimum_frames}, got {T}."
        )

    # ---------------------------------------------------------------
    # Dynamic programming table.
    #
    # dp[t, s] = maximum log probability of a path ending at
    # frame t and expanded target state s.
    # ---------------------------------------------------------------

    dp = np.full(
        (T, S),
        -np.inf,
        dtype=np.float64
    )

    # ---------------------------------------------------------------
    # Backpointer table.
    #
    # backpointer[t, s] stores the predecessor state.
    #
    # -1 means unreachable.
    # ---------------------------------------------------------------

    backpointer = np.full(
        (T, S),
        -1,
        dtype=np.int64
    )

    # ---------------------------------------------------------------
    # Initialization at frame zero.
    #
    # Initially, the model can emit:
    #
    #   1. Blank
    #   2. First target token
    #
    # ---------------------------------------------------------------

    dp[0, 0] = log_probs[
        0,
        expanded[0]
    ]

    dp[0, 1] = log_probs[
        0,
        expanded[1]
    ]

    # ---------------------------------------------------------------
    # Dynamic programming.
    # ---------------------------------------------------------------

    for t in range(1, T):

        for s in range(S):

            candidates = []

            predecessor_states = []

            # -------------------------------------------------------
            # Transition 1:
            #
            # Stay in the same state.
            #
            # Example:
            #
            # H H H
            #
            # -------------------------------------------------------

            candidates.append(
                dp[t - 1, s]
            )

            predecessor_states.append(s)

            # -------------------------------------------------------
            # Transition 2:
            #
            # Move from the previous state.
            #
            # Example:
            #
            # blank -> H
            #
            # -------------------------------------------------------

            if s >= 1:

                candidates.append(
                    dp[t - 1, s - 1]
                )

                predecessor_states.append(
                    s - 1
                )

            # -------------------------------------------------------
            # Transition 3:
            #
            # Skip one state.
            #
            # This transition is permitted only when:
            #
            # 1. Destination is not blank.
            # 2. Destination differs from state s-2.
            #
            # This is required by CTC topology.
            # -------------------------------------------------------

            if s >= 2:

                destination_token = expanded[s]

                previous_token = expanded[s - 2]

                can_skip = (

                    destination_token != blank_id

                    and destination_token != previous_token

                )

                if can_skip:

                    candidates.append(
                        dp[t - 1, s - 2]
                    )

                    predecessor_states.append(
                        s - 2
                    )

            # -------------------------------------------------------
            # Choose the best predecessor.
            # -------------------------------------------------------

            best_index = int(
                np.argmax(candidates)
            )

            best_previous_score = candidates[
                best_index
            ]

            best_previous_state = predecessor_states[
                best_index
            ]

            if np.isfinite(best_previous_score):

                dp[t, s] = (

                    best_previous_score

                    + log_probs[
                        t,
                        expanded[s]
                    ]

                )

                backpointer[
                    t,
                    s
                ] = best_previous_state

    # ---------------------------------------------------------------
    # Select final state.
    #
    # A valid CTC path must finish in:
    #
    # 1. Final target token state.
    # 2. Final blank state.
    # ---------------------------------------------------------------

    final_candidates = [

        S - 1,

        S - 2

    ]

    final_scores = [

        dp[T - 1, state]

        for state in final_candidates

    ]

    best_final_index = int(
        np.argmax(final_scores)
    )

    final_state = final_candidates[
        best_final_index
    ]

    final_score = final_scores[
        best_final_index
    ]

    if not np.isfinite(final_score):

        return AlignmentResult(

            target_tokens=target,

            expanded_target=expanded,

            frame_path=[],

            token_alignments=[],

            total_log_probability=-np.inf,

            success=False,

            message=(
                "No valid CTC alignment path was found."
            )

        )

    # ---------------------------------------------------------------
    # Backtracking.
    #
    # Recover the best state for every frame.
    # ---------------------------------------------------------------

    state_path = np.zeros(
        T,
        dtype=np.int64
    )

    state_path[T - 1] = final_state

    for t in range(T - 1, 0, -1):

        previous_state = backpointer[
            t,
            state_path[t]
        ]

        if previous_state < 0:

            raise RuntimeError(
                "Invalid backpointer during backtracking."
            )

        state_path[t - 1] = previous_state

    # ---------------------------------------------------------------
    # Convert expanded states into token IDs.
    # ---------------------------------------------------------------

    frame_path = [

        expanded[state]

        for state in state_path

    ]

    # ---------------------------------------------------------------
    # Build token timestamps.
    #
    # Every target token occupies an odd state:
    #
    #   1, 3, 5, 7, ...
    #
    # We extract the frames assigned to each target state.
    # ---------------------------------------------------------------

    token_alignments = []

    for target_index, token_id in enumerate(target):

        state_index = (
            2 * target_index + 1
        )

        frame_indices = np.where(
            state_path == state_index
        )[0]

        if len(frame_indices) == 0:

            continue

        start_frame = int(
            frame_indices[0]
        )

        end_frame = int(
            frame_indices[-1] + 1
        )

        # -----------------------------------------------------------
        # Mean frame probability for the aligned token.
        #
        # This is an approximate local confidence measure.
        # It is not a calibrated probability that the timestamp
        # is correct.
        # -----------------------------------------------------------

        frame_probabilities = np.exp(
            log_probs[
                frame_indices,
                token_id
            ]
        )

        confidence = float(
            np.mean(frame_probabilities)
        )

        start_time = (
            start_frame * frame_duration
        )

        end_time = (
            end_frame * frame_duration
        )

        token_alignments.append(

            TokenAlignment(

                token=vocabulary[token_id],

                token_id=token_id,

                start_frame=start_frame,

                end_frame=end_frame,

                start_time=start_time,

                end_time=end_time,

                confidence=confidence

            )

        )

    return AlignmentResult(

        target_tokens=target,

        expanded_target=expanded,

        frame_path=frame_path,

        token_alignments=token_alignments,

        total_log_probability=float(
            final_score
        ),

        success=True,

        message="Alignment completed successfully."

    )


# =====================================================================
# SECTION 7 — WORD ALIGNMENT
# =====================================================================

def group_tokens_into_words(
    token_alignments,
    word_token_counts
):

    """
    Group character/phoneme timestamps into word timestamps.

    Example:

        Token sequence:

            H E L L O W O R L D

        Word token counts:

            [5, 5]

        Words:

            HELLO
            WORLD

    This function assumes that the token order matches the word order.

    Real multilingual systems require language-aware tokenization.
    """

    word_alignments = []

    cursor = 0

    for word_index, count in enumerate(
        word_token_counts
    ):

        word_tokens = token_alignments[
            cursor:cursor + count
        ]

        cursor += count

        if not word_tokens:

            continue

        word_alignments.append({

            "word_index": word_index,

            "start_time": min(
                token.start_time
                for token in word_tokens
            ),

            "end_time": max(
                token.end_time
                for token in word_tokens
            ),

            "duration": (

                max(
                    token.end_time
                    for token in word_tokens
                )

                -

                min(
                    token.start_time
                    for token in word_tokens
                )

            ),

            "tokens": [
                token.token
                for token in word_tokens
            ]

        })

    if cursor != len(token_alignments):

        print(
            "WARNING: Some tokens were not assigned to a word."
        )

    return word_alignments


# =====================================================================
# SECTION 8 — SYNTHETIC ACOUSTIC PROBABILITIES
# =====================================================================

def create_synthetic_emissions(
    target,
    vocabulary,
    frames_per_token=5,
    blank_frames=2,
    noise_scale=0.8
):

    """
    Generate synthetic frame-level acoustic probabilities.

    This function simulates the output of an acoustic neural network.

    It does NOT process real audio.

    Each target token receives a period of relatively high probability.

    Blank periods are inserted between tokens.

    Small random noise is added to make the problem less trivial.

    Returns:

        log_probs
        true_frame_path
    """

    target = list(target)

    blank_id = TOKEN_TO_ID[BLANK_TOKEN]

    frame_path = []

    for index, token in enumerate(target):

        if index > 0:

            # Insert blank frames between tokens.
            frame_path.extend(
                [blank_id] * blank_frames
            )

        frame_path.extend(
            [token] * frames_per_token
        )

    # Add leading and trailing blanks.
    frame_path = (
        [blank_id, blank_id]
        + frame_path
        + [blank_id, blank_id]
    )

    T = len(frame_path)

    V = len(vocabulary)

    logits = np.random.normal(
        loc=0.0,
        scale=noise_scale,
        size=(T, V)
    )

    # ---------------------------------------------------------------
    # Increase the logit for the intended token at each frame.
    # ---------------------------------------------------------------

    for t, token_id in enumerate(frame_path):

        logits[t, token_id] += 5.0

    probabilities = softmax(
        logits,
        axis=1
    )

    log_probs = np.log(
        np.maximum(
            probabilities,
            1e-12
        )
    )

    return log_probs, frame_path


# =====================================================================
# SECTION 9 — PRINT ALIGNMENT
# =====================================================================

def print_alignment(result):

    """
    Print token-level alignment in a readable table.
    """

    if not result.success:

        print(result.message)

        return

    print("\n====================================================")

    print("TOKEN-LEVEL FORCED ALIGNMENT")

    print("====================================================")

    print(

        f"{'Token':<12}"

        f"{'Start':>12}"

        f"{'End':>12}"

        f"{'Duration':>12}"

        f"{'Confidence':>14}"

    )

    print("-" * 62)

    for item in result.token_alignments:

        duration = (
            item.end_time - item.start_time
        )

        print(

            f"{item.token:<12}"

            f"{item.start_time:>12.3f}"

            f"{item.end_time:>12.3f}"

            f"{duration:>12.3f}"

            f"{item.confidence:>14.3f}"

        )

    print("-" * 62)

    print(
        "Total log probability:",
        result.total_log_probability
    )


# =====================================================================
# SECTION 10 — VISUALIZATION
# =====================================================================

def visualize_alignment(
    log_probs,
    result,
    vocabulary,
    title="CTC Forced Alignment"
):

    """
    Visualize:

        1. Acoustic probability matrix.
        2. Viterbi alignment path.
        3. Token timestamps.

    The heatmap shows the probability of each vocabulary token
    at every frame.

    The red line shows the token selected by Viterbi alignment.
    """

    if not result.success:

        print(
            "Cannot visualize unsuccessful alignment."
        )

        return

    probabilities = np.exp(
        log_probs
    )

    T, V = probabilities.shape

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        gridspec_kw={
            "height_ratios": [2, 1]
        }
    )

    # ---------------------------------------------------------------
    # Heatmap
    # ---------------------------------------------------------------

    axes[0].imshow(

        probabilities.T,

        aspect="auto",

        origin="lower",

        interpolation="nearest"

    )

    axes[0].set_yticks(
        range(V)
    )

    axes[0].set_yticklabels(
        vocabulary
    )

    axes[0].set_xlabel(
        "Acoustic frame"
    )

    axes[0].set_ylabel(
        "Token"
    )

    axes[0].set_title(
        "Frame-Level Acoustic Probabilities"
    )

    # ---------------------------------------------------------------
    # Viterbi path
    # ---------------------------------------------------------------

    axes[0].plot(

        range(T),

        result.frame_path,

        color="red",

        linewidth=2,

        label="Viterbi alignment"

    )

    axes[0].legend()

    # ---------------------------------------------------------------
    # Token duration bars
    # ---------------------------------------------------------------

    for index, item in enumerate(
        result.token_alignments
    ):

        axes[1].barh(

            y=index,

            width=(
                item.end_time - item.start_time
            ),

            left=item.start_time,

            height=0.6

        )

        axes[1].text(

            item.start_time,

            index,

            f" {item.token}",

            va="center"

        )

    axes[1].set_xlabel(
        "Time (seconds)"
    )

    axes[1].set_ylabel(
        "Target token"
    )

    axes[1].set_title(
        "Estimated Token Time Intervals"
    )

    axes[1].grid(
        axis="x",
        alpha=0.3
    )

    plt.suptitle(title)

    plt.tight_layout()

    plt.show()


# =====================================================================
# SECTION 11 — EXPERIMENT 1
# BASIC CTC COLLAPSE
# =====================================================================

def experiment_1():

    """
    Demonstrate CTC collapse.
    """

    print("\nEXPERIMENT 1: CTC COLLAPSE")

    print("=" * 50)

    example = [

        0,

        1, 1,

        0,

        2, 2,

        3, 3,

        0,

        3,

        4, 4,

        0

    ]

    print("Input IDs:")

    print(example)

    print("\nInput tokens:")

    print([
        ID_TO_TOKEN[x]
        for x in example
    ])

    collapsed = ctc_collapse(
        example,
        blank_id=0
    )

    print("\nCollapsed IDs:")

    print(collapsed)

    print("\nCollapsed tokens:")

    print([
        ID_TO_TOKEN[x]
        for x in collapsed
    ])


# =====================================================================
# SECTION 12 — EXPERIMENT 2
# TARGET EXPANSION
# =====================================================================

def experiment_2():

    """
    Demonstrate CTC target expansion.
    """

    print("\nEXPERIMENT 2: CTC TARGET EXPANSION")

    print("=" * 50)

    target_text = "HELLO"

    target = [
        TOKEN_TO_ID[c]
        for c in target_text
    ]

    expanded = expand_ctc_target(
        target,
        blank_id=0
    )

    print("Target:")

    print(target_text)

    print("\nExpanded sequence:")

    print([
        ID_TO_TOKEN[x]
        for x in expanded
    ])

    print("\nNumber of target tokens:", len(target))

    print("Number of expanded states:", len(expanded))


# =====================================================================
# SECTION 13 — EXPERIMENT 3
# FORCED ALIGNMENT
# =====================================================================

def experiment_3():

    """
    Align HELLO using synthetic acoustic probabilities.
    """

    print("\nEXPERIMENT 3: FORCED ALIGNMENT")

    print("=" * 50)

    target_text = "HELLO"

    target = [

        TOKEN_TO_ID[c]

        for c in target_text

    ]

    log_probs, true_path = create_synthetic_emissions(

        target=target,

        vocabulary=VOCABULARY,

        frames_per_token=6,

        blank_frames=2

    )

    result = ctc_viterbi_align(

        log_probs=log_probs,

        target=target,

        vocabulary=VOCABULARY,

        blank_id=0,

        frame_duration=FRAME_DURATION

    )

    print_alignment(result)

    print("\nTrue synthetic frame path:")

    print([
        ID_TO_TOKEN[x]
        for x in true_path
    ])

    print("\nRecovered frame path:")

    print([
        ID_TO_TOKEN[x]
        for x in result.frame_path
    ])

    visualize_alignment(
        log_probs,
        result,
        VOCABULARY,
        title="HELLO — Synthetic CTC Alignment"
    )


# =====================================================================
# SECTION 14 — EXPERIMENT 4
# WORD TIMESTAMPS
# =====================================================================

def experiment_4():

    """
    Align HELLO WORLD and combine character timestamps into words.
    """

    print("\nEXPERIMENT 4: WORD-LEVEL ALIGNMENT")

    print("=" * 50)

    target_text = "HELLOWORLD"

    target = [

        TOKEN_TO_ID[c]

        for c in target_text

    ]

    log_probs, true_path = create_synthetic_emissions(

        target=target,

        vocabulary=VOCABULARY,

        frames_per_token=5,

        blank_frames=2

    )

    result = ctc_viterbi_align(

        log_probs=log_probs,

        target=target,

        vocabulary=VOCABULARY,

        blank_id=0,

        frame_duration=FRAME_DURATION

    )

    print_alignment(result)

    word_alignments = group_tokens_into_words(

        token_alignments=result.token_alignments,

        word_token_counts=[5, 5]

    )

    print("\n====================================================")

    print("WORD-LEVEL TIMESTAMPS")

    print("====================================================")

    print(
        f"{'Word':<12}"
        f"{'Start':>12}"
        f"{'End':>12}"
        f"{'Duration':>12}"
    )

    print("-" * 48)

    words = [
        "HELLO",
        "WORLD"
    ]

    for word, alignment in zip(
        words,
        word_alignments
    ):

        print(

            f"{word:<12}"

            f"{alignment['start_time']:>12.3f}"

            f"{alignment['end_time']:>12.3f}"

            f"{alignment['duration']:>12.3f}"

        )

    visualize_alignment(

        log_probs,

        result,

        VOCABULARY,

        title="HELLO WORLD — Word-Level Alignment"

    )


# =====================================================================
# SECTION 15 — EXPERIMENT 5
# REPEATED TOKENS
# =====================================================================

def experiment_5():

    """
    Demonstrate why repeated characters require blank separation.

    Target:

        L L

    A valid CTC path must include a blank between the two L tokens.

    Example:

        L _ L

    """

    print("\nEXPERIMENT 5: REPEATED TOKENS")

    print("=" * 50)

    target = [

        TOKEN_TO_ID["L"],

        TOKEN_TO_ID["L"]

    ]

    expanded = expand_ctc_target(
        target,
        blank_id=0
    )

    print("Target:")

    print(["L", "L"])

    print("\nExpanded target:")

    print([
        ID_TO_TOKEN[x]
        for x in expanded
    ])

    print("\nMinimum valid number of frames:")

    print(3)

    print("\nValid path:")

    print([
        "L",
        "<blank>",
        "L"
    ])

    print("\nInvalid path for producing two L tokens:")

    print([
        "L",
        "L"
    ])

    print(
        "\nThe invalid path collapses to only one L."
    )


# =====================================================================
# SECTION 16 — EXPERIMENT 6
# ALIGNMENT SENSITIVITY TO ACOUSTIC PROBABILITIES
# =====================================================================

def experiment_6():

    """
    Compare alignment results under different levels of acoustic
    uncertainty.

    This experiment changes the synthetic noise level.

    A real acoustic model may produce uncertain token probabilities
    due to noise, pronunciation variation, or poor recording quality.

    The Viterbi algorithm still returns the best valid path if one
    exists, but that does not guarantee the alignment is correct.
    """

    print("\nEXPERIMENT 6: ACOUSTIC UNCERTAINTY")

    print("=" * 50)

    target_text = "HELLO"

    target = [
        TOKEN_TO_ID[c]
        for c in target_text
    ]

    noise_levels = [
        0.3,
        1.0,
        2.0,
        3.0
    ]

    for noise in noise_levels:

        log_probs, _ = create_synthetic_emissions(

            target=target,

            vocabulary=VOCABULARY,

            frames_per_token=6,

            blank_frames=2,

            noise_scale=noise

        )

        result = ctc_viterbi_align(

            log_probs=log_probs,

            target=target,

            vocabulary=VOCABULARY,

            blank_id=0,

            frame_duration=FRAME_DURATION

        )

        print(
            f"\nNoise scale: {noise}"
        )

        print(
            "Alignment success:",
            result.success
        )

        if result.success:

            print(
                "Log probability:",
                result.total_log_probability
            )

            print(
                "Recovered path:",
                [
                    ID_TO_TOKEN[x]
                    for x in result.frame_path
                ]
            )

            print(
                "Token durations:",
                [
                    round(
                        item.end_time - item.start_time,
                        3
                    )
                    for item in result.token_alignments
                ]
            )


# =====================================================================
# SECTION 17 — RUN ALL EXPERIMENTS
# =====================================================================

def run_all():

    experiment_1()

    experiment_2()

    experiment_3()

    experiment_4()

    experiment_5()

    experiment_6()


# =====================================================================
# SECTION 18 — MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "CTC Forced Alignment Educational Implementation"
        )

    )

    parser.add_argument(

        "--experiment",

        type=int,

        choices=[
            1,
            2,
            3,
            4,
            5,
            6
        ],

        help="Run a specific experiment."

    )

    args = parser.parse_args()

    if args.experiment is None:

        run_all()

    elif args.experiment == 1:

        experiment_1()

    elif args.experiment == 2:

        experiment_2()

    elif args.experiment == 3:

        experiment_3()

    elif args.experiment == 4:

        experiment_4()

    elif args.experiment == 5:

        experiment_5()

    elif args.experiment == 6:

        experiment_6()


if __name__ == "__main__":

    main()


"""
=======================================================================
END OF LESSON 9
=======================================================================

WHAT WE IMPLEMENTED
-------------------

You implemented:

    1. CTC collapse.
    2. CTC target expansion.
    3. Log probability calculations.
    4. Dynamic programming.
    5. Viterbi decoding.
    6. Backtracking.
    7. Frame-level token alignment.
    8. Token timestamps.
    9. Word timestamps.
    10. Alignment visualization.

WHAT WE DID NOT IMPLEMENT
-------------------------

We did not implement a trained acoustic model.

We used synthetic probabilities to isolate and understand
the alignment algorithm.

A complete real-world forced alignment pipeline needs:

    Audio
      |
      v
    Feature extraction
      |
      v
    Trained acoustic model
      |
      v
    Frame-level log probabilities
      |
      v
    CTC forced alignment
      |
      v
    Token timestamps

NEXT LEARNING OBJECTIVE
-----------------------

The next step is to understand how speech is represented as
machine-learning input and how neural acoustic models extract
useful speech information.

This will prepare us for implementing the first neural
speech representation model.

=======================================================================
"""