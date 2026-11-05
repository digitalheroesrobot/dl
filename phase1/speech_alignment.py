"""
File: phase1_lesson13_speech_alignment.py

PHASE 1 — LESSON 13
SPEECH ALIGNMENT, DURATION MODELING,
AND PHONEME-TO-AUDIO TIMING
============================================================

PURPOSE
-------
Understand how a TTS training system obtains phoneme timing
information from a known transcript and its corresponding audio.

This lesson implements an educational CTC-style Viterbi aligner
using NumPy and dynamic programming.

IMPORTANT LIMITATION
--------------------
The acoustic probabilities in this experiment are SYNTHETIC.

The script demonstrates the alignment algorithm, but it does NOT
perform real forced alignment on an arbitrary recording.

Real forced alignment requires:
    1. A known transcript.
    2. A phoneme sequence produced by a language frontend.
    3. Acoustic probabilities from a trained acoustic model,
       such as a CTC speech recognition model.
    4. A decoding/alignment algorithm.

No pretrained model is used in this lesson.

============================================================
1. FOUR DIFFERENT CONCEPTS
============================================================

A. UTTERANCE-LEVEL PAIRING

    recording.wav <-> "Hello world"

This only tells us that the complete recording corresponds to
the complete transcript.

It does not tell us where individual phonemes occur.

B. FORCED ALIGNMENT

Given:
    - known audio
    - known transcript
    - known phoneme sequence

Estimate:
    - start time of each phoneme
    - end time of each phoneme
    - word boundaries

C. ASR TIMESTAMPING

An automatic speech recognition system recognizes speech and
may estimate word or segment timestamps.

This is not necessarily the same as forced alignment because
ASR must also infer the words.

D. TTS DURATION PREDICTION

Given a phoneme sequence, predict how long each phoneme should
last before generating speech.

For example:

    Phonemes:       h   ə   l   oʊ
    Durations:     4   3   5    8 frames

At a 10 ms frame hop:

    h   = 40 ms
    ə   = 30 ms
    l   = 50 ms
    oʊ  = 80 ms

The duration targets can be obtained from aligned training audio.

============================================================
2. WHAT IS ACOUSTIC ALIGNMENT?
============================================================

An audio waveform is a sequence of samples:

    x[0], x[1], x[2], ..., x[N-1]

A speech feature extractor converts it into acoustic frames:

    X = [x_1, x_2, ..., x_T]

Each frame represents a short time region of audio.

For example, with:
    sample rate = 24,000 Hz
    frame hop   = 240 samples

The frame interval is:

    240 / 24000 = 0.01 seconds = 10 ms

An acoustic model estimates probabilities such as:

    P(phone = "h" | acoustic frame t)

For CTC, the model generally predicts probabilities over:

    phoneme labels + a special blank symbol

The blank symbol means that a frame does not emit a new
phoneme label.

============================================================
3. CTC LABEL EXPANSION
============================================================

Suppose the target phoneme sequence is:

    h, ə, l

CTC expands it by inserting blank symbols:

    blank, h, blank, ə, blank, l, blank

The expanded sequence is called the CTC state sequence.

A valid alignment path may remain in a state for multiple
frames.

Example:

    blank, blank, h, h, blank, ə, ə, blank, l, l, blank

Collapsing repeated labels and removing blanks produces:

    h, ə, l

Repeated adjacent phonemes require a blank between their
emissions in a valid CTC path.

For example, target:

    l, l

Expanded states:

    blank, l, blank, l, blank

The blank is necessary to distinguish two adjacent occurrences
of the same phoneme.

============================================================
4. VITERBI DYNAMIC PROGRAMMING
============================================================

Let:
    T = number of acoustic frames
    S = number of expanded CTC states
    z_s = label associated with expanded state s
    logp[t, k] = log probability of label k at frame t

Define:

    V[t, s]

as the maximum log probability of any valid path that ends
at expanded state s at frame t.

The recurrence is:

    V[t,s] = logp[t,z_s] + max(
        V[t-1,s],
        V[t-1,s-1],
        V[t-1,s-2] if skip is allowed
    )

The three transitions mean:

    1. Stay in the same state.
    2. Move forward by one state.
    3. Skip a blank state and move forward by two states.

The two-state skip is permitted only when:
    - the destination state is not blank
    - the destination label differs from the label two
      states earlier

That restriction prevents repeated adjacent labels from
being incorrectly merged.

After filling the dynamic programming table, backtracking
recovers the best state sequence.

This is Viterbi decoding: it finds the single highest-scoring
alignment path.

CTC training also uses a forward algorithm that sums the
probabilities of valid paths. Viterbi instead selects the
single best path.

============================================================
5. FROM FRAME INDICES TO TIME
============================================================

If the frame hop is H samples and the sample rate is Fs:

    frame_time = frame_index * H / Fs

For example:

    H  = 240 samples
    Fs = 24000 Hz

    frame 10 = 10 * 240 / 24000
             = 0.10 seconds

A phoneme's duration is:

    duration = end_time - start_time

The frame interval is treated as:

    [start_frame, end_frame)

The end frame is exclusive.

============================================================
6. ALIGNMENT TARGETS FOR TTS
============================================================

Once phoneme timing has been estimated, we can obtain:

    d_i = number of frames assigned to phoneme i

A duration predictor learns:

    predicted_duration_i = f(text/phoneme representation)

A common educational loss is log-duration L1:

    L_duration =
        mean(abs(log(d_i + epsilon)
                 - log(predicted_d_i + epsilon)))

Log durations reduce the influence of very long phonemes
and make relative duration errors easier to model.

In a complete neural TTS system, duration is also affected by:
    - speaking rate
    - stress
    - lexical tone
    - pitch accent
    - phrase position
    - pauses
    - emotion and speaking style

============================================================
7. LANGUAGE EXAMPLES
============================================================

This script includes simplified token sequences for:

    English
    Mandarin Chinese
    Japanese
    Russian

These examples are intentionally small.

They are NOT complete phoneme inventories or full G2P systems.

A production system needs a language-specific frontend that
handles pronunciation, stress, tone, pitch accent, and
text normalization.

============================================================
REQUIREMENTS
============================================================

    pip install numpy matplotlib

RUN
---

    python phase1_lesson13_speech_alignment.py

    python phase1_lesson13_speech_alignment.py --language en

    python phase1_lesson13_speech_alignment.py --language zh

    python phase1_lesson13_speech_alignment.py --language ja

    python phase1_lesson13_speech_alignment.py --language ru

    python phase1_lesson13_speech_alignment.py --no-plot

============================================================
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 24000
HOP_LENGTH = 240

FRAME_INTERVAL = HOP_LENGTH / SAMPLE_RATE

BLANK = "<blank>"

EPSILON = 1e-8


# ============================================================
# LANGUAGE EXAMPLES
# ============================================================

LANGUAGE_EXAMPLES = {
    "en": {
        "name": "English",
        "words": ["hello", "world"],
        "phones": [
            "h", "ə", "l", "oʊ",
            "w", "ɜː", "r", "l", "d"
        ],
        "word_ids": [
            0, 0, 0, 0,
            1, 1, 1, 1, 1
        ],
    },

    "zh": {
        "name": "Mandarin Chinese",
        "words": ["ni", "hao"],
        "phones": [
            "n", "i˧˩˧",
            "x", "ɑʊ˥˩"
        ],
        "word_ids": [
            0, 0,
            1, 1
        ],
    },

    "ja": {
        "name": "Japanese",
        "words": ["konnichiwa"],
        "phones": [
            "k", "o", "N", "n",
            "i", "tɕ", "i", "w", "a"
        ],
        "word_ids": [
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
    },

    "ru": {
        "name": "Russian",
        "words": ["privet"],
        "phones": [
            "p", "r", "ʲ", "i",
            "v", "ʲ", "e", "t"
        ],
        "word_ids": [
            0, 0, 0, 0, 0, 0, 0, 0
        ],
    },
}


# ============================================================
# CTC STATE EXPANSION
# ============================================================

def expand_ctc_labels(phones):
    """
    Convert:

        ["h", "ə", "l"]

    into:

        [
            "<blank>",
            "h",
            "<blank>",
            "ə",
            "<blank>",
            "l",
            "<blank>"
        ]

    State indices:
        0 = blank
        1 = h
        2 = blank
        3 = ə
        4 = blank
        5 = l
        6 = blank
    """

    expanded = [BLANK]

    for phone in phones:
        expanded.append(phone)
        expanded.append(BLANK)

    return expanded


# ============================================================
# SYNTHETIC ACOUSTIC EVIDENCE
# ============================================================

def stable_softmax(logits):
    """
    Numerically stable softmax.
    """

    shifted = logits - np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=-1,
        keepdims=True
    )


def generate_synthetic_emissions(
    phones,
    seed=42,
    min_phone_frames=3,
    max_phone_frames=9,
):
    """
    Generate synthetic frame-level acoustic probabilities.

    This function is ONLY a teaching substitute for a trained
    acoustic model.

    Each phoneme receives a random duration.

    Blank frames are inserted between phonemes.

    The synthetic model produces high probabilities for the
    expected frame labels, with noise added to make the alignment
    problem less trivial.

    Returns:
        log_probs:
            Shape (T, vocabulary_size)

        vocabulary:
            List of output labels

        true_frame_labels:
            Synthetic ground-truth label for each frame
    """

    rng = np.random.default_rng(seed)

    vocabulary = [BLANK] + sorted(set(phones))

    label_to_id = {
        label: index
        for index, label in enumerate(vocabulary)
    }

    true_frame_labels = []

    # Initial blank frames.
    initial_blank = 2

    true_frame_labels.extend(
        [BLANK] * initial_blank
    )

    true_phone_durations = []

    for phone in phones:

        duration = int(
            rng.integers(
                min_phone_frames,
                max_phone_frames + 1
            )
        )

        true_phone_durations.append(duration)

        true_frame_labels.extend(
            [phone] * duration
        )

        # Blank separator between phones.
        blank_duration = int(
            rng.integers(1, 3)
        )

        true_frame_labels.extend(
            [BLANK] * blank_duration
        )

    # Final blank frames.
    true_frame_labels.extend(
        [BLANK] * 2
    )

    num_frames = len(true_frame_labels)
    num_labels = len(vocabulary)

    logits = rng.normal(
        loc=0.0,
        scale=0.9,
        size=(num_frames, num_labels)
    )

    # Make the synthetic target label more likely.
    for frame_index, label in enumerate(
        true_frame_labels
    ):

        target_id = label_to_id[label]

        logits[frame_index, target_id] += 5.0

    probabilities = stable_softmax(logits)

    log_probs = np.log(
        np.maximum(probabilities, EPSILON)
    )

    return (
        log_probs,
        vocabulary,
        true_frame_labels,
        true_phone_durations
    )


# ============================================================
# CTC VITERBI ALIGNMENT
# ============================================================

def ctc_viterbi_align(
    log_probs,
    vocabulary,
    phones,
):
    """
    Find the highest-scoring CTC alignment path.

    Inputs:
        log_probs:
            (T, V) matrix of frame-level log probabilities.

        vocabulary:
            Model output labels.

        phones:
            Known target phoneme sequence.

    Returns:
        state_path:
            Expanded-state index for every frame.

        expanded:
            CTC-expanded target sequence.

        best_score:
            Score of the best alignment.

    This function implements Viterbi decoding, not CTC training.
    """

    num_frames, vocab_size = log_probs.shape

    label_to_id = {
        label: index
        for index, label in enumerate(vocabulary)
    }

    for phone in phones:
        if phone not in label_to_id:
            raise ValueError(
                f"Phone {phone!r} is missing from vocabulary."
            )

    expanded = expand_ctc_labels(phones)

    num_states = len(expanded)

    # Dynamic programming table.
    #
    # dp[t, s] = best log score for paths ending at state s
    #            at acoustic frame t.
    dp = np.full(
        (num_frames, num_states),
        -np.inf,
        dtype=np.float64
    )

    # Backpointer stores the previous state.
    backpointer = np.full(
        (num_frames, num_states),
        -1,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    # At frame zero, a valid path can begin at:
    #   state 0: blank
    #   state 1: first phoneme
    #
    # All other states are unreachable initially.

    first_label_id = label_to_id[expanded[0]]

    dp[0, 0] = log_probs[0, first_label_id]

    if num_states > 1:

        second_label_id = label_to_id[expanded[1]]

        dp[0, 1] = log_probs[0, second_label_id]

    # --------------------------------------------------------
    # DYNAMIC PROGRAMMING
    # --------------------------------------------------------

    for t in range(1, num_frames):

        for s in range(num_states):

            current_label = expanded[s]

            current_label_id = label_to_id[current_label]

            emission_score = log_probs[
                t,
                current_label_id
            ]

            # Candidate 1: stay in current state.
            candidates = [
                (dp[t - 1, s], s)
            ]

            # Candidate 2: advance by one state.
            if s >= 1:

                candidates.append(
                    (dp[t - 1, s - 1], s - 1)
                )

            # Candidate 3: skip a blank state.
            #
            # Skip is valid only if:
            #   - destination is not blank
            #   - destination label differs from
            #     the label two states earlier
            #
            # This prevents repeated labels from being
            # incorrectly merged.
            if s >= 2:

                previous_two_label = expanded[s - 2]

                skip_allowed = (
                    current_label != BLANK
                    and current_label != previous_two_label
                )

                if skip_allowed:

                    candidates.append(
                        (dp[t - 1, s - 2], s - 2)
                    )

            candidate_scores = [
                item[0]
                for item in candidates
            ]

            best_candidate_index = int(
                np.argmax(candidate_scores)
            )

            best_previous_score, best_previous_state = (
                candidates[best_candidate_index]
            )

            dp[t, s] = (
                best_previous_score + emission_score
            )

            backpointer[t, s] = best_previous_state

    # --------------------------------------------------------
    # TERMINAL STATE
    # --------------------------------------------------------

    # The final valid CTC path can terminate at:
    #   1. final blank state
    #   2. final phoneme state
    #
    # Choose whichever has the larger score.

    terminal_candidates = [
        num_states - 1,
        num_states - 2
    ]

    terminal_scores = [
        dp[-1, state]
        for state in terminal_candidates
    ]

    best_terminal_index = int(
        np.argmax(terminal_scores)
    )

    best_terminal_state = terminal_candidates[
        best_terminal_index
    ]

    best_score = dp[
        -1,
        best_terminal_state
    ]

    if not np.isfinite(best_score):
        raise RuntimeError(
            "No valid CTC alignment path was found."
        )

    # --------------------------------------------------------
    # BACKTRACKING
    # --------------------------------------------------------

    state_path = np.zeros(
        num_frames,
        dtype=np.int32
    )

    state_path[-1] = best_terminal_state

    current_state = best_terminal_state

    for t in range(num_frames - 1, 0, -1):

        previous_state = backpointer[
            t,
            current_state
        ]

        if previous_state < 0:
            raise RuntimeError(
                "Invalid backpointer during traceback."
            )

        state_path[t - 1] = previous_state

        current_state = previous_state

    return (
        state_path,
        expanded,
        float(best_score),
        dp
    )


# ============================================================
# FRAME-TO-PHONEME TIMING
# ============================================================

def extract_phone_timestamps(
    state_path,
    expanded,
    frame_interval,
):
    """
    Convert the Viterbi state path into phoneme timestamps.

    The expanded sequence has this structure:

        blank, phone1, blank, phone2, blank, ...

    Therefore phoneme states have odd indices.

    Timestamps use half-open frame intervals:

        [start_frame, end_frame)

    The end frame is exclusive.
    """

    results = []

    for state_index, phone in enumerate(expanded):

        if phone == BLANK:
            continue

        frame_indices = np.where(
            state_path == state_index
        )[0]

        if len(frame_indices) == 0:
            raise RuntimeError(
                f"Phoneme {phone} received no frames."
            )

        start_frame = int(frame_indices[0])

        end_frame = int(frame_indices[-1] + 1)

        start_time = start_frame * frame_interval

        end_time = end_frame * frame_interval

        duration = end_time - start_time

        results.append({
            "phone": phone,
            "state": state_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "frames": end_frame - start_frame,
        })

    return results


# ============================================================
# PHONEME-TO-WORD AGGREGATION
# ============================================================

def aggregate_word_timestamps(
    phone_results,
    words,
    word_ids,
):
    """
    Aggregate phoneme timestamps into word timestamps.

    word_ids[i] identifies which word contains phone i.

    This is a simplified mapping supplied by the example.
    A production text frontend must generate this mapping.
    """

    if len(phone_results) != len(word_ids):
        raise ValueError(
            "Phone result count and word_ids count differ."
        )

    word_results = []

    for word_index, word in enumerate(words):

        selected = [
            result
            for result, mapped_word_id in zip(
                phone_results,
                word_ids
            )
            if mapped_word_id == word_index
        ]

        if not selected:
            continue

        start = min(
            item["start"]
            for item in selected
        )

        end = max(
            item["end"]
            for item in selected
        )

        word_results.append({
            "word": word,
            "start": start,
            "end": end,
            "duration": end - start,
            "phones": [
                item["phone"]
                for item in selected
            ],
        })

    return word_results


# ============================================================
# DURATION MODELING DEMONSTRATION
# ============================================================

def demonstrate_duration_loss(phone_results, seed=7):
    """
    Demonstrate how aligned durations become training targets.

    A real duration predictor would be a neural network.

    Here we create deliberately noisy predictions to calculate
    an example log-duration loss.

    This is NOT training a neural duration predictor.
    """

    rng = np.random.default_rng(seed)

    target_durations = np.array(
        [
            item["frames"]
            for item in phone_results
        ],
        dtype=np.float64
    )

    predicted_durations = (
        target_durations
        * rng.uniform(
            0.75,
            1.30,
            size=len(target_durations)
        )
    )

    target_log = np.log(
        target_durations + EPSILON
    )

    predicted_log = np.log(
        predicted_durations + EPSILON
    )

    loss = np.mean(
        np.abs(
            target_log - predicted_log
        )
    )

    return (
        target_durations,
        predicted_durations,
        float(loss)
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_alignment_results(
    language_name,
    words,
    phones,
    phone_results,
    word_results,
    true_durations,
    target_durations,
    predicted_durations,
    duration_loss,
    best_score,
):
    print("\n" + "=" * 72)
    print("SPEECH ALIGNMENT RESULTS")
    print("=" * 72)

    print(f"Language: {language_name}")

    print(f"Transcript: {' '.join(words)}")

    print(f"Phone sequence: {' '.join(phones)}")

    print(f"Viterbi score: {best_score:.3f}")

    print("\nPHONEME ALIGNMENT")
    print("-" * 72)

    print(
        f"{'Phone':<12}"
        f"{'Start':>10}"
        f"{'End':>10}"
        f"{'Duration':>12}"
        f"{'Frames':>10}"
    )

    for item in phone_results:

        print(
            f"{item['phone']:<12}"
            f"{item['start']:>9.3f}s"
            f"{item['end']:>9.3f}s"
            f"{item['duration']:>10.3f}s"
            f"{item['frames']:>10}"
        )

    print("\nWORD ALIGNMENT")
    print("-" * 72)

    for item in word_results:

        print(
            f"{item['word']:<15}"
            f"{item['start']:.3f}s - "
            f"{item['end']:.3f}s"
            f"  duration={item['duration']:.3f}s"
        )

    print("\nDURATION PREDICTOR TRAINING EXAMPLE")
    print("-" * 72)

    print(
        f"{'Phone':<12}"
        f"{'Target':>12}"
        f"{'Predicted':>14}"
        f"{'Target ms':>14}"
    )

    for index, item in enumerate(phone_results):

        target = target_durations[index]

        predicted = predicted_durations[index]

        target_ms = target * FRAME_INTERVAL * 1000.0

        print(
            f"{item['phone']:<12}"
            f"{target:>12.1f}"
            f"{predicted:>14.2f}"
            f"{target_ms:>13.1f}"
        )

    print(f"\nExample log-duration L1 loss: {duration_loss:.5f}")

    print("\nSynthetic source durations:")
    print(true_durations)

    print("\nAligned target durations:")
    print(target_durations.astype(int))

    print("\nNOTE:")
    print(
        "The synthetic source durations and aligned durations "
        "may differ slightly because Viterbi chooses the "
        "highest-scoring path."
    )


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_alignment(
    log_probs,
    vocabulary,
    state_path,
    expanded,
    phone_results,
    word_results,
    language_name,
):
    """
    Plot:
        1. Synthetic acoustic log probabilities.
        2. Viterbi state path.
        3. Phoneme and word timing.
    """

    num_frames = log_probs.shape[0]

    frame_times = (
        np.arange(num_frames) * FRAME_INTERVAL
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 13),
        constrained_layout=True
    )

    fig.suptitle(
        f"CTC-Style Speech Alignment — {language_name}",
        fontsize=16
    )

    # --------------------------------------------------------
    # Plot 1: Acoustic log probabilities
    # --------------------------------------------------------

    axes[0].imshow(
        log_probs.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[
            0,
            num_frames * FRAME_INTERVAL,
            -0.5,
            len(vocabulary) - 0.5
        ]
    )

    axes[0].set_yticks(
        np.arange(len(vocabulary))
    )

    axes[0].set_yticklabels(vocabulary)

    axes[0].set_xlabel("Time (seconds)")

    axes[0].set_ylabel("Output label")

    axes[0].set_title(
        "Synthetic acoustic model log probabilities"
    )

    # --------------------------------------------------------
    # Plot 2: Viterbi state path
    # --------------------------------------------------------

    axes[1].step(
        frame_times,
        state_path,
        where="post"
    )

    axes[1].set_yticks(
        np.arange(len(expanded))
    )

    axes[1].set_yticklabels(expanded)

    axes[1].set_xlabel("Time (seconds)")

    axes[1].set_ylabel("Expanded CTC state")

    axes[1].set_title(
        "Best Viterbi alignment path"
    )

    axes[1].grid(True, alpha=0.3)

    # --------------------------------------------------------
    # Plot 3: Phoneme and word spans
    # --------------------------------------------------------

    for index, item in enumerate(phone_results):

        start = item["start"]

        duration = item["duration"]

        axes[2].barh(
            y=1,
            width=duration,
            left=start,
            height=0.35
        )

        center = start + duration / 2

        axes[2].text(
            center,
            1,
            item["phone"],
            ha="center",
            va="center",
            fontsize=9
        )

    for item in word_results:

        center = (
            item["start"] + item["end"]
        ) / 2

        axes[2].text(
            center,
            0,
            item["word"],
            ha="center",
            va="center",
            fontsize=10
        )

        axes[2].plot(
            [item["start"], item["end"]],
            [0.25, 0.25],
            linewidth=5
        )

    axes[2].set_yticks([0, 1])

    axes[2].set_yticklabels([
        "Words",
        "Phonemes"
    ])

    axes[2].set_xlabel("Time (seconds)")

    axes[2].set_title(
        "Phoneme and word timestamps"
    )

    axes[2].grid(
        axis="x",
        alpha=0.3
    )

    axes[2].set_xlim(
        0,
        num_frames * FRAME_INTERVAL
    )

    plt.show()


# ============================================================
# REPEATED-PHONEME EXPERIMENT
# ============================================================

def repeated_phone_experiment():
    """
    Demonstrate why CTC needs blank states between repeated labels.

    Target:
        l, l

    Expanded:
        blank, l, blank, l, blank

    A valid path must pass through the middle blank state
    to emit the second l.
    """

    phones = ["l", "l"]

    (
        log_probs,
        vocabulary,
        true_labels,
        true_durations
    ) = generate_synthetic_emissions(
        phones,
        seed=123
    )

    (
        path,
        expanded,
        score,
        dp
    ) = ctc_viterbi_align(
        log_probs,
        vocabulary,
        phones
    )

    print("\nREPEATED-PHONEME EXPERIMENT")
    print("-" * 60)

    print("Target phones:", phones)

    print("Expanded CTC states:", expanded)

    print("Best state path:")

    print([
        expanded[state]
        for state in path
    ])

    print(
        "\nNotice that the two adjacent 'l' labels are "
        "separated by a blank state."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Educational CTC-style phoneme alignment "
            "and duration modeling."
        )
    )

    parser.add_argument(
        "--language",
        choices=["en", "zh", "ja", "ru"],
        default="en"
    )

    parser.add_argument(
        "--no-plot",
        action="store_true"
    )

    parser.add_argument(
        "--repeated-phone-demo",
        action="store_true"
    )

    args = parser.parse_args()

    if args.repeated_phone_demo:

        repeated_phone_experiment()

        return

    example = LANGUAGE_EXAMPLES[
        args.language
    ]

    language_name = example["name"]

    words = example["words"]

    phones = example["phones"]

    word_ids = example["word_ids"]

    print("=" * 72)
    print("PHASE 1 — LESSON 13")
    print("SPEECH ALIGNMENT AND DURATION MODELING")
    print("=" * 72)

    print(f"\nLanguage: {language_name}")

    print("Transcript:", " ".join(words))

    print("Phonemes:", " ".join(phones))

    print(f"Frame interval: {FRAME_INTERVAL * 1000:.2f} ms")

    # --------------------------------------------------------
    # STEP 1: Generate synthetic acoustic evidence
    # --------------------------------------------------------

    (
        log_probs,
        vocabulary,
        true_frame_labels,
        true_phone_durations
    ) = generate_synthetic_emissions(
        phones
    )

    print(
        f"\nGenerated {log_probs.shape[0]} acoustic frames."
    )

    print(
        f"Acoustic vocabulary size: {len(vocabulary)}"
    )

    # --------------------------------------------------------
    # STEP 2: Expand CTC target sequence
    # --------------------------------------------------------

    expanded = expand_ctc_labels(phones)

    print("\nExpanded CTC sequence:")

    print(" ".join(expanded))

    # --------------------------------------------------------
    # STEP 3: Viterbi alignment
    # --------------------------------------------------------

    (
        state_path,
        expanded,
        best_score,
        dp
    ) = ctc_viterbi_align(
        log_probs,
        vocabulary,
        phones
    )

    # --------------------------------------------------------
    # STEP 4: Extract phoneme timestamps
    # --------------------------------------------------------

    phone_results = extract_phone_timestamps(
        state_path,
        expanded,
        FRAME_INTERVAL
    )

    # --------------------------------------------------------
    # STEP 5: Aggregate phonemes into words
    # --------------------------------------------------------

    word_results = aggregate_word_timestamps(
        phone_results,
        words,
        word_ids
    )

    # --------------------------------------------------------
    # STEP 6: Demonstrate duration modeling
    # --------------------------------------------------------

    (
        target_durations,
        predicted_durations,
        duration_loss
    ) = demonstrate_duration_loss(
        phone_results
    )

    # --------------------------------------------------------
    # STEP 7: Print results
    # --------------------------------------------------------

    print_alignment_results(
        language_name,
        words,
        phones,
        phone_results,
        word_results,
        true_phone_durations,
        target_durations,
        predicted_durations,
        duration_loss,
        best_score
    )

    # --------------------------------------------------------
    # STEP 8: Visualize
    # --------------------------------------------------------

    if not args.no_plot:

        visualize_alignment(
            log_probs,
            vocabulary,
            state_path,
            expanded,
            phone_results,
            word_results,
            language_name
        )

    print("\n" + "=" * 72)
    print("LESSON 13 COMPLETE")
    print("=" * 72)

    print(
        "\nNext conceptual step:"
        "\nUse real acoustic-model probabilities to align "
        "actual recorded speech."
    )


if __name__ == "__main__":
    main()