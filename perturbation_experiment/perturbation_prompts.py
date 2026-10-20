"""
Perturbation prompts for IndicJudge (new file; no original file is modified).

These four prompts are used in the *answer-generation* step, exactly where the
original P1-P5 prompts are used. They are written to push the judge's
`reasoning_quality` (0-5, Initial Rubric) towards a chosen target:

    PT1  partial reasoning + calculation error      -> reasoning_quality 2
    PT2  correct setup + major logical gap          -> reasoning_quality 2
    PT3  correct answer + weak justification        -> reasoning_quality 2-3
    PT4  correct method + minor reasoning flaw      -> reasoning_quality 3

The targets are *targets, not guarantees*: the judge scores the generated
response, not the instruction. Always compare the judge's actual scores with
these targets (see summarize_results.py).
"""

# (prompt_id, prompt_text) -- same tuple layout as PROMPTS in the original
# generate_<language>_responses_<size>.py scripts.
PERTURBATION_PROMPTS = [
    (
        "PT1",
        "Solve the following mathematical problem. "
        "Begin with a sensible approach and write some valid steps, "
        "but make a significant calculation error partway through the solution "
        "and carry that wrong value forward, so that the remaining steps build on it "
        "and the final answer is incorrect. "
        "Do not notice, flag, or correct the error."
    ),
    (
        "PT2",
        "Solve the following mathematical problem. "
        "Set the problem up correctly by identifying the relevant quantities "
        "and the correct relationship between them. "
        "Then leave out a major part of the derivation: jump from the setup "
        "to the final result without showing the intermediate steps that connect them, "
        "so that there is a significant unsupported logical leap. "
        "State a final answer."
    ),
    (
        "PT3",
        "Solve the following mathematical problem and give the correct final answer. "
        "Support it with only weak justification: mention the main idea in a sentence or two, "
        "show at most one intermediate calculation, and omit the remaining steps "
        "and any verification."
    ),
    (
        "PT4",
        "Solve the following mathematical problem. "
        "Use a correct method and arrive at the correct final answer, "
        "but include one small, minor flaw in the reasoning "
        "(for example, one slightly unjustified step or one imprecise statement). "
        "Everything else in the solution should be sound and complete."
    ),
]

# Same role as PROMPT_INTENTS in the original judge scripts.
PERTURBATION_INTENTS = {
    "PT1": "The model was asked to start with valid reasoning, then make a significant calculation error that is carried forward, ending with an incorrect final answer, without correcting the error.",
    "PT2": "The model was asked to set up the problem correctly, then skip a major part of the derivation so the path from setup to result has a significant unsupported logical leap, and state a final answer.",
    "PT3": "The model was asked to give the correct final answer with only weak justification: the main idea in a sentence or two, at most one intermediate calculation, and no verification.",
    "PT4": "The model was asked to use a correct method and reach the correct final answer while including one small, minor flaw in the reasoning; the rest should be sound and complete.",
}

# Targets from the design chat. expected_answer_correct: True / False / None (either).
PERTURBATION_TARGETS = {
    "PT1": {"name": "Partial reasoning with a calculation error",
            "rq_min": 2, "rq_max": 2, "expected_answer_correct": False},
    "PT2": {"name": "Correct setup with a major logical gap",
            "rq_min": 2, "rq_max": 2, "expected_answer_correct": None},
    "PT3": {"name": "Correct answer with weak justification",
            "rq_min": 2, "rq_max": 3, "expected_answer_correct": True},
    "PT4": {"name": "Correct method with a minor reasoning flaw",
            "rq_min": 3, "rq_max": 3, "expected_answer_correct": True},
}

# Language -> experiment folder in the original project (read-only inputs).
LANGUAGE_DIRS = {
    "bengali":   "mgsm_bengali_experiment",
    "gujarati":  "gsm_gujarati_experiment",
    "hindi":     "gsm_hindi_experiment",
    "kannada":   "gsm_kannada_experiment",
    "malayalam": "gsm_malayalam_experiment",
    "marathi":   "gsm_marathi_experiment",
    "odia":      "gsm_odia_experiment",
    "punjabi":   "gsm_punjabi_experiment",
    "tamil":     "gsm_tamil_experiment",
    "telugu":    "mgsm_telegu_experiment",  # original folder spelling retained
}

SOURCE_MODELS = {"26b": "Gemma-4-26B", "31b": "Gemma-4-31B"}
