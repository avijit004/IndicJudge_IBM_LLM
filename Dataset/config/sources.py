"""
Declarative (language, task) -> ordered source list.

Coverage sets below were verified live against the Hugging Face Hub on
2026-08-25 (see scripts/probe_schema.py runs during development) -- not
guessed from dataset names. Two datasets named in the original brief turned
out to be GATED and inaccessible without an accepted-license HF token in
this environment: `openlanguagedata/flores_plus` and `ai4bharat/BPCC`.
`ai4bharat/MILU` is also gated. `ai4bharat/IndicXNLI` does not exist under
that exact repo id on the Hub; the equivalent, ungated `Divyanshu/indicxnli`
is used instead. All of this is recorded here and surfaced again in the
run report / README.
"""

LANGUAGES = [
    # (name, iso_code, resource_tier)
    ("Hindi", "hi", "high"),
    ("Bengali", "bn", "high"),
    ("Marathi", "mr", "high"),
    ("Telugu", "te", "high"),
    ("Tamil", "ta", "high"),
    ("Gujarati", "gu", "high"),
    ("Urdu", "ur", "high"),
    ("Kannada", "kn", "high"),
    ("Malayalam", "ml", "high"),
    ("Punjabi", "pa", "high"),
    ("Odia", "or", "mid"),
    ("Assamese", "as", "mid"),
    ("Nepali", "ne", "mid"),
    ("Sindhi", "sd", "mid"),
    ("Kashmiri", "ks", "mid"),
    ("Maithili", "mai", "low"),
    ("Bhojpuri", "bho", "low"),
    ("Konkani", "gom", "low"),
    ("Dogri", "doi", "low"),
    ("Sanskrit", "sa", "low"),
    ("Santali", "sat", "low"),
    ("Bodo", "brx", "low"),
]

ISO_TO_NAME = {iso: name for name, iso, _ in LANGUAGES}
ISO_TO_TIER = {iso: tier for name, iso, tier in LANGUAGES}

TASKS = ["qa", "reasoning", "summarization", "translation"]
TASK_LABELS = {
    "qa": "Question Answering",
    "reasoning": "Reasoning",
    "summarization": "Summarization",
    "translation": "Translation Quality",
}

# ---------------------------------------------------------------------------
# Coverage sets per source, verified live (see probe log summary above).
# ---------------------------------------------------------------------------

XORQA_IN_LANGS = {"hi", "bn", "mr", "te", "ta", "gu", "ur", "kn", "ml", "pa",
                   "or", "as", "mai", "bho", "gom", "sa", "sat", "brx"}
XQUAD_IN_LANGS = {"hi", "bn", "mr", "te", "ta", "gu", "ur", "kn", "ml", "pa", "or", "as"}
INDICQA_LANGS = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}
TYDIQA_LANGS = {"bn": "bengali", "te": "telugu"}  # secondary_task 'language' field values
XQUAD_GOOGLE_LANGS = {"hi"}

INDICCOPA_LANGS = {"as", "bn", "gom", "gu", "hi", "kn", "mai", "ml", "mr",
                    "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"}
# IndicParam: single split, filter by exam 'subject' name -> our iso code
INDICPARAM_SUBJECT_TO_ISO = {
    "Konkani": "gom", "Sanskrit": "sa", "Bodo": "brx", "Maithili": "mai",
    "Nepali": "ne", "Dogri": "doi", "Santali": "sat", "Oriya": "or",
}
INDICXNLI_LANGS = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}
GLOBAL_MMLU_LANGS = {"bn", "hi", "ne", "te"}

CROSSSUM_IN_LANGS = {"hi", "bn", "mr", "te", "ta", "gu", "ur", "kn", "ml", "pa",
                      "or", "as", "ne", "mai", "bho", "gom", "sa", "sat", "brx"}
INDIC_HEADLINE_LANGS = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}
XLSUM_LANG_NAMES = {  # iso -> xlsum config name
    "bn": "bengali", "gu": "gujarati", "hi": "hindi", "mr": "marathi",
    "ne": "nepali", "pa": "punjabi", "ta": "tamil", "te": "telugu", "ur": "urdu",
}
CROSSSUM_HF_LANG_NAMES = XLSUM_LANG_NAMES  # same provider/lang set, config "<name>-english"
INDICWIKIBIO_LANGS = {"as", "bn", "hi", "kn", "ml", "or", "pa", "ta", "te"}

FLORES_IN_LANGS = {"hi", "bn", "mr", "te", "ta", "gu", "ur", "kn", "ml", "pa",
                    "or", "as", "ne", "mai", "bho", "gom", "sa", "sat", "brx"}
SAMANANTAR_LANGS = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}
# rahular/itihasa is Sanskrit(sa)-English only, translation dict key 'sn'

GATED_NOTES = {
    "openlanguagedata/flores_plus": "gated on HF Hub (license click-through) -- would add ks, sd, doi translation coverage",
    "ai4bharat/BPCC": "gated on HF Hub (license click-through)",
    "ai4bharat/MILU": "gated on HF Hub (license click-through)",
}
