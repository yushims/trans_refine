---
name: asr-patch-lexical
description: Fix typos and phonetic ASR errors in a transcript segment while strictly preserving meaning, pronunciation similarity, token order, and original semantic role. Never reorders tokens to match English syntax. Never replaces a valid token with a different valid token unless ORIGINAL evidence forces the change. Supports standard and conservative modes (conservative forbids register upgrades, tense changes, relative-pronoun swaps, and dialect-to-MSA word substitutions). Use as the LEXICAL step of the ASR patch chain.
---

# LEXICAL step

## Inputs
- Current text (post-NO_TOUCH)
- `no_touch_tokens`
- `ENGLISH_TRANSLATION` (reference only)
- `mode`: `standard` or `conservative`
- `active_language_profile` (from router) — e.g., `arabic` activates the Arabic profile below

## Core rules (both modes)

- Fix typos and phonetic ASR errors **only**.
- Preserve pronunciation similarity. Use `ENGLISH_TRANSLATION` for disambiguation only.
- Do **NOT** replace a valid token with a different valid token unless `ORIGINAL` evidence clearly shows the original is incorrect.
- Do **NOT** normalize brand / proper-name forms from `ENGLISH_TRANSLATION` alone. Require explicit in-line `ORIGINAL` evidence.
- Do **NOT** edit any span in `no_touch_tokens` (no replace / insert / delete / reorder).
- **Strictly forbidden** to reorder tokens to match English syntax. Maintain the `ORIGINAL` token sequence even when `ENGLISH_TRANSLATION` suggests otherwise.

## Arabic profile (both modes)

Activated when `active_language_profile == arabic`.

- Limit lexical edits to spelling/orthography standardization of the **same** token toward formal MSA writing, preserving meaning and style.
- Remove diacritics. Do **NOT** replace base letters except for clear spelling errors with `ORIGINAL` evidence.

## Conservative mode additions

Every edit MUST preserve the original semantic role. Additionally:
- Do **NOT** replace a valid token with another valid token that changes nuance, register, or meaning.
- For ambiguous tokens, prefer **no change**.
- **Strictly forbidden**:
  - changing verb tense / aspect markers
  - replacing relative pronouns
  - upgrading informal tokens into formal equivalents
- Arabic: treat dialectal tokens as valid lexical items. Do **NOT** normalize dialect → MSA unless it is a clear spelling error of the **same** word.

## Output

```json
{"edits": [["before", "after"]], "result": "string"}
```

## Standalone invocation (full top-level schema)

When this skill is invoked directly via `--use-skill asr-patch-lexical --chain-steps LEXICAL` (no router), you MUST still emit the full ASR-patch top-level JSON schema below. Returning only the step-local fragment fails the pipeline's schema check.

- Put this step's edits/result in `ct_lexical`.
- Every other `ct_*` MUST be `{"edits": [], "result": "<unchanged source text>"}` (no prior step ran).
- `tokenization` MUST be `{"tokens": [...]}` — best-effort tokenization of the source text.
- `translation` MUST be a string; use `""` if you cannot produce one.
- `aggressiveness_level`, `speaker_scope`, `seg_start`, `seg_end` MUST be present as strings using the allowed values.
- `no_touch_tokens` MUST be `[]` unless you also identify protected spans.
- No markdown fencing, no extra keys, no commentary outside the JSON object.

```json
{
  "tokenization": {"tokens": []},
  "translation": "string",
  "aggressiveness_level": "low/medium/high",
  "speaker_scope": "single/multi",
  "seg_start": "high/medium/low",
  "seg_end": "high/medium/low",
  "ct_speaker":    {"edits": [], "result": "string"},
  "ct_combine":    {"edits": [], "result": "string"},
  "no_touch_tokens": [],
  "ct_lexical":    {"edits": [["before","after"]], "result": "string"},
  "ct_disfluency": {"edits": [], "result": "string"},
  "ct_format":     {"edits": [], "result": "string"},
  "ct_numeral":    {"edits": [], "result": "string"},
  "ct_punct":      {"edits": [], "result": "string"},
  "ct_casing":     {"edits": [], "result": "string"},
  "ct_remain_fix": {"edits": [], "result": "string"}
}
```
