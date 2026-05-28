---
name: asr-patch-casing
description: Capitalize sentence starts and proper nouns in an ASR transcript segment. Mirrors capitalization of proper nouns/entities/acronyms from ENGLISH_TRANSLATION onto the corresponding ORIGINAL tokens (for scripts with case distinction). Standardizes casing for no_touch_tokens. Uses seg_start to decide segment-initial capitalization. Strictly letter-case only — no punctuation changes. Use as the CASING step of the ASR patch chain.
---

# CASING step

## Inputs
- Current text
- `aggressiveness_level`
- `seg_start`
- `no_touch_tokens`
- `ENGLISH_TRANSLATION`

## Rules

- Capitalize sentence starts and proper nouns.
- Without a sentence boundary, do **NOT** introduce mid-sentence capitalization except for proper nouns/entities.
- Use `aggressiveness_level` strictly to scale casing intensity.
- If `ENGLISH_TRANSLATION` capitalizes a token (proper noun, entity, or acronym), you **MUST** capitalize the corresponding token in the output regardless of its form in `ORIGINAL`, including when embedded in non-Latin scripts. Applies only to tokens with case distinction.
- Standardize casing for tokens in `no_touch_tokens`.
- At segment start, apply `seg_start`: `high` → capitalize the first token as a sentence start; `low` → preserve original casing.
- **Strictly no punctuation changes** in this step. Modify letter case only.

## Language profile hooks

Applied only when `active_language_profile` matches. Stricter interpretation wins on any conflict.

- `arabic`, `cjk`: case-less scripts. This step **MUST pass through unchanged** — `edits: []`, `result` equal to input. The router should also deactivate this step under these profiles; if it is somehow active, still emit pass-through.
- `latin-other`: apply casing rules normally; do not import English-specific capitalization conventions that conflict with the locale's typography (e.g., German common nouns).

## Output

```json
{"edits": [["before", "after"]], "result": "string"}
```

## Standalone invocation (full top-level schema)

When this skill is invoked directly via `--use-skill asr-patch-casing --chain-steps CASING` (no router), you MUST still emit the full ASR-patch top-level JSON schema below. Returning only the step-local fragment fails the pipeline's schema check.

- Put this step's edits/result in `ct_casing`.
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
  "ct_lexical":    {"edits": [], "result": "string"},
  "ct_disfluency": {"edits": [], "result": "string"},
  "ct_format":     {"edits": [], "result": "string"},
  "ct_numeral":    {"edits": [], "result": "string"},
  "ct_punct":      {"edits": [], "result": "string"},
  "ct_casing":     {"edits": [["before","after"]], "result": "string"},
  "ct_remain_fix": {"edits": [], "result": "string"}
}
```
