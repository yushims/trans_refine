---
name: asr-patch-remain-fix
description: Final residual cleanup of an ASR transcript segment. Repairs only misses from earlier ACTIVE chain steps already executed in this run — never backfills edits belonging to inactive steps. Preserves no_touch_tokens and legal/entity protections, preserves language/script variant, avoids adjacent redundant punctuation, and prefers no-edit when confidence is low. Aligns intensity with aggressiveness_level. Use as the final REMAIN_FIX step of the ASR patch chain.
---

# REMAIN_FIX step

## Inputs
- Current text (post-CASING)
- List of active earlier steps in this run
- `no_touch_tokens`
- `aggressiveness_level`
- `mode`: `standard` or `conservative`

## Rules

- Final residual cleanup for misses in earlier **ACTIVE** steps only.
- Prioritize conservative edits; if confidence is low, keep input unchanged.
- Apply a residual fix only when explicit in `ORIGINAL` or strongly supported by `ENGLISH_TRANSLATION` structure.
- **Scope gate**: may only fix issues that belong to the active chain steps already executed earlier in this run.
- **Forbidden** from introducing edits that belong exclusively to any inactive step.
- Preserve `no_touch_tokens` spans and legal/entity protections exactly.
- Keep meaning unchanged: no semantic rewrites, paraphrases, reordering, style rewriting, or unsupported additions.
- Preserve language/script variant; do **NOT** perform script conversion or script-variant substitution unless explicitly required by `ORIGINAL` evidence.
- Avoid adjacent redundant punctuation; align with `aggressiveness_level`.

## Conservative mode tie-breaks

When multiple valid edits exist, prefer in order:
1. minimal lexical change
2. highest spoken-form preservation
3. lowest risk of semantic drift

If confidence is not high → **do not edit**.

## Output

```json
{"edits": [["before", "after"]], "result": "string"}
```

## Standalone invocation (full top-level schema)

When this skill is invoked directly via `--use-skill asr-patch-remain-fix --chain-steps REMAIN_FIX` (no router), you MUST still emit the full ASR-patch top-level JSON schema below. Returning only the step-local fragment fails the pipeline's schema check.

- Put this step's edits/result in `ct_remain_fix`.
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
  "ct_casing":     {"edits": [], "result": "string"},
  "ct_remain_fix": {"edits": [["before","after"]], "result": "string"}
}
```
