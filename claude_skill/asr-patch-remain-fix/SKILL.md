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
