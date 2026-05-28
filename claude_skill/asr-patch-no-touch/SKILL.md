---
name: asr-patch-no-touch
description: Extract immutable spans (proper names, legal entities, technical brands) from the COMBINE-step output of an ASR transcript. These spans become protected for the remainder of the patch chain — no replace, insert, delete, reorder, or script-swap is allowed within them. Use as the NO_TOUCH step of the ASR patch chain.
---

# NO_TOUCH step

## Rules

- Extract immutable spans from the **COMBINE step result** for proper names and legal entities.
- Extract spans as **exact substrings of the COMBINE output**. Do **NOT** extract from `ENGLISH_TRANSLATION`.
- Total ban on lexical editing, script-swapping, or reordering within these spans for every later step in the chain.
- Text content itself is unchanged by this step.

## Output

```json
{
  "no_touch_tokens": ["span1", "span2"],
  "edits": [],
  "result": "<unchanged input text>"
}
```

If `NO_TOUCH` is inactive, `no_touch_tokens` MUST be `[]`.

## Standalone invocation (full top-level schema)

When this skill is invoked directly via `--use-skill asr-patch-no-touch --chain-steps NO_TOUCH` (no router), you MUST still emit the full ASR-patch top-level JSON schema below. Returning only the step-local fragment fails the pipeline's schema check.

- Put identified protected spans in the top-level `no_touch_tokens` array.
- Every `ct_*` MUST be `{"edits": [], "result": "<unchanged source text>"}` (this step makes no textual edits).
- `tokenization` MUST be `{"tokens": [...]}` — best-effort tokenization of the source text.
- `translation` MUST be a string; use `""` if you cannot produce one.
- `aggressiveness_level`, `speaker_scope`, `seg_start`, `seg_end` MUST be present as strings using the allowed values.
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
  "no_touch_tokens": ["span1", "span2"],
  "ct_lexical":    {"edits": [], "result": "string"},
  "ct_disfluency": {"edits": [], "result": "string"},
  "ct_format":     {"edits": [], "result": "string"},
  "ct_numeral":    {"edits": [], "result": "string"},
  "ct_punct":      {"edits": [], "result": "string"},
  "ct_casing":     {"edits": [], "result": "string"},
  "ct_remain_fix": {"edits": [], "result": "string"}
}
```
