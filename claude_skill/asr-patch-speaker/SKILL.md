---
name: asr-patch-speaker
description: Normalize speaker labels in an ASR transcript segment. Converts existing labels like "Speaker 1:" or "Abc:" to canonical "[SPK1]:" / "[ABC]:" form, or adds [SPK1]:, [SPK2]: tags when speaker_scope is multi but labels are missing. Pass-through when speaker_scope is single and no labels exist. Use as the SPEAKER step of the ASR patch chain.
---

# SPEAKER step

## Inputs
- Current text (output of previous step, or ORIGINAL if first active step)
- `speaker_scope` from preprocess: `single` or `multi`

## Rules

- Normalize existing labels to `[LABEL]:` format. Examples: `Speaker 1:` → `[SPK1]:`; `Abc:` → `[ABC]:` when the mapping is unambiguous.
- If `speaker_scope` is `single` (or not `multi`) and no labels exist: pass-through unchanged.
- If `speaker_scope` is `multi` and no labels exist: add `[SPK1]:`, `[SPK2]:`, etc., in turn order.
- Preserve turn order and boundaries exactly. Do not merge or split turns.
- Do not rewrite lexical content, punctuation, numerals, casing, or spacing except as strictly required to attach/normalize speaker tags.

## Output

```json
{"edits": [["before", "after"]], "result": "string"}
```

## Standalone invocation (full top-level schema)

When this skill is invoked directly via `--use-skill asr-patch-speaker --chain-steps SPEAKER` (no router), you MUST still emit the full ASR-patch top-level JSON schema below. Returning only the step-local fragment fails the pipeline's schema check.

- Put this step's edits/result in `ct_speaker`.
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
  "ct_speaker":    {"edits": [["before","after"]], "result": "string"},
  "ct_combine":    {"edits": [], "result": "string"},
  "no_touch_tokens": [],
  "ct_lexical":    {"edits": [], "result": "string"},
  "ct_disfluency": {"edits": [], "result": "string"},
  "ct_format":     {"edits": [], "result": "string"},
  "ct_numeral":    {"edits": [], "result": "string"},
  "ct_punct":      {"edits": [], "result": "string"},
  "ct_casing":     {"edits": [], "result": "string"},
  "ct_remain_fix": {"edits": [], "result": "string"}
}
```
