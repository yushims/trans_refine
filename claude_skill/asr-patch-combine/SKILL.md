---
name: asr-patch-combine
description: Merge separated fragments of a single entity in an ASR transcript (e.g., "10 0" -> "100", spaced-out IDs, products, names, currencies, dates, times, handles, multipliers) where the whitespace is purely a formatting artifact. Does NOT merge letter-by-letter spellings, does NOT convert numerals or expand years (defer to NUMERAL). Use as the COMBINE step of the ASR patch chain.
---

# COMBINE step

## Rules

- Merge consecutive parts that form a single entity when split is purely a formatting artifact. Examples: `10 0` → `100`.
- Scope includes identifier / product / name / entity / number / currency / date / time / handle / multiplier fragments.
- Only merge where whitespace creates a meaningless break within one entity.
- Do **NOT** merge letter-by-letter spellings; preserve the spaced-out form as-is.
- **Forbidden**: year expansion or numeral conversion (defer to the `NUMERAL` step).

## Language profile hooks

Applied only when `active_language_profile` matches. Stricter interpretation wins on any conflict.

- `cjk`: in no-whitespace scripts, only merge when the inter-character space is clearly an ASR artifact within a single entity. Do **NOT** merge across what would normally be separate tokens in the language.

## Output

```json
{"edits": [["before", "after"]], "result": "string"}
```

## Standalone invocation (full top-level schema)

When this skill is invoked directly via `--use-skill asr-patch-combine --chain-steps COMBINE` (no router), you MUST still emit the full ASR-patch top-level JSON schema below. Returning only the step-local fragment fails the pipeline's schema check.

- Put this step's edits/result in `ct_combine`.
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
  "ct_combine":    {"edits": [["before","after"]], "result": "string"},
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
