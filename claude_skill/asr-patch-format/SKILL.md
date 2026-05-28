---
name: asr-patch-format
description: Fix spacing artifacts in symbols, abbreviations, and compact forms in an ASR transcript segment. Removes unnecessary quotation marks (including locale-specific) without changing meaning, grammar, wording, language, script, or order. Preserves compact forms for fixed compounds and technical brands. Treats legal names as protected (format-level only). Use as the FORMAT step of the ASR patch chain.
---

# FORMAT step

## Rules

- Fix spacing artifacts in symbols/abbreviations.
- Remove unnecessary quotation marks (including locale-specific) without changing meaning, grammar, wording, language, script, or text order.
- For English abbreviations/compact forms already present in `ORIGINAL`, follow `ENGLISH_TRANSLATION`-aligned formatting **only when unambiguous**. Do **NOT** translate non-English forms.
- Maintain compact forms for fixed compounds/markers when the spacing change is purely formatting.
- Keep technical brands in standard compact form; do not split lexicalized compounds.
- Treat legal names as protected spans: allow only format-level edits (spacing / punctuation / casing). If the lexical form is uncertain, keep unchanged.

## Language profile hooks

Applied only when `active_language_profile` matches. Stricter interpretation wins on any conflict.

- `arabic`: prefer Arabic-style quotation marks (`«»`, `“”`); do **NOT** Latinize. Do not introduce spaces around Arabic punctuation.
- `cjk`: prefer CJK quotation marks (`「」`, `『』`, `《》`) when removing unnecessary marks; preserve halfwidth vs. fullwidth forms exactly as in input — do **NOT** normalize.
- `latin-other`: prefer locale-appropriate quotation marks (`«»`, `„“`, etc.) over English curly quotes when removing unnecessary marks.

## Output

```json
{"edits": [["before", "after"]], "result": "string"}
```

## Standalone invocation (full top-level schema)

When this skill is invoked directly via `--use-skill asr-patch-format --chain-steps FORMAT` (no router), you MUST still emit the full ASR-patch top-level JSON schema below. Returning only the step-local fragment fails the pipeline's schema check.

- Put this step's edits/result in `ct_format`.
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
  "ct_format":     {"edits": [["before","after"]], "result": "string"},
  "ct_numeral":    {"edits": [], "result": "string"},
  "ct_punct":      {"edits": [], "result": "string"},
  "ct_casing":     {"edits": [], "result": "string"},
  "ct_remain_fix": {"edits": [], "result": "string"}
}
```
