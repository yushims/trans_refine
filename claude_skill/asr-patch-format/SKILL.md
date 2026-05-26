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
