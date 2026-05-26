---
name: asr-patch-numeral
description: Format-only normalization of years, dates, and percentages to canonical digits in an ASR transcript segment. Keeps 2-digit years unless a century anchor disambiguates. Normalizes unambiguous percentages to number+%. Preserves readout: never alters pronunciation or relative position of numbers. Aligns numeral style with ENGLISH_TRANSLATION only when multiple valid forms exist and meaning is unchanged. Use as the NUMERAL step of the ASR patch chain.
---

# NUMERAL step

## Rules

- Normalize years, dates, and percentages to canonical digits — **format-only**.
- Keep 2-digit years unless a nearby century anchor disambiguates.
- Normalize unambiguous percentages to number+%. Avoid mixed word-marker + digit percent forms.
- Prefer numeral styles aligned with `ENGLISH_TRANSLATION` when multiple valid forms exist and meaning is unchanged.
- Use customary written styles for the target language/register; keep consistency within the segment.
- Convert mixed digit+multiplier forms to canonical digits while preserving original unit/script conventions.
- **Preserve readout**: if a change would alter pronunciation or intent is uncertain, keep the original wording.
- Convert explicit full-year expressions to digits only when unambiguous.
- You **MUST** preserve each number's original relative position. Do **NOT** move a number to a different clause or sentence even if it seems more "logical".

## Language profile hooks

Applied only when `active_language_profile` matches. Stricter interpretation wins on any conflict.

- `arabic`: preserve the digit family used in `ORIGINAL` (Arabic-Indic `٠-٩` vs Latin `0-9`); do **NOT** transliterate between them. Percent sign placement follows the form already in `ORIGINAL`.
- `cjk`: preserve original digit form (Han numerals `一二三...` vs Arabic numerals) — normalize only when split-fragment combination is required and meaning is unchanged.
- `latin-other`: preserve locale decimal and thousands separators present in `ORIGINAL` (e.g., `,` decimal vs `.` decimal); do **NOT** swap separator conventions.

## Output

```json
{"edits": [["before", "after"]], "result": "string"}
```
