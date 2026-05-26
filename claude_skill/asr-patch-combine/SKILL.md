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
