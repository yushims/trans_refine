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
