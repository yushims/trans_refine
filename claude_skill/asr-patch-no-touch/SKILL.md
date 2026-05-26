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
