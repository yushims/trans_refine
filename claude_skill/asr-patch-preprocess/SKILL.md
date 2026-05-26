---
name: asr-patch-preprocess
description: PHASE 1 preprocessing for ASR transcript patching. Tokenizes ORIGINAL, produces a faithful English reference translation, estimates aggressiveness_level (low/medium/high), determines speaker_scope (single/multi), and predicts segment-start / segment-end sentence-boundary confidence. Always run once before any PHASE 2 patch step. Outputs tokenization, translation, aggressiveness_level, speaker_scope, seg_start, seg_end.
---

# Preprocess (PHASE 1)

Run exactly once before any patch step. All outputs are locked for the remainder of the chain.

## Inputs

- `ORIGINAL` (required)
- `needs_translation` (bool, from router) — whether any active downstream step needs the English reference
- `active_language_profile` (from router) — e.g., `generic`, `arabic`, `cjk`, `latin-other`
- `{locale}` (optional; resolved by router)
- `{prev_context}`, `{next_context}` (optional)

## Tasks

### tokenization
Tokenize `ORIGINAL` once. Tokens must be exact substrings of `ORIGINAL`. Priority:
1. Protected-span longest match (URLs, emails, technical brands, currency, dates).
2. Whitespace-separated words for Latin/Arabic/etc.
3. Minimal stable character units for non-whitespace scripts (Han/Kana/...).
4. Standalone punctuation.

### translation
**Conditional on `needs_translation`.**

- If `needs_translation == false` → set `translation` to `""` and skip translation work entirely. No active downstream step will reference it.
- If `needs_translation == true` → produce a faithful English reference translation of `ORIGINAL`:
  - If `ORIGINAL` is already English, produce a corrected English version (do NOT translate to another language).
  - Keep names, numbers, and protected spans unchanged.
  - Reference artifact only — does not trigger schema changes downstream.

English-dependent steps (router activates translation when any of these is active): `lexical`, `format`, `numeral`, `punct`, `casing`, `remain-fix`.

### aggressiveness_level
One of `low` / `medium` / `high`, estimated from transcript-quality signals (ASR disfluencies, punctuation sparsity, boundary ambiguity, token noise).
- `low` = conservative, `medium` = balanced, `high` = aggressive punctuation/fluency.
- Locked for the full chain. Forbidden constraints are never overridden by this level.

### speaker_scope
Exactly `single` or `multi`. Use `multi` only with explicit turn-taking evidence in `ORIGINAL`; otherwise `single`.

### seg_start, seg_end (segment boundary prediction)
Each is `high` / `medium` / `low`.
- If `ORIGINAL` has display-format cues in the middle, respect them and set the corresponding judgment to `high`.
- Otherwise use clause completeness, `ENGLISH_TRANSLATION` structure, and surrounding tokens (including `{prev_context}` / `{next_context}`) to judge.

## Output

```json
{
  "tokenization": {"tokens": ["..."]},
  "translation": "string",
  "aggressiveness_level": "low|medium|high",
  "speaker_scope": "single|multi",
  "seg_start": "high|medium|low",
  "seg_end": "high|medium|low"
}
```
