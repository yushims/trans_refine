---
name: asr-patch-punct
description: Restore sentence-ending marks, commas after discourse markers, and clause-boundary punctuation in an ASR transcript segment. Mirrors the punctuation density of ENGLISH_TRANSLATION, scaled by aggressiveness_level. Handles run-ons by splitting into multiple sentences. Downgrades spurious full stops in over-segmented input. Uses seg_end to decide terminal punctuation. Use as the PUNCT step of the ASR patch chain.
---

# PUNCT step

## Inputs
- Current text
- `aggressiveness_level`
- `seg_start`, `seg_end`
- `ENGLISH_TRANSLATION`
- `{prev_context}`, `{next_context}` (boundary decisions only)

## Rules

- Use language-appropriate sentence-end marks; prefer neutral declarative punctuation.
- You **MUST** set off surviving discourse markers and interjections with commas to distinguish them from the main clause.
- Insert a mandatory comma after introductory markers that orient the sentence; treat these as structural breaks regardless of original ASR spacing.
- Insert a comma before coordinating conjunctions when they connect two independent clauses.
- Punctuation density **MUST** match the structural rhythm of `ENGLISH_TRANSLATION`. If the translation uses a comma to join ideas, you are **forbidden** from upgrading it to a period.
- Use `aggressiveness_level` to control density matching; when `high`, mirror the full cadence of `ENGLISH_TRANSLATION` even with higher edit count.
- Insert punctuation at the earliest clear boundary where a new Subject-Verb clause begins.
- Add missing sentence-ending marks before any sentence-start capitalization.
- **Boundary restoration**: when `ENGLISH_TRANSLATION` clearly indicates multiple sentence boundaries, mirror them with language-appropriate marks unless contradicted by strong `ORIGINAL` evidence.
- If a translation boundary resolves run-on ambiguity and is supported by `ORIGINAL`, prioritize that boundary over the fewest-edits tie-break.
- **Run-on control**: if the segment is long and contains 2+ clear clause boundaries, split into multiple sentences instead of one comma chain.
- For scripts without whitespace, resolve all inter-character spaces into punctuation (preferring commas for flow) or remove them.
- **Conditional downgrade**: if `ORIGINAL` is over-segmented (spurious full stops between tightly connected fragments of one thought), downgrade those specific sentence-end marks to commas.
- Do **NOT** downgrade true sentence boundaries; keep full stops for complete thoughts, topic shifts, or boundaries supported by `ENGLISH_TRANSLATION`.
- Do **NOT** insert punctuation adjacent to existing marks (no stacked / redundant punctuation).
- At segment end, apply `seg_end`: `high` → ensure sentence-ending mark; `low` → prefer no terminal punctuation change.

## Language profile hooks

Applied only when `active_language_profile` matches. Stricter interpretation wins on any conflict.

- `arabic`: use Arabic punctuation — `،` (comma), `؛` (semicolon), `؟` (question mark). Do **NOT** insert Latin `,` `;` `?` into Arabic-script text.
- `cjk`: use full-width punctuation — `。` (full stop), `，` (comma), `？` (question), `！` (exclamation). Do **NOT** insert ASCII sentence marks into CJK text.
- `latin-other`: Spanish only — add inverted opening `¿` / `¡` only when `ORIGINAL` clearly omits them and structure supports it; otherwise leave unchanged.

## Output

```json
{"edits": [["before", "after"]], "result": "string"}
```
