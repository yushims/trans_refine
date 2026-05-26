---
name: asr-patch-router
description: Orchestrator for ASR/STT transcript patching. Inspects an input transcript segment, decides which per-step patch skills are needed (preprocess, speaker, combine, no-touch, lexical, disfluency, format, numeral, punct, casing, remain-fix), and runs them in the canonical order. Use whenever the user asks to clean up, patch, normalize, or post-process an ASR transcript, or supplies fields like ORIGINAL, ENGLISH_TRANSLATION, {chain_steps}, {input_transcript}, {locale}, {prev_context}, {next_context}. Selects standard vs conservative mode and assembles the final JSON output that matches the fixed schema (tokenization, translation, aggressiveness_level, speaker_scope, seg_start, seg_end, ct_speaker, ct_combine, no_touch_tokens, ct_lexical, ct_disfluency, ct_format, ct_numeral, ct_punct, ct_casing, ct_remain_fix).
---

# ASR Patch Router

You are orchestrating a deterministic, schema-bound transcript patch. Each step in the chain is a separate skill. Your job is to (1) detect which step-skills are needed for this input and (2) invoke them in order, then assemble one JSON object.

## Canonical chain order (NEVER reorder)

1. `asr-patch-preprocess` — always runs first (produces tokenization, translation, aggressiveness_level, speaker_scope, seg_start, seg_end).
2. `asr-patch-speaker`
3. `asr-patch-combine`
4. `asr-patch-no-touch`
5. `asr-patch-lexical`
6. `asr-patch-disfluency`
7. `asr-patch-format`
8. `asr-patch-numeral`
9. `asr-patch-punct`
10. `asr-patch-casing`
11. `asr-patch-remain-fix` — always runs last when any earlier step is active.

## Step 1 — Language & locale detection

Resolve `locale` and `active_language_profile` before anything else.

1. If `{locale}` is supplied, accept it verbatim (e.g., `ar-SA`, `en-US`, `zh-CN`, `ja-JP`).
2. Otherwise infer locale from `ORIGINAL`:
   - Dominant script (Arabic, Han, Kana, Hangul, Cyrillic, Devanagari, Latin, ...).
   - For Latin script, use lexical/grammatical evidence to refine (en, es, fr, de, pt, ...). Default to `und` (undetermined) if no clear signal.
   - Record this inference internally; the final JSON schema is unchanged.
3. Select `active_language_profile` from the registry below using the locale's primary subtag and script. If no profile matches, use `generic`.

### Language profile registry

Each profile is a small set of locale-conditional constraints. **Profiles never permit semantic rewrites, paraphrases, reordering, or hallucinated additions.** When a profile rule conflicts with a generic step rule, **the stricter interpretation wins**. Apply a profile rule only when `ORIGINAL` language/script evidence is explicit and unambiguous.

| Profile | Trigger | Hooks (which step skills must honor it) |
|---------|---------|-----------------------------------------|
| `generic` | default / unknown locale | none beyond step rules |
| `arabic` | locale starts with `ar`, OR dominant script is Arabic | `lexical` (orthographic normalization toward MSA at character level; remove diacritics; conservative mode forbids dialect→MSA word swaps), `format` (Arabic quotation marks `«»` and `“”`; do not Latinize), `numeral` (preserve Arabic-Indic `٠-٩` vs Latin `0-9` per `ORIGINAL`; do not transliterate digits), `punct` (use `،` comma, `؛` semicolon, `؟` question mark; not Latin equivalents), `casing` (case-less script → no casing edits), `disfluency` (whitespace-sensitive; treat tag removal carefully) |
| `cjk` | locale starts with `zh`, `ja`, `ko`, OR dominant script is Han/Kana/Hangul | `format` (CJK quotation marks `「」` `『』` `《》`; halfwidth/fullwidth preservation), `numeral` (preserve original digit form), `punct` (use `。`, `，`, `？`, `！` where appropriate; no Latin sentence marks), `casing` (case-less script → no casing edits), `combine` (no-whitespace handling — only resolve obvious inter-character spaces) |
| `latin-other` | locale uses Latin script but not English | `format` (locale quotation marks `«»`, `„“`, etc.), `punct` (Spanish inverted `¿` `¡` when warranted by `ORIGINAL` evidence), `numeral` (locale decimal/thousands separators per `ORIGINAL`; do not normalize) |

To add a new profile later, append a row here and add a matching "Language profile hooks" subsection in each affected step skill.

### Cross-cutting overrides policy

- Apply an override **only** when `ORIGINAL` language/script evidence is explicit and unambiguous.
- Overrides narrow behavior; they never permit semantic rewrites, paraphrases, reordering, or hallucinated additions.
- On conflict with a generic step rule, the **stricter** interpretation wins.
- Pass `active_language_profile` to every step skill alongside `mode`.
- For case-less scripts (Arabic, Han, Kana, Hangul, etc.) the `casing` step must pass through unchanged.

## Step 2 — Mode selection

Pick exactly one mode:

- `conservative` — when the user mentions "spoken", "dialect", "preserve register", "low confidence", "do not rewrite", or explicitly asks for it. Stricter on lexical/dialect.
- `standard` — default. Balanced patching.

Pass the chosen mode to every step skill you invoke (some skills branch on it).

## Step 3 — Active-step detection

If the user provides `{chain_steps}` / `{active_chain_ids}` / `{active_step_keys}`, use those literally. Otherwise auto-detect using these signals on `ORIGINAL`:

| Step | Activate when ORIGINAL shows |
|------|------------------------------|
| `speaker` | Existing labels like `Speaker 1:`, `Abc:`, OR multi-speaker turn-taking evidence. |
| `combine` | Spaced-out fragments of one entity (e.g. `10 0`, `micro soft`, `2 0 2 4`). |
| `no-touch` | Proper names, legal entities, technical brands present. |
| `lexical` | Probable typos or phonetic ASR errors. |
| `disfluency` | Fillers (`uh`, `um`, etc.), false starts, noise tags (`[NOISE]`, `_hmm_`, `[SPN/]`, ...), adjacent short repeats. |
| `format` | Spacing artifacts in symbols/abbreviations, unnecessary quote marks. |
| `numeral` | Spoken-out years/dates/percentages, mixed digit+word multipliers. |
| `punct` | Missing sentence-end marks, run-ons, missing commas after discourse markers, over-segmentation. |
| `casing` | Sentence-starts lowercased, proper nouns lowercased. |
| `remain-fix` | Any active step above. |

When unsure, **activate** the step — its rules already require no-op on ambiguous cases.

Additional locale-driven gating: if `active_language_profile` indicates a case-less script (Arabic, Han, Kana, Hangul, ...), **deactivate** `casing` regardless of detection.

## Step 4 — English-translation gating

The English reference translation is only produced when at least one **active** step depends on it. English-dependent steps are:

- `lexical` (disambiguation)
- `format` (English-aligned compact forms)
- `numeral` (preferred numeral styles)
- `punct` (punctuation density matching)
- `casing` (proper-noun / entity / acronym capitalization)
- `remain-fix` (residual cleanup supported by translation structure)

Compute `needs_translation = any(active step ∈ {lexical, format, numeral, punct, casing, remain-fix})`.

Pass `needs_translation` to `asr-patch-preprocess`:
- If `true` → preprocess produces a faithful English reference translation.
- If `false` → preprocess sets `translation` to `""` (empty string) and skips translation work. Downstream steps must not reference `ENGLISH_TRANSLATION` in this case.

The `translation` key remains present in the final JSON regardless.

## Step 5 — Execute

1. Invoke `asr-patch-preprocess` first with `needs_translation` and `active_language_profile`. Carry forward `tokenization`, `translation`, `aggressiveness_level`, `speaker_scope`, `seg_start`, `seg_end`.
2. For each step in canonical order:
   - If active: invoke its skill with `mode`, `active_language_profile`, the carried preprocessing fields, the current intermediate text (output of the previous active step, or `ORIGINAL` for the first active step), `locale`, `{prev_context}`, `{next_context}`, and (from `no-touch` onward) the `no_touch_tokens` list.
   - If inactive: emit pass-through `{"edits": [], "result": <unchanged input-to-that-step>}`. If `NO_TOUCH` is inactive, `no_touch_tokens` must be `[]`.
3. After `remain-fix`, assemble the final JSON.

## Step 6 — Output contract

Return **exactly one** JSON object, no markdown/prose, with these exact keys:

```
tokenization, translation, aggressiveness_level, speaker_scope, seg_start, seg_end,
ct_speaker, ct_combine, no_touch_tokens, ct_lexical, ct_disfluency, ct_format,
ct_numeral, ct_punct, ct_casing, ct_remain_fix
```

Each `ct_*` value is `{"edits": [[before, after], ...], "result": "string"}`.

## Non-negotiables (enforced across all step skills)

- Lexical content must come from `ORIGINAL`. `ENGLISH_TRANSLATION` is reference-only.
- Never reorder tokens to match English syntax.
- Never delete long repeated spans.
- Never edit `no_touch_tokens` spans after the `NO_TOUCH` step.
- No keys outside the schema. No markdown wrapping.
- On ambiguity, prefer **no change**.

## Context handling

`{prev_context}` and `{next_context}` are used only for boundary decisions (casing, punctuation). Never include or edit them in the output.

---

## Runtime invocation (auto-filled by the harness)

The fields below are substituted at request time by `build_patch_prompt`. When this skill is invoked directly via `--use-skill asr-patch-router`, treat the values literally.

### [RUNTIME_CHAIN_POLICY]

- Active chain IDs: {active_chain_ids}
- Active payload step keys: {active_step_keys}
- Inactive payload step keys: {inactive_step_keys}
- For every inactive payload step key, edits MUST be [] and result MUST be pass-through unchanged.
- If any inactive payload step has non-empty edits, the output is invalid and will be retried.
- If NO_TOUCH is inactive, no_touch_tokens MUST be [].
- Active chain steps (id:name): {chain_steps}

### [LOCALE]

- The input audio locale is: {locale}
- If known, use the locale information to guide language-specific editing decisions and to select `active_language_profile` per Step 1.

### [CONTEXT]

- Previous segment ending: `{prev_context}`
- Next segment starting: `{next_context}`
- If context is empty, no neighboring segments are available; rely on `[SEGMENT_BOUNDARY_PREDICTION]` for boundary decisions.
- Use context only for boundary decisions (casing, punctuation). Do NOT edit or reference context content in the output.

### Input Transcript

```text
{input_transcript}
```

