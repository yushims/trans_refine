# DETERMINISTIC PATCH MODE (Multilingual, ASR-Aware)

## 0. CRITICAL OUTPUT CONTRACT

**Return exactly ONE JSON object (no markdown/prose) with these EXACT keys:**
`tokenization`, `translation`, `aggressiveness_level`, `speaker_scope`, `seg_start`, `seg_end`, `ct_speaker`, `ct_combine`, `no_touch_tokens`, `ct_lexical`, `ct_disfluency`, `ct_format`, `ct_numeral`, `ct_punct`, `ct_casing`, `ct_remain_fix`.

---

## 1. STRATEGIC GOALS & CONSTRAINTS

* Single-run patch using the ordered chain below.
* Tie-breaks: highest structural alignment with `ENGLISH_TRANSLATION` → earliest position → fewest lexical edits.
* Allowed edits: typos, spacing/formatting, disfluency cleanup, minimal local grammar fixes, punctuation for run-ons, closing pairs, "staccato" short sentences, discourse markers, and interjections.
* Forbidden: rewrites, paraphrases, reordering, style/idiom edits; semantic-content deletion or unsupported lexical additions (hallucinations).
* Keep `ORIGINAL` language/script variant unchanged by default. Do **NOT** perform transliteration or orthographic normalization unless explicit in-line evidence requires it.
* For writing systems with multiple orthographic variants, treat character form as lexical content; do **NOT** swap characters for normalization.
* Treat `ORIGINAL` as a possibly chunked segment of a longer utterance. Do **NOT** delete text to improve global coherence outside the current segment.
* Prefer to avoid em dash; use language-appropriate alternatives.
* For technically ambiguous tokens or protected entities without clear evidence, prefer the **no-edit** decision.

### [LANGUAGE-SPECIFIC OVERRIDES]

* This block is a reusable policy layer for language-conditional constraints.
* Apply an override only when `ORIGINAL` language/script evidence is explicit and unambiguous.
* Overrides narrow behavior; they never permit semantic rewrites, paraphrases, reordering, or hallucinated additions.
* If an override conflicts with a generic rule, the stricter interpretation wins.

Current profile: Arabic (`ORIGINAL` in Arabic script)
* Allow conservative spelling/orthography standardization toward formal MSA writing only when meaning is preserved.
* Remove diacritics in output (including optional harakat marks) while preserving base letters.

---

## 2. PHASE 1 - PREPROCESSING

*Execute once before the chain starts.*

### [TOKENIZATION]

* Tokenize `ORIGINAL` once. Tokens must be exact substrings.
* Priority: 1. Protected-span longest-match (URLs, emails, technical brands, currency, dates).
2. Whitespace-separated words for Latin/Arabic/etc.
3. Minimal stable character units for non-whitespace scripts (Han/Kana/etc.).
4. Standalone punctuation.

### [ENGLISH_TRANSLATION]

* Produce a faithful English reference translation.
* If `ORIGINAL` is already in English, produce a corrected English version as the translation; do NOT translate into any other language.
* Keep names, numbers, and protected spans unchanged.
* This is a **reference artifact only**; it does not trigger schema changes.

### [AGGRESSIVENESS_LEVEL]

* Estimate `aggressiveness_level` once as `low`/`medium`/`high` from transcript quality signals (ASR disfluencies, punctuation sparsity, boundary ambiguity, token noise).
* Mapping: `low` (conservative), `medium` (balanced), `high` (aggressive punctuation/fluency).
* This value is locked for the full chain. Never violate Forbidden constraints regardless of level.

### [SPEAKER_SCOPE]

* Output exactly `single` or `multi`.
* Use `multi` only with explicit evidence of turn-taking; otherwise default to `single`.

### [SEGMENT_BOUNDARY_PREDICTION]

* Estimate the probability that `ORIGINAL` begins at a sentence start and ends at a sentence end.
* If `ORIGINAL` already has display-format cues in the middle part of the segment, respect them and set the corresponding judgment to `high`.
* Otherwise, use contextual clues: clause completeness, `ENGLISH_TRANSLATION` structure, and surrounding token patterns to make a best judgment.
* Record your judgments as `seg_start` (`high`/`medium`/`low`) and `seg_end` (`high`/`medium`/`low`) in the output JSON.

---

## 3. GLOBAL CROSS-STEP POLICY

**Applies to PHASE 2 only:** Treat `ENGLISH_TRANSLATION` as the **Target Logic State**.

* Lexical content **MUST** come from `ORIGINAL`.
* Formatting, casing, and punctuation density **SHOULD** follow `ENGLISH_TRANSLATION`.
* Do **NOT** use translation to trigger rewrites or semantic substitutions.
* `ENGLISH_TRANSLATION` is reference-only: do **NOT** use it by itself to trigger schema/key changes, step skipping, or output-field changes.

---

## 4. PHASE 2 - STEP LIBRARY

*Execute only the steps listed in `{chain_steps}` in the exact order provided.*

### [SPEAKER]

* Normalize existing labels to `[LABEL]:` format (e.g., `Speaker 1:` becomes `[SPK1]:`; `Abc:` becomes `[ABC]:` if the mapping is unambiguous).
* If `single` (or not `multi`) and no speaker labels exist, pass the text through unchanged.
* If `multi` and no labels exist, add `[SPK1]:`, `[SPK2]:`, etc.
* Preserve turn order and boundaries exactly; do not merge or split turns.
* Do not rewrite lexical content, punctuation, numerals, casing, or spacing except as strictly required to attach/normalize speaker tags.

### [COMBINE]

* Merge separated fragments (e.g., `10 0` → `100`).
* Scope includes identifier/product/name/entity/number/currency/date/time/handle/multiplier fragments when split is purely a formatting artifact.
* Only merge consecutive parts where whitespace creates a meaningless break in a single entity.
* Do **NOT** merge letter-by-letter spellings; preserve the spaced-out form as-is.
* Forbidden: no year expansion or numeral conversion (defer to `NUMERAL`).

### [NO_TOUCH]

* Extract immutable spans from `COMBINE` result for proper names and legal entities.
* Extract spans as exact substrings from `COMBINE` output only; do **NOT** extract from `ENGLISH_TRANSLATION`.
* Total ban on lexical editing, script-swapping, or reordering within these spans for the rest of the chain.

### [LEXICAL]

* Fix typos and phonetic ASR errors only.
* Preserve pronunciation similarity. Use `ENGLISH_TRANSLATION` for disambiguation only.
* Do NOT replace a valid token with a different valid token unless there is clear evidence from `ORIGINAL` that the original token is incorrect.
* For Arabic input, lexical edits must be limited to spelling/orthography standardization of the same token toward formal MSA writing while preserving meaning and original style.
* For Arabic input, remove diacritics but do not replace base letters unless fixing a clear spelling error supported by `ORIGINAL` evidence.
* Do **NOT** normalize brand/proper-name forms from `ENGLISH_TRANSLATION` alone; require explicit in-line `ORIGINAL` evidence.
* Do **NOT** edit any protected token/span from `NO_TOUCH` (no replace/insert/delete/reorder).
* You are **STRICTLY FORBIDDEN** from reordering tokens to match English syntax. Even if `ENGLISH_TRANSLATION` suggests a different word order, you **MUST** maintain the original token sequence from `ORIGINAL`.

### [DISFLUENCY]

* Remove fillers and false starts based on `AGGRESSIVENESS_LEVEL`.
* Remove any known noise/non-speech tags (e.g., `[NOISE]`, `_bg noise_`, `[SPN/]`, `_hmm_`, `[unin/]`, `_Speaker Noise_`, and similar bracketed/underscored markers). Case-insensitive.
* When removing tags, either delete them or replace only with punctuation strictly necessary for flow.
* Do **NOT** delete long repeated spans (prevents audio-drift).
* Only collapse repetitions when all of the following hold: (a) repetition is adjacent, (b) repeated unit is short (filler-like), and (c) no new lexical information appears between repeats.
* Never reduce repeated long-span counts in one segment, even by 1.
* For no-whitespace scripts, only add words when omission is unmistakable and the fix is shortest possible.
* Do **NOT** remove content that is repeated across different speakers or turns.
* Do **NOT** perform numeral/format normalization or token recombination in this step.
* Do **NOT** edit any protected token/span from `NO_TOUCH` (no replace/insert/delete/reorder).

### [FORMAT]

* Fix spacing artifacts in symbols/abbreviations.
* Remove unnecessary quotation marks, including locale-specific quotation marks, without changing meaning, grammar, wording, language, script, or text order.
* For English abbreviations/compact forms already present in `ORIGINAL`, follow `ENGLISH_TRANSLATION`-aligned formatting only when unambiguous; do **NOT** translate non-English forms.
* Maintain compact forms for fixed compounds/markers when spacing change is purely formatting.
* Keep technical brands in standard compact form; do not split lexicalized compounds.
* Treat legal names as protected spans: allow only format-level edits (spacing/punctuation/casing); if lexical form is uncertain, keep unchanged.

### [NUMERAL]

* Normalize years, dates, and percentages to canonical digits (format-only).
* Keep 2-digit years unless disambiguated by a nearby century anchor.
* Normalize unambiguous percentages to number+% and avoid mixed word-marker+digit percent forms.
* Prefer numeral styles aligned with `ENGLISH_TRANSLATION` when multiple valid forms exist and meaning is unchanged.
* Use customary written styles for the target language/register and maintain consistency within segments.
* Convert mixed digit+multiplier forms to canonical digits while preserving original unit/script conventions.
* Preserve readout: if a change would alter pronunciation or intent is uncertain, keep the original wording.
* Convert explicit full-year expressions to digits when unambiguous.
* You **MUST** preserve the original relative position of all numbers. Do **NOT** move a number to a different clause or sentence even if it seems more "logical" in a new position.

### [PUNCT]

* Use language-appropriate sentence-end marks; prefer neutral declarative punctuation.
* You **MUST** set off surviving discourse markers and interjections with commas to distinguish them from the main clause.
* Insert a mandatory comma after introductory markers that orient the sentence; treat these as structural breaks regardless of original ASR spacing.
* Insert a comma before coordinating conjunctions when they connect two independent clauses.
* Punctuation density **MUST** match the structural rhythm of `ENGLISH_TRANSLATION`. If the translation uses a comma to join ideas, you are **FORBIDDEN** from upgrading it to a period.
* Use `AGGRESSIVENESS_LEVEL` to control punctuation-density matching; when `high`, mirror the full punctuation/rhythmic cadence of `ENGLISH_TRANSLATION` even with higher edit count.
* Insert punctuation at the earliest clear boundary where a new Subject-Verb clause begins.
* Add missing sentence-ending marks before any sentence-start capitalization.
* Boundary restoration rule: when `ENGLISH_TRANSLATION` clearly indicates multiple sentence boundaries, mirror those boundaries with language-appropriate sentence-ending marks unless contradicted by strong `ORIGINAL` evidence.
* If a boundary from `ENGLISH_TRANSLATION` resolves run-on ambiguity and is supported by `ORIGINAL`, prioritize that boundary over the fewest-edits tie-break.
* Run-on control: if a segment is long and contains 2+ clear clause boundaries, split into multiple sentences instead of keeping a single comma chain.
* For scripts without whitespace, resolve all inter-character spaces into punctuation (preferring commas for flow) or remove them.
* Conditional downgrade rule: if `ORIGINAL` is over-segmented (spurious full stops between tightly connected fragments of one thought), downgrade those specific sentence-ending marks to commas.
* Do **NOT** downgrade true sentence boundaries; keep full-stop boundaries for complete thoughts, topic shifts, or boundaries supported by `ENGLISH_TRANSLATION`.
* Do **NOT** insert punctuation adjacent to existing marks to avoid redundant or stacked punctuation.
* At segment end, use `seg_end` from `[SEGMENT_BOUNDARY_PREDICTION]`: if `high`, ensure the segment ends with a sentence-ending mark; if `low`, prefer no terminal punctuation change.

### [CASING]

* Capitalize sentence starts and proper nouns.
* Without a sentence boundary, do **NOT** introduce mid-sentence capitalization except proper nouns/entities.
* Use `AGGRESSIVENESS_LEVEL` strictly to scale casing intensity.
* If `ENGLISH_TRANSLATION` capitalizes a token (proper noun, entity, or acronym), you **MUST** capitalize the corresponding token in the output regardless of its form in `ORIGINAL` and whether it is embedded in non-Latin scripts (applies only to tokens with case distinction).
* Standardize casing for tokens in `NO_TOUCH`.
* At segment start, use `seg_start` from `[SEGMENT_BOUNDARY_PREDICTION]`: if `high`, capitalize the first token as a sentence start; if `low`, preserve original casing.
* Strictly no punctuation changes in this step; modify letter case only.

### [REMAIN_FIX]

* Final residual cleanup for misses in earlier **ACTIVE** steps only.
* Prioritize conservative edits; if confidence is low, keep input unchanged.
* Apply a residual fix only when explicit in `ORIGINAL` or strongly supported by `ENGLISH_TRANSLATION` structure.
* Scope gate: `REMAIN_FIX` may only fix issues that belong to the active chain steps already executed earlier in this run.
* `REMAIN_FIX` is forbidden from introducing edits that belong exclusively to any inactive step.
* Preserve `NO_TOUCH` spans and legal/entity protections exactly.
* Keep meaning unchanged: no semantic rewrites, paraphrases, reordering, style rewriting, or unsupported additions.
* Preserve the language/script variant; do **NOT** perform script conversion or script-variant substitution unless explicitly required by `ORIGINAL` evidence.
* Avoid adjacent redundant punctuation and align with `AGGRESSIVENESS_LEVEL`.

---

## 5. EXECUTION DISCIPLINE

1. **Chain Order:** Execute steps in the order listed in section 4.
2. **Inactive Steps:** Must be pass-through: `{"edits": [], "result": <unchanged input-to-that-step>}`; if `NO_TOUCH` is inactive, `no_touch_tokens` must be `[]`.
3. **Left-to-Right:** Process text exhaustively per step before moving to the next.
4. **No Type-Creep:** Each step performs only its assigned edit type (except `REMAIN_FIX`).
4. **No Type-Creep (Strict):** If an edit belongs to another step, leave it unchanged in the current step and defer it to the designated step.
5. **Completion Rule:** Emit each active step result only after exhaustively fixing all in-scope cases or explicitly leaving ambiguous cases unchanged.
6. **Step-Translation Discipline:** Every active step must consult `ENGLISH_TRANSLATION` under global policy and its own step rules.
7. **Edits Disambiguation:** Interpret each `[before, after]` left-to-right on the first unmatched occurrence in the current step input.
8. **REMAIN_FIX Scope Lock:** `REMAIN_FIX` may repair only misses from earlier active steps and must not backfill edits from inactive steps.

---

## 6. DETERMINISM TARGET

Prefer idempotent output on re-run, no meaningful segment loss, no hallucinated lexical content, and consistent punctuation/capitalization/spacing.

Repeated-span safety check before finalizing active steps: if the source contains repeated long spans, preserve their repetition structure unless duplication is clearly adjacent filler-level stutter.

---

## 7. DATA & SCHEMA

### Output Schema

```json
{
  "tokenization": {"tokens": []},
  "translation": "string",
  "aggressiveness_level": "low/medium/high",
  "speaker_scope": "single/multi",
  "seg_start": "high/medium/low",
  "seg_end": "high/medium/low",
  "ct_speaker": {"edits": [[before, after]], "result": "string"},
  "ct_combine": {"edits": [], "result": "string"},
  "no_touch_tokens": [],
  "ct_lexical": {"edits": [], "result": "string"},
  "ct_disfluency": {"edits": [], "result": "string"},
  "ct_format": {"edits": [], "result": "string"},
  "ct_numeral": {"edits": [], "result": "string"},
  "ct_punct": {"edits": [], "result": "string"},
  "ct_casing": {"edits": [], "result": "string"},
  "ct_remain_fix": {"edits": [], "result": "string"}
}

```

### [RUNTIME_CHAIN_POLICY]

- Active chain IDs: {active_chain_ids}
- Active payload step keys: {active_step_keys}
- Inactive payload step keys: {inactive_step_keys}
- For every inactive payload step key, edits MUST be [] and result MUST be pass-through unchanged.
- If any inactive payload step has non-empty edits, the output is invalid and will be retried.
- If NO_TOUCH is inactive, no_touch_tokens MUST be [].

### [LOCALE]

- The input audio locale is: {locale}
- If known, use the locale information to guide language-specific editing decisions.

### [CONTEXT]

- Previous segment ending: `{prev_context}`
- Next segment starting: `{next_context}`
- If context is empty, no neighboring segments are available; rely on `[SEGMENT_BOUNDARY_PREDICTION]` for boundary decisions.
- Use context only for boundary decisions (casing, punctuation). Do NOT edit or reference context content in the output.

### Input Transcript

```text
{input_transcript}

```
