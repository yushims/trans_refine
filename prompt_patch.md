# DETERMINISTIC PATCH MODE (Multilingual, ASR-Aware)

## 0. CRITICAL OUTPUT CONTRACT

**Return exactly ONE JSON object (no markdown/prose) with these EXACT keys:**
`tokenization`, `translation`, `aggressiveness_level`, `speaker_scope`, `ct_speaker`, `ct_combine`, `no_touch_tokens`, `ct_lexical`, `ct_disfluency`, `ct_format`, `ct_numeral`, `ct_punct`, `ct_casing`, `ct_remain_fix`.

---

## 1. STRATEGIC GOALS & CONSTRAINTS

* Single-run patch using the ordered chain below.
* Tie-breaks: highest structural alignment with `ENGLISH_TRANSLATION` → earliest position → fewest lexical edits.
* Allowed edits: typos, spacing/formatting, disfluency cleanup, minimal local grammar fixes, punctuation for run-ons, closing pairs, "staccato" short sentences, discourse markers, and interjections.
* Forbidden: rewrites, paraphrases, reordering, style/idiom edits; semantic-content deletion or unsupported lexical additions (hallucinations).
* Keep `ORIGINAL` language/script variant unchanged by default. Do **NOT** perform transliteration or orthographic normalization unless explicit in-line evidence requires it.
* For writing systems with multiple orthographic variants, treat character form as lexical content; do **NOT** swap characters for normalization.
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
* Keep names, numbers, and protected spans unchanged.
* This is a **reference artifact only**; it does not trigger schema changes.

### [AGGRESSIVENESS_LEVEL]

* Mapping: `low` (conservative), `medium` (balanced), `high` (aggressive punctuation/fluency).
* This value is locked for the full chain. Never violate Forbidden constraints regardless of level.

### [SPEAKER_SCOPE]

* Output exactly `single` or `multi`.
* Use `multi` only with explicit evidence of turn-taking; otherwise default to `single`.

---

## 3. GLOBAL CROSS-STEP POLICY

**Applies to PHASE 2 only:** Treat `ENGLISH_TRANSLATION` as the **Target Logic State**.

* Lexical content **MUST** come from `ORIGINAL`.
* Formatting, casing, and punctuation density **SHOULD** follow `ENGLISH_TRANSLATION`.
* Do **NOT** use translation to trigger rewrites or semantic substitutions.

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
* Forbidden: no year expansion or numeral conversion (defer to `NUMERAL`).

### [NO_TOUCH]

* Extract immutable spans from `COMBINE` result for proper names and legal entities.
* Extract spans as exact substrings from `COMBINE` output only; do **NOT** extract from `ENGLISH_TRANSLATION`.
* Total ban on lexical editing, script-swapping, or reordering within these spans for the rest of the chain.

### [LEXICAL]

* Fix typos and phonetic ASR errors.
* Preserve pronunciation similarity. Use `ENGLISH_TRANSLATION` for disambiguation only.
* For Arabic input, lexical edits must be limited to spelling/orthography standardization toward formal MSA writing while preserving meaning and original style.
* For Arabic input, remove diacritics but do not replace base letters unless fixing a clear spelling error supported by `ORIGINAL` evidence.
* Forbidden: do **NOT** normalize brand/proper-name forms from `ENGLISH_TRANSLATION` alone; require explicit in-line `ORIGINAL` evidence.
* Do **NOT** edit any protected token/span from `NO_TOUCH` (no replace/insert/delete/reorder).
* You are **STRICTLY FORBIDDEN** from reordering tokens to match English syntax. Even if `ENGLISH_TRANSLATION` suggests a different word order, you **MUST** maintain the original token sequence from `ORIGINAL`.

### [DISFLUENCY]

* Remove fillers and false starts based on `AGGRESSIVENESS_LEVEL`.
* Forbidden: do not delete long repeated spans (prevents audio-drift).
* For no-whitespace scripts, only add words when omission is unmistakable and the fix is shortest possible.
* Do **NOT** remove content that is repeated across different speakers or turns.
* Do **NOT** perform numeral/format normalization or token recombination in this step.
* Do **NOT** edit any protected token/span from `NO_TOUCH` (no replace/insert/delete/reorder).

### [FORMAT]

* Fix spacing artifacts in symbols/abbreviations.
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
* For scripts without whitespace, resolve all inter-character spaces into punctuation (preferring commas for flow) or remove them.
* Downgrade sentence-ending marks to commas unless there is a significant topic shift or thought finalization.
* Do **NOT** insert punctuation adjacent to existing marks to avoid redundant or stacked punctuation.
* At segment end, avoid changing terminal punctuation by default; allow change only when strongly supported by clear clause completion within the current segment.

### [CASING]

* Capitalize sentence starts and proper nouns.
* Without a sentence boundary, do **NOT** introduce mid-sentence capitalization except proper nouns/entities.
* Use `AGGRESSIVENESS_LEVEL` strictly to scale casing intensity.
* Standardize casing for technical platforms even if input is lowercase.
* Standardize casing for tokens in `NO_TOUCH`.
* If `ENGLISH_TRANSLATION` capitalizes it, you **MUST** capitalize it here.
* At segment start, avoid changing first-token casing by default; allow change only when sentence/clause start is clearly indicated within the current segment.
* Strictly no punctuation changes in this step; modify letter case only.

### [REMAIN_FIX]

* Final residual cleanup for misses in earlier **ACTIVE** steps only.
* Prioritize conservative edits; if confidence is low, keep input unchanged.
* Apply a residual fix only when explicit in `ORIGINAL` or strongly supported by `ENGLISH_TRANSLATION` structure.
* Scope gate: `REMAIN_FIX` may only fix issues that belong to steps included in `{chain_steps}` and already executed earlier in this run.
* `REMAIN_FIX` is forbidden from introducing edits that belong exclusively to any inactive step.
* Preserve `NO_TOUCH` spans and legal/entity protections exactly.
* Keep meaning unchanged: no semantic rewrites, paraphrases, reordering, style rewriting, or unsupported additions.
* Preserve the language/script variant; do **NOT** perform script conversion or script-variant substitution unless explicitly required by `ORIGINAL` evidence.
* Avoid adjacent redundant punctuation and align with `AGGRESSIVENESS_LEVEL`.

---

## 5. EXECUTION DISCIPLINE

1. **Chain Order:** `{chain_steps}`.
2. **Inactive Steps:** Must be pass-through: `{"edits": [], "result": <unchanged>}`.
3. **Left-to-Right:** Process text exhaustively per step before moving to the next.
4. **No Type-Creep:** Each step performs only its assigned edit type (except `REMAIN_FIX`).
5. **Completion Rule:** Emit each active step result only after exhaustively fixing all in-scope cases or explicitly leaving ambiguous cases unchanged.
6. **Step-Translation Discipline:** Every active step must consult `ENGLISH_TRANSLATION` under global policy and its own step rules.
7. **Edits Disambiguation:** Interpret each `[before, after]` left-to-right on the first unmatched occurrence in the current step input.
8. **REMAIN_FIX Scope Lock:** `REMAIN_FIX` may repair only misses from earlier active steps in `{chain_steps}` and must not backfill edits from inactive steps.

---

## 6. OUTPUT FIELDS (ALWAYS PRESENT)

* `ct_speaker` = `{"edits": [[before, after], ...], "result": text after SPEAKER}`
* `ct_combine` = `{"edits": [[before, after], ...], "result": text after COMBINE}`
* `no_touch_tokens` = unique exact substrings extracted in `NO_TOUCH` first-appearance order (or `[]` if inactive)
* `ct_lexical` = `{"edits": [[before, after], ...], "result": text after LEXICAL}`
* `ct_disfluency` = `{"edits": [[before, after], ...], "result": text after DISFLUENCY}`
* `ct_format` = `{"edits": [[before, after], ...], "result": text after FORMAT}`
* `ct_numeral` = `{"edits": [[before, after], ...], "result": text after NUMERAL}`
* `ct_punct` = `{"edits": [[before, after], ...], "result": text after PUNCT}`
* `ct_casing` = `{"edits": [[before, after], ...], "result": text after CASING}`
* `ct_remain_fix` = `{"edits": [[before, after], ...], "result": text after REMAIN_FIX}`
* Use `{"edits": [], "result": <unchanged>}` when no edits were made.

---

## 7. DETERMINISM TARGET

Prefer idempotent output on re-run, no meaningful segment loss, no hallucinated lexical content, and consistent punctuation/capitalization/spacing.

---

## 8. DATA & SCHEMA

### Output Schema

```json
{
  "tokenization": {"tokens": []},
  "translation": "string",
  "aggressiveness_level": "low/medium/high",
  "speaker_scope": "single/multi",
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

### Input Transcript

```text
{input_transcript}

```
