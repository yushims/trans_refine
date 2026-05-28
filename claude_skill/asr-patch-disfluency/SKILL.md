---
name: asr-patch-disfluency
description: Remove fillers, false starts, and noise/non-speech tags (e.g., [NOISE], _bg noise_, [SPN/], _hmm_, [unin/]) from an ASR transcript segment. Preserves long repeated spans (prevents audio-drift) and only collapses short adjacent filler-like repetitions. Never removes content repeated across different speakers/turns. Scales with aggressiveness_level. Use as the DISFLUENCY step of the ASR patch chain.
---

# DISFLUENCY step

## Inputs
- Current text
- `aggressiveness_level`
- `no_touch_tokens`
- `mode`: `standard` or `conservative`

## Rules

- Remove fillers and false starts based on `aggressiveness_level`.
- Remove any known noise / non-speech tags (case-insensitive). Examples: `[NOISE]`, `_bg noise_`, `[SPN/]`, `_hmm_`, `[unin/]`, `_Speaker Noise_`, and similar bracketed/underscored markers.
- When removing tags, either delete them or replace only with punctuation strictly necessary for flow.
- Do **NOT** delete long repeated spans (prevents audio-drift).
- Only collapse repetitions when **all** hold: (a) adjacent, (b) repeated unit is short and filler-like, (c) no new lexical information appears between repeats.
- Never reduce repeated long-span counts in one segment, even by 1.
- For no-whitespace scripts, only add words when omission is unmistakable and the fix is shortest possible.
- Do **NOT** remove content repeated across different speakers or turns.
- Do **NOT** perform numeral/format normalization or token recombination here.
- Do **NOT** edit any span in `no_touch_tokens`.

## Conservative mode addition

- **Strictly forbidden** to replace fillers with other lexical tokens.

## Output

```json
{"edits": [["before", "after"]], "result": "string"}
```

## Standalone invocation (full top-level schema)

When this skill is invoked directly via `--use-skill asr-patch-disfluency --chain-steps DISFLUENCY` (no router), you MUST still emit the full ASR-patch top-level JSON schema below. Returning only the step-local fragment fails the pipeline's schema check.

- Put this step's edits/result in `ct_disfluency`.
- Every other `ct_*` MUST be `{"edits": [], "result": "<unchanged source text>"}` (no prior step ran).
- `tokenization` MUST be `{"tokens": [...]}` — best-effort tokenization of the source text.
- `translation` MUST be a string; use `""` if you cannot produce one.
- `aggressiveness_level`, `speaker_scope`, `seg_start`, `seg_end` MUST be present as strings using the allowed values.
- `no_touch_tokens` MUST be `[]` unless you also identify protected spans.
- No markdown fencing, no extra keys, no commentary outside the JSON object.

```json
{
  "tokenization": {"tokens": []},
  "translation": "string",
  "aggressiveness_level": "low/medium/high",
  "speaker_scope": "single/multi",
  "seg_start": "high/medium/low",
  "seg_end": "high/medium/low",
  "ct_speaker":    {"edits": [], "result": "string"},
  "ct_combine":    {"edits": [], "result": "string"},
  "no_touch_tokens": [],
  "ct_lexical":    {"edits": [], "result": "string"},
  "ct_disfluency": {"edits": [["before","after"]], "result": "string"},
  "ct_format":     {"edits": [], "result": "string"},
  "ct_numeral":    {"edits": [], "result": "string"},
  "ct_punct":      {"edits": [], "result": "string"},
  "ct_casing":     {"edits": [], "result": "string"},
  "ct_remain_fix": {"edits": [], "result": "string"}
}
```
