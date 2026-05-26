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
