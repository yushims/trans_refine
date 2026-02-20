# Transcription Refinement Pipeline (AOAI + Copilot)

Deterministic transcription refinement pipeline with shared JSON-schema validation and one-pass strict JSON repair.

## What this repo contains

- `run_aoai.py`: AOAI pipeline runner.
- `run_copilot.py`: Copilot pipeline runner.
- `pipeline_common.py`: shared parsing/validation/output utilities.
- `run-aoai.ps1`: AOAI launcher + env wiring.
- `run-copilot.ps1`: Copilot launcher + env wiring.
- `copilot-env.ps1`: helper for Copilot process env vars.
- Prompt templates:
	- `prompt_patch_aoai.txt`
	- `prompt_repair_aoai.txt`
	- `prompt_patch_copilot.txt`
	- `prompt_repair_copilot.txt`

## Prerequisites

- Python 3.13 preferred (or available Python from PATH).
- PowerShell on Windows.
- AOAI key for AOAI flow (`AZURE_OPENAI_API_KEY`).
- Copilot runtime dependency available for Copilot flow (`copilot` package/module in your environment).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

### AOAI

```powershell
./run-aoai.ps1 -HelpOnly
./run-aoai.ps1
```

Defaults:
- deployment: `gpt-5-chat`
- deterministic knobs: `Temperature=0`, `TopP=1`
- strict JSON repair: enabled
- output validation: enabled

### Copilot

```powershell
./run-copilot.ps1 -HelpOnly
./run-copilot.ps1
```

Defaults:
- model: `gpt-5.2-thinking`
- strict JSON repair: enabled
- output validation: enabled

Model note:
- `COPILOT_MODEL` / `--model` is treated as a requested model name.
- In this SDK path, backend may resolve/fallback the model and may not expose the final effective model ID in response metadata.

## Input / output

- Default input: `sample_multi_input.txt` (one transcription per line).
- JSON output path is configurable via launcher args.
- Plain text output contains joined `corrected_text` lines.

## Common flags

- `--prompt-file`
- `--repair-prompt-file`
- `--input-file`
- `--output-file`
- `--output-plain-file`
- `--validate-output {0,1}`

Use `-HelpOnly` on either PowerShell launcher to see effective CLI options.

## Notes on secrets

- Keep secrets in `.env` or process env vars.
- Do not commit `.env` or secret-bearing files (for example `env.txt` with real keys).

## Minimal files for repro handoff

- `requirements.txt`
- `pipeline_common.py`
- `run_aoai.py`
- `run_copilot.py`
- `run-aoai.ps1`
- `run-copilot.ps1`
- `copilot-env.ps1`
- `prompt_patch_aoai.txt`
- `prompt_repair_aoai.txt`
- `prompt_patch_copilot.txt`
- `prompt_repair_copilot.txt`
- `sample_multi_input.txt` (optional, for quick demo)