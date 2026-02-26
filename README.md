# Transcription Refinement Pipeline (AOAI + Copilot)

Transcription refinement pipeline with shared JSON-schema validation and one-pass strict JSON repair.

## What this repo contains

- `run_aoai.py`: AOAI pipeline runner.
- `run_copilot.py`: Copilot pipeline runner.
- `pipeline_common.py`: shared parsing/validation/output utilities.
- `run-aoai.ps1`: AOAI launcher.
- `run-copilot.ps1`: Copilot launcher.
- Prompt templates:
	- `prompt_patch.txt`
	- `prompt_repair.txt`

## Prerequisites

- Python 3.13 preferred (or available Python from PATH).
- PowerShell on Windows.
- AOAI credentials available for AOAI flow.
- Copilot runtime dependency available for Copilot flow (`copilot` package/module).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dependency check

After install, run a quick AOAI import check:

```powershell
python -c "from openai import AzureOpenAI; print('AOAI import OK')"
```

Then run a Copilot import check:

```powershell
python -c "from copilot import CopilotClient; print('Copilot import OK')"
```

## Quick start

### AOAI

```powershell
./run-aoai.ps1 -HelpOnly
./run-aoai.ps1
```

Defaults:
- Deployment: `gpt-5-chat`.
- Deterministic knobs: `Temperature=0`, `TopP=1` (override with `-Temperature`, `-TopP`).
- Concurrency: `10` (override with `-Concurrency`).
- Timeout: `600s` (override with `-Timeout`).
- Timeout retries: `2` (override with `-TimeoutRetries`).
- Empty-result retries: `2` (override with `-EmptyResultRetries`).
- Output base path: `sample_multi_output_aoai` (override with `-OutputFile`).
- Derived outputs: JSON (`.json`) and plain text (`.txt`) are auto-generated.

### Copilot

```powershell
./run-copilot.ps1 -HelpOnly
./run-copilot.ps1 -ListModelsOnly
./run-copilot.ps1 -PrintModels
./run-copilot.ps1 -Concurrency 10
./run-copilot.ps1
```

Defaults:
- Model: `gpt-5.2` (override with `-Model`).
- Concurrency: `10` (override with `-Concurrency`).
- Timeout: `600s` (override with `-Timeout`).
- Timeout retries: `2` (override with `-TimeoutRetries`).
- Empty-result retries: `2` (override with `-EmptyResultRetries`).
- Output file base path: `sample_multi_output_copilot` (override with `-OutputFile`).
- Derived output files: JSON (`.json`) and plain text (`.txt`) are auto-generated.
- Model list printing at startup: disabled by default.

## E2E evaluation

Use the unified retry flag across all evaluated models:

```powershell
python .\run_e2e_evaluation.py --empty-result-retries 2
python .\run_e2e_evaluation.py --empty-result-retries 2 --timeout 600 --timeout-retries 2
```

## Input / output

- Default input: `sample_multi_input.txt` (one transcription per line).
- Output file path is configurable via launcher args.
- Text output file contains joined `corrected_text` lines.

## Common flags

- `--patch-prompt-file`
- `--repair-prompt-file`
- `--input-file`
- `--output-file`
- `--list-models-only`
- `--print-models`
- `--concurrency`

Use `-HelpOnly` on either PowerShell launcher to see effective CLI options.

## Notes on secrets

- Keep secrets out of source-controlled files.
- Do not commit any secret-bearing files.

## Minimal files for repro handoff

- `requirements.txt`
- `pipeline_common.py`
- `run_aoai.py`
- `run_copilot.py`
- `launcher-common.ps1`
- `run-aoai.ps1`
- `run-copilot.ps1`
- `prompt_patch.txt`
- `prompt_repair.txt`
- `sample_multi_input.txt` (optional, for quick demo)