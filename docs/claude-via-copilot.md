# Using Claude via GitHub Copilot

`run_copilot.py` drives the patch pipeline through the Copilot SDK (`copilot` package). It can target any Copilot-served model, including Claude (`claude-*`). The prompt sent to the model can come from two sources:

1. **Normal prompt mode** — a flat markdown prompt file (default: `prompt_patch.md`).
2. **Skill mode** — a Claude Agent Skill folder containing `SKILL.md` (and optional sub-skills).

Both modes feed through the same template loader (`load_prompt_template`) and the same `build_patch_prompt` substitution path, so the runtime contract (chain steps, retries, JSON schema, repair, JSONL output) is identical.

## Calling Claude through GitHub Copilot

Claude models are not called directly against Anthropic's API. They are reached through the **GitHub Copilot SDK** (`github-copilot-sdk`, imported in code as `from copilot import CopilotClient`), which proxies requests to whichever Claude deployments your Copilot entitlement exposes. The same `CopilotClient` is used for GPT-class and Claude-class model ids — only the `--model` string changes.

### Prerequisites

1. **GitHub Copilot subscription** with access to Claude models (Business / Enterprise plans, or an Individual plan with Claude enabled). Verify in your GitHub account: **Settings → Copilot → Models**.
2. **GitHub CLI signed in** — the SDK reads the cached Copilot OAuth token created by `gh auth login` (or by signing in to Copilot in VS Code). On Windows the token typically lives under `%LOCALAPPDATA%\github-copilot\`.

   ```powershell
   gh auth login          # if not already signed in
   gh auth status         # confirm token is present
   ```

3. **Install the SDK** (already pinned in [requirements.txt](../requirements.txt)):

   ```powershell
   pip install -r requirements.txt
   ```

   Key package: `github-copilot-sdk` (the `copilot` Python module).

### How the call is made

[run_copilot.py](../run_copilot.py) instantiates a `CopilotClient`, creates a session per request via [`build_copilot_session_parameters`](../common_copilot.py) (which wraps `CopilotClient.create_session(**params)`), and sends the patch prompt through [`send_copilot_once`](../common_copilot.py). Retries, timeouts, model-mismatch fallback, and repair-on-malformed-JSON are all handled inside [common_copilot.py](../common_copilot.py).

The `--model` value is forwarded verbatim to the SDK — Claude ids like `claude-haiku-4.5` or `claude-sonnet-4.5` are routed by Copilot to the corresponding Anthropic deployment.

### Listing available Claude models

Available model ids depend on your Copilot plan and change over time. List them with:

```powershell
python run_copilot.py --print-models
```

Pick any id starting with `claude-` for Claude-via-Copilot. The examples in this document use `claude-haiku-4.5`.

### Selecting the model

Either pass `--model` on the command line:

```powershell
python run_copilot.py --model claude-haiku-4.5 --input-file sample_multi_input.txt --output-file outputs/sample_multi_output.txt
```

…or set the `COPILOT_MODEL` environment variable (loaded via `python-dotenv`, so a `.env` file in the workspace root also works):

```powershell
$env:COPILOT_MODEL = "claude-haiku-4.5"
python run_copilot.py --input-file sample_multi_input.txt --output-file outputs/sample_multi_output.txt
```

CLI `--model` takes precedence over the env var.

### Minimal smoke test

To confirm Claude-via-Copilot is wired correctly before running the full pipeline:

```powershell
python run_copilot.py `
  --model claude-haiku-4.5 `
  --input-file sample_multi_input.txt `
  --output-file outputs/sample_multi_smoke.txt
```

A successful run prints `Wrote ... .jsonl` and `Wrote ... .txt` and produces both files under [outputs/](../outputs/). Authentication failures surface as HTTP 401 from the SDK; entitlement failures surface as `Model '<id>' is not available`.

## Common setup

Pick a Claude model exposed by your Copilot deployment. **Always list available models first** — exact model IDs depend on your Copilot entitlement and change over time:

```powershell
python run_copilot.py --list-models-only
```

Then set it via `--model` or the `COPILOT_MODEL` env var. The examples below use `claude-haiku-4.5` (verified available in this deployment — same id used by [run_copilot.bat](../run_copilot.bat)). Substitute another id (e.g. `claude-haiku-4.5`) if your `--list-models-only` output lists it:

```powershell
$env:COPILOT_MODEL = "claude-haiku-4.5"
```

The input/output paths in the examples below use the bundled [sample_multi_input.txt](../sample_multi_input.txt) (one transcript per line, supports `# locale` headers). Outputs go to [outputs/](../outputs/) (create the folder if missing). Replace with your own files as needed.

## Mode 1 — Normal prompt (flat markdown)

This is the default. The patch prompt is `prompt_patch.md` (or `prompt_patch_conservative.md` etc.), with `{input_transcript}`, `{chain_steps}`, `{locale}`, `{prev_context}`, `{next_context}` placeholders.

```powershell
python run_copilot.py `
  --model claude-haiku-4.5 `
  --input-file sample_multi_input.txt `
  --output-file outputs/sample_multi_output.txt `
  --patch-prompt-file prompt_patch.md `
  --repair-prompt-file prompt_repair.md
```

Key prompt-related flags ([common.py `add_run_pipeline_cli_arguments`](common.py)):

| Flag | Default | Purpose |
|------|---------|---------|
| `--patch-prompt-file` | `prompt_patch.md` | Patch prompt template. |
| `--repair-prompt-file` | `prompt_repair.md` | Strict-JSON repair prompt used when the model returns malformed JSON. |
| `--chain-steps` | all | Repeatable selector; ids `0`-`9` or step names (`SPEAKER`, `COMBINE`, `NO_TOUCH`, `LEXICAL`, `DISFLUENCY`, `FORMAT`, `NUMERAL`, `PUNCT`, `CASING`, `REMAIN_FIX`). |
| `--locale` | `None` | Locale hint substituted into the prompt. |

The conservative variant of the prompt is just another file — swap `--patch-prompt-file`:

```powershell
python run_copilot.py `
  --model claude-haiku-4.5 `
  --input-file sample_multi_input.txt `
  --output-file outputs/sample_multi_output_conservative.txt `
  --patch-prompt-file prompt_patch_conservative.md `
  --repair-prompt-file prompt_repair.md
```

## Mode 2 — Claude Agent Skill

A skill is a directory containing `SKILL.md` with YAML frontmatter (`name`, `description`) plus instructions. This repo's skills live under [claude_skill/](claude_skill/):

```
claude_skill/
  asr-patch-router/SKILL.md       # orchestrator (recommended default)
  asr-patch-preprocess/SKILL.md
  asr-patch-speaker/SKILL.md
  asr-patch-combine/SKILL.md
  asr-patch-no-touch/SKILL.md
  asr-patch-lexical/SKILL.md
  asr-patch-disfluency/SKILL.md
  asr-patch-format/SKILL.md
  asr-patch-numeral/SKILL.md
  asr-patch-punct/SKILL.md
  asr-patch-casing/SKILL.md
  asr-patch-remain-fix/SKILL.md
```

Invoke the router as the patch prompt:

```powershell
python run_copilot.py `
  --model claude-haiku-4.5 `
  --input-file sample_multi_input.txt `
  --output-file outputs/sample_multi_output_skill.txt `
  --use-skill asr-patch-router
```

Invoke a single step skill in isolation (useful for unit-testing one step). Per-step `SKILL.md` files don't contain the router's `Runtime invocation` template block, so `build_patch_prompt` falls back to appending the ASR disclaimer + transcript automatically. The step's rules still apply, but `{chain_steps}` / `{prev_context}` / `{next_context}` are **not** substituted:

```powershell
python run_copilot.py `
  --model claude-haiku-4.5 `
  --input-file sample_multi_input.txt `
  --output-file outputs/sample_multi_output_lexical.txt `
  --use-skill asr-patch-lexical `
  --chain-steps LEXICAL
```

Use a non-default skill root directory (the folder must contain `<use-skill>/SKILL.md`):

```powershell
python run_copilot.py `
  --model claude-haiku-4.5 `
  --input-file sample_multi_input.txt `
  --output-file outputs/sample_multi_output_skill_dir.txt `
  --use-skill asr-patch-router `
  --skill-dir claude_skill
```

### New CLI flags

Added in [common.py](common.py) `add_run_pipeline_cli_arguments` and wired in [run_copilot.py](run_copilot.py):

| Flag | Default | Purpose |
|------|---------|---------|
| `--use-skill <name>` | `None` | Load patch prompt from `<skill-dir>/<name>/SKILL.md`. |
| `--skill-dir <path>` | `claude_skill` | Root directory holding skill folders. Used only when `--use-skill` is set. |

### Resolution rules

Implemented by `resolve_skill_patch_prompt_override(use_skill, skill_dir, explicit_patch_prompt_file)`:

1. If `--use-skill` is **not** set → use `--patch-prompt-file` (or its default).
2. If `--use-skill` is set **and** `--patch-prompt-file` is also explicitly set → the explicit file wins (skill flag is ignored).
3. If `--use-skill` is set and `--patch-prompt-file` is not → resolve to `<skill-dir>/<use-skill>/SKILL.md`. Missing path → error and exit.

The startup banner prints `Using Claude skill: <name> (skill-dir: <dir>)` when skill mode is active.

### How skill mode reuses the existing pipeline

- `SKILL.md` is read by the same `load_prompt_template` used for flat prompts.
- `build_patch_prompt` substitutes `{input_transcript}`, `{chain_steps}`, `{active_chain_ids}`, `{active_step_keys}`, `{inactive_step_keys}`, `{locale}`, `{prev_context}`, `{next_context}` wherever they appear.
- The router's `SKILL.md` ends with a **Runtime invocation** section that contains exactly those placeholders, so a single substitution pass produces a self-contained prompt for the model.
- Repair flow, JSON schema validation, retry/timeout policy, long-span guard, hallucination guard, and JSONL output all behave identically.

## Picking standard vs conservative behavior

| Goal | Normal prompt | Skill |
|------|---------------|-------|
| Default balanced patching | `--patch-prompt-file prompt_patch.md` | `--use-skill asr-patch-router` (default mode `standard`) |
| Spoken-form / dialect preservation | `--patch-prompt-file prompt_patch_conservative.md` | `--use-skill asr-patch-router` — say "conservative" / "preserve dialect" in your input, or pre-set the active mode in the router-input metadata |

In skill mode the router selects mode based on cues in the input (see [claude_skill/asr-patch-router/SKILL.md](claude_skill/asr-patch-router/SKILL.md), Step 2). To force a mode, add `mode=conservative` to your input metadata or use the conservative variant of the affected step skills directly.

## Locale / language profile

- `--locale ar-SA` (or any BCP-47 tag) is substituted into the prompt and used by the skill router's Step 1 to pick an `active_language_profile` (`generic`, `arabic`, `cjk`, `latin-other`).
- If `--locale` is omitted, the router infers from script + lexical evidence on the input.
- Profile rules **narrow** behavior — they never enable rewrites or reorders. On conflict with a generic step rule, the stricter rule wins. See the profile registry in the router for the full table.

## When to choose which mode

| Use the **normal prompt** when… | Use a **skill** when… |
|---|---|
| You want the smallest possible context (single flat file). | You want per-step modularity and the router to gate each step on detected signals. |
| You're iterating on a single monolithic prompt. | You want to swap or A/B individual steps without re-editing the master prompt. |
| You're comparing against the historical baseline. | You want locale-aware activation (e.g., auto-disable `casing` for case-less scripts). |
| You target non-Claude Copilot models that don't benefit from the skill metaphor. | You target Claude and want to leverage the Agent Skill conventions (frontmatter `name` + `description` discovery, progressive disclosure). |

## Notes & limitations

- Skill mode is currently wired into `run_copilot.py` only. `run_aoai.py` still uses the flat prompt path; the same 3-line change there will enable it if you want symmetry.
- Sub-skill invocation is **logical**, not transport-level: the router skill instructs the model to follow the per-step rules within one response. There is no separate API call per step skill.
- The skill `description` is what Claude's discovery mechanism matches on — keep it rich with trigger phrases and key field names.
- `--patch-prompt-file` always wins over `--use-skill` when both are explicit, so you can force a particular SKILL.md path or fall back to a flat prompt for a single run without removing the skill flag.
