import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AOAI/Copilot/Gemini E2E evaluation using prompt-based checks."
    )
    parser.add_argument("--input-file", default="sample_multi_input.txt")
    parser.add_argument("--prefix", default="eval")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--timeout-retries", type=int, default=2)
    parser.add_argument("--empty-result-retries", type=int, default=2)
    parser.add_argument("--skip-existing-results", action="store_true")
    parser.add_argument("--copilot-model", default="gpt-5.2")
    parser.add_argument("--gemini-model", default="gemini-3-pro-preview")
    parser.add_argument("--prompt-file", default="prompt_eval.md")
    parser.add_argument("--aoai-deployment", default="gpt-5-chat")
    parser.add_argument("--aoai-endpoint", default="https://adaptationdev-resource.openai.azure.com/")
    parser.add_argument("--aoai-api-version", default="2025-01-01-preview")
    parser.add_argument("--eval-timeout", type=float, default=600.0)
    parser.add_argument("--eval-timeout-retries", type=int, default=2)
    parser.add_argument("--eval-empty-result-retries", type=int, default=2)
    parser.add_argument("--eval-model-mismatch-retries", type=int, default=2)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-retry-temperature-jitter", type=float, default=0.08)
    parser.add_argument("--eval-retry-top-p-jitter", type=float, default=0.03)
    parser.add_argument("--eval-concurrency", type=int, default=10)
    parser.add_argument("--eval-repair-prompt-file", default="prompt_repair.md")
    parser.add_argument(
        "--chain-steps",
        action="append",
        help="Repeatable active-chain selector (ids 1-8 or step names like COMBINE, NO_TOUCH).",
    )
    return parser.parse_args()


def non_empty_line_count(file_path: Path) -> int:
    return sum(1 for line in file_path.read_text(encoding="utf-8").split('\n') if line.strip())


def run_command(
    command: list[str],
    log_path: Path,
) -> tuple[int, str, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"[E2E] Command: {' '.join(command)}\n")
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed_seconds = time.perf_counter() - start_time
        log_file.write(f"\n[E2E] Attempt runtime: {elapsed_seconds:.2f}s\n")
        log_file.write(f"[E2E] Exit code: {process.returncode}\n")

    output = log_path.read_text(encoding="utf-8", errors="replace")
    return process.returncode, output, elapsed_seconds


def extract_json_error_description(output: str) -> str:
    if not output:
        return ""

    key_detail_matches = re.findall(r"JSON_TOP_LEVEL_KEY_ERROR:\s*(.+)", output)
    if key_detail_matches:
        return f"JSON_TOP_LEVEL_KEY_ERROR: {key_detail_matches[-1].strip()}"

    marker_matches = re.findall(r"JSON_FORMAT_ERROR:\s*(.+)", output)
    if marker_matches:
        return marker_matches[-1].strip()

    schema_match = re.search(r"Schema error:\s*(.+)", output)
    if schema_match:
        return f"Schema error: {schema_match.group(1).strip()}"

    repair_invalid_match = re.search(r"Repair attempt did not return valid JSON\.\s*(.+)", output)
    if repair_invalid_match:
        detail = repair_invalid_match.group(1).strip()
        if detail and not detail.startswith("Raw output:"):
            return f"Repair attempt did not return valid JSON: {detail}"
        return "Repair attempt did not return valid JSON"

    if "Initial response was not valid strict JSON." in output:
        return "Initial response was not valid strict JSON"

    return ""


def resolve_default_prompt_path(default_filename: str) -> Path:
    default_path = Path(__file__).with_name(default_filename)
    return default_path


def run_model(
    model: str,
    script_path: Path,
    input_file: Path,
    prefix: str,
    runs: int,
    attempts: int,
    timeout: int,
    timeout_retries: int,
    empty_result_retries: int,
    expected_lines: int,
    skip_existing_results: bool,
    extra_args: list[str] | None = None,
) -> dict[str, str]:

    run_error_descriptions: dict[str, str] = {}
    logs_dir = Path(f"{prefix}_results")
    patch_prompt_path = resolve_default_prompt_path("prompt_patch.md")
    repair_prompt_path = resolve_default_prompt_path("prompt_repair.md")

    for run_index in range(1, runs + 1):
        out_txt = logs_dir / f"run{run_index}_{model}.txt"
        out_json = logs_dir / f"run{run_index}_{model}.json"

        if skip_existing_results and out_txt.exists() and out_json.exists():
            lines = out_txt.read_text(encoding="utf-8").split('\n')
            if lines and lines[-1] == '':
                lines.pop()
            non_empty = sum(1 for line in lines if line.strip())
            if len(lines) >= expected_lines and non_empty >= expected_lines:
                print(
                    f"{model.upper()} run {run_index} existing result detected; "
                    "skipping run_model execution"
                )
                continue

        ok = False
        last_json_error = ""
        for attempt_index in range(1, attempts + 1):
            print(f"{model.upper()} run {run_index} attempt {attempt_index}")

            command = [
                sys.executable,
                str(script_path),
                "--input-file",
                str(input_file),
                "--output-file",
                str(out_json),
                "--patch-prompt-file",
                str(patch_prompt_path),
                "--repair-prompt-file",
                str(repair_prompt_path),
                "--timeout",
                str(timeout),
                "--timeout-retries",
                str(timeout_retries),
                "--empty-result-retries",
                str(empty_result_retries),
            ]
            if extra_args:
                command.extend(extra_args)

            log_path = logs_dir / f"run{run_index}_{model}_attempt{attempt_index}.log"
            exit_code, output, attempt_runtime_seconds = run_command(command, log_path)
            print(f"{model.upper()} log: {log_path}")
            print(
                f"{model.upper()} run {run_index} attempt {attempt_index} "
                f"summary: exit={exit_code}, runtime={attempt_runtime_seconds:.2f}s"
            )
            if output:
                json_error = extract_json_error_description(output)
                if json_error:
                    last_json_error = json_error

            if exit_code != 0:
                print(f"{model.upper()} run {run_index} attempt {attempt_index} failed with exit={exit_code}")
                continue

            if not out_txt.exists():
                print(f"{model.upper()} run {run_index} attempt {attempt_index} missing text output")
                continue

            lines = out_txt.read_text(encoding="utf-8").split('\n')
            if lines and lines[-1] == '':
                lines.pop()
            non_empty = sum(1 for line in lines if line.strip())
            if len(lines) >= expected_lines and non_empty >= expected_lines:
                ok = True
                print(f"{model.upper()} run {run_index} success with non-empty output")
                break

            print(
                f"{model.upper()} run {run_index} incomplete: "
                f"lines={len(lines)}, non_empty={non_empty}, expected={expected_lines}"
            )

        if not ok:
            print(f"{model.upper()} run {run_index} did not reach complete non-empty output after retries")
            out_txt_name = f"run{run_index}_{model}.txt"
            if last_json_error:
                run_error_descriptions[out_txt_name] = last_json_error

    return run_error_descriptions


def run_eval(
    script_path: Path,
    script_args: list[str],
    prefix: str,
    log_name: str | None = None,
) -> None:
    command = [
        sys.executable,
        str(script_path),
        *script_args,
    ]
    logs_dir = Path(f"{prefix}_results")
    log_filename = log_name or f"{script_path.stem}.log"
    log_path = logs_dir / log_filename
    exit_code, _output, runtime_seconds = run_command(command, log_path)
    print(f"EVAL script {script_path} log: {log_path}")
    print(f"EVAL script {script_path} summary: exit={exit_code}, runtime={runtime_seconds:.2f}s")
    if exit_code != 0:
        raise RuntimeError(f"Evaluation script failed: {script_path}. See {log_path}")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    expected_lines = non_empty_line_count(input_file)
    chain_steps = [step for step in (args.chain_steps or []) if isinstance(step, str) and step.strip()]
    run_chain_step_args: list[str] = []
    for step in chain_steps:
        run_chain_step_args.extend(["--chain-steps", step])

    eval_chain_steps_text = ", ".join(chain_steps) if chain_steps else "ALL"

    aoai_errors = run_model(
        model=f"aoai-{args.aoai_deployment}",
        script_path=Path("run_aoai.py"),
        input_file=input_file,
        prefix=args.prefix,
        runs=args.runs,
        attempts=args.attempts,
        timeout=args.timeout,
        timeout_retries=args.timeout_retries,
        empty_result_retries=args.empty_result_retries,
        expected_lines=expected_lines,
        skip_existing_results=args.skip_existing_results,
        extra_args=[
            "--deployment", args.aoai_deployment,
            "--endpoint", args.aoai_endpoint,
            "--api-version", args.aoai_api_version,
            *run_chain_step_args,
        ],
    )

    copilot_errors = run_model(
        model=f"copilot-{args.copilot_model}",
        script_path=Path("run_copilot.py"),
        input_file=input_file,
        prefix=args.prefix,
        runs=args.runs,
        attempts=args.attempts,
        timeout=args.timeout,
        timeout_retries=args.timeout_retries,
        empty_result_retries=args.empty_result_retries,
        expected_lines=expected_lines,
        skip_existing_results=args.skip_existing_results,
        extra_args=["--model", args.copilot_model, *run_chain_step_args],
    )

    gemini_errors = run_model(
        model=f"copilot-{args.gemini_model}",
        script_path=Path("run_copilot.py"),
        input_file=input_file,
        prefix=args.prefix,
        runs=args.runs,
        attempts=args.attempts,
        timeout=args.timeout,
        timeout_retries=args.timeout_retries,
        empty_result_retries=args.empty_result_retries,
        expected_lines=expected_lines,
        skip_existing_results=args.skip_existing_results,
        extra_args=["--model", args.gemini_model, *run_chain_step_args],
    )

    all_errors = {}
    all_errors.update(aoai_errors)
    all_errors.update(copilot_errors)
    all_errors.update(gemini_errors)

    run_errors_path = Path(f"{args.prefix}_results") / f"{args.prefix}_run_errors.json"
    run_errors_path.parent.mkdir(parents=True, exist_ok=True)
    run_errors_path.write_text(json.dumps(all_errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    patch_result_dir = Path(f"{args.prefix}_results")
    generated_output_models = [
        f"aoai-{args.aoai_deployment}",
        f"copilot-{args.copilot_model}",
        f"copilot-{args.gemini_model}",
    ]
    patch_result_files: list[str] = []
    for model_name in generated_output_models:
        patch_result_files.extend(
            [
                str(patch_result_dir / f"run{i}_{model_name}.txt")
                for i in range(1, args.runs + 1)
            ]
        )
    patch_result_file_value = ",".join(patch_result_files)
    eval_output_prefix = Path(str(args.prefix)).stem or args.prefix

    common_eval_args = [
        "--orginal-trans-file", str(input_file),
        "--prefix", eval_output_prefix,
        "--patch-result-file", patch_result_file_value,
        "--prompt-file", args.prompt_file,
        "--chain-steps", eval_chain_steps_text,
    ]

    aoai_eval_args = [
        *common_eval_args,
        "--repair-prompt-file", args.eval_repair_prompt_file,
        "--deployment", args.aoai_deployment,
        "--endpoint", args.aoai_endpoint,
        "--api-version", args.aoai_api_version,
        "--timeout", str(args.eval_timeout),
        "--timeout-retries", str(args.eval_timeout_retries),
        "--empty-result-retries", str(args.eval_empty_result_retries),
        "--temperature", str(args.eval_temperature),
        "--top-p", str(args.eval_top_p),
        "--retry-temperature-jitter", str(args.eval_retry_temperature_jitter),
        "--retry-top-p-jitter", str(args.eval_retry_top_p_jitter),
        "--concurrency", str(max(1, args.eval_concurrency)),
    ]

    run_eval(
        Path("eval_aoai.py"),
        aoai_eval_args,
        args.prefix,
        log_name=f"eval_aoai-{args.aoai_deployment}.log",
    )

    run_eval(
        Path("eval_copilot.py"),
        [
            *common_eval_args,
            "--model", args.copilot_model,
            "--timeout", str(args.eval_timeout),
            "--timeout-retries", str(args.eval_timeout_retries),
            "--empty-result-retries", str(args.eval_empty_result_retries),
            "--model-mismatch-retries", str(args.eval_model_mismatch_retries),
            "--concurrency", str(max(1, args.eval_concurrency)),
        ],
        args.prefix,
        log_name=f"eval_copilot-{args.copilot_model}.log",
    )

    run_eval(
        Path("eval_copilot.py"),
        [
            *common_eval_args,
            "--model", args.gemini_model,
            "--timeout", str(args.eval_timeout),
            "--timeout-retries", str(args.eval_timeout_retries),
            "--empty-result-retries", str(args.eval_empty_result_retries),
            "--model-mismatch-retries", str(args.eval_model_mismatch_retries),
            "--concurrency", str(max(1, args.eval_concurrency)),
        ],
        args.prefix,
        log_name=f"eval_copilot-{args.gemini_model}.log",
    )

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Total running time: {elapsed_seconds:.2f}s")


if __name__ == "__main__":
    main()
