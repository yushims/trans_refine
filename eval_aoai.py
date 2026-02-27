import argparse
import asyncio
import os
import json
import random
from pathlib import Path

from openai import AzureOpenAI

from common_eval import (
    EVAL_CHECK_FIELDS,
    build_failed_eval_payload,
    get_aoai_eval_payload_with_repair as get_payload_with_repair_common,
    load_run_errors,
    validate_eval_payload,
    write_eval_outputs,
)
from common import load_prompt_template

EVAL_SCHEMA = {
    "name": "patch_eval_output",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pass": {"type": "boolean"},
            "fail_reasons": {
                "type": "array",
                "items": {"type": "string"},
            },
            "checks": {
                "type": "object",
                "additionalProperties": False,
                "properties": {field: {"type": "boolean"} for field in EVAL_CHECK_FIELDS},
                "required": EVAL_CHECK_FIELDS,
            },
            "diff_summary": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["pass", "fail_reasons", "checks", "diff_summary"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orginal-trans-file", dest="orginal_trans_file", default="sample_multi_input.txt")
    parser.add_argument("--patch-result-file", dest="patch_result_file", required=True)
    parser.add_argument("--output-file", dest="output_file")
    parser.add_argument("--prompt-file", dest="prompt_file", default="prompt_eval.txt")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file", default="prompt_repair.txt")
    parser.add_argument("--deployment", dest="deployment", default="gpt-5-chat")
    parser.add_argument("--endpoint", dest="endpoint", default="https://adaptationdev-resource.openai.azure.com/")
    parser.add_argument("--api-version", dest="api_version", default="2025-01-01-preview")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--concurrency", dest="concurrency", type=int, default=10)
    parser.add_argument("--timeout", dest="timeout", type=float, default=600.0)
    parser.add_argument("--timeout-retries", dest="timeout_retries", type=int, default=2)
    parser.add_argument("--empty-result-retries", dest="empty_result_retries", type=int, default=2)
    parser.add_argument("--temperature", dest="temperature", type=float, default=0.0)
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--retry-temperature-jitter", dest="retry_temperature_jitter", type=float, default=0.08)
    parser.add_argument("--retry-top-p-jitter", dest="retry_top_p_jitter", type=float, default=0.03)
    return parser.parse_args()


async def get_payload_with_repair(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    processing_id: str,
    repair_prompt_template: str,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    retry_temperature_jitter: float,
    retry_top_p_jitter: float,
) -> dict:
    return await get_payload_with_repair_common(
        client=client,
        deployment=deployment,
        prompt=prompt,
        processing_id=processing_id,
        repair_prompt_template=repair_prompt_template,
        schema=EVAL_SCHEMA,
        timeout_seconds=timeout_seconds,
        timeout_retries=timeout_retries,
        empty_result_retries=empty_result_retries,
        temperature=temperature,
        top_p=top_p,
        retry_temperature_jitter=retry_temperature_jitter,
        retry_top_p_jitter=retry_top_p_jitter,
        validate_payload=validate_eval_payload,
        build_failed_payload=build_failed_eval_payload,
    )


async def main() -> None:
    args = parse_args()

    input_file_value = args.orginal_trans_file
    input_file = Path(input_file_value)
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    prompt_file_value = args.prompt_file
    prompt_file = Path(prompt_file_value)
    if not prompt_file.exists():
        print(f"Evaluation prompt file not found: {prompt_file}")
        return

    repair_prompt_file_value = args.repair_prompt_file
    repair_prompt_file = Path(repair_prompt_file_value)
    if not repair_prompt_file.exists():
        print(f"Repair prompt file not found: {repair_prompt_file}")
        return

    endpoint = args.endpoint
    deployment = args.deployment
    api_version = args.api_version
    configured_concurrency = args.concurrency
    concurrency = max(1, configured_concurrency)
    timeout_seconds = args.timeout
    timeout_retries = max(0, args.timeout_retries)
    empty_result_retries = max(0, args.empty_result_retries)
    temperature = max(0.0, min(1.0, float(args.temperature)))
    top_p = max(0.0, min(1.0, float(args.top_p)))
    retry_temperature_jitter = max(0.0, float(args.retry_temperature_jitter))
    retry_top_p_jitter = max(0.0, float(args.retry_top_p_jitter))

    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("--api-key is required (or set AZURE_OPENAI_API_KEY).")
        return

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
    )

    print(f"Using deployment: {deployment}")
    print(f"Using endpoint: {endpoint}")
    print(f"API version: {api_version}")
    print(f"Temperature: {temperature}")
    print(f"Top p: {top_p}")
    print(f"Retry jitter: temperature<=+{retry_temperature_jitter}, top_p±{retry_top_p_jitter}")
    print(f"Using eval prompt file: {prompt_file}")
    print(f"Using repair prompt file: {repair_prompt_file}")
    print(f"Concurrency: {concurrency}")
    print(f"Timeout: {timeout_seconds}s, retries: {timeout_retries}")
    print(f"Empty-result retries: {empty_result_retries}")

    src_lines = input_file.read_text(encoding="utf-8").splitlines()
    eval_template = load_prompt_template(prompt_file)
    repair_prompt_template = load_prompt_template(repair_prompt_file)
    patch_result_files = [value.strip() for value in str(args.patch_result_file).split(",") if value.strip()]
    if not patch_result_files:
        print("Missing patch result file(s). Provide --patch-result-file.")
        return

    first_name = Path(patch_result_files[0]).name
    prefix = "eval"
    for marker in ("_aoai_run", "_copilot_run", "_gemini_run"):
        if marker in first_name:
            prefix = first_name.split(marker, 1)[0]
            break
    if prefix == "eval" and first_name.startswith("run"):
        for model in ("aoai", "copilot", "gemini"):
            model_marker = f"_{model}"
            if model_marker in first_name:
                left = first_name.split(model_marker, 1)[0]
                if "_" in left:
                    prefix = left.split("_", 1)[1]
                break

    run_errors_path = Path(f"{prefix}_results") / f"{prefix}_run_errors.json"
    error_descriptions = load_run_errors(str(run_errors_path))
    files = patch_result_files
    evaluator_name = deployment
    output_prefix = Path(str(args.output_file)).stem if args.output_file else prefix
    if not output_prefix:
        output_prefix = prefix

    report: list[dict] = []
    for file_name in files:
        path = Path(file_name)
        if not path.exists():
            reason = "Missing output file"
            json_error = error_descriptions.get(file_name, "") or error_descriptions.get(Path(file_name).name, "")
            if json_error:
                reason = f"Missing output file. Last JSON format error: {json_error}"
            report.append(
                {
                    "evaluator": evaluator_name,
                    "evaluator_model": deployment,
                    "file": file_name,
                    "missing": True,
                    "line_results": [
                        {
                            "line": None,
                            "pass": False,
                            "fail_reasons": [reason],
                            "checks": {field: False for field in EVAL_CHECK_FIELDS},
                            "diff_summary": [],
                        }
                    ],
                }
            )
            continue

        out_lines = path.read_text(encoding="utf-8").splitlines()
        line_results: list[dict] = []
        max_lines = max(len(src_lines), len(out_lines))
        semaphore = asyncio.Semaphore(concurrency)
        results_by_line: dict[int, dict] = {}

        async def evaluate_line(idx: int) -> None:
            src = src_lines[idx] if idx < len(src_lines) else ""
            out = out_lines[idx] if idx < len(out_lines) else ""
            line_no = idx + 1

            if not src.strip() and not out.strip():
                return

            prompt = (
                eval_template
                .replace("{original_transcript}", src)
                .replace("{patch_transcript}", out)
                .replace("{patch_json}", out)
            )

            if max_lines > 1:
                print(f"Processing line {line_no}/{max_lines}...")

            async with semaphore:
                processing_id = f"{line_no}/{max_lines}"
                payload = await get_payload_with_repair(
                    client,
                    deployment,
                    prompt,
                    processing_id,
                    repair_prompt_template,
                    temperature,
                    top_p,
                    timeout_seconds,
                    timeout_retries,
                    empty_result_retries,
                    retry_temperature_jitter,
                    retry_top_p_jitter,
                )

            results_by_line[line_no] = {
                "line": line_no,
                "pass": bool(payload.get("pass")),
                "fail_reasons": payload.get("fail_reasons", []),
                "checks": payload.get("checks", {field: False for field in EVAL_CHECK_FIELDS}),
                "diff_summary": payload.get("diff_summary", []),
                "source_excerpt": src[:160],
                "output_excerpt": out[:160],
            }

        await asyncio.gather(*(evaluate_line(idx) for idx in range(max_lines)))
        line_results = [results_by_line[line_no] for line_no in sorted(results_by_line)]

        report.append(
            {
                "evaluator": evaluator_name,
                "evaluator_model": deployment,
                "file": file_name,
                "missing": False,
                "line_results": line_results,
            }
        )

    results_path, scores_path, summary_path = write_eval_outputs(
        output_prefix,
        "aoai",
        deployment,
        report,
    )
    print(f"Wrote {results_path}")
    print(f"Wrote {scores_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
