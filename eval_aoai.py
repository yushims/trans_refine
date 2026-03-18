import argparse
import asyncio
import json
import random
from pathlib import Path

from openai import AzureOpenAI

from common_eval import (
    build_aligned_edits,
    build_eval_prompt,
    build_failed_eval_payload,
    get_aoai_eval_payload_with_repair,
    load_run_errors,
    validate_eval_payload,
    write_eval_outputs,
)
from common import (
    _CHAIN_ID_TO_NAME,
    _resolve_active_chain_ids,
    DEFAULT_AOAI_API_VERSION,
    DEFAULT_AOAI_DEPLOYMENT,
    DEFAULT_AOAI_ENDPOINT,
    add_aoai_sampling_cli_arguments,
    add_chain_steps_cli_argument,
    add_common_runtime_cli_arguments,
    format_resolved_chain_steps,
    load_prompt_template,
    resolve_required_template_path,
)


def build_eval_schema(chain_steps: list[str] | None) -> dict:
    active_chain_ids = _resolve_active_chain_ids(chain_steps)
    active_step_names = [_CHAIN_ID_TO_NAME.get(chain_id, "") for chain_id in active_chain_ids]
    active_step_enum = [name for name in active_step_names if name]
    valid_step_enum = list(active_step_enum)
    valid_step_enum.append(None)

    return {
        "name": "patch_eval_output",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "edit": {"type": "string"},
                            "valid_step": {
                                "type": ["string", "null"],
                                "enum": valid_step_enum,
                            },
                            "rule": {"type": "string"},
                            "note": {"type": "string"},
                        },
                        "required": ["edit", "valid_step", "rule", "note"],
                    },
                },
                "missing_edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "span": {"type": "string"},
                            "expected_edit": {"type": "string"},
                            "step": {
                                "type": "string",
                                "enum": active_step_enum,
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["span", "expected_edit", "step", "reason"],
                    },
                },
            },
            "required": [
                "edits",
                "missing_edits",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orginal-trans-file", dest="orginal_trans_file", default="sample_multi_input.txt")
    parser.add_argument("--patch-result-file", dest="patch_result_file", required=True)
    parser.add_argument("--prefix", dest="prefix", default="eval")
    parser.add_argument("--prompt-file", dest="prompt_file", default="prompt_eval.md")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file", default="prompt_repair.md")
    add_chain_steps_cli_argument(parser)
    parser.add_argument("--deployment", dest="deployment", default=DEFAULT_AOAI_DEPLOYMENT)
    parser.add_argument("--endpoint", dest="endpoint", default=DEFAULT_AOAI_ENDPOINT)
    parser.add_argument("--api-version", dest="api_version", default=DEFAULT_AOAI_API_VERSION)
    add_common_runtime_cli_arguments(parser)
    add_aoai_sampling_cli_arguments(parser)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    input_file_value = args.orginal_trans_file
    input_file = Path(input_file_value)
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    prompt_file, prompt_error = resolve_required_template_path(
        args.prompt_file,
        "prompt_eval.md",
        "Evaluation prompt",
    )
    if prompt_error:
        print(prompt_error)
        return

    repair_prompt_file, repair_prompt_error = resolve_required_template_path(
        args.repair_prompt_file,
        "prompt_repair.md",
        "Repair prompt",
    )
    if repair_prompt_error:
        print(repair_prompt_error)
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
    chain_steps_text = format_resolved_chain_steps(args.chain_steps)
    eval_schema = build_eval_schema(args.chain_steps)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
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
    print(f"Active eval chain steps: {chain_steps_text}")

    src_lines = input_file.read_text(encoding="utf-8").splitlines()
    eval_template = load_prompt_template(prompt_file)
    repair_prompt_template = load_prompt_template(repair_prompt_file)
    patch_result_files = [value.strip() for value in str(args.patch_result_file).split(",") if value.strip()]
    if not patch_result_files:
        print("Missing patch result file(s). Provide --patch-result-file.")
        return

    prefix = Path(str(args.prefix)).stem or "eval"

    run_errors_path = Path(f"{prefix}_results") / f"{prefix}_run_errors.json"
    error_descriptions = load_run_errors(str(run_errors_path))

    report: list[dict] = []
    for file_name in patch_result_files:
        print(f"Evaluating patch result file: {file_name}")
        path = Path(file_name)
        if not path.exists():
            reason = "Missing output file"
            json_error = error_descriptions.get(file_name, "") or error_descriptions.get(Path(file_name).name, "")
            if json_error:
                reason = f"Missing output file. Last JSON format error: {json_error}"
            report.append(
                {
                    "evaluator": deployment,
                    "evaluator_model": deployment,
                    "result_file": file_name,
                    "missing": True,
                    "line_results": [
                        {
                            "line": None,
                            "pass": False,
                            "fail_reasons": [reason],
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

            extracted_edits_text = build_aligned_edits(src, out)

            prompt = build_eval_prompt(
                eval_template,
                src,
                out,
                chain_steps_text,
                extracted_edits_text,
            )

            if max_lines > 1:
                print(f"Evaluating line {line_no}/{max_lines}...")

            async with semaphore:
                processing_id = f"eval-{file_name}-L{line_no}"
                payload = await get_aoai_eval_payload_with_repair(
                    client=client,
                    deployment=deployment,
                    prompt=prompt,
                    processing_id=processing_id,
                    repair_prompt_template=repair_prompt_template,
                    schema=eval_schema,
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

            results_by_line[line_no] = {
                "line": line_no,
                "pass": bool(payload.get("pass")),
                "edit_error_rate": float(payload.get("edit_error_rate", 1.0)),
                "fail_reasons": payload.get("fail_reasons", []),
                "diff_summary": payload.get("diff_summary", []),
                "edits": payload.get("edits", []),
                "missing_edits": payload.get("missing_edits", []),
                "source_excerpt": src,
                "output_excerpt": out,
            }

        await asyncio.gather(*(evaluate_line(idx) for idx in range(max_lines)))
        line_results = [results_by_line[line_no] for line_no in sorted(results_by_line)]

        report.append(
            {
                "evaluator": deployment,
                "evaluator_model": deployment,
                "result_file": file_name,
                "missing": False,
                "line_results": line_results,
            }
        )

    results_path, scores_path, summary_path = write_eval_outputs(
        prefix,
        "aoai",
        deployment,
        report,
    )
    print(f"Wrote {results_path}")
    print(f"Wrote {scores_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
