import argparse
import asyncio
import json
from pathlib import Path

from copilot import CopilotClient  # pyright: ignore[reportMissingImports]

from common import load_prompt_template
from common_eval import (
    EVAL_CHECK_FIELDS,
    get_copilot_eval_payload_with_repair,
    load_run_errors,
    write_eval_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-evaluate outputs with Copilot API evaluator model")
    parser.add_argument("--orginal-trans-file", default="sample_multi_input.txt")
    parser.add_argument("--patch-result-file", required=True)
    parser.add_argument("--output-file")
    parser.add_argument("--prompt-file", default="prompt_eval.txt")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--timeout-retries", type=int, default=2)
    parser.add_argument("--empty-result-retries", type=int, default=2)
    parser.add_argument("--model-mismatch-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=10)
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()

    input_file = Path(args.orginal_trans_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Evaluation prompt file not found: {prompt_file}")

    timeout_seconds = float(args.timeout)
    timeout_retries = max(0, int(args.timeout_retries))
    empty_result_retries = max(0, int(args.empty_result_retries))
    model_mismatch_retries = max(0, int(args.model_mismatch_retries))
    concurrency = max(1, int(args.concurrency))

    src_lines = input_file.read_text(encoding="utf-8").splitlines()
    eval_template = load_prompt_template(prompt_file)
    patch_result_files = [value.strip() for value in str(args.patch_result_file).split(",") if value.strip()]
    if not patch_result_files:
        raise ValueError("Missing patch result file(s). Provide --patch-result-file.")

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
    evaluator_name = args.model
    output_prefix = Path(str(args.output_file)).stem if args.output_file else prefix
    if not output_prefix:
        output_prefix = prefix

    print(f"Using model: {args.model}")
    print(f"Using eval prompt file: {prompt_file}")
    print(f"Concurrency: {concurrency}")
    print(f"Timeout: {timeout_seconds}s, retries: {timeout_retries}")
    print(f"Empty-result retries: {empty_result_retries}")
    print(f"Model-mismatch retries: {model_mismatch_retries}")

    report: list[dict] = []

    client = CopilotClient()
    await client.start()
    try:
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
                        "evaluator_model": args.model,
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
                    processing_id = f"eval-{file_name}-L{line_no}"
                    payload = await get_copilot_eval_payload_with_repair(
                        client=client,
                        model_name=args.model,
                        prompt=prompt,
                        processing_id=processing_id,
                        timeout_seconds=timeout_seconds,
                        timeout_retries=timeout_retries,
                        empty_result_retries=empty_result_retries,
                        model_mismatch_retries=model_mismatch_retries,
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
                    "evaluator_model": args.model,
                    "file": file_name,
                    "missing": False,
                    "line_results": line_results,
                }
            )
    finally:
        await client.stop()

    results_path, scores_path, summary_path = write_eval_outputs(
        output_prefix,
        "copilot",
        args.model,
        report,
    )
    print(f"Wrote {results_path}")
    print(f"Wrote {scores_path}")
    print(f"Wrote {summary_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
