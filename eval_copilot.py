import argparse
import asyncio
import json
from pathlib import Path

from copilot import CopilotClient  # pyright: ignore[reportMissingImports]

from common import (
    DEFAULT_COPILOT_MODEL,
    add_chain_steps_cli_argument,
    add_common_runtime_cli_arguments,
    add_model_mismatch_retries_cli_argument,
    format_resolved_chain_steps,
    load_prompt_template,
    resolve_required_template_path,
)
from common_eval import (
    build_aligned_edits,
    build_eval_prompt,
    get_copilot_eval_payload_with_retries,
    load_run_errors,
    write_eval_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-evaluate outputs with Copilot API evaluator model")
    parser.add_argument("--orginal-trans-file", default="sample_multi_input.txt")
    parser.add_argument("--patch-result-file", required=True)
    parser.add_argument("--prefix", default="eval")
    parser.add_argument("--prompt-file", default="prompt_eval.md")
    add_chain_steps_cli_argument(parser)
    parser.add_argument("--model", default=DEFAULT_COPILOT_MODEL)
    add_common_runtime_cli_arguments(parser, timeout_default=120.0)
    add_model_mismatch_retries_cli_argument(parser)
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()

    input_file = Path(args.orginal_trans_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    prompt_file, prompt_error = resolve_required_template_path(
        args.prompt_file,
        "prompt_eval.md",
        "Evaluation prompt",
    )
    if prompt_error:
        raise FileNotFoundError(prompt_error)

    timeout_seconds = float(args.timeout)
    timeout_retries = max(0, int(args.timeout_retries))
    empty_result_retries = max(0, int(args.empty_result_retries))
    model_mismatch_retries = max(0, int(args.model_mismatch_retries))
    concurrency = max(1, int(args.concurrency))
    chain_steps_text = format_resolved_chain_steps(args.chain_steps)

    src_lines = input_file.read_text(encoding="utf-8").splitlines()
    eval_template = load_prompt_template(prompt_file)
    patch_result_files = [value.strip() for value in str(args.patch_result_file).split(",") if value.strip()]
    if not patch_result_files:
        raise ValueError("Missing patch result file(s). Provide --patch-result-file.")

    prefix = Path(str(args.prefix)).stem or "eval"

    run_errors_path = Path(f"{prefix}_results") / f"{prefix}_run_errors.json"
    error_descriptions = load_run_errors(str(run_errors_path))

    print(f"Using model: {args.model}")
    print(f"Using eval prompt file: {prompt_file}")
    print(f"Concurrency: {concurrency}")
    print(f"Timeout: {timeout_seconds}s, retries: {timeout_retries}")
    print(f"Empty-result retries: {empty_result_retries}")
    print(f"Model-mismatch retries: {model_mismatch_retries}")
    print(f"Active eval chain steps: {chain_steps_text}")

    report: list[dict] = []

    client = CopilotClient()
    await client.start()
    try:
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
                        "evaluator": args.model,
                        "evaluator_model": args.model,
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
                    payload = await get_copilot_eval_payload_with_retries(
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
                    "evaluator": args.model,
                    "evaluator_model": args.model,
                    "result_file": file_name,
                    "missing": False,
                    "line_results": line_results,
                }
            )
    finally:
        await client.stop()

    results_path, scores_path, summary_path = write_eval_outputs(
        prefix,
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
