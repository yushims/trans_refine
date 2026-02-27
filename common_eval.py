import csv
import json
import random
import re
from pathlib import Path
from collections.abc import Callable
from typing import Any

from common_aoai import aoai_send_with_timeout_retry
from common_copilot import (
    ModelMismatchError,
    build_copilot_session_parameters,
    handle_copilot_model_mismatch_retry,
    send_copilot_once,
)
from common import (
    build_repair_prompt_after_invalid_json,
    run_with_timeout_retry,
    should_retry_after_failure,
)


EVAL_CHECK_FIELDS = [
    "forbidden_ok",
    "line_end_ok",
    "first_token_casing_ok",
    "punct_adjacency_ok",
    "span_ok",
    "compact_spacing_ok",
    "consistency_ok",
    "minimality_ok",
    "asr_pronunciation_ok",
    "numeral_ok",
]

EVAL_HARD_CHECK_FIELDS = [
    "forbidden_ok",
    "line_end_ok",
    "first_token_casing_ok",
    "punct_adjacency_ok",
    "span_ok",
    "compact_spacing_ok",
    "consistency_ok",
]


async def send_copilot_prompt_once(
    client: Any,
    model_name: str,
    prompt: str,
    timeout_seconds: float,
) -> str:
    session = await client.create_session(build_copilot_session_parameters(model_name))
    return await send_copilot_once(
        session,
        prompt,
        timeout_seconds,
        requested_model=model_name,
    )


async def get_copilot_eval_payload_with_retries(
    client: Any,
    model_name: str,
    prompt: str,
    processing_id: str,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    model_mismatch_retries: int,
) -> dict:
    for mismatch_attempt in range(model_mismatch_retries + 1):
        for empty_attempt in range(empty_result_retries + 1):
            async def operation() -> dict:
                content = await send_copilot_prompt_once(client, model_name, prompt, timeout_seconds)
                if not content:
                    raise ValueError("evaluation model returned empty content")

                payload = json.loads(content)
                is_valid, validation_error = validate_eval_payload(payload)
                if not is_valid:
                    raise ValueError(f"invalid evaluator payload: {validation_error}")
                return payload

            try:
                result = await run_with_timeout_retry(
                    operation,
                    timeout_retries,
                    processing_id=processing_id,
                )

                if result is None:
                    should_retry = should_retry_after_failure(
                        empty_attempt,
                        empty_result_retries,
                        "Model call timed out after retries.",
                        processing_id=processing_id,
                    )
                    if should_retry:
                        continue
                    return build_failed_eval_payload("Prompt-eval failure: timeout after retries")

                return result

            except ModelMismatchError as mismatch_error:
                should_retry = await handle_copilot_model_mismatch_retry(
                    mismatch_error,
                    mismatch_attempt,
                    model_mismatch_retries,
                    processing_id,
                    "emitting failed eval payload",
                )
                if should_retry:
                    break

                return build_failed_eval_payload(
                    f"Prompt-eval failure: model mismatch requested={mismatch_error.requested_model} actual={mismatch_error.actual_model}"
                )

            except Exception as error:
                message = str(error).strip()
                lowered = message.lower()
                is_empty = "empty content" in lowered

                if is_empty:
                    should_retry = should_retry_after_failure(
                        empty_attempt,
                        empty_result_retries,
                        "Model returned empty content after retries.",
                        processing_id=processing_id,
                    )
                    if should_retry:
                        continue

                return build_failed_eval_payload(f"Prompt-eval failure: {error}")

    return build_failed_eval_payload("Prompt-eval failure: exhausted retries")


async def get_copilot_eval_payload_with_repair_on_session(
    client: Any,
    model_name: str,
    prompt: str,
    processing_id: str,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
) -> dict:
    return await get_copilot_eval_payload_with_retries(
        client=client,
        model_name=model_name,
        prompt=prompt,
        processing_id=processing_id,
        timeout_seconds=timeout_seconds,
        timeout_retries=timeout_retries,
        empty_result_retries=empty_result_retries,
        model_mismatch_retries=0,
    )


async def get_copilot_eval_payload_with_repair(
    client: Any,
    model_name: str,
    prompt: str,
    processing_id: str,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    model_mismatch_retries: int,
) -> dict:
    return await get_copilot_eval_payload_with_retries(
        client=client,
        model_name=model_name,
        prompt=prompt,
        processing_id=processing_id,
        timeout_seconds=timeout_seconds,
        timeout_retries=timeout_retries,
        empty_result_retries=empty_result_retries,
        model_mismatch_retries=model_mismatch_retries,
    )


async def get_aoai_eval_payload_with_repair(
    client: object,
    deployment: str,
    prompt: str,
    processing_id: str,
    repair_prompt_template: str,
    schema: dict,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    temperature: float,
    top_p: float,
    retry_temperature_jitter: float,
    retry_top_p_jitter: float,
    validate_payload: Callable[[dict], tuple[bool, str]],
    build_failed_payload: Callable[[str], dict],
) -> dict:
    for empty_attempt in range(empty_result_retries + 1):
        attempt_temperature = temperature
        attempt_top_p = top_p
        if empty_attempt > 0:
            attempt_temperature = max(
                0.0,
                min(1.0, temperature + random.uniform(0.0, retry_temperature_jitter)),
            )
            attempt_top_p = max(
                0.0,
                min(1.0, top_p + random.uniform(-retry_top_p_jitter, retry_top_p_jitter)),
            )
            print(
                f"[{processing_id}] Retry decode jitter applied: temperature={attempt_temperature:.3f}, "
                f"top_p={attempt_top_p:.3f}"
            )

        try:
            content = await aoai_send_with_timeout_retry(
                client,
                deployment,
                prompt,
                schema,
                attempt_temperature,
                attempt_top_p,
                timeout_seconds,
                timeout_retries,
                processing_id,
            )

            if content is None:
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    "Model call timed out after retries.",
                    processing_id=processing_id,
                )
                if should_retry:
                    continue
                return build_failed_payload("Prompt-eval failure: timeout after retries")

            if not content:
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    "Model returned empty content after retries.",
                    processing_id=processing_id,
                )
                if should_retry:
                    continue
                return build_failed_payload("Prompt-eval failure: empty content after retries")

            try:
                payload = json.loads(content)
                is_valid, validation_error = validate_payload(payload)
                if not is_valid:
                    raise ValueError(f"invalid evaluator payload: {validation_error}")
                return payload
            except Exception as parse_error:
                repair_prompt = build_repair_prompt_after_invalid_json(
                    repair_prompt_template,
                    str(parse_error),
                    content,
                    target_schema=json.dumps(schema, ensure_ascii=False, indent=2),
                    processing_id=processing_id,
                )
                repaired_content = await aoai_send_with_timeout_retry(
                    client,
                    deployment,
                    repair_prompt,
                    schema,
                    attempt_temperature,
                    attempt_top_p,
                    timeout_seconds,
                    timeout_retries,
                    processing_id,
                )

                if not repaired_content:
                    should_retry = should_retry_after_failure(
                        empty_attempt,
                        empty_result_retries,
                        "Repair returned empty output or timed out.",
                        processing_id=processing_id,
                    )
                    if should_retry:
                        continue
                    return build_failed_payload("Prompt-eval failure: repair returned empty output or timed out")

                try:
                    payload = json.loads(repaired_content)
                    is_valid, validation_error = validate_payload(payload)
                    if not is_valid:
                        raise ValueError(f"invalid evaluator payload: {validation_error}")
                    return payload
                except Exception as repaired_parse_error:
                    should_retry = should_retry_after_failure(
                        empty_attempt,
                        empty_result_retries,
                        f"Repair attempt did not return valid JSON ({repaired_parse_error}).",
                        processing_id=processing_id,
                    )
                    if should_retry:
                        continue
                    return build_failed_payload(f"Prompt-eval failure: {repaired_parse_error}")
        except Exception as error:
            return build_failed_payload(f"Prompt-eval failure: {error}")

    return build_failed_payload("Prompt-eval failure: exhausted retries")


def validate_eval_payload(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "evaluation output is not an object"

    if not isinstance(payload.get("pass"), bool):
        return False, "field 'pass' must be boolean"
    if not isinstance(payload.get("fail_reasons"), list):
        return False, "field 'fail_reasons' must be an array"
    if not isinstance(payload.get("diff_summary"), list):
        return False, "field 'diff_summary' must be an array"

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        return False, "field 'checks' must be an object"

    for field in EVAL_CHECK_FIELDS:
        if not isinstance(checks.get(field), bool):
            return False, f"checks.{field} must be boolean"

    hard_pass = all(bool(checks.get(field)) for field in EVAL_HARD_CHECK_FIELDS)
    if bool(payload.get("pass")) != hard_pass:
        return False, "field 'pass' is inconsistent with hard checks"

    return True, ""


def build_failed_eval_payload(reason: str) -> dict:
    return {
        "pass": False,
        "fail_reasons": [reason],
        "checks": {field: False for field in EVAL_CHECK_FIELDS},
        "diff_summary": [],
    }


def target_model_from_file(file_name: str) -> str:
    lowered = Path(file_name).name.lower()
    if "_aoai_run" in lowered or re.search(r"(^|[_\-.])aoai($|[_\-.])", lowered):
        return "aoai"
    if "_copilot_run" in lowered or re.search(r"(^|[_\-.])copilot($|[_\-.])", lowered):
        return "copilot"
    if "_gemini_run" in lowered or re.search(r"(^|[_\-.])gemini($|[_\-.])", lowered):
        return "gemini"
    return "unknown"


def load_run_errors(path_value: str | None) -> dict[str, str]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def write_eval_outputs(
    prefix: str,
    evaluator_api: str,
    evaluator_model: str,
    report: list[dict],
) -> tuple[Path, Path, Path]:
    output_dir = Path(f"{prefix}_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{prefix}_results_{evaluator_api}-{evaluator_model}.json"
    results_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    for item in report:
        line_results = item.get("line_results", [])
        total_lines = len(line_results)
        passed_lines = sum(1 for line_result in line_results if line_result.get("pass") is True)
        failed_lines = total_lines - passed_lines
        pass_rate = round((passed_lines / total_lines * 100.0), 2) if total_lines else 0.0

        rows.append(
            {
                "evaluator_api": evaluator_api,
                "evaluator_model": evaluator_model,
                "file": item.get("file", ""),
                "line_count": total_lines,
                "passed_lines": passed_lines,
                "failed_lines": failed_lines,
                "pass_rate_percent": pass_rate,
            }
        )

    scores_path = output_dir / f"{prefix}_scores_{evaluator_api}-{evaluator_model}.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["evaluator_api", "evaluator_model", "file", "line_count", "passed_lines", "failed_lines", "pass_rate_percent"],
        )
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, dict[str, float]] = {}
    for row in rows:
        file_name = str(row["file"])
        target_model = target_model_from_file(file_name)
        aggregate.setdefault(target_model, {"n": 0, "score": 0.0, "failed": 0.0, "lines": 0.0})
        aggregate[target_model]["n"] += 1
        aggregate[target_model]["score"] += float(row["pass_rate_percent"])
        aggregate[target_model]["failed"] += float(row["failed_lines"])
        aggregate[target_model]["lines"] += float(row["line_count"])

    summary_rows: list[dict[str, object]] = []
    for target_model, stats in aggregate.items():
        overall_pass = round((1.0 - (stats["failed"] / stats["lines"])) * 100.0, 2) if stats["lines"] > 0 else 0.0
        summary_rows.append(
            {
                "evaluator_api": evaluator_api,
                "evaluator_model": evaluator_model,
                "target_model": target_model,
                "runs": int(stats["n"]),
                "avg_file_pass_rate_percent": round(stats["score"] / stats["n"], 2),
                "overall_line_pass_rate_percent": overall_pass,
            }
        )

    summary_path = output_dir / f"{prefix}_summary_{evaluator_api}-{evaluator_model}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["evaluator_api", "evaluator_model", "target_model", "runs", "avg_file_pass_rate_percent", "overall_line_pass_rate_percent"],
        )
        writer.writeheader()
        writer.writerows(sorted(summary_rows, key=lambda row: row["target_model"]))

    return results_path, scores_path, summary_path
