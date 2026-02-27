import asyncio
import json
import random
import re

from common import (
    build_repair_prompt_after_invalid_json,
    extract_text_content,
    handle_invalid_repair_json_result,
    is_non_repairable_validation_error,
    log_json_validation_with_key_error,
    non_repairable_prefix,
    parse_validate_and_apply_text_fixes,
    print_timeout_and_retry_guidance,
    resolve_payload_or_retry_on_empty_corrected_text,
    run_with_timeout_retry,
    should_retry_after_failure,
)


async def aoai_send_once(
    client: object,
    deployment: str,
    prompt: str,
    schema: dict,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
) -> str:
    request_kwargs = {
        "model": deployment,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": schema,
        },
        "temperature": temperature,
        "top_p": top_p,
    }

    completion = await asyncio.wait_for(
        asyncio.to_thread(client.chat.completions.create, **request_kwargs),
        timeout=timeout_seconds,
    )

    choices = getattr(completion, "choices", None)
    if not choices:
        return ""

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    return extract_text_content(content).strip()


def aoai_extract_retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        retry_after_header = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after_header is not None:
            try:
                retry_after_seconds = float(str(retry_after_header).strip())
                if retry_after_seconds > 0:
                    return retry_after_seconds
            except ValueError:
                pass

    message = str(error)
    match = re.search(r"retry after\s+(\d+(?:\.\d+)?)\s+second", message, flags=re.IGNORECASE)
    if not match:
        return None

    try:
        retry_after_seconds = float(match.group(1))
        return retry_after_seconds if retry_after_seconds > 0 else None
    except ValueError:
        return None


def is_aoai_rate_limited_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    message = str(error).lower()
    return (
        "ratelimitreached" in message
        or "rate limit" in message
        or "error code: 429" in message
    )


async def aoai_send_with_timeout_retry(
    client: object,
    deployment: str,
    prompt: str,
    schema: dict,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
    timeout_retries: int,
    processing_id: str | None = None,
) -> str | None:
    async def operation() -> str:
        return await aoai_send_once(
            client,
            deployment,
            prompt,
            schema,
            temperature,
            top_p,
            timeout_seconds,
        )

    return await run_with_timeout_retry(
        operation,
        timeout_retries,
        processing_id,
        is_retryable_error=is_aoai_rate_limited_error,
        resolve_backoff_seconds=lambda error, _attempt: aoai_extract_retry_after_seconds(error),
    )


async def get_patch_payload_with_repair(
    client: object,
    deployment: str,
    prompt: str,
    transcription: str,
    processing_id: str,
    repair_prompt_template: str,
    patch_schema: dict,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    retry_temperature_jitter: float,
    retry_top_p_jitter: float,
) -> dict | None:
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

        content = await aoai_send_with_timeout_retry(
            client,
            deployment,
            prompt,
            patch_schema,
            attempt_temperature,
            attempt_top_p,
            timeout_seconds,
            timeout_retries,
            processing_id,
        )

        if content is None:
            print_timeout_and_retry_guidance(timeout_seconds, timeout_retries, processing_id)
            return None

        payload, validation_error, content = parse_validate_and_apply_text_fixes(
            content,
            transcription,
            processing_id,
        )

        if payload is None:
            log_json_validation_with_key_error(validation_error, content, processing_id)

            if is_non_repairable_validation_error(validation_error):
                reason = validation_error.removeprefix(non_repairable_prefix()).strip() if isinstance(validation_error, str) else "Model output is not repairable."
                print(f"[{processing_id}] Skipping repair: {reason}")
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    "Model output is not repairable.",
                    processing_id=processing_id,
                )
                if not should_retry:
                    return None
                continue

            repair_prompt = build_repair_prompt_after_invalid_json(
                repair_prompt_template,
                validation_error,
                content,
                target_schema=json.dumps(patch_schema, ensure_ascii=False, indent=2),
                processing_id=processing_id,
            )

            repaired_content = await aoai_send_with_timeout_retry(
                client,
                deployment,
                repair_prompt,
                patch_schema,
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
                if not should_retry:
                    return None
                continue

            payload, validation_error, repaired_content = parse_validate_and_apply_text_fixes(
                repaired_content,
                transcription,
                processing_id,
            )
            if payload is None:
                should_retry = handle_invalid_repair_json_result(
                    empty_attempt,
                    empty_result_retries,
                    validation_error,
                    repaired_content,
                    processing_id,
                )
                if not should_retry:
                    return None
                continue

        result_payload, should_retry = resolve_payload_or_retry_on_empty_corrected_text(
            payload,
            transcription,
            empty_attempt,
            empty_result_retries,
            processing_id,
        )
        if not should_retry:
            return result_payload

    return None
