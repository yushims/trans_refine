import asyncio
import json
import random

from common import (
    extract_retry_after_seconds,
    extract_text_content,
    get_patch_payload_with_repair_generic,
    run_with_timeout_retry,
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
        resolve_backoff_seconds=lambda error, _attempt: extract_retry_after_seconds(error),
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
    skip_first_token_casing_preservation: bool = False,
) -> dict | None:
    attempt_temperatures: list[float] = []
    attempt_top_ps: list[float] = []

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
            # print(
            #     f"[{processing_id}] Retry decode jitter applied: temperature={attempt_temperature:.3f}, "
            #     f"top_p={attempt_top_p:.3f}"
            # )
        attempt_temperatures.append(attempt_temperature)
        attempt_top_ps.append(attempt_top_p)

    async def _send_prompt(attempt_prompt: str, empty_attempt: int) -> str | None:
        return await aoai_send_with_timeout_retry(
            client,
            deployment,
            attempt_prompt,
            patch_schema,
            attempt_temperatures[empty_attempt],
            attempt_top_ps[empty_attempt],
            timeout_seconds,
            timeout_retries,
            processing_id,
        )

    async def _send_repair_prompt(repair_prompt: str, empty_attempt: int) -> str | None:
        return await aoai_send_with_timeout_retry(
            client,
            deployment,
            repair_prompt,
            patch_schema,
            attempt_temperatures[empty_attempt],
            attempt_top_ps[empty_attempt],
            timeout_seconds,
            timeout_retries,
            processing_id,
        )

    return await get_patch_payload_with_repair_generic(
        prompt=prompt,
        transcription=transcription,
        processing_id=processing_id,
        repair_prompt_template=repair_prompt_template,
        target_schema=json.dumps(patch_schema, ensure_ascii=False, indent=2),
        timeout_seconds=timeout_seconds,
        timeout_retries=timeout_retries,
        empty_result_retries=empty_result_retries,
        send_prompt=_send_prompt,
        send_repair_prompt=_send_repair_prompt,
        skip_first_token_casing_preservation=skip_first_token_casing_preservation,
        repair_timeout_message="Repair returned empty output or timed out.",
        repair_empty_message="Repair returned empty output or timed out.",
    )
