import argparse
import asyncio
import random
import re

from openai import AzureOpenAI
from pipeline_common import (
    assign_payload_or_emit_empty,
    build_empty_payload,
    build_repair_prompt_after_invalid_json,
    collect_transcriptions_from_input,
    extract_text_content,
    finalize_payloads_and_write,
    handle_invalid_repair_json_result,
    is_non_repairable_validation_error,
    load_patch_and_repair_templates,
    log_json_validation_with_key_error,
    non_repairable_prefix,
    parse_validate_and_apply_text_fixes,
    print_common_runtime_settings,
    print_timeout_and_retry_guidance,
    resolve_payload_or_retry_on_empty_corrected_text,
    resolve_patch_and_repair_template_paths,
    run_transcriptions_with_concurrency,
    run_with_timeout_retry,
    should_retry_after_failure,
)


PATCH_SCHEMA = {
    "name": "deterministic_patch_output",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tokenization": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tokens": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["tokens"]
            },
            "ct_combine": {"type": "string"},
            "ct_fix": {"type": "string"},
            "ct_punct": {"type": "string"},
            "ct_casing": {"type": "string"},
            "verification": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op_details": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["op_details"]
            },
            "machine_transcription_probability": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            }
        },
        "required": [
            "tokenization",
            "ct_combine",
            "ct_fix",
            "ct_punct",
            "ct_casing",
            "verification",
            "machine_transcription_probability"
        ]
    }
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", dest="input_file")
    parser.add_argument("--output-file", dest="output_file")
    parser.add_argument("--patch-prompt-file", dest="patch_prompt_file")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file")
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


async def send_once(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
) -> str:
    request_kwargs = {
        "model": deployment,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": PATCH_SCHEMA
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


async def send_with_timeout_retry(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
    timeout_retries: int,
    processing_id: str | None = None,
) -> str | None:
    def _extract_retry_after_seconds(error: Exception) -> float | None:
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

    def _is_rate_limited_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return True

        message = str(error).lower()
        return (
            "ratelimitreached" in message
            or "rate limit" in message
            or "error code: 429" in message
        )

    async def operation() -> str:
        return await send_once(
            client,
            deployment,
            prompt,
            temperature,
            top_p,
            timeout_seconds,
        )

    return await run_with_timeout_retry(
        operation,
        timeout_retries,
        processing_id,
        is_retryable_error=_is_rate_limited_error,
        resolve_backoff_seconds=lambda error, _attempt: _extract_retry_after_seconds(error),
    )


async def get_payload_with_repair(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    transcription: str,
    processing_id: str,
    repair_prompt_template: str,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    retry_temperature_jitter: float,
    retry_top_p_jitter: float,
):
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

        content = await send_with_timeout_retry(
            client,
            deployment,
            prompt,
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
                processing_id,
            )

            repaired_content = await send_with_timeout_retry(
                client,
                deployment,
                repair_prompt,
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


async def main():
    args = parse_args()

    input_file_value = args.input_file
    output_file_value = args.output_file

    transcriptions: list[str]
    loaded_transcriptions = collect_transcriptions_from_input(input_file_value)
    if loaded_transcriptions is None:
        return
    transcriptions = loaded_transcriptions

    endpoint = args.endpoint
    deployment = args.deployment
    api_version = args.api_version
    api_key = args.api_key
    if not api_key:
        print("--api-key is required.")
        return

    configured_concurrency = args.concurrency
    concurrency = max(1, configured_concurrency)

    timeout_seconds = args.timeout
    timeout_retries = max(0, args.timeout_retries)
    empty_result_retries = max(0, args.empty_result_retries)

    temperature = args.temperature
    top_p = args.top_p
    retry_temperature_jitter = max(0.0, args.retry_temperature_jitter)
    retry_top_p_jitter = max(0.0, args.retry_top_p_jitter)

    prompt_template_path, repair_prompt_template_path, template_error = resolve_patch_and_repair_template_paths(
        args.patch_prompt_file,
        args.repair_prompt_file,
    )
    if template_error:
        print(template_error)
        return
    if prompt_template_path is None or repair_prompt_template_path is None:
        return

    print(f"Using deployment: {deployment}")
    print(f"Using endpoint: {endpoint}")
    print(f"API version: {api_version}")
    print(f"Temperature: {temperature}")
    print(f"Top p: {top_p}")
    print(f"Retry jitter: temperature<=+{retry_temperature_jitter}, top_p±{retry_top_p_jitter}")
    print_common_runtime_settings(
        prompt_template_path,
        repair_prompt_template_path,
        concurrency,
        timeout_seconds,
        timeout_retries,
        empty_result_retries,
    )
    prompt_template, repair_prompt_template = load_patch_and_repair_templates(
        prompt_template_path,
        repair_prompt_template_path,
    )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    payloads: list[dict | None] = [None] * len(transcriptions)

    async def process_item(index: int, transcription: str, total: int) -> None:
        slot = index - 1
        processing_id = f"{index}/{total}"
        if not transcription.strip():
            payloads[slot] = build_empty_payload()
            print(
                f"Input transcription {index}/{total} is empty; "
                "emitting empty payload."
            )
            return

        prompt = prompt_template + transcription
        try:
            payload = await get_payload_with_repair(
                client,
                deployment,
                prompt,
                transcription,
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

            assign_payload_or_emit_empty(payload, payloads, slot, index, total)
        except asyncio.CancelledError:
            payloads[slot] = build_empty_payload()
            print(
                f"Cancelled while processing transcription {index}/{total}; "
                "emitting empty payload."
            )
            return
        except Exception as error:
            payloads[slot] = build_empty_payload()
            print(
                f"Unexpected error on transcription {index}/{total}: {error}; "
                "emitting empty payload."
            )
            return

    await run_transcriptions_with_concurrency(transcriptions, concurrency, process_item)

    if not finalize_payloads_and_write(payloads, output_file_value):
        return


if __name__ == "__main__":
    asyncio.run(main())
