import argparse
import asyncio
import random
from typing import Any, Callable

from copilot import CopilotClient  # pyright: ignore[reportMissingImports]
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


STRICT_JSON_SYSTEM = """
You are a JSON formatter.
Return ONLY one strict JSON object.
No markdown, no prose, no code fences, no comments.
No extra keys.
If any field is unknown, use valid defaults instead of explanation.
""".strip()                    


class ModelMismatchError(RuntimeError):
    def __init__(self, requested_model: str, actual_model: str):
        self.requested_model = requested_model
        self.actual_model = actual_model
        super().__init__(
            f"MODEL_MISMATCH requested={requested_model} actual={actual_model}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", dest="input_file")
    parser.add_argument("--output-file", dest="output_file")
    parser.add_argument("--patch-prompt-file", dest="patch_prompt_file")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file")
    parser.add_argument("--model", dest="model", default="gpt-5.2")
    parser.add_argument("--concurrency", dest="concurrency", type=int, default=10)
    parser.add_argument("--timeout", dest="timeout", type=float, default=600.0)
    parser.add_argument("--timeout-retries", dest="timeout_retries", type=int, default=2)
    parser.add_argument("--empty-result-retries", dest="empty_result_retries", type=int, default=2)
    parser.add_argument("--model-mismatch-retries", dest="model_mismatch_retries", type=int, default=2)
    parser.add_argument("--list-models-only", dest="list_models_only", action="store_true")
    parser.add_argument("--print-models", dest="print_models", action="store_true")
    return parser.parse_args()


async def send_once(
    session: Any,
    prompt: str,
    timeout_seconds: float,
    requested_model: str | None = None,
) -> str:
    existing_events = await session.get_messages()
    seen_ids = {
        str(getattr(event, "id", ""))
        for event in existing_events
        if getattr(event, "id", None) is not None
    }

    await session.send({"prompt": prompt, "mode": "immediate"})

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    latest_content = ""
    delta_parts: list[str] = []

    while True:
        events = await session.get_messages()
        new_events = []
        for event in events:
            event_id = getattr(event, "id", None)
            if event_id is None:
                continue
            event_id_text = str(event_id)
            if event_id_text in seen_ids:
                continue
            seen_ids.add(event_id_text)
            new_events.append(event)

        for event in new_events:
            event_type = getattr(getattr(event, "type", None), "value", getattr(event, "type", ""))
            data = getattr(event, "data", None)

            if event_type == "session.error":
                error_message = (
                    getattr(data, "message", None)
                    or getattr(data, "error_reason", None)
                    or getattr(data, "error", None)
                    or "session.error"
                )
                raise OSError(str(error_message))

            if event_type == "assistant.message":
                content = extract_text_content(getattr(data, "content", None)).strip()
                if content:
                    latest_content = content

            if event_type == "assistant.message_delta":
                delta = extract_text_content(
                    getattr(data, "delta_content", None) or getattr(data, "content", None)
                )
                if delta:
                    delta_parts.append(delta)

            if event_type == "assistant.usage" and requested_model:
                actual_model = (
                    getattr(data, "model", None)
                    or getattr(data, "model_id", None)
                )
                if actual_model and str(actual_model) != requested_model:
                    raise ModelMismatchError(requested_model, str(actual_model))

            if event_type == "assistant.turn_end":
                if latest_content:
                    return latest_content
                if delta_parts:
                    combined = "".join(delta_parts).strip()
                    if combined:
                        return combined

        if latest_content:
            return latest_content

        if loop.time() >= deadline:
            raise asyncio.TimeoutError()

        await asyncio.sleep(0.1)


async def send_with_timeout_retry(
    session: Any,
    prompt: str,
    timeout_seconds: float,
    timeout_retries: int,
    processing_id: str | None = None,
    requested_model: str | None = None,
) -> str | None:
    async def operation() -> str:
        return await send_once(
            session,
            prompt,
            timeout_seconds,
            requested_model,
        )

    return await run_with_timeout_retry(operation, timeout_retries, processing_id)


async def _get_payload_with_repair_on_session(
    session: Any,
    prompt: str,
    transcription: str,
    processing_id: str,
    requested_model: str,
    repair_prompt_template: str,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
) -> dict | None:
    for empty_attempt in range(empty_result_retries + 1):
        attempt_prompt = prompt
        if empty_attempt > 0:
            retry_nonce = random.randint(1, 1_000_000_000)
            attempt_prompt = f"{prompt}\n\nRetry nonce: {retry_nonce}"
        json_instruction_boost = "".join(["NO prose/reasoning!!! Only JSON!!! "] * (empty_attempt+1) * 2)
        attempt_prompt = f"{attempt_prompt}\n\nDirectly output JSON, {json_instruction_boost}Output JSON here:"

        content = await send_with_timeout_retry(
            session,
            attempt_prompt,
            timeout_seconds,
            timeout_retries,
            processing_id,
            requested_model,
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
                session,
                repair_prompt,
                timeout_seconds,
                timeout_retries,
                processing_id,
                requested_model,
            )
            if repaired_content is None:
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    "Repair attempt timed out.",
                    processing_id=processing_id,
                )
                if not should_retry:
                    return None
                continue

            repaired_content = repaired_content.strip()
            if not repaired_content:
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    "Repair retry returned empty output.",
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


async def get_payload_with_repair(
    create_session: Callable[[], Any],
    prompt: str,
    transcription: str,
    processing_id: str,
    requested_model: str,
    repair_prompt_template: str,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    model_mismatch_retries: int,
) -> dict | None:
    for mismatch_attempt in range(model_mismatch_retries + 1):
        session = await create_session()
        try:
            return await _get_payload_with_repair_on_session(
                session,
                prompt,
                transcription,
                processing_id,
                requested_model,
                repair_prompt_template,
                timeout_seconds,
                timeout_retries,
                empty_result_retries,
            )
        except ModelMismatchError as mismatch_error:
            is_last_mismatch_attempt = mismatch_attempt == model_mismatch_retries
            if is_last_mismatch_attempt:
                print(
                    f"[{processing_id}] Failed due to model mismatch "
                    f"(requested={mismatch_error.requested_model}, actual={mismatch_error.actual_model}) "
                    f"after {model_mismatch_retries + 1} attempt(s); emitting empty payload."
                )
                return None

            backoff_seconds = 2 ** mismatch_attempt
            backoff_seconds *= 10
            print(
                f"[{processing_id}] Model mismatch "
                f"(requested={mismatch_error.requested_model}, actual={mismatch_error.actual_model}). "
                f"Retrying item {mismatch_attempt + 2}/{model_mismatch_retries + 1} "
                f"in {backoff_seconds}s..."
            )
            await asyncio.sleep(backoff_seconds)

    return None


async def print_available_models(client: CopilotClient) -> None:
    try:
        models_result = client.list_models()
        if asyncio.iscoroutine(models_result):
            models_result = await models_result

        models = models_result
        if models is None:
            print("Available models: <none returned>")
            return

        if isinstance(models, dict):
            for key in ("models", "items", "data"):
                candidate = models.get(key)
                if isinstance(candidate, list):
                    models = candidate
                    break

        if not isinstance(models, list):
            print(f"Available models (raw): {models}")
            return

        print("Available models from client.list_models():")
        for item in models:
            if isinstance(item, dict):
                model_id = item.get("id") or item.get("model") or item.get("name")
                display_name = item.get("display_name") or item.get("displayName") or item.get("description")
            else:
                model_id = getattr(item, "id", None) or getattr(item, "model", None) or getattr(item, "name", None)
                display_name = (
                    getattr(item, "display_name", None)
                    or getattr(item, "displayName", None)
                    or getattr(item, "description", None)
                )

            if not model_id:
                model_id = str(item)

            if display_name and display_name != model_id:
                print(f"- {model_id}: {display_name}")
            else:
                print(f"- {model_id}")
    except Exception as error:
        print(f"Warning: client.list_models() failed: {error}")


async def main():
    args = parse_args()

    input_file_value = args.input_file
    output_file_value = args.output_file

    transcriptions: list[str]
    if args.list_models_only:
        transcriptions = []
    else:
        loaded_transcriptions = collect_transcriptions_from_input(input_file_value)
        if loaded_transcriptions is None:
            return
        transcriptions = loaded_transcriptions

    model = args.model
    configured_concurrency = args.concurrency
    concurrency = max(1, configured_concurrency)

    timeout_seconds = args.timeout
    timeout_retries = max(0, args.timeout_retries)
    empty_result_retries = max(0, args.empty_result_retries)
    model_mismatch_retries = max(0, args.model_mismatch_retries)

    prompt_template_path, repair_prompt_template_path, template_error = resolve_patch_and_repair_template_paths(
        args.patch_prompt_file,
        args.repair_prompt_file,
    )
    if template_error:
        print(template_error)
        return
    if prompt_template_path is None or repair_prompt_template_path is None:
        return

    print(f"Using model: {model}")
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

    client = CopilotClient()
    await client.start()

    try:
        should_print_models = (
            args.list_models_only
            or args.print_models
        )
        if should_print_models:
            await print_available_models(client)

        if args.list_models_only:
            return

        payloads: list[dict | None] = [None] * len(transcriptions)

        async def process_item(index: int, transcription: str, total: int) -> None:
            requested_model = model
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
                session_parameters = {
                    "model": model, 
                    "tools": [], 
                    "streaming": False,
                    "system_message": STRICT_JSON_SYSTEM, 
                }
                if model in ["gpt-5.2"]:
                    session_parameters["reasoning_effort"] = "low"

                async def create_session() -> Any:
                    return await client.create_session(session_parameters)

                payload = await get_payload_with_repair(
                    create_session,
                    prompt,
                    transcription,
                    processing_id,
                    requested_model,
                    repair_prompt_template,
                    timeout_seconds,
                    timeout_retries,
                    empty_result_retries,
                    model_mismatch_retries,
                )

                assign_payload_or_emit_empty(payload, payloads, slot, index, total)
                return
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

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
