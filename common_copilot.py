import asyncio
import json
import random
import unicodedata
from collections.abc import Callable
from typing import Any

from common import (
    build_patch_payload_schema,
    extract_retry_after_seconds,
    extract_text_content,
    get_patch_payload_with_repair_generic,
    run_with_timeout_retry,
)


STRICT_JSON_SYSTEM = """
You are a JSON formatter.
Return ONLY one strict JSON object.
No markdown, no prose, no code fences, no comments.
No extra keys.
If any field is unknown, use valid defaults instead of explanation.
""".strip()


def build_copilot_session_parameters(model_name: str) -> dict[str, Any]:
    session_parameters: dict[str, Any] = {
        "model": model_name,
        "tools": [],
        "streaming": False,
        "system_message": STRICT_JSON_SYSTEM,
        "response_format": {"type": "json_object"},
    }
    if model_name in ["gpt-5.2"]:
        session_parameters["reasoning_effort"] = "low"
    return session_parameters


async def print_available_models(client: Any) -> None:
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


class ModelMismatchError(RuntimeError):
    def __init__(self, requested_model: str, actual_model: str):
        self.requested_model = requested_model
        self.actual_model = actual_model
        super().__init__(
            f"MODEL_MISMATCH requested={requested_model} actual={actual_model}"
        )


def is_copilot_retryable_error(error: Exception) -> bool:
    if isinstance(error, ModelMismatchError):
        return False

    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    message = str(error).lower()
    return (
        "rate limit" in message
        or "too many requests" in message
        or "throttl" in message
        or "error code: 429" in message
    )


async def send_copilot_once(
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
                    latest_content = unicodedata.normalize("NFC", content)

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
                    combined = unicodedata.normalize("NFC", "".join(delta_parts).strip())
                    if combined:
                        return combined

        if latest_content:
            return latest_content

        if loop.time() >= deadline:
            raise asyncio.TimeoutError()

        await asyncio.sleep(0.1)


async def send_copilot_with_timeout_retry(
    session: Any,
    prompt: str,
    timeout_seconds: float,
    timeout_retries: int,
    processing_id: str | None = None,
    requested_model: str | None = None,
) -> str | None:
    async def operation() -> str:
        return await send_copilot_once(
            session,
            prompt,
            timeout_seconds,
            requested_model,
        )

    return await run_with_timeout_retry(
        operation,
        timeout_retries,
        processing_id,
        is_retryable_error=is_copilot_retryable_error,
        resolve_backoff_seconds=lambda error, _attempt: extract_retry_after_seconds(error),
    )


async def handle_copilot_model_mismatch_retry(
    mismatch_error: ModelMismatchError,
    mismatch_attempt: int,
    model_mismatch_retries: int,
    processing_id: str,
    failure_action: str,
) -> bool:
    is_last_mismatch_attempt = mismatch_attempt == model_mismatch_retries
    if is_last_mismatch_attempt:
        print(
            f"[{processing_id}] Failed due to model mismatch "
            f"(requested={mismatch_error.requested_model}, actual={mismatch_error.actual_model}) "
            f"after {model_mismatch_retries + 1} attempt(s); {failure_action}."
        )
        return False

    backoff_seconds = 2 ** mismatch_attempt
    backoff_seconds *= 10
    print(
        f"[{processing_id}] Model mismatch "
        f"(requested={mismatch_error.requested_model}, actual={mismatch_error.actual_model}). "
        f"Retrying item {mismatch_attempt + 2}/{model_mismatch_retries + 1} "
        f"in {backoff_seconds}s..."
    )
    await asyncio.sleep(backoff_seconds)
    return True


async def _get_copilot_patch_payload_with_repair_on_session(
    session: Any,
    prompt: str,
    transcription: str,
    processing_id: str,
    requested_model: str,
    repair_prompt_template: str,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    skip_first_token_casing_preservation: bool = False,
    active_step_keys: set[str] | None = None,
    on_final_failure: Callable[[str], None] | None = None,
) -> dict | None:
    patch_target_schema = json.dumps(
        build_patch_payload_schema(),
        ensure_ascii=False,
        indent=2,
    )

    def _build_attempt_prompt(base_prompt: str, empty_attempt: int) -> str:
        attempt_prompt = base_prompt
        if empty_attempt > 0:
            retry_nonce = random.randint(1, 1_000_000_000)
            attempt_prompt = f"{base_prompt}\n\nRetry nonce: {retry_nonce}"
        json_instruction_boost = "".join(["NO prose/reasoning!!! Only JSON!!! "] * (empty_attempt + 1) * 2)
        return f"{attempt_prompt}\n\nDirectly output JSON, {json_instruction_boost}Output JSON here:"

    async def _send_prompt(attempt_prompt: str, _empty_attempt: int) -> str | None:
        return await send_copilot_with_timeout_retry(
            session,
            attempt_prompt,
            timeout_seconds,
            timeout_retries,
            processing_id,
            requested_model,
        )

    async def _send_repair_prompt(repair_prompt: str, _empty_attempt: int) -> str | None:
        return await send_copilot_with_timeout_retry(
            session,
            repair_prompt,
            timeout_seconds,
            timeout_retries,
            processing_id,
            requested_model,
        )

    return await get_patch_payload_with_repair_generic(
        prompt=prompt,
        transcription=transcription,
        processing_id=processing_id,
        repair_prompt_template=repair_prompt_template,
        target_schema=patch_target_schema,
        timeout_seconds=timeout_seconds,
        timeout_retries=timeout_retries,
        empty_result_retries=empty_result_retries,
        send_prompt=_send_prompt,
        send_repair_prompt=_send_repair_prompt,
        build_attempt_prompt=_build_attempt_prompt,
        skip_first_token_casing_preservation=skip_first_token_casing_preservation,
        active_step_keys=active_step_keys,
        on_final_failure=on_final_failure,
        repair_timeout_message="Repair attempt timed out.",
        repair_empty_message="Repair retry returned empty output.",
        strip_repair_content=True,
    )


async def get_copilot_patch_payload_with_repair(
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
    skip_first_token_casing_preservation: bool = False,
    active_step_keys: set[str] | None = None,
    on_final_failure: Callable[[str], None] | None = None,
) -> dict | None:
    for mismatch_attempt in range(model_mismatch_retries + 1):
        session = await create_session()
        try:
            return await _get_copilot_patch_payload_with_repair_on_session(
                session,
                prompt,
                transcription,
                processing_id,
                requested_model,
                repair_prompt_template,
                timeout_seconds,
                timeout_retries,
                empty_result_retries,
                skip_first_token_casing_preservation,
                active_step_keys,
                on_final_failure,
            )
        except ModelMismatchError as mismatch_error:
            should_retry = await handle_copilot_model_mismatch_retry(
                mismatch_error,
                mismatch_attempt,
                model_mismatch_retries,
                processing_id,
                "emitting empty payload",
            )
            if not should_retry:
                return None

    return None
