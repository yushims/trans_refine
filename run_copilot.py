import argparse
import asyncio
import os
import random

from copilot import CopilotClient  # pyright: ignore[reportMissingImports]
from pipeline_common import (
    apply_corrected_text_fallback,
    get_env_bool,
    get_env_float,
    get_env_int,
    format_repair_prompt,
    load_prompt_template,
    parse_and_validate_json,
    parse_transcriptions_from_file,
    resolve_bool_with_override,
    resolve_path,
    resolve_required_template_path,
    validate_output_payloads,
    validate_patch_payload,
    write_output_artifacts,
)
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", dest="input_file")
    parser.add_argument("--output-file", dest="output_file")
    parser.add_argument("--output-plain-file", dest="output_plain_file")
    parser.add_argument("--prompt-file", dest="prompt_file")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--list-models-only", dest="list_models_only", action="store_true")
    parser.add_argument("--validate-output", dest="validate_output", choices=["0", "1"])
    return parser.parse_args()


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


async def send_with_timeout_retry(
    session,
    prompt: str,
    timeout_seconds: float,
    timeout_retries: int,
):
    for attempt in range(timeout_retries + 1):
        try:
            return await session.send_and_wait(
                {"prompt": prompt, "mode": "immediate"},
                timeout=timeout_seconds,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            is_last_attempt = attempt == timeout_retries
            if is_last_attempt:
                return None

            backoff_seconds = 2 ** attempt
            print(
                f"Timeout/cancel on attempt {attempt + 1}/{timeout_retries + 1}. "
                f"Retrying in {backoff_seconds}s..."
            )
            await asyncio.sleep(backoff_seconds)


async def get_payload_with_repair(
    session,
    prompt: str,
    transcription: str,
    repair_prompt_template: str,
    timeout_seconds: float,
    timeout_retries: int,
    strict_json_repair: bool,
    empty_result_retries: int,
):
    for empty_attempt in range(empty_result_retries + 1):
        attempt_prompt = prompt
        if empty_attempt > 0:
            retry_nonce = random.randint(1, 1_000_000_000)
            attempt_prompt = f"{prompt}\n\nRetry nonce: {retry_nonce}"
            print(f"Retry prompt nonce applied: {retry_nonce}")

        response = await send_with_timeout_retry(
            session,
            attempt_prompt,
            timeout_seconds,
            timeout_retries,
        )
        if response is None:
            print(
                f"Request timed out after {timeout_seconds}s "
                f"for {timeout_retries + 1} attempt(s)."
            )
            print(
                "Increase COPILOT_TIMEOUT or COPILOT_TIMEOUT_RETRIES "
                "and try again."
            )
            return None

        if response.data is None:
            print("No response received from the model.")
            return None

        content = response.data.content.strip()
        if content:
            payload, validation_error = parse_and_validate_json(content)
        else:
            payload = None
            validation_error = "Model returned empty output."

        if payload is None:
            if not strict_json_repair:
                print("Initial response was not valid strict JSON.")
                if validation_error:
                    print(validation_error)
                print("Raw output:")
                print(content)
                return None

            print("Initial response was not valid strict JSON. Running one repair attempt...")
            repair_prompt = format_repair_prompt(
                repair_prompt_template,
                validation_error,
                content,
            )

            repair_response = await send_with_timeout_retry(
                session,
                repair_prompt,
                timeout_seconds,
                timeout_retries,
            )
            if repair_response is None:
                is_last_empty_attempt = empty_attempt == empty_result_retries
                if is_last_empty_attempt:
                    print("Repair attempt timed out.")
                    return None
                print(
                    f"Repair attempt timed out. "
                    f"Retrying item {empty_attempt + 1}/{empty_result_retries + 1}..."
                )
                continue

            if repair_response.data is None:
                is_last_empty_attempt = empty_attempt == empty_result_retries
                if is_last_empty_attempt:
                    print("Repair retry returned no response data.")
                    return None
                print(
                    f"Repair retry returned no response data. "
                    f"Retrying item {empty_attempt + 1}/{empty_result_retries + 1}..."
                )
                continue

            repaired_content = repair_response.data.content.strip()
            if not repaired_content:
                is_last_empty_attempt = empty_attempt == empty_result_retries
                if is_last_empty_attempt:
                    print("Repair retry returned empty output.")
                    return None
                print(
                    f"Repair retry returned empty output. "
                    f"Retrying item {empty_attempt + 1}/{empty_result_retries + 1}..."
                )
                continue

            payload, validation_error = parse_and_validate_json(repaired_content)
            if payload is None:
                is_last_empty_attempt = empty_attempt == empty_result_retries
                if is_last_empty_attempt:
                    print("Repair attempt did not return valid JSON.")
                    print(validation_error)
                    print("Raw output:")
                    print(repaired_content)
                    return None
                print(
                    f"Repair attempt did not return valid JSON ({validation_error}). "
                    f"Retrying item {empty_attempt + 1}/{empty_result_retries + 1}..."
                )
                continue

        corrected_text = payload.get("corrected_text") if isinstance(payload, dict) else None
        if isinstance(corrected_text, str) and corrected_text.strip():
            return payload

        is_last_empty_attempt = empty_attempt == empty_result_retries
        if is_last_empty_attempt:
            print("Model returned empty corrected_text after retries. Applying fallback to original transcription text.")
            return apply_corrected_text_fallback(payload, transcription)

        print(
            f"Model returned empty corrected_text. "
            f"Retrying item {empty_attempt + 1}/{empty_result_retries + 1}..."
        )

    return None


async def main():
    client = CopilotClient()
    await client.start()

    try:
        args = parse_args()
        await print_available_models(client)

        if args.list_models_only:
            return

        input_file_value = args.input_file or os.getenv("INPUT_FILE")
        output_file_value = args.output_file or os.getenv("OUTPUT_FILE")
        output_plain_file_value = args.output_plain_file or os.getenv("OUTPUT_PLAIN_FILE")

        transcriptions: list[str]
        if input_file_value:
            input_path = resolve_path(input_file_value)
            if not input_path.exists():
                print(f"Input file not found: {input_path}")
                return
            transcriptions = parse_transcriptions_from_file(input_path)
            print(f"Read {len(transcriptions)} transcription(s) from: {input_path}")
        else:
            transcription = input("Enter transcription: ").strip()
            transcriptions = [transcription] if transcription else []

        if not transcriptions:
            print("No transcription provided.")
            return

        model = args.model or os.getenv("COPILOT_MODEL", "gpt-5.2")
        timeout_seconds = get_env_float("COPILOT_TIMEOUT", 180.0)
        timeout_retries = max(0, get_env_int("COPILOT_TIMEOUT_RETRIES", 2))
        empty_result_retries = max(0, get_env_int("COPILOT_EMPTY_RESULT_RETRIES", 1))
        strict_json_repair = get_env_bool("STRICT_JSON_REPAIR", True)
        validate_output = resolve_bool_with_override(args.validate_output, "COPILOT_VALIDATE_OUTPUT", True)

        prompt_template_path, prompt_path_error = resolve_required_template_path(
            args.prompt_file or os.getenv("PROMPT_FILE"),
            "prompt_patch_copilot.txt",
            "Prompt",
        )
        if prompt_path_error:
            print(prompt_path_error)
            return

        repair_prompt_template_path, repair_path_error = resolve_required_template_path(
            args.repair_prompt_file or os.getenv("COPILOT_REPAIR_PROMPT_FILE"),
            "prompt_repair_copilot.txt",
            "Repair prompt",
        )
        if repair_path_error:
            print(repair_path_error)
            return

        print(f"Using model: {model}")
        print("Note: Copilot treats this as a requested model; backend may resolve/fallback and may not expose final model ID.")
        print(f"Using prompt file: {prompt_template_path}")
        print(f"Using repair prompt file: {repair_prompt_template_path}")
        print(f"Timeout: {timeout_seconds}s, retries: {timeout_retries}")
        print(f"Empty-result retries: {empty_result_retries}")
        print(f"Strict JSON repair: {strict_json_repair}")
        print(f"Validate output: {validate_output}")

        prompt_template = load_prompt_template(prompt_template_path)
        repair_prompt_template = load_prompt_template(repair_prompt_template_path)

        payloads: list[dict] = []
        total = len(transcriptions)
        for index, transcription in enumerate(transcriptions, start=1):
            if total > 1:
                print(f"Processing transcription {index}/{total}...")

            session = await client.create_session({"model": model})

            requested_model = model
            resolved_model_holder = {"value": None}

            def handle_event(event):
                event_type = getattr(event.type, "value", event.type)

                if event_type == "assistant.usage":
                    actual_model = (
                        getattr(event.data, "model", None)
                        or getattr(event.data, "model_id", None)
                    )
                    if actual_model:
                        resolved_model_holder["value"] = actual_model
                        # output only when requested model is not equal to resolved model
                        if actual_model != requested_model:
                            print(
                                f"[MODEL MISMATCH] requested={requested_model}, "
                                f"actual={actual_model}"
                            )
            session.on(handle_event)

            prompt = prompt_template + transcription

            payload = await get_payload_with_repair(
                session,
                prompt,
                transcription,
                repair_prompt_template,
                timeout_seconds,
                timeout_retries,
                strict_json_repair,
                empty_result_retries,
            )
            if payload is None:
                print(f"Failed on transcription {index}/{total}.")
                return

            if validate_output:
                is_valid, validation_error = validate_patch_payload(payload)
                if not is_valid:
                    print(
                        f"Output validation failed on transcription {index}/{total}: "
                        f"{validation_error}"
                    )
                    return

            payloads.append(payload)

        if validate_output:
            is_valid, validation_error = validate_output_payloads(payloads)
            if not is_valid:
                print(validation_error)
                return

        write_output_artifacts(payloads, output_file_value, output_plain_file_value)

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
