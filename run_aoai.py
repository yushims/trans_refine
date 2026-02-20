import argparse
import asyncio
import os
import random

from dotenv import load_dotenv
from openai import AzureOpenAI
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
    resolve_float_with_fallback,
    resolve_path,
    resolve_required_template_path,
    validate_output_payloads,
    validate_patch_payload,
    write_output_artifacts,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", dest="input_file")
    parser.add_argument("--output-file", dest="output_file")
    parser.add_argument("--output-plain-file", dest="output_plain_file")
    parser.add_argument("--prompt-file", dest="prompt_file")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file")
    parser.add_argument("--deployment", dest="deployment")
    parser.add_argument("--endpoint", dest="endpoint")
    parser.add_argument("--api-version", dest="api_version")
    parser.add_argument("--validate-output", dest="validate_output", choices=["0", "1"])
    return parser.parse_args()


def _extract_text_content(completion) -> str:
    choices = getattr(completion, "choices", None)
    if not choices:
        return ""

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return ""


def _chat_completion_create(client: AzureOpenAI, request_kwargs: dict):
    return client.chat.completions.create(**request_kwargs)


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
        "temperature": temperature,
        "top_p": top_p,
    }

    completion = await asyncio.wait_for(
        asyncio.to_thread(_chat_completion_create, client, request_kwargs),
        timeout=timeout_seconds,
    )

    return _extract_text_content(completion).strip()


async def send_with_timeout_retry(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
    timeout_retries: int,
) -> str | None:
    current_timeout_seconds = timeout_seconds
    for attempt in range(timeout_retries + 1):
        try:
            return await send_once(
                client,
                deployment,
                prompt,
                temperature,
                top_p,
                current_timeout_seconds,
            )
        except asyncio.TimeoutError:
            is_last_attempt = attempt == timeout_retries
            if is_last_attempt:
                return None

            backoff_seconds = 2 ** attempt
            next_timeout_seconds = current_timeout_seconds + timeout_seconds
            print(
                f"Timeout on attempt {attempt + 1}/{timeout_retries + 1}. "
                f"Retrying in {backoff_seconds}s with timeout={next_timeout_seconds}s..."
            )
            current_timeout_seconds = next_timeout_seconds
            await asyncio.sleep(backoff_seconds)


async def get_payload_with_repair(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    transcription: str,
    repair_prompt_template: str,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
    timeout_retries: int,
    strict_json_repair: bool,
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
                f"Retry decode jitter applied: temperature={attempt_temperature:.3f}, "
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
        )

        if content is None:
            print(
                f"Request timed out after {timeout_seconds}s "
                f"for {timeout_retries + 1} attempt(s)."
            )
            print(
                "Increase AOAI_TIMEOUT or AOAI_TIMEOUT_RETRIES "
                "and try again."
            )
            return None

        if not content:
            validation_error = "Model returned empty output."
            payload = None
        else:
            payload, validation_error = parse_and_validate_json(content)

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

            repaired_content = await send_with_timeout_retry(
                client,
                deployment,
                repair_prompt,
                attempt_temperature,
                attempt_top_p,
                timeout_seconds,
                timeout_retries,
            )

            if not repaired_content:
                is_last_empty_attempt = empty_attempt == empty_result_retries
                if is_last_empty_attempt:
                    print("Repair returned empty output or timed out.")
                    return None
                print(
                    f"Repair returned empty output or timed out. "
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
    args = parse_args()

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

    endpoint = args.endpoint or os.getenv(
        "ENDPOINT_URL",
        "https://adaptationdev-resource.openai.azure.com/",
    )
    deployment = args.deployment or os.getenv("DEPLOYMENT_NAME", "gpt-5-chat")
    api_version = args.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("AZURE_OPENAI_API_KEY is not set.")
        return

    temperature_default = 0.0
    top_p_default = 1.0

    temperature, temperature_source = resolve_float_with_fallback(
        "AOAI_TEMPERATURE",
        "TEMPERATURE",
        temperature_default,
    )

    top_p, top_p_source = resolve_float_with_fallback(
        "AOAI_TOP_P",
        "TOP_P",
        top_p_default,
    )

    timeout_seconds = get_env_float("AOAI_TIMEOUT", 180.0)
    timeout_retries = max(0, get_env_int("AOAI_TIMEOUT_RETRIES", 2))
    empty_result_retries = max(0, get_env_int("AOAI_EMPTY_RESULT_RETRIES", 1))
    retry_temperature_jitter = max(0.0, get_env_float("AOAI_RETRY_TEMPERATURE_JITTER", 0.08))
    retry_top_p_jitter = max(0.0, get_env_float("AOAI_RETRY_TOP_P_JITTER", 0.03))
    strict_json_repair = get_env_bool("STRICT_JSON_REPAIR", True)
    validate_output = resolve_bool_with_override(args.validate_output, "AOAI_VALIDATE_OUTPUT", True)

    prompt_template_path, prompt_path_error = resolve_required_template_path(
        args.prompt_file or os.getenv("PROMPT_FILE"),
        "prompt_patch_aoai.txt",
        "Prompt",
    )
    if prompt_path_error:
        print(prompt_path_error)
        return

    repair_prompt_template_path, repair_path_error = resolve_required_template_path(
        args.repair_prompt_file or os.getenv("AOAI_REPAIR_PROMPT_FILE"),
        "prompt_repair_aoai.txt",
        "Repair prompt",
    )
    if repair_path_error:
        print(repair_path_error)
        return

    print(f"Using deployment: {deployment}")
    print(f"Using endpoint: {endpoint}")
    print(f"API version: {api_version}")
    print(f"Temperature: {temperature}")
    print(f"Top p: {top_p}")
    print(f"Using prompt file: {prompt_template_path}")
    print(f"Using repair prompt file: {repair_prompt_template_path}")
    print(f"Timeout: {timeout_seconds}s, retries: {timeout_retries}")
    print(f"Empty-result retries: {empty_result_retries}")
    print(f"Retry jitter: temperature<=+{retry_temperature_jitter}, top_p±{retry_top_p_jitter}")
    print(f"Strict JSON repair: {strict_json_repair}")
    print(f"Validate output: {validate_output}")
    if temperature_source is not None and temperature != temperature_default:
        print(
            f"Warning: {temperature_source} overrides deterministic "
            f"temperature default {temperature_default} -> {temperature}"
        )
    if top_p_source is not None and top_p != top_p_default:
        print(
            f"Warning: {top_p_source} overrides deterministic "
            f"top_p default {top_p_default} -> {top_p}"
        )
    prompt_template = load_prompt_template(prompt_template_path)
    repair_prompt_template = load_prompt_template(repair_prompt_template_path)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    payloads: list[dict] = []
    total = len(transcriptions)
    for index, transcription in enumerate(transcriptions, start=1):
        if total > 1:
            print(f"Processing transcription {index}/{total}...")

        prompt = prompt_template + transcription

        result = await get_payload_with_repair(
            client,
            deployment,
            prompt,
            transcription,
            repair_prompt_template,
            temperature,
            top_p,
            timeout_seconds,
            timeout_retries,
            strict_json_repair,
            empty_result_retries,
            retry_temperature_jitter,
            retry_top_p_jitter,
        )
        if result is None:
            print(f"Failed on transcription {index}/{total}.")
            return

        if validate_output:
            is_valid, validation_error = validate_patch_payload(result)
            if not is_valid:
                print(
                    f"Output validation failed on transcription {index}/{total}: "
                    f"{validation_error}"
                )
                return

        payloads.append(result)

    if validate_output:
        is_valid, validation_error = validate_output_payloads(payloads)
        if not is_valid:
            print(validation_error)
            return

    write_output_artifacts(payloads, output_file_value, output_plain_file_value)


if __name__ == "__main__":
    asyncio.run(main())
