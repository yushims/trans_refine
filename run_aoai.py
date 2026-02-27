import argparse
import asyncio
import json
import os
import random
from pathlib import Path

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
    resolve_bool_with_override,
    resolve_float_with_fallback,
    resolve_path,
    resolve_required_template_path,
    validate_output_payloads,
    validate_patch_payload,
    write_output_artifacts,
)

load_dotenv()


def _initialize_output_paths(
    output_file_value: str | None,
    output_plain_file_value: str | None,
):
    if not output_file_value:
        return None, None

    output_path = resolve_path(output_file_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    if output_plain_file_value:
        output_plain_path = resolve_path(output_plain_file_value)
    else:
        output_plain_path = output_path.with_suffix(".txt")

    output_plain_path.parent.mkdir(parents=True, exist_ok=True)
    output_plain_path.write_text("", encoding="utf-8")

    print(f"Streaming JSON output to: {output_path}")
    print(f"Streaming plain text output to: {output_plain_path}")

    return output_path, output_plain_path

def _append_output_artifacts(
    result: dict,
    output_path,
    output_plain_path,
    utt_id: str | None = None,
) -> None:
    output_result = result
    if utt_id is not None:
        output_result = dict(result)
        output_result["utt_id"] = utt_id

    output_line = json.dumps(output_result, ensure_ascii=False)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(output_line)
        output_file.write("\n")

    corrected_text = result.get("corrected_text") if isinstance(result, dict) else ""
    corrected_text_value = corrected_text if isinstance(corrected_text, str) else ""
    with output_plain_path.open("a", encoding="utf-8") as output_plain_file:
        if utt_id is not None:
            output_plain_file.write(f"{utt_id}\t{corrected_text_value}")
        else:
            output_plain_file.write(corrected_text_value)
        output_plain_file.write("\n")


def _append_failed_output_artifacts(
    transcription: str,
    output_path,
    output_plain_path,
    utt_id: str | None = None,
) -> None:
    with output_path.open("a", encoding="utf-8") as output_file:
        if utt_id is not None:
            output_file.write(json.dumps({"utt_id": utt_id, "corrected_text": transcription}, ensure_ascii=False))
        output_file.write("\n")

    with output_plain_path.open("a", encoding="utf-8") as output_plain_file:
        if utt_id is not None:
            output_plain_file.write(f"{utt_id}\t{transcription}")
        else:
            output_plain_file.write(transcription)
        output_plain_file.write("\n")


def parse_transcription_records_from_file(input_path: Path) -> list[tuple[str | None, str]]:
    raw_text = input_path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return []

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [(None, item.strip()) for item in parsed if isinstance(item, str) and item.strip()]

    if isinstance(parsed, dict):
        items = parsed.get("transcriptions")
        if isinstance(items, list):
            return [(None, item.strip()) for item in items if isinstance(item, str) and item.strip()]

    records: list[tuple[str | None, str]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "\t" in line:
            utt_id_part, text_part = line.split("\t", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt_id_part, text_part = parts
            else:
                utt_id_part, text_part = "", line

        utt_id = utt_id_part.strip()
        text = text_part.strip()
        if not text:
            continue

        records.append((utt_id if utt_id else None, text))

    return records


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

    transcription_records: list[tuple[str | None, str]]
    if input_file_value:
        input_path = resolve_path(input_file_value)
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            return
        transcription_records = parse_transcription_records_from_file(input_path)
        print(f"Read {len(transcription_records)} transcription(s) from: {input_path}")
    else:
        transcription = input("Enter transcription: ").strip()
        transcription_records = [(None, transcription)] if transcription else []

    if not transcription_records:
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

    timeout_seconds = get_env_float("AOAI_TIMEOUT", 60.0)
    timeout_retries = max(0, get_env_int("AOAI_TIMEOUT_RETRIES", 4))
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

    output_path, output_plain_path = _initialize_output_paths(
        output_file_value,
        output_plain_file_value,
    )

    payloads: list[dict] = []
    failed_count = 0
    total = len(transcription_records)
    for index, record in enumerate(transcription_records, start=1):
        utt_id, transcription = record
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
            if output_path is not None and output_plain_path is not None:
                _append_failed_output_artifacts(transcription, output_path, output_plain_path, utt_id)
                failed_count += 1
                print(
                    f"Failed on transcription {index}/{total} after retries. "
                    "Wrote fallback outputs and continuing..."
                )
                continue

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

        if output_path is not None and output_plain_path is not None:
            _append_output_artifacts(result, output_path, output_plain_path, utt_id)

    if validate_output:
        is_valid, validation_error = validate_output_payloads(payloads)
        if not is_valid:
            print(validation_error)
            return

    if output_path is not None and output_plain_path is not None:
        print(f"Wrote {len(payloads)} result(s) to: {output_path}")
        print(f"Wrote {len(payloads)} plain text line(s) to: {output_plain_path}")
        if failed_count > 0:
            print(f"Fallback lines written for {failed_count} failed transcription(s).")
        return

    write_output_artifacts(payloads, output_file_value, output_plain_file_value)


if __name__ == "__main__":
    asyncio.run(main())
