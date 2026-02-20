import json
import os
from pathlib import Path


def get_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def get_env_optional_float(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def resolve_float_with_fallback(
    primary_env_name: str,
    fallback_env_name: str,
    default: float,
) -> tuple[float, str | None]:
    primary_value = get_env_optional_float(primary_env_name)
    if primary_value is not None:
        return primary_value, primary_env_name

    fallback_raw = os.getenv(fallback_env_name)
    source: str | None = None
    if fallback_raw is not None and fallback_raw.strip():
        source = fallback_env_name

    return get_env_float(fallback_env_name, default), source


def resolve_bool_with_override(
    override_value: str | None,
    env_name: str,
    default: bool,
) -> bool:
    if override_value is not None:
        return override_value == "1"
    return get_env_bool(env_name, default)


def get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def get_env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def validate_patch_payload(payload: dict) -> tuple[bool, str]:
    required_keys = {
        "tokenization",
        "ops",
        "ct_combine",
        "ct_fix",
        "ct_punct",
        "corrected_text",
        "verification",
        "machine_transcription_probability",
    }
    payload_keys = set(payload.keys())

    if payload_keys != required_keys:
        return False, (
            "Top-level keys must be exactly: "
            "tokenization, ops, ct_combine, ct_fix, "
            "ct_punct, corrected_text, verification, "
            "machine_transcription_probability"
        )

    tokenization = payload.get("tokenization")
    if not isinstance(tokenization, dict):
        return False, "tokenization must be an object"
    if tokenization.get("scheme") != "universal_v1":
        return False, "tokenization.scheme must be 'universal_v1'"
    if not isinstance(tokenization.get("tokens"), list):
        return False, "tokenization.tokens must be an array"

    if not isinstance(payload.get("ops"), list):
        return False, "ops must be an array"
    if not isinstance(payload.get("ct_combine"), str):
        return False, "ct_combine must be a string"
    if not isinstance(payload.get("ct_fix"), str):
        return False, "ct_fix must be a string"
    if not isinstance(payload.get("ct_punct"), str):
        return False, "ct_punct must be a string"
    if not isinstance(payload.get("corrected_text"), str):
        return False, "corrected_text must be a string"

    verification = payload.get("verification")
    if not isinstance(verification, dict):
        return False, "verification must be an object"
    if not isinstance(verification.get("edit_count"), int):
        return False, "verification.edit_count must be an integer"
    if not isinstance(verification.get("op_details"), list):
        return False, "verification.op_details must be an array"
    if verification.get("edit_count") != len(payload.get("ops", [])):
        return False, "verification.edit_count must equal len(ops)"

    op_details = verification.get("op_details", [])
    if len(op_details) < 4:
        return False, "verification.op_details must include TOKEN_COMBINE/ERROR_FIX/PUNCTUATION/CASING entries"
    if not all(isinstance(item, str) for item in op_details):
        return False, "verification.op_details entries must be strings"
    if not op_details[0].startswith("TOKEN_COMBINE_RESULT:"):
        return False, "verification.op_details[0] must start with 'TOKEN_COMBINE_RESULT:'"
    if not op_details[1].startswith("ERROR_FIX_RESULT:"):
        return False, "verification.op_details[1] must start with 'ERROR_FIX_RESULT:'"
    if not op_details[2].startswith("PUNCTUATION_RESULT:"):
        return False, "verification.op_details[2] must start with 'PUNCTUATION_RESULT:'"
    if not op_details[3].startswith("CASING_RESULT:"):
        return False, "verification.op_details[3] must start with 'CASING_RESULT:'"

    machine_probability = payload.get("machine_transcription_probability")
    if not isinstance(machine_probability, (int, float)):
        return False, "machine_transcription_probability must be a number"
    if machine_probability < 0 or machine_probability > 1:
        return False, "machine_transcription_probability must be in [0,1]"

    return True, ""


def validate_output_payloads(payloads: list[dict]) -> tuple[bool, str]:
    for index, payload in enumerate(payloads, start=1):
        if not isinstance(payload, dict):
            return False, f"Output item {index} must be an object"

        is_valid, validation_error = validate_patch_payload(payload)
        if not is_valid:
            return False, f"Output item {index} schema error: {validation_error}"

    return True, ""


def normalize_payload_for_schema(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload

    ops = payload.get("ops")
    verification = payload.get("verification")
    if isinstance(ops, list) and isinstance(verification, dict):
        verification["edit_count"] = len(ops)

    return payload


def extract_first_json_object(text: str) -> str | None:
    start_index = text.find("{")
    if start_index == -1:
        return None

    depth = 0
    in_string = False
    is_escaped = False

    for index in range(start_index, len(text)):
        char = text[index]

        if in_string:
            if is_escaped:
                is_escaped = False
            elif char == "\\":
                is_escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index:index + 1]

    return None


def parse_and_validate_json(content: str) -> tuple[dict | None, str | None]:
    trimmed = content.strip()
    if trimmed.startswith("```"):
        lines = trimmed.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            trimmed = "\n".join(lines[1:-1]).strip()

    extracted = extract_first_json_object(trimmed)
    candidate = extracted if extracted is not None else trimmed

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        return None, f"JSON parse error: {error}"

    payload = normalize_payload_for_schema(payload)

    is_valid, validation_error = validate_patch_payload(payload)
    if not is_valid:
        return None, f"Schema error: {validation_error}"

    return payload, None


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent / candidate


def resolve_template_path(
    override_value: str | None,
    default_filename: str,
) -> Path:
    if override_value:
        template_path = Path(override_value)
        if not template_path.is_absolute():
            template_path = Path(__file__).parent / template_path
        return template_path

    return Path(__file__).with_name(default_filename)


def resolve_required_template_path(
    override_value: str | None,
    default_filename: str,
    label: str,
) -> tuple[Path, str | None]:
    template_path = resolve_template_path(override_value, default_filename)
    if not template_path.exists():
        return template_path, f"{label} file not found: {template_path}"
    return template_path, None


def parse_transcriptions_from_file(input_path: Path) -> list[str]:
    raw_text = input_path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return []

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        transcriptions = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
        return transcriptions

    if isinstance(parsed, dict):
        items = parsed.get("transcriptions")
        if isinstance(items, list):
            transcriptions = [item.strip() for item in items if isinstance(item, str) and item.strip()]
            return transcriptions

    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def strip_prompt_comments(prompt_text: str) -> str:
    filtered_lines: list[str] = []
    for line in prompt_text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def load_prompt_template(template_path: Path) -> str:
    return strip_prompt_comments(template_path.read_text(encoding="utf-8"))


def write_output_artifacts(
    payloads: list[dict],
    output_file_value: str | None,
    output_plain_file_value: str | None,
) -> None:
    output_payload = payloads[0] if len(payloads) == 1 else payloads

    corrected_text_lines: list[str] = []
    for item in payloads:
        corrected_text = item.get("corrected_text") if isinstance(item, dict) else ""
        corrected_text_lines.append(corrected_text if isinstance(corrected_text, str) else "")
    plain_output_content = "\n".join(corrected_text_lines)

    output_content = json.dumps(output_payload, ensure_ascii=False, indent=2)
    if output_file_value:
        output_path = resolve_path(output_file_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_content, encoding="utf-8")
        print(f"Wrote result to: {output_path}")

        if output_plain_file_value:
            output_plain_path = resolve_path(output_plain_file_value)
        else:
            output_plain_path = output_path.with_suffix(".txt")

        output_plain_path.parent.mkdir(parents=True, exist_ok=True)
        output_plain_path.write_text(plain_output_content, encoding="utf-8")
        print(f"Wrote plain text result to: {output_plain_path}")
    else:
        print(output_content)


def format_repair_prompt(
    repair_prompt_template: str,
    validation_error: str | None,
    previous_output: str,
) -> str:
    if "{validation_error}" in repair_prompt_template or "{previous_output}" in repair_prompt_template:
        return (
            repair_prompt_template
            .replace("{validation_error}", validation_error or "")
            .replace("{previous_output}", previous_output)
        )

    return (
        f"{repair_prompt_template}\n\n"
        f"Validation error:\n{validation_error}\n\n"
        f"Previous output:\n{previous_output}"
    )


def apply_corrected_text_fallback(payload: dict, transcription: str) -> dict | None:
    if not isinstance(payload, dict):
        return None

    fallback_text = transcription.strip()
    if not fallback_text:
        return None

    if not isinstance(payload.get("ct_combine"), str) or not payload.get("ct_combine", "").strip():
        payload["ct_combine"] = fallback_text
    if not isinstance(payload.get("ct_fix"), str) or not payload.get("ct_fix", "").strip():
        payload["ct_fix"] = fallback_text
    if not isinstance(payload.get("ct_punct"), str) or not payload.get("ct_punct", "").strip():
        payload["ct_punct"] = fallback_text
    payload["corrected_text"] = fallback_text
    return payload
