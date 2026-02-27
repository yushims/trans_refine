import asyncio
import difflib
import json
import math
import re
import unicodedata
from collections.abc import Awaitable, Callable
from pathlib import Path


_REQUIRED_TOP_LEVEL_KEYS = {
    "ct_casing",
}

_OPTIONAL_TOP_LEVEL_KEYS = {
    "tokenization",
    "ct_combine",
    "ct_fix",
    "ct_punct",
    "verification",
    "machine_transcription_probability",
    "corrected_text",
}


_ALLOWED_INSERTION_FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "it", "as", "by", "with", "from", "if", "then", "so", "but", "not",
    "de", "la", "el", "y", "en", "un", "una", "con", "le", "les", "des", "et",
    "du", "au", "aux", "das", "der", "die", "und", "mit", "zu", "da", "do", "dos",
    "e", "di", "del", "della", "van", "het",
}
_URL_EMAIL_PATTERN = re.compile(
    r"(?:https?://\S+|www\.\S+|[^\s@]+@(?:[^\s@.]+\.)+[^\s@.]{2,})",
    flags=re.IGNORECASE,
)
_COMBINING_MARK_RANGES = "\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f"
_WORD_SEGMENT_PATTERN = rf"[^\W\d_](?:[^\W\d_]|[{_COMBINING_MARK_RANGES}])*"
_NUMBER_LIKE_PATTERN = re.compile(
    rf"\d+(?:[.,:٫٬]\d+)*(?:%|{_WORD_SEGMENT_PATTERN})?",
    flags=re.UNICODE,
)
_LATIN_WORD_PATTERN = re.compile(
    rf"{_WORD_SEGMENT_PATTERN}(?:['’`.-]{_WORD_SEGMENT_PATTERN})*",
    flags=re.UNICODE,
)


def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            extracted = extract_text_content(item)
            if extracted:
                parts.append(extracted)
        return "".join(parts)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

        for key in ("content", "delta_content", "message"):
            nested = content.get(key)
            extracted = extract_text_content(nested)
            if extracted:
                return extracted
        return ""

    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text

    nested_content = getattr(content, "content", None)
    extracted = extract_text_content(nested_content)
    if extracted:
        return extracted

    nested_delta = getattr(content, "delta_content", None)
    extracted = extract_text_content(nested_delta)
    if extracted:
        return extracted

    return ""


async def run_with_timeout_retry(
    operation: Callable[[], Awaitable[str]],
    timeout_retries: int,
    processing_id: str | None = None,
    is_retryable_error: Callable[[Exception], bool] | None = None,
    resolve_backoff_seconds: Callable[[Exception, int], float | None] | None = None,
) -> str | None:
    prefix = f"[{processing_id}] " if processing_id else ""

    for attempt in range(timeout_retries + 1):
        try:
            return await operation()
        except (asyncio.TimeoutError, asyncio.CancelledError, OSError) as error:
            is_last_attempt = attempt == timeout_retries
            if is_last_attempt:
                return None

            configured_backoff = (
                resolve_backoff_seconds(error, attempt)
                if resolve_backoff_seconds is not None
                else None
            )
            backoff_seconds = configured_backoff if isinstance(configured_backoff, (int, float)) and configured_backoff > 0 else 2 ** attempt
            backoff_seconds *= 10
            if isinstance(error, OSError):
                error_text = str(error).strip() or error.__class__.__name__
                print(
                    f"{prefix}Transport error on attempt {attempt + 1}/{timeout_retries + 1}: {error_text}. "
                    f"Retrying in {backoff_seconds}s..."
                )
            else:
                print(
                    f"{prefix}Timeout/cancel on attempt {attempt + 1}/{timeout_retries + 1}. "
                    f"Retrying in {backoff_seconds}s..."
                )
            await asyncio.sleep(backoff_seconds)
        except Exception as error:
            if is_retryable_error is None or not is_retryable_error(error):
                raise

            is_last_attempt = attempt == timeout_retries
            if is_last_attempt:
                return None

            configured_backoff = (
                resolve_backoff_seconds(error, attempt)
                if resolve_backoff_seconds is not None
                else None
            )
            backoff_seconds = configured_backoff if isinstance(configured_backoff, (int, float)) and configured_backoff > 0 else 2 ** attempt
            backoff_seconds *= 10
            error_text = str(error).strip() or error.__class__.__name__
            print(
                f"{prefix}Retryable error on attempt {attempt + 1}/{timeout_retries + 1}: {error_text}. "
                f"Retrying in {backoff_seconds}s..."
            )
            await asyncio.sleep(backoff_seconds)

    return None


def should_retry_after_failure(
    empty_attempt: int,
    empty_result_retries: int,
    final_message: str,
    retry_message: str | None = None,
    processing_id: str | None = None,
) -> bool:
    prefix = f"[{processing_id}] " if processing_id else ""
    is_last_empty_attempt = empty_attempt == empty_result_retries
    if is_last_empty_attempt:
        print(f"{prefix}{final_message}")
        return False

    message_for_retry = retry_message or final_message

    print(
        f"{prefix}{message_for_retry} "
        f"Retrying item {empty_attempt + 2}/{empty_result_retries + 1}..."
    )
    return True


def resolve_payload_or_retry_on_empty_corrected_text(
    payload: dict | None,
    transcription: str,
    empty_attempt: int,
    empty_result_retries: int,
    processing_id: str | None = None,
) -> tuple[dict | None, bool]:
    prefix = f"[{processing_id}] " if processing_id else ""
    corrected_text = payload.get("corrected_text") if isinstance(payload, dict) else None
    if isinstance(corrected_text, str) and corrected_text.strip():
        return payload, False

    is_last_empty_attempt = empty_attempt == empty_result_retries
    if is_last_empty_attempt:
        print(
            f"{prefix}Model returned empty corrected_text after retries. "
            "Applying fallback to original transcription text."
        )
        return apply_corrected_text_fallback(payload, transcription), False

    print(
        f"{prefix}Model returned empty corrected_text. "
        f"Retrying item {empty_attempt + 2}/{empty_result_retries + 1}..."
    )
    return None, True


def _is_punct_token(token: str) -> bool:
    return isinstance(token, str) and len(token) == 1 and unicodedata.category(token).startswith("P")


def _is_cjk_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _is_cjk_token(token: str) -> bool:
    return bool(token) and all(_is_cjk_char(char) for char in token)


def _is_hangul_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return 0xAC00 <= codepoint <= 0xD7AF


def _is_kana_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return 0x3040 <= codepoint <= 0x30FF


def _is_han_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return (0x3400 <= codepoint <= 0x4DBF) or (0x4E00 <= codepoint <= 0x9FFF)


def _is_non_latin_letter(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    if not char.isalpha():
        return False
    return "LATIN" not in unicodedata.name(char, "")


def _is_hallucinated_token(token: str, source_lexicon_folded: set[str]) -> bool:
    if not isinstance(token, str) or not token.strip():
        return True
    if len(token) > 40:
        return True
    if any(char.isspace() for char in token):
        return True

    folded = token.casefold()
    if folded in source_lexicon_folded:
        return False
    if folded in _ALLOWED_INSERTION_FUNCTION_WORDS:
        return False

    if all(unicodedata.category(char).startswith(("P", "S", "N")) for char in token):
        return False

    if not any(char.isalpha() for char in token):
        return False

    if len(token) == 1 and _is_cjk_token(token):
        return False

    if len(token) <= 2 and token.isalpha() and not _is_cjk_token(token):
        return False

    if any(_is_non_latin_letter(char) for char in token) and len(token) <= 2:
        return False

    return True


def _tokenize_for_hallucination(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue

        url_email_match = _URL_EMAIL_PATTERN.match(text, index)
        if url_email_match:
            token = url_email_match.group(0)
            tokens.append(token)
            index += len(token)
            continue

        number_match = _NUMBER_LIKE_PATTERN.match(text, index)
        if number_match:
            token = number_match.group(0)
            tokens.append(token)
            index += len(token)
            continue

        if _is_han_char(char) or _is_kana_char(char):
            tokens.append(char)
            index += 1
            continue

        if _is_hangul_char(char):
            next_index = index + 1
            while next_index < length and _is_hangul_char(text[next_index]):
                next_index += 1
            tokens.append(text[index:next_index])
            index = next_index
            continue

        latin_match = _LATIN_WORD_PATTERN.match(text, index)
        if latin_match:
            token = latin_match.group(0)
            tokens.append(token)
            index += len(token)
            continue

        category = unicodedata.category(char)
        if category.startswith(("P", "S")):
            tokens.append(char)
            index += 1
            continue

        if _is_non_latin_letter(char):
            next_index = index + 1
            while next_index < length and _is_non_latin_letter(text[next_index]):
                next_index += 1
            tokens.append(text[index:next_index])
            index = next_index
            continue

        tokens.append(char)
        index += 1

    return tokens


def validate_corrected_text_hallucination(corrected_text: str, source_text: str) -> tuple[bool, str]:
    if not isinstance(corrected_text, str):
        return False, "corrected_text must be a string"
    if not corrected_text.strip():
        return True, ""
    if not isinstance(source_text, str):
        return False, "source_text must be a string"
    if not source_text.strip():
        return True, ""

    source_tokens = [
        token
        for token in _tokenize_for_hallucination(source_text)
        if token.strip() and not _is_punct_token(token)
    ]
    corrected_tokens = [
        token
        for token in _tokenize_for_hallucination(corrected_text)
        if token.strip() and not _is_punct_token(token)
    ]

    source_lexicon_folded = {token.casefold() for token in source_tokens}
    source_folded_stream = [token.casefold() for token in source_tokens]
    corrected_folded_stream = [token.casefold() for token in corrected_tokens]

    suspicious_tokens: list[str] = []
    matcher = difflib.SequenceMatcher(a=source_folded_stream, b=corrected_folded_stream, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "insert":
            continue
        for token in corrected_tokens[j1:j2]:
            if _is_hallucinated_token(token, source_lexicon_folded):
                suspicious_tokens.append(token)

    suspicious_insert_count = len(suspicious_tokens)
    many_new_tokens_threshold = max(2, math.ceil(len(source_tokens) * 0.12))
    if suspicious_insert_count >= many_new_tokens_threshold:
        unique = sorted(set(suspicious_tokens))[:8]
        return False, (
            "corrected_text contains too many potential hallucinated inserted token(s): "
            f"count={suspicious_insert_count}, threshold={many_new_tokens_threshold}, "
            f"examples={', '.join(unique)}"
        )

    return True, ""


def _is_unicode_cased_letter(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    return unicodedata.category(char) in {"Lu", "Ll", "Lt"}


def _leading_cased_word_span(text: str) -> tuple[str, int, int] | None:
    if not isinstance(text, str):
        return None

    stripped = text.lstrip()
    if not stripped:
        return None

    leading_ws_len = len(text) - len(stripped)
    if not _is_unicode_cased_letter(stripped[0]):
        return None

    index = 1
    while index < len(stripped):
        char = stripped[index]
        if _is_unicode_cased_letter(char):
            index += 1
            continue

        if char in "'’`.-":
            has_prev = index - 1 >= 0 and _is_unicode_cased_letter(stripped[index - 1])
            has_next = index + 1 < len(stripped) and _is_unicode_cased_letter(stripped[index + 1])
            if has_prev and has_next:
                index += 1
                continue

        break

    token = stripped[:index]
    start = leading_ws_len
    end = leading_ws_len + index
    return token, start, end


def _has_case_distinction(token: str) -> bool:
    if not isinstance(token, str):
        return False
    has_letter = any(char.isalpha() for char in token)
    return has_letter and token.lower() != token.upper()


def validate_first_token_casing_preserved(corrected_text: str, source_text: str) -> tuple[bool, str]:
    if not isinstance(corrected_text, str):
        return False, "corrected_text must be a string"
    if not isinstance(source_text, str):
        return False, "source_text must be a string"

    source_span = _leading_cased_word_span(source_text)
    corrected_span = _leading_cased_word_span(corrected_text)
    if source_span is None or corrected_span is None:
        return True, ""

    source_token, _, _ = source_span
    corrected_token, _, _ = corrected_span

    if source_token == corrected_token:
        return True, ""

    if (
        _has_case_distinction(source_token)
        and _has_case_distinction(corrected_token)
        and source_token.casefold() == corrected_token.casefold()
    ):
        return False, (
            "first token casing changed: "
            f"source='{source_token}' corrected='{corrected_token}'"
        )

    return True, ""


def _terminal_punctuation_char(text: str) -> str:
    if not isinstance(text, str):
        return ""
    trimmed = text.rstrip()
    if not trimmed:
        return ""
    tail = trimmed[-1]
    return tail if unicodedata.category(tail).startswith("P") else ""


def _lexical_tokens_for_terminal_punctuation(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"[^\W_]+(?:['’`.-][^\W_]+)*", text, flags=re.UNICODE)


def _last_lexical_token_for_terminal_punctuation(text: str) -> str | None:
    tokens = _lexical_tokens_for_terminal_punctuation(text)
    if not tokens:
        return None
    return tokens[-1]


def validate_terminal_punctuation_preserved(corrected_text: str, source_text: str) -> tuple[bool, str]:
    if not isinstance(corrected_text, str):
        return False, "corrected_text must be a string"
    if not isinstance(source_text, str):
        return False, "source_text must be a string"

    source_terminal = _terminal_punctuation_char(source_text)
    corrected_terminal = _terminal_punctuation_char(corrected_text)
    source_has_terminal = bool(source_terminal)
    corrected_has_terminal = bool(corrected_terminal)

    if not source_has_terminal and corrected_has_terminal:
        return False, (
            "terminal punctuation added: "
            f"source_terminal='{source_terminal}' corrected_terminal='{corrected_terminal}'"
        )
    if source_has_terminal and not corrected_has_terminal:
        return False, (
            "terminal punctuation removed: "
            f"source_terminal='{source_terminal}' corrected_terminal='{corrected_terminal}'"
        )

    if source_has_terminal and corrected_has_terminal and source_terminal != corrected_terminal:
        return False, (
            "terminal punctuation changed: "
            f"source_terminal='{source_terminal}' corrected_terminal='{corrected_terminal}'"
        )

    return True, ""


def preserve_terminal_punctuation(corrected_text: str, source_text: str) -> tuple[str, bool]:
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return corrected_text, False

    source_trimmed = source_text.rstrip()
    corrected_trimmed = corrected_text.rstrip()
    if not source_trimmed or not corrected_trimmed:
        return corrected_text, False

    source_terminal = _terminal_punctuation_char(source_text)
    corrected_terminal = _terminal_punctuation_char(corrected_text)
    source_has_terminal = bool(source_terminal)
    corrected_has_terminal = bool(corrected_terminal)

    if source_has_terminal != corrected_has_terminal:
        corrected_trailing_len = len(corrected_text) - len(corrected_trimmed)
        corrected_trailing = corrected_text[-corrected_trailing_len:] if corrected_trailing_len > 0 else ""

        corrected_base = corrected_trimmed
        while corrected_base and unicodedata.category(corrected_base[-1]).startswith("P"):
            corrected_base = corrected_base[:-1]

        replacement = corrected_base + (source_terminal if source_has_terminal else "")
        if replacement != corrected_trimmed:
            return replacement + corrected_trailing, True
        return corrected_text, False

    if source_terminal == corrected_terminal:
        return corrected_text, False

    corrected_trailing_len = len(corrected_text) - len(corrected_trimmed)
    corrected_trailing = corrected_text[-corrected_trailing_len:] if corrected_trailing_len > 0 else ""

    base = corrected_trimmed
    while base and unicodedata.category(base[-1]).startswith("P"):
        base = base[:-1]

    replacement = base + source_terminal if source_terminal else base
    return replacement + corrected_trailing, True


def preserve_first_token_casing(corrected_text: str, source_text: str) -> tuple[str, bool]:
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return corrected_text, False

    source_span = _leading_cased_word_span(source_text)
    corrected_span = _leading_cased_word_span(corrected_text)
    if source_span is None or corrected_span is None:
        return corrected_text, False

    source_token, _, _ = source_span
    corrected_token, corrected_start, corrected_end = corrected_span
    if source_token == corrected_token:
        return corrected_text, False

    if (
        _has_case_distinction(source_token)
        and _has_case_distinction(corrected_token)
        and source_token.casefold() == corrected_token.casefold()
    ):
        return corrected_text[:corrected_start] + source_token + corrected_text[corrected_end:], True

    return corrected_text, False


def _materialize_corrected_text(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload

    corrected_text = payload.get("corrected_text")
    if isinstance(corrected_text, str) and corrected_text.strip():
        return payload

    for key in ("ct_casing", "ct_punct", "ct_fix", "ct_combine"):
        value = payload.get(key)
        if isinstance(value, str):
            payload["corrected_text"] = value
            return payload

    payload["corrected_text"] = ""
    return payload


def validate_patch_payload(payload: dict) -> tuple[bool, str]:
    payload_keys = set(payload.keys())

    missing_required = sorted(_REQUIRED_TOP_LEVEL_KEYS - payload_keys)
    unexpected_keys = sorted(payload_keys - _REQUIRED_TOP_LEVEL_KEYS - _OPTIONAL_TOP_LEVEL_KEYS)
    if missing_required or unexpected_keys:
        return False, (
            "Top-level keys must be exactly: "
            "ct_casing "
            "(optional: tokenization, ct_combine, ct_fix, "
            "ct_punct, verification, "
            "machine_transcription_probability, corrected_text)"
        )

    if not isinstance(payload.get("ct_casing"), str):
        return False, "ct_casing must be a string"
    if not isinstance(payload.get("corrected_text", ""), str):
        return False, "corrected_text must be a string"

    return True, ""


def validate_output_payloads(payloads: list[dict]) -> tuple[bool, str]:
    for index, payload in enumerate(payloads, start=1):
        if not isinstance(payload, dict):
            return False, f"Output item {index} must be an object"

        if len(payload) == 0:
            continue

        is_valid, validation_error = validate_patch_payload(payload)
        if not is_valid:
            return False, f"Output item {index} schema error: {validation_error}"

    return True, ""


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


def is_json_like_repairable_response(content: str) -> tuple[bool, str]:
    if not isinstance(content, str):
        return False, "content is not a string"

    trimmed = content.strip()
    if not trimmed:
        return False, "content is empty after trim"

    lowered = trimmed.lower()
    if any(key in lowered for key in (
        '"tokenization"',
        '"ct_combine"',
        '"ct_fix"',
        '"ct_punct"',
        '"ct_casing"',
        '"verification"',
        '"machine_transcription_probability"',
    )):
        return True, "contains expected JSON schema keys"

    return False, (
        "does not look JSON-like/repairable"
    )


def non_repairable_prefix() -> str:
    return "NON_REPAIRABLE:"


def is_non_repairable_validation_error(validation_error: str | None) -> bool:
    if not isinstance(validation_error, str):
        return False
    return validation_error.startswith(non_repairable_prefix())


def _mark_non_repairable_validation_error(validation_error: str) -> str:
    return f"{non_repairable_prefix()} {validation_error}"


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
        is_json_like, json_like_reason = is_json_like_repairable_response(trimmed)
        if not is_json_like:
            return None, _mark_non_repairable_validation_error(
                f"{json_like_reason}; JSON parse error: {error}; raw content: {content[:10]} ..."
            )
        return None, f"JSON parse error: {error}"

    payload = _materialize_corrected_text(payload)

    is_valid, validation_error = validate_patch_payload(payload)
    if not is_valid:
        return None, f"Schema error: {validation_error}"

    return payload, None


def parse_validate_and_apply_text_fixes(
    raw_content: str,
    source_text: str,
    processing_id: str,
) -> tuple[dict | None, str | None, str]:
    content = raw_content.strip()
    if not content:
        return None, _mark_non_repairable_validation_error("Model returned empty output."), content

    payload, validation_error = parse_and_validate_json(content)
    if payload is None:
        return None, validation_error, content

    corrected_text = payload.get("corrected_text", "") if isinstance(payload, dict) else ""
    corrected_text, casing_normalized = preserve_first_token_casing(
        corrected_text,
        source_text,
    )
    corrected_text, punctuation_normalized = preserve_terminal_punctuation(
        corrected_text,
        source_text,
    )

    if (casing_normalized or punctuation_normalized) and isinstance(payload, dict):
        payload["corrected_text"] = corrected_text

    hallucination_ok, hallucination_error = validate_corrected_text_hallucination(
        corrected_text,
        source_text,
    )
    if not hallucination_ok:
        return None, _mark_non_repairable_validation_error(
            f"Hallucination check failed: {hallucination_error}"
        ), content

    punctuation_ok, punctuation_error = validate_terminal_punctuation_preserved(
        corrected_text,
        source_text,
    )
    if not punctuation_ok:
        print(
            "⚠️⚠️⚠️ WARNING [PUNCTUATION_VALIDATION]: "
            f"[{processing_id}] {punctuation_error}"
        )

    casing_ok, casing_error = validate_first_token_casing_preserved(
        corrected_text,
        source_text,
    )
    if not casing_ok:
        print(
            "⚠️⚠️⚠️ WARNING [CASING_VALIDATION]: "
            f"[{processing_id}] {casing_error}"
        )

    return payload, None, content


def build_empty_payload() -> dict:
    return {
        # "ct_combine": "",
        # "ct_fix": "",
        # "ct_punct": "",
        "ct_casing": "",
        # "corrected_text": "",
        # "verification": {
        #     "op_details": [
        #         "TOKEN_COMBINE_RESULT:",
        #         "ERROR_FIX_RESULT:",
        #         "PUNCTUATION_RESULT:",
        #         "CASING_RESULT:",
        #     ],
        # },
        # "machine_transcription_probability": 0.0,
    }


def describe_top_level_key_error(
    validation_error: str | None,
    raw_output: str,
    processing_id: str | None = None,
) -> str | None:
    prefix = f"[{processing_id}] " if processing_id else ""
    if not validation_error or "Top-level keys must be exactly:" not in validation_error:
        return None

    trimmed = raw_output.strip()
    candidate = extract_first_json_object(trimmed) or trimmed

    try:
        payload = json.loads(candidate)
    except Exception:
        return f"{prefix}JSON_TOP_LEVEL_KEY_ERROR: unable to parse payload object for key diff"

    if not isinstance(payload, dict):
        return f"{prefix}JSON_TOP_LEVEL_KEY_ERROR: top-level payload is not an object"

    allowed_keys = _REQUIRED_TOP_LEVEL_KEYS | _OPTIONAL_TOP_LEVEL_KEYS
    payload_keys = {str(key) for key in payload.keys()}
    missing_keys = sorted(_REQUIRED_TOP_LEVEL_KEYS - payload_keys)
    unexpected_keys = sorted(payload_keys - allowed_keys)
    return (
        f"{prefix}JSON_TOP_LEVEL_KEY_ERROR: "
        f"missing_keys={missing_keys}; unexpected_keys={unexpected_keys}"
    )


def log_json_validation_with_key_error(
    validation_error: str | None,
    raw_output: str,
    processing_id: str | None = None,
) -> str | None:
    prefix = f"[{processing_id}] " if processing_id else ""
    if validation_error:
        print(f"{prefix}JSON_FORMAT_ERROR: {validation_error}")

    key_error_detail = describe_top_level_key_error(
        validation_error,
        raw_output,
        processing_id,
    )
    if key_error_detail:
        print(key_error_detail)
    return key_error_detail


def handle_invalid_repair_json_result(
    empty_attempt: int,
    empty_result_retries: int,
    validation_error: str | None,
    repaired_content: str,
    processing_id: str | None = None,
) -> bool:
    prefix = f"[{processing_id}] " if processing_id else ""
    repaired_key_error_detail = log_json_validation_with_key_error(
        validation_error,
        repaired_content,
        processing_id,
    )

    should_retry = should_retry_after_failure(
        empty_attempt,
        empty_result_retries,
        "Repair attempt did not return valid JSON.",
        f"Repair attempt did not return valid JSON ({validation_error}).",
        processing_id,
    )
    if should_retry:
        return True

    print(f"{prefix}{validation_error}")
    if not repaired_key_error_detail:
        print(f"{prefix}Raw output:")
        print(repaired_content)
    return False


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


def resolve_patch_and_repair_template_paths(
    patch_override_value: str | None,
    repair_override_value: str | None,
) -> tuple[Path | None, Path | None, str | None]:
    prompt_template_path, prompt_path_error = resolve_required_template_path(
        patch_override_value,
        "prompt_patch.txt",
        "Patch prompt",
    )
    if prompt_path_error:
        return None, None, prompt_path_error

    repair_prompt_template_path, repair_path_error = resolve_required_template_path(
        repair_override_value,
        "prompt_repair.txt",
        "Repair prompt",
    )
    if repair_path_error:
        return None, None, repair_path_error

    return prompt_template_path, repair_prompt_template_path, None


def print_common_runtime_settings(
    prompt_template_path: Path,
    repair_prompt_template_path: Path,
    concurrency: int,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
) -> None:
    print(f"Using patch prompt file: {prompt_template_path}")
    print(f"Using repair prompt file: {repair_prompt_template_path}")
    print(f"Concurrency: {concurrency}")
    print(f"Timeout: {timeout_seconds}s, retries: {timeout_retries}")
    print(f"Empty-result retries: {empty_result_retries}")


def print_processing_progress(index: int, total: int) -> None:
    if total > 1:
        print(f"Processing transcription {index}/{total}...")


async def run_indexed_tasks(
    items: list[str],
    process_item: Callable[[int, str], Awaitable[None]],
) -> None:
    tasks = [
        asyncio.create_task(process_item(index, item))
        for index, item in enumerate(items, start=1)
    ]
    await asyncio.gather(*tasks)


async def run_transcriptions_with_concurrency(
    transcriptions: list[str],
    concurrency: int,
    process_item: Callable[[int, str, int], Awaitable[None]],
) -> int:
    total = len(transcriptions)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_item_guarded(index: int, transcription: str) -> None:
        async with semaphore:
            print_processing_progress(index, total)
            await process_item(index, transcription, total)

    await run_indexed_tasks(transcriptions, process_item_guarded)
    return total


def parse_transcriptions_from_file(input_path: Path) -> list[str]:
    raw_text = input_path.read_text(encoding="utf-8")
    if raw_text == "":
        return []

    stripped = raw_text.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        transcriptions = [item.strip() for item in parsed if isinstance(item, str)]
        return transcriptions

    if isinstance(parsed, dict):
        items = parsed.get("transcriptions")
        if isinstance(items, list):
            transcriptions = [item.strip() for item in items if isinstance(item, str)]
            return transcriptions

    return [line.strip() for line in raw_text.rstrip().split("\n")]

def collect_transcriptions_from_input(input_file_value: str | None) -> list[str] | None:
    transcriptions: list[str]
    if input_file_value:
        input_path = resolve_path(input_file_value)
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            return None
        transcriptions = parse_transcriptions_from_file(input_path)
        print(f"Read {len(transcriptions)} transcription(s) from: {input_path}")
    else:
        transcription = input("Enter transcription: ").strip()
        transcriptions = [transcription] if transcription else []

    if not transcriptions:
        print("No transcription provided.")
        return None

    return transcriptions

def assign_payload_or_emit_empty(
    payload: dict | None,
    payloads: list[dict | None],
    slot: int,
    index: int,
    total: int,
) -> bool:
    if payload is None:
        payloads[slot] = build_empty_payload()
        print(
            f"Failed on transcription {index}/{total}; "
            f"emitting empty payload."
        )
        return False

    is_valid, validation_error = validate_patch_payload(payload)
    if not is_valid:
        payloads[slot] = build_empty_payload()
        print(
            f"Output validation failed on transcription {index}/{total}: "
            f"{validation_error}. Emitting empty payload."
        )
        return False

    payloads[slot] = payload
    return True

def finalize_payloads_and_write(
    payloads: list[dict | None],
    output_file_value: str | None,
) -> bool:
    if any(payload is None for payload in payloads):
        print("Failed to produce output for one or more transcriptions.")
        return False

    final_payloads = [payload for payload in payloads if payload is not None]

    is_valid, validation_error = validate_output_payloads(final_payloads)
    if not is_valid:
        print(validation_error)
        return False

    write_output_artifacts(final_payloads, output_file_value)
    return True


def strip_prompt_comments(prompt_text: str) -> str:
    filtered_lines: list[str] = []
    for line in prompt_text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def load_prompt_template(template_path: Path) -> str:
    return strip_prompt_comments(template_path.read_text(encoding="utf-8"))


def load_patch_and_repair_templates(
    prompt_template_path: Path,
    repair_prompt_template_path: Path,
) -> tuple[str, str]:
    return (
        load_prompt_template(prompt_template_path),
        load_prompt_template(repair_prompt_template_path),
    )


def write_output_artifacts(
    payloads: list[dict],
    output_file_value: str | None,
) -> None:
    sanitized_payloads: list[dict] = []
    for item in payloads:
        if isinstance(item, dict):
            payload_copy = dict(item)
            payload_copy.pop("tokenization", None)
            verification = payload_copy.get("verification")
            if isinstance(verification, dict):
                verification_copy = dict(verification)
                payload_copy["verification"] = verification_copy
            sanitized_payloads.append(payload_copy)

    output_payload = sanitized_payloads[0] if len(sanitized_payloads) == 1 else sanitized_payloads

    corrected_text_lines: list[str] = []
    for item in sanitized_payloads:
        corrected_text = item.get("corrected_text") if isinstance(item, dict) else ""
        corrected_text_lines.append(corrected_text if isinstance(corrected_text, str) else "")
    text_output_content = "\n".join(corrected_text_lines)

    output_content = json.dumps(output_payload, ensure_ascii=False, indent=2)
    if output_file_value:
        output_path = resolve_path(output_file_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_content, encoding="utf-8")
        print(f"Wrote result to: {output_path}")

        output_text_path = output_path.with_suffix(".txt")

        output_text_path.parent.mkdir(parents=True, exist_ok=True)
        output_text_path.write_text(text_output_content, encoding="utf-8")
        print(f"Wrote text output file to: {output_text_path}")
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


def print_timeout_and_retry_guidance(
    timeout_seconds: float,
    timeout_retries: int,
    processing_id: str,
) -> None:
    print(
        f"[{processing_id}] Request timed out after {timeout_seconds}s "
        f"for {timeout_retries + 1} attempt(s)."
    )
    print(f"[{processing_id}] Increase TIMEOUT or TIMEOUT_RETRIES and try again.")


def build_repair_prompt_after_invalid_json(
    repair_prompt_template: str,
    validation_error: str | None,
    previous_output: str,
    processing_id: str | None = None,
) -> str:
    prefix = f"[{processing_id}] " if processing_id else ""
    print(f"{prefix}Initial response was not valid strict JSON. Running one repair attempt...")
    return format_repair_prompt(
        repair_prompt_template,
        validation_error,
        previous_output,
    )


def apply_corrected_text_fallback(payload: dict, transcription: str) -> dict | None:
    if not isinstance(payload, dict):
        return None

    if not isinstance(payload.get("tokenization"), dict):
        payload["tokenization"] = {"tokens": []}
    else:
        tokenization = payload["tokenization"]
        if not isinstance(tokenization.get("tokens"), list):
            tokenization["tokens"] = []

    if not isinstance(payload.get("ct_combine"), str):
        payload["ct_combine"] = ""
    if not isinstance(payload.get("ct_fix"), str):
        payload["ct_fix"] = ""
    if not isinstance(payload.get("ct_punct"), str):
        payload["ct_punct"] = ""
    if not isinstance(payload.get("ct_casing"), str):
        payload["ct_casing"] = ""

    verification = payload.get("verification")
    if not isinstance(verification, dict):
        verification = {}
        payload["verification"] = verification
    verification["op_details"] = [
        "TOKEN_COMBINE_RESULT:",
        "ERROR_FIX_RESULT:",
        "PUNCTUATION_RESULT:",
        "CASING_RESULT:",
    ]

    machine_probability = payload.get("machine_transcription_probability")
    if not isinstance(machine_probability, (int, float)):
        payload["machine_transcription_probability"] = 0.0
    elif machine_probability < 0 or machine_probability > 1:
        payload["machine_transcription_probability"] = 0.0

    payload["corrected_text"] = ""
    return payload
