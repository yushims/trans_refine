import asyncio
import argparse
import ast
import csv
import difflib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path


_STEP_CHAIN_KEYS = (
    "ct_speaker",
    "ct_combine",
    "ct_lexical",
    "ct_disfluency",
    "ct_format",
    "ct_numeral",
    "ct_punct",
    "ct_casing",
    "ct_remain_fix",
)

_REQUIRED_TOP_LEVEL_KEY_ORDER = (
    "tokenization",
    "translation",
    "aggressiveness_level",
    "speaker_scope",
    "ct_speaker",
    "ct_combine",
    "no_touch_tokens",
    *(
        step_key
        for step_key in _STEP_CHAIN_KEYS
        if step_key not in {"ct_speaker", "ct_combine"}
    ),
)

_REQUIRED_TOP_LEVEL_KEYS = set(_REQUIRED_TOP_LEVEL_KEY_ORDER)

_OPTIONAL_TOP_LEVEL_KEY_ORDER = (
    "source_filename",
    "source_text",
    "corrected_text",
)

_OPTIONAL_TOP_LEVEL_KEYS = set(_OPTIONAL_TOP_LEVEL_KEY_ORDER)


DEFAULT_CONCURRENCY = 10
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_TIMEOUT_RETRIES = 2
DEFAULT_EMPTY_RESULT_RETRIES = 2
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_RETRY_TEMPERATURE_JITTER = 0.08
DEFAULT_RETRY_TOP_P_JITTER = 0.03
DEFAULT_MODEL_MISMATCH_RETRIES = 2
DEFAULT_AOAI_ENDPOINT = "https://adaptationdev-resource.openai.azure.com/"
DEFAULT_AOAI_API_VERSION = "2025-01-01-preview"
DEFAULT_AOAI_DEPLOYMENT = "gpt-5-chat"
DEFAULT_COPILOT_MODEL = "gpt-5.2"


def add_common_runtime_cli_arguments(
    parser: argparse.ArgumentParser,
    timeout_default: float = DEFAULT_TIMEOUT_SECONDS,
    concurrency_default: int = DEFAULT_CONCURRENCY,
    timeout_retries_default: int = DEFAULT_TIMEOUT_RETRIES,
    empty_result_retries_default: int = DEFAULT_EMPTY_RESULT_RETRIES,
) -> None:
    parser.add_argument("--concurrency", dest="concurrency", type=int, default=concurrency_default)
    parser.add_argument("--timeout", dest="timeout", type=float, default=timeout_default)
    parser.add_argument("--timeout-retries", dest="timeout_retries", type=int, default=timeout_retries_default)
    parser.add_argument(
        "--empty-result-retries",
        dest="empty_result_retries",
        type=int,
        default=empty_result_retries_default,
    )


def add_chain_steps_cli_argument(
    parser: argparse.ArgumentParser,
    default: str = "ALL",
) -> None:
    parser.add_argument("--chain-steps", dest="chain_steps", default=default)


def add_aoai_sampling_cli_arguments(
    parser: argparse.ArgumentParser,
    temperature_default: float = DEFAULT_TEMPERATURE,
    top_p_default: float = DEFAULT_TOP_P,
    retry_temperature_jitter_default: float = DEFAULT_RETRY_TEMPERATURE_JITTER,
    retry_top_p_jitter_default: float = DEFAULT_RETRY_TOP_P_JITTER,
) -> None:
    parser.add_argument("--temperature", dest="temperature", type=float, default=temperature_default)
    parser.add_argument("--top-p", dest="top_p", type=float, default=top_p_default)
    parser.add_argument(
        "--retry-temperature-jitter",
        dest="retry_temperature_jitter",
        type=float,
        default=retry_temperature_jitter_default,
    )
    parser.add_argument(
        "--retry-top-p-jitter",
        dest="retry_top_p_jitter",
        type=float,
        default=retry_top_p_jitter_default,
    )


def add_model_mismatch_retries_cli_argument(
    parser: argparse.ArgumentParser,
    default: int = DEFAULT_MODEL_MISMATCH_RETRIES,
) -> None:
    parser.add_argument("--model-mismatch-retries", dest="model_mismatch_retries", type=int, default=default)


def _new_empty_step_payload() -> dict:
    return {"edits": [], "result": ""}


def _step_field_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string"},
                },
            },
            "result": {"type": "string"},
        },
        "required": ["edits", "result"],
    }


def build_patch_payload_schema() -> dict:
    properties: dict[str, dict] = {
        "tokenization": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tokens": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["tokens"],
        },
        "translation": {
            "type": "string",
        },
        "aggressiveness_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "speaker_scope": {
            "type": "string",
            "enum": ["single", "multi", "unknown"],
        },
    }

    # Keep field order stable for readability/debugging: ct_speaker, ct_combine, no_touch_tokens, then remaining steps.
    properties["ct_speaker"] = _step_field_schema()
    properties["ct_combine"] = _step_field_schema()
    properties["no_touch_tokens"] = {
        "type": "array",
        "items": {"type": "string"},
    }
    for step_key in _STEP_CHAIN_KEYS:
        if step_key in {"ct_speaker", "ct_combine"}:
            continue
        properties[step_key] = _step_field_schema()

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(_REQUIRED_TOP_LEVEL_KEY_ORDER),
    }


def build_patch_response_format_schema() -> dict:
    return {
        "name": "deterministic_patch_output",
        "strict": True,
        "schema": build_patch_payload_schema(),
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
# Common in-word connectors across multilingual text:
# apostrophe variants, dot, middle dots, and dash variants.
_WORD_CONNECTOR_CHARS = "'’`.·・ʻʼ‐‑‒–—―-"
_WORD_CONNECTOR_CHAR_CLASS = re.escape(_WORD_CONNECTOR_CHARS)
_WORD_CONNECTOR_SET = set(_WORD_CONNECTOR_CHARS)
_NUMBER_LIKE_PATTERN = re.compile(
    rf"\d+(?:[.,:٫٬]\d+)*(?:%|{_WORD_SEGMENT_PATTERN})?",
    flags=re.UNICODE,
)
_NUMBER_CORE_PATTERN = re.compile(
    r"\d+(?:[.,:٫٬]\d+)*",
    flags=re.UNICODE,
)
_LATIN_WORD_PATTERN = re.compile(
    rf"{_WORD_SEGMENT_PATTERN}(?:[{_WORD_CONNECTOR_CHAR_CLASS}]{_WORD_SEGMENT_PATTERN})*",
    flags=re.UNICODE,
)
_SENTENCE_END_PUNCTUATION = {
    ".",
    "!",
    "?",
    "。",
    "！",
    "？",
    "؟",
    "۔",
    "।",
    "॥",
    "։",
    "።",
}
_SENTENCE_CLOSERS = {
    '"',
    "'",
    "”",
    "’",
    "»",
    "›",
    "」",
    "』",
    "》",
    "】",
    "〉",
    "）",
    "］",
    "｝",
    ")",
    "]",
    "}",
}
_OPENING_PUNCT_TOKENS = {
    "(",
    "[",
    "{",
    "<",
    '"',
    "'",
    "“",
    "‘",
    "«",
    "‹",
    "「",
    "『",
    "《",
    "【",
    "〈",
    "（",
    "［",
    "｛",
}
_CHAR_BASED_NO_SPACE_GROUPS = {
    "han",
    "kana",
    "hangul",
    "thai",
    "lao",
    "khmer",
    "myanmar",
}
_DOT_ABBREVIATION_WORD_PATTERN = re.compile(
    r"\b(?:mr|mrs|ms|dr|prof|sr|jr|st|etc|vs|fig|no)\.$",
    flags=re.IGNORECASE,
)
_DOT_ABBREVIATION_COMPOUND_PATTERN = re.compile(
    r"\b(?:e\.g|i\.e|a\.k\.a|u\.s|u\.k)\.$",
    flags=re.IGNORECASE,
)


def _is_hangul_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return 0xAC00 <= codepoint <= 0xD7AF


def _is_kana_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xFF66 <= codepoint <= 0xFF9D
        or 0x1B000 <= codepoint <= 0x1B16F
    )


def _is_han_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return (
        (0x3400 <= codepoint <= 0x4DBF)
        or (0x4E00 <= codepoint <= 0x9FFF)
        or (0xF900 <= codepoint <= 0xFAFF)
        or (0x20000 <= codepoint <= 0x2A6DF)
        or (0x2A700 <= codepoint <= 0x2B73F)
        or (0x2B740 <= codepoint <= 0x2B81F)
        or (0x2B820 <= codepoint <= 0x2CEAF)
        or (0x2CEB0 <= codepoint <= 0x2EBEF)
        or (0x30000 <= codepoint <= 0x3134F)
        or (0x2F800 <= codepoint <= 0x2FA1F)
    )


def _is_thai_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return 0x0E00 <= codepoint <= 0x0E7F


def _is_lao_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return 0x0E80 <= codepoint <= 0x0EFF


def _is_khmer_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return 0x1780 <= codepoint <= 0x17FF


def _is_myanmar_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return (
        (0x1000 <= codepoint <= 0x109F)
        or (0xAA60 <= codepoint <= 0xAA7F)
        or (0xA9E0 <= codepoint <= 0xA9FF)
    )


def _char_based_script_group(char: str) -> str | None:
    # Group name is used to ensure spacing cleanup is same-script only.
    if _is_han_char(char):
        return "han"
    if _is_kana_char(char):
        return "kana"
    if _is_hangul_char(char):
        return "hangul"
    if _is_thai_char(char):
        return "thai"
    if _is_lao_char(char):
        return "lao"
    if _is_khmer_char(char):
        return "khmer"
    if _is_myanmar_char(char):
        return "myanmar"
    return None


def _is_inner_word_connector(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    # Include explicit connector set plus all Unicode dash punctuation.
    return char in _WORD_CONNECTOR_SET or unicodedata.category(char) == "Pd"


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


def extract_retry_after_seconds(error: Exception) -> float | None:
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
        return apply_corrected_text_fallback(payload), False

    print(
        f"{prefix}Model returned empty corrected_text. "
        f"Retrying item {empty_attempt + 2}/{empty_result_retries + 1}..."
    )
    return None, True


def _is_punct_token(token: str) -> bool:
    return isinstance(token, str) and len(token) == 1 and unicodedata.category(token).startswith("P")


def _is_cjk_char(char: str) -> bool:
    return _is_han_char(char) or _is_kana_char(char) or _is_hangul_char(char)


def _is_cjk_token(token: str) -> bool:
    return bool(token) and all(_is_cjk_char(char) for char in token)


def _is_non_latin_letter(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    if not char.isalpha():
        return False
    return "LATIN" not in unicodedata.name(char, "")


def _is_non_latin_word_char(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    if _is_non_latin_letter(char):
        return True
    # Keep combining marks attached to the preceding non-Latin base letter.
    return unicodedata.category(char) in {"Mn", "Mc", "Me"}


def _contains_latin_letter(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    for char in text:
        if char.isalpha() and "LATIN" in unicodedata.name(char, ""):
            return True
    return False


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


_COMPACT_SPEAKER_LABEL_PATTERN = re.compile(
    r"^(?:spk|speaker|spkr)[_-]?\d+$",
    flags=re.IGNORECASE,
)
_FREEFORM_SPEAKER_LABEL_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_-]{0,15}$",
)


def _is_colon_token(token: str) -> bool:
    return token in {":", "："}


def _normalize_model_result_newlines(text: str, source_text: str) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove only model-inserted line breaks before likely speaker labels.
    normalized = re.sub(
        r"\n\s*(?=(?:\[[^\]\r\n]{1,24}\]|(?:spk|speaker|spkr)[_-]?\d+|[A-Za-z][A-Za-z0-9_-]{0,15})\s*[:\uff1a])",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )

    source_has_newline = isinstance(source_text, str) and ("\n" in source_text or "\r" in source_text)
    if not source_has_newline:
        # Keep single-line inputs single-line in model output.
        normalized = re.sub(r"\s*\n\s*", " ", normalized)
        normalized = re.sub(r" {2,}", " ", normalized)
        return normalized.strip()

    return normalized


def normalize_payload_result_newlines(payload: dict, source_text: str) -> None:
    if not isinstance(payload, dict):
        return

    for step_key in _STEP_CHAIN_KEYS:
        step_payload = payload.get(step_key)
        if not isinstance(step_payload, dict):
            continue
        step_result = step_payload.get("result")
        if isinstance(step_result, str):
            step_payload["result"] = _normalize_model_result_newlines(step_result, source_text)

    corrected_text = payload.get("corrected_text")
    if isinstance(corrected_text, str):
        payload["corrected_text"] = _normalize_model_result_newlines(corrected_text, source_text)


def _looks_like_speaker_label_token(token: str) -> bool:
    if not isinstance(token, str):
        return False

    stripped = token.strip()
    if not stripped:
        return False

    def _is_identifier_char(char: str) -> bool:
        if char.isalnum() or char in {"_", "-"}:
            return True
        return unicodedata.category(char) in {"Mn", "Mc", "Me"}

    if _COMPACT_SPEAKER_LABEL_PATTERN.match(stripped):
        return True

    if not _FREEFORM_SPEAKER_LABEL_PATTERN.match(stripped):
        if not all(_is_identifier_char(char) for char in stripped):
            return False

        has_letter = any(char.isalpha() for char in stripped)
        has_non_latin_letter = any(
            char.isalpha() and "LATIN" not in unicodedata.name(char, "")
            for char in stripped
        )
        if has_letter and has_non_latin_letter and len(stripped) <= 12:
            return True

        # Allow multilingual Latin-with-diacritics speaker names like "José:".
        has_non_ascii = any(ord(char) > 127 for char in stripped)
        has_digit = any(char.isdigit() for char in stripped)
        if has_letter and has_non_ascii and len(stripped) <= 16 and (has_digit or stripped[0].isupper()):
            return True

        return False

    # Accept speaker-like identifiers such as LWX, S1, USER_2.
    has_digit = any(char.isdigit() for char in stripped)
    return stripped.isupper() or has_digit


def _is_single_char_letter_token(token: str) -> bool:
    if not isinstance(token, str) or len(token) != 1:
        return False
    category = unicodedata.category(token)
    return (
        token.isalpha()
        or category in {"Mn", "Mc", "Me"}
    )


def _normalize_speaker_label_signature(parts: list[str]) -> tuple[str, ...]:
    normalized_parts: list[str] = []
    for part in parts:
        if not isinstance(part, str):
            continue
        stripped = part.strip()
        if stripped:
            normalized_parts.append(stripped.casefold())
    return tuple(normalized_parts)


def _is_high_confidence_speaker_signature(signature: tuple[str, ...]) -> bool:
    if not signature:
        return False

    text = "".join(signature)
    if not text:
        return False

    if _COMPACT_SPEAKER_LABEL_PATTERN.match(text):
        return True

    if text in {"speaker", "spk", "spkr"}:
        return True

    # Numeric/symbolic ids are usually deliberate speaker markers (S1, USER_2, host-1).
    if any(char.isdigit() or char in {"_", "-"} for char in text):
        return True

    return False


def _match_speaker_label_span(tokens: list[str], index: int) -> tuple[int, tuple[str, ...]] | None:
    if index < 0 or index >= len(tokens):
        return None

    token = tokens[index]
    if not isinstance(token, str):
        return None

    next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
    next_next_token = tokens[index + 2] if index + 2 < len(tokens) else ""
    next_third_token = tokens[index + 3] if index + 3 < len(tokens) else ""
    next_fourth_token = tokens[index + 4] if index + 4 < len(tokens) else ""

    # Canonical bracketed form: [LABEL]:
    if (
        token == "["
        and isinstance(next_token, str)
        and next_token
        and next_next_token == "]"
        and _is_colon_token(next_third_token)
        and _looks_like_speaker_label_token(next_token)
    ):
        signature = _normalize_speaker_label_signature([next_token])
        return (index + 4, signature) if signature else None

    # Canonical bracketed split form: [SPK1]: tokenized as [, SPK, 1, ], :
    folded = token.casefold()
    if (
        token == "["
        and isinstance(next_token, str)
        and next_token.casefold() in {"speaker", "spk", "spkr"}
        and isinstance(next_next_token, str)
        and next_next_token.isdigit()
        and next_third_token == "]"
        and _is_colon_token(next_fourth_token)
    ):
        signature = _normalize_speaker_label_signature([f"{next_token}{next_next_token}"])
        return (index + 5, signature) if signature else None

    # Handle freeform labels such as "LWX:".
    if _is_colon_token(next_token) and _looks_like_speaker_label_token(token):
        signature = _normalize_speaker_label_signature([token])
        return (index + 2, signature) if signature else None

    # Handle multilingual labels tokenized into multiple single-char letter tokens,
    # e.g. "王小明:", "Аня:", "علي:".
    if _is_single_char_letter_token(token):
        end = index
        while end < len(tokens) and _is_single_char_letter_token(tokens[end]) and end - index < 12:
            end += 1
        if end > index + 1 and end < len(tokens) and _is_colon_token(tokens[end]):
            label_text = "".join(tokens[index:end])
            signature = _normalize_speaker_label_signature([label_text])
            return (end + 1, signature) if signature else None

    # Handle plain labels tokenized as "speaker" + "1" + ":".
    if folded in {"speaker", "spk", "spkr"}:
        if (
            isinstance(next_token, str)
            and next_token.isdigit()
            and _is_colon_token(next_next_token)
        ):
            signature = _normalize_speaker_label_signature([f"{token}{next_token}"])
            return (index + 3, signature) if signature else None

    return None


def _collect_and_remove_speaker_labels(tokens: list[str]) -> tuple[list[str], set[tuple[str, ...]]]:
    candidate_spans: list[tuple[int, int, tuple[str, ...]]] = []
    index = 0

    while index < len(tokens):
        matched = _match_speaker_label_span(tokens, index)
        if matched is None:
            index += 1
            continue

        end, signature = matched
        if signature:
            candidate_spans.append((index, end, signature))
        index = end

    # Be conservative for multilingual freeform labels: only strip low-confidence labels
    # when the text clearly looks multi-speaker (2+ label spans) or signature repeats.
    signature_counts = Counter(signature for _, _, signature in candidate_spans)
    total_candidates = len(candidate_spans)

    filtered: list[str] = []
    collected_signatures: set[tuple[str, ...]] = set()
    span_by_start = {
        start: (end, signature)
        for start, end, signature in candidate_spans
    }

    index = 0
    while index < len(tokens):
        span = span_by_start.get(index)
        if span is None:
            filtered.append(tokens[index])
            index += 1
            continue

        end, signature = span
        should_strip = (
            _is_high_confidence_speaker_signature(signature)
            or signature_counts[signature] >= 2
            or total_candidates >= 2
        )
        if should_strip:
            collected_signatures.add(signature)
            index = end
            continue

        filtered.extend(tokens[index:end])
        index = end

    return filtered, collected_signatures


def _remove_speaker_labels_by_reference(
    tokens: list[str],
    reference_signatures: set[tuple[str, ...]],
) -> list[str]:
    if not tokens or not reference_signatures:
        return tokens

    filtered: list[str] = []
    index = 0

    while index < len(tokens):
        matched = _match_speaker_label_span(tokens, index)
        if matched is None:
            filtered.append(tokens[index])
            index += 1
            continue

        end, signature = matched
        if signature in reference_signatures:
            index = end
            continue

        filtered.append(tokens[index])
        index += 1

    return filtered


def _remove_speaker_label_tokens(tokens: list[str]) -> list[str]:
    filtered, _ = _collect_and_remove_speaker_labels(tokens)
    return filtered


def _is_reasonable_number_like_token(token: str) -> bool:
    if not isinstance(token, str) or not token:
        return False

    core_match = _NUMBER_CORE_PATTERN.match(token)
    if core_match is None:
        return False
    if core_match.end() == len(token):
        return True

    suffix = token[core_match.end():]
    if suffix == "%":
        return True

    # Keep very short unit-like suffixes (for example 年, 月, kg, ms).
    letter_count = sum(1 for char in suffix if char.isalpha())
    if letter_count == 0:
        return True
    if letter_count <= 2:
        return True

    # Keep longer short units script-agnostic to avoid ASCII-biased behavior.
    if letter_count <= 4:
        return True

    # Otherwise treat as over-greedy and fall back to numeric core only.
    return False


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
            if _is_reasonable_number_like_token(token):
                tokens.append(token)
                index += len(token)
                continue

            core_match = _NUMBER_CORE_PATTERN.match(text, index)
            if core_match:
                core_token = core_match.group(0)
                tokens.append(core_token)
                index += len(core_token)
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

        script_group = _char_based_script_group(char)
        if script_group in {"thai", "lao", "khmer", "myanmar"}:
            next_index = index + 1
            while (
                next_index < length
                and _char_based_script_group(text[next_index]) == script_group
            ):
                next_index += 1
            tokens.append(text[index:next_index])
            index = next_index
            continue

        latin_match = _LATIN_WORD_PATTERN.match(text, index)
        if latin_match and _contains_latin_letter(latin_match.group(0)):
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
            while next_index < length and _is_non_latin_word_char(text[next_index]):
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
        return False, ""

    source_tokens_raw = _tokenize_for_hallucination(source_text)
    corrected_tokens_raw = _tokenize_for_hallucination(corrected_text)

    corrected_tokens, corrected_speaker_label_signatures = _collect_and_remove_speaker_labels(
        corrected_tokens_raw
    )
    source_tokens = _remove_speaker_labels_by_reference(
        source_tokens_raw,
        corrected_speaker_label_signatures,
    )

    source_tokens = [token for token in source_tokens if token.strip() and not _is_punct_token(token)]
    corrected_tokens = [token for token in corrected_tokens if token.strip() and not _is_punct_token(token)]

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

    source_token_count = len(source_tokens)
    corrected_token_count = len(corrected_tokens)
    if source_token_count > 0:
        estimated_deleted_tokens = max(0, source_token_count - corrected_token_count)
        large_deletion_threshold = max(4, math.ceil(source_token_count * 0.35))
        retained_ratio = corrected_token_count / source_token_count

        # Flag only significant shrinkage to avoid penalizing small local cleanup.
        if estimated_deleted_tokens >= large_deletion_threshold and retained_ratio < 0.60:
            return False, (
                "corrected_text appears to delete too much source content: "
                f"source_tokens={source_token_count}, corrected_tokens={corrected_token_count}, "
                f"deleted={estimated_deleted_tokens}, threshold={large_deletion_threshold}, "
                f"retained_ratio={retained_ratio:.3f}"
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

        if _is_inner_word_connector(char):
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


def _cased_word_span_at(text: str, start: int) -> tuple[str, int, int] | None:
    if not isinstance(text, str):
        return None
    if not isinstance(start, int) or start < 0 or start >= len(text):
        return None
    if not _is_unicode_cased_letter(text[start]):
        return None

    index = start + 1
    while index < len(text):
        char = text[index]
        if _is_unicode_cased_letter(char):
            index += 1
            continue

        if _is_inner_word_connector(char):
            has_prev = index - 1 >= start and _is_unicode_cased_letter(text[index - 1])
            has_next = index + 1 < len(text) and _is_unicode_cased_letter(text[index + 1])
            if has_prev and has_next:
                index += 1
                continue

        break

    return text[start:index], start, index


def _has_case_distinction(token: str) -> bool:
    if not isinstance(token, str):
        return False
    has_letter = any(char.isalpha() for char in token)
    return has_letter and token.lower() != token.upper()


def _is_first_token_in_no_touch_entities(
    source_token: str,
    corrected_token: str,
    no_touch_tokens: object,
) -> bool:
    if not isinstance(no_touch_tokens, list):
        return False

    token_candidates = [token for token in (source_token, corrected_token) if isinstance(token, str) and token]
    if not token_candidates:
        return False

    for entity_span in no_touch_tokens:
        if not isinstance(entity_span, str):
            continue
        span = entity_span.strip()
        if not span:
            continue

        for token in token_candidates:
            if span.casefold() == token.casefold():
                return True

            # Also treat multi-token protected spans that start with this first token as protected.
            if re.match(rf"^{re.escape(token)}(?:\b|\s|$)", span, flags=re.IGNORECASE):
                return True

    return False


def is_all_uppercase_cased_input(text: str) -> bool:
    if not isinstance(text, str):
        return False

    stripped = text.strip()
    if not stripped:
        return False

    cased_letter_count = 0
    uppercase_cased_letter_count = 0
    for char in stripped:
        if not _is_unicode_cased_letter(char):
            continue
        cased_letter_count += 1
        if char == char.upper() and char != char.lower():
            uppercase_cased_letter_count += 1

    return cased_letter_count > 0 and uppercase_cased_letter_count == cased_letter_count


def is_all_lowercase_cased_input(text: str) -> bool:
    if not isinstance(text, str):
        return False

    stripped = text.strip()
    if not stripped:
        return False

    cased_letter_count = 0
    lowercase_cased_letter_count = 0
    for char in stripped:
        if not _is_unicode_cased_letter(char):
            continue
        cased_letter_count += 1
        if char == char.lower() and char != char.upper():
            lowercase_cased_letter_count += 1

    return cased_letter_count > 0 and lowercase_cased_letter_count == cased_letter_count


def _terminal_punctuation_char(text: str) -> str:
    if not isinstance(text, str):
        return ""
    trimmed = text.rstrip()
    if not trimmed:
        return ""
    tail = trimmed[-1]
    return tail if unicodedata.category(tail).startswith("P") else ""


def _contains_any_punctuation(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return any(unicodedata.category(char).startswith("P") for char in text)


def preserve_terminal_punctuation(corrected_text: str, source_text: str) -> tuple[str, bool]:
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return corrected_text, False

    # Do not force terminal punctuation parity for punctuation-free inputs.
    if not _contains_any_punctuation(source_text):
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


def preserve_first_token_casing(
    corrected_text: str,
    source_text: str,
    no_touch_tokens: object = None,
) -> tuple[str, bool]:
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return corrected_text, False

    source_span = _leading_cased_word_span(source_text)
    if source_span is None:
        return corrected_text, False

    source_token, source_start, _ = source_span
    corrected_span = _cased_word_span_at(corrected_text, source_start)
    if corrected_span is None:
        return corrected_text, False

    corrected_token, corrected_start, corrected_end = corrected_span
    if source_token == corrected_token:
        return corrected_text, False

    if _is_first_token_in_no_touch_entities(source_token, corrected_token, no_touch_tokens):
        return corrected_text, False

    if (
        _has_case_distinction(source_token)
        and _has_case_distinction(corrected_token)
        and source_token.casefold() == corrected_token.casefold()
    ):
        return corrected_text[:corrected_start] + source_token + corrected_text[corrected_end:], True

    return corrected_text, False


def validate_terminal_punctuation_preserved(
    corrected_text: str,
    source_text: str,
) -> tuple[bool, str]:
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return True, ""

    source_terminal = _terminal_punctuation_char(source_text)
    corrected_terminal = _terminal_punctuation_char(corrected_text)
    if source_terminal == corrected_terminal:
        return True, ""

    return (
        False,
        "terminal punctuation changed "
        f"(source='{source_terminal or '<none>'}', corrected='{corrected_terminal or '<none>'}')",
    )


def validate_first_token_casing_preserved(
    corrected_text: str,
    source_text: str,
) -> tuple[bool, str]:
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return True, ""

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
        return False, f"first token casing changed (source='{source_token}', corrected='{corrected_token}')"

    return True, ""


def _is_index_inside_url_or_email(text: str, index: int) -> bool:
    if not isinstance(text, str) or index < 0 or index >= len(text):
        return False

    for match in _URL_EMAIL_PATTERN.finditer(text):
        if match.start() <= index < match.end():
            return True
    return False


# def _looks_like_dot_abbreviation_at(text: str, index: int) -> bool:
#     # Disabled by request: abbreviation detection is too heuristic.
#     ...


# def _is_sentence_boundary_punctuation_at(text: str, index: int) -> bool:
#     # Disabled by request: sentence-boundary detection currently depends on
#     # _looks_like_dot_abbreviation_at and is considered too heuristic.
#     ...


# def preserve_sentence_start_casing(
#     corrected_text: str,
#     no_touch_tokens: object = None,
# ) -> tuple[str, bool]:
#     # Disabled by request: sentence-start casing depends on punctuation boundary
#     # heuristics that are currently considered too brittle.
#     ...


def _materialize_corrected_text(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload

    corrected_text = payload.get("corrected_text")
    if isinstance(corrected_text, str) and corrected_text.strip():
        return payload

    for key in reversed(_STEP_CHAIN_KEYS):
        value = payload.get(key)
        if isinstance(value, dict):
            result = value.get("result")
            if isinstance(result, str):
                payload["corrected_text"] = result
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
            f"{', '.join(_REQUIRED_TOP_LEVEL_KEY_ORDER)} "
            "(optional internal key: corrected_text)"
        )

    tokenization_value = payload.get("tokenization")
    if not isinstance(tokenization_value, dict):
        return False, "tokenization must be an object"
    if set(tokenization_value.keys()) != {"tokens"}:
        return False, "tokenization must contain exactly key tokens"
    tokens = tokenization_value.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        return False, "tokenization.tokens must be an array of strings"

    translation = payload.get("translation")
    if not isinstance(translation, str):
        return False, "translation must be a string"

    aggressiveness_level = payload.get("aggressiveness_level")
    if not isinstance(aggressiveness_level, str):
        return False, 'aggressiveness_level must be one of "low", "medium", "high"'
    if aggressiveness_level not in {"low", "medium", "high"}:
        return False, 'aggressiveness_level must be one of "low", "medium", "high"'

    speaker_scope = payload.get("speaker_scope")
    if speaker_scope is not None:
        if not isinstance(speaker_scope, str):
            return False, 'speaker_scope must be one of "single", "multi", "unknown"'
        if speaker_scope not in {"single", "multi", "unknown"}:
            return False, 'speaker_scope must be one of "single", "multi", "unknown"'

    no_touch_tokens = payload.get("no_touch_tokens")
    if not isinstance(no_touch_tokens, list) or not all(isinstance(token, str) for token in no_touch_tokens):
        return False, "no_touch_tokens must be an array of strings"

    def _validate_step_field(step_name: str) -> tuple[bool, str]:
        value = payload.get(step_name)
        if value is None:
            return True, ""
        if not isinstance(value, dict):
            return False, f"{step_name} value must be dict with keys edits/result"

        if set(value.keys()) != {"edits", "result"}:
            return False, f"{step_name} must contain exactly keys edits/result"

        edits = value.get("edits")
        if not isinstance(edits, list):
            return False, f"{step_name}.edits must be an array"

        for index, pair in enumerate(edits, start=1):
            if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(part, str) for part in pair):
                return False, f"{step_name}.edits[{index}] must be [before, after] strings"

        result = value.get("result")
        if not isinstance(result, str):
            return False, f"{step_name}.result must be a string"

        return True, ""

    for step_key in _STEP_CHAIN_KEYS:
        is_step_valid, step_error = _validate_step_field(step_key)
        if not is_step_valid:
            return False, step_error

    if not isinstance(payload.get("corrected_text", ""), str):
        return False, "corrected_text must be a string"

    source_text = payload.get("source_text")
    if source_text is not None and not isinstance(source_text, str):
        return False, "source_text must be a string"

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
    if any(f'"{key}"' in lowered for key in _REQUIRED_TOP_LEVEL_KEYS):
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


def strip_markdown_code_fence(text: str) -> str:
    if not isinstance(text, str):
        return ""

    trimmed = text.strip()
    if not trimmed:
        return trimmed

    html_block = re.match(
        r"^\s*<pre>\s*<code[^>]*>(.*?)</code>\s*</pre>\s*$",
        trimmed,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if html_block:
        return html_block.group(1).strip()

    full_fence_match = re.match(
        r"^\s*(```|~~~)[^\n]*\n(.*?)\n\1\s*$",
        trimmed,
        flags=re.DOTALL,
    )
    if full_fence_match:
        return full_fence_match.group(2).strip()

    embedded_fence_match = re.search(
        r"(```|~~~)[^\n]*\n(.*?)\n\1",
        trimmed,
        flags=re.DOTALL,
    )
    if embedded_fence_match:
        return embedded_fence_match.group(2).strip()

    lines = trimmed.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines and all(line.lstrip().startswith(">") for line in non_empty_lines):
        unquoted = "\n".join(re.sub(r"^\s*>\s?", "", line) for line in lines).strip()
        if unquoted and unquoted != trimmed:
            return strip_markdown_code_fence(unquoted)

    return trimmed


def parse_and_validate_json(content: str) -> tuple[dict | None, str | None]:
    trimmed = strip_markdown_code_fence(content)

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
    skip_first_token_casing_preservation: bool = False,
) -> tuple[dict | None, str | None, str]:
    content = raw_content.strip()
    if not content:
        return None, _mark_non_repairable_validation_error("Model returned empty output."), content

    payload, validation_error = parse_and_validate_json(content)
    if payload is None:
        return None, validation_error, content

    normalize_payload_result_newlines(payload, source_text)

    corrected_text = ""
    no_touch_tokens = None
    if isinstance(payload, dict):
        no_touch_tokens = payload.get("no_touch_tokens")

        # Final chain result is the last active ct_* step in chain order.
        for step_key in reversed(_STEP_CHAIN_KEYS):
            step_payload = payload.get(step_key)
            step_result = step_payload.get("result") if isinstance(step_payload, dict) else None
            if isinstance(step_result, str):
                corrected_text = step_result
                break

        if not corrected_text:
            existing_corrected_text = payload.get("corrected_text")
            corrected_text = existing_corrected_text if isinstance(existing_corrected_text, str) else ""
        payload["corrected_text"] = corrected_text

    casing_normalized = False
    spacing_normalized = False
    if not skip_first_token_casing_preservation:
        corrected_text, casing_normalized = preserve_first_token_casing(
            corrected_text,
            source_text,
            no_touch_tokens,
        )
    corrected_text, spacing_normalized = normalize_char_based_spacing_input(corrected_text)
    corrected_text, punctuation_normalized = preserve_terminal_punctuation(
        corrected_text,
        source_text,
    )
    # Disable this as it calls _looks_like_dot_abbreviation_at which is too heuristic.
    # corrected_text, sentence_casing_normalized = preserve_sentence_start_casing(
    #     corrected_text,
    #     no_touch_tokens,
    # )

    if (
        casing_normalized
        or spacing_normalized
        or punctuation_normalized
        # or sentence_casing_normalized
    ) and isinstance(payload, dict):
        payload["corrected_text"] = corrected_text

    hallucination_ok, hallucination_error = validate_corrected_text_hallucination(
        corrected_text,
        source_text,
    )
    if not hallucination_ok:
        return None, _mark_non_repairable_validation_error(
            f"Hallucination check failed: {hallucination_error}"
        ), content

    return payload, None, content


def build_empty_payload() -> dict:
    payload = {
        "tokenization": {"tokens": []},
        "translation": "",
        "aggressiveness_level": "low",
        "speaker_scope": "unknown",
    }

    for step_key in _STEP_CHAIN_KEYS:
        payload[step_key] = _new_empty_step_payload()

    return payload


def describe_top_level_key_error(
    raw_output: str,
    processing_id: str | None = None,
) -> str | None:
    prefix = f"[{processing_id}] " if processing_id else ""
    trimmed = strip_markdown_code_fence(raw_output)
    extracted = extract_first_json_object(trimmed)
    candidate = extracted if extracted is not None else trimmed

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

    default_path = Path(__file__).with_name(default_filename)
    return default_path


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
        "prompt_patch.md",
        "Patch prompt",
    )
    if prompt_path_error:
        return None, None, prompt_path_error

    repair_prompt_template_path, repair_path_error = resolve_required_template_path(
        repair_override_value,
        "prompt_repair.md",
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


def parse_transcriptions_from_file(
    input_path: Path,
) -> tuple[list[str], list[str | None], list[list[str] | None]]:
    def _select_source_identifier(cells: list[str]) -> str | None:
        if not cells or len(cells) < 2:
            return None

        # Treat the last column as the segment text; prefer the metadata column
        # immediately before it as a stable row-level identifier.
        preferred = cells[-2].strip()
        if preferred:
            return preferred

        for value in reversed(cells[:-1]):
            candidate = value.strip()
            if candidate:
                return candidate

        return None

    def _strip_emphasis_tags(text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return re.sub(r"</?\s*(?:b|strong|em|i)\b[^>]*>", "", text, flags=re.IGNORECASE)

    raw_text = input_path.read_text(encoding="utf-8")
    if raw_text == "":
        return [], [], []

    if input_path.suffix.lower() == ".tsv":
        transcriptions: list[str] = []
        source_filenames: list[str | None] = []
        source_rows: list[list[str] | None] = []

        reader = csv.reader(raw_text.splitlines(), delimiter="\t")
        first_row_is_header = False
        for row_index, row in enumerate(reader):
            if not row or not any(cell.strip() for cell in row):
                continue

            first_cell = row[0].strip() if len(row) > 0 else ""
            last_cell = row[-1].strip() if len(row) > 0 else ""
            if row_index == 0:
                first_label = first_cell.lower().replace(" ", "_")
                last_label = last_cell.lower().replace(" ", "_")
                if first_label in {"filename", "file", "file_name"} and last_label in {
                    "input",
                    "input_segment",
                    "segment",
                    "transcription",
                    "text",
                }:
                    first_row_is_header = True
                    continue
                if len(row) == 1 and first_label in {
                    "input",
                    "input_segment",
                    "segment",
                    "transcription",
                    "text",
                }:
                    first_row_is_header = True
                    continue

            if len(row) >= 2:
                filename = _select_source_identifier(row)
                segment = _strip_emphasis_tags(last_cell)
            else:
                filename = None
                segment = _strip_emphasis_tags(first_cell)
            transcriptions.append(segment)
            source_filenames.append(filename)
            source_rows.append(list(row))

        if first_row_is_header and not transcriptions:
            print("TSV input contains only header and no data rows.")

        return transcriptions, source_filenames, source_rows

    stripped = raw_text.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        transcriptions = [_strip_emphasis_tags(item.strip()) for item in parsed if isinstance(item, str)]
        return transcriptions, [None] * len(transcriptions), [None] * len(transcriptions)

    if isinstance(parsed, dict):
        items = parsed.get("transcriptions")
        if isinstance(items, list):
            transcriptions = [_strip_emphasis_tags(item.strip()) for item in items if isinstance(item, str)]
            return transcriptions, [None] * len(transcriptions), [None] * len(transcriptions)

    transcriptions: list[str] = []
    source_filenames: list[str | None] = []
    source_rows: list[list[str] | None] = []
    for line in raw_text.splitlines():
        if line.lstrip().startswith("#"):
            # Preserve comment lines verbatim so they can be emitted unchanged.
            transcriptions.append(line)
            source_filenames.append(None)
            source_rows.append(None)
            continue

        stripped_line = line.strip()
        if "\t" in stripped_line:
            parts = stripped_line.split("\t")
            filename = _select_source_identifier(parts)
            segment = _strip_emphasis_tags(parts[-1].strip())
            transcriptions.append(segment)
            source_filenames.append(filename)
            source_rows.append(parts)
            continue

        transcriptions.append(_strip_emphasis_tags(stripped_line))
        source_filenames.append(None)
        source_rows.append(None)
    return transcriptions, source_filenames, source_rows


def is_input_comment_line(transcription: str) -> bool:
    return isinstance(transcription, str) and transcription.lstrip().startswith("#")


def normalize_char_based_spacing_input(transcription: str) -> tuple[str, bool]:
    """Remove spacing artifacts for same-script char-based languages while preserving mixed-script boundaries."""
    if not isinstance(transcription, str) or not transcription:
        return transcription, False

    normalized_parts: list[str] = []
    index = 0
    changed = False
    length = len(transcription)

    while index < length:
        char = transcription[index]
        if not char.isspace():
            normalized_parts.append(char)
            index += 1
            continue

        run_start = index
        while index < length and transcription[index].isspace():
            index += 1

        prev_char = normalized_parts[-1] if normalized_parts else ""
        next_char = transcription[index] if index < length else ""
        prev_group = _char_based_script_group(prev_char)
        next_group = _char_based_script_group(next_char)

        if prev_group is not None and prev_group == next_group:
            changed = True
            continue

        normalized_parts.append(transcription[run_start:index])

    if not changed:
        return transcription, False

    return "".join(normalized_parts), True


def normalize_all_uppercase_input(transcription: str) -> tuple[str, bool]:
    """Convert all-uppercase sentence-like input to lowercase before prompting."""
    if not isinstance(transcription, str):
        return transcription, False

    stripped = transcription.strip()
    if not stripped:
        return transcription, False

    cased_letter_count = 0
    uppercase_cased_letter_count = 0
    for char in stripped:
        if not _is_unicode_cased_letter(char):
            continue
        cased_letter_count += 1
        if char == char.upper() and char != char.lower():
            uppercase_cased_letter_count += 1

    if cased_letter_count == 0:
        return transcription, False
    if uppercase_cased_letter_count != cased_letter_count:
        return transcription, False

    # Avoid rewriting inputs that contain only one casable token (for example "NASA").
    cased_word_matches = re.findall(r"[^\W\d_]+", stripped, flags=re.UNICODE)
    cased_word_count = sum(1 for token in cased_word_matches if _has_case_distinction(token))
    if cased_word_count <= 1:
        return transcription, False

    lowered = transcription.lower()
    # Some locale-sensitive mappings (for example Turkish dotted I) expand to multiple code points.
    # Skip normalization in these cases to avoid introducing display artifacts.
    if len(lowered) != len(transcription):
        return transcription, False

    # Avoid rewriting short acronym-like single-token inputs such as "NASA".
    # Expect prompt to convert it back
    # cased_word_matches = re.findall(r"[^\W\d_]+", stripped, flags=re.UNICODE)
    # cased_word_count = sum(1 for token in cased_word_matches if _has_case_distinction(token))
    # if cased_word_count < 2 and cased_letter_count < 10:
    #     return transcription, False

    return lowered, True


def collect_transcriptions_from_input(
    input_file_value: str | None,
) -> tuple[list[str], list[str | None], list[list[str] | None]] | None:
    transcriptions: list[str]
    source_filenames: list[str | None]
    source_rows: list[list[str] | None]
    if input_file_value:
        input_path = resolve_path(input_file_value)
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            return None
        transcriptions, source_filenames, source_rows = parse_transcriptions_from_file(input_path)
        print(f"Read {len(transcriptions)} transcription(s) from: {input_path}")
    else:
        transcription = input("Enter transcription: ").strip()
        transcriptions = [transcription] if transcription else []
        source_filenames = [None] * len(transcriptions)
        source_rows = [None] * len(transcriptions)

    if not transcriptions:
        print("No transcription provided.")
        return None

    return transcriptions, source_filenames, source_rows

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
    text_output_lines: list[str] | None = None,
    source_filenames: list[str | None] | None = None,
    source_rows: list[list[str] | None] | None = None,
    output_as_tsv: bool | None = None,
) -> bool:
    if text_output_lines is not None and len(text_output_lines) != len(payloads):
        print("WARNING: text output line count does not match payload count; normalizing output lines.")
        if len(text_output_lines) < len(payloads):
            text_output_lines = [*text_output_lines, *([""] * (len(payloads) - len(text_output_lines)))]
        else:
            text_output_lines = text_output_lines[:len(payloads)]

    if source_filenames is not None and len(source_filenames) != len(payloads):
        print("WARNING: source filename count does not match payload count; omitting filename metadata in output.")
        source_filenames = None

    if source_rows is not None and len(source_rows) != len(payloads):
        print("WARNING: source row count does not match payload count; omitting row metadata in output.")
        source_rows = None

    if text_output_lines is None and any(payload is None for payload in payloads):
        print("Failed to produce output for one or more transcriptions.")
        return False

    if text_output_lines is not None:
        for index, payload in enumerate(payloads):
            if payload is None and not isinstance(text_output_lines[index], str):
                print(f"WARNING: invalid text output line for transcription {index + 1}; using empty text fallback.")
                text_output_lines[index] = ""

    fallback_text_lines: list[str]
    if text_output_lines is None:
        fallback_text_lines = []
        for payload in payloads:
            if isinstance(payload, dict):
                corrected_text = payload.get("corrected_text")
                fallback_text_lines.append(corrected_text if isinstance(corrected_text, str) else "")
            else:
                fallback_text_lines.append("")
    else:
        fallback_text_lines = [line if isinstance(line, str) else "" for line in text_output_lines]

    final_payloads = [payload for payload in payloads if isinstance(payload, dict)]

    is_valid, validation_error = validate_output_payloads(final_payloads)
    if not is_valid:
        print(validation_error)
        write_fallback_text_output(
            output_file_value,
            fallback_text_lines,
            source_filenames,
            source_rows,
            reason="Output JSON validation failed",
            output_as_tsv=output_as_tsv,
        )
        return True

    try:
        write_output_artifacts(
            final_payloads,
            output_file_value,
            text_output_lines,
            source_filenames,
            source_rows,
            output_as_tsv,
        )
    except Exception as error:
        print(f"Failed to write JSON output: {error}")
        write_fallback_text_output(
            output_file_value,
            fallback_text_lines,
            source_filenames,
            source_rows,
            reason="Output JSON write failed",
            output_as_tsv=output_as_tsv,
        )
        return True
    return True


def write_fallback_text_output(
    output_file_value: str | None,
    text_lines: list[str],
    source_filenames: list[str | None] | None = None,
    source_rows: list[list[str] | None] | None = None,
    reason: str | None = None,
    output_as_tsv: bool | None = None,
) -> None:
    normalized_text_lines = [
        sanitize_output_string(line if isinstance(line, str) else "")
        for line in text_lines
    ]
    is_tsv_output = (
        output_as_tsv
        if isinstance(output_as_tsv, bool)
        else bool(
            source_filenames
            and any(isinstance(filename, str) and filename.strip() for filename in source_filenames)
        )
    )

    if source_filenames is not None and len(source_filenames) == len(normalized_text_lines):
        fallback_lines = [
                f"{sanitize_output_string(filename if isinstance(filename, str) else '')}\t{text}"
            for filename, text in zip(source_filenames, normalized_text_lines)
        ]
    else:
        fallback_lines = normalized_text_lines

    fallback_rows: list[list[str]] | None = None
    if source_rows is not None and len(source_rows) == len(normalized_text_lines):
        fallback_rows = []
        for original_row, corrected_text in zip(source_rows, normalized_text_lines):
            if isinstance(original_row, list) and original_row:
                rebuilt_row = [sanitize_output_string(value) for value in original_row]
                rebuilt_row[-1] = corrected_text
                fallback_rows.append(rebuilt_row)
            else:
                fallback_rows.append([corrected_text])

    fallback_content = "\n".join(fallback_lines)

    if output_file_value:
        output_path = resolve_path(output_file_value)
        output_text_path = output_path.with_suffix(".tsv") if is_tsv_output else output_path.with_suffix(".txt")
        output_text_path.parent.mkdir(parents=True, exist_ok=True)
        if is_tsv_output and fallback_rows is not None:
            with output_text_path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
                writer.writerows(fallback_rows)
        else:
            output_text_path.write_text(fallback_content, encoding="utf-8")
        if reason:
            print(f"{reason}; wrote fallback text output to: {output_text_path}")
        else:
            print(f"Wrote fallback text output to: {output_text_path}")
        return

    if reason:
        print(f"{reason}; emitting fallback text output to stdout.")
    print(fallback_content)


def strip_prompt_comments(prompt_text: str) -> str:
    filtered_lines: list[str] = []
    for line in prompt_text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def load_prompt_template(template_path: Path) -> str:
    prompt_text = template_path.read_text(encoding="utf-8")
    if template_path.suffix.lower() == ".md":
        # Keep markdown headings/content intact for markdown-based prompts.
        return prompt_text
    return strip_prompt_comments(prompt_text)


def load_patch_and_repair_templates(
    prompt_template_path: Path,
    repair_prompt_template_path: Path,
) -> tuple[str, str]:
    return (
        load_prompt_template(prompt_template_path),
        load_prompt_template(repair_prompt_template_path),
    )


_CHAIN_ID_TO_NAME = {
    "0": "SPEAKER",
    "1": "COMBINE",
    "2": "NO_TOUCH",
    "3": "LEXICAL",
    "4": "DISFLUENCY",
    "5": "FORMAT",
    "6": "NUMERAL",
    "7": "PUNCT",
    "8": "CASING",
    "9": "REMAIN_FIX",
}

_CHAIN_NAME_TO_ID = {value: key for key, value in _CHAIN_ID_TO_NAME.items()}
_DEFAULT_CHAIN_IDS = list(_CHAIN_ID_TO_NAME.keys())
_NO_TOUCH_CHAIN_ID = "2"
_LEXICAL_CHAIN_ID = "3"
_DISFLUENCY_CHAIN_ID = "4"


def _expand_chain_step_values(chain_steps: list[str] | None) -> list[str]:
    expanded_steps: list[str] = []
    for raw_step in chain_steps or []:
        if not isinstance(raw_step, str):
            continue

        step_value = raw_step.strip()
        if not step_value:
            continue

        parsed_list: list[str] | None = None
        if step_value.startswith("[") and step_value.endswith("]"):
            parsed: object
            try:
                parsed = json.loads(step_value)
            except Exception:
                try:
                    parsed = ast.literal_eval(step_value)
                except Exception:
                    parsed = None

            if isinstance(parsed, (list, tuple)):
                parsed_list = [str(item).strip() for item in parsed if str(item).strip()]

        if parsed_list is not None:
            expanded_steps.extend(parsed_list)
            continue

        if "," in step_value:
            split_values = [item.strip() for item in step_value.split(",") if item.strip()]
            expanded_steps.extend(split_values)
            continue

        expanded_steps.append(step_value)

    return expanded_steps


def _normalize_chain_step_token(token: str) -> str | None:
    if not isinstance(token, str):
        return None
    normalized = token.strip()
    if not normalized:
        return None

    bracket_match = re.match(r"^\[\s*([A-Za-z_]+)\s*\]$", normalized)
    if bracket_match:
        normalized = bracket_match.group(1)

    upper = normalized.upper()
    if upper in _CHAIN_NAME_TO_ID:
        return _CHAIN_NAME_TO_ID[upper]

    if normalized in _CHAIN_ID_TO_NAME:
        return normalized

    return None


def _resolve_active_chain_ids(chain_steps: list[str] | None) -> list[str]:
    expanded = _expand_chain_step_values(chain_steps)
    if not expanded:
        return _DEFAULT_CHAIN_IDS

    resolved_ids: list[str] = []
    seen_ids: set[str] = set()

    for token in expanded:
        chain_id = _normalize_chain_step_token(token)
        if chain_id is None:
            # If any selector token is invalid, fall back to full default chain.
            return _DEFAULT_CHAIN_IDS
        if chain_id in seen_ids:
            continue
        seen_ids.add(chain_id)
        resolved_ids.append(chain_id)

    if resolved_ids:
        requires_no_touch = (
            _LEXICAL_CHAIN_ID in resolved_ids
            or _DISFLUENCY_CHAIN_ID in resolved_ids
        )
        if requires_no_touch and _NO_TOUCH_CHAIN_ID not in resolved_ids:
            insert_before = len(resolved_ids)
            for index, chain_id in enumerate(resolved_ids):
                if chain_id in {_LEXICAL_CHAIN_ID, _DISFLUENCY_CHAIN_ID}:
                    insert_before = index
                    break
            resolved_ids.insert(insert_before, _NO_TOUCH_CHAIN_ID)

        return resolved_ids

    return _DEFAULT_CHAIN_IDS


def format_resolved_chain_steps(chain_steps: list[str] | None) -> str:
    """Return a stable display string for resolved active chain steps."""
    active_chain_ids = _resolve_active_chain_ids(chain_steps)
    return ", ".join(
        f"{chain_id}:{_CHAIN_ID_TO_NAME.get(chain_id, 'UNKNOWN')}"
        for chain_id in active_chain_ids
    )


def build_patch_prompt(
    prompt_template: str,
    transcription: str,
    chain_steps: list[str] | None = None,
) -> str:
    active_chain_ids = _resolve_active_chain_ids(chain_steps)
    active_chain_names = [_CHAIN_ID_TO_NAME[chain_id] for chain_id in active_chain_ids]

    chain_steps_text = "\n".join(
        f"{chain_id}) {chain_name}"
        for chain_id, chain_name in zip(active_chain_ids, active_chain_names)
    )

    prompt = prompt_template
    if "{chain_steps}" in prompt:
        prompt = prompt.replace("{chain_steps}", chain_steps_text)
    if "{input_transcript}" in prompt:
        prompt = prompt.replace("{input_transcript}", transcription)
        return prompt

    return prompt + transcription


def write_output_artifacts(
    payloads: list[dict],
    output_file_value: str | None,
    text_output_lines: list[str] | None = None,
    source_filenames: list[str | None] | None = None,
    source_rows: list[list[str] | None] | None = None,
    output_as_tsv: bool | None = None,
) -> None:
    is_tsv_output = (
        output_as_tsv
        if isinstance(output_as_tsv, bool)
        else bool(
            source_filenames
            and any(isinstance(filename, str) and filename.strip() for filename in source_filenames)
        )
    )

    def _order_top_level_payload_keys(payload: dict) -> dict:
        ordered_payload: dict = {}

        # Emit source_filename first in serialized output when present.
        if "source_filename" in payload:
            ordered_payload["source_filename"] = payload["source_filename"]

        # Emit source_text next in serialized output when present.
        if "source_text" in payload:
            ordered_payload["source_text"] = payload["source_text"]

        # Keep canonical required key order for deterministic JSON diffs.
        for key in _REQUIRED_TOP_LEVEL_KEY_ORDER:
            if key in payload:
                ordered_payload[key] = payload[key]

        # Keep known optional keys next (except source_text already emitted first).
        for key in _OPTIONAL_TOP_LEVEL_KEY_ORDER:
            if key == "source_text":
                continue
            if key in payload:
                ordered_payload[key] = payload[key]

        # Preserve any unexpected keys at the end in original insertion order.
        for key, value in payload.items():
            if key not in ordered_payload:
                ordered_payload[key] = value

        return ordered_payload

    def _sanitize_output_payload_value(value: object) -> object:
        if isinstance(value, str):
            return sanitize_output_string(value)
        if isinstance(value, list):
            return [_sanitize_output_payload_value(item) for item in value]
        if isinstance(value, tuple):
            return [_sanitize_output_payload_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: _sanitize_output_payload_value(item)
                for key, item in value.items()
            }
        return value

    can_backfill_source_filename = (
        source_filenames is not None
        and len(source_filenames) == len(payloads)
    )

    sanitized_payloads: list[dict] = []
    for index, item in enumerate(payloads):
        if isinstance(item, dict):
            payload_copy = _sanitize_output_payload_value(item)
            if not isinstance(payload_copy, dict):
                continue
            if can_backfill_source_filename:
                source_filename = source_filenames[index]
                if isinstance(source_filename, str) and source_filename.strip():
                    payload_copy["source_filename"] = sanitize_output_string(source_filename)
            payload_copy.pop("tokenization", None)
            sanitized_payloads.append(_order_top_level_payload_keys(payload_copy))

    output_payload = sanitized_payloads[0] if len(sanitized_payloads) == 1 else sanitized_payloads

    corrected_text_lines: list[str] = []
    for item in sanitized_payloads:
        corrected_text = item.get("corrected_text") if isinstance(item, dict) else ""
        corrected_text_lines.append(corrected_text if isinstance(corrected_text, str) else "")

    if text_output_lines is None:
        normalized_text_lines = corrected_text_lines
    else:
        normalized_text_lines = [
            sanitize_output_string(line if isinstance(line, str) else "")
            for line in text_output_lines
        ]
    text_output_content = "\n".join(normalized_text_lines)

    normalized_source_filenames: list[str] = [""] * len(normalized_text_lines)
    if source_filenames is not None:
        if len(source_filenames) != len(normalized_text_lines):
            print("WARNING: source filename count does not match output line count; omitting TSV filename column.")
        else:
            normalized_source_filenames = [
                sanitize_output_string(value if isinstance(value, str) else "")
                for value in source_filenames
            ]

    normalized_source_rows: list[list[str] | None] | None = None
    if source_rows is not None:
        if len(source_rows) != len(normalized_text_lines):
            print("WARNING: source row count does not match output line count; omitting full-row TSV reconstruction.")
        else:
            normalized_source_rows = []
            for row in source_rows:
                if isinstance(row, list):
                    normalized_source_rows.append([sanitize_output_string(value) for value in row])
                else:
                    normalized_source_rows.append(None)

    output_content = json.dumps(output_payload, ensure_ascii=False, indent=2)
    if output_file_value:
        output_path = resolve_path(output_file_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_json_path = output_path.with_suffix(".json")
        output_json_path.write_text(output_content, encoding="utf-8")

        if is_tsv_output:
            output_tsv_path = output_path.with_suffix(".tsv")
            with output_tsv_path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
                for row_index, corrected_text in enumerate(normalized_text_lines):
                    if (
                        normalized_source_rows is not None
                        and row_index < len(normalized_source_rows)
                        and isinstance(normalized_source_rows[row_index], list)
                        and normalized_source_rows[row_index]
                    ):
                        row = list(normalized_source_rows[row_index])
                        row[-1] = corrected_text
                        writer.writerow(row)
                        continue

                    filename = (
                        normalized_source_filenames[row_index]
                        if row_index < len(normalized_source_filenames)
                        else ""
                    )
                    writer.writerow([filename, corrected_text])
        else:
            output_text_path = output_path.with_suffix(".txt")
            output_text_path.parent.mkdir(parents=True, exist_ok=True)
            output_text_path.write_text(text_output_content, encoding="utf-8")
    else:
        print(output_content)


def sanitize_output_string(value: str) -> str:
    if not isinstance(value, str):
        return ""

    normalized = (
        value.replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace('"', " ")
    )
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized.strip()


def format_repair_prompt(
    repair_prompt_template: str,
    validation_error: str | None,
    previous_output: str,
    target_schema: str | None = None,
    processing_id: str | None = None,
) -> str:
    supported_placeholders = {"validation_error", "previous_output", "target_schema"}
    template_placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", repair_prompt_template))
    unresolved_placeholders = sorted(template_placeholders - supported_placeholders)
    if unresolved_placeholders:
        prefix = f"[{processing_id}] " if processing_id else ""
        print(
            f"{prefix}WARNING: unresolved repair template placeholders detected: "
            f"{', '.join(unresolved_placeholders)}"
        )

    schema_text = target_schema
    if not isinstance(schema_text, str) or not schema_text.strip():
        schema_text = json.dumps(
            {
                "type": "object",
                "required": list(_REQUIRED_TOP_LEVEL_KEY_ORDER),
                "additionalProperties": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    if any(
        placeholder in repair_prompt_template
        for placeholder in ("{validation_error}", "{previous_output}", "{target_schema}")
    ):
        return (
            repair_prompt_template
            .replace("{validation_error}", validation_error or "")
            .replace("{previous_output}", previous_output)
            .replace("{target_schema}", schema_text)
        )

    return (
        f"{repair_prompt_template}\n\n"
        f"Target schema:\n{schema_text}\n\n"
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
    target_schema: str | None = None,
    processing_id: str | None = None,
) -> str:
    prefix = f"[{processing_id}] " if processing_id else ""
    print(f"{prefix}Initial response was not valid strict JSON. Running one repair attempt...")
    return format_repair_prompt(
        repair_prompt_template,
        validation_error,
        previous_output,
        target_schema,
        processing_id,
    )


async def get_patch_payload_with_repair_generic(
    prompt: str,
    transcription: str,
    processing_id: str,
    repair_prompt_template: str,
    target_schema: str,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    send_prompt: Callable[[str, int], Awaitable[str | None]],
    send_repair_prompt: Callable[[str, int], Awaitable[str | None]],
    build_attempt_prompt: Callable[[str, int], str] | None = None,
    skip_first_token_casing_preservation: bool = False,
    repair_timeout_message: str = "Repair attempt timed out.",
    repair_empty_message: str = "Repair retry returned empty output.",
    strip_repair_content: bool = False,
) -> dict | None:
    for empty_attempt in range(empty_result_retries + 1):
        attempt_prompt = (
            build_attempt_prompt(prompt, empty_attempt)
            if build_attempt_prompt is not None
            else prompt
        )

        content = await send_prompt(attempt_prompt, empty_attempt)
        if content is None:
            print_timeout_and_retry_guidance(timeout_seconds, timeout_retries, processing_id)
            return None

        payload, validation_error, content = parse_validate_and_apply_text_fixes(
            content,
            transcription,
            processing_id,
            skip_first_token_casing_preservation=skip_first_token_casing_preservation,
        )

        if payload is None:
            log_json_validation_with_key_error(validation_error, content, processing_id)

            if is_non_repairable_validation_error(validation_error):
                reason = (
                    validation_error.removeprefix(non_repairable_prefix()).strip()
                    if isinstance(validation_error, str)
                    else "Model output is not repairable."
                )
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
                target_schema=target_schema,
                processing_id=processing_id,
            )

            repaired_content = await send_repair_prompt(repair_prompt, empty_attempt)
            if repaired_content is None:
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    repair_timeout_message,
                    processing_id=processing_id,
                )
                if not should_retry:
                    return None
                continue

            if strip_repair_content:
                repaired_content = repaired_content.strip()

            if not repaired_content:
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    repair_empty_message,
                    processing_id=processing_id,
                )
                if not should_retry:
                    return None
                continue

            payload, validation_error, repaired_content = parse_validate_and_apply_text_fixes(
                repaired_content,
                transcription,
                processing_id,
                skip_first_token_casing_preservation=skip_first_token_casing_preservation,
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
            empty_attempt,
            empty_result_retries,
            processing_id,
        )
        if not should_retry:
            return result_payload

    return None


def apply_corrected_text_fallback(payload: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None

    if not isinstance(payload.get("tokenization"), dict):
        payload["tokenization"] = {"tokens": []}
    else:
        tokenization = payload["tokenization"]
        if not isinstance(tokenization.get("tokens"), list):
            tokenization["tokens"] = []

    for step_key in _STEP_CHAIN_KEYS:
        if not isinstance(payload.get(step_key), dict):
            payload[step_key] = _new_empty_step_payload()

    no_touch_tokens = payload.get("no_touch_tokens")
    if not isinstance(no_touch_tokens, list) or not all(isinstance(token, str) for token in no_touch_tokens):
        payload["no_touch_tokens"] = []

    translation = payload.get("translation")
    if not isinstance(translation, str):
        payload["translation"] = ""

    aggressiveness_level = payload.get("aggressiveness_level")
    if not isinstance(aggressiveness_level, str) or aggressiveness_level not in {"low", "medium", "high"}:
        payload["aggressiveness_level"] = "low"

    payload["corrected_text"] = ""
    return payload
