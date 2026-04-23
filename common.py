import asyncio
import argparse
import ast
import builtins
import csv
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
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
    "seg_start",
    "seg_end",
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


DEFAULT_CONCURRENCY = 5
DEFAULT_TIMEOUT_SECONDS = 1000.0
DEFAULT_TIMEOUT_RETRIES = 1
DEFAULT_EMPTY_RESULT_RETRIES = 1
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_RETRY_TEMPERATURE_JITTER = 0.08
DEFAULT_RETRY_TOP_P_JITTER = 0.03
DEFAULT_MODEL_MISMATCH_RETRIES = 1
DEFAULT_MAX_INPUT_CHARS_PER_CALL = 300
DEFAULT_LONG_SPAN_MIN_DELETED_TOKENS = 3
DEFAULT_HALLUCINATION_MAX_INSERTED_TOKENS = 3


_ORIGINAL_PRINT = builtins.print
_SAFE_PRINT_INSTALLED = False


def install_safe_console_output() -> None:
    """Make console logging robust on Windows code pages that cannot encode Unicode."""
    global _SAFE_PRINT_INSTALLED
    if _SAFE_PRINT_INSTALLED:
        return

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass

    def _safe_print(*args, **kwargs) -> None:
        # Default to flush=True so output is visible immediately when redirected to a file.
        if "flush" not in kwargs:
            kwargs["flush"] = True
        try:
            _ORIGINAL_PRINT(*args, **kwargs)
            return
        except UnicodeEncodeError:
            pass

        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        target_stream = kwargs.get("file", sys.stdout)
        flush = kwargs.get("flush", False)

        try:
            raw_text = sep.join(str(arg) for arg in args)
        except Exception:
            raw_text = " ".join(repr(arg) for arg in args)

        encoding = getattr(target_stream, "encoding", None) or "utf-8"
        safe_text = raw_text.encode(encoding, errors="backslashreplace").decode(encoding, errors="ignore")
        _ORIGINAL_PRINT(safe_text, end=end, file=target_stream, flush=flush)

    builtins.print = _safe_print
    _SAFE_PRINT_INSTALLED = True


_long_span_min_deleted_tokens = DEFAULT_LONG_SPAN_MIN_DELETED_TOKENS
_hallucination_max_inserted_tokens = DEFAULT_HALLUCINATION_MAX_INSERTED_TOKENS


def configure_long_span_preservation_guard(
    min_deleted_tokens: int | None = None,
) -> None:
    global _long_span_min_deleted_tokens

    if isinstance(min_deleted_tokens, int):
        _long_span_min_deleted_tokens = max(1, min_deleted_tokens)


def get_long_span_preservation_guard_config() -> int:
    return _long_span_min_deleted_tokens


def configure_hallucination_guard(
    max_inserted_tokens: int | None = None,
) -> None:
    global _hallucination_max_inserted_tokens

    if isinstance(max_inserted_tokens, int):
        _hallucination_max_inserted_tokens = max(1, max_inserted_tokens)


def get_hallucination_guard_config() -> int:
    return _hallucination_max_inserted_tokens


def add_common_runtime_cli_arguments(
    parser: argparse.ArgumentParser,
    timeout_default: float = DEFAULT_TIMEOUT_SECONDS,
    concurrency_default: int = DEFAULT_CONCURRENCY,
    timeout_retries_default: int = DEFAULT_TIMEOUT_RETRIES,
    empty_result_retries_default: int = DEFAULT_EMPTY_RESULT_RETRIES,
    max_input_chars_per_call_default: int = DEFAULT_MAX_INPUT_CHARS_PER_CALL,
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
    parser.add_argument(
        "--max-input-chars-per-call",
        dest="max_input_chars_per_call",
        type=int,
        default=max_input_chars_per_call_default,
        help=(
            "Split very long input into smaller segments and call the model once per segment "
            f"(default: {DEFAULT_MAX_INPUT_CHARS_PER_CALL}; 0 disables segmentation)."
        ),
    )
    parser.add_argument(
        "--long-span-min-deleted-tokens",
        dest="long_span_min_deleted_tokens",
        type=int,
        default=DEFAULT_LONG_SPAN_MIN_DELETED_TOKENS,
        help=(
            "Retry if a contiguous deleted source span has at least this many non-punctuation tokens "
            f"(default: {DEFAULT_LONG_SPAN_MIN_DELETED_TOKENS})."
        ),
    )
    parser.add_argument(
        "--hallucination-max-inserted-tokens",
        dest="hallucination_max_inserted_tokens",
        type=int,
        default=DEFAULT_HALLUCINATION_MAX_INSERTED_TOKENS,
        help=(
            "Fail hallucination check if a contiguous inserted span exceeds this many tokens "
            f"(default: {DEFAULT_HALLUCINATION_MAX_INSERTED_TOKENS})."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="no_resume",
        action="store_true",
        default=False,
        help=(
            "Disable automatic resume from existing output. "
            "By default, if an output text file (.txt or .tsv) exists, "
            "rows with non-empty output are skipped and only empty rows are retried."
        ),
    )
    parser.add_argument(
        "--retry-content-filtered",
        dest="retry_content_filtered",
        action="store_true",
        default=False,
        help=(
            "Treat content-filtered results as failures so they are retried: "
            "on resume, lines starting with %%CF%% are re-submitted, and "
            "%%PS%% partial-segment markers with filtered segments have those "
            "segments retried (instead of kept as-is)."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        dest="output_jsonl",
        action="store_true",
        default=False,
        help="Enable writing JSONL structured output (.jsonl) alongside text output.",
    )


def add_chain_steps_cli_argument(
    parser: argparse.ArgumentParser,
    default: str = "ALL",
) -> None:
    parser.add_argument("--chain-steps", dest="chain_steps", default=default)


def add_run_pipeline_cli_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add CLI arguments shared by run_aoai.py and run_copilot.py."""
    parser.add_argument("--input-file", dest="input_file")
    parser.add_argument("--output-file", dest="output_file")
    parser.add_argument("--patch-prompt-file", dest="patch_prompt_file")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file")
    parser.add_argument(
        "--progress-write-every",
        dest="progress_write_every",
        type=int,
        default=None,
        help=(
            "Write incremental text snapshot (.txt/.tsv) and JSONL progress (.jsonl) "
            "every N completed items "
            "(default: 1, or 100 when resume is active)."
        ),
    )
    parser.add_argument(
        "--locale",
        dest="locale",
        default=None,
        help="Locale of the input audio (e.g. en-US, zh-CN). Adds locale context to the prompt.",
    )
    parser.add_argument(
        "--chain-steps",
        dest="chain_steps",
        action="append",
        help="Repeatable active-chain selector (ids 1-8 or step names like COMBINE, NO_TOUCH).",
    )
    parser.add_argument(
        "--apply-safe-edits",
        dest="apply_safe_edits",
        action="store_true",
        default=False,
        help="Apply character-level diff post-processing to accept only safe edits (punctuation/casing) and reject word-content changes.",
    )


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
        "seg_start": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "seg_end": {
            "type": "string",
            "enum": ["high", "medium", "low"],
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
_SEGMENT_BOUNDARY_SENTENCE_CHARS = "".join(
    sorted(_SENTENCE_END_PUNCTUATION | {";", ":", "；", "：", "؛"})
)
_SEGMENT_BOUNDARY_SENTENCE_CHAR_SET = set(_SEGMENT_BOUNDARY_SENTENCE_CHARS)
_SEGMENT_BOUNDARY_CLOSERS = "".join(sorted(_SENTENCE_CLOSERS))
_SEGMENT_BOUNDARY_PATTERN = re.compile(
    rf"[{re.escape(_SEGMENT_BOUNDARY_SENTENCE_CHARS)}](?:[{re.escape(_SEGMENT_BOUNDARY_CLOSERS)}]*)",
    flags=re.UNICODE,
)
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

# CJK numeral characters — converting these to Arabic digits (e.g. 二零二零 → 2020)
# is a valid numeral normalization edit, not a long-span deletion.
_CJK_NUMERAL_CHARS = frozenset(
    "零一二三四五六七八九十百千万亿兆"  # Simplified
    "壹贰叁肆伍陆柒捌玖拾佰仟萬億"    # Formal/traditional
    "〇"                               # Ideographic zero
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


def _extract_text_content_recursive(content: object, visited_ids: set[int]) -> str:
    if isinstance(content, str):
        return content

    if content is None:
        return ""

    if isinstance(content, (list, dict)) or hasattr(content, "__dict__"):
        object_id = id(content)
        if object_id in visited_ids:
            return ""
        visited_ids.add(object_id)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            extracted = _extract_text_content_recursive(item, visited_ids)
            if extracted:
                parts.append(extracted)
        return "".join(parts)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

        for key in ("content", "delta_content", "message"):
            nested = content.get(key)
            extracted = _extract_text_content_recursive(nested, visited_ids)
            if extracted:
                return extracted
        return ""

    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text

    nested_content = getattr(content, "content", None)
    extracted = _extract_text_content_recursive(nested_content, visited_ids)
    if extracted:
        return extracted

    nested_delta = getattr(content, "delta_content", None)
    extracted = _extract_text_content_recursive(nested_delta, visited_ids)
    if extracted:
        return extracted

    return ""


def _extract_text_content_iterative(content: object) -> str:
    if isinstance(content, str):
        return content

    if content is None:
        return ""

    visited_ids: set[int] = set()
    stack: list[object] = [content]
    parts: list[str] = []

    while stack:
        current = stack.pop()

        if isinstance(current, str):
            parts.append(current)
            continue

        if current is None:
            continue

        if isinstance(current, (list, dict)) or hasattr(current, "__dict__"):
            object_id = id(current)
            if object_id in visited_ids:
                continue
            visited_ids.add(object_id)

        if isinstance(current, list):
            # Preserve left-to-right order when popping from stack.
            for item in reversed(current):
                stack.append(item)
            continue

        if isinstance(current, dict):
            text = current.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue

            # Preserve recursive preference order.
            for key in ("message", "delta_content", "content"):
                nested = current.get(key)
                if nested is not None:
                    stack.append(nested)
            continue

        text = getattr(current, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue

        nested_delta = getattr(current, "delta_content", None)
        if nested_delta is not None:
            stack.append(nested_delta)
        nested_content = getattr(current, "content", None)
        if nested_content is not None:
            stack.append(nested_content)

    return "".join(parts)


def extract_text_content(content: object) -> str:
    try:
        return _extract_text_content_recursive(content, set())
    except RecursionError:
        return _extract_text_content_iterative(content)


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


_COMPACT_SPEAKER_LABEL_PATTERN = re.compile(
    r"^(?:spk|speaker|spkr)[_-]?\d+$",
    flags=re.IGNORECASE,
)
_FREEFORM_SPEAKER_LABEL_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_-]{0,15}$",
)


def _is_colon_token(token: str) -> bool:
    return token in {":", "："}


# Unicode line-break variants that appear mid-token in model output and should
# be removed without inserting whitespace (otherwise adjacent characters in
# no-space scripts like Chinese/Japanese get incorrectly split). ASCII line
# breaks (\n, \r, \v, \f) are still flattened to a space since they typically
# separate words.
_UNICODE_LINE_BREAK_ZERO_WIDTH_RE = re.compile(
    r"[\x1c\x1d\x1e\x85\u2028\u2029]"
)


def _normalize_model_result_newlines(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # The source is one transcription per line, so the output must also stay on
    # a single line. The model may still emit multi-line output. Unicode
    # variants (NEL, LS, PS, FS/GS/RS) are dropped without inserting whitespace
    # so no-space scripts are not artificially split; ASCII line breaks (\n,
    # \r\n, \r, \v, \f) collapse to a single space since they typically mark
    # word boundaries.
    normalized = _UNICODE_LINE_BREAK_ZERO_WIDTH_RE.sub("", text)
    normalized = re.sub(r"\r\n|[\r\v\f]", "\n", normalized)
    normalized = re.sub(r"\s*\n\s*", " ", normalized)
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized.strip()


def normalize_payload_result_newlines(payload: dict) -> None:
    if not isinstance(payload, dict):
        return

    for step_key in _STEP_CHAIN_KEYS:
        step_payload = payload.get(step_key)
        if not isinstance(step_payload, dict):
            continue
        step_result = step_payload.get("result")
        if isinstance(step_result, str):
            step_payload["result"] = _normalize_model_result_newlines(step_result)

    corrected_text = payload.get("corrected_text")
    if isinstance(corrected_text, str):
        payload["corrected_text"] = _normalize_model_result_newlines(corrected_text)


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


def _tokenize_for_content_guards(text: str) -> list[str]:
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
            # Split URL/email into components so content guards can compare
            # individual parts (scheme, host segments, path) against spelled-out
            # ASR tokens like "h t t p kettőspont per per v v v pont vam pont hu".
            # Split on punctuation boundaries, emitting each punct char separately
            # so _is_punct_token can filter them.
            url_text = url_email_match.group(0)
            buf: list[str] = []
            for ch in url_text:
                cat = unicodedata.category(ch)
                if cat.startswith(("P", "S")) or ch in ":/.,@":
                    if buf:
                        tokens.append("".join(buf))
                        buf.clear()
                    tokens.append(ch)
                else:
                    buf.append(ch)
            if buf:
                tokens.append("".join(buf))
            index += len(url_text)
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


def _is_repeated_context_span(
    source_tokens: list[str],
    i1: int,
    i2: int,
    max_pattern_len: int = 3,
) -> bool:
    """Check if source_tokens[i1:i2] is mostly repeated copies of preceding/following context.

    ASR disfluency: "ez egy ez egy ez egy ez egy lét" → "ez egy lét"
    The deleted span ["ez","egy","ez","egy","ez","egy"] is 3 copies of the
    preceding ["ez","egy"].

    When adjacent opcodes are merged by callers, the merged span may have a
    small non-repeating head or tail.  This function tolerates up to *max_pattern_len* leftover tokens at either
    end, as long as the core repeated portion has at least 2 full copies and
    matches the preceding or following context.
    """
    deleted = source_tokens[i1:i2]
    span_len = len(deleted)
    if span_len == 0:
        return False

    for pat_len in range(1, min(max_pattern_len, span_len) + 1):
        # Try the pattern from the start of the span.
        pattern = deleted[:pat_len]
        # Count how many full copies match from the start.
        full_copies = 0
        for k in range(0, span_len, pat_len):
            if deleted[k:k + pat_len] == pattern:
                full_copies += 1
            else:
                break
        repeated_len = full_copies * pat_len
        leftover = span_len - repeated_len
        if full_copies >= 2 and leftover <= max_pattern_len:
            # Verify pattern matches preceding or following context.
            if i1 >= pat_len and source_tokens[i1 - pat_len:i1] == pattern:
                return True
            if i2 + pat_len <= len(source_tokens) and source_tokens[i2:i2 + pat_len] == pattern:
                return True

        # Try the pattern from the end of the span (handles non-repeating head).
        pattern_end = deleted[-pat_len:]
        full_copies_end = 0
        for k in range(span_len - pat_len, -1, -pat_len):
            if deleted[k:k + pat_len] == pattern_end:
                full_copies_end += 1
            else:
                break
        repeated_len_end = full_copies_end * pat_len
        leftover_end = span_len - repeated_len_end
        if full_copies_end >= 2 and leftover_end <= max_pattern_len:
            if i1 >= pat_len and source_tokens[i1 - pat_len:i1] == pattern_end:
                return True
            if i2 + pat_len <= len(source_tokens) and source_tokens[i2:i2 + pat_len] == pattern_end:
                return True

    return False


# Pattern for splitting tokens on URL-delimiter and punctuation boundaries so
# the word-level diff can see spelled-out components separately.
_WORD_DIFF_SPLIT_RE = re.compile(r"(://|[.:,;/])")


def _tokenize_for_word_diff(text: str) -> list[str]:
    """Split text into tokens for word-level diff used by content-guard cleaning.

    Whitespace-splits first, then further splits on URL-delimiter/punctuation
    boundaries so e.g. "http://www.foo" → ["http", "://", "www", ".", "foo"].
    """
    tokens: list[str] = []
    for word in text.lower().split():
        for part in _WORD_DIFF_SPLIT_RE.sub(r" \1 ", word).split():
            if part:
                tokens.append(part)
    return tokens


def _is_conversion_span(words: list[str]) -> bool:
    """True if ``words`` are digit-containing or pure-symbol tokens.

    Such spans represent spelled-out→symbolic/numeric conversions that should
    be tolerated by the content guard (e.g. "kettőspont"→":" "pont"→"."
    "ezer"→"1000").
    """
    for w in words:
        if any(c.isdigit() for c in w):
            return True
        if not any(c.isalpha() for c in w):
            return True
    return False


def _is_disfluency_delete(src_tokens: list[str], start: int, end: int) -> bool:
    """True if ``src_tokens[start:end]`` is a repeated-pattern disfluency.

    Complements :func:`_is_repeated_context_span` for content-guard cleaning:
    uses shorter pattern range (1–4) and checks strict whole-span repetition
    that matches adjacent context.
    """
    deleted = src_tokens[start:end]
    span_len = len(deleted)
    if span_len == 0:
        return False
    for pat_len in range(1, min(4, span_len) + 1):
        if span_len % pat_len != 0:
            continue
        pattern = deleted[:pat_len]
        if all(deleted[k:k + pat_len] == pattern for k in range(0, span_len, pat_len)):
            if start >= pat_len and src_tokens[start - pat_len:start] == pattern:
                return True
            if end + pat_len <= len(src_tokens) and src_tokens[end:end + pat_len] == pattern:
                return True
    return False


def strip_conversion_and_disfluency_spans(
    source_text: str,
    corrected_text: str,
) -> tuple[str, str]:
    """Strip spelled-out→symbol/numeric conversions and repeated-word
    disfluencies from a (source, corrected) pair before content-guard checks.

    Word-level diff identifies ``replace`` spans whose corrected side is
    purely digits or symbols, and ``delete``/``replace`` spans that are
    repeated-pattern disfluencies — those are removed from both sides.
    Returns cleaned (source, corrected).  If all content was conversion,
    returns ``("", "")`` signalling "no content remains to check".
    """
    src_words = _tokenize_for_word_diff(source_text)
    out_words = _tokenize_for_word_diff(corrected_text)
    if not src_words or not out_words:
        return source_text, corrected_text

    matcher = difflib.SequenceMatcher(a=src_words, b=out_words, autojunk=False)
    src_keep: list[str] = []
    out_keep: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and _is_conversion_span(out_words[j1:j2]):
            continue
        if tag in ("delete", "replace") and _is_disfluency_delete(src_words, i1, i2):
            out_keep.extend(out_words[j1:j2])
            continue
        src_keep.extend(src_words[i1:i2])
        out_keep.extend(out_words[j1:j2])

    if not src_keep and not out_keep:
        return "", ""
    return " ".join(src_keep), " ".join(out_keep)


def validate_corrected_text_hallucination(corrected_text: str, source_text: str) -> tuple[bool, str]:
    if not isinstance(corrected_text, str):
        return False, "corrected_text must be a string"
    if not corrected_text.strip():
        return True, ""
    if not isinstance(source_text, str):
        return False, "source_text must be a string"
    if not source_text.strip():
        return False, ""

    source_tokens_raw = _tokenize_for_content_guards(source_text)
    corrected_tokens_raw = _tokenize_for_content_guards(corrected_text)

    corrected_tokens, corrected_speaker_label_signatures = _collect_and_remove_speaker_labels(
        corrected_tokens_raw
    )
    source_tokens = _remove_speaker_labels_by_reference(
        source_tokens_raw,
        corrected_speaker_label_signatures,
    )

    source_tokens = [token for token in source_tokens if token.strip() and not _is_punct_token(token)]
    corrected_tokens = [token for token in corrected_tokens if token.strip() and not _is_punct_token(token)]

    source_folded_stream = [token.casefold() for token in source_tokens]
    corrected_folded_stream = [token.casefold() for token in corrected_tokens]

    matcher = difflib.SequenceMatcher(a=source_folded_stream, b=corrected_folded_stream, autojunk=False)
    max_inserted_tokens = get_hallucination_guard_config()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            net_inserted = j2 - j1
        elif tag == "replace":
            net_inserted = (j2 - j1) - (i2 - i1)
        else:
            continue

        if net_inserted <= max_inserted_tokens:
            continue

        # Space-collapse check: joining corrected tokens == joining source tokens
        # means the change is just whitespace insertion, not hallucination.
        src_joined = "".join(source_folded_stream[i1:i2])
        cor_joined = "".join(corrected_folded_stream[j1:j2])
        if src_joined and cor_joined and difflib.SequenceMatcher(None, src_joined, cor_joined, autojunk=False).ratio() >= 0.5:
            continue

        # Numeral conversion check: if inserted tokens contain digits,
        # the insertion is likely digit expansion from spelled-out numbers.
        if any(any(c.isdigit() for c in t) for t in corrected_folded_stream[j1:j2]):
            continue

        inserted_tokens = corrected_tokens[j1:j2]
        snippet = " ".join(inserted_tokens[:8])
        return False, (
            "corrected_text contains long inserted span: "
            f"inserted_tokens={net_inserted}, "
            f"max_allowed={max_inserted_tokens}, span='{snippet}'"
        )

    return True, ""


def _folded_non_punct_tokens(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = _tokenize_for_content_guards(text)
    return [token.casefold() for token in tokens if token.strip() and not _is_punct_token(token)]


def _no_space_script_char_stream(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []

    chars: list[str] = []
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char).startswith(("P", "S")):
            continue
        if _char_based_script_group(char) in _CHAR_BASED_NO_SPACE_GROUPS:
            chars.append(char)

    return chars


def _no_space_script_char_positions(text: str) -> list[tuple[str, int]]:
    """Like _no_space_script_char_stream but also returns each char's position in *text*."""
    if not isinstance(text, str) or not text:
        return []

    items: list[tuple[str, int]] = []
    for i, char in enumerate(text):
        if char.isspace():
            continue
        if unicodedata.category(char).startswith(("P", "S")):
            continue
        if _char_based_script_group(char) in _CHAR_BASED_NO_SPACE_GROUPS:
            items.append((char, i))

    return items


def _deleted_span_replaced_by_digits(
    source_text: str,
    corrected_text: str,
    src_start: int,
    src_end: int,
) -> bool:
    """Check if the source region [src_start:src_end] was replaced by digits in corrected_text.

    Uses a full-text SequenceMatcher on the local region (with some context)
    to see if the source chars were substituted by digit characters.
    """
    # Expand context window around the deleted span for the diff.
    ctx = 30
    s_lo = max(0, src_start - ctx)
    s_hi = min(len(source_text), src_end + ctx)
    src_window = source_text[s_lo:s_hi]

    # Corresponding corrected window: approximate by same offsets, clamped.
    c_lo = max(0, src_start - ctx)
    c_hi = min(len(corrected_text), src_end + ctx)
    cor_window = corrected_text[c_lo:c_hi]

    if not cor_window:
        return False

    sm = difflib.SequenceMatcher(a=src_window, b=cor_window, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        inserted = cor_window[j1:j2]
        # If the replacement consists mainly of digits (allow some punctuation
        # like commas, periods for formatted numbers, % for percentages),
        # treat as numeral conversion.
        non_digit_non_punct = [c for c in inserted if not c.isdigit() and c not in ".,، %"]
        if len(non_digit_non_punct) == 0 and any(c.isdigit() for c in inserted):
            return True

    return False


def validate_no_long_span_removed_no_space_scripts(
    corrected_text: str,
    source_text: str,
    processing_id: str | None = None,
) -> tuple[bool, str]:
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return True, ""

    source_items = _no_space_script_char_positions(source_text)
    corrected_chars = _no_space_script_char_stream(corrected_text)
    source_chars = [c for c, _ in source_items]
    if not source_chars:
        return True, ""

    min_deleted_no_space_chars = get_long_span_preservation_guard_config()

    matcher = difflib.SequenceMatcher(a=source_chars, b=corrected_chars, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("delete", "replace"):
            continue
        net_deleted = (i2 - i1) - (j2 - j1)
        if net_deleted < min_deleted_no_space_chars:
            continue

        deleted_chars = source_chars[i1:i2]

        if all(c in _CJK_NUMERAL_CHARS for c in deleted_chars):
            continue

        src_start = source_items[i1][1]
        src_end = source_items[i2 - 1][1] + 1
        if _deleted_span_replaced_by_digits(source_text, corrected_text, src_start, src_end):
            continue
        # Disfluency repetition check: if deleted chars are repeated
        # copies of the preceding/following context, it's stutter removal.
        if _is_repeated_context_span(source_chars, i1, i2):
            continue

        snippet = "".join(deleted_chars[:20])
        return (
            False,
            "long no-space-script span was removed "
            f"(deleted_chars={net_deleted}, threshold={min_deleted_no_space_chars}, "
            f"span='{snippet}')",
        )

    return True, ""


def _inserted_span_replaces_digits(
    source_text: str,
    corrected_text: str,
    cor_start: int,
    cor_end: int,
) -> bool:
    """Check if the corrected region [cor_start:cor_end] replaced digits in source_text.

    This is the inverse of _deleted_span_replaced_by_digits: digit → no-space
    script conversion (e.g. 2020 → 二零二零) is not hallucination.
    """
    ctx = 30
    s_lo = max(0, cor_start - ctx)
    s_hi = min(len(source_text), cor_end + ctx)
    src_window = source_text[s_lo:s_hi]

    c_lo = max(0, cor_start - ctx)
    c_hi = min(len(corrected_text), cor_end + ctx)
    cor_window = corrected_text[c_lo:c_hi]

    if not src_window:
        return False

    sm = difflib.SequenceMatcher(a=src_window, b=cor_window, autojunk=False)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        deleted = src_window[i1:i2]
        non_digit_non_punct = [c for c in deleted if not c.isdigit() and c not in ".,، %"]
        if len(non_digit_non_punct) == 0 and any(c.isdigit() for c in deleted):
            return True

    return False


def validate_corrected_text_hallucination_no_space_scripts(
    corrected_text: str,
    source_text: str,
) -> tuple[bool, str]:
    """Detect hallucinated no-space-script characters inserted into corrected text.

    Mirrors validate_no_long_span_removed_no_space_scripts but checks for
    inserted chars (hallucination) instead of deleted chars (truncation).
    """
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return True, ""

    source_chars = _no_space_script_char_stream(source_text)
    corrected_items = _no_space_script_char_positions(corrected_text)
    corrected_chars = [c for c, _ in corrected_items]
    if not corrected_chars:
        return True, ""

    max_inserted_chars = get_hallucination_guard_config()

    matcher = difflib.SequenceMatcher(a=source_chars, b=corrected_chars, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            net_inserted = j2 - j1
        elif tag == "replace":
            net_inserted = (j2 - j1) - (i2 - i1)
        else:
            continue

        if net_inserted <= max_inserted_chars:
            continue

        inserted_chars = corrected_chars[j1:j2]

        # Skip if inserted span is entirely CJK numeral characters — likely
        # converted from Arabic digits (e.g. 2020 → 二零二零).
        if all(c in _CJK_NUMERAL_CHARS for c in inserted_chars):
            continue

        # General numeral conversion check: if the inserted no-space chars
        # replaced digits in the source, it's numeral conversion, not hallucination.
        cor_start = corrected_items[j1][1]
        cor_end = corrected_items[j2 - 1][1] + 1
        if _inserted_span_replaces_digits(source_text, corrected_text, cor_start, cor_end):
            continue

        snippet = "".join(inserted_chars[:20])
        return (
            False,
            "corrected_text contains long inserted no-space-script span: "
            f"inserted_chars={net_inserted}, "
            f"max_allowed={max_inserted_chars}, span='{snippet}'",
        )

    return True, ""


# Content-guard failure categories returned by validate_all_content_guards.
GUARD_CATEGORY_LONG_SPAN = "long_span"           # word-level truncation
GUARD_CATEGORY_HALLUCINATION = "hallucination"   # word-level hallucination
GUARD_CATEGORY_LONG_SPAN_NS = "long_span_ns"     # no-space-script truncation
GUARD_CATEGORY_HALLUCINATION_NS = "hallucination_ns"  # no-space-script hallucination


def validate_all_content_guards(
    corrected_text: str,
    source_text: str,
    processing_id: str | None = None,
    _apply_cleaning: bool = True,
) -> tuple[bool, str, str | None]:
    """Run all 4 content guard checks with shared tokenization.

    Checks long-span removal and hallucination for both space-delimited and
    no-space scripts, tokenizing each text only once per tokenizer.
    Returns ``(True, "", None)`` if all pass, or ``(False, error_message,
    category)`` on first failure, where ``category`` is one of
    ``GUARD_CATEGORY_*`` (long_span / hallucination / long_span_ns /
    hallucination_ns) so callers can dispatch (e.g. repairable vs not).

    When ``_apply_cleaning`` is True (default), first tries the checks on a
    cleaned (source, corrected) pair with spelled-out→symbolic/numeric
    conversions and repeated-word disfluencies removed.  If the cleaned pair
    passes, the overall check passes; otherwise falls through to the raw pair.
    This tolerates e.g. "kettőspont"→":", "h t t p"→"http", "ezer"→"1000".
    """
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return True, "", None

    # Try cleaned pair first — spelled-out→symbolic/numeric conversions and
    # repeated-word disfluencies are removed before the guard so valid edits
    # aren't flagged as truncation/hallucination.
    if _apply_cleaning:
        cleaned_src, cleaned_cor = strip_conversion_and_disfluency_spans(
            source_text, corrected_text,
        )
        if not cleaned_src and not cleaned_cor:
            return True, "", None  # every token was a conversion span
        if (cleaned_src and cleaned_cor
                and (cleaned_src != source_text or cleaned_cor != corrected_text)):
            ok, _err, _cat = validate_all_content_guards(
                cleaned_cor, cleaned_src, processing_id, _apply_cleaning=False,
            )
            if ok:
                return True, "", None
            # Cleaned pair also fails — fall through to raw to produce the
            # most accurate error message.

    min_deleted_tokens = get_long_span_preservation_guard_config()
    max_inserted_tokens = get_hallucination_guard_config()

    # --- Word-level checks (space-delimited scripts) ---
    # Tokenize once for both long-span-removed and hallucination checks.
    source_tokens_raw = _tokenize_for_content_guards(source_text)
    corrected_tokens_raw = _tokenize_for_content_guards(corrected_text)

    # Speaker label handling (for hallucination check).
    corrected_tokens_no_spk, corrected_spk_sigs = _collect_and_remove_speaker_labels(
        corrected_tokens_raw
    )
    source_tokens_no_spk = _remove_speaker_labels_by_reference(
        source_tokens_raw, corrected_spk_sigs,
    )

    source_folded = [t.casefold() for t in source_tokens_no_spk if t.strip() and not _is_punct_token(t)]
    corrected_folded = [t.casefold() for t in corrected_tokens_no_spk if t.strip() and not _is_punct_token(t)]

    if source_folded:
        matcher = difflib.SequenceMatcher(a=source_folded, b=corrected_folded, autojunk=False)
        opcodes = matcher.get_opcodes()

        # Check 1: long span removed (word-level).
        for tag, i1, i2, j1, j2 in opcodes:
            if tag not in ("delete", "replace"):
                continue
            net_deleted = (i2 - i1) - (j2 - j1)
            if net_deleted > min_deleted_tokens:
                # Space-collapse check: if joining source tokens is similar to joining
                # corrected tokens, the change is mostly whitespace removal.
                src_joined = "".join(source_folded[i1:i2])
                cor_joined = "".join(corrected_folded[j1:j2])
                if src_joined and cor_joined and difflib.SequenceMatcher(None, src_joined, cor_joined, autojunk=False).ratio() >= 0.5:
                    continue
                # Numeral conversion check: if corrected tokens contain digits,
                # the deletion is likely spelled-out number → digit conversion.
                if any(any(c.isdigit() for c in t) for t in corrected_folded[j1:j2]):
                    continue
                # URL reconstruction check: if the corrected side starts with a URL
                # prefix, the deletion is spelled-out URL components → actual URL.
                if cor_joined.startswith(("http", "www", "ftp")):
                    continue
                # Disfluency repetition check: if deleted tokens are repeated
                # copies of the preceding/following context, it's stutter removal.
                if _is_repeated_context_span(source_folded, i1, i2):
                    continue
                deleted = source_folded[i1:i2]
                snippet = " ".join(deleted[:12])
                return False, (
                    f"long span was removed (deleted_tokens={net_deleted}, span='{snippet}')"
                ), GUARD_CATEGORY_LONG_SPAN

        # Check 2: hallucination (word-level).
        corrected_tokens_filtered = [t for t in corrected_tokens_no_spk if t.strip() and not _is_punct_token(t)]
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "insert":
                net_inserted = j2 - j1
            elif tag == "replace":
                net_inserted = (j2 - j1) - (i2 - i1)
            else:
                continue
            if net_inserted <= max_inserted_tokens:
                continue
            # Space-collapse check: joining corrected tokens == joining source tokens
            # means the change is just whitespace insertion, not hallucination.
            src_joined = "".join(source_folded[i1:i2])
            cor_joined = "".join(corrected_folded[j1:j2])
            if src_joined and cor_joined and difflib.SequenceMatcher(None, src_joined, cor_joined, autojunk=False).ratio() >= 0.5:
                continue
            # Numeral conversion check: if inserted tokens contain digits,
            # the insertion is likely digit expansion from spelled-out numbers.
            if any(any(c.isdigit() for c in t) for t in corrected_folded[j1:j2]):
                continue
            inserted = corrected_tokens_filtered[j1:j2]
            snippet = " ".join(inserted[:8])
            return False, (
                f"corrected_text contains long inserted span: "
                f"inserted_tokens={net_inserted}, max_allowed={max_inserted_tokens}, span='{snippet}'"
            ), GUARD_CATEGORY_HALLUCINATION

    # --- Char-level checks (no-space scripts: CJK, Thai, etc.) ---
    # Tokenize once for both no-space checks.
    source_ns_items = _no_space_script_char_positions(source_text)
    source_ns_chars = [c for c, _ in source_ns_items]
    corrected_ns_items = _no_space_script_char_positions(corrected_text)
    corrected_ns_chars = [c for c, _ in corrected_ns_items]

    if source_ns_chars or corrected_ns_chars:
        matcher_ns = difflib.SequenceMatcher(a=source_ns_chars, b=corrected_ns_chars, autojunk=False)
        opcodes_ns = matcher_ns.get_opcodes()

        # Check 3: long no-space span removed.
        if source_ns_chars:
            for tag, i1, i2, j1, j2 in opcodes_ns:
                if tag not in ("delete", "replace"):
                    continue
                net_deleted = (i2 - i1) - (j2 - j1)
                if net_deleted < min_deleted_tokens:
                    continue
                deleted = source_ns_chars[i1:i2]
                if all(c in _CJK_NUMERAL_CHARS for c in deleted):
                    continue
                src_start = source_ns_items[i1][1]
                src_end = source_ns_items[i2 - 1][1] + 1
                if _deleted_span_replaced_by_digits(source_text, corrected_text, src_start, src_end):
                    continue
                # Disfluency repetition check.
                if _is_repeated_context_span(source_ns_chars, i1, i2):
                    continue
                snippet = "".join(deleted[:20])
                return False, (
                    f"long no-space-script span was removed "
                    f"(deleted_chars={net_deleted}, threshold={min_deleted_tokens}, span='{snippet}')"
                ), GUARD_CATEGORY_LONG_SPAN_NS

        # Check 4: hallucinated no-space chars inserted.
        if corrected_ns_chars:
            for tag, i1, i2, j1, j2 in opcodes_ns:
                if tag == "insert":
                    net_inserted = j2 - j1
                elif tag == "replace":
                    net_inserted = (j2 - j1) - (i2 - i1)
                else:
                    continue
                if net_inserted <= max_inserted_tokens:
                    continue
                inserted = corrected_ns_chars[j1:j2]
                if all(c in _CJK_NUMERAL_CHARS for c in inserted):
                    continue
                cor_start = corrected_ns_items[j1][1]
                cor_end = corrected_ns_items[j2 - 1][1] + 1
                if _inserted_span_replaces_digits(source_text, corrected_text, cor_start, cor_end):
                    continue
                snippet = "".join(inserted[:20])
                return False, (
                    f"corrected_text contains long inserted no-space-script span: "
                    f"inserted_chars={net_inserted}, max_allowed={max_inserted_tokens}, span='{snippet}'"
                ), GUARD_CATEGORY_HALLUCINATION_NS

    return True, "", None


def validate_inactive_step_edits_empty(
    payload: dict,
    active_step_keys: set[str] | None,
) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return True, ""
    if not isinstance(active_step_keys, set):
        return True, ""

    inactive_steps_with_edits: list[str] = []
    for step_key in _STEP_CHAIN_KEYS:
        if step_key in active_step_keys:
            continue

        step_payload = payload.get(step_key)
        if not isinstance(step_payload, dict):
            continue

        edits = step_payload.get("edits")
        if isinstance(edits, list) and len(edits) > 0:
            inactive_steps_with_edits.append(step_key)

    if inactive_steps_with_edits:
        return (
            False,
            "inactive chain step contains non-empty edits "
            f"(steps={', '.join(inactive_steps_with_edits)})",
        )

    return True, ""


def validate_no_long_span_removed(
    corrected_text: str,
    source_text: str,
    processing_id: str | None = None,
) -> tuple[bool, str]:
    if not isinstance(corrected_text, str) or not isinstance(source_text, str):
        return True, ""

    source_tokens = _folded_non_punct_tokens(source_text)
    corrected_tokens = _folded_non_punct_tokens(corrected_text)
    if not source_tokens:
        return True, ""

    matcher = difflib.SequenceMatcher(a=source_tokens, b=corrected_tokens, autojunk=False)

    min_deleted_tokens = get_long_span_preservation_guard_config()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("delete", "replace"):
            continue
        net_deleted = (i2 - i1) - (j2 - j1)
        if net_deleted <= min_deleted_tokens:
            continue

        # Space-collapse check: joining source tokens is similar to joining
        # corrected tokens means the change is mostly whitespace removal.
        src_joined = "".join(source_tokens[i1:i2])
        cor_joined = "".join(corrected_tokens[j1:j2])
        if src_joined and cor_joined and difflib.SequenceMatcher(None, src_joined, cor_joined, autojunk=False).ratio() >= 0.5:
            continue

        # Numeral conversion check: if corrected tokens contain digits,
        # the deletion is likely spelled-out number → digit conversion.
        if any(any(c.isdigit() for c in t) for t in corrected_tokens[j1:j2]):
            continue

        # URL reconstruction check: if the corrected side starts with a URL
        # prefix, the deletion is spelled-out URL components → actual URL.
        if cor_joined.startswith(("http", "www", "ftp")):
            continue

        # Disfluency repetition check: if deleted tokens are repeated
        # copies of the preceding/following context, it's stutter removal.
        if _is_repeated_context_span(source_tokens, i1, i2):
            continue

        deleted_tokens = source_tokens[i1:i2]
        snippet = " ".join(deleted_tokens[:12])
        return (
            False,
            "long span was removed "
            f"(deleted_tokens={net_deleted}, "
            f"span='{snippet}')",
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

    seg_start = payload.get("seg_start")
    if seg_start is not None:
        if not isinstance(seg_start, str):
            return False, 'seg_start must be one of "high", "medium", "low"'
        if seg_start not in {"high", "medium", "low"}:
            return False, 'seg_start must be one of "high", "medium", "low"'

    seg_end = payload.get("seg_end")
    if seg_end is not None:
        if not isinstance(seg_end, str):
            return False, 'seg_end must be one of "high", "medium", "low"'
        if seg_end not in {"high", "medium", "low"}:
            return False, 'seg_end must be one of "high", "medium", "low"'

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


def is_long_span_preservation_validation_error(validation_error: str | None) -> bool:
    if not isinstance(validation_error, str):
        return False
    return "Long-span preservation check failed:" in validation_error


def is_json_recursion_validation_error(validation_error: str | None) -> bool:
    if not isinstance(validation_error, str):
        return False
    return validation_error.startswith("JSON parse recursion error:")


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

    lines = trimmed.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines and all(line.lstrip().startswith(">") for line in non_empty_lines):
        unquoted = "\n".join(re.sub(r"^\s*>\s?", "", line) for line in lines).strip()
        if unquoted and unquoted != trimmed:
            return strip_markdown_code_fence(unquoted)

    return trimmed


# Regex for \u escapes with 5+ hex digits — these are invalid JSON and
# indicate the model zero-padded a codepoint (e.g. \u000159 for U+0159 ř).
_MALFORMED_UESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{5,})")


def fix_malformed_unicode_escapes(json_str: str) -> str:
    r"""Normalize \u escapes with 5+ hex digits to valid 4-digit \uXXXX.

    Only touches sequences that are already invalid JSON (5+ hex digits).
    Valid 4-digit \uXXXX sequences are left untouched so json.loads() can
    decode them normally.

    The model often zero-pads codepoints (e.g. \u000159 for U+0159 ř).
    When trailing hex chars are actually literal text (e.g. \u000159e = ř + "e"),
    we find the shortest prefix that yields a printable BMP character.

    >>> fix_malformed_unicode_escapes(r'"p\u000159\u0000edli\u000161"')
    '"p\\u0159\\u00edli\\u0161"'
    >>> fix_malformed_unicode_escapes(r'"\u000159edn\u0000edc"')
    '"\\u0159edn\\u00edc"'
    """
    def _normalize(m: re.Match) -> str:
        hex_digits = m.group(1)
        for n in range(4, len(hex_digits) + 1):
            cp = int(hex_digits[:n], 16)
            if cp >= 0x20 and cp <= 0xFFFF and not (0xD800 <= cp <= 0xDFFF):
                remaining = hex_digits[n:]
                return f"\\u{cp:04x}" + remaining
        return m.group(0)

    return _MALFORMED_UESCAPE_RE.sub(_normalize, json_str)


def parse_and_validate_json(content: str) -> tuple[dict | None, str | None]:
    trimmed = strip_markdown_code_fence(content)

    extracted = extract_first_json_object(trimmed)
    candidate = extracted if extracted is not None else trimmed
    candidate = fix_malformed_unicode_escapes(candidate)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        is_json_like, json_like_reason = is_json_like_repairable_response(trimmed)
        if not is_json_like:
            return None, _mark_non_repairable_validation_error(
                f"{json_like_reason}; JSON parse error: {error}; raw content: {content[:10]} ..."
            )
        return None, f"JSON parse error: {error}"
    except RecursionError as error:
        return None, f"JSON parse recursion error: {error}"

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
    active_step_keys: set[str] | None = None,
    allow_long_span_preservation_failure: bool = False,
) -> tuple[dict | None, str | None, str]:
    content = raw_content.strip()
    if not content:
        return None, _mark_non_repairable_validation_error("Model returned empty output."), content

    payload, validation_error = parse_and_validate_json(content)
    if payload is None:
        return None, validation_error, content

    normalize_payload_result_newlines(payload)

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
    # Only revert first-token casing when seg_start is not "high" (LLM shouldn't have changed it).
    seg_start_value = payload.get("seg_start") if isinstance(payload, dict) else None
    should_preserve_casing = (
        not skip_first_token_casing_preservation
        and seg_start_value != "high"
    )
    if should_preserve_casing:
        corrected_text, casing_normalized = preserve_first_token_casing(
            corrected_text,
            source_text,
            no_touch_tokens,
        )
    corrected_text, spacing_normalized = normalize_char_based_spacing_input(corrected_text)
    # Only revert terminal punctuation when seg_end is not "high" (LLM shouldn't have changed it).
    seg_end_value = payload.get("seg_end") if isinstance(payload, dict) else None
    punctuation_normalized = False
    if seg_end_value != "high":
        corrected_text, punctuation_normalized = preserve_terminal_punctuation(
            corrected_text,
            source_text,
        )

    if (
        casing_normalized
        or spacing_normalized
        or punctuation_normalized
    ) and isinstance(payload, dict):
        payload["corrected_text"] = corrected_text

    inactive_edits_ok, inactive_edits_error = validate_inactive_step_edits_empty(
        payload,
        active_step_keys,
    )
    if not inactive_edits_ok:
        return None, f"Inactive-step edits check failed: {inactive_edits_error}", content

    # Shared content guards (with spelled-out→symbol/numeral/disfluency cleaning).
    # Dispatch on category so hallucination stays non-repairable and
    # long-span failures honour ``allow_long_span_preservation_failure``.
    guard_ok, guard_err, guard_category = validate_all_content_guards(
        corrected_text, source_text, processing_id,
    )
    if guard_ok:
        return payload, None, content

    if guard_category in (GUARD_CATEGORY_LONG_SPAN, GUARD_CATEGORY_LONG_SPAN_NS):
        if allow_long_span_preservation_failure:
            print(
                f"[{processing_id}] WARNING: Long-span preservation check failed "
                f"but accepting final retry result: {guard_err}"
            )
            return payload, None, content
        return None, f"Long-span preservation check failed: {guard_err}", content

    if guard_category == GUARD_CATEGORY_HALLUCINATION:
        return None, _mark_non_repairable_validation_error(
            f"Hallucination check failed: corrected_text contains long inserted span: {guard_err}"
        ), content

    if guard_category == GUARD_CATEGORY_HALLUCINATION_NS:
        return None, _mark_non_repairable_validation_error(
            f"Hallucination check (no-space scripts) failed: {guard_err}"
        ), content

    # Unknown category — fall back to non-repairable with raw error.
    return None, _mark_non_repairable_validation_error(
        f"Content guard failed: {guard_err}"
    ), content


def build_empty_payload() -> dict:
    payload = {
        "tokenization": {"tokens": []},
        "translation": "",
        "aggressiveness_level": "low",
        "speaker_scope": "unknown",
        "seg_start": "medium",
        "seg_end": "medium",
        "no_touch_tokens": [],
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
    max_input_chars_per_call: int,
) -> None:
    min_deleted_tokens = get_long_span_preservation_guard_config()
    max_inserted_tokens = get_hallucination_guard_config()
    print(f"Using patch prompt file: {prompt_template_path}")
    print(f"Using repair prompt file: {repair_prompt_template_path}")
    print(f"Concurrency: {concurrency}")
    print(f"Timeout: {timeout_seconds}s, retries: {timeout_retries}")
    print(f"Empty-result retries: {empty_result_retries}")
    if max_input_chars_per_call > 0:
        print(f"Long-input segmentation: enabled, max chars per call={max_input_chars_per_call}")
    else:
        print("Long-input segmentation: disabled")
    print(
        "Long-span preservation guard: "
        f"min_deleted_tokens={min_deleted_tokens}"
    )
    print(
        "Hallucination guard: "
        f"max_inserted_tokens={max_inserted_tokens}"
    )


def _find_segment_cut_index(text: str, start: int, max_chars: int) -> int:
    text_length = len(text)
    hard_limit = min(text_length, start + max_chars)
    if hard_limit >= text_length:
        return text_length

    min_candidate = start + max(1, max_chars // 2)
    window = text[start:hard_limit]

    def _candidate_from_find(fragment: str) -> int | None:
        index = window.rfind(fragment)
        if index < 0:
            return None
        candidate = start + index + len(fragment)
        if candidate < min_candidate:
            return None
        return candidate

    for marker in ("\n\n", "\r\n\r\n", "\n", "\r\n"):
        candidate = _candidate_from_find(marker)
        if candidate is not None:
            return _stabilize_segment_cut_index(text, start, candidate, hard_limit)

    # Sentence punctuation followed by whitespace (e.g. ". ", "! ", "? ") —
    # strong sentence boundary that won't split decimals (3.14) or
    # abbreviations (U.S.A.).
    # Additional guard: only accept when both sides of the boundary are
    # language (letter) characters — reject when a digit or symbol is
    # adjacent (e.g. "3. 14" or "$. x").
    sentence_ws_candidates: list[int] = []
    for match in _SEGMENT_BOUNDARY_PATTERN.finditer(window):
        end_pos = match.end()
        # Check if the match is followed by whitespace (or is at the window end).
        if end_pos < len(window) and window[end_pos].isspace():
            # Verify language chars around the boundary.
            char_before = window[match.start() - 1] if match.start() > 0 else ""
            # Skip past whitespace to find the first non-space char after.
            after_idx = end_pos + 1
            while after_idx < len(window) and window[after_idx].isspace():
                after_idx += 1
            char_after = window[after_idx] if after_idx < len(window) else ""
            if char_before and not char_before.isalpha():
                continue
            if char_after and not char_after.isalpha():
                continue
            candidate = start + end_pos
            if candidate >= min_candidate:
                sentence_ws_candidates.append(candidate)
    if sentence_ws_candidates:
        return _stabilize_segment_cut_index(text, start, sentence_ws_candidates[-1], hard_limit)

    # Plain whitespace with letters on both sides — clean word boundary that
    # avoids splitting numeric/symbolic tokens like "3 .14" or "$ 100".
    for index in range(len(window) - 1, -1, -1):
        if window[index].isspace():
            char_before = window[index - 1] if index > 0 else ""
            after_idx = index + 1
            while after_idx < len(window) and window[after_idx].isspace():
                after_idx += 1
            char_after = window[after_idx] if after_idx < len(window) else ""
            if char_before and not char_before.isalpha():
                continue
            if char_after and not char_after.isalpha():
                continue
            candidate = start + index + 1
            if candidate >= min_candidate:
                return _stabilize_segment_cut_index(text, start, candidate, hard_limit)

    # Prefer whitespace boundaries over bare sentence punctuation to avoid splitting
    # decimal numbers (3.14), abbreviations (Dr.), or compound tokens (U.S.A.).
    for index in range(len(window) - 1, -1, -1):
        if window[index].isspace():
            candidate = start + index + 1
            if candidate >= min_candidate:
                return _stabilize_segment_cut_index(text, start, candidate, hard_limit)

    # Fall back to sentence-ending punctuation if no whitespace was found.
    sentence_candidates: list[int] = []
    for match in _SEGMENT_BOUNDARY_PATTERN.finditer(window):
        candidate = start + match.end()
        if candidate >= min_candidate:
            sentence_candidates.append(candidate)
    if sentence_candidates:
        return _stabilize_segment_cut_index(text, start, sentence_candidates[-1], hard_limit)

    # Last resort: any sentence boundary character.
    for index in range(len(window) - 1, -1, -1):
        if window[index] in _SEGMENT_BOUNDARY_SENTENCE_CHAR_SET:
            candidate = start + index + 1
            if candidate >= min_candidate:
                return _stabilize_segment_cut_index(text, start, candidate, hard_limit)

    return _stabilize_segment_cut_index(text, start, hard_limit, hard_limit)


def _is_unicode_join_or_modifier(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    code = ord(char)
    category = unicodedata.category(char)
    if category in {"Mn", "Mc", "Me"}:
        return True
    if char in {"\u200d", "\u200c"}:
        return True
    if 0xFE00 <= code <= 0xFE0F:
        return True
    if 0xE0100 <= code <= 0xE01EF:
        return True
    if 0x1F3FB <= code <= 0x1F3FF:
        return True
    return False


def _is_regional_indicator(char: str) -> bool:
    if not isinstance(char, str) or len(char) != 1:
        return False
    code = ord(char)
    return 0x1F1E6 <= code <= 0x1F1FF


def _splits_regional_indicator_pair(text: str, cut_index: int) -> bool:
    if not isinstance(text, str):
        return False
    if cut_index <= 0 or cut_index >= len(text):
        return False
    if not (_is_regional_indicator(text[cut_index - 1]) and _is_regional_indicator(text[cut_index])):
        return False

    run_start = cut_index - 1
    while run_start - 1 >= 0 and _is_regional_indicator(text[run_start - 1]):
        run_start -= 1

    run_length_before_cut = cut_index - run_start
    return run_length_before_cut % 2 == 1


def _stabilize_segment_cut_index(text: str, start: int, candidate: int, hard_limit: int) -> int:
    text_length = len(text)
    minimum = start + 1
    cut_index = max(minimum, min(candidate, hard_limit))

    while cut_index < text_length:
        previous_char = text[cut_index - 1] if cut_index - 1 >= 0 else ""
        current_char = text[cut_index]
        if previous_char == "\u200d" or _is_unicode_join_or_modifier(current_char):
            if cut_index >= hard_limit:
                break
            cut_index += 1
            continue
        if _splits_regional_indicator_pair(text, cut_index):
            if cut_index >= hard_limit:
                break
            cut_index += 1
            continue
        break

    while cut_index > start:
        previous_char = text[cut_index - 1] if cut_index - 1 >= 0 else ""
        current_is_modifier = cut_index < text_length and _is_unicode_join_or_modifier(text[cut_index])
        if previous_char == "\u200d" or current_is_modifier:
            cut_index -= 1
            continue
        break

    while (
        cut_index < text_length
        and cut_index > start
        and _splits_regional_indicator_pair(text, cut_index)
    ):
        cut_index -= 1

    if cut_index <= start:
        return min(text_length, start + 1)
    return cut_index


def take_next_transcription_segment_for_llm(
    transcription: str,
    start_offset: int,
    max_chars_per_call: int,
) -> tuple[str, int]:
    if not isinstance(transcription, str):
        return "", 0

    text_length = len(transcription)
    start = max(0, min(int(start_offset), text_length))
    if start >= text_length:
        return "", text_length

    max_chars = max(0, int(max_chars_per_call))
    if max_chars == 0 or (text_length - start) <= max_chars * 1.5:
        return transcription[start:], text_length

    cut_index = _find_segment_cut_index(transcription, start, max_chars)
    if cut_index <= start:
        cut_index = min(text_length, start + max_chars)

    return transcription[start:cut_index], cut_index


def split_transcription_for_llm(transcription: str, max_chars_per_call: int) -> list[str]:
    if not isinstance(transcription, str):
        return [""]

    segments: list[str] = []
    start = 0
    text_length = len(transcription)
    while start < text_length:
        segment, cut_index = take_next_transcription_segment_for_llm(
            transcription,
            start,
            max_chars_per_call,
        )
        if cut_index <= start:
            break
        segments.append(segment)
        start = cut_index

    return segments if segments else [transcription]


def _needs_space_between_segment_parts(left: str, right: str) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if not left or not right:
        return False

    left_char = left[-1]
    right_char = right[0]
    if left_char.isspace() or right_char.isspace():
        return False

    # Same-script char-based languages (CJK, Thai, etc.) don't use spaces.
    left_group = _char_based_script_group(left_char)
    right_group = _char_based_script_group(right_char)
    if (
        left_group is not None
        and right_group is not None
        and left_group == right_group
        and left_group in _CHAR_BASED_NO_SPACE_GROUPS
    ):
        return False

    # At segment boundaries, insert a space unless both sides are punctuation/symbols
    # (e.g. ")(" or "...—" should not get a space).
    left_category = unicodedata.category(left_char)
    right_category = unicodedata.category(right_char)
    left_is_punct = left_category.startswith(("P", "S"))
    right_is_punct = right_category.startswith(("P", "S"))
    if left_is_punct and right_is_punct:
        return False

    # If at least one side has a word-like character, insert a space.
    return True


def join_segment_text_parts(parts: list[str]) -> str:
    merged_parts: list[str] = []
    previous_part = ""

    for part in parts:
        if not isinstance(part, str) or not part:
            continue

        if merged_parts and _needs_space_between_segment_parts(previous_part, part):
            merged_parts.append(" ")

        merged_parts.append(part)
        previous_part = part

    return "".join(merged_parts)


def merge_segment_payloads(
    segment_payloads: list[dict],
    segment_sources: list[str],
) -> dict | None:
    if not segment_payloads:
        return None

    merged_payload = build_empty_payload()

    merged_tokens: list[str] = []
    merged_no_touch_tokens: list[str] = []
    seen_no_touch_tokens: set[str] = set()
    merged_translations: list[str] = []
    merged_corrected_parts: list[str] = []
    merged_step_results: dict[str, list[str]] = {step_key: [] for step_key in _STEP_CHAIN_KEYS}
    merged_step_edits: dict[str, list[list[str]]] = {step_key: [] for step_key in _STEP_CHAIN_KEYS}

    aggressiveness_rank = {"low": 0, "medium": 1, "high": 2}
    ranked_aggressiveness_values: list[int] = []
    saw_single_scope = False
    saw_multi_scope = False

    for payload in segment_payloads:
        if not isinstance(payload, dict):
            return None

        tokenization = payload.get("tokenization")
        tokens = tokenization.get("tokens") if isinstance(tokenization, dict) else None
        if isinstance(tokens, list):
            for token in tokens:
                if isinstance(token, str):
                    merged_tokens.append(token)

        translation = payload.get("translation")
        if isinstance(translation, str) and translation:
            merged_translations.append(translation)

        aggressiveness_level = payload.get("aggressiveness_level")
        if isinstance(aggressiveness_level, str) and aggressiveness_level in aggressiveness_rank:
            ranked_aggressiveness_values.append(aggressiveness_rank[aggressiveness_level])

        speaker_scope = payload.get("speaker_scope")
        if speaker_scope == "single":
            saw_single_scope = True
        elif speaker_scope == "multi":
            saw_multi_scope = True

        no_touch_tokens = payload.get("no_touch_tokens")
        if isinstance(no_touch_tokens, list):
            for token in no_touch_tokens:
                if isinstance(token, str) and token not in seen_no_touch_tokens:
                    seen_no_touch_tokens.add(token)
                    merged_no_touch_tokens.append(token)

        for step_key in _STEP_CHAIN_KEYS:
            step_payload = payload.get(step_key)
            if not isinstance(step_payload, dict):
                continue
            step_result = step_payload.get("result")
            if isinstance(step_result, str):
                merged_step_results[step_key].append(step_result)
            step_edits = step_payload.get("edits")
            if isinstance(step_edits, list):
                for pair in step_edits:
                    if (
                        isinstance(pair, list)
                        and len(pair) == 2
                        and isinstance(pair[0], str)
                        and isinstance(pair[1], str)
                    ):
                        merged_step_edits[step_key].append([pair[0], pair[1]])

        corrected_text = payload.get("corrected_text")
        if isinstance(corrected_text, str):
            merged_corrected_parts.append(corrected_text)

    merged_payload["tokenization"]["tokens"] = merged_tokens
    merged_payload["translation"] = "\n\n".join(merged_translations) if merged_translations else ""

    if ranked_aggressiveness_values:
        max_rank = max(ranked_aggressiveness_values)
        reverse_rank = {rank: level for level, rank in aggressiveness_rank.items()}
        merged_payload["aggressiveness_level"] = reverse_rank.get(max_rank, "low")
    else:
        merged_payload["aggressiveness_level"] = "low"

    if saw_multi_scope:
        merged_payload["speaker_scope"] = "multi"
    elif saw_single_scope:
        merged_payload["speaker_scope"] = "single"
    else:
        merged_payload["speaker_scope"] = "unknown"

    # For merged segments: use first segment's start prediction, last segment's end prediction.
    if segment_payloads:
        first_starts = segment_payloads[0].get("seg_start")
        merged_payload["seg_start"] = first_starts if isinstance(first_starts, str) else "medium"
        last_ends = segment_payloads[-1].get("seg_end")
        merged_payload["seg_end"] = last_ends if isinstance(last_ends, str) else "medium"

    merged_payload["no_touch_tokens"] = merged_no_touch_tokens

    for step_key in _STEP_CHAIN_KEYS:
        merged_payload[step_key]["edits"] = merged_step_edits[step_key]
        merged_payload[step_key]["result"] = join_segment_text_parts(merged_step_results[step_key])

    merged_payload["corrected_text"] = join_segment_text_parts(merged_corrected_parts)

    if segment_sources:
        merged_payload["source_text"] = join_segment_text_parts(segment_sources)

    is_valid, _validation_error = validate_patch_payload(merged_payload)
    if not is_valid:
        return None

    return merged_payload


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
    global_offset: int = 0,
    global_total: int | None = None,
) -> int:
    total = len(transcriptions)
    display_total = global_total if global_total is not None else total
    semaphore = asyncio.Semaphore(concurrency)

    async def process_item_guarded(index: int, transcription: str) -> None:
        async with semaphore:
            display_index = global_offset + index
            print_processing_progress(display_index, display_total)
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

    # Detect encoding: try UTF-8 first, fall back to common legacy encodings.
    # Single bulk read is fastest even for OneDrive/network-backed filesystems.
    try:
        raw_text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = None
        for candidate in ("utf-8-sig", "cp1250", "cp1252", "latin-1"):
            try:
                raw_text = input_path.read_text(encoding=candidate)
                print(f"Input file is not UTF-8; using fallback encoding: {candidate}")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if raw_text is None:
            print(f"Could not decode input file with any supported encoding: {input_path}")
            return [], [], []

    if not raw_text:
        return [], [], []

    if input_path.suffix.lower() == ".tsv":
        transcriptions: list[str] = []
        source_filenames: list[str | None] = []

        print(f"Parsing TSV ({len(raw_text):,} chars)...")
        first_row_is_header = False
        lines = raw_text.split('\n')
        if lines and lines[-1] == '':
            lines.pop()
        del raw_text  # Free memory before building the row list.
        total_lines = len(lines)
        # For large files, store only prefix columns (without text) to save memory.
        _store_full_rows = total_lines <= 500_000
        source_rows: list[list[str] | None] = []
        if not _store_full_rows:
            print(f"  Large file ({total_lines:,} lines): storing prefix columns only to save memory.")
        reader = csv.reader(lines, delimiter="\t")
        for row_index, row in enumerate(reader):
            if row_index > 0 and row_index % 1_000_000 == 0:
                print(f"  Parsed {row_index:,}/{total_lines:,} rows...")
            row = [unicodedata.normalize("NFC", cell) for cell in row]
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
            if _store_full_rows:
                source_rows.append(list(row))
            elif len(row) >= 2:
                # Store prefix columns + empty placeholder for text to preserve
                # all columns while avoiding memory cost of storing transcription text.
                source_rows.append(row[:-1] + [""])
            else:
                source_rows.append(None)

        # Free the lines list now that parsing is done.
        del lines

        if first_row_is_header and not transcriptions:
            print("TSV input contains only header and no data rows.")

        return transcriptions, source_filenames, source_rows

    # Non-TSV: try JSON first, then plain text line-by-line (all in-memory).
    stripped = raw_text.strip()
    is_json = stripped[:1] in ("{", "[")

    if is_json:
        raw_text = unicodedata.normalize("NFC", raw_text)
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

    # Plain text: process from in-memory raw_text, apply NFC per-line.
    print(f"Parsing plain text ({len(raw_text):,} chars)...")
    transcriptions: list[str] = []
    source_filenames: list[str | None] = []
    source_rows: list[list[str] | None] = []
    lines = raw_text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    del raw_text
    total_lines = len(lines)
    for line_index, raw_line in enumerate(lines):
        if line_index > 0 and line_index % 1_000_000 == 0:
            print(f"  Parsed {line_index:,}/{total_lines:,} lines...")
        line = unicodedata.normalize("NFC", raw_line)

        if line.lstrip().startswith("#"):
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
    del lines
    return transcriptions, source_filenames, source_rows


def is_input_comment_line(transcription: str) -> bool:
    return isinstance(transcription, str) and transcription.lstrip().startswith("#")


# Emoji stripping: covers all major emoji Unicode ranges.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess, extended-A
    "\U0001FA70-\U0001FAFF"  # extended-B
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000023EA-\U000023F3"  # misc symbols
    "\U0000231A-\U0000231B"
    "\U00002328"
    "\U000023CF"
    "\U000023E9-\U000023F3"
    "\U000023F8-\U000023FA"
    "\U000025AA-\U000025AB"
    "\U000025B6"
    "\U000025C0"
    "\U000025FB-\U000025FE"
    "\U00002600-\U000027BF"
    "\U00002934-\U00002935"
    "\U00002B05-\U00002B07"
    "\U00002B1B-\U00002B1C"
    "\U00002B50"
    "\U00002B55"
    "\U00003030"
    "\U0000303D"
    "\U00003297"
    "\U00003299"
    "\U0001F3FB-\U0001F3FF"  # skin tone modifiers
    "]+",
    flags=re.UNICODE,
)


def strip_emojis(text: str) -> tuple[str, bool]:
    """Remove all emoji characters from text. Returns (cleaned, changed)."""
    if not isinstance(text, str) or not text:
        return text, False
    cleaned = _EMOJI_PATTERN.sub("", text)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    return cleaned, cleaned != text.strip()


def insert_spaces_at_script_boundaries(text: str) -> tuple[str, bool]:
    """Insert a space at boundaries between different script/digit categories.

    Handles two boundary types:
      1) Latin ↔ non-Latin letters
      2) digit ↔ non-Latin letters

    Digit ↔ Latin boundaries are intentionally skipped (e.g. "abc123def"
    stays as-is) because they frequently appear in natural text like
    "3km", "mp3", "V2" where no space is expected.

    Limitation: does not insert spaces between two different non-Latin
    scripts (e.g. Cyrillic↔CJK "мир你好" stays as-is). All non-Latin
    scripts are treated as a single "nonlatin" category. This is rarely
    an issue in ASR data.

    Punctuation, symbols, and decimal points reset the boundary tracker
    so no spaces are inserted around them.
    E.g. "hello你好world" -> "hello 你好 world",
         "123你好" -> "123 你好",
         "abc123def" -> "abc123def",  (digit↔Latin, no space)
         "abc.123" -> "abc.123",  (dot resets, both sides are same-cat after reset)
         "abc,你好" -> "abc,你好".  (comma resets)
    """
    if not isinstance(text, str) or len(text) < 2:
        return text, False

    parts: list[str] = []
    changed = False
    # Categories: "latin", "nonlatin", "digit", or None (space/punct/symbol)
    prev_cat: str | None = None

    for ch in text:
        if ch.isdigit():
            cur_cat = "digit"
        elif ch.isalpha():
            cur_cat = "nonlatin" if _is_non_latin_letter(ch) else "latin"
        else:
            cat = unicodedata.category(ch)
            if cat in {"Mn", "Mc", "Me"}:
                # Combining marks inherit the previous category (e.g. Thai
                # vowel marks attached to a Thai consonant).
                parts.append(ch)
                continue
            # Spaces, punctuation, symbols, decimal points — reset category.
            parts.append(ch)
            prev_cat = None
            continue

        if prev_cat is not None and cur_cat != prev_cat:
            # Skip digit ↔ Latin boundaries.
            is_digit_latin = (
                (prev_cat == "digit" and cur_cat == "latin")
                or (prev_cat == "latin" and cur_cat == "digit")
            )
            if not is_digit_latin and parts and not parts[-1][-1:].isspace():
                parts.append(" ")
                changed = True
        parts.append(ch)
        prev_cat = cur_cat

    if not changed:
        return text, False
    return "".join(parts), True


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


# ---------------------------------------------------------------------------
# Noise token set (used by post-processing)
# ---------------------------------------------------------------------------

# Noise / non-speech tokens whose removal is always safe.
_PREPROCESSING_NOISE_TOKENS: set[str] = {t.lower() for t in [
    "_bg noise_", "Bg Noise_", "_B.G. noise_", "_BG Noise_", "_BG speech",
    "[/CNON]", "[CNON]", "[/CNPS]", "[CNPS]", "[/CSPN]", "[CSPN]", "_CSPN_",
    "[FILL/]", "[hmm]", "_hmm_", "_mouth noise_", "_Mouth_noise_",
    "__Mouth Noise__", "_Mouth_Noise_", "[noise]", "_/noise_", "_noise_",
    "__noise__", "@Noise@", "_Noise_", "@NOISE@", "[NOISE/]",
    "[non/]", "[non]", "[NON/]", "[/NPS]", "[NPS]", "[S/]", "[SN/]",
    "[Speaker Noise]", "_Speaker Noise_", "_Speaker_Noise_",
    "[SPN/]", "[unin/]", "_noise_noise_", "_bg_noise_", "[_noise_]",
    "_NS_NOISE_", "_/click_",
]}


# ---------------------------------------------------------------------------
# Post-processing: character-level diff merge (from Databricks pipeline)
# ---------------------------------------------------------------------------

# Abbreviation pattern: 2+ single letters each followed by a dot (e.g. O.E.E., A.O.K.)
_POSTPROC_ABBREV_RE = re.compile(r'(?:[A-Za-z]\.){2,}')
# Spaced abbreviation pattern: e.g. "S. M. S." or "H. S. V."
_POSTPROC_SPACED_ABBREV_RE = re.compile(r'(?<![A-Za-z])(?:[A-Za-z]\. ){1,}[A-Za-z]\.')
# Decorative punctuation that can be safely stripped from deletions
_POSTPROC_DECORATIVE_PUNCT_RE = re.compile(r"""["'"\u201c\u201d\u2018\u2019\u00ab\u00bb()\[\]{}&~*]""")
# Punctuation that can be safely deleted when the entire chunk is pure punct.
_POSTPROC_DELETABLE_PUNCT_RE = re.compile(r"""[."'\u201c\u201d\u2018\u2019\u00ab\u00bb()\[\]{}&~*]""")


def _postproc_strip_alnum(s: str) -> str:
    return re.sub(r'[^\w]', '', s, flags=re.UNICODE)


def _postproc_strip_word_chars(s: str) -> str:
    return re.sub(r'\w', '', s, flags=re.UNICODE)


def _postproc_has_alnum(s: str) -> bool:
    return bool(re.search(r'\w', s, flags=re.UNICODE))


def _postproc_contains_noise_token(s: str) -> bool:
    s_lower = s.lower()
    return any(tok in s_lower for tok in _PREPROCESSING_NOISE_TOKENS)


def _postproc_greedy_abbrev_split(letters: str, llm_output: str) -> list[str] | None:
    if not letters:
        return []
    for end in range(len(letters), 1, -1):
        prefix = letters[:end]
        if re.search(
            r'(?<![A-Za-z])' + re.escape(prefix) + r'(?![A-Za-z])',
            llm_output,
            re.IGNORECASE,
        ):
            rest = _postproc_greedy_abbrev_split(letters[end:], llm_output)
            if rest is not None:
                return [prefix] + rest
    return None


def _postproc_normalize_abbreviations(original: str, llm_output: str) -> str:
    """Remove abbreviation dots in *original* when the LLM removed them too."""
    # Collapse spaced abbreviations "S. M. S." -> "S.M.S."
    original = _POSTPROC_SPACED_ABBREV_RE.sub(
        lambda m: m.group().replace(' ', ''), original
    )

    def _replace(match: re.Match) -> str:
        abbrev = match.group()
        letters = abbrev.replace('.', '')
        if re.search(
            r'(?<![A-Za-z])' + re.escape(letters) + r'(?![A-Za-z])',
            llm_output,
            re.IGNORECASE,
        ):
            return letters
        parts = _postproc_greedy_abbrev_split(letters, llm_output)
        if parts is not None:
            return ' '.join(parts)
        return abbrev

    return _POSTPROC_ABBREV_RE.sub(_replace, original)


_POSTPROC_QUOTE_PAIRS = {
    '"': '"', '\u201c': '\u201d', '\u2018': '\u2019',
    '\u00ab': '\u00bb', '\u2039': '\u203a',
    '\u300c': '\u300d', '\u300e': '\u300f',
    '\u201e': '\u201d',
}
_POSTPROC_OPENING_QUOTES = set(_POSTPROC_QUOTE_PAIRS.keys())


def _postproc_unquote_field(s: str) -> str:
    if len(s) < 2:
        return s
    opening = s[0]
    if opening not in _POSTPROC_OPENING_QUOTES:
        return s
    expected_closing = _POSTPROC_QUOTE_PAIRS[opening]
    if s[-1] == expected_closing:
        inner = s[1:-1]
        if opening == '"':
            inner = inner.replace('\\"', '"')
        return inner
    return s


def postprocess_apply_safe_edits(
    original: str,
    llm_output: str,
    verbose: bool = False,
) -> tuple[str, str | None]:
    """Filter LLM output via character-level diff: accept only punctuation
    and casing changes, reject word-content changes.

    For each diff opcode:
      - equal:   Keep as-is.
      - replace: If the alphanumeric letters are the same (ignoring case),
                 it's a pure punct/cap change -> accept the LLM version.
                 If the original is pure punctuation and the LLM introduced
                 words, accept the punct removal but reject the new words.
                 Otherwise the LLM changed actual word content -> keep original.
      - insert:  Accept only if no word characters (pure punct/space addition).
      - delete:  Accept only if no word characters (pure punct removal) or noise token.

    Returns ``(merged_text, report_or_None)``.
    """
    original = _postproc_unquote_field(original)
    original = _postproc_normalize_abbreviations(original, llm_output)

    matcher = difflib.SequenceMatcher(None, original, llm_output, autojunk=False)

    result: list[str] = []
    kept_changes: list[str] = []
    rejected_changes: list[str] = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_chunk = original[i1:i2]
        llm_chunk = llm_output[j1:j2]

        if op == "equal":
            result.append(orig_chunk)

        elif op == "replace":
            orig_alnum = _postproc_strip_alnum(orig_chunk)
            llm_alnum = _postproc_strip_alnum(llm_chunk)

            if orig_alnum.lower() == llm_alnum.lower():
                result.append(llm_chunk)
                if orig_chunk != llm_chunk:
                    kept_changes.append(f"  '{orig_chunk}' -> '{llm_chunk}'")
            elif not orig_alnum:
                spacing = _postproc_strip_word_chars(llm_chunk)
                if spacing:
                    result.append(spacing)
                kept_changes.append(f"  Removed punct '{orig_chunk}'")
                if llm_alnum:
                    rejected_changes.append(f"  Inserted '{llm_alnum}' (insertion rejected)")
            else:
                cleaned = re.sub(r'[^\w\s]', '', orig_chunk, flags=re.UNICODE)
                result.append(cleaned)
                if cleaned != orig_chunk:
                    kept_changes.append(f"  Stripped punct: '{orig_chunk}' -> '{cleaned}'")
                rejected_changes.append(f"  '{orig_chunk}' -> '{llm_chunk}' (content change rejected)")

        elif op == "insert":
            if not _postproc_has_alnum(llm_chunk):
                result.append(llm_chunk)
                kept_changes.append(f"  Inserted '{llm_chunk}'")
            else:
                rejected_changes.append(f"  Inserted '{llm_chunk}' (insertion rejected)")

        elif op == "delete":
            if _postproc_contains_noise_token(orig_chunk):
                kept_changes.append(f"  Removed noise token '{orig_chunk}'")
            elif not _postproc_has_alnum(orig_chunk):
                if _POSTPROC_DELETABLE_PUNCT_RE.search(orig_chunk):
                    kept_changes.append(f"  Removed decorative punct '{orig_chunk}'")
                else:
                    result.append(orig_chunk)
            else:
                cleaned = _POSTPROC_DECORATIVE_PUNCT_RE.sub('', orig_chunk)
                if cleaned != orig_chunk:
                    kept_changes.append(f"  Stripped decorative punct: '{orig_chunk}' -> '{cleaned}'")
                if cleaned:
                    result.append(cleaned)

    merged = "".join(result)
    merged = re.sub(r' {2,}', ' ', merged).strip()

    report = None
    if verbose:
        lines = ["=== Transcript Merge Report ===", ""]
        if kept_changes:
            lines.append(f"Accepted {len(kept_changes)} punctuation/capitalization change(s):")
            lines.extend(kept_changes)
        else:
            lines.append("No punctuation/capitalization changes to apply.")
        lines.append("")
        if rejected_changes:
            lines.append(f"Rejected {len(rejected_changes)} content change(s):")
            lines.extend(rejected_changes)
        else:
            lines.append("No content changes to reject.")
        report = "\n".join(lines)

    return merged, report


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


def count_input_lines(input_file_value: str) -> int:
    """Fast line count without loading the entire file."""
    input_path = resolve_path(input_file_value)
    if not input_path.exists():
        return 0
    count = 0
    with open(input_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def _estimate_segments(text: str, max_input_chars: int) -> int:
    """Estimate how many segments a transcription will produce."""
    if max_input_chars <= 0 or not text:
        return 1
    text_len = len(text)
    if text_len <= max_input_chars:
        return 1
    # Use ceiling division; matches the segmentation loop in run_batch_pipeline.
    return -(-text_len // max_input_chars)


def iter_transcription_chunks(
    input_file_value: str,
    chunk_size: int,
    max_input_chars_per_call: int = 0,
) -> Iterator[tuple[int, list[str], list[str | None], list[list[str] | None]]]:
    """Yield (global_offset, transcriptions, source_filenames, source_rows) chunks.

    Reads the file in a single bulk read, then yields slices of *chunk_size*
    estimated segments.  When *max_input_chars_per_call* > 0, each line's
    estimated segment count (``ceil(len / max_chars)``) is accumulated and a
    chunk is yielded once the total reaches *chunk_size*.  When
    *max_input_chars_per_call* is 0 (no segmentation), *chunk_size* counts
    lines (1 segment per line).

    ``source_rows`` preserves the full original row (all columns) for TSV
    reconstruction so that leading columns are not lost in the output.
    """
    input_path = resolve_path(input_file_value)

    # Detect encoding (single bulk read — fast even on OneDrive).
    try:
        raw_text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = None
        for candidate in ("utf-8-sig", "cp1250", "cp1252", "latin-1"):
            try:
                raw_text = input_path.read_text(encoding=candidate)
                print(f"Input file is not UTF-8; using fallback encoding: {candidate}")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if raw_text is None:
            print(f"Could not decode input file: {input_path}")
            return

    if not raw_text:
        return

    def _select_source_identifier(cells: list[str]) -> str | None:
        if not cells or len(cells) < 2:
            return None
        preferred = cells[-2].strip()
        if preferred:
            return preferred
        for value in reversed(cells[:-1]):
            candidate = value.strip()
            if candidate:
                return candidate
        return None

    # Parse lines and yield chunks as they fill up — no need to store the whole file.
    chunk_transcriptions: list[str] = []
    chunk_filenames: list[str | None] = []
    chunk_source_rows: list[list[str] | None] = []
    chunk_estimated_segments = 0
    global_offset = 0

    if input_path.suffix.lower() == ".tsv":
        lines = raw_text.split('\n')
        if lines and lines[-1] == '':
            lines.pop()
        del raw_text
        for row_index, raw_line in enumerate(lines):
            row = [unicodedata.normalize("NFC", cell) for cell in raw_line.split("\t")]
            if not row or not any(cell.strip() for cell in row):
                continue
            first_cell = row[0].strip() if row else ""
            last_cell = row[-1].strip() if row else ""
            # Skip header.
            if row_index == 0:
                fl = first_cell.lower().replace(" ", "_")
                ll = last_cell.lower().replace(" ", "_")
                if (fl in {"filename", "file", "file_name"} and ll in {"input", "input_segment", "segment", "transcription", "text"}):
                    continue
                if len(row) == 1 and fl in {"input", "input_segment", "segment", "transcription", "text"}:
                    continue
            if len(row) >= 2:
                chunk_filenames.append(_select_source_identifier(row))
                chunk_transcriptions.append(last_cell)
                chunk_source_rows.append(list(row))
                chunk_estimated_segments += _estimate_segments(last_cell, max_input_chars_per_call)
            else:
                chunk_filenames.append(None)
                chunk_transcriptions.append(first_cell)
                chunk_source_rows.append(None)
                chunk_estimated_segments += _estimate_segments(first_cell, max_input_chars_per_call)
            if chunk_estimated_segments >= chunk_size:
                yield (global_offset, chunk_transcriptions, chunk_filenames, chunk_source_rows)
                global_offset += len(chunk_transcriptions)
                chunk_transcriptions = []
                chunk_filenames = []
                chunk_source_rows = []
                chunk_estimated_segments = 0
        del lines
    else:
        lines = raw_text.split('\n')
        if lines and lines[-1] == '':
            lines.pop()
        del raw_text
        for raw_line in lines:
            line = unicodedata.normalize("NFC", raw_line)
            stripped_line = line.strip()
            if "\t" in stripped_line:
                parts = stripped_line.split("\t")
                chunk_filenames.append(_select_source_identifier(parts))
                text = parts[-1].strip()
                chunk_transcriptions.append(text)
                chunk_source_rows.append(parts)
                chunk_estimated_segments += _estimate_segments(text, max_input_chars_per_call)
            else:
                chunk_filenames.append(None)
                chunk_transcriptions.append(stripped_line)
                chunk_source_rows.append(None)
                chunk_estimated_segments += _estimate_segments(stripped_line, max_input_chars_per_call)
            if chunk_estimated_segments >= chunk_size:
                yield (global_offset, chunk_transcriptions, chunk_filenames, chunk_source_rows)
                global_offset += len(chunk_transcriptions)
                chunk_transcriptions = []
                chunk_filenames = []
                chunk_source_rows = []
                chunk_estimated_segments = 0
        del lines

    # Yield remaining items.
    if chunk_transcriptions:
        yield (global_offset, chunk_transcriptions, chunk_filenames, chunk_source_rows)


def load_existing_output_text_lines(
    output_file_value: str | None,
    expected_count: int,
    output_as_tsv: bool,
) -> list[str] | None:
    if not output_file_value:
        print("Resume requested but no --output-file was provided; ignoring resume option.")
        return None

    output_path = resolve_path(output_file_value)
    output_text_path = output_path.with_suffix(".tsv") if output_as_tsv else output_path.with_suffix(".txt")
    if not output_text_path.exists():
        # If the preferred extension doesn't exist, try the other one.
        # This handles cases where multi-column .txt input was auto-detected
        # and written as .tsv.
        alt_path = output_path.with_suffix(".txt") if output_as_tsv else output_path.with_suffix(".tsv")
        if alt_path.exists():
            output_text_path = alt_path
            output_as_tsv = alt_path.suffix.lower() == ".tsv"
        else:
            print(f"Resume requested but output file does not exist: {output_text_path}")
            return None

    try:
        raw_text = output_text_path.read_text(encoding="utf-8")
    except Exception as error:
        print(f"Failed to read resume output file {output_text_path}: {error}")
        return None

    loaded_lines: list[str] = []
    # Always extract last column — even .txt files may contain tab-separated columns.
    from common_aoai import is_partial_segments_marker as _is_partial_marker
    _lines = raw_text.split('\n')
    if _lines and _lines[-1] == '':
        _lines.pop()
    reader = csv.reader(_lines, delimiter="\t")
    for row in reader:
        if not row:
            loaded_lines.append("")
            continue
        cell = row[-1]
        # Preserve partial segment markers verbatim (they contain JSON).
        if _is_partial_marker(cell):
            loaded_lines.append(cell)
        else:
            loaded_lines.append(sanitize_output_string(cell))

    if len(loaded_lines) < expected_count:
        loaded_lines.extend([""] * (expected_count - len(loaded_lines)))
    elif len(loaded_lines) > expected_count:
        loaded_lines = loaded_lines[:expected_count]

    non_empty_count = sum(1 for line in loaded_lines if isinstance(line, str) and line.strip())
    print(
        f"Loaded resume output rows from {output_text_path}: "
        f"{non_empty_count}/{expected_count} non-empty."
    )
    return loaded_lines

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
    active_step_keys: set[str] | None = None,
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
            active_step_keys,
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

    # Determine file extension: follow caller's explicit setting or input extension.
    is_tsv_extension = (
        output_as_tsv
        if isinstance(output_as_tsv, bool)
        else bool(
            source_filenames
            and any(isinstance(filename, str) and filename.strip() for filename in source_filenames)
        )
    )

    # Build output lines preserving all source columns when available.
    # This applies regardless of the file extension — a .txt file can contain
    # tab-separated columns.
    fallback_rows: list[list[str]] | None = None
    if source_rows is not None and len(source_rows) == len(normalized_text_lines):
        fallback_rows = []
        for original_row, corrected_text in zip(source_rows, normalized_text_lines):
            if isinstance(original_row, list) and original_row:
                # Preserve prefix columns exactly; only strip TSV-breaking chars.
                rebuilt_row = [
                    v.replace("\t", " ").replace("\n", " ").replace("\r", " ")
                    if isinstance(v, str) else ""
                    for v in original_row[:-1]
                ]
                rebuilt_row.append(corrected_text)
                fallback_rows.append(rebuilt_row)
            else:
                fallback_rows.append([corrected_text])

    if fallback_rows is not None:
        fallback_lines = ["\t".join(row) for row in fallback_rows]
    elif is_tsv_extension and source_filenames is not None and len(source_filenames) == len(normalized_text_lines):
        fallback_lines = [
            f"{sanitize_output_string(filename if isinstance(filename, str) else '')}\t{text}"
            for filename, text in zip(source_filenames, normalized_text_lines)
        ]
    else:
        fallback_lines = normalized_text_lines

    fallback_content = "\n".join(fallback_lines)

    if output_file_value:
        output_path = resolve_path(output_file_value)
        output_text_path = output_path.with_suffix(".tsv") if is_tsv_extension else output_path.with_suffix(".txt")
        output_text_path.parent.mkdir(parents=True, exist_ok=True)
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
    for line in prompt_text.split('\n'):
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


def resolve_active_chain_step_keys(chain_steps: list[str] | None) -> set[str]:
    """Resolve active chain selectors into payload step keys that own `edits` fields."""
    active_chain_ids = _resolve_active_chain_ids(chain_steps)
    chain_id_to_step_key = {
        "0": "ct_speaker",
        "1": "ct_combine",
        "3": "ct_lexical",
        "4": "ct_disfluency",
        "5": "ct_format",
        "6": "ct_numeral",
        "7": "ct_punct",
        "8": "ct_casing",
        "9": "ct_remain_fix",
    }

    resolved_step_keys: set[str] = set()
    for chain_id in active_chain_ids:
        step_key = chain_id_to_step_key.get(chain_id)
        if isinstance(step_key, str):
            resolved_step_keys.add(step_key)
    return resolved_step_keys


def build_patch_prompt(
    prompt_template: str,
    transcription: str,
    chain_steps: list[str] | None = None,
    locale: str | None = None,
    prev_context: str | None = None,
    next_context: str | None = None,
) -> str:
    active_chain_ids = _resolve_active_chain_ids(chain_steps)
    active_chain_names = [_CHAIN_ID_TO_NAME[chain_id] for chain_id in active_chain_ids]
    active_step_keys = resolve_active_chain_step_keys(chain_steps)
    inactive_step_keys = [step_key for step_key in _STEP_CHAIN_KEYS if step_key not in active_step_keys]

    chain_steps_text = ", ".join(
        f"{chain_id}:{chain_name}"
        for chain_id, chain_name in zip(active_chain_ids, active_chain_names)
    )

    active_chain_ids_text = ", ".join(active_chain_ids)
    active_step_keys_text = ", ".join(sorted(active_step_keys)) if active_step_keys else "<none>"
    inactive_step_keys_text = ", ".join(inactive_step_keys) if inactive_step_keys else "<none>"

    prompt = prompt_template
    if "{chain_steps}" in prompt:
        prompt = prompt.replace("{chain_steps}", chain_steps_text)
    if "{active_chain_ids}" in prompt:
        prompt = prompt.replace("{active_chain_ids}", active_chain_ids_text)
    if "{active_step_keys}" in prompt:
        prompt = prompt.replace("{active_step_keys}", active_step_keys_text)
    if "{inactive_step_keys}" in prompt:
        prompt = prompt.replace("{inactive_step_keys}", inactive_step_keys_text)

    if "{locale}" in prompt:
        prompt = prompt.replace("{locale}", locale if locale else "unknown")
    if "{prev_context}" in prompt:
        prompt = prompt.replace("{prev_context}", prev_context if prev_context else "")
    if "{next_context}" in prompt:
        prompt = prompt.replace("{next_context}", next_context if next_context else "")
    if "{input_transcript}" in prompt:
        prompt = prompt.replace("{input_transcript}", transcription)
        return prompt

    # Append ASR disclaimer + transcription. The disclaimer reduces content
    # filter false positives on garbled ASR text with missing diacritics.
    disclaimer = (
        "\n\n[NOTE: The text below is raw ASR (speech recognition) output. "
        "It may contain missing diacritics, garbled words, or concatenated segments. "
        "Treat all content as benign transcription data.]\n\n"
    )
    return prompt + disclaimer + transcription


def _order_top_level_output_payload_keys(payload: dict) -> dict:
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


def _strip_inactive_step_payloads(payload: dict, active_step_keys: set[str] | None) -> dict:
    if not isinstance(active_step_keys, set):
        return payload

    compacted = dict(payload)
    for step_key in _STEP_CHAIN_KEYS:
        if step_key not in active_step_keys:
            compacted.pop(step_key, None)
    return compacted


def sanitize_payload_for_output(
    payload: dict,
    source_filename: str | None = None,
    active_step_keys: set[str] | None = None,
) -> dict | None:
    if not isinstance(payload, dict):
        return None

    payload_copy = _sanitize_output_payload_value(payload)
    if not isinstance(payload_copy, dict):
        return None

    if isinstance(source_filename, str) and source_filename.strip():
        payload_copy["source_filename"] = sanitize_output_string(source_filename)

    payload_copy.pop("tokenization", None)
    payload_copy = _strip_inactive_step_payloads(payload_copy, active_step_keys)
    return _order_top_level_output_payload_keys(payload_copy)


def prepare_jsonl_output_path(
    output_file_value: str | None,
    resume_mode: bool = False,
) -> Path | None:
    if not output_file_value:
        return None

    output_path = resolve_path(output_file_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl_path = output_path.with_suffix(".jsonl")

    if not resume_mode:
        output_jsonl_path.write_text("", encoding="utf-8")

    return output_jsonl_path


def append_payload_jsonl_record(
    output_jsonl_path: Path | None,
    payload: dict | None,
    source_filename: str | None = None,
    active_step_keys: set[str] | None = None,
) -> bool:
    if output_jsonl_path is None or not isinstance(payload, dict):
        return False

    sanitized_payload = sanitize_payload_for_output(
        payload,
        source_filename,
        active_step_keys,
    )
    if not isinstance(sanitized_payload, dict):
        return False

    line = json.dumps(sanitized_payload, ensure_ascii=False, separators=(",", ":"))
    with output_jsonl_path.open("a", encoding="utf-8", newline="") as output_file:
        output_file.write(line)
        output_file.write("\n")
    return True


def write_output_artifacts(
    payloads: list[dict],
    output_file_value: str | None,
    text_output_lines: list[str] | None = None,
    source_filenames: list[str | None] | None = None,
    source_rows: list[list[str] | None] | None = None,
    output_as_tsv: bool | None = None,
    active_step_keys: set[str] | None = None,
) -> None:
    # The authoritative row count is text_output_lines (one per input item).
    # payloads may be shorter (filtered to dicts only by the caller).
    expected_row_count = len(text_output_lines) if text_output_lines is not None else len(payloads)

    # File extension follows caller's setting (input extension).
    is_tsv_extension = (
        output_as_tsv
        if isinstance(output_as_tsv, bool)
        else bool(
            source_filenames
            and any(isinstance(filename, str) and filename.strip() for filename in source_filenames)
        )
    )

    can_backfill_source_filename = (
        source_filenames is not None
        and len(source_filenames) == len(payloads)
    )

    sanitized_payloads: list[dict] = []
    for index, item in enumerate(payloads):
        if isinstance(item, dict):
            source_filename = source_filenames[index] if can_backfill_source_filename else None
            sanitized_payload = sanitize_payload_for_output(
                item,
                source_filename,
                active_step_keys,
            )
            if isinstance(sanitized_payload, dict):
                sanitized_payloads.append(sanitized_payload)

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

    # Build output lines preserving all source columns when available.
    normalized_source_rows: list[list[str] | None] | None = None
    if source_rows is not None:
        if len(source_rows) != len(normalized_text_lines):
            print("WARNING: source row count does not match output line count; omitting full-row TSV reconstruction.")
        else:
            normalized_source_rows = []
            for row in source_rows:
                if isinstance(row, list):
                    # Preserve prefix columns exactly; only strip TSV-breaking chars.
                    normalized_source_rows.append([
                        v.replace("\t", " ").replace("\n", " ").replace("\r", " ")
                        if isinstance(v, str) else ""
                        for v in row
                    ])
                else:
                    normalized_source_rows.append(None)

    # Build final text lines: use full source rows when available, else filename+text or text-only.
    final_text_lines: list[str] = []
    for row_index, corrected_text in enumerate(normalized_text_lines):
        if (
            normalized_source_rows is not None
            and row_index < len(normalized_source_rows)
            and isinstance(normalized_source_rows[row_index], list)
            and normalized_source_rows[row_index]
        ):
            row = list(normalized_source_rows[row_index])
            row[-1] = corrected_text
            final_text_lines.append("\t".join(row))
        elif is_tsv_extension:
            normalized_source_filenames = [""] * len(normalized_text_lines)
            if source_filenames is not None and len(source_filenames) == len(normalized_text_lines):
                normalized_source_filenames = [
                    sanitize_output_string(value if isinstance(value, str) else "")
                    for value in source_filenames
                ]
            filename = (
                normalized_source_filenames[row_index]
                if row_index < len(normalized_source_filenames)
                else ""
            )
            final_text_lines.append(f"{filename}\t{corrected_text}")
        else:
            final_text_lines.append(corrected_text)
    text_output_content = "\n".join(final_text_lines)

    if output_file_value:
        output_path = resolve_path(output_file_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSONL (one payload per line) as the default structured output.
        output_jsonl_path = output_path.with_suffix(".jsonl")
        with output_jsonl_path.open("w", encoding="utf-8", newline="") as jsonl_file:
            for sanitized_payload in sanitized_payloads:
                line = json.dumps(sanitized_payload, ensure_ascii=False, separators=(",", ":"))
                jsonl_file.write(line)
                jsonl_file.write("\n")

        # Write text output (.txt or .tsv depending on input extension).
        output_text_path = output_path.with_suffix(".tsv") if is_tsv_extension else output_path.with_suffix(".txt")
        output_text_path.parent.mkdir(parents=True, exist_ok=True)
        output_text_path.write_text(text_output_content, encoding="utf-8")
    else:
        for sanitized_payload in sanitized_payloads:
            print(json.dumps(sanitized_payload, ensure_ascii=False, separators=(",", ":")))


def sanitize_output_string(value: str) -> str:
    if not isinstance(value, str):
        return ""

    # Strip all ASCII control characters (0x00-0x1F, 0x7F) and Unicode line
    # separators that would break line-based output formats.
    # Preserve only \t temporarily (replaced with space below).
    normalized = "".join(
        " " if (
            c == "\t"
            or c == "\n"
            or c == "\r"
            or (c < " " and c != "\t")      # ASCII C0 controls (0x00-0x1F)
            or c == "\x7f"                    # DEL
            or c == "\x85"                    # NEL (Next Line)
            or c == "\u2028"                  # LINE SEPARATOR
            or c == "\u2029"                  # PARAGRAPH SEPARATOR
        ) else c
        for c in value
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
    active_step_keys: set[str] | None = None,
    on_final_failure: Callable[[str], None] | None = None,
    repair_timeout_message: str = "Repair attempt timed out.",
    repair_empty_message: str = "Repair retry returned empty output.",
    strip_repair_content: bool = False,
) -> dict | None:
    def _notify_final_failure(reason: str) -> None:
        if on_final_failure is None:
            return
        try:
            on_final_failure(reason)
        except Exception:
            pass

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
            active_step_keys=active_step_keys,
        )

        if payload is None:
            log_json_validation_with_key_error(validation_error, content, processing_id)

            if is_json_recursion_validation_error(validation_error):
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    "JSON parse recursion depth exceeded.",
                    "JSON parse recursion depth exceeded. Rerunning patch without repair.",
                    processing_id,
                )
                if not should_retry:
                    return None
                continue

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

            if is_long_span_preservation_validation_error(validation_error):
                print(
                    f"[{processing_id}] Long-span preservation check failed. "
                    "Retrying with shortened segment."
                )
                _notify_final_failure("long_span")
                return None

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
                active_step_keys=active_step_keys,
            )
            if payload is None:
                if is_long_span_preservation_validation_error(validation_error):
                    print(
                        f"[{processing_id}] Long-span preservation check failed after repair. "
                        "Retrying with shortened segment."
                    )
                    _notify_final_failure("long_span")
                    return None

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
