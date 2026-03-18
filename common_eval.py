import csv
import difflib
import json
import random
import re
import unicodedata
from pathlib import Path
from collections.abc import Callable
from typing import Any

from common_aoai import aoai_send_with_timeout_retry
from common_copilot import (
    ModelMismatchError,
    build_copilot_session_parameters,
    handle_copilot_model_mismatch_retry,
    is_copilot_retryable_error,
    send_copilot_once,
)
from common import (
    _CHAR_BASED_NO_SPACE_GROUPS,
    _OPENING_PUNCT_TOKENS,
    _char_based_script_group,
    _is_punct_token,
    _tokenize_for_hallucination,
    build_repair_prompt_after_invalid_json,
    extract_retry_after_seconds,
    strip_markdown_code_fence,
    run_with_timeout_retry,
    should_retry_after_failure,
)


_STEP_NAME_TO_PAYLOAD_KEY = {
    "SPEAKER": "ct_speaker",
    "COMBINE": "ct_combine",
    "LEXICAL": "ct_lexical",
    "DISFLUENCY": "ct_disfluency",
    "FORMAT": "ct_format",
    "NUMERAL": "ct_numeral",
    "PUNCT": "ct_punct",
    "CASING": "ct_casing",
    "REMAIN_FIX": "ct_remain_fix",
}

def _compute_edit_error_rate(
    total_edits: int,
    invalid_count: int,
    missing_count: int,
) -> tuple[float, int]:
    total_opportunities = total_edits + missing_count
    if total_opportunities <= 0:
        return 0.0, total_opportunities

    rate = (invalid_count + missing_count) / total_opportunities
    rate = max(0.0, min(1.0, rate))
    return rate, total_opportunities

def _infer_trail_outcome(item: dict) -> str:
    # New contract uses chain step in `valid_step`; outcome is carried by rule/note.
    rule = str(item.get("rule", "")).strip().upper()
    note = str(item.get("note", "")).strip().upper()
    if "STEP_MISMATCH" in rule or "STEP_MISMATCH" in note:
        return "STEP_MISMATCH"

    valid_step_raw = item.get("valid_step")
    if valid_step_raw is None:
        return "INVALID"

    if "INVALID" in rule or "INVALID" in note:
        return "INVALID"
    return "VALID"


def _is_valid_edit_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("edit"), str):
        return False
    if not isinstance(item.get("rule"), str):
        return False
    if not isinstance(item.get("note"), str):
        return False
    valid_step = item.get("valid_step")
    return valid_step is None or isinstance(valid_step, str)


def _is_valid_missing_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return all(isinstance(item.get(field), str) for field in ("span", "expected_edit", "step", "reason"))


def _normalize_compare_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", normalized.strip())


def _split_edit_text(edit_text: str) -> tuple[str, str] | None:
    if not isinstance(edit_text, str):
        return None

    text = edit_text.strip()
    if not text.startswith("["):
        return None

    separator = "] -> ["
    depth = 0
    split_index = -1
    for index, char in enumerate(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0 and text.startswith(separator, index):
                split_index = index
                break

    if split_index < 0:
        return None

    before = text[1:split_index]
    after_start = split_index + len(separator)

    depth = 1
    after_end = -1
    for index in range(after_start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                after_end = index
                break

    if after_end < 0:
        return None
    if text[after_end + 1:].strip():
        return None

    after = text[after_start:after_end]
    return before, after


def _extract_marked_change_segments(marked_text: str) -> list[str]:
    if not isinstance(marked_text, str) or not marked_text:
        return []
    return re.findall(r"<chg>(.*?)</chg>", marked_text, flags=re.DOTALL)


def _contains_no_space_script_char(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for char in text:
        if _char_based_script_group(char) in _CHAR_BASED_NO_SPACE_GROUPS:
            return True
    return False


def _join_change_segments(segments: list[str]) -> str:
    cleaned = [segment.strip() for segment in segments if isinstance(segment, str) and segment.strip()]
    if not cleaned:
        return ""

    # Avoid inserting spaces between fragments in no-space scripts.
    if any(_contains_no_space_script_char(segment) for segment in cleaned):
        return "".join(cleaned)

    return " ".join(cleaned)


def _find_contradictory_invalid_target_index(
    missing_item: dict[str, str],
    invalid_edit_targets: list[dict[str, str]],
) -> int | None:
    missing_expected = _normalize_compare_text(str(missing_item.get("expected_edit", "")))
    missing_span = _normalize_compare_text(str(missing_item.get("span", "")))
    if not missing_expected:
        return None

    for index, target in enumerate(invalid_edit_targets):
        after_change = _normalize_compare_text(target.get("after_change", ""))
        after_full = _normalize_compare_text(target.get("after_full", ""))
        if missing_expected not in {after_change, after_full}:
            continue

        before_change = _normalize_compare_text(target.get("before_change", ""))
        before_full = _normalize_compare_text(target.get("before_full", ""))
        if missing_span and missing_span not in {before_change, before_full}:
            continue
        return index

    return None


def _is_noop_missing_edit(missing_item: dict[str, str]) -> bool:
    span = _normalize_compare_text(str(missing_item.get("span", "")))
    expected_edit = _normalize_compare_text(str(missing_item.get("expected_edit", "")))
    if not span or not expected_edit:
        return False
    return span == expected_edit


def normalize_eval_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return build_failed_eval_payload("Prompt-eval failure: evaluator output is not an object")

    edits = payload.get("edits")
    missing_edits = payload.get("missing_edits")
    if not isinstance(edits, list) or not isinstance(missing_edits, list):
        return build_failed_eval_payload("Prompt-eval failure: missing required fields 'edits' or 'missing_edits'")
    if not all(_is_valid_edit_item(item) for item in edits):
        return build_failed_eval_payload("Prompt-eval failure: malformed item in 'edits'")
    if not all(_is_valid_missing_item(item) for item in missing_edits):
        return build_failed_eval_payload("Prompt-eval failure: malformed item in 'missing_edits'")

    invalid_edits = 0
    step_mismatch_edits = 0
    fail_reasons: list[str] = []
    edits_with_outcome: list[dict[str, Any]] = []
    invalid_issue_records: list[dict[str, Any]] = []
    invalid_edit_targets: list[dict[str, str]] = []
    for item in edits:
        outcome = _infer_trail_outcome(item)
        item_with_outcome = dict(item)
        item_with_outcome["outcome"] = outcome
        edits_with_outcome.append(item_with_outcome)

        if outcome == "INVALID":
            invalid_edits += 1
        elif outcome == "STEP_MISMATCH":
            step_mismatch_edits += 1
        if outcome in {"INVALID", "STEP_MISMATCH"}:
            edit = str(item.get("edit", "")).strip()
            valid_step = str(item.get("valid_step", "")).strip()
            rule = str(item.get("rule", "")).strip()
            issue_record: dict[str, Any] = {
                "outcome": outcome,
                "valid_step": valid_step,
                "edit": edit,
                "rule": rule,
                "target_index": None,
            }

            split_edit = _split_edit_text(edit)
            if split_edit is not None:
                before_full, after_full = split_edit
                before_segments = _extract_marked_change_segments(before_full)
                after_segments = _extract_marked_change_segments(after_full)
                target_index = len(invalid_edit_targets)
                invalid_edit_targets.append(
                    {
                        "before_full": before_full,
                        "after_full": after_full,
                        "before_change": _join_change_segments(before_segments),
                        "after_change": _join_change_segments(after_segments),
                    }
                )
                issue_record["target_index"] = target_index

            invalid_issue_records.append(issue_record)

    filtered_missing_edits: list[dict[str, str]] = []
    contradiction_count = 0
    noop_missing_count = 0
    contradicted_invalid_target_indices: set[int] = set()
    for missing in missing_edits:
        if not isinstance(missing, dict):
            continue
        if _is_noop_missing_edit(missing):
            noop_missing_count += 1
            continue
        matched_invalid_index = _find_contradictory_invalid_target_index(missing, invalid_edit_targets)
        if matched_invalid_index is not None:
            contradiction_count += 1
            contradicted_invalid_target_indices.add(matched_invalid_index)
            continue
        filtered_missing_edits.append(missing)

    effective_invalid_edits = 0
    effective_step_mismatch_edits = 0
    for issue_record in invalid_issue_records:
        target_index = issue_record.get("target_index")
        if isinstance(target_index, int) and target_index in contradicted_invalid_target_indices:
            # An overlapped missing+invalid judgment is ambiguous and can indicate a correct edit;
            # do not count it as an invalid failure.
            continue

        outcome = str(issue_record.get("outcome", ""))
        if outcome == "INVALID":
            effective_invalid_edits += 1
        elif outcome == "STEP_MISMATCH":
            effective_step_mismatch_edits += 1

        valid_step = str(issue_record.get("valid_step", "")).strip()
        edit = str(issue_record.get("edit", "")).strip()
        rule = str(issue_record.get("rule", "")).strip()
        fail_reasons.append(f"{outcome}: valid_step={valid_step}; edit={edit}; rule={rule}")

    invalid_count = effective_invalid_edits + effective_step_mismatch_edits
    missing_count = len(filtered_missing_edits)
    edit_error_rate, total_opportunities = _compute_edit_error_rate(
        total_edits=len(edits),
        invalid_count=invalid_count,
        missing_count=missing_count,
    )

    diff_summary: list[str] = []
    diff_summary.append(
        (
            f"edit_error_rate={edit_error_rate * 100:.2f}%"
            f"; invalid={invalid_count}; missing={missing_count}; opportunities={total_opportunities}"
        )
    )
    diff_summary.append(
        f"edits_count={len(edits)}; invalid={invalid_edits}; step_mismatch={step_mismatch_edits}"
    )
    if contradiction_count > 0:
        diff_summary.append(
            "contradiction_resolved_overlap="
            f"{contradiction_count}; effective_invalid={effective_invalid_edits}; "
            f"effective_step_mismatch={effective_step_mismatch_edits}"
        )
    if noop_missing_count > 0:
        diff_summary.append(f"noop_filtered_missing={noop_missing_count}")

    for missing in filtered_missing_edits:
        if not isinstance(missing, dict):
            continue
        span = str(missing.get("span", "")).strip()
        step = str(missing.get("step", "")).strip()
        reason = str(missing.get("reason", "")).strip()
        fail_reasons.append(f"MISSING: step={step}; span={span}; reason={reason}")

    pass_value = invalid_count == 0 and missing_count == 0

    if not pass_value and not fail_reasons:
        fail_reasons = ["Derived FAIL from audit counts"]

    return {
        "pass": pass_value,
        "fail_reasons": fail_reasons,
        "edit_error_rate": edit_error_rate,
        "diff_summary": diff_summary,
        "edits": edits_with_outcome,
        "missing_edits": filtered_missing_edits,
    }


def load_patch_result_json_payloads(patch_result_path: Path) -> list[dict]:
    json_path: Path | None = None
    if patch_result_path.suffix.lower() == ".json":
        json_path = patch_result_path
    elif patch_result_path.suffix.lower() == ".txt":
        candidate = patch_result_path.with_suffix(".json")
        if candidate.exists():
            json_path = candidate

    if json_path is None or not json_path.exists():
        return []

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def build_python_extracted_edits_by_step(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return "{}"

    extracted: dict[str, object] = {}
    for step_name, key in _STEP_NAME_TO_PAYLOAD_KEY.items():
        step_payload = payload.get(key)
        edits = step_payload.get("edits") if isinstance(step_payload, dict) else []
        extracted[step_name] = edits if isinstance(edits, list) else []

    no_touch_tokens = payload.get("no_touch_tokens")
    extracted["NO_TOUCH"] = no_touch_tokens if isinstance(no_touch_tokens, list) else []
    return json.dumps(extracted, ensure_ascii=False)


def build_aligned_edits(original_text: str, patched_text: str) -> str:
    source = original_text or ""
    target = patched_text or ""
    change_open_marker = "<chg>"
    change_close_marker = "</chg>"

    extracted: list[str] = []

    def _last_char(text: str) -> str:
        return text[-1] if isinstance(text, str) and text else ""

    def _first_char(text: str) -> str:
        return text[0] if isinstance(text, str) and text else ""

    def _should_compact_adjacent(left: str, right: str) -> bool:
        left_group = _char_based_script_group(_last_char(left))
        right_group = _char_based_script_group(_first_char(right))
        return (
            left_group is not None
            and left_group == right_group
            and left_group in _CHAR_BASED_NO_SPACE_GROUPS
        )

    def _expand_alignment_units(units: list[str]) -> list[str]:
        expanded: list[str] = []
        for unit in units:
            if not isinstance(unit, str) or not unit:
                continue

            first_group = _char_based_script_group(unit[0])
            if first_group not in _CHAR_BASED_NO_SPACE_GROUPS:
                expanded.append(unit)
                continue

            if all(_char_based_script_group(char) == first_group for char in unit):
                expanded.extend(list(unit))
            else:
                expanded.append(unit)

        return expanded

    def _render_edit_units(units: list[str]) -> str:
        if not units:
            return ""

        parts: list[str] = [units[0]]
        for token in units[1:]:
            previous = parts[-1]
            if (
                _is_punct_token(token)
                or token in {change_open_marker, change_close_marker}
                or previous in {change_open_marker, change_close_marker}
                or previous in _OPENING_PUNCT_TOKENS
                or _should_compact_adjacent(previous, token)
            ):
                parts.append(token)
                continue

            parts.append(" ")
            parts.append(token)

        return "".join(parts)

    def _render_full_change(changed_units: list[str]) -> str:
        if not changed_units:
            return ""
        changed_text = _render_edit_units(changed_units)
        return f"{change_open_marker}{changed_text}{change_close_marker}"

    def _looks_like_empty_after_edit(edit_text: str) -> bool:
        if not isinstance(edit_text, str):
            return False
        return bool(re.search(r"\]\s*->\s*\[\s*\]\s*$", edit_text))

    def _needs_space_between(left_token: str, right_token: str) -> bool:
        if not left_token or not right_token:
            return False
        if _is_punct_token(right_token):
            return False
        if left_token in _OPENING_PUNCT_TOKENS:
            return False
        if _should_compact_adjacent(left_token, right_token):
            return False
        return True

    def _render_marked_units(
        left_context: list[str],
        changed_units: list[str],
        right_context: list[str],
    ) -> str:
        parts: list[str] = []
        left_text = _render_edit_units(left_context)
        right_text = _render_edit_units(right_context)

        if changed_units:
            changed_text = _render_edit_units(changed_units)
            marked_changed = f"{change_open_marker}{changed_text}{change_close_marker}"

            if left_text:
                parts.append(left_text)
                if _needs_space_between(left_context[-1], changed_units[0]):
                    parts.append(" ")

            parts.append(marked_changed)

            if right_text:
                if _needs_space_between(changed_units[-1], right_context[0]):
                    parts.append(" ")
                parts.append(right_text)

            return "".join(parts)

        if left_text:
            parts.append(left_text)
        if right_text:
            if left_text and _needs_space_between(left_context[-1], right_context[0]):
                parts.append(" ")
            parts.append(right_text)
        return "".join(parts)

    def _shared_left_context(i1: int, j1: int) -> list[str]:
        context: list[str] = []
        left_index = i1 - 1
        right_index = j1 - 1
        while (
            left_index >= 0
            and right_index >= 0
            and source_units[left_index] == target_units[right_index]
        ):
            context.insert(0, source_units[left_index])
            left_index -= 1
            right_index -= 1
        return context

    def _shared_right_context(i2: int, j2: int) -> list[str]:
        context: list[str] = []
        left_index = i2
        right_index = j2
        while (
            left_index < len(source_units)
            and right_index < len(target_units)
            and source_units[left_index] == target_units[right_index]
        ):
            context.append(source_units[left_index])
            left_index += 1
            right_index += 1
        return context

    def _count_occurrences(haystack: list[str], needle: list[str]) -> int:
        if not needle:
            return len(haystack) + 1
        if len(needle) > len(haystack):
            return 0

        count = 0
        end = len(haystack) - len(needle) + 1
        for index in range(end):
            if haystack[index:index + len(needle)] == needle:
                count += 1
        return count

    def _is_unambiguous_span(
        source_span: list[str],
        target_span: list[str],
    ) -> bool:
        source_unique = _count_occurrences(source_units, source_span) == 1
        target_unique = _count_occurrences(target_units, target_span) == 1
        return source_unique and target_unique

    def _minimal_context_window(
        source_change: list[str],
        target_change: list[str],
        left_shared: list[str],
        right_shared: list[str],
    ) -> tuple[list[str], list[str]]:
        max_total = len(left_shared) + len(right_shared)
        for total in range(max_total + 1):
            left_min = max(0, total - len(right_shared))
            left_max = min(total, len(left_shared))
            for left_count in range(left_min, left_max + 1):
                right_count = total - left_count

                left_context = left_shared[len(left_shared) - left_count:] if left_count > 0 else []
                right_context = right_shared[:right_count] if right_count > 0 else []

                source_span = left_context + source_change + right_context
                target_span = left_context + target_change + right_context
                if _is_unambiguous_span(source_span, target_span):
                    return left_context, right_context

        return left_shared, right_shared

    def _is_punctuation_only_change(
        source_change: list[str],
        target_change: list[str],
    ) -> bool:
        changed_units = source_change + target_change
        if not changed_units:
            return False
        return all(_is_punct_token(unit) for unit in changed_units)

    def _is_short_deletion_change(
        source_change: list[str],
        target_change: list[str],
        max_units: int,
    ) -> bool:
        if not source_change or target_change:
            return False
        # Keep this focused on short phrase drops (common disfluency edits).
        return len(source_change) <= max_units

    def _is_speaker_label_unit(unit: str) -> bool:
        if not isinstance(unit, str):
            return False
        text = unit.strip()
        if not text:
            return False
        if re.match(r"^\[[^\]]+\]:?$", text):
            return True
        upper = text.upper()
        return upper.startswith("[SPK") or upper.startswith("SPK")

    def _is_speaker_label_change(source_change: list[str], target_change: list[str]) -> bool:
        return any(_is_speaker_label_unit(unit) for unit in (source_change + target_change))

    def _contains_digit(units: list[str]) -> bool:
        return any(any(char.isdigit() for char in token) for token in units if isinstance(token, str))

    def _is_numeral_context_change(
        source_change: list[str],
        target_change: list[str],
        left_shared: list[str],
        right_shared: list[str],
    ) -> bool:
        local_units = source_change + target_change + left_shared[-2:] + right_shared[:2]
        return _contains_digit(local_units)

    def _is_non_space_short_lexical_change(
        source_change: list[str],
        target_change: list[str],
        max_units: int,
    ) -> bool:
        if not source_change or not target_change:
            return False
        if len(source_change) + len(target_change) > max_units:
            return False
        changed_units = source_change + target_change
        has_no_space_script = any(
            isinstance(unit, str)
            and any(
                _char_based_script_group(char) in _CHAR_BASED_NO_SPACE_GROUPS
                for char in unit
            )
            for unit in changed_units
        )
        return has_no_space_script and not _is_punctuation_only_change(source_change, target_change)

    def _is_combine_like_change(
        source_change: list[str],
        target_change: list[str],
        max_units: int,
    ) -> bool:
        if not source_change or not target_change:
            return False
        if len(source_change) + len(target_change) > max_units:
            return False
        if _is_punctuation_only_change(source_change, target_change):
            return False
        has_no_space_script = any(
            isinstance(unit, str)
            and any(
                _char_based_script_group(char) in _CHAR_BASED_NO_SPACE_GROUPS
                for char in unit
            )
            for unit in (source_change + target_change)
        )
        if has_no_space_script:
            return False

        combined_text = "".join(
            unit for unit in (source_change + target_change) if isinstance(unit, str)
        )
        has_non_ascii_alpha = any(
            char.isalpha() and not (("A" <= char <= "Z") or ("a" <= char <= "z"))
            for char in combined_text
        )
        if has_non_ascii_alpha:
            return False

        # Small bilateral rewrite is often where fragment-combine attribution is ambiguous,
        # but avoid broadening simple 1-to-1 token substitutions.
        return len(source_change) >= 2 or len(target_change) >= 2

    def _resolve_context_expansion_min_units(
        source_change: list[str],
        target_change: list[str],
        left_shared: list[str],
        right_shared: list[str],
    ) -> int:
        min_units = 0

        if _is_punctuation_only_change(source_change, target_change):
            min_units = max(min_units, 6)
        if _is_short_deletion_change(source_change, target_change, max_units=4):
            min_units = max(min_units, 10)
        if _is_speaker_label_change(source_change, target_change):
            min_units = max(min_units, 12)
        if _is_numeral_context_change(source_change, target_change, left_shared, right_shared):
            min_units = max(min_units, 10)
        if _is_combine_like_change(source_change, target_change, max_units=4):
            min_units = max(min_units, 10)

        return min_units

    def _expand_context_for_clarity(
        left_shared: list[str],
        right_shared: list[str],
        left_context: list[str],
        right_context: list[str],
        min_units: int,
    ) -> tuple[list[str], list[str]]:
        expanded_left = left_context
        expanded_right = right_context

        if len(expanded_left) < min_units:
            expanded_left = left_shared[max(0, len(left_shared) - min_units):]
        if len(expanded_right) < min_units:
            expanded_right = right_shared[:min_units]

        return expanded_left, expanded_right

    # Reuse hallucination tokenization so eval alignment matches detection granularity.
    source_units = _expand_alignment_units(_tokenize_for_hallucination(source))
    target_units = _expand_alignment_units(_tokenize_for_hallucination(target))

    matcher = difflib.SequenceMatcher(a=source_units, b=target_units, autojunk=False)
    opcodes = matcher.get_opcodes()

    def _should_force_full_replace_alignment() -> bool:
        if not source_units or not target_units:
            return False

        if len(target_units) > 2:
            return False

        retained_ratio = len(target_units) / max(1, len(source_units))
        if retained_ratio > 0.15:
            return False

        non_equal_ops = [opcode for opcode in opcodes if opcode[0] != "equal"]
        if len(non_equal_ops) <= 1:
            return False

        # If most source content is removed and only a tiny target remains, avoid
        # split pseudo-edits caused by single-token anchoring (e.g., retaining "2").
        return len(source_units) >= 8

    if _should_force_full_replace_alignment():
        before = _render_full_change(source_units)
        after = _render_full_change(target_units)
        return json.dumps([f"[{before}] -> [{after}]"], ensure_ascii=False)

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue

        left_shared = _shared_left_context(i1, j1)
        right_shared = _shared_right_context(i2, j2)
        source_change = source_units[i1:i2]
        target_change = target_units[j1:j2]

        left_context, right_context = _minimal_context_window(
            source_change,
            target_change,
            left_shared,
            right_shared,
        )

        clarity_min_units = _resolve_context_expansion_min_units(
            source_change,
            target_change,
            left_shared,
            right_shared,
        )
        if clarity_min_units > 0:
            left_context, right_context = _expand_context_for_clarity(
                left_shared,
                right_shared,
                left_context,
                right_context,
                min_units=clarity_min_units,
            )

        before = _render_marked_units(left_context, source_change, right_context)
        after = _render_marked_units(left_context, target_change, right_context)
        edit_text = f"[{before}] -> [{after}]"
        extracted.append(edit_text)

    # Defensive fallback: when FINAL_RESULT is non-empty, never emit a sole
    # delete-to-empty aligned edit. This avoids stale/edge tokenization artifacts
    # surfacing as `-> []` even though patched text exists.
    if target.strip() and len(extracted) == 1 and _looks_like_empty_after_edit(extracted[0]):
        before = _render_full_change(source_units) if source_units else f"{change_open_marker}{source}{change_close_marker}"
        after = (
            _render_full_change(target_units)
            if target_units
            else f"{change_open_marker}{target.strip()}{change_close_marker}"
        )
        return json.dumps([f"[{before}] -> [{after}]"], ensure_ascii=False)

    return json.dumps(extracted, ensure_ascii=False)


def build_eval_prompt(
    eval_template: str,
    original_text: str,
    patched_text: str,
    chain_steps_text: str | None = None,
    extracted_edits_text: str | None = None,
) -> str:
    resolved_chain_steps = chain_steps_text if isinstance(chain_steps_text, str) and chain_steps_text.strip() else "ALL"
    resolved_extracted_edits = (
        extracted_edits_text
        if isinstance(extracted_edits_text, str) and extracted_edits_text.strip()
        else "[]"
    )

    original_placeholder = "{original_transcript}"
    patched_placeholder = "{final_result}"
    extracted_placeholder = "{extracted_edits}"

    has_original_placeholder = original_placeholder in eval_template
    has_patched_placeholder = patched_placeholder in eval_template
    has_extracted_placeholder = extracted_placeholder in eval_template

    prompt = (
        eval_template
        .replace("{original_transcript}", original_text)
        .replace("{final_result}", patched_text)
        .replace("{chain_steps}", resolved_chain_steps)
        .replace("{extracted_edits}", resolved_extracted_edits)
    )

    # Fallback for templates that do not expose placeholders for runtime inputs.
    fallback_sections: list[str] = []
    if not has_original_placeholder:
        fallback_sections.append(f"ORIGINAL:\n{original_text}")
    if not has_patched_placeholder:
        fallback_sections.append(f"FINAL_RESULT:\n{patched_text}")
    if not has_extracted_placeholder:
        fallback_sections.append(f"EXTRACTED_EDITS:\n{resolved_extracted_edits}")

    if fallback_sections:
        prompt = prompt.rstrip() + "\n\n" + "\n\n".join(fallback_sections)

    return prompt


async def send_copilot_prompt_once(
    client: Any,
    model_name: str,
    prompt: str,
    timeout_seconds: float,
) -> str:
    session = await client.create_session(build_copilot_session_parameters(model_name))
    return await send_copilot_once(
        session,
        prompt,
        timeout_seconds,
        requested_model=model_name,
    )


async def get_copilot_eval_payload_with_retries(
    client: Any,
    model_name: str,
    prompt: str,
    processing_id: str,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    model_mismatch_retries: int,
) -> dict:
    for mismatch_attempt in range(model_mismatch_retries + 1):
        for empty_attempt in range(empty_result_retries + 1):
            async def operation() -> dict:
                content = await send_copilot_prompt_once(client, model_name, prompt, timeout_seconds)
                if not content:
                    raise ValueError("evaluation model returned empty content")

                payload = json.loads(strip_markdown_code_fence(content))
                payload = normalize_eval_payload(payload)
                is_valid, validation_error = validate_eval_payload(payload)
                if not is_valid:
                    raise ValueError(f"invalid evaluator payload: {validation_error}")
                return payload

            try:
                result = await run_with_timeout_retry(
                    operation,
                    timeout_retries,
                    processing_id=processing_id,
                    is_retryable_error=is_copilot_retryable_error,
                    resolve_backoff_seconds=lambda error, _attempt: extract_retry_after_seconds(error),
                )

                if result is None:
                    should_retry = should_retry_after_failure(
                        empty_attempt,
                        empty_result_retries,
                        "Model call timed out after retries.",
                        processing_id=processing_id,
                    )
                    if should_retry:
                        continue
                    return build_failed_eval_payload("Prompt-eval failure: timeout after retries")

                return result

            except ModelMismatchError as mismatch_error:
                should_retry = await handle_copilot_model_mismatch_retry(
                    mismatch_error,
                    mismatch_attempt,
                    model_mismatch_retries,
                    processing_id,
                    "emitting failed eval payload",
                )
                if should_retry:
                    break

                return build_failed_eval_payload(
                    f"Prompt-eval failure: model mismatch requested={mismatch_error.requested_model} actual={mismatch_error.actual_model}"
                )

            except Exception as error:
                message = str(error).strip()
                lowered = message.lower()
                is_empty = "empty content" in lowered

                if is_empty:
                    should_retry = should_retry_after_failure(
                        empty_attempt,
                        empty_result_retries,
                        "Model returned empty content after retries.",
                        processing_id=processing_id,
                    )
                    if should_retry:
                        continue

                return build_failed_eval_payload(f"Prompt-eval failure: {error}")

    return build_failed_eval_payload("Prompt-eval failure: exhausted retries")


async def get_aoai_eval_payload_with_repair(
    client: object,
    deployment: str,
    prompt: str,
    processing_id: str,
    repair_prompt_template: str,
    schema: dict,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    temperature: float,
    top_p: float,
    retry_temperature_jitter: float,
    retry_top_p_jitter: float,
    validate_payload: Callable[[dict], tuple[bool, str]],
    build_failed_payload: Callable[[str], dict],
) -> dict:
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

        try:
            content = await aoai_send_with_timeout_retry(
                client,
                deployment,
                prompt,
                schema,
                attempt_temperature,
                attempt_top_p,
                timeout_seconds,
                timeout_retries,
                processing_id,
            )

            if content is None:
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    "Model call timed out after retries.",
                    processing_id=processing_id,
                )
                if should_retry:
                    continue
                return build_failed_payload("Prompt-eval failure: timeout after retries")

            if not content:
                should_retry = should_retry_after_failure(
                    empty_attempt,
                    empty_result_retries,
                    "Model returned empty content after retries.",
                    processing_id=processing_id,
                )
                if should_retry:
                    continue
                return build_failed_payload("Prompt-eval failure: empty content after retries")

            try:
                payload = json.loads(strip_markdown_code_fence(content))
                payload = normalize_eval_payload(payload)
                is_valid, validation_error = validate_payload(payload)
                if not is_valid:
                    raise ValueError(f"invalid evaluator payload: {validation_error}")
                return payload
            except Exception as parse_error:
                repair_prompt = build_repair_prompt_after_invalid_json(
                    repair_prompt_template,
                    str(parse_error),
                    content,
                    target_schema=json.dumps(schema, ensure_ascii=False, indent=2),
                    processing_id=processing_id,
                )
                repaired_content = await aoai_send_with_timeout_retry(
                    client,
                    deployment,
                    repair_prompt,
                    schema,
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
                    if should_retry:
                        continue
                    return build_failed_payload("Prompt-eval failure: repair returned empty output or timed out")

                try:
                    payload = json.loads(strip_markdown_code_fence(repaired_content))
                    payload = normalize_eval_payload(payload)
                    is_valid, validation_error = validate_payload(payload)
                    if not is_valid:
                        raise ValueError(f"invalid evaluator payload: {validation_error}")
                    return payload
                except Exception as repaired_parse_error:
                    should_retry = should_retry_after_failure(
                        empty_attempt,
                        empty_result_retries,
                        f"Repair attempt did not return valid JSON ({repaired_parse_error}).",
                        processing_id=processing_id,
                    )
                    if should_retry:
                        continue
                    return build_failed_payload(f"Prompt-eval failure: {repaired_parse_error}")
        except Exception as error:
            return build_failed_payload(f"Prompt-eval failure: {error}")

    return build_failed_payload("Prompt-eval failure: exhausted retries")


def validate_eval_payload(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "evaluation output is not an object"

    if not isinstance(payload.get("pass"), bool):
        return False, "field 'pass' must be boolean"
    if not isinstance(payload.get("fail_reasons"), list):
        return False, "field 'fail_reasons' must be an array"
    if not isinstance(payload.get("diff_summary"), list):
        return False, "field 'diff_summary' must be an array"
    if not isinstance(payload.get("edit_error_rate"), (int, float)):
        return False, "field 'edit_error_rate' must be numeric"

    return True, ""


def build_failed_eval_payload(reason: str) -> dict:
    return {
        "pass": False,
        "fail_reasons": [reason],
        "edit_error_rate": 1.0,
        "diff_summary": [],
        "edits": [],
        "missing_edits": [],
    }


def load_run_errors(path_value: str | None) -> dict[str, str]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def write_eval_outputs(
    prefix: str,
    evaluator_api: str,
    evaluator_model: str,
    report: list[dict],
) -> tuple[Path, Path, Path]:
    output_dir = Path(f"{prefix}_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = f"{evaluator_api}-{evaluator_model}"
    normalized_report: list[dict] = []
    for item in report:
        if not isinstance(item, dict):
            continue
        normalized_item = dict(item)
        normalized_item["evaluator"] = evaluator
        normalized_item.pop("evaluator_model", None)
        normalized_report.append(normalized_item)

    results_path = output_dir / f"{prefix}_{evaluator}_results.json"
    results_path.write_text(json.dumps(normalized_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows: list[dict[str, object]] = []
    total_runs = 0
    total_pass_rate = 0.0
    total_failed = 0.0
    total_line_count = 0.0
    for item in normalized_report:
        line_results = item.get("line_results", [])
        total_lines = len(line_results)
        passed_lines = sum(1 for line_result in line_results if line_result.get("pass") is True)
        failed_lines = total_lines - passed_lines
        pass_rate = round((passed_lines / total_lines * 100.0), 2) if total_lines else 0.0

        rows.append(
            {
                "evaluator": evaluator,
                "result_file": item.get("result_file", item.get("file", "")),
                "line_count": total_lines,
                "passed_lines": passed_lines,
                "failed_lines": failed_lines,
                "pass_rate_percent": pass_rate,
            }
        )

        total_runs += 1
        total_pass_rate += float(pass_rate)
        total_failed += float(failed_lines)
        total_line_count += float(total_lines)

    scores_path = output_dir / f"{prefix}_{evaluator}_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["evaluator", "result_file", "line_count", "passed_lines", "failed_lines", "pass_rate_percent"],
        )
        writer.writeheader()
        writer.writerows(rows)

    overall_pass = round((1.0 - (total_failed / total_line_count)) * 100.0, 2) if total_line_count > 0 else 0.0
    avg_file_pass_rate = round(total_pass_rate / total_runs, 2) if total_runs > 0 else 0.0
    summary_rows: list[dict[str, object]] = [
        {
            "evaluator": evaluator,
            "runs": int(total_runs),
            "avg_file_pass_rate_percent": avg_file_pass_rate,
            "overall_line_pass_rate_percent": overall_pass,
        }
    ]

    summary_path = output_dir / f"{prefix}_{evaluator}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["evaluator", "runs", "avg_file_pass_rate_percent", "overall_line_pass_rate_percent"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return results_path, scores_path, summary_path
