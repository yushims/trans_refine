"""Fix missing spaces at segment boundaries in processed output.

Three-phase workflow:
  extract  Scan for punct+word patterns (no space), deduplicate, write patterns JSON.
  judge    Send unique patterns to LLM for YES/NO space verdicts.
  apply    Apply verdicts to fix the output file.

Usage:
  python fix_segment_spaces.py extract --input outputs/file.tsv --output patterns.json
  python fix_segment_spaces.py judge   --patterns patterns.json --output verdicts.json
  python fix_segment_spaces.py apply   --input outputs/file.tsv --verdicts verdicts.json --output fixed.tsv
"""

import argparse
import asyncio
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from common import (
    _CHAR_BASED_NO_SPACE_GROUPS,
    _SEGMENT_BOUNDARY_SENTENCE_CHAR_SET,
    _SENTENCE_CLOSERS,
    _SENTENCE_END_PUNCTUATION,
    _URL_EMAIL_PATTERN,
    _char_based_script_group,
    install_safe_console_output,
    resolve_path,
)

# Abbreviation pattern: 2+ single letters each followed by a dot (e.g. U.S.A., O.E.E.)
_ABBREV_DOT_PATTERN = re.compile(r"(?:[A-Za-z]\.){2,}")


# ---------------------------------------------------------------------------
# Shared scanning logic
# ---------------------------------------------------------------------------

def _extract_touching_word(text: str, pos: int, direction: int, max_len: int = 20) -> str:
    """Extract the contiguous word touching position *pos*.

    *direction*: ``-1`` for backwards (word before), ``+1`` for forwards (word after).
    Stops at whitespace, punctuation, or symbols.
    """
    length = len(text)
    if direction == -1:
        i = pos - 1
        count = 0
        while i >= 0 and count < max_len:
            ch = text[i]
            if ch.isspace():
                break
            cat = unicodedata.category(ch)
            if cat.startswith(("P", "S", "Z")):
                break
            i -= 1
            count += 1
        return text[i + 1 : pos]
    else:
        j = pos
        count = 0
        while j < length and count < max_len:
            ch = text[j]
            if ch.isspace():
                break
            cat = unicodedata.category(ch)
            if cat.startswith(("P", "S", "Z")):
                break
            j += 1
            count += 1
        return text[pos:j]


def scan_missing_space_candidates(text: str) -> list[dict]:
    """Find positions where sentence punct touches a word char with no space.

    Returns a list of dicts with keys:
        pos          index of the boundary punct char
        closers_end  index after any closing quotes/brackets (space goes here)
        key          normalized dedup key
        left         left context string (~30 chars)
        right        right context string (~30 chars)
        punct        the punctuation + closers substring
    """
    if not isinstance(text, str) or len(text) < 2:
        return []

    length = len(text)

    # Build exclusion sets.
    url_positions: set[int] = set()
    for m in _URL_EMAIL_PATTERN.finditer(text):
        url_positions.update(range(m.start(), m.end()))

    abbrev_positions: set[int] = set()
    for m in _ABBREV_DOT_PATTERN.finditer(text):
        abbrev_positions.update(range(m.start(), m.end()))

    # Use the full old segmentation boundary set (sentence-end punct + ; : etc.)
    # to match all positions where the old code could have cut.
    boundary_chars = _SEGMENT_BOUNDARY_SENTENCE_CHAR_SET

    candidates: list[dict] = []
    i = 0

    while i < length:
        ch = text[i]

        if ch not in boundary_chars:
            i += 1
            continue

        # Skip inside URL/email or abbreviation.
        if i in url_positions or i in abbrev_positions:
            i += 1
            continue

        # Skip decimal: digit.digit (e.g. 3.14).
        if ch == "." and i > 0 and text[i - 1].isdigit():
            i += 1
            continue

        # Consume optional closing quotes/brackets after the punct.
        j = i + 1
        while j < length and text[j] in _SENTENCE_CLOSERS:
            j += 1

        # Check if next char is word-like with no space.
        if j < length and not text[j].isspace():
            cat = unicodedata.category(text[j])
            is_word = cat.startswith("L") or cat.startswith("N")

            if is_word:
                # Skip same-script no-space languages (CJK, Thai, etc.).
                # Treat Han and Kana as compatible (Japanese mixes these
                # freely without spaces). Hangul is excluded here because
                # ko-KR uses spaces between words.
                char_before = text[i - 1] if i > 0 else ""
                char_after = text[j]
                left_group = _char_based_script_group(char_before)
                right_group = _char_based_script_group(char_after)
                east_asian_compat = {"han", "kana"}
                if left_group in east_asian_compat and right_group in east_asian_compat:
                    i = j
                    continue
                if (
                    left_group is not None
                    and right_group is not None
                    and left_group == right_group
                    and left_group in _CHAR_BASED_NO_SPACE_GROUPS
                ):
                    i = j
                    continue

                punct_str = text[i:j]
                word_before = _extract_touching_word(text, i, -1)
                word_after = _extract_touching_word(text, j, +1)
                key = f"{word_before.casefold()}{punct_str}{word_after.casefold()}"

                left_start = max(0, i - 30)
                right_end = min(length, j + 30)
                left_ctx = text[left_start:i]
                right_ctx = text[j:right_end]

                candidates.append(
                    {
                        "pos": i,
                        "closers_end": j,
                        "key": key,
                        "left": left_ctx,
                        "right": right_ctx,
                        "punct": punct_str,
                    }
                )

        i = j if j > i else i + 1

    return candidates


# ---------------------------------------------------------------------------
# Phase 1: EXTRACT
# ---------------------------------------------------------------------------

def _extract_text_from_line(line: str, is_jsonl: bool) -> str | None:
    if is_jsonl:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        ct = payload.get("corrected_text", "")
        return ct if isinstance(ct, str) else None
    else:
        if "\t" in line:
            parts = line.split("\t")
            return parts[-1].strip()
        return line.strip()


def run_extract(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    is_jsonl = input_path.suffix.lower() == ".jsonl"

    pattern_data: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "examples": []}
    )
    total_occurrences = 0
    lines_scanned = 0
    lines_with_candidates = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.rstrip("\n\r")
            if not raw_line:
                continue

            text = _extract_text_from_line(raw_line, is_jsonl)
            if not text:
                continue

            lines_scanned += 1
            candidates = scan_missing_space_candidates(text)

            if candidates:
                lines_with_candidates += 1

            for c in candidates:
                total_occurrences += 1
                key = c["key"]
                pattern_data[key]["count"] += 1
                if len(pattern_data[key]["examples"]) < 3:
                    pattern_data[key]["examples"].append(
                        {
                            "left": c["left"],
                            "punct": c["punct"],
                            "right": c["right"],
                            "line": line_num,
                        }
                    )

            if lines_scanned % 500_000 == 0:
                print(
                    f"  Scanned {lines_scanned:,} lines, "
                    f"{total_occurrences:,} occurrences, "
                    f"{len(pattern_data):,} unique patterns..."
                )

    sorted_patterns = sorted(pattern_data.items(), key=lambda x: -x[1]["count"])

    output = {
        "total_lines_scanned": lines_scanned,
        "lines_with_candidates": lines_with_candidates,
        "total_occurrences": total_occurrences,
        "unique_patterns": len(sorted_patterns),
        "patterns": [{"key": key, **data} for key, data in sorted_patterns],
    }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Scanned {lines_scanned:,} lines.")
    print(f"Lines with candidates: {lines_with_candidates:,}")
    print(f"Total occurrences: {total_occurrences:,}")
    print(f"Unique patterns: {len(sorted_patterns):,}")
    print(f"Patterns written to: {output_path}")

    print("\nTop 20 patterns:")
    for rank, (key, data) in enumerate(sorted_patterns[:20], 1):
        ex = data["examples"][0]
        left_snippet = ex["left"][-15:]
        right_snippet = ex["right"][:15]
        print(
            f"  {rank:>3}. [{data['count']:>6,}x]  "
            f"...{left_snippet}{ex['punct']}|{right_snippet}..."
        )


# ---------------------------------------------------------------------------
# Phase 2: JUDGE
# ---------------------------------------------------------------------------

def _build_judge_prompt(batch_patterns: list[dict], minimal_context: bool = False) -> str:
    items: list[str] = []
    for idx, p in enumerate(batch_patterns, 1):
        if minimal_context:
            ex = p["examples"][0]
            left = " ".join(ex["left"].split()[-2:])[-15:]
            right = " ".join(ex["right"].split()[:2])[:15]
            snippet = f"{left}{ex['punct']}\u25b8{right}"
            items.append(f'{idx}. "{snippet}"')
        else:
            ex = p["examples"][0]
            left = ex["left"][-10:] if len(ex["left"]) > 10 else ex["left"]
            right = ex["right"][:10] if len(ex["right"]) > 10 else ex["right"]
            snippet = f"{left}{ex['punct']}\u25b8{right}"
            items.append(f'{idx}. "{snippet}"')

    return (
        "You are a text proofreading assistant. Below are text snippets where "
        "punctuation directly touches the next character (marked with \u25b8) "
        "with no space between them.\n\n"
        "For each snippet, determine if a SPACE should be inserted at the \u25b8 position.\n\n"
        "Rules:\n"
        "- YES: Sentence boundaries where a new sentence or clause starts after the punctuation\n"
        "- NO: Abbreviations (U.S., Dr., etc.), decimal-like patterns, compound expressions, "
        "or any case where no space is correct in that language\n\n"
        "Respond ONLY with a JSON object mapping the item number (as string) to \"YES\" or \"NO\".\n"
        'Example: {"1": "YES", "2": "NO"}\n\n'
        "Items:\n" + "\n".join(items)
    )


async def run_judge(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    from openai import AsyncAzureOpenAI

    patterns_path = resolve_path(args.patterns)
    if not patterns_path.exists():
        print(f"Patterns file not found: {patterns_path}")
        sys.exit(1)

    data = json.loads(patterns_path.read_text(encoding="utf-8"))
    patterns = data.get("patterns", [])

    if not patterns:
        print("No patterns to judge.")
        sys.exit(0)

    print(f"Judging {len(patterns)} unique patterns...")

    endpoint = args.endpoint or os.environ.get(
        "AZURE_OPENAI_ENDPOINT"
    )
    deployment = args.deployment or os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT"
    )
    api_version = args.api_version or os.environ.get(
        "API_VERSION"
    )
    batch_size = args.batch_size
    concurrency = args.concurrency

    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    semaphore = asyncio.Semaphore(concurrency)
    verdicts: dict[str, bool] = {}
    failed_batches = 0

    # Resume: load existing verdicts if output file already exists.
    output_path = resolve_path(args.output)
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            for k, v in existing.get("verdicts", {}).items():
                if isinstance(v, str) and v != "UNRESOLVED":
                    verdicts[k] = v == "YES"
            print(f"  Resumed {len(verdicts)} existing verdicts from {output_path.name}")
        except (json.JSONDecodeError, KeyError):
            pass

    # Filter to patterns that still need judging.
    remaining = [p for p in patterns if p["key"] not in verdicts]
    if not remaining:
        print("All patterns already judged. Nothing to do.")
    else:
        print(f"  {len(remaining)} patterns remaining to judge...")

    filtered_keys: set[str] = set()

    async def judge_batch(
        batch_patterns: list[dict], batch_idx: int,
        use_minimal_context: bool = False,
    ) -> None:
        nonlocal failed_batches
        async with semaphore:
            prompt = _build_judge_prompt(batch_patterns, minimal_context=use_minimal_context)
            try:
                response = await client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                result = json.loads(content)

                for idx, p in enumerate(batch_patterns, 1):
                    verdict_str = str(result.get(str(idx), "")).strip().upper()
                    verdicts[p["key"]] = verdict_str == "YES"

                yes_in_batch = sum(
                    1
                    for idx in range(1, len(batch_patterns) + 1)
                    if str(result.get(str(idx), "")).strip().upper() == "YES"
                )
                print(
                    f"  Batch {batch_idx + 1}: "
                    f"{len(batch_patterns)} patterns, "
                    f"{yes_in_batch} YES / {len(batch_patterns) - yes_in_batch} NO"
                )
            except Exception as e:
                failed_batches += 1
                is_permanent = getattr(e, 'status_code', 0) == 400
                if is_permanent:
                    print(f"  Batch {batch_idx + 1} FILTERED (400): will retry with minimal context")
                    for p in batch_patterns:
                        filtered_keys.add(p["key"])
                else:
                    print(f"  Batch {batch_idx + 1} FAILED: {e}")
                for p in batch_patterns:
                    if p["key"] not in verdicts:
                        verdicts[p["key"]] = None

    # Build batches from remaining patterns.
    batches: list[list[dict]] = []
    for i in range(0, len(remaining), batch_size):
        batches.append(remaining[i : i + batch_size])

    tasks = [judge_batch(batch, idx) for idx, batch in enumerate(batches)]
    await asyncio.gather(*tasks)

    # Auto-retry unresolved patterns (up to 3 attempts).
    # Use minimal context for patterns that were filtered (400).
    for retry_round in range(1, 4):
        unresolved_patterns = [p for p in remaining if verdicts.get(p["key"]) is None]
        if not unresolved_patterns:
            break
        use_minimal = any(p["key"] in filtered_keys for p in unresolved_patterns)
        print(f"\n  Auto-retry {retry_round}: {len(unresolved_patterns)} unresolved patterns"
              f"{' (minimal context)' if use_minimal else ''}...")
        retry_batches = [unresolved_patterns[i:i + batch_size] for i in range(0, len(unresolved_patterns), batch_size)]
        retry_tasks = [judge_batch(batch, idx, use_minimal_context=use_minimal) for idx, batch in enumerate(retry_batches)]
        await asyncio.gather(*retry_tasks)

    # Any patterns still filtered after retries → default to NO (safe).
    for key in filtered_keys:
        if verdicts.get(key) is None:
            verdicts[key] = False

    await client.close()

    yes_count = sum(1 for v in verdicts.values() if v is True)
    no_count = sum(1 for v in verdicts.values() if v is False)
    unresolved_count = sum(1 for v in verdicts.values() if v is None)

    output_data = {
        "total_patterns": len(verdicts),
        "space_yes": yes_count,
        "space_no": no_count,
        "unresolved": unresolved_count,
        "verdicts": {
            k: ("YES" if v is True else "NO" if v is False else "UNRESOLVED")
            for k, v in verdicts.items()
        },
    }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nVerdicts: {yes_count} YES, {no_count} NO, {unresolved_count} unresolved"
    )
    if failed_batches:
        print(f"WARNING: {failed_batches} batch(es) failed.")
    print(f"Written to: {output_path}")


# ---------------------------------------------------------------------------
# Phase 3: APPLY
# ---------------------------------------------------------------------------

def _load_verdicts(verdicts_path: Path) -> dict[str, bool]:
    data = json.loads(verdicts_path.read_text(encoding="utf-8"))
    raw_verdicts = data.get("verdicts", {})
    return {
        k: (v == "YES" if isinstance(v, str) else bool(v))
        for k, v in raw_verdicts.items()
    }


def apply_verdicts_to_text(
    text: str, verdicts: dict[str, bool]
) -> tuple[str, int]:
    """Apply space verdicts to a text string. Returns (fixed_text, num_fixes)."""
    candidates = scan_missing_space_candidates(text)
    if not candidates:
        return text, 0

    # Filter to YES verdicts only.
    fix_positions: list[int] = []
    for c in candidates:
        verdict = verdicts.get(c["key"])
        if verdict is True:
            fix_positions.append(c["closers_end"])

    if not fix_positions:
        return text, 0

    # Build result by inserting spaces at fix positions (ascending order).
    fix_set = set(fix_positions)
    parts: list[str] = []
    for idx, ch in enumerate(text):
        if idx in fix_set:
            parts.append(" ")
        parts.append(ch)

    return "".join(parts), len(fix_positions)


def run_apply(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    verdicts_path = resolve_path(args.verdicts)
    if not verdicts_path.exists():
        print(f"Verdicts file not found: {verdicts_path}")
        sys.exit(1)

    verdicts = _load_verdicts(verdicts_path)
    yes_count = sum(1 for v in verdicts.values() if v)
    print(f"Loaded {len(verdicts)} verdicts ({yes_count} YES)")

    is_jsonl = input_path.suffix.lower() == ".jsonl"

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_fixes = 0
    lines_fixed = 0
    lines_processed = 0
    missing_verdict_keys: set[str] = set()

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        for raw_line in fin:
            line = raw_line.rstrip("\n\r")
            if not line:
                fout.write("\n")
                continue

            lines_processed += 1

            if is_jsonl:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    fout.write(line + "\n")
                    continue

                if not isinstance(payload, dict):
                    fout.write(line + "\n")
                    continue

                line_fixes = 0

                # Fix corrected_text.
                ct = payload.get("corrected_text", "")
                if isinstance(ct, str) and ct:
                    fixed, n = apply_verdicts_to_text(ct, verdicts)
                    if n > 0:
                        payload["corrected_text"] = fixed
                        line_fixes += n

                if line_fixes > 0:
                    total_fixes += line_fixes
                    lines_fixed += 1

                fout.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            else:
                # TSV/TXT: fix last column.
                if "\t" in line:
                    parts = line.split("\t")
                    last_col = parts[-1]
                    for c in scan_missing_space_candidates(last_col):
                        if c["key"] not in verdicts:
                            missing_verdict_keys.add(c["key"])
                    fixed, n = apply_verdicts_to_text(last_col, verdicts)
                    if n > 0:
                        parts[-1] = fixed
                        total_fixes += n
                        lines_fixed += 1
                    fout.write("\t".join(parts) + "\n")
                else:
                    for c in scan_missing_space_candidates(line):
                        if c["key"] not in verdicts:
                            missing_verdict_keys.add(c["key"])
                    fixed, n = apply_verdicts_to_text(line, verdicts)
                    if n > 0:
                        total_fixes += n
                        lines_fixed += 1
                    fout.write(fixed + "\n")

            if lines_processed % 500_000 == 0:
                print(
                    f"  Applied {total_fixes:,} fixes in "
                    f"{lines_fixed:,}/{lines_processed:,} lines..."
                )

    print(f"\nProcessed {lines_processed:,} lines.")
    print(f"Fixed {lines_fixed:,} lines ({total_fixes:,} space insertions).")
    if missing_verdict_keys:
        print(f"Missing verdicts: {len(missing_verdict_keys):,} unique pattern(s) had no verdict (treated as NO).")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    install_safe_console_output()

    parser = argparse.ArgumentParser(
        description="Fix missing spaces at segment boundaries in processed output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- extract --
    extract_parser = subparsers.add_parser(
        "extract", help="Scan output for punct+word patterns"
    )
    extract_parser.add_argument(
        "--input", required=True, help="Input file (JSONL, TSV, or TXT)"
    )
    extract_parser.add_argument(
        "--output", required=True, help="Output patterns JSON file"
    )

    # -- judge --
    judge_parser = subparsers.add_parser(
        "judge", help="Send patterns to LLM for space verdicts"
    )
    judge_parser.add_argument(
        "--patterns", required=True, help="Input patterns JSON file"
    )
    judge_parser.add_argument(
        "--output", required=True, help="Output verdicts JSON file"
    )
    judge_parser.add_argument("--endpoint", default=None)
    judge_parser.add_argument("--deployment", default=None)
    judge_parser.add_argument("--api-version", default=None)
    judge_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of patterns per LLM call (default: 50)",
    )
    judge_parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent LLM calls (default: 5)",
    )

    # -- apply --
    apply_parser = subparsers.add_parser(
        "apply", help="Apply verdicts to fix output file"
    )
    apply_parser.add_argument(
        "--input", required=True, help="Input file to fix (JSONL, TSV, or TXT)"
    )
    apply_parser.add_argument(
        "--verdicts", required=True, help="Verdicts JSON file from judge phase"
    )
    apply_parser.add_argument(
        "--output", required=True, help="Fixed output file"
    )

    args = parser.parse_args()

    if args.command == "extract":
        run_extract(args)
    elif args.command == "judge":
        asyncio.run(run_judge(args))
    elif args.command == "apply":
        run_apply(args)


if __name__ == "__main__":
    main()
