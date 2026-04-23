"""Fix incorrect digit-letter spacing in processed output.

Three-phase workflow:
  extract  Scan for digit<space>letter and letter<space>digit patterns,
           deduplicate, write patterns JSON.
  judge    Send unique patterns to LLM for YES/NO space verdicts.
  apply    Apply verdicts to fix the output file.

Usage:
  python fix_digit_letter_spaces.py extract --input outputs/file.tsv --output patterns.json
  python fix_digit_letter_spaces.py judge   --patterns patterns.json --output verdicts.json
  python fix_digit_letter_spaces.py apply   --input outputs/file.tsv --verdicts verdicts.json --output fixed.tsv
"""

import argparse
import asyncio
import json
import os
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from common import (
    _is_non_latin_letter,
    install_safe_console_output,
    resolve_path,
)


# ---------------------------------------------------------------------------
# Scanning: find digit<space>Latin patterns
# ---------------------------------------------------------------------------

def _extract_token(text: str, pos: int, direction: int, max_len: int = 20) -> str:
    """Extract the contiguous token touching position *pos*.

    *direction*: -1 for backwards, +1 for forwards.
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


def scan_digit_letter_space_candidates(text: str) -> list[dict]:
    """Find positions where a single space sits between a digit and a Latin letter.

    Returns a list of dicts with keys:
        pos    index of the space character
        key    normalized dedup key (e.g. "abc 123" or "3 km")
        left   left context string (~30 chars)
        right  right context string (~30 chars)
    """
    if not isinstance(text, str) or len(text) < 3:
        return []

    length = len(text)
    candidates: list[dict] = []

    for i in range(1, length - 1):
        if text[i] != " ":
            continue
        # Skip if part of a whitespace run.
        if text[i - 1].isspace() or text[i + 1].isspace():
            continue

        prev_ch = text[i - 1]
        next_ch = text[i + 1]

        # Check digit<space>Latin or Latin<space>digit.
        is_digit_latin = prev_ch.isdigit() and next_ch.isalpha() and not _is_non_latin_letter(next_ch)
        is_latin_digit = next_ch.isdigit() and prev_ch.isalpha() and not _is_non_latin_letter(prev_ch)

        if not (is_digit_latin or is_latin_digit):
            continue

        # Skip speaker labels: e.g. "[Speaker 2]:", "[John 1]:", "SPK 1:".
        word_before = _extract_token(text, i, -1)
        word_after = _extract_token(text, i + 1, +1)

        # Check if inside a bracketed speaker label [*digit*]:
        # by scanning backwards for '[' and forwards for ']:'.
        if is_latin_digit:
            # word_before is Latin, word_after is digit — e.g. "Speaker 2"
            # Check if enclosed in brackets followed by colon.
            after_end = i + 1 + len(word_after)
            scan_fwd = after_end
            while scan_fwd < length and text[scan_fwd] in {"]", ")", "}", " "}:
                scan_fwd += 1
            has_colon_after = scan_fwd < length and text[scan_fwd] in {":", "\uff1a"}

            # Check if there's an opening bracket before the word.
            scan_back = i - len(word_before) - 1
            while scan_back >= 0 and text[scan_back] in {"[", "(", " "}:
                scan_back -= 1
            has_bracket_before = (i - len(word_before) - 1 >= 0 and
                                  text[i - len(word_before) - 1] in {"[", "("})

            wb_lower = word_before.casefold()
            if has_bracket_before:
                if has_colon_after:
                    continue  # Skip — inside bracketed label with colon.
                if wb_lower in {"speaker", "spk", "spkr", "s", "sp",
                                "\u0433\u043e\u0432\u043e\u0440\u044f\u0449\u0438\u0439"}:
                    continue  # Skip — bracketed speaker label without colon.

            # Also skip common speaker prefixes without brackets but with colon.
            if has_colon_after and wb_lower in {"speaker", "spk", "spkr", "s", "sp",
                                                 "\u0433\u043e\u0432\u043e\u0440\u044f\u0449\u0438\u0439"}:
                continue

        key = f"{word_before.casefold()} {word_after.casefold()}"

        left_start = max(0, i - 30)
        right_end = min(length, i + 1 + 30)
        left_ctx = text[left_start:i]
        right_ctx = text[i + 1 : right_end]

        candidates.append({
            "pos": i,
            "key": key,
            "left": left_ctx,
            "right": right_ctx,
        })

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
            candidates = scan_digit_letter_space_candidates(text)

            if candidates:
                lines_with_candidates += 1

            for c in candidates:
                total_occurrences += 1
                key = c["key"]
                pattern_data[key]["count"] += 1
                if len(pattern_data[key]["examples"]) < 3:
                    pattern_data[key]["examples"].append({
                        "left": c["left"],
                        "right": c["right"],
                        "line": line_num,
                    })

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
            f"...{left_snippet}|{right_snippet}..."
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
            key_parts = p["key"].split(" ", 1)
            if len(key_parts) == 2:
                snippet = f"{left}{key_parts[0]}\u25b8{key_parts[1]}{right}"
            else:
                snippet = f"{left}\u25b8{right}"
            items.append(f'{idx}. "{snippet}"')
        else:
            ex = p["examples"][0]
            left = ex["left"][-10:] if len(ex["left"]) > 10 else ex["left"]
            right = ex["right"][:10] if len(ex["right"]) > 10 else ex["right"]
            key_parts = p["key"].split(" ", 1)
            if len(key_parts) == 2:
                snippet = f"{left}{key_parts[0]}\u25b8{key_parts[1]}{right}"
            else:
                snippet = f"{left}\u25b8{right}"
            items.append(f'{idx}. "{snippet}"')

    return (
        "You are a text proofreading assistant. Below are text snippets where "
        "a space exists between a digit and a letter (marked with \u25b8). "
        "The space may or may not be correct.\n\n"
        "For each snippet, determine if the space at the \u25b8 position should be REMOVED.\n\n"
        "Rules:\n"
        "- REMOVE: If the digit and letter form a single token/compound "
        "(e.g. '3 km' should be '3km', 'mp 3' should be 'mp3', 'v 2' should be 'v2')\n"
        "- REMOVE: If the digit+letter form a version, model name, unit, or identifier\n"
        "- REMOVE: Chess algebraic notation — a letter (a-h/A-H) + digit (1-8) or "
        "superscript digit denoting a board square (e.g. 'b 7'→'b7', 'c ⁴'→'c⁴', "
        "'f 3'→'f3'). Context clues: words like слон/конь/ферзь/ладья/пешка/bishop/"
        "knight/queen/rook/pawn nearby, or chess-related discussion.\n"
        "- KEEP: If the space is natural in that language context "
        "(e.g. 'I have 3 dogs', 'section 5 above', 'page 42 onwards')\n"
        "- KEEP: If the digit is part of a number and the letter starts a new word\n\n"
        "Respond ONLY with a JSON object mapping the item number (as string) to "
        '"REMOVE" or "KEEP".\n'
        'Example: {"1": "REMOVE", "2": "KEEP"}\n\n'
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
                    verdicts[k] = v == "REMOVE"
            print(f"  Resumed {len(verdicts)} existing verdicts from {output_path.name}")
        except (json.JSONDecodeError, KeyError):
            pass

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
                    # REMOVE means space should be removed (True = remove space)
                    verdicts[p["key"]] = verdict_str == "REMOVE"

                remove_in_batch = sum(
                    1
                    for idx in range(1, len(batch_patterns) + 1)
                    if str(result.get(str(idx), "")).strip().upper() == "REMOVE"
                )
                print(
                    f"  Batch {batch_idx + 1}: "
                    f"{len(batch_patterns)} patterns, "
                    f"{remove_in_batch} REMOVE / {len(batch_patterns) - remove_in_batch} KEEP"
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

    batches: list[list[dict]] = []
    for i in range(0, len(remaining), batch_size):
        batches.append(remaining[i : i + batch_size])

    tasks = [judge_batch(batch, idx) for idx, batch in enumerate(batches)]
    await asyncio.gather(*tasks)

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

    for key in filtered_keys:
        if verdicts.get(key) is None:
            verdicts[key] = False

    await client.close()

    remove_count = sum(1 for v in verdicts.values() if v is True)
    keep_count = sum(1 for v in verdicts.values() if v is False)
    unresolved_count = sum(1 for v in verdicts.values() if v is None)

    output_data = {
        "total_patterns": len(verdicts),
        "space_remove": remove_count,
        "space_keep": keep_count,
        "unresolved": unresolved_count,
        "verdicts": {
            k: ("REMOVE" if v is True else "KEEP" if v is False else "UNRESOLVED")
            for k, v in verdicts.items()
        },
    }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nVerdicts: {remove_count} REMOVE, {keep_count} KEEP, {unresolved_count} unresolved"
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
        k: (v == "REMOVE" if isinstance(v, str) else bool(v))
        for k, v in raw_verdicts.items()
    }


def apply_verdicts_to_text(
    text: str, verdicts: dict[str, bool]
) -> tuple[str, int]:
    """Remove spaces where verdicts say REMOVE. Returns (fixed_text, num_fixes)."""
    candidates = scan_digit_letter_space_candidates(text)
    if not candidates:
        return text, 0

    # Collect positions to remove (REMOVE verdicts only).
    remove_positions: set[int] = set()
    for c in candidates:
        verdict = verdicts.get(c["key"])
        if verdict is True:
            remove_positions.add(c["pos"])

    if not remove_positions:
        return text, 0

    # Build result by skipping space chars at remove positions.
    parts: list[str] = []
    for idx, ch in enumerate(text):
        if idx in remove_positions:
            continue  # Skip this space.
        parts.append(ch)

    return "".join(parts), len(remove_positions)


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
    remove_count = sum(1 for v in verdicts.values() if v)
    print(f"Loaded {len(verdicts)} verdicts ({remove_count} REMOVE)")

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
                    for c in scan_digit_letter_space_candidates(last_col):
                        if c["key"] not in verdicts:
                            missing_verdict_keys.add(c["key"])
                    fixed, n = apply_verdicts_to_text(last_col, verdicts)
                    if n > 0:
                        parts[-1] = fixed
                        total_fixes += n
                        lines_fixed += 1
                    fout.write("\t".join(parts) + "\n")
                else:
                    for c in scan_digit_letter_space_candidates(line):
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
    print(f"Fixed {lines_fixed:,} lines ({total_fixes:,} spaces removed).")
    if missing_verdict_keys:
        print(f"Missing verdicts: {len(missing_verdict_keys):,} unique pattern(s) had no verdict (treated as KEEP).")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    install_safe_console_output()

    parser = argparse.ArgumentParser(
        description="Fix incorrect digit-letter spacing in processed output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- extract --
    extract_parser = subparsers.add_parser(
        "extract", help="Scan output for digit<space>letter patterns"
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
