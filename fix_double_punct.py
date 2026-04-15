"""Fix double/stacked punctuation issues in processed output.

Catches cases where two punctuation marks are adjacent and one should be removed,
e.g. ".," → ".", ",." → ".", ".." → ".", "!." → "!" etc.

Three-phase workflow:
  extract  Scan for adjacent punctuation patterns, deduplicate, write patterns JSON.
  judge    Send unique patterns to LLM for verdicts on which form to keep.
  apply    Apply verdicts to fix the output file.

Usage:
  python fix_double_punct.py extract --input outputs/file.tsv --output patterns.json
  python fix_double_punct.py judge   --patterns patterns.json --output verdicts.json
  python fix_double_punct.py apply   --input outputs/file.tsv --verdicts verdicts.json --output fixed.tsv
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
    DEFAULT_AOAI_API_VERSION,
    DEFAULT_AOAI_DEPLOYMENT,
    DEFAULT_AOAI_ENDPOINT,
    install_safe_console_output,
    resolve_path,
)


# Punctuation characters to consider (sentence-ending + common mid-sentence).
_PUNCT_CHARS = set(".,;:!?。，；：！？、·…\u2026\uff0c\uff0e\uff01\uff1f\uff1b\uff1a")

# Pattern: 2+ punctuation chars (possibly with whitespace between).
# We capture the full run to let the LLM decide what to keep.
_DOUBLE_PUNCT_RE = re.compile(
    r"([.,;:!?。，；：！？、·…\uff0c\uff0e\uff01\uff1f\uff1b\uff1a])"
    r"(\s*)"
    r"([.,;:!?。，；：！？、·…\uff0c\uff0e\uff01\uff1f\uff1b\uff1a])"
)

# Legitimate multi-char punctuation to skip.
_LEGIT_MULTI = {"...", "…", "!!", "??", "!?", "?!", ".."}


def scan_double_punct_candidates(text: str) -> list[dict]:
    """Find positions where two punctuation marks are adjacent (with optional space).

    Returns dicts with keys: pos, key, left, right, match_text, replacement_hint.
    *pos* is the start index of the matched double-punct sequence.
    """
    if not isinstance(text, str) or len(text) < 2:
        return []

    candidates: list[dict] = []

    for m in _DOUBLE_PUNCT_RE.finditer(text):
        p1 = m.group(1)
        space = m.group(2)
        p2 = m.group(3)
        full = m.group(0)

        # Skip legitimate multi-char punctuation.
        stripped = p1 + p2
        if stripped in _LEGIT_MULTI:
            continue
        # Skip ellipsis patterns (3+ dots).
        if p1 == "." and p2 == ".":
            # Check if part of "..." by looking ahead/behind.
            start = m.start()
            end = m.end()
            if start > 0 and text[start - 1] == ".":
                continue
            if end < len(text) and text[end] == ".":
                continue

        pos = m.start()
        key = f"{p1}{space}{p2}"

        left_start = max(0, pos - 25)
        right_end = min(len(text), m.end() + 25)
        left_ctx = text[left_start:pos]
        right_ctx = text[m.end():right_end]

        candidates.append({
            "pos": pos,
            "key": key,
            "left": left_ctx,
            "right": right_ctx,
            "match_text": full,
        })

    return candidates


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Phase 1: EXTRACT
# ---------------------------------------------------------------------------

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
            candidates = scan_double_punct_candidates(text)

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
                        "match": c["match_text"],
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
            f"...{left_snippet}[{repr(key)}]{right_snippet}..."
        )


# ---------------------------------------------------------------------------
# Phase 2: JUDGE
# ---------------------------------------------------------------------------

def _build_judge_prompt(batch_patterns: list[dict]) -> str:
    items: list[str] = []
    for idx, p in enumerate(batch_patterns, 1):
        ex = p["examples"][0]
        left = ex["left"][-25:] if len(ex["left"]) > 25 else ex["left"]
        right = ex["right"][:25] if len(ex["right"]) > 25 else ex["right"]
        match = ex["match"]
        snippet = f"{left}\u25b8{match}\u25c2{right}"
        items.append(f"{idx}. key={repr(p['key'])} context: \"{snippet}\"")

    return (
        "You are a text proofreading assistant. Below are text snippets where "
        "two punctuation marks appear adjacent to each other (between \u25b8 and \u25c2). "
        "One of them may be redundant.\n\n"
        "For each snippet, determine what the punctuation between the markers should be replaced with.\n\n"
        "Rules:\n"
        "- If one mark is redundant, keep the stronger/more appropriate one "
        "(e.g. '.,'->'.' , ',.'->'.' , '!.'->'!' , '?.'->'?')\n"
        "- A period after a stronger mark (!?) is usually redundant\n"
        "- A comma next to a period is usually redundant - keep the period\n"
        "- If the double punctuation is intentional or meaningful, keep it as-is "
        "(e.g. '...' for ellipsis, '?!' for rhetorical emphasis)\n"
        "- Consider language conventions\n\n"
        "Respond ONLY with a JSON object mapping the item number (as string) to "
        "the replacement string.\n"
        'Example: {"1": ".", "2": "!", "3": ".,"} (use the original if keeping as-is)\n\n'
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
        "AZURE_OPENAI_ENDPOINT", DEFAULT_AOAI_ENDPOINT
    )
    deployment = args.deployment or os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT", DEFAULT_AOAI_DEPLOYMENT
    )
    api_version = args.api_version or os.environ.get(
        "AZURE_OPENAI_API_VERSION", DEFAULT_AOAI_API_VERSION
    )
    batch_size = args.batch_size
    concurrency = args.concurrency

    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    semaphore = asyncio.Semaphore(concurrency)
    verdicts: dict[str, str] = {}
    failed_batches = 0

    async def judge_batch(
        batch_patterns: list[dict], batch_idx: int
    ) -> None:
        nonlocal failed_batches
        async with semaphore:
            prompt = _build_judge_prompt(batch_patterns)
            try:
                response = await client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                result = json.loads(content)

                changed = 0
                for idx, p in enumerate(batch_patterns, 1):
                    replacement = str(result.get(str(idx), p["key"])).strip()
                    verdicts[p["key"]] = replacement
                    if replacement != p["key"]:
                        changed += 1

                print(
                    f"  Batch {batch_idx + 1}: "
                    f"{len(batch_patterns)} patterns, "
                    f"{changed} changed / {len(batch_patterns) - changed} kept"
                )
            except Exception as e:
                failed_batches += 1
                print(f"  Batch {batch_idx + 1} FAILED: {e}")
                for p in batch_patterns:
                    if p["key"] not in verdicts:
                        verdicts[p["key"]] = None

    batches: list[list[dict]] = []
    for i in range(0, len(patterns), batch_size):
        batches.append(patterns[i : i + batch_size])

    tasks = [judge_batch(batch, idx) for idx, batch in enumerate(batches)]
    await asyncio.gather(*tasks)

    # Auto-retry unresolved.
    for retry_round in range(1, 4):
        unresolved = [p for p in patterns if verdicts.get(p["key"]) is None]
        if not unresolved:
            break
        print(f"\n  Auto-retry {retry_round}: {len(unresolved)} unresolved...")
        retry_batches = [unresolved[i:i + batch_size] for i in range(0, len(unresolved), batch_size)]
        retry_tasks = [judge_batch(batch, idx) for idx, batch in enumerate(retry_batches)]
        await asyncio.gather(*retry_tasks)

    await client.close()

    changed_count = sum(1 for k, v in verdicts.items() if v is not None and v != k)
    kept_count = sum(1 for k, v in verdicts.items() if v is not None and v == k)
    unresolved_count = sum(1 for v in verdicts.values() if v is None)

    output_data = {
        "total_patterns": len(verdicts),
        "changed": changed_count,
        "kept": kept_count,
        "unresolved": unresolved_count,
        "verdicts": {k: v if v is not None else "UNRESOLVED" for k, v in verdicts.items()},
    }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nVerdicts: {changed_count} changed, {kept_count} kept, {unresolved_count} unresolved"
    )
    if failed_batches:
        print(f"WARNING: {failed_batches} batch(es) failed.")
    print(f"Written to: {output_path}")


# ---------------------------------------------------------------------------
# Phase 3: APPLY
# ---------------------------------------------------------------------------

def _load_verdicts(verdicts_path: Path) -> dict[str, str]:
    data = json.loads(verdicts_path.read_text(encoding="utf-8"))
    raw = data.get("verdicts", {})
    return {k: v for k, v in raw.items() if isinstance(v, str) and v != "UNRESOLVED"}


def apply_verdicts_to_text(
    text: str, verdicts: dict[str, str]
) -> tuple[str, int]:
    """Replace double-punct patterns per verdicts. Returns (fixed_text, num_fixes)."""
    candidates = scan_double_punct_candidates(text)
    if not candidates:
        return text, 0

    # Collect replacements sorted by position descending (right-to-left).
    replacements: list[tuple[int, int, str]] = []
    for c in candidates:
        verdict = verdicts.get(c["key"])
        if verdict is None or verdict == c["key"]:
            continue  # No change or no verdict.
        start = c["pos"]
        end = start + len(c["match_text"])
        replacements.append((start, end, verdict))

    if not replacements:
        return text, 0

    # Apply right-to-left so positions stay valid.
    result = list(text)
    num_fixes = 0
    for start, end, replacement in sorted(replacements, reverse=True):
        result[start:end] = list(replacement)
        num_fixes += 1

    return "".join(result), num_fixes


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
    changed = sum(1 for k, v in verdicts.items() if v != k)
    print(f"Loaded {len(verdicts)} verdicts ({changed} to change)")

    is_jsonl = input_path.suffix.lower() == ".jsonl"

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_fixes = 0
    lines_fixed = 0
    lines_processed = 0

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
                if "\t" in line:
                    parts = line.split("\t")
                    fixed, n = apply_verdicts_to_text(parts[-1], verdicts)
                    if n > 0:
                        parts[-1] = fixed
                        total_fixes += n
                        lines_fixed += 1
                    fout.write("\t".join(parts) + "\n")
                else:
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
    print(f"Fixed {lines_fixed:,} lines ({total_fixes:,} replacements).")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    install_safe_console_output()

    parser = argparse.ArgumentParser(
        description="Fix double/stacked punctuation in processed output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- extract --
    extract_parser = subparsers.add_parser(
        "extract", help="Scan output for double punctuation patterns"
    )
    extract_parser.add_argument(
        "--input", required=True, help="Input file (JSONL, TSV, or TXT)"
    )
    extract_parser.add_argument(
        "--output", required=True, help="Output patterns JSON file"
    )

    # -- judge --
    judge_parser = subparsers.add_parser(
        "judge", help="Send patterns to LLM for punctuation verdicts"
    )
    judge_parser.add_argument(
        "--patterns", required=True, help="Input patterns JSON file"
    )
    judge_parser.add_argument(
        "--output", required=True, help="Output verdicts JSON file"
    )
    judge_parser.add_argument(
        "--endpoint", default=None, help="Azure OpenAI endpoint"
    )
    judge_parser.add_argument(
        "--deployment", default=None, help="Azure OpenAI deployment"
    )
    judge_parser.add_argument(
        "--api-version", default=None, help="Azure OpenAI API version"
    )
    judge_parser.add_argument(
        "--batch-size", type=int, default=30, help="Patterns per LLM batch"
    )
    judge_parser.add_argument(
        "--concurrency", type=int, default=5, help="Max concurrent LLM requests"
    )

    # -- apply --
    apply_parser = subparsers.add_parser(
        "apply", help="Apply verdicts to fix the output file"
    )
    apply_parser.add_argument(
        "--input", required=True, help="Input file to fix"
    )
    apply_parser.add_argument(
        "--verdicts", required=True, help="Verdicts JSON file"
    )
    apply_parser.add_argument(
        "--output", required=True, help="Output fixed file"
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
