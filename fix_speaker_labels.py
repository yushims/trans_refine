"""Fix speaker label spacing by comparing against source input.

If the source has [SPK1]: (no space), the output should match.
If the source has [SPK 1]: (with space), the output should match.
If the label was added by the LLM (not in source), remove the space.

Usage:
  python fix_speaker_labels.py --input inputs/file.tsv --output-to-fix outputs/file.tsv --output outputs/fixed.tsv
"""

import argparse
import re
import sys

from common import install_safe_console_output, resolve_path


# Match speaker labels like [SPK1]:, [SPK 1]:, [Speaker2]:, [Speaker 2]:
# Colon is optional — some data uses [SPK 1] without colon.
_SPEAKER_LABEL_PATTERN = re.compile(
    r"\[([^\]]*?)\b(\w+)\s+(\d+)\b([^\]]*?)\]\s*[:\uff1a]?"
)
_SPEAKER_LABEL_NO_SPACE_PATTERN = re.compile(
    r"\[([^\]]*?)\b(\w+)(\d+)\b([^\]]*?)\]\s*[:\uff1a]?"
)


def _extract_speaker_labels(text: str) -> dict[str, str]:
    """Extract speaker labels and their exact format from text.
    
    Returns dict mapping normalized label (e.g. 'spk1') to exact text (e.g. 'SPK1' or 'SPK 1').
    """
    labels: dict[str, str] = {}
    
    # Match labels with space: [SPK 1]:
    for m in _SPEAKER_LABEL_PATTERN.finditer(text):
        prefix, word, digit, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        normalized = f"{word.lower()}{digit}"
        labels[normalized] = f"{word} {digit}"
    
    # Match labels without space: [SPK1]:
    for m in _SPEAKER_LABEL_NO_SPACE_PATTERN.finditer(text):
        prefix, word, digit, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        normalized = f"{word.lower()}{digit}"
        if normalized not in labels:  # Don't overwrite space version
            labels[normalized] = f"{word}{digit}"
    
    return labels


def fix_speaker_labels_in_text(
    output_text: str,
    source_labels: dict[str, str],
) -> tuple[str, int]:
    """Fix speaker label spacing in output_text based on source labels.
    
    Returns (fixed_text, num_fixes).
    """
    fixes = 0
    
    def _replace_with_space(m: re.Match) -> str:
        nonlocal fixes
        prefix, word, digit, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        normalized = f"{word.lower()}{digit}"
        colon_part = m.group(0)[m.group(0).rindex("]"):]  # ]: or ]:
        bracket_content = m.group(0)[:m.group(0).rindex("]")]
        
        if normalized in source_labels:
            source_form = source_labels[normalized]
            # Check if source has space or not
            if " " in source_form:
                # Source has space — keep space (already has it)
                return m.group(0)
            else:
                # Source has no space — remove space
                fixed = m.group(0).replace(f"{word} {digit}", f"{word}{digit}", 1)
                if fixed != m.group(0):
                    fixes += 1
                return fixed
        else:
            # Label not in source (LLM-added) — default to no space
            fixed = m.group(0).replace(f"{word} {digit}", f"{word}{digit}", 1)
            if fixed != m.group(0):
                fixes += 1
            return fixed
    
    result = _SPEAKER_LABEL_PATTERN.sub(_replace_with_space, output_text)
    return result, fixes


def main() -> None:
    install_safe_console_output()

    parser = argparse.ArgumentParser(
        description="Fix speaker label spacing by comparing against source input.",
    )
    parser.add_argument("--input", required=True, help="Source input file (TSV/TXT).")
    parser.add_argument("--output-to-fix", required=True, help="Output file to fix.")
    parser.add_argument("--output", required=True, help="Fixed output file.")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    fix_path = resolve_path(args.output_to_fix)
    output_path = resolve_path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    if not fix_path.exists():
        print(f"Output-to-fix file not found: {fix_path}")
        sys.exit(1)

    # Build global source label map from all input lines.
    print(f"Reading input: {input_path}")
    inp_lines = input_path.read_text(encoding="utf-8").split('\n')
    if inp_lines and inp_lines[-1] == '':
        inp_lines.pop()
    print(f"  {len(inp_lines):,} lines")

    source_labels: dict[str, str] = {}
    for line in inp_lines:
        parts = line.split("\t")
        text = parts[-1] if parts else line
        labels = _extract_speaker_labels(text)
        source_labels.update(labels)

    print(f"  Found {len(source_labels)} unique speaker label formats in source")
    for norm, form in sorted(source_labels.items())[:10]:
        print(f"    {norm} -> {form}")

    # Fix output file.
    print(f"Reading output to fix: {fix_path}")
    fix_lines = fix_path.read_text(encoding="utf-8").split('\n')
    if fix_lines and fix_lines[-1] == '':
        fix_lines.pop()
    print(f"  {len(fix_lines):,} lines")

    fixed_lines: list[str] = []
    total_fixes = 0
    lines_fixed = 0

    for line in fix_lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            last_col = parts[-1]
            fixed, n = fix_speaker_labels_in_text(last_col, source_labels)
            if n > 0:
                parts[-1] = fixed
                total_fixes += n
                lines_fixed += 1
            fixed_lines.append("\t".join(parts))
        else:
            fixed, n = fix_speaker_labels_in_text(line, source_labels)
            if n > 0:
                total_fixes += n
                lines_fixed += 1
            fixed_lines.append(fixed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(fixed_lines), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Fixed {lines_fixed:,} lines ({total_fixes:,} label fixes).")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
