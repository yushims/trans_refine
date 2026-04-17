"""Split long input lines into segments for independent processing.

For each input line longer than --max-chars, splits it into segments using
the same logic as run_aoai.py (sentence/word boundary splitting). Short lines
pass through unchanged.

Outputs:
  - Segmented input file: one segment per line, preserving prefix columns.
  - Line map file (.json): maps each output line back to its source line
    and segment index, enabling reassembly later.

Usage:
  python segment_input.py --input inputs/file.txt --output inputs/file_segments.txt --max-chars 30000
"""

import argparse
import json
import sys

from common import (
    install_safe_console_output,
    resolve_path,
    take_next_transcription_segment_for_llm,
    DEFAULT_MAX_INPUT_CHARS_PER_CALL,
)


def main() -> None:
    install_safe_console_output()

    parser = argparse.ArgumentParser(
        description="Split long input lines into segments.",
    )
    parser.add_argument("--input", required=True, help="Input file.")
    parser.add_argument("--output", required=True, help="Segmented output file.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_INPUT_CHARS_PER_CALL,
        help=f"Max chars per segment (default: {DEFAULT_MAX_INPUT_CHARS_PER_CALL}).",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    max_chars = args.max_chars

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    print(f"Reading input: {input_path}")
    lines = input_path.read_text(encoding="utf-8").splitlines()
    print(f"  {len(lines):,} lines, max-chars={max_chars:,}")

    segmented_lines: list[str] = []
    line_map: list[dict] = []  # {"src": source_line_index, "seg": segment_index, "total": total_segments}
    split_count = 0

    for li, line in enumerate(lines):
        parts = line.split("\t")
        text = parts[-1].strip() if parts else ""
        prefix = "\t".join(parts[:-1]) if len(parts) >= 2 else ""

        if len(text) <= max_chars or max_chars <= 0:
            # Short line — pass through.
            segmented_lines.append(line)
            line_map.append({"src": li, "seg": 0, "total": 1})
            continue

        # Split into segments.
        segments: list[str] = []
        offset = 0
        while offset < len(text):
            seg, _ = take_next_transcription_segment_for_llm(text, offset, max_chars)
            if not seg:
                seg = text[offset:offset + 1]
            segments.append(seg)
            offset += len(seg)

        for si, seg in enumerate(segments):
            if prefix:
                # Append segment info to the filename (first column).
                fn = parts[0]
                rest = "\t".join(parts[1:-1])
                tagged_fn = f"{fn}_seg{si}of{len(segments)}"
                tagged_prefix = f"{tagged_fn}\t{rest}" if rest else tagged_fn
                segmented_lines.append(f"{tagged_prefix}\t{seg}")
            else:
                segmented_lines.append(seg)
            line_map.append({"src": li, "seg": si, "total": len(segments)})

        split_count += 1
        if split_count <= 10:
            print(f"  Line {li + 1}: {len(text):,} chars -> {len(segments)} segments")

    # Write segmented output.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(segmented_lines), encoding="utf-8")

    # Write line map.
    map_path = output_path.with_suffix(".map.json")
    map_path.write_text(json.dumps(line_map, separators=(",", ":")), encoding="utf-8")

    print(f"\nDone.")
    print(f"  Input lines: {len(lines):,}")
    print(f"  Output lines: {len(segmented_lines):,}")
    print(f"  Lines split: {split_count:,}")
    print(f"  Output: {output_path}")
    print(f"  Line map: {map_path}")


if __name__ == "__main__":
    main()
