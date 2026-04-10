import argparse
import asyncio
import os
import signal

from openai import AzureOpenAI
from dotenv import load_dotenv
from common import (
    DEFAULT_AOAI_API_VERSION,
    DEFAULT_AOAI_DEPLOYMENT,
    DEFAULT_AOAI_ENDPOINT,
    add_aoai_sampling_cli_arguments,
    add_common_runtime_cli_arguments,
    add_run_pipeline_cli_arguments,
    append_payload_jsonl_record,
    assign_payload_or_emit_empty,
    build_patch_prompt,
    build_patch_response_format_schema,
    build_empty_payload,
    collect_transcriptions_from_input,
    configure_hallucination_guard,
    configure_long_span_preservation_guard,
    finalize_payloads_and_write,
    format_resolved_chain_steps,
    is_all_lowercase_cased_input,
    is_all_uppercase_cased_input,
    normalize_all_uppercase_input,
    is_input_comment_line,
    install_safe_console_output,
    load_patch_and_repair_templates,
    load_existing_output_text_lines,
    merge_segment_payloads,
    prepare_jsonl_output_path,
    postprocess_apply_safe_edits,
    print_common_runtime_settings,
    resolve_active_chain_step_keys,
    resolve_patch_and_repair_template_paths,
    run_transcriptions_with_concurrency,
    strip_emojis,
    take_next_transcription_segment_for_llm,
    write_fallback_text_output,
)
from common_aoai import get_patch_payload_with_repair, run_batch_pipeline, BATCH_DEFAULT_SIZE

PATCH_SCHEMA = build_patch_response_format_schema()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_run_pipeline_cli_arguments(parser)
    parser.add_argument("--deployment", dest="deployment", default=DEFAULT_AOAI_DEPLOYMENT)
    parser.add_argument("--endpoint", dest="endpoint", default=DEFAULT_AOAI_ENDPOINT)
    parser.add_argument("--api-version", dest="api_version", default=DEFAULT_AOAI_API_VERSION)
    add_common_runtime_cli_arguments(parser)
    add_aoai_sampling_cli_arguments(parser)
    parser.add_argument(
        "--batch",
        dest="batch_mode",
        action="store_true",
        default=False,
        help="Use the Azure OpenAI Batch API instead of real-time calls.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=BATCH_DEFAULT_SIZE,
        help=f"Number of items per Batch API partition (default: {BATCH_DEFAULT_SIZE}).",
    )
    parser.add_argument(
        "--batch-max-retries",
        dest="batch_max_retries",
        type=int,
        default=2,
        help="Maximum retry passes for failed items after all chunks complete (default: 2, 0 disables retries).",
    )
    return parser.parse_args()


async def main() -> None:
    install_safe_console_output()
    load_dotenv()
    loop = asyncio.get_running_loop()
    shutdown_requested = asyncio.Event()
    installed_signal_handlers: list[tuple[signal.Signals, object]] = []

    def _request_shutdown(signal_name: str) -> None:
        if shutdown_requested.is_set():
            return
        shutdown_requested.set()
        print(
            f"Shutdown requested ({signal_name}); "
            "stopping new items and flushing progress."
        )

    def _handle_shutdown_signal(signum: int, _frame: object) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except Exception:
            signal_name = str(signum)
        loop.call_soon_threadsafe(_request_shutdown, signal_name)

    for candidate_signal in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if candidate_signal is None:
            continue
        try:
            previous_handler = signal.getsignal(candidate_signal)
            signal.signal(candidate_signal, _handle_shutdown_signal)
            installed_signal_handlers.append((candidate_signal, previous_handler))
        except Exception:
            continue

    args = parse_args()

    input_file_value = args.input_file
    output_file_value = args.output_file
    output_as_tsv = bool(input_file_value and str(input_file_value).lower().endswith(".tsv"))

    # In batch mode, always use chunked processing for consistent behavior.
    from common import count_input_lines, iter_transcription_chunks
    use_chunked = False
    if args.batch_mode and input_file_value:
        use_chunked = True

    if use_chunked:
        # Chunked batch mode: process batch_size * concurrency lines at a time.
        pass  # Will be handled after client setup below.

    if not use_chunked:
        input_data = collect_transcriptions_from_input(input_file_value)
        if input_data is None:
            return
        transcriptions, source_filenames, source_rows = input_data

    endpoint = args.endpoint
    deployment = args.deployment
    api_version = args.api_version

    concurrency = max(1, args.concurrency)

    timeout_seconds = args.timeout
    timeout_retries = max(0, args.timeout_retries)
    empty_result_retries = max(0, args.empty_result_retries)
    max_input_chars_per_call = max(0, int(args.max_input_chars_per_call))
    long_span_min_deleted_tokens = max(1, int(args.long_span_min_deleted_tokens))
    hallucination_max_inserted_tokens = max(1, int(args.hallucination_max_inserted_tokens))
    configure_long_span_preservation_guard(
        min_deleted_tokens=long_span_min_deleted_tokens,
    )
    configure_hallucination_guard(
        max_inserted_tokens=hallucination_max_inserted_tokens,
    )

    temperature = args.temperature
    top_p = args.top_p
    retry_temperature_jitter = max(0.0, args.retry_temperature_jitter)
    retry_top_p_jitter = max(0.0, args.retry_top_p_jitter)
    chain_steps = [step for step in (args.chain_steps or []) if isinstance(step, str) and step.strip()]
    active_step_keys = resolve_active_chain_step_keys(chain_steps)
    resume_from_output = not bool(args.no_resume)
    skip_jsonl_output = bool(args.skip_jsonl_output)
    progress_write_every_arg = args.progress_write_every
    if isinstance(progress_write_every_arg, int):
        progress_write_every = max(1, progress_write_every_arg)
    else:
        progress_write_every = 100 if resume_from_output else 1
    base_progress_write_every = progress_write_every

    prompt_template_path, repair_prompt_template_path, template_error = resolve_patch_and_repair_template_paths(
        args.patch_prompt_file,
        args.repair_prompt_file,
    )
    if template_error:
        print(template_error)
        return
    if prompt_template_path is None or repair_prompt_template_path is None:
        return

    print(f"Using deployment: {deployment}")
    print(f"Using endpoint: {endpoint}")
    print(f"API version: {api_version}")
    print(f"Temperature: {temperature}")
    print(f"Top p: {top_p}")
    print(f"Retry jitter: temperature<=+{retry_temperature_jitter}, top_p±{retry_top_p_jitter}")
    if chain_steps:
        print(f"Chain step selector count: {len(chain_steps)}")
    print(f"Resolved active chain: {format_resolved_chain_steps(chain_steps)}")
    if skip_jsonl_output:
        print("JSONL progress output: disabled (--skip-jsonl-output)")
    print_common_runtime_settings(
        prompt_template_path,
        repair_prompt_template_path,
        concurrency,
        timeout_seconds,
        timeout_retries,
        empty_result_retries,
        max_input_chars_per_call,
    )

    prompt_template, repair_prompt_template = load_patch_and_repair_templates(
        prompt_template_path,
        repair_prompt_template_path,
    )

    if not os.environ.get("AZURE_OPENAI_API_KEY") and not os.environ.get("AZURE_OPENAI_AD_TOKEN"):
        print(
            "Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY (or AZURE_OPENAI_AD_TOKEN) "
            "in the current environment or .env file."
        )
        return

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    # -----------------------------------------------------------------------
    #  Chunked batch mode (large files)
    # -----------------------------------------------------------------------
    if use_chunked:
        from common import DEFAULT_AOAI_BATCH_DEPLOYMENT, sanitize_output_string
        import csv as _csv
        batch_deployment = DEFAULT_AOAI_BATCH_DEPLOYMENT
        chunk_size = args.batch_size * concurrency
        print(f"Chunked batch mode (deployment={batch_deployment}, chunk_size={chunk_size:,}, batch_size={args.batch_size:,}, concurrency={concurrency})")

        def _sanitize_tsv_cell(value: str) -> str:
            """Strip only characters that break TSV structure, preserving content."""
            if not isinstance(value, str):
                return ""
            return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")

        def _build_output_line(
            corrected_text: str,
            source_row: list[str] | None,
            filename: str | None,
        ) -> str:
            """Build a single output line, preserving all source columns for multi-column input.

            Only the last column (corrected_text) is fully sanitized.
            Prefix columns are kept as-is except for tab/newline characters
            that would break TSV structure.
            """
            ct = sanitize_output_string(corrected_text)
            if isinstance(source_row, list) and source_row:
                # Preserve prefix columns exactly, only sanitize TSV-breaking chars.
                rebuilt = [_sanitize_tsv_cell(v) for v in source_row[:-1]]
                rebuilt.append(ct)
                return "\t".join(rebuilt)
            if isinstance(filename, str) and filename:
                return f"{_sanitize_tsv_cell(filename)}\t{ct}"
            return ct

        # Load resume lines if output file exists.
        resume_lines: list[str] | None = None
        if resume_from_output and output_file_value:
            from common import resolve_path as _resolve_path
            _out_path = _resolve_path(output_file_value)
            _out_text_path = _out_path.with_suffix(".tsv") if output_as_tsv else _out_path.with_suffix(".txt")
            if not _out_text_path.exists():
                # Try the other extension (multi-column .txt input writes .tsv).
                _alt_path = _out_path.with_suffix(".txt") if output_as_tsv else _out_path.with_suffix(".tsv")
                if _alt_path.exists():
                    _out_text_path = _alt_path
            if _out_text_path.exists():
                try:
                    _raw = _out_text_path.read_text(encoding="utf-8")
                    _is_tsv_resume = _out_text_path.suffix.lower() == ".tsv"
                    if _is_tsv_resume:
                        # For TSV, extract only the last column (transcription).
                        import csv as _csv
                        resume_lines = []
                        for row in _csv.reader(_raw.splitlines(), delimiter="\t"):
                            resume_lines.append(sanitize_output_string(row[-1]) if row else "")
                    else:
                        resume_lines = [sanitize_output_string(line) for line in _raw.splitlines()]
                    non_empty = sum(1 for l in resume_lines if l.strip())
                    print(f"Loaded resume output: {non_empty:,} non-empty lines.")
                except Exception as e:
                    print(f"Could not load resume file: {e}")

        # Open output file in append mode for incremental writes.
        from common import resolve_path as _resolve_path
        out_path = _resolve_path(output_file_value) if output_file_value else None
        out_text_path = None
        if out_path:
            out_text_path = out_path.with_suffix(".tsv") if output_as_tsv else out_path.with_suffix(".txt")
            out_text_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine how many chunks were already fully written in a previous run.
        # On resume, we rewrite the entire output file from scratch to avoid
        # duplicates from partial appends.
        chunks_completed_in_resume = 0
        total_written = 0
        first_write = True
        # Track failed items across all chunks for end-of-run retry.
        # Each entry: (global_idx, preprocessed_text, original_text, filename, skip_casing_flag)
        all_failed_items: list[tuple[int, str, str, str | None, bool]] = []

        for global_offset, chunk_transcriptions, chunk_filenames, chunk_source_rows in iter_transcription_chunks(
            input_file_value, chunk_size
        ):
            chunk_len = len(chunk_transcriptions)
            chunk_end = global_offset + chunk_len
            print(f"\n--- Chunk [{global_offset + 1:,}..{chunk_end:,}] ({chunk_len:,} items) ---")

            # Check resume: skip entire chunk if all lines are already resolved.
            chunk_output_lines: list[str] | None = None
            if resume_lines is not None:
                chunk_resume = resume_lines[global_offset:chunk_end] if global_offset < len(resume_lines) else []
                all_resolved = (
                    len(chunk_resume) == chunk_len
                    and all(isinstance(l, str) and l.strip() for l in chunk_resume)
                )
                if all_resolved:
                    print(f"  Chunk already completed in resume output; skipping.")
                    # Still write these lines to the output to maintain correct order.
                    chunk_output_lines = list(chunk_resume)
                    chunks_completed_in_resume += 1
                    # Write to output.
                    if out_text_path:
                        out_lines = [
                            _build_output_line(
                                text,
                                chunk_source_rows[i] if i < len(chunk_source_rows) else None,
                                chunk_filenames[i] if i < len(chunk_filenames) else None,
                            )
                            for i, text in enumerate(chunk_output_lines)
                        ]
                        content = "\n".join(out_lines) + "\n"
                        if first_write:
                            out_text_path.write_text(content, encoding="utf-8")
                            first_write = False
                        else:
                            with out_text_path.open("a", encoding="utf-8") as fh:
                                fh.write(content)
                    total_written += chunk_len
                    continue

            # Preprocess chunk.
            skip_flags: list[bool] = []
            batch_texts: list[str] = []
            pre_resolved: set[int] = set()
            for i, t in enumerate(chunk_transcriptions):
                stripped_t, _ = strip_emojis(t)
                prompt_t, _ = normalize_all_uppercase_input(stripped_t)
                batch_texts.append(prompt_t)
                skip_flags.append(
                    is_all_uppercase_cased_input(t) or is_all_lowercase_cased_input(t)
                )
                if is_input_comment_line(t):
                    pre_resolved.add(i)
                elif resume_lines is not None:
                    ri = global_offset + i
                    if ri < len(resume_lines) and isinstance(resume_lines[ri], str) and resume_lines[ri].strip():
                        pre_resolved.add(i)

            # Run batch pipeline for this chunk.
            chunk_payloads = await run_batch_pipeline(
                client=client,
                deployment=batch_deployment,
                transcriptions=batch_texts,
                prompt_template=prompt_template,
                chain_steps=chain_steps,
                locale=args.locale,
                schema=PATCH_SCHEMA,
                temperature=temperature,
                top_p=top_p,
                batch_size=args.batch_size,
                build_patch_prompt_fn=build_patch_prompt,
                skip_first_token_casing_preservation_flags=skip_flags,
                active_step_keys=active_step_keys,
                max_input_chars_per_call=max_input_chars_per_call,
                concurrency=concurrency,
                pre_resolved_indices=pre_resolved,
            )
            assert len(chunk_payloads) == chunk_len, (
                f"Batch pipeline alignment error: expected {chunk_len} payloads, got {len(chunk_payloads)}"
            )
            chunk_output_lines: list[str] = []
            for i, payload in enumerate(chunk_payloads):
                t = chunk_transcriptions[i]
                if payload is None:
                    if is_input_comment_line(t):
                        chunk_output_lines.append(t)
                    elif resume_lines is not None:
                        ri = global_offset + i
                        chunk_output_lines.append(resume_lines[ri] if ri < len(resume_lines) else "")
                    else:
                        chunk_output_lines.append("")
                    # Track as failed for end-of-run retry (skip comments and pre-resolved).
                    if i not in pre_resolved and not is_input_comment_line(t):
                        fn = chunk_filenames[i] if chunk_filenames and i < len(chunk_filenames) else None
                        all_failed_items.append((global_offset + i, batch_texts[i], t, fn, skip_flags[i]))
                else:
                    ct = payload.get("corrected_text")
                    if isinstance(ct, str) and ct.strip():
                        # Post-processing: character-level diff merge to reject content changes.
                        if args.apply_safe_edits:
                            original = chunk_transcriptions[i]
                            if isinstance(original, str) and original.strip():
                                merged, _ = postprocess_apply_safe_edits(original, ct)
                                ct = merged
                        chunk_output_lines.append(ct)
                    else:
                        # Payload present but corrected_text empty — treat as failed.
                        chunk_output_lines.append("")
                        if i not in pre_resolved and not is_input_comment_line(t):
                            fn = chunk_filenames[i] if chunk_filenames and i < len(chunk_filenames) else None
                            all_failed_items.append((global_offset + i, batch_texts[i], t, fn, skip_flags[i]))

            # Write chunk results to output file.
            assert len(chunk_output_lines) == chunk_len, (
                f"Output alignment error: expected {chunk_len} lines, got {len(chunk_output_lines)}"
            )
            if out_text_path:
                out_lines = [
                    _build_output_line(
                        text,
                        chunk_source_rows[i] if i < len(chunk_source_rows) else None,
                        chunk_filenames[i] if i < len(chunk_filenames) else None,
                    )
                    for i, text in enumerate(chunk_output_lines)
                ]
                content = "\n".join(out_lines) + "\n"

                if first_write:
                    out_text_path.write_text(content, encoding="utf-8")
                    first_write = False
                else:
                    with out_text_path.open("a", encoding="utf-8") as fh:
                        fh.write(content)

            total_written += chunk_len
            print(f"  Chunk complete. Total written: {total_written:,}")

        print(f"\nAll chunks done. Total: {total_written:,} items.")
        if chunks_completed_in_resume > 0:
            print(f"  ({chunks_completed_in_resume} chunks were already completed from resume.)")

        # ---------------------------------------------------------------------
        #  End-of-run retry for items that failed during chunk processing.
        # ---------------------------------------------------------------------
        batch_max_retries = getattr(args, "batch_max_retries", 0)
        if all_failed_items and batch_max_retries > 0 and out_text_path and out_text_path.exists():
            import random
            retry_remaining = list(all_failed_items)
            for retry_pass in range(1, batch_max_retries + 1):
                if not retry_remaining:
                    break
                print(f"\n--- Retry pass {retry_pass}: {len(retry_remaining):,} failed items ---")

                # Temperature jitter for retries.
                pass_temperature = temperature
                if retry_temperature_jitter > 0:
                    pass_temperature = max(
                        0.0,
                        min(1.0, temperature + random.uniform(0.0, retry_temperature_jitter)),
                    )
                    print(f"  Temperature jitter applied: {pass_temperature:.3f}")

                # Halve max_input_chars on each retry pass: pass1 = half, pass2 = quarter, ...
                retry_max_chars = max(1, max_input_chars_per_call // (2 ** retry_pass)) if max_input_chars_per_call > 0 else 0

                retry_texts = [item[1] for item in retry_remaining]
                retry_skip_flags = [item[4] for item in retry_remaining]

                retry_payloads = await run_batch_pipeline(
                    client=client,
                    deployment=batch_deployment,
                    transcriptions=retry_texts,
                    prompt_template=prompt_template,
                    chain_steps=chain_steps,
                    locale=args.locale,
                    schema=PATCH_SCHEMA,
                    temperature=pass_temperature,
                    top_p=top_p,
                    batch_size=args.batch_size,
                    build_patch_prompt_fn=build_patch_prompt,
                    skip_first_token_casing_preservation_flags=retry_skip_flags,
                    active_step_keys=active_step_keys,
                    max_input_chars_per_call=retry_max_chars,
                    concurrency=concurrency,
                )
                assert len(retry_payloads) == len(retry_remaining), (
                    f"Retry alignment error: expected {len(retry_remaining)} payloads, got {len(retry_payloads)}"
                )

                # Collect successful retries and patch output file.
                patched_lines: dict[int, str] = {}  # global_idx -> corrected text
                still_failed: list[tuple[int, str, str, str | None, bool]] = []
                for item, payload in zip(retry_remaining, retry_payloads, strict=True):
                    global_idx = item[0]
                    if payload is not None:
                        ct = payload.get("corrected_text", "")
                        if isinstance(ct, str) and ct.strip():
                            if args.apply_safe_edits:
                                original = item[2]
                                if isinstance(original, str) and original.strip():
                                    merged, _ = postprocess_apply_safe_edits(original, ct)
                                    ct = merged
                            patched_lines[global_idx] = ct
                        else:
                            still_failed.append(item)
                    else:
                        still_failed.append(item)

                if patched_lines:
                    raw = out_text_path.read_text(encoding="utf-8")
                    output_lines = raw.split("\n")
                    if output_lines and output_lines[-1] == "":
                        output_lines.pop()
                    for global_idx, ct in patched_lines.items():
                        if global_idx < len(output_lines):
                            line = output_lines[global_idx]
                            if "\t" in line:
                                # Replace only the last column (transcription),
                                # preserving all leading columns.
                                prefix, _old_text = line.rsplit("\t", 1)
                                output_lines[global_idx] = f"{prefix}\t{sanitize_output_string(ct)}"
                            else:
                                output_lines[global_idx] = sanitize_output_string(ct)
                    out_text_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
                    print(f"  Patched {len(patched_lines):,} lines in output file.")

                retry_remaining = still_failed

            if retry_remaining:
                print(f"\n{len(retry_remaining):,} items still failed after {batch_max_retries} retry pass(es).")
        elif all_failed_items:
            print(f"\n{len(all_failed_items):,} items failed (retries disabled).")

        return

    # -----------------------------------------------------------------------
    #  Real-time mode (default)
    # -----------------------------------------------------------------------
    payloads: list[dict | None] = [None] * len(transcriptions)
    text_output_lines: list[str] = [""] * len(transcriptions)
    if resume_from_output:
        loaded_lines = load_existing_output_text_lines(
            output_file_value,
            len(transcriptions),
            output_as_tsv,
        )
        if loaded_lines is not None:
            text_output_lines = loaded_lines
    progress_write_lock = asyncio.Lock()
    pending_progress_writes = 0
    resume_progress_write_every = base_progress_write_every
    output_jsonl_path = (
        None
        if skip_jsonl_output
        else prepare_jsonl_output_path(output_file_value, resume_mode=resume_from_output)
    )
    streamed_jsonl_slots: set[int] = set()
    pending_jsonl_slots: list[int] = []

    async def maybe_write_progress_snapshot(
        *,
        slot: int | None = None,
        used_existing_output_line: bool = False,
        force: bool = False,
    ) -> None:
        nonlocal pending_progress_writes, resume_progress_write_every
        if not output_file_value:
            return
        async with progress_write_lock:
            if (
                isinstance(slot, int)
                and slot >= 0
                and slot not in streamed_jsonl_slots
                and slot not in pending_jsonl_slots
            ):
                pending_jsonl_slots.append(slot)

            current_progress_write_every = base_progress_write_every
            if resume_from_output and used_existing_output_line:
                current_progress_write_every = resume_progress_write_every

            if not force:
                pending_progress_writes += 1
                if pending_progress_writes < current_progress_write_every:
                    if resume_from_output and used_existing_output_line:
                        resume_progress_write_every = max(
                            base_progress_write_every,
                            resume_progress_write_every * 2,
                        )
                    return
                pending_progress_writes = 0

            remaining_jsonl_slots: list[int] = []
            for pending_slot in pending_jsonl_slots:
                try:
                    source_filename = source_filenames[pending_slot] if pending_slot < len(source_filenames) else None
                    if append_payload_jsonl_record(
                        output_jsonl_path,
                        payloads[pending_slot],
                        source_filename,
                        active_step_keys,
                    ):
                        streamed_jsonl_slots.add(pending_slot)
                    else:
                        remaining_jsonl_slots.append(pending_slot)
                except Exception as error:
                    print(f"Failed to append JSONL progress record: {error}")
                    remaining_jsonl_slots.append(pending_slot)
            pending_jsonl_slots.clear()
            pending_jsonl_slots.extend(remaining_jsonl_slots)

            if not any(isinstance(line, str) and line.strip() for line in text_output_lines):
                if resume_from_output and used_existing_output_line:
                    resume_progress_write_every = max(
                        base_progress_write_every,
                        resume_progress_write_every * 2,
                    )
                return

            try:
                write_fallback_text_output(
                    output_file_value,
                    text_output_lines,
                    source_filenames,
                    source_rows,
                    output_as_tsv=output_as_tsv,
                )
            except Exception as error:
                print(f"Failed to write progress snapshot: {error}")
            finally:
                if resume_from_output and used_existing_output_line:
                    resume_progress_write_every = max(
                        base_progress_write_every,
                        resume_progress_write_every * 2,
                    )

    async def reset_resume_progress_write_interval() -> None:
        nonlocal resume_progress_write_every
        if not resume_from_output:
            return
        async with progress_write_lock:
            resume_progress_write_every = base_progress_write_every

    async def process_item(index: int, transcription: str, total: int) -> None:
        if shutdown_requested.is_set():
            return
        slot = index - 1
        processing_id = f"{index}/{total}"
        if is_input_comment_line(transcription):
            text_output_lines[slot] = transcription
            await maybe_write_progress_snapshot(slot=slot)
            return

        if not transcription.strip():
            payloads[slot] = build_empty_payload()
            payloads[slot]["source_text"] = transcription
            print(
                f"Input transcription {index}/{total} is empty; "
                "emitting empty payload."
            )
            await maybe_write_progress_snapshot(slot=slot)
            return

        if resume_from_output:
            existing_line = text_output_lines[slot] if slot < len(text_output_lines) else ""
            if isinstance(existing_line, str) and existing_line.strip():
                payloads[slot] = build_empty_payload()
                payloads[slot]["source_text"] = transcription
                payloads[slot]["corrected_text"] = existing_line
                source_filename = source_filenames[slot]
                if isinstance(source_filename, str) and source_filename:
                    payloads[slot]["source_filename"] = source_filename
                print(
                    f"[{processing_id}] Resume enabled: existing output is non-empty; "
                    "skipping transcription."
                )
                await maybe_write_progress_snapshot(slot=slot, used_existing_output_line=True)
                return
            await reset_resume_progress_write_interval()

        stripped_transcription, _ = strip_emojis(transcription)
        prompt_transcription, case_normalized = normalize_all_uppercase_input(stripped_transcription)
        source_was_all_uppercase = is_all_uppercase_cased_input(transcription)
        source_was_all_lowercase = is_all_lowercase_cased_input(transcription)
        skip_first_token_casing_preservation = source_was_all_uppercase or source_was_all_lowercase
        # if case_normalized:
        #     print(f"[{processing_id}] Normalized all-uppercase input to display casing before prompt.")

        try:
            source_length = len(prompt_transcription)
            if max_input_chars_per_call > 0 and source_length > max_input_chars_per_call:
                print(
                    f"[{processing_id}] Input length {source_length} exceeded "
                    f"max-input-chars-per-call={max_input_chars_per_call}; "
                    "processing with adaptive segmentation."
                )

            segment_payloads: list[dict] = []
            segment_sources: list[str] = []
            payload: dict | None = None
            segment_index = 1
            start_offset = 0

            while start_offset < source_length:
                remaining_length = source_length - start_offset
                if max_input_chars_per_call > 0:
                    current_limit = min(max_input_chars_per_call, remaining_length)
                else:
                    current_limit = remaining_length

                segment_succeeded = False
                while True:
                    segment_transcription, _ = take_next_transcription_segment_for_llm(
                        prompt_transcription,
                        start_offset,
                        current_limit,
                    )
                    if not segment_transcription:
                        segment_transcription = prompt_transcription[start_offset:start_offset + 1]

                    segment_processing_id = f"{processing_id} seg {segment_index}"
                    if source_length > len(segment_transcription):
                        print(f"[{segment_processing_id}] Processing segment ({len(segment_transcription)} chars).")

                    # Compute context snippets for boundary decisions.
                    _CTX_LEN = 50
                    seg_prev_ctx = segment_sources[-1][-_CTX_LEN:] if segment_sources else None
                    seg_next_end = start_offset + len(segment_transcription)
                    seg_next_ctx = prompt_transcription[seg_next_end:seg_next_end + _CTX_LEN] if seg_next_end < source_length else None

                    prompt = build_patch_prompt(
                        prompt_template,
                        segment_transcription,
                        chain_steps,
                        locale=args.locale,
                        prev_context=seg_prev_ctx,
                        next_context=seg_next_ctx,
                    )

                    failure_reasons: list[str] = []

                    def _on_final_failure(reason: str) -> None:
                        failure_reasons.append(reason)

                    segment_payload = await get_patch_payload_with_repair(
                        client=client,
                        deployment=deployment,
                        prompt=prompt,
                        transcription=segment_transcription,
                        processing_id=segment_processing_id,
                        repair_prompt_template=repair_prompt_template,
                        patch_schema=PATCH_SCHEMA,
                        timeout_seconds=timeout_seconds,
                        timeout_retries=timeout_retries,
                        empty_result_retries=empty_result_retries,
                        temperature=temperature,
                        top_p=top_p,
                        retry_temperature_jitter=retry_temperature_jitter,
                        retry_top_p_jitter=retry_top_p_jitter,
                        skip_first_token_casing_preservation=skip_first_token_casing_preservation,
                        active_step_keys=active_step_keys,
                        on_final_failure=_on_final_failure,
                    )

                    if segment_payload is not None:
                        segment_payloads.append(segment_payload)
                        segment_sources.append(segment_transcription)
                        start_offset += len(segment_transcription)
                        segment_index += 1
                        segment_succeeded = True
                        break

                    should_shorten = (
                        len(segment_transcription) > 1
                        and current_limit > 1
                    )
                    if should_shorten:
                        next_limit = max(1, min(current_limit - 1, len(segment_transcription) - 1, current_limit // 2))
                        if next_limit < current_limit:
                            print(
                                f"[{segment_processing_id}] Segment failed after retries; "
                                f"shortening segment and retrying (limit {current_limit} -> {next_limit})."
                            )
                            current_limit = next_limit
                            continue

                    print(
                        f"[{segment_processing_id}] Segment failed after retries; "
                        "skipping remaining segments for this transcription."
                    )
                    payload = None
                    break

                if not segment_succeeded:
                    break

            if len(segment_sources) > 1:
                if len(segment_payloads) != len(segment_sources):
                    payload = None
                else:
                    payload = merge_segment_payloads(segment_payloads, segment_sources)
                if payload is None:
                    print(
                        f"[{processing_id}] Failed to merge segmented payloads; "
                        "emitting empty payload."
                    )
            elif segment_payloads:
                payload = segment_payloads[0]
            else:
                payload = None

            assign_payload_or_emit_empty(payload, payloads, slot, index, total)
            resolved_payload = payloads[slot]
            if isinstance(resolved_payload, dict):
                source_filename = source_filenames[slot]
                if isinstance(source_filename, str) and source_filename:
                    resolved_payload["source_filename"] = source_filename
                resolved_payload["source_text"] = transcription
                corrected_text = resolved_payload.get("corrected_text")
                # Post-processing: character-level diff merge to reject content changes.
                if args.apply_safe_edits and isinstance(corrected_text, str) and corrected_text.strip() and transcription.strip():
                    merged, _ = postprocess_apply_safe_edits(transcription, corrected_text)
                    if merged != corrected_text:
                        resolved_payload["corrected_text"] = merged
                        corrected_text = merged
                text_output_lines[slot] = corrected_text if isinstance(corrected_text, str) else ""
            await maybe_write_progress_snapshot(slot=slot)
        except asyncio.CancelledError:
            payloads[slot] = build_empty_payload()
            payloads[slot]["source_text"] = transcription
            text_output_lines[slot] = ""
            print(
                f"Cancelled while processing transcription {index}/{total}; "
                "emitting empty payload."
            )
            await maybe_write_progress_snapshot(slot=slot)
            return
        except Exception as error:
            payloads[slot] = build_empty_payload()
            payloads[slot]["source_text"] = transcription
            text_output_lines[slot] = ""
            print(
                f"Unexpected error on transcription {index}/{total}: {error}; "
                "emitting empty payload."
            )
            await maybe_write_progress_snapshot(slot=slot)
            return

    try:
        await run_transcriptions_with_concurrency(transcriptions, concurrency, process_item)
    finally:
        await maybe_write_progress_snapshot(force=True)
        for candidate_signal, previous_handler in installed_signal_handlers:
            try:
                signal.signal(candidate_signal, previous_handler)
            except Exception:
                continue

    if shutdown_requested.is_set():
        print("Shutdown handler completed final progress flush.")

    if not finalize_payloads_and_write(
        payloads,
        output_file_value,
        text_output_lines,
        source_filenames,
        source_rows,
        output_as_tsv,
        active_step_keys,
    ):
        return


if __name__ == "__main__":
    asyncio.run(main())
