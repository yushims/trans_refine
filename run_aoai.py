import argparse
import asyncio
import os
import signal

from openai import AzureOpenAI
from dotenv import load_dotenv
from common import (
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
from common_aoai import (
    get_patch_payload_with_repair,
    run_batch_pipeline,
    set_shutdown_event,
    BATCH_DEFAULT_SIZE,
    is_partial_segments_marker,
    decode_partial_segments,
    encode_partial_segments,
)

PATCH_SCHEMA = build_patch_response_format_schema()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_run_pipeline_cli_arguments(parser)
    parser.add_argument("--deployment", dest="deployment", default=None)
    parser.add_argument("--endpoint", dest="endpoint", default=None)
    parser.add_argument("--api-version", dest="api_version", default=None)
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
        help=f"Max segments (API requests) per Batch API partition upload (default: {BATCH_DEFAULT_SIZE}). One input line may produce multiple segments when segmentation is active.",
    )
    parser.add_argument(
        "--batch-max-retries",
        dest="batch_max_retries",
        type=int,
        default=2,
        help="Maximum retry passes for failed items after all chunks complete (default: 2, 0 disables retries).",
    )
    parser.add_argument(
        "--batch-deployment",
        dest="batch_deployment",
        default=None,
        help="Deployment name for batch API calls (from BATCH_DEPLOYMENT env var).",
    )
    parser.add_argument(
        "--batch-endpoint",
        dest="batch_endpoint",
        default=None,
        help="Azure OpenAI endpoint for batch API calls (defaults to BATCH_ENDPOINT env or --endpoint).",
    )
    parser.add_argument(
        "--batch-api-version",
        dest="batch_api_version",
        default=None,
        help="API version for batch API calls (defaults to BATCH_API_VERSION env or --api-version).",
    )
    parser.add_argument(
        "--batch-use-realtime",
        dest="batch_use_realtime",
        action="store_true",
        default=False,
        help="Use real-time chat completion calls instead of the Batch API (faster for debugging).",
    )
    parser.add_argument(
        "--batch-debug-dir",
        dest="batch_debug_dir",
        default=None,
        help="Directory to save raw LLM batch results as JSONL files for debugging (one file per chunk).",
    )
    return parser.parse_args()


async def main() -> None:
    install_safe_console_output()
    load_dotenv()
    loop = asyncio.get_running_loop()
    shutdown_requested = asyncio.Event()
    set_shutdown_event(shutdown_requested)  # Register with common_aoai for batch polling.
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

    # Resolve defaults from env vars (loaded by load_dotenv above).
    if args.endpoint is None:
        args.endpoint = os.environ.get("ENDPOINT")
    if args.api_version is None:
        args.api_version = os.environ.get("API_VERSION")
    if args.deployment is None:
        args.deployment = os.environ.get("DEPLOYMENT")
    if args.batch_deployment is None:
        args.batch_deployment = os.environ.get("BATCH_DEPLOYMENT")
    # Batch-specific endpoint/api-version/api-key: fall back to real-time values.
    if args.batch_endpoint is None:
        args.batch_endpoint = os.environ.get("BATCH_ENDPOINT", args.endpoint)
    if args.batch_api_version is None:
        args.batch_api_version = os.environ.get("BATCH_API_VERSION", args.api_version)

    input_file_value = args.input_file
    output_file_value = args.output_file
    output_as_tsv = bool(input_file_value and str(input_file_value).lower().endswith(".tsv"))

    # Use chunked/streaming processing when an input file is provided.
    # Batch mode always uses chunked; real-time mode also uses chunked when
    # there is an input file so that large files don't have to be fully loaded
    # before LLM calls begin.
    from common import count_input_lines, iter_transcription_chunks
    use_chunked = bool(input_file_value)

    if use_chunked:
        # Chunked mode: process lines in streaming chunks.
        pass  # Will be handled after client setup below.

    if not use_chunked:
        # No input file (stdin) — must pre-load everything.
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
    retry_content_filtered = bool(getattr(args, "retry_content_filtered", False))
    output_jsonl = bool(args.output_jsonl)
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
    if output_jsonl:
        print("JSONL structured output: enabled (--output-jsonl)")
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

    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("Missing Azure OpenAI credentials. Set API_KEY in the current environment or .env file.")
        return

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
    )

    # Create a separate client for batch API if endpoint/api-version/api-key differ.
    batch_api_key = os.environ.get("BATCH_API_KEY")
    if (
        args.batch_mode
        and (args.batch_endpoint != endpoint or args.batch_api_version != api_version or batch_api_key)
    ):
        batch_client_kwargs: dict = {
            "azure_endpoint": args.batch_endpoint,
            "api_version": args.batch_api_version,
        }
        if batch_api_key:
            batch_client_kwargs["api_key"] = batch_api_key
        batch_client = AzureOpenAI(**batch_client_kwargs)
        print(f"Using separate batch endpoint: {args.batch_endpoint}")
    else:
        batch_client = client

    # Metadata attached to each batch job for identification.
    _input_basename = os.path.basename(args.input_file) if args.input_file else ""
    batch_metadata = {
        "user": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
        "input_file": _input_basename,
    }

    # -----------------------------------------------------------------------
    #  Chunked batch mode (large files)
    # -----------------------------------------------------------------------
    if use_chunked and args.batch_mode:
        from common import sanitize_output_string
        import csv as _csv
        # Use regular deployment for realtime calls; batch SKU doesn't support chat completions.
        batch_deployment = args.deployment if args.batch_use_realtime else args.batch_deployment
        # batch_size = number of segments (API requests) per partition upload.
        # chunk_size = target estimated segments per iteration before submitting.
        # iter_transcription_chunks accumulates lines until their estimated
        # segment count reaches chunk_size, so we load exactly the right
        # number of lines to fill batch_size * concurrency segments.
        chunk_size = args.batch_size * concurrency
        print(
            f"Chunked batch mode (deployment={batch_deployment}, "
            f"chunk_size={chunk_size:,} segments, "
            f"batch_size={args.batch_size:,} segments/partition, "
            f"concurrency={concurrency})"
        )

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
            # All text is sanitized to prevent line splitting in the output file.
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
                    # Always extract last column for resume — even .txt files
                    # may contain tab-separated columns.
                    # Use .split('\n') instead of .splitlines() because
                    # splitlines() also splits on \x1c, \x1d, \x1e, \x85,
                    # \u2028, \u2029 which can appear in ASR transcription
                    # text and would corrupt the line count.
                    import csv as _csv
                    resume_lines = []
                    _raw_lines = _raw.split("\n")
                    if _raw_lines and _raw_lines[-1] == "":
                        _raw_lines.pop()
                    for row in _csv.reader(_raw_lines, delimiter="\t"):
                        cell = row[-1] if row else ""
                        # Sanitize all resume lines unconditionally to strip embedded
                        # control characters that would corrupt line alignment.
                        resume_lines.append(sanitize_output_string(cell))
                    non_empty = sum(1 for l in resume_lines if l.strip())
                    print(f"Loaded resume output: {non_empty:,} non-empty lines.")
                    # Validate resume line count against input to detect corrupt files
                    # (e.g. from the newline-splitting bug).
                    input_line_count = count_input_lines(input_file_value)
                    if input_line_count > 0 and len(resume_lines) > input_line_count:
                        print(
                            f"  WARNING: Resume file has {len(resume_lines):,} lines but input has {input_line_count:,}. "
                            "Resume file appears corrupted (line splitting). Discarding resume data."
                        )
                        resume_lines = None
                except Exception as e:
                    print(f"Could not load resume file: {e}")

        # Open output file in append mode for incremental writes.
        from common import resolve_path as _resolve_path
        out_path = _resolve_path(output_file_value) if output_file_value else None
        out_text_path = None
        out_jsonl_path = None
        if out_path:
            out_text_path = out_path.with_suffix(".tsv") if output_as_tsv else out_path.with_suffix(".txt")
            out_text_path.parent.mkdir(parents=True, exist_ok=True)
            if output_jsonl:
                out_jsonl_path = out_path.with_suffix(".jsonl")
                out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine how many chunks were already fully written in a previous run.
        # On resume, we rewrite the entire output file from scratch to avoid
        # duplicates from partial appends.
        chunks_completed_in_resume = 0
        total_written = 0
        first_write = True
        first_jsonl_write = True
        # Track failed items across all chunks for end-of-run retry.
        # Each entry: (global_idx, preprocessed_text, original_text, filename, skip_casing_flag)
        all_failed_items: list[tuple[int, str, str, str | None, bool]] = []
        # Carry-over: unresolved items from resumed chunks, to be
        # injected into the next chunk that actually runs.
        # Each entry: (global_idx, source_text, filename, source_row, resume_line)
        carryover_items: list[tuple[int, str, str | None, list[str] | None, str]] = []

        for global_offset, chunk_transcriptions, chunk_filenames, chunk_source_rows in iter_transcription_chunks(
            input_file_value, chunk_size, max_input_chars_per_call=max_input_chars_per_call
        ):
            chunk_len = len(chunk_transcriptions)
            chunk_end = global_offset + chunk_len
            print(f"\n--- Chunk [{global_offset + 1:,}..{chunk_end:,}] ({chunk_len:,} items) ---")

            # Check resume: skip chunk if fully resolved, otherwise carry over
            # unresolved items to be processed with the next chunk's batch.
            chunk_output_lines: list[str] | None = None
            if resume_lines is not None:
                chunk_resume = resume_lines[global_offset:chunk_end] if global_offset < len(resume_lines) else []
                if len(chunk_resume) == chunk_len:
                    # Count unresolved lines (empty or partial markers).
                    unresolved_in_chunk: list[int] = []
                    for ci, cl in enumerate(chunk_resume):
                        if not (isinstance(cl, str) and cl.strip()) or is_partial_segments_marker(cl):
                            unresolved_in_chunk.append(ci)

                    if not unresolved_in_chunk:
                        # Fully resolved — skip entirely.
                        print(f"  Chunk already completed in resume output; skipping.")
                        chunk_output_lines = list(chunk_resume)
                        chunks_completed_in_resume += 1
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

                    # Carry over unresolved items to the next chunk's batch.
                    # Write resolved lines now; unresolved lines get empty placeholders
                    # that will be patched when carry-over results come back.
                    for ci in unresolved_in_chunk:
                        fn = chunk_filenames[ci] if chunk_filenames and ci < len(chunk_filenames) else None
                        sr = chunk_source_rows[ci] if chunk_source_rows and ci < len(chunk_source_rows) else None
                        rl = chunk_resume[ci] if ci < len(chunk_resume) else ""
                        carryover_items.append((global_offset + ci, chunk_transcriptions[ci], fn, sr, rl))

                    print(
                        f"  Chunk has {chunk_len - len(unresolved_in_chunk)}/{chunk_len} resolved; "
                        f"carrying over {len(unresolved_in_chunk)} item(s) to next batch."
                    )
                    # Write the chunk with resolved lines; unresolved get their
                    # current value (empty or partial marker) as placeholder.
                    chunk_output_lines = list(chunk_resume)
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

            # Process accumulated carry-over items. If there are more than
            # chunk_size, process them in standalone batches first to avoid
            # overloading the current chunk's batch.
            chunk_carryover: list[tuple[int, str, str | None, list[str] | None, str]] = []
            if carryover_items:
                while len(carryover_items) > chunk_size:
                    # Flush a standalone batch of carry-over items.
                    co_batch = carryover_items[:chunk_size]
                    carryover_items = carryover_items[chunk_size:]
                    print(f"  Processing {len(co_batch)} carry-over item(s) in standalone batch ({len(carryover_items)} remaining)...")
                    co_texts: list[str] = []
                    co_skip: list[bool] = []
                    for _, co_text, _, _, _ in co_batch:
                        stripped_co, _ = strip_emojis(co_text)
                        prompt_co, _ = normalize_all_uppercase_input(stripped_co)
                        co_texts.append(prompt_co)
                        co_skip.append(is_all_uppercase_cased_input(co_text) or is_all_lowercase_cased_input(co_text))
                    co_payloads = await run_batch_pipeline(
                        client=batch_client,
                        deployment=batch_deployment,
                        transcriptions=co_texts,
                        prompt_template=prompt_template,
                        chain_steps=chain_steps,
                        locale=args.locale,
                        schema=PATCH_SCHEMA,
                        temperature=temperature,
                        top_p=top_p,
                        batch_size=args.batch_size,
                        build_patch_prompt_fn=build_patch_prompt,
                        skip_first_token_casing_preservation_flags=co_skip,
                        active_step_keys=active_step_keys,
                        max_input_chars_per_call=max_input_chars_per_call,
                        concurrency=concurrency,
                        use_realtime=args.batch_use_realtime,
                        batch_metadata=batch_metadata,
                    )
                    # Patch results at original lines immediately.
                    co_patches: dict[int, str] = {}
                    for ci, ((co_gidx, co_text, co_fn, co_sr, co_rl), co_payload) in enumerate(zip(co_batch, co_payloads)):
                        if co_payload is not None:
                            co_ct = co_payload.get("corrected_text", "")
                            if isinstance(co_ct, str) and co_ct.strip() and not is_partial_segments_marker(co_ct):
                                if args.apply_safe_edits and isinstance(co_text, str) and co_text.strip():
                                    merged, _ = postprocess_apply_safe_edits(co_text, co_ct)
                                    co_ct = merged
                                co_patches[co_gidx] = co_ct
                            elif isinstance(co_ct, str) and is_partial_segments_marker(co_ct):
                                co_patches[co_gidx] = co_ct
                                all_failed_items.append((co_gidx, co_texts[ci], co_text, co_fn, co_skip[ci]))
                            else:
                                all_failed_items.append((co_gidx, co_texts[ci], co_text, co_fn, co_skip[ci]))
                        else:
                            all_failed_items.append((co_gidx, co_texts[ci], co_text, co_fn, co_skip[ci]))
                    if co_patches and out_text_path and out_text_path.exists():
                        from common import sanitize_output_string as _co_san2
                        co_fl = out_text_path.read_text(encoding="utf-8").split("\n")
                        if co_fl and co_fl[-1] == "":
                            co_fl.pop()
                        for gidx, ct in co_patches.items():
                            if gidx < len(co_fl):
                                s_ct = ct if is_partial_segments_marker(ct) else _co_san2(ct)
                                ln = co_fl[gidx]
                                if "\t" in ln:
                                    pfx, _ = ln.rsplit("\t", 1)
                                    co_fl[gidx] = f"{pfx}\t{s_ct}"
                                else:
                                    co_fl[gidx] = s_ct
                        out_text_path.write_text("\n".join(co_fl) + "\n", encoding="utf-8")
                        resolved = sum(1 for ct in co_patches.values() if not is_partial_segments_marker(ct))
                        print(f"  Standalone carry-over: patched {len(co_patches)} item(s) at original lines ({resolved} resolved).")
                # Remaining carry-over items (≤ chunk_size) will be appended to the chunk batch.
                chunk_carryover = list(carryover_items)
                carryover_items.clear()

            carryover_batch_offset = len(chunk_transcriptions)  # where carry-over starts in batch arrays

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
                        resume_cell = resume_lines[ri]
                        if retry_content_filtered and resume_cell.startswith("%%CF%%"):
                            # Treat content-filtered cell as empty so it gets retried.
                            pass
                        elif is_partial_segments_marker(resume_cell):
                            # Partial segment markers: mark as pre_resolved so
                            # run_batch_pipeline skips them. Partial payload will
                            # be injected into chunk_payloads after pipeline returns.
                            pre_resolved.add(i)
                        else:
                            pre_resolved.add(i)

            # Inject resumed partial-segment payloads into chunk_payloads slots
            # so the per-chunk retry loop can pick them up.
            resumed_partial_indices: list[int] = []
            for i, t in enumerate(chunk_transcriptions):
                if resume_lines is not None:
                    ri = global_offset + i
                    if ri < len(resume_lines) and is_partial_segments_marker(resume_lines[ri] if ri < len(resume_lines) else ""):
                        resumed_partial_indices.append(i)

            # Append carry-over items to the batch arrays.
            for co_global_idx, co_text, co_fn, co_sr, co_resume_line in chunk_carryover:
                stripped_co, _ = strip_emojis(co_text)
                prompt_co, _ = normalize_all_uppercase_input(stripped_co)
                batch_texts.append(prompt_co)
                skip_flags.append(
                    is_all_uppercase_cased_input(co_text) or is_all_lowercase_cased_input(co_text)
                )
                # If carry-over has a partial marker, handle like resumed partial.
                co_batch_idx = len(batch_texts) - 1
                if is_partial_segments_marker(co_resume_line):
                    pre_resolved.add(co_batch_idx)
                    resumed_partial_indices.append(co_batch_idx)

            if chunk_carryover:
                print(f"  Including {len(chunk_carryover)} carry-over item(s) from previous chunks.")

            # Run batch pipeline for this chunk.
            chunk_payloads = await run_batch_pipeline(
                client=batch_client,
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
                use_realtime=args.batch_use_realtime,
                batch_metadata=batch_metadata,
                batch_debug_dir=args.batch_debug_dir,
                batch_debug_chunk_id=f"{global_offset + 1}_{chunk_end}",
            )
            total_batch_len = chunk_len + len(chunk_carryover)
            assert len(chunk_payloads) == total_batch_len, (
                f"Batch pipeline alignment error: expected {total_batch_len} payloads, got {len(chunk_payloads)}"
            )

            # Inject resumed partial-segment payloads so per-chunk retry picks them up.
            for i in resumed_partial_indices:
                if i < chunk_len:
                    ri = global_offset + i
                    if ri < len(resume_lines):
                        partial_p = build_empty_payload()
                        partial_p["corrected_text"] = resume_lines[ri]
                        chunk_payloads[i] = partial_p
                else:
                    # Carry-over partial: get resume line from chunk_carryover.
                    co_idx = i - carryover_batch_offset
                    if co_idx < len(chunk_carryover):
                        co_resume_line = chunk_carryover[co_idx][4]
                        partial_p = build_empty_payload()
                        partial_p["corrected_text"] = co_resume_line
                        chunk_payloads[i] = partial_p

            # Per-chunk retry: retry failed items within this chunk immediately
            # so that segmented long lines can be fully resolved before moving on.
            chunk_max_retries = getattr(args, "batch_max_retries", 0)
            # Include resumed partial-segment items in retry (they were in pre_resolved
            # but now have partial payloads injected above).
            resumed_partial_set = set(resumed_partial_indices)

            # Helper: build output lines from current chunk_payloads and write to file.
            def _write_chunk_snapshot(label: str = "") -> list[str]:
                nonlocal first_write, total_written
                snap_output_lines: list[str] = []
                for i in range(chunk_len):
                    payload = chunk_payloads[i]
                    t = chunk_transcriptions[i]
                    if payload is None:
                        if is_input_comment_line(t):
                            snap_output_lines.append(t)
                        elif resume_lines is not None:
                            ri = global_offset + i
                            snap_output_lines.append(resume_lines[ri] if ri < len(resume_lines) else "")
                        else:
                            snap_output_lines.append("")
                    else:
                        ct = sanitize_output_string(payload.get("corrected_text") or "")
                        if is_partial_segments_marker(ct):
                            snap_output_lines.append(ct)
                        elif ct.strip():
                            if args.apply_safe_edits:
                                original = chunk_transcriptions[i]
                                if isinstance(original, str) and original.strip():
                                    merged, _ = postprocess_apply_safe_edits(original, ct)
                                    ct = merged
                            snap_output_lines.append(ct)
                        else:
                            snap_output_lines.append("")
                if out_text_path:
                    out_lines = [
                        _build_output_line(
                            text,
                            chunk_source_rows[i] if i < len(chunk_source_rows) else None,
                            chunk_filenames[i] if i < len(chunk_filenames) else None,
                        )
                        for i, text in enumerate(snap_output_lines)
                    ]
                    content = "\n".join(out_lines) + "\n"
                    if first_write:
                        out_text_path.write_text(content, encoding="utf-8")
                        first_write = False
                    else:
                        # Re-read file, drop last chunk_len lines if already written, re-append.
                        if label:
                            # Overwrite the last chunk's lines in-place.
                            existing = out_text_path.read_text(encoding="utf-8")
                            existing_lines = existing.split("\n")
                            if existing_lines and existing_lines[-1] == "":
                                existing_lines.pop()
                            # Remove previously written chunk lines.
                            prefix_lines = existing_lines[:len(existing_lines) - chunk_len]
                            new_content = "\n".join(prefix_lines + out_lines) + "\n"
                            out_text_path.write_text(new_content, encoding="utf-8")
                        else:
                            with out_text_path.open("a", encoding="utf-8") as fh:
                                fh.write(content)
                    if label:
                        non_empty = sum(1 for l in snap_output_lines if l.strip() and not is_partial_segments_marker(l))
                        print(f"  Snapshot after {label}: {non_empty}/{chunk_len} resolved lines written.")
                # Verify output line count after write and sync total_written.
                if out_text_path and out_text_path.exists():
                    actual_lines = sum(1 for _ in out_text_path.open("r", encoding="utf-8"))
                    expected_lines = total_written + chunk_len
                    if actual_lines != expected_lines:
                        print(f"  WARNING: Output file has {actual_lines:,} lines but expected {expected_lines:,}. Syncing.")
                    # Sync total_written from actual file to prevent drift.
                    total_written = actual_lines - chunk_len
                return snap_output_lines

            # Write initial snapshot before retries so results are persisted.
            _write_chunk_snapshot()

            for chunk_retry in range(1, chunk_max_retries + 1):
                # Collect chunk-local failed indices (including carry-over at end).
                chunk_failed_indices: list[int] = []
                for i, payload in enumerate(chunk_payloads):
                    if i in pre_resolved and i not in resumed_partial_set:
                        continue
                    # Only check comment lines for original chunk items (not carry-over).
                    if i < chunk_len and is_input_comment_line(chunk_transcriptions[i]):
                        continue
                    if payload is None:
                        chunk_failed_indices.append(i)
                        continue
                    ct = payload.get("corrected_text")
                    if not (isinstance(ct, str) and ct.strip()) or is_partial_segments_marker(ct if isinstance(ct, str) else ""):
                        chunk_failed_indices.append(i)

                if not chunk_failed_indices:
                    break

                print(f"  Chunk retry {chunk_retry}/{chunk_max_retries}: {len(chunk_failed_indices)} failed items...")
                retry_texts = [batch_texts[i] for i in chunk_failed_indices]
                retry_skip = [skip_flags[i] for i in chunk_failed_indices]
                retry_max_chars = max(1, max_input_chars_per_call // (2 ** chunk_retry)) if max_input_chars_per_call > 0 else 0

                retry_payloads = await run_batch_pipeline(
                    client=batch_client,
                    deployment=batch_deployment,
                    transcriptions=retry_texts,
                    prompt_template=prompt_template,
                    chain_steps=chain_steps,
                    locale=args.locale,
                    schema=PATCH_SCHEMA,
                    temperature=temperature,
                    top_p=top_p,
                    batch_size=args.batch_size,
                    build_patch_prompt_fn=build_patch_prompt,
                    skip_first_token_casing_preservation_flags=retry_skip,
                    active_step_keys=active_step_keys,
                    max_input_chars_per_call=retry_max_chars,
                    concurrency=concurrency,
                    use_realtime=args.batch_use_realtime,
                    batch_metadata=batch_metadata,
                )

                resolved_count = 0
                for fi, rp in zip(chunk_failed_indices, retry_payloads):
                    if rp is not None:
                        rct = rp.get("corrected_text", "")
                        if isinstance(rct, str) and rct.strip() and not is_partial_segments_marker(rct):
                            chunk_payloads[fi] = rp
                            resolved_count += 1
                        elif isinstance(rct, str) and is_partial_segments_marker(rct):
                            # Updated partial marker — keep for next retry pass.
                            chunk_payloads[fi] = rp
                print(f"  Chunk retry {chunk_retry}: resolved {resolved_count}/{len(chunk_failed_indices)}")
                # Write snapshot after each retry so results are persisted.
                _write_chunk_snapshot(label=f"retry {chunk_retry}")
                if resolved_count == 0:
                    break  # No progress — stop retrying this chunk.

            # Build final chunk output lines (already written by _write_chunk_snapshot)
            # and collect failed items for end-of-run retry.
            chunk_output_lines: list[str] = []
            for i in range(chunk_len):
                payload = chunk_payloads[i]
                t = chunk_transcriptions[i]
                if payload is None:
                    if is_input_comment_line(t):
                        chunk_output_lines.append(t)
                    elif resume_lines is not None:
                        ri = global_offset + i
                        chunk_output_lines.append(resume_lines[ri] if ri < len(resume_lines) else "")
                    else:
                        chunk_output_lines.append("")
                    # Track as failed for end-of-run retry (skip comments and
                    # non-resumed pre-resolved items).
                    if (i not in pre_resolved or i in resumed_partial_set) and not is_input_comment_line(t):
                        fn = chunk_filenames[i] if chunk_filenames and i < len(chunk_filenames) else None
                        all_failed_items.append((global_offset + i, batch_texts[i], t, fn, skip_flags[i]))
                else:
                    ct = payload.get("corrected_text")
                    if isinstance(ct, str) and is_partial_segments_marker(ct):
                        chunk_output_lines.append(ct)
                        if (i not in pre_resolved or i in resumed_partial_set) and not is_input_comment_line(t):
                            fn = chunk_filenames[i] if chunk_filenames and i < len(chunk_filenames) else None
                            all_failed_items.append((global_offset + i, batch_texts[i], t, fn, skip_flags[i]))
                    elif isinstance(ct, str) and ct.strip():
                        if args.apply_safe_edits:
                            original = chunk_transcriptions[i]
                            if isinstance(original, str) and original.strip():
                                merged, _ = postprocess_apply_safe_edits(original, ct)
                                ct = merged
                        chunk_output_lines.append(ct)
                    else:
                        chunk_output_lines.append("")
                        if (i not in pre_resolved or i in resumed_partial_set) and not is_input_comment_line(t):
                            fn = chunk_filenames[i] if chunk_filenames and i < len(chunk_filenames) else None
                            all_failed_items.append((global_offset + i, batch_texts[i], t, fn, skip_flags[i]))

            # Final snapshot write (ensures last retry state is persisted).
            # Skip if no retries were attempted — initial snapshot already wrote.
            if chunk_max_retries > 0:
                _write_chunk_snapshot(label="final")

            # Write JSONL records for this chunk.
            if out_jsonl_path:
                if first_jsonl_write and not resume_from_output:
                    out_jsonl_path.write_text("", encoding="utf-8")
                    first_jsonl_write = False
                for i in range(len(chunk_payloads)):
                    payload = chunk_payloads[i]
                    if payload is not None and isinstance(payload, dict):
                        ct = payload.get("corrected_text", "")
                        if isinstance(ct, str) and ct.strip() and not is_partial_segments_marker(ct):
                            if i < chunk_len:
                                source_fn = chunk_filenames[i] if chunk_filenames and i < len(chunk_filenames) else None
                            else:
                                co_idx = i - carryover_batch_offset
                                source_fn = chunk_carryover[co_idx][2] if co_idx < len(chunk_carryover) else None
                            append_payload_jsonl_record(
                                out_jsonl_path, payload, source_fn, active_step_keys,
                            )

            # Patch carry-over results back into their original lines in the output file.
            if chunk_carryover and out_text_path and out_text_path.exists():
                carryover_patches: dict[int, str] = {}
                for co_idx, (co_global_idx, co_text, co_fn, co_sr, co_resume_line) in enumerate(chunk_carryover):
                    batch_idx = carryover_batch_offset + co_idx
                    if batch_idx < len(chunk_payloads):
                        co_payload = chunk_payloads[batch_idx]
                        if co_payload is not None:
                            co_ct = co_payload.get("corrected_text", "")
                            if isinstance(co_ct, str) and co_ct.strip() and not is_partial_segments_marker(co_ct):
                                if args.apply_safe_edits and isinstance(co_text, str) and co_text.strip():
                                    merged, _ = postprocess_apply_safe_edits(co_text, co_ct)
                                    co_ct = merged
                                carryover_patches[co_global_idx] = co_ct
                            elif isinstance(co_ct, str) and is_partial_segments_marker(co_ct):
                                carryover_patches[co_global_idx] = co_ct
                                all_failed_items.append((co_global_idx, batch_texts[batch_idx], co_text, co_fn, skip_flags[batch_idx]))
                            else:
                                all_failed_items.append((co_global_idx, batch_texts[batch_idx], co_text, co_fn, skip_flags[batch_idx]))
                        else:
                            all_failed_items.append((co_global_idx, batch_texts[batch_idx], co_text, co_fn, skip_flags[batch_idx]))

                if carryover_patches:
                    from common import sanitize_output_string as _co_san
                    co_file_lines = out_text_path.read_text(encoding="utf-8").split("\n")
                    if co_file_lines and co_file_lines[-1] == "":
                        co_file_lines.pop()
                    for gidx, ct in carryover_patches.items():
                        if gidx < len(co_file_lines):
                            sanitized_ct = ct if is_partial_segments_marker(ct) else _co_san(ct)
                            co_line = co_file_lines[gidx]
                            if "\t" in co_line:
                                co_prefix, _ = co_line.rsplit("\t", 1)
                                co_file_lines[gidx] = f"{co_prefix}\t{sanitized_ct}"
                            else:
                                co_file_lines[gidx] = sanitized_ct
                    out_text_path.write_text("\n".join(co_file_lines) + "\n", encoding="utf-8")
                    resolved_co = sum(1 for ct in carryover_patches.values() if not is_partial_segments_marker(ct))
                    print(f"  Patched {len(carryover_patches)} carry-over item(s) at original lines ({resolved_co} resolved).")

            total_written += chunk_len
            print(f"  Chunk complete. Total written: {total_written:,}")

        print(f"\nAll chunks done. Total: {total_written:,} items.")
        if chunks_completed_in_resume > 0:
            print(f"  ({chunks_completed_in_resume} chunks were already completed from resume.)")

        # Flush any remaining carry-over items to all_failed_items.
        if carryover_items:
            print(f"  {len(carryover_items)} carry-over item(s) remaining after all chunks (will retry at end).")
            for co_global_idx, co_text, co_fn, co_sr, co_resume_line in carryover_items:
                stripped_co, _ = strip_emojis(co_text)
                prompt_co, _ = normalize_all_uppercase_input(stripped_co)
                co_skip = is_all_uppercase_cased_input(co_text) or is_all_lowercase_cased_input(co_text)
                all_failed_items.append((co_global_idx, prompt_co, co_text, co_fn, co_skip))
            carryover_items.clear()

        # ---------------------------------------------------------------------
        #  End-of-run retry for items that failed during chunk processing.
        #  Handles both fully-failed items and partial-segment markers.
        # ---------------------------------------------------------------------
        batch_max_retries = getattr(args, "batch_max_retries", 0)
        if all_failed_items and batch_max_retries > 0 and out_text_path and out_text_path.exists():
            import random
            from common import join_segment_text_parts

            # Read current output to detect partial segment markers.
            def _read_output_lines() -> list[str]:
                raw = out_text_path.read_text(encoding="utf-8")
                lines = raw.split("\n")
                if lines and lines[-1] == "":
                    lines.pop()
                return lines

            def _extract_last_column(line: str) -> str:
                if "\t" in line:
                    return line.rsplit("\t", 1)[1]
                return line

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

                retry_max_chars = max(1, max_input_chars_per_call // (2 ** retry_pass)) if max_input_chars_per_call > 0 else 0

                # Read output to check for partial segment markers.
                current_output = _read_output_lines()

                # Separate items into: partial-segment retries vs full retries.
                partial_items: list[tuple[int, list[tuple[bool, str]], int, str, bool]] = []
                full_retry_items: list[tuple[int, str, str, str | None, bool]] = []

                for item in retry_remaining:
                    global_idx = item[0]
                    if global_idx < len(current_output):
                        last_col = _extract_last_column(current_output[global_idx])
                        decoded = decode_partial_segments(last_col)
                        if decoded is not None:
                            seg_limit, seg_parts = decoded
                            partial_items.append((global_idx, seg_parts, seg_limit, item[2], item[4]))
                            continue
                    full_retry_items.append(item)

                patched_lines: dict[int, str] = {}
                still_failed: list[tuple[int, str, str, str | None, bool]] = []

                # --- Handle partial-segment retries ---
                if partial_items:
                    # Collect only the failed segments from all partial items.
                    # Track: (partial_item_index, segment_index, source_text)
                    failed_seg_texts: list[str] = []
                    failed_seg_map: list[tuple[int, int]] = []  # (partial_idx, seg_idx)
                    for pi, (global_idx, seg_parts, seg_limit, orig_text, skip_flag) in enumerate(partial_items):
                        for si, (ok, text) in enumerate(seg_parts):
                            if ok is True:
                                continue  # Skip successful segments.
                            if ok == "filtered" and not retry_content_filtered:
                                continue  # Skip content-filtered segments (unless retry requested).
                            failed_seg_texts.append(text)
                            failed_seg_map.append((pi, si))

                    if failed_seg_texts:
                        print(f"  Retrying {len(failed_seg_texts)} failed segments from {len(partial_items)} partial items...")
                        seg_skip_flags = [False] * len(failed_seg_texts)
                        seg_payloads = await run_batch_pipeline(
                            client=batch_client,
                            deployment=batch_deployment,
                            transcriptions=failed_seg_texts,
                            prompt_template=prompt_template,
                            chain_steps=chain_steps,
                            locale=args.locale,
                            schema=PATCH_SCHEMA,
                            temperature=pass_temperature,
                            top_p=top_p,
                            batch_size=args.batch_size,
                            build_patch_prompt_fn=build_patch_prompt,
                            skip_first_token_casing_preservation_flags=seg_skip_flags,
                            active_step_keys=active_step_keys,
                            max_input_chars_per_call=retry_max_chars,
                            concurrency=concurrency,
                            use_realtime=args.batch_use_realtime,
                            batch_metadata=batch_metadata,
                        )

                        # Apply retry results back into partial items.
                        for (pi, si), payload in zip(failed_seg_map, seg_payloads):
                            if payload is not None:
                                ct = payload.get("corrected_text", "")
                                if isinstance(ct, str) and ct.strip():
                                    partial_items[pi][1][si] = (True, ct)

                    # Check which partial items are now fully resolved.
                    for global_idx, seg_parts, seg_limit, orig_text, skip_flag in partial_items:
                        still_has_failures = any(not ok for ok, _ in seg_parts)
                        if not still_has_failures:
                            # All segments now succeeded — merge.
                            merged_text = join_segment_text_parts([text for _, text in seg_parts])
                            if args.apply_safe_edits and isinstance(orig_text, str) and orig_text.strip():
                                merged_text, _ = postprocess_apply_safe_edits(orig_text, merged_text)
                            patched_lines[global_idx] = merged_text
                        else:
                            # Still has failures — re-encode partial marker.
                            new_marker = encode_partial_segments(seg_parts, seg_limit)
                            patched_lines[global_idx] = new_marker
                            # Keep in retry list for next pass.
                            still_failed.append((global_idx, orig_text, orig_text, None, skip_flag))

                # --- Handle full retries (non-segmented failed items) ---
                if full_retry_items:
                    retry_texts = [item[1] for item in full_retry_items]
                    retry_skip_flags = [item[4] for item in full_retry_items]

                    retry_payloads = await run_batch_pipeline(
                        client=batch_client,
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
                        use_realtime=args.batch_use_realtime,
                        batch_metadata=batch_metadata,
                    )

                    for item, payload in zip(full_retry_items, retry_payloads, strict=True):
                        global_idx = item[0]
                        if payload is not None:
                            ct = payload.get("corrected_text", "")
                            if isinstance(ct, str) and is_partial_segments_marker(ct):
                                # New partial result from retry — write marker.
                                patched_lines[global_idx] = ct
                                still_failed.append(item)
                            elif isinstance(ct, str) and ct.strip():
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

                # --- Write patched lines to output file ---
                if patched_lines:
                    output_lines = _read_output_lines()
                    for global_idx, ct in patched_lines.items():
                        if global_idx < len(output_lines):
                            # Partial markers must be written verbatim (contain JSON).
                            sanitized_ct = ct if is_partial_segments_marker(ct) else sanitize_output_string(ct)
                            line = output_lines[global_idx]
                            if "\t" in line:
                                prefix, _old_text = line.rsplit("\t", 1)
                                output_lines[global_idx] = f"{prefix}\t{sanitized_ct}"
                            else:
                                output_lines[global_idx] = sanitized_ct
                    out_text_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
                    resolved_count = sum(1 for ct in patched_lines.values() if not is_partial_segments_marker(ct))
                    print(f"  Patched {len(patched_lines):,} lines ({resolved_count:,} fully resolved).")

                retry_remaining = still_failed

            if retry_remaining:
                # Final pass: resolve remaining partial markers to merged text
                # with original text for still-failed segments.
                final_output = _read_output_lines()
                final_patches: dict[int, str] = {}
                for item in retry_remaining:
                    global_idx = item[0]
                    if global_idx < len(final_output):
                        last_col = _extract_last_column(final_output[global_idx])
                        decoded = decode_partial_segments(last_col)
                        if decoded is not None:
                            _, seg_parts = decoded
                            merged = join_segment_text_parts([text for _, text in seg_parts])
                            final_patches[global_idx] = merged
                if final_patches:
                    for global_idx, ct in final_patches.items():
                        if global_idx < len(final_output):
                            line = final_output[global_idx]
                            if "\t" in line:
                                prefix, _ = line.rsplit("\t", 1)
                                final_output[global_idx] = f"{prefix}\t{sanitize_output_string(ct)}"
                            else:
                                final_output[global_idx] = sanitize_output_string(ct)
                    out_text_path.write_text("\n".join(final_output) + "\n", encoding="utf-8")
                    print(f"  Resolved {len(final_patches):,} remaining partial markers to merged text.")
                print(f"\n{len(retry_remaining):,} items still had partial failures after {batch_max_retries} retry pass(es).")
        elif all_failed_items:
            print(f"\n{len(all_failed_items):,} items failed (retries disabled).")

        return

    # -----------------------------------------------------------------------
    #  Chunked real-time mode (streaming from file)
    # -----------------------------------------------------------------------
    if use_chunked and not args.batch_mode:
        from common import sanitize_output_string, resolve_path as _resolve_path

        # Use the same chunk_size as batch mode but for real-time calls.
        rt_chunk_size = args.batch_size * concurrency

        def _sanitize_tsv_cell(value: str) -> str:
            if not isinstance(value, str):
                return ""
            return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")

        def _build_output_line(
            corrected_text: str,
            source_row: list[str] | None,
            filename: str | None,
        ) -> str:
            ct = sanitize_output_string(corrected_text)
            if isinstance(source_row, list) and source_row:
                rebuilt = [_sanitize_tsv_cell(v) for v in source_row[:-1]]
                rebuilt.append(ct)
                return "\t".join(rebuilt)
            if isinstance(filename, str) and filename:
                return f"{_sanitize_tsv_cell(filename)}\t{ct}"
            return ct

        # Load resume lines if output file exists.
        import csv as _rt_csv
        resume_lines: list[str] | None = None
        if resume_from_output and output_file_value:
            _out_path = _resolve_path(output_file_value)
            _out_text_path = _out_path.with_suffix(".tsv") if output_as_tsv else _out_path.with_suffix(".txt")
            if not _out_text_path.exists():
                _alt_path = _out_path.with_suffix(".txt") if output_as_tsv else _out_path.with_suffix(".tsv")
                if _alt_path.exists():
                    _out_text_path = _alt_path
            if _out_text_path.exists():
                try:
                    _raw = _out_text_path.read_text(encoding="utf-8")
                    resume_lines = []
                    _raw_lines = _raw.split("\n")
                    if _raw_lines and _raw_lines[-1] == "":
                        _raw_lines.pop()
                    for row in _rt_csv.reader(_raw_lines, delimiter="\t"):
                        cell = row[-1] if row else ""
                        resume_lines.append(sanitize_output_string(cell))
                    non_empty = sum(1 for l in resume_lines if l.strip())
                    print(f"Loaded resume output: {non_empty:,} non-empty lines.")
                    input_line_count = count_input_lines(input_file_value)
                    if input_line_count > 0 and len(resume_lines) > input_line_count:
                        print(
                            f"  WARNING: Resume file has {len(resume_lines):,} lines but input has {input_line_count:,}. "
                            "Resume file appears corrupted. Discarding resume data."
                        )
                        resume_lines = None
                except Exception as e:
                    print(f"Could not load resume file: {e}")

        # Prepare output paths.
        out_path = _resolve_path(output_file_value) if output_file_value else None
        out_text_path = None
        if out_path:
            out_text_path = out_path.with_suffix(".tsv") if output_as_tsv else out_path.with_suffix(".txt")
            out_text_path.parent.mkdir(parents=True, exist_ok=True)

        first_write = True
        total_written = 0
        global_line_count = count_input_lines(input_file_value)

        print(
            f"Streaming real-time mode (deployment={deployment}, "
            f"chunk_size={rt_chunk_size:,}, concurrency={concurrency}, "
            f"total_lines={global_line_count:,})"
        )

        for global_offset, chunk_transcriptions, chunk_filenames, chunk_source_rows in iter_transcription_chunks(
            input_file_value, rt_chunk_size, max_input_chars_per_call=max_input_chars_per_call
        ):
            if shutdown_requested.is_set():
                break

            chunk_len = len(chunk_transcriptions)
            chunk_end = global_offset + chunk_len
            print(f"\n--- Chunk [{global_offset + 1:,}..{chunk_end:,}] ({chunk_len:,} items) ---")

            # Check resume: skip chunk if fully resolved.
            chunk_resume: list[str] | None = None
            if resume_lines is not None:
                cr = resume_lines[global_offset:chunk_end] if global_offset < len(resume_lines) else []
                if len(cr) == chunk_len and all(isinstance(cl, str) and cl.strip() for cl in cr):
                    print(f"  Chunk already completed in resume output; skipping.")
                    if out_text_path:
                        out_lines = [
                            _build_output_line(
                                cr[i],
                                chunk_source_rows[i] if i < len(chunk_source_rows) else None,
                                chunk_filenames[i] if i < len(chunk_filenames) else None,
                            )
                            for i in range(chunk_len)
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
                # Partial resume — use what we have.
                if len(cr) == chunk_len:
                    chunk_resume = cr

            # Per-chunk arrays.
            chunk_payloads: list[dict | None] = [None] * chunk_len
            chunk_output_lines: list[str] = [""] * chunk_len

            # Pre-fill from resume.
            if chunk_resume is not None:
                for ci, cl in enumerate(chunk_resume):
                    if isinstance(cl, str) and cl.strip():
                        chunk_output_lines[ci] = cl

            async def _process_chunk_item(index: int, transcription: str, total: int) -> None:
                if shutdown_requested.is_set():
                    return
                slot = index - 1  # index is 1-based within chunk
                global_index = global_offset + slot + 1
                processing_id = f"{global_index:,}/{chunk_end:,}"

                if is_input_comment_line(transcription):
                    chunk_output_lines[slot] = transcription
                    return

                if not transcription.strip():
                    chunk_payloads[slot] = build_empty_payload()
                    chunk_payloads[slot]["source_text"] = transcription
                    return

                # Resume check.
                if chunk_resume is not None:
                    existing = chunk_output_lines[slot]
                    if isinstance(existing, str) and existing.strip():
                        chunk_payloads[slot] = build_empty_payload()
                        chunk_payloads[slot]["source_text"] = transcription
                        chunk_payloads[slot]["corrected_text"] = existing
                        return

                stripped_transcription, _ = strip_emojis(transcription)
                prompt_transcription, _ = normalize_all_uppercase_input(stripped_transcription)
                source_was_all_uppercase = is_all_uppercase_cased_input(transcription)
                source_was_all_lowercase = is_all_lowercase_cased_input(transcription)
                skip_casing = source_was_all_uppercase or source_was_all_lowercase

                try:
                    source_length = len(prompt_transcription)
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
                            seg_text, _ = take_next_transcription_segment_for_llm(
                                prompt_transcription, start_offset, current_limit,
                            )
                            if not seg_text:
                                seg_text = prompt_transcription[start_offset:start_offset + 1]

                            seg_id = f"{processing_id} seg {segment_index}"
                            _CTX_LEN = 50
                            seg_prev_ctx = segment_sources[-1][-_CTX_LEN:] if segment_sources else None
                            seg_next_end = start_offset + len(seg_text)
                            seg_next_ctx = prompt_transcription[seg_next_end:seg_next_end + _CTX_LEN] if seg_next_end < source_length else None

                            prompt = build_patch_prompt(
                                prompt_template, seg_text, chain_steps,
                                locale=args.locale,
                                prev_context=seg_prev_ctx,
                                next_context=seg_next_ctx,
                            )

                            seg_payload = await get_patch_payload_with_repair(
                                client=client, deployment=deployment,
                                prompt=prompt, transcription=seg_text,
                                processing_id=seg_id,
                                repair_prompt_template=repair_prompt_template,
                                patch_schema=PATCH_SCHEMA,
                                timeout_seconds=timeout_seconds,
                                timeout_retries=timeout_retries,
                                empty_result_retries=empty_result_retries,
                                temperature=temperature, top_p=top_p,
                                retry_temperature_jitter=retry_temperature_jitter,
                                retry_top_p_jitter=retry_top_p_jitter,
                                skip_first_token_casing_preservation=skip_casing,
                                active_step_keys=active_step_keys,
                            )

                            if seg_payload is not None:
                                segment_payloads.append(seg_payload)
                                segment_sources.append(seg_text)
                                start_offset += len(seg_text)
                                segment_index += 1
                                segment_succeeded = True
                                break

                            if len(seg_text) > 1 and current_limit > 1:
                                next_limit = max(1, min(current_limit - 1, len(seg_text) - 1, current_limit // 2))
                                if next_limit < current_limit:
                                    current_limit = next_limit
                                    continue

                            payload = None
                            break

                        if not segment_succeeded:
                            break

                    if len(segment_sources) > 1:
                        payload = merge_segment_payloads(segment_payloads, segment_sources) if len(segment_payloads) == len(segment_sources) else None
                    elif segment_payloads:
                        payload = segment_payloads[0]
                    else:
                        payload = None

                    assign_payload_or_emit_empty(payload, chunk_payloads, slot, global_index, chunk_end)
                    resolved = chunk_payloads[slot]
                    if isinstance(resolved, dict):
                        fn = chunk_filenames[slot] if slot < len(chunk_filenames) else None
                        if isinstance(fn, str) and fn:
                            resolved["source_filename"] = fn
                        resolved["source_text"] = transcription
                        ct = resolved.get("corrected_text")
                        if args.apply_safe_edits and isinstance(ct, str) and ct.strip() and transcription.strip():
                            merged, _ = postprocess_apply_safe_edits(transcription, ct)
                            if merged != ct:
                                resolved["corrected_text"] = merged
                                ct = merged
                        chunk_output_lines[slot] = ct if isinstance(ct, str) else ""

                except asyncio.CancelledError:
                    chunk_payloads[slot] = build_empty_payload()
                    chunk_payloads[slot]["source_text"] = transcription
                    chunk_output_lines[slot] = ""
                except Exception as error:
                    chunk_payloads[slot] = build_empty_payload()
                    chunk_payloads[slot]["source_text"] = transcription
                    chunk_output_lines[slot] = ""
                    print(f"[{processing_id}] Error: {error}; emitting empty payload.")

            # Process this chunk with concurrency.
            await run_transcriptions_with_concurrency(
                chunk_transcriptions, concurrency, _process_chunk_item,
                global_offset=global_offset,
                global_total=global_line_count,
            )

            # Write chunk output.
            if out_text_path:
                out_lines = [
                    _build_output_line(
                        chunk_output_lines[i],
                        chunk_source_rows[i] if i < len(chunk_source_rows) else None,
                        chunk_filenames[i] if i < len(chunk_filenames) else None,
                    )
                    for i in range(chunk_len)
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

        if shutdown_requested.is_set():
            print("Shutdown handler completed final progress flush.")
        return

    # -----------------------------------------------------------------------
    #  Real-time mode — stdin fallback (no input file)
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
        if not output_jsonl
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
