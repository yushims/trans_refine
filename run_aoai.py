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
    insert_spaces_at_script_boundaries,
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
    #  Batch API mode
    # -----------------------------------------------------------------------
    if args.batch_mode:
        from common import DEFAULT_AOAI_BATCH_DEPLOYMENT
        batch_deployment = DEFAULT_AOAI_BATCH_DEPLOYMENT
        print(f"Batch mode enabled (deployment={batch_deployment}, batch_size={args.batch_size})")

        # Load existing output for resume support.
        batch_resume_lines: list[str] | None = None
        if resume_from_output:
            batch_resume_lines = load_existing_output_text_lines(
                output_file_value,
                len(transcriptions),
                output_as_tsv,
            )

        # Pre-compute per-item preprocessing and casing flags.
        # Mark already-completed items so the batch pipeline can skip them.
        skip_casing_flags: list[bool] = []
        batch_transcriptions: list[str] = []
        pre_resolved_indices: set[int] = set()
        for idx, t in enumerate(transcriptions):
            stripped_t, _ = strip_emojis(t)
            prompt_t, _ = normalize_all_uppercase_input(stripped_t)
            prompt_t, _ = insert_spaces_at_script_boundaries(prompt_t)
            batch_transcriptions.append(prompt_t)
            skip_casing_flags.append(
                is_all_uppercase_cased_input(t) or is_all_lowercase_cased_input(t)
            )
            # Skip comment lines (e.g. "# zh-CN") — treat as pre-resolved.
            if is_input_comment_line(t):
                pre_resolved_indices.add(idx)
                continue
            # If resume has a non-empty line for this index, mark it pre-resolved.
            if batch_resume_lines is not None:
                existing = batch_resume_lines[idx] if idx < len(batch_resume_lines) else ""
                if isinstance(existing, str) and existing.strip():
                    pre_resolved_indices.add(idx)

        if pre_resolved_indices:
            print(f"Batch mode: resuming — {len(pre_resolved_indices)} items already completed, skipping them.")

        def _on_batch_pass_complete(payloads_snapshot: list[dict | None]) -> None:
            """Write a progress text snapshot after each batch pass."""
            if not output_file_value:
                return
            progress_lines: list[str] = [""] * len(transcriptions)
            for i, p in enumerate(payloads_snapshot):
                if isinstance(p, dict):
                    ct = p.get("corrected_text")
                    progress_lines[i] = ct if isinstance(ct, str) else ""
                elif batch_resume_lines is not None and i < len(batch_resume_lines):
                    progress_lines[i] = batch_resume_lines[i] or ""
            write_fallback_text_output(
                output_file_value,
                progress_lines,
                source_filenames,
                source_rows,
                output_as_tsv=output_as_tsv,
            )

        payloads_batch = await run_batch_pipeline(
            client=client,
            deployment=batch_deployment,
            transcriptions=batch_transcriptions,
            prompt_template=prompt_template,
            chain_steps=chain_steps,
            locale=args.locale,
            schema=PATCH_SCHEMA,
            temperature=temperature,
            top_p=top_p,
            batch_size=args.batch_size,
            build_patch_prompt_fn=build_patch_prompt,
            skip_first_token_casing_preservation_flags=skip_casing_flags,
            active_step_keys=active_step_keys,
            max_input_chars_per_call=max_input_chars_per_call,
            retry_temperature_jitter=retry_temperature_jitter,
            concurrency=concurrency,
            pre_resolved_indices=pre_resolved_indices,
            on_pass_complete=_on_batch_pass_complete,
        )

        # Merge resume lines and comment lines into results.
        if batch_resume_lines is not None:
            for idx in pre_resolved_indices:
                if payloads_batch[idx] is None and not is_input_comment_line(transcriptions[idx]):
                    payloads_batch[idx] = build_empty_payload()
                    payloads_batch[idx]["source_text"] = transcriptions[idx]
                    payloads_batch[idx]["corrected_text"] = batch_resume_lines[idx]
        for idx in pre_resolved_indices:
            if is_input_comment_line(transcriptions[idx]) and payloads_batch[idx] is None:
                payloads_batch[idx] = build_empty_payload()
                payloads_batch[idx]["source_text"] = transcriptions[idx]
                payloads_batch[idx]["corrected_text"] = transcriptions[idx]

        # Post-processing: character-level diff merge to reject content changes.
        if args.apply_safe_edits:
            for idx, payload in enumerate(payloads_batch):
                if not isinstance(payload, dict):
                    continue
                corrected = payload.get("corrected_text")
                if not isinstance(corrected, str) or not corrected.strip():
                    continue
                original = transcriptions[idx]
                if not isinstance(original, str) or not original.strip():
                    continue
                merged, _ = postprocess_apply_safe_edits(original, corrected)
                if merged != corrected:
                    payload["corrected_text"] = merged

        # Attach source metadata and build text output lines.
        text_lines_batch: list[str] = [""] * len(transcriptions)
        for idx, payload in enumerate(payloads_batch):
            if payload is None:
                payload = build_empty_payload()
                payloads_batch[idx] = payload
            payload["source_text"] = transcriptions[idx]
            source_fn = source_filenames[idx] if idx < len(source_filenames) else None
            if isinstance(source_fn, str) and source_fn:
                payload["source_filename"] = source_fn
            corrected = payload.get("corrected_text")
            text_lines_batch[idx] = corrected if isinstance(corrected, str) else ""

        finalize_payloads_and_write(
            payloads_batch,
            output_file_value,
            text_lines_batch,
            source_filenames,
            source_rows,
            output_as_tsv,
            active_step_keys,
        )
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
        prompt_transcription, _ = insert_spaces_at_script_boundaries(prompt_transcription)
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
