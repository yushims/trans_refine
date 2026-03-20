import argparse
import asyncio
from typing import Any

from copilot import CopilotClient  # pyright: ignore[reportMissingImports]
from common import (
    DEFAULT_COPILOT_MODEL,
    add_common_runtime_cli_arguments,
    add_model_mismatch_retries_cli_argument,
    assign_payload_or_emit_empty,
    build_patch_prompt,
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
    load_patch_and_repair_templates,
    load_existing_output_text_lines,
    merge_segment_payloads,
    print_common_runtime_settings,
    resolve_active_chain_step_keys,
    resolve_patch_and_repair_template_paths,
    run_transcriptions_with_concurrency,
    take_next_transcription_segment_for_llm,
    write_output_artifacts,
)
from common_copilot import (
    build_copilot_session_parameters,
    get_copilot_patch_payload_with_repair,
    print_available_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", dest="input_file")
    parser.add_argument("--output-file", dest="output_file")
    parser.add_argument("--patch-prompt-file", dest="patch_prompt_file")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file")
    parser.add_argument("--model", dest="model", default=DEFAULT_COPILOT_MODEL)
    parser.add_argument(
        "--progress-write-every",
        dest="progress_write_every",
        type=int,
        default=None,
        help=(
            "Write incremental output snapshot every N completed items "
            "(default: 1, or 100 when --resume-from-output is enabled)."
        ),
    )
    add_common_runtime_cli_arguments(parser)
    add_model_mismatch_retries_cli_argument(parser)
    parser.add_argument("--list-models-only", dest="list_models_only", action="store_true")
    parser.add_argument("--print-models", dest="print_models", action="store_true")
    parser.add_argument(
        "--chain-steps",
        dest="chain_steps",
        action="append",
        help="Repeatable active-chain selector (ids 1-8 or step names like COMBINE, NO_TOUCH).",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    input_file_value = args.input_file
    output_file_value = args.output_file
    output_as_tsv = bool(input_file_value and str(input_file_value).lower().endswith(".tsv"))

    transcriptions: list[str]
    source_filenames: list[str | None]
    source_rows: list[list[str] | None]
    if args.list_models_only:
        transcriptions = []
        source_filenames = []
        source_rows = []
    else:
        input_data = collect_transcriptions_from_input(input_file_value)
        if input_data is None:
            return
        transcriptions, source_filenames, source_rows = input_data

    model = args.model
    configured_concurrency = args.concurrency
    concurrency = max(1, configured_concurrency)

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
    model_mismatch_retries = max(0, args.model_mismatch_retries)
    chain_steps = [step for step in (args.chain_steps or []) if isinstance(step, str) and step.strip()]
    active_step_keys = resolve_active_chain_step_keys(chain_steps)
    resume_from_output = bool(args.resume_from_output)
    progress_write_every_arg = args.progress_write_every
    if isinstance(progress_write_every_arg, int):
        progress_write_every = max(1, progress_write_every_arg)
    else:
        progress_write_every = 100 if resume_from_output else 1

    prompt_template_path, repair_prompt_template_path, template_error = resolve_patch_and_repair_template_paths(
        args.patch_prompt_file,
        args.repair_prompt_file,
    )
    if template_error:
        print(template_error)
        return
    if prompt_template_path is None or repair_prompt_template_path is None:
        return

    print(f"Using model: {model}")
    if chain_steps:
        print(f"Chain step selector count: {len(chain_steps)}")
    print(f"Resolved active chain: {format_resolved_chain_steps(chain_steps)}")
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

    client = CopilotClient()
    await client.start()

    try:
        should_print_models = (
            args.list_models_only
            or args.print_models
        )
        if should_print_models:
            await print_available_models(client)

        if args.list_models_only:
            return

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

        async def maybe_write_progress_snapshot() -> None:
            nonlocal pending_progress_writes
            if not output_file_value:
                return
            async with progress_write_lock:
                pending_progress_writes += 1
                if pending_progress_writes < progress_write_every:
                    return

                completed_payloads = [payload for payload in payloads if isinstance(payload, dict)]
                if not completed_payloads:
                    return

                pending_progress_writes = 0
                try:
                    write_output_artifacts(
                        completed_payloads,
                        output_file_value,
                        text_output_lines,
                        source_filenames,
                        source_rows,
                        output_as_tsv,
                        active_step_keys,
                    )
                except Exception as error:
                    print(f"Failed to write progress snapshot: {error}")

        async def process_item(index: int, transcription: str, total: int) -> None:
            requested_model = model
            slot = index - 1
            processing_id = f"{index}/{total}"
            if is_input_comment_line(transcription):
                text_output_lines[slot] = transcription
                await maybe_write_progress_snapshot()
                return

            if not transcription.strip():
                payloads[slot] = build_empty_payload()
                payloads[slot]["source_text"] = transcription
                print(
                    f"Input transcription {index}/{total} is empty; "
                    "emitting empty payload."
                )
                await maybe_write_progress_snapshot()
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
                    await maybe_write_progress_snapshot()
                    return

            prompt_transcription, case_normalized = normalize_all_uppercase_input(transcription)
            source_was_all_uppercase = is_all_uppercase_cased_input(transcription)
            source_was_all_lowercase = is_all_lowercase_cased_input(transcription)
            skip_first_token_casing_preservation = source_was_all_uppercase or source_was_all_lowercase
            # if case_normalized:
            #     print(f"[{processing_id}] Normalized all-uppercase input to display casing before prompt.")

            try:
                session_parameters = build_copilot_session_parameters(model)

                async def create_session() -> Any:
                    return await client.create_session(session_parameters)

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

                        prompt = build_patch_prompt(
                            prompt_template,
                            segment_transcription,
                            chain_steps,
                        )

                        failure_reasons: list[str] = []

                        def _on_final_failure(reason: str) -> None:
                            failure_reasons.append(reason)

                        segment_payload = await get_copilot_patch_payload_with_repair(
                            create_session,
                            prompt,
                            segment_transcription,
                            segment_processing_id,
                            requested_model,
                            repair_prompt_template,
                            timeout_seconds,
                            timeout_retries,
                            empty_result_retries,
                            model_mismatch_retries,
                            skip_first_token_casing_preservation,
                            active_step_keys,
                            _on_final_failure,
                            False,
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
                    text_output_lines[slot] = corrected_text if isinstance(corrected_text, str) else ""
                await maybe_write_progress_snapshot()
                return
            except asyncio.CancelledError:
                payloads[slot] = build_empty_payload()
                payloads[slot]["source_text"] = transcription
                text_output_lines[slot] = ""
                print(
                    f"Cancelled while processing transcription {index}/{total}; "
                    "emitting empty payload."
                )
                await maybe_write_progress_snapshot()
                return
            except Exception as error:
                payloads[slot] = build_empty_payload()
                payloads[slot]["source_text"] = transcription
                text_output_lines[slot] = ""
                print(
                    f"Unexpected error on transcription {index}/{total}: {error}; "
                    "emitting empty payload."
                )
                await maybe_write_progress_snapshot()
                return

        await run_transcriptions_with_concurrency(transcriptions, concurrency, process_item)

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

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
