import argparse
import asyncio
from typing import Any

from copilot import CopilotClient  # pyright: ignore[reportMissingImports]
from common import (
    assign_payload_or_emit_empty,
    build_patch_prompt,
    build_empty_payload,
    collect_transcriptions_from_input,
    finalize_payloads_and_write,
    is_all_lowercase_cased_input,
    is_all_uppercase_cased_input,
    normalize_char_based_spacing_input,
    normalize_all_uppercase_input,
    is_input_comment_line,
    load_patch_and_repair_templates,
    print_common_runtime_settings,
    resolve_patch_and_repair_template_paths,
    run_transcriptions_with_concurrency,
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
    parser.add_argument("--model", dest="model", default="gpt-5.2")
    parser.add_argument("--concurrency", dest="concurrency", type=int, default=10)
    parser.add_argument("--timeout", dest="timeout", type=float, default=600.0)
    parser.add_argument("--timeout-retries", dest="timeout_retries", type=int, default=2)
    parser.add_argument("--empty-result-retries", dest="empty_result_retries", type=int, default=2)
    parser.add_argument("--model-mismatch-retries", dest="model_mismatch_retries", type=int, default=2)
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

    transcriptions: list[str]
    if args.list_models_only:
        transcriptions = []
    else:
        transcriptions = collect_transcriptions_from_input(input_file_value)
        if transcriptions is None:
            return

    model = args.model
    configured_concurrency = args.concurrency
    concurrency = max(1, configured_concurrency)

    timeout_seconds = args.timeout
    timeout_retries = max(0, args.timeout_retries)
    empty_result_retries = max(0, args.empty_result_retries)
    model_mismatch_retries = max(0, args.model_mismatch_retries)
    chain_steps = [step for step in (args.chain_steps or []) if isinstance(step, str) and step.strip()]

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
    print_common_runtime_settings(
        prompt_template_path,
        repair_prompt_template_path,
        concurrency,
        timeout_seconds,
        timeout_retries,
        empty_result_retries,
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

        async def process_item(index: int, transcription: str, total: int) -> None:
            requested_model = model
            slot = index - 1
            processing_id = f"{index}/{total}"
            if is_input_comment_line(transcription):
                text_output_lines[slot] = transcription
                return

            if not transcription.strip():
                payloads[slot] = build_empty_payload()
                print(
                    f"Input transcription {index}/{total} is empty; "
                    "emitting empty payload."
                )
                return

            prompt_transcription, spacing_normalized = normalize_char_based_spacing_input(transcription)
            prompt_transcription, case_normalized = normalize_all_uppercase_input(prompt_transcription)
            source_was_all_uppercase = is_all_uppercase_cased_input(transcription)
            source_was_all_lowercase = is_all_lowercase_cased_input(transcription)
            skip_first_token_casing_preservation = source_was_all_uppercase or source_was_all_lowercase
            if spacing_normalized:
                print(f"[{processing_id}] Normalized char-based intra-script spacing artifacts before prompt.")
            if case_normalized:
                print(f"[{processing_id}] Normalized all-uppercase input to display casing before prompt.")

            prompt = build_patch_prompt(
                prompt_template,
                prompt_transcription,
                chain_steps,
            )
            try:
                session_parameters = build_copilot_session_parameters(model)

                async def create_session() -> Any:
                    return await client.create_session(session_parameters)

                payload = await get_copilot_patch_payload_with_repair(
                    create_session,
                    prompt,
                    prompt_transcription,
                    processing_id,
                    requested_model,
                    repair_prompt_template,
                    timeout_seconds,
                    timeout_retries,
                    empty_result_retries,
                    model_mismatch_retries,
                    skip_first_token_casing_preservation,
                )

                assign_payload_or_emit_empty(payload, payloads, slot, index, total)
                resolved_payload = payloads[slot]
                if isinstance(resolved_payload, dict):
                    corrected_text = resolved_payload.get("corrected_text")
                    text_output_lines[slot] = corrected_text if isinstance(corrected_text, str) else ""
                return
            except asyncio.CancelledError:
                payloads[slot] = build_empty_payload()
                text_output_lines[slot] = ""
                print(
                    f"Cancelled while processing transcription {index}/{total}; "
                    "emitting empty payload."
                )
                return
            except Exception as error:
                payloads[slot] = build_empty_payload()
                text_output_lines[slot] = ""
                print(
                    f"Unexpected error on transcription {index}/{total}: {error}; "
                    "emitting empty payload."
                )
                return

        await run_transcriptions_with_concurrency(transcriptions, concurrency, process_item)

        if not finalize_payloads_and_write(payloads, output_file_value, text_output_lines):
            return

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
