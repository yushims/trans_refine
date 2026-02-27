import argparse
import asyncio
from typing import Any

from copilot import CopilotClient  # pyright: ignore[reportMissingImports]
from common import (
    assign_payload_or_emit_empty,
    build_empty_payload,
    collect_transcriptions_from_input,
    finalize_payloads_and_write,
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
    return parser.parse_args()


async def main():
    args = parse_args()

    input_file_value = args.input_file
    output_file_value = args.output_file

    transcriptions: list[str]
    if args.list_models_only:
        transcriptions = []
    else:
        loaded_transcriptions = collect_transcriptions_from_input(input_file_value)
        if loaded_transcriptions is None:
            return
        transcriptions = loaded_transcriptions

    model = args.model
    configured_concurrency = args.concurrency
    concurrency = max(1, configured_concurrency)

    timeout_seconds = args.timeout
    timeout_retries = max(0, args.timeout_retries)
    empty_result_retries = max(0, args.empty_result_retries)
    model_mismatch_retries = max(0, args.model_mismatch_retries)

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

        async def process_item(index: int, transcription: str, total: int) -> None:
            requested_model = model
            slot = index - 1
            processing_id = f"{index}/{total}"
            if not transcription.strip():
                payloads[slot] = build_empty_payload()
                print(
                    f"Input transcription {index}/{total} is empty; "
                    "emitting empty payload."
                )
                return

            prompt = prompt_template + transcription
            try:
                session_parameters = build_copilot_session_parameters(model)

                async def create_session() -> Any:
                    return await client.create_session(session_parameters)

                payload = await get_copilot_patch_payload_with_repair(
                    create_session,
                    prompt,
                    transcription,
                    processing_id,
                    requested_model,
                    repair_prompt_template,
                    timeout_seconds,
                    timeout_retries,
                    empty_result_retries,
                    model_mismatch_retries,
                )

                assign_payload_or_emit_empty(payload, payloads, slot, index, total)
                return
            except asyncio.CancelledError:
                payloads[slot] = build_empty_payload()
                print(
                    f"Cancelled while processing transcription {index}/{total}; "
                    "emitting empty payload."
                )
                return
            except Exception as error:
                payloads[slot] = build_empty_payload()
                print(
                    f"Unexpected error on transcription {index}/{total}: {error}; "
                    "emitting empty payload."
                )
                return

        await run_transcriptions_with_concurrency(transcriptions, concurrency, process_item)

        if not finalize_payloads_and_write(payloads, output_file_value):
            return

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
