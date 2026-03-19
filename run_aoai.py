import argparse
import asyncio
import os

from openai import AzureOpenAI
from dotenv import load_dotenv
from common import (
    DEFAULT_AOAI_API_VERSION,
    DEFAULT_AOAI_DEPLOYMENT,
    DEFAULT_AOAI_ENDPOINT,
    add_aoai_sampling_cli_arguments,
    add_common_runtime_cli_arguments,
    assign_payload_or_emit_empty,
    build_patch_prompt,
    build_patch_response_format_schema,
    build_empty_payload,
    collect_transcriptions_from_input,
    finalize_payloads_and_write,
    format_resolved_chain_steps,
    is_all_lowercase_cased_input,
    is_all_uppercase_cased_input,
    normalize_all_uppercase_input,
    is_input_comment_line,
    load_patch_and_repair_templates,
    print_common_runtime_settings,
    resolve_patch_and_repair_template_paths,
    run_transcriptions_with_concurrency,
    write_output_artifacts,
)
from common_aoai import get_patch_payload_with_repair

PATCH_SCHEMA = build_patch_response_format_schema()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", dest="input_file")
    parser.add_argument("--output-file", dest="output_file")
    parser.add_argument("--patch-prompt-file", dest="patch_prompt_file")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file")
    parser.add_argument("--deployment", dest="deployment", default=DEFAULT_AOAI_DEPLOYMENT)
    parser.add_argument("--endpoint", dest="endpoint", default=DEFAULT_AOAI_ENDPOINT)
    parser.add_argument("--api-version", dest="api_version", default=DEFAULT_AOAI_API_VERSION)
    parser.add_argument(
        "--progress-write-every",
        dest="progress_write_every",
        type=int,
        default=1,
        help="Write incremental output snapshot every N completed items (default: 1).",
    )
    add_common_runtime_cli_arguments(parser)
    add_aoai_sampling_cli_arguments(parser)
    parser.add_argument(
        "--chain-steps",
        dest="chain_steps",
        action="append",
        help="Repeatable active-chain selector (ids 1-8 or step names like COMBINE, NO_TOUCH).",
    )
    return parser.parse_args()


async def main() -> None:
    load_dotenv()

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

    temperature = args.temperature
    top_p = args.top_p
    retry_temperature_jitter = max(0.0, args.retry_temperature_jitter)
    retry_top_p_jitter = max(0.0, args.retry_top_p_jitter)
    chain_steps = [step for step in (args.chain_steps or []) if isinstance(step, str) and step.strip()]
    progress_write_every = max(1, args.progress_write_every)

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

    payloads: list[dict | None] = [None] * len(transcriptions)
    text_output_lines: list[str] = [""] * len(transcriptions)
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
                )
            except Exception as error:
                print(f"Failed to write progress snapshot: {error}")

    async def process_item(index: int, transcription: str, total: int) -> None:
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

        prompt_transcription, case_normalized = normalize_all_uppercase_input(transcription)
        source_was_all_uppercase = is_all_uppercase_cased_input(transcription)
        source_was_all_lowercase = is_all_lowercase_cased_input(transcription)
        skip_first_token_casing_preservation = source_was_all_uppercase or source_was_all_lowercase
        # if case_normalized:
        #     print(f"[{processing_id}] Normalized all-uppercase input to display casing before prompt.")

        prompt = build_patch_prompt(
            prompt_template,
            prompt_transcription,
            chain_steps,
        )
        try:
            payload = await get_patch_payload_with_repair(
                client=client,
                deployment=deployment,
                prompt=prompt,
                transcription=prompt_transcription,
                processing_id=processing_id,
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
            )

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
    ):
        return


if __name__ == "__main__":
    asyncio.run(main())
