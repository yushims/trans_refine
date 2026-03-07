import argparse
import asyncio

from openai import AzureOpenAI
from common import (
    assign_payload_or_emit_empty,
    build_patch_prompt,
    build_patch_response_format_schema,
    build_empty_payload,
    collect_transcriptions_from_input,
    finalize_payloads_and_write,
    is_input_comment_line,
    load_patch_and_repair_templates,
    print_common_runtime_settings,
    resolve_patch_and_repair_template_paths,
    run_transcriptions_with_concurrency,
)
from common_aoai import get_patch_payload_with_repair

PATCH_SCHEMA = build_patch_response_format_schema()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", dest="input_file")
    parser.add_argument("--output-file", dest="output_file")
    parser.add_argument("--patch-prompt-file", dest="patch_prompt_file")
    parser.add_argument("--repair-prompt-file", dest="repair_prompt_file")
    parser.add_argument("--deployment", dest="deployment", default="gpt-5-chat")
    parser.add_argument("--endpoint", dest="endpoint", default="https://adaptationdev-resource.openai.azure.com/")
    parser.add_argument("--api-version", dest="api_version", default="2025-01-01-preview")
    parser.add_argument("--concurrency", dest="concurrency", type=int, default=10)
    parser.add_argument("--timeout", dest="timeout", type=float, default=600.0)
    parser.add_argument("--timeout-retries", dest="timeout_retries", type=int, default=2)
    parser.add_argument("--empty-result-retries", dest="empty_result_retries", type=int, default=2)
    parser.add_argument("--temperature", dest="temperature", type=float, default=0.0)
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--retry-temperature-jitter", dest="retry_temperature_jitter", type=float, default=0.08)
    parser.add_argument("--retry-top-p-jitter", dest="retry_top_p_jitter", type=float, default=0.03)
    parser.add_argument(
        "--chain-steps",
        dest="chain_steps",
        action="append",
        help="Repeatable active-chain selector (ids 1-8 or step names like COMBINE, NO_TOUCH).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    input_file_value = args.input_file
    output_file_value = args.output_file

    transcriptions = collect_transcriptions_from_input(input_file_value)
    if transcriptions is None:
        return

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

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    payloads: list[dict | None] = [None] * len(transcriptions)
    text_output_lines: list[str] = [""] * len(transcriptions)

    async def process_item(index: int, transcription: str, total: int) -> None:
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

        prompt = build_patch_prompt(
            prompt_template,
            transcription,
            chain_steps,
        )
        try:
            payload = await get_patch_payload_with_repair(
                client=client,
                deployment=deployment,
                prompt=prompt,
                transcription=transcription,
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
            )

            assign_payload_or_emit_empty(payload, payloads, slot, index, total)
            resolved_payload = payloads[slot]
            if isinstance(resolved_payload, dict):
                corrected_text = resolved_payload.get("corrected_text")
                text_output_lines[slot] = corrected_text if isinstance(corrected_text, str) else ""
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


if __name__ == "__main__":
    asyncio.run(main())
