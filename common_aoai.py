import asyncio
import json
import os
import random
import tempfile
import time
from collections.abc import Callable

from common import (
    DEFAULT_MAX_INPUT_CHARS_PER_CALL,
    extract_retry_after_seconds,
    extract_text_content,
    get_patch_payload_with_repair_generic,
    merge_segment_payloads,
    parse_validate_and_apply_text_fixes,
    run_with_timeout_retry,
    take_next_transcription_segment_for_llm,
)


async def aoai_send_once(
    client: object,
    deployment: str,
    prompt: str,
    schema: dict,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
) -> str:
    request_kwargs = {
        "model": deployment,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": schema,
        },
        "temperature": temperature,
        "top_p": top_p,
    }

    completion = await asyncio.wait_for(
        asyncio.to_thread(client.chat.completions.create, **request_kwargs),
        timeout=timeout_seconds,
    )

    choices = getattr(completion, "choices", None)
    if not choices:
        return ""

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    return extract_text_content(content).strip()


def is_aoai_rate_limited_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    message = str(error).lower()
    return (
        "ratelimitreached" in message
        or "rate limit" in message
        or "error code: 429" in message
    )


async def aoai_send_with_timeout_retry(
    client: object,
    deployment: str,
    prompt: str,
    schema: dict,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
    timeout_retries: int,
    processing_id: str | None = None,
) -> str | None:
    async def operation() -> str:
        return await aoai_send_once(
            client,
            deployment,
            prompt,
            schema,
            temperature,
            top_p,
            timeout_seconds,
        )

    return await run_with_timeout_retry(
        operation,
        timeout_retries,
        processing_id,
        is_retryable_error=is_aoai_rate_limited_error,
        resolve_backoff_seconds=lambda error, _attempt: extract_retry_after_seconds(error),
    )


async def get_patch_payload_with_repair(
    client: object,
    deployment: str,
    prompt: str,
    transcription: str,
    processing_id: str,
    repair_prompt_template: str,
    patch_schema: dict,
    temperature: float,
    top_p: float,
    timeout_seconds: float,
    timeout_retries: int,
    empty_result_retries: int,
    retry_temperature_jitter: float,
    retry_top_p_jitter: float,
    skip_first_token_casing_preservation: bool = False,
    active_step_keys: set[str] | None = None,
    on_final_failure: Callable[[str], None] | None = None,
) -> dict | None:
    attempt_temperatures: list[float] = []
    attempt_top_ps: list[float] = []

    for empty_attempt in range(empty_result_retries + 1):
        attempt_temperature = temperature
        attempt_top_p = top_p
        if empty_attempt > 0:
            attempt_temperature = max(
                0.0,
                min(1.0, temperature + random.uniform(0.0, retry_temperature_jitter)),
            )
            attempt_top_p = max(
                0.0,
                min(1.0, top_p + random.uniform(-retry_top_p_jitter, retry_top_p_jitter)),
            )
            # print(
            #     f"[{processing_id}] Retry decode jitter applied: temperature={attempt_temperature:.3f}, "
            #     f"top_p={attempt_top_p:.3f}"
            # )
        attempt_temperatures.append(attempt_temperature)
        attempt_top_ps.append(attempt_top_p)

    async def _send_prompt(attempt_prompt: str, empty_attempt: int) -> str | None:
        return await aoai_send_with_timeout_retry(
            client,
            deployment,
            attempt_prompt,
            patch_schema,
            attempt_temperatures[empty_attempt],
            attempt_top_ps[empty_attempt],
            timeout_seconds,
            timeout_retries,
            processing_id,
        )

    async def _send_repair_prompt(repair_prompt: str, empty_attempt: int) -> str | None:
        return await aoai_send_with_timeout_retry(
            client,
            deployment,
            repair_prompt,
            patch_schema,
            attempt_temperatures[empty_attempt],
            attempt_top_ps[empty_attempt],
            timeout_seconds,
            timeout_retries,
            processing_id,
        )

    return await get_patch_payload_with_repair_generic(
        prompt=prompt,
        transcription=transcription,
        processing_id=processing_id,
        repair_prompt_template=repair_prompt_template,
        target_schema=json.dumps(patch_schema, ensure_ascii=False, indent=2),
        timeout_seconds=timeout_seconds,
        timeout_retries=timeout_retries,
        empty_result_retries=empty_result_retries,
        send_prompt=_send_prompt,
        send_repair_prompt=_send_repair_prompt,
        skip_first_token_casing_preservation=skip_first_token_casing_preservation,
        active_step_keys=active_step_keys,
        on_final_failure=on_final_failure,
        repair_timeout_message="Repair returned empty output or timed out.",
        repair_empty_message="Repair returned empty output or timed out.",
    )


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

BATCH_DEFAULT_SIZE = 10000  # 1000: ~2-5MB, 5000: ~10-25 MB, 10000: ~20-50 MB, 20000: ~40-100 MB, 50000: 100+ MB
BATCH_POLL_INTERVAL_SECONDS = 1800  # 30 minutes


def _call_with_retry(func, *args, retries: int = 5, **kwargs):
    """Call *func* with simple retry logic and linear backoff."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            delay = min(2 ** attempt, 30)
            print(f"  Retry {attempt}/{retries} after error: {exc}  (backoff {delay}s)")
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Call failed without exception details.")


def cleanup_stale_batch_resources(client) -> None:
    """Cancel non-terminal batch jobs and delete all uploaded files.

    Call this before starting a new batch run to avoid orphan resources
    from previously crashed runs.
    """
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}

    # Cancel pending batch jobs.
    try:
        for batch in client.batches.list():
            status = getattr(batch, "status", None)
            if status in terminal_statuses:
                continue
            try:
                resp = client.batches.cancel(batch.id)
                print(f"  Cancelled stale batch job: {batch.id} (was {status}, now {resp.status})")
            except Exception as exc:
                print(f"  Could not cancel batch {batch.id}: {exc}")
    except Exception as exc:
        print(f"  Could not list batch jobs for cleanup: {exc}")

    # Delete orphan files.
    try:
        for f in client.files.list():
            try:
                deleted = client.files.delete(f.id)
                print(f"  Deleted file: {f.id} (deleted={getattr(deleted, 'deleted', True)})")
            except Exception as exc:
                print(f"  Could not delete file {f.id}: {exc}")
    except Exception as exc:
        print(f"  Could not list files for cleanup: {exc}")


async def run_batch_pipeline(
    client,
    deployment: str,
    transcriptions: list[str],
    prompt_template: str,
    chain_steps: list[str] | None,
    locale: str | None,
    schema: dict,
    temperature: float,
    top_p: float,
    batch_size: int,
    build_patch_prompt_fn: Callable,
    skip_first_token_casing_preservation_flags: list[bool],
    active_step_keys: set[str] | None = None,
    max_retry_passes: int = 2,
    max_input_chars_per_call: int = DEFAULT_MAX_INPUT_CHARS_PER_CALL,
    retry_temperature_jitter: float = 0.08,
    concurrency: int = 1,
    poll_interval_seconds: int = BATCH_POLL_INTERVAL_SECONDS,
    pre_resolved_indices: set[int] | None = None,
    on_pass_complete: Callable[[list], None] | None = None,
) -> list[dict | None]:
    """Run the full batch pipeline: partition, submit, parse results.

    Failed items are collected and resubmitted in up to *max_retry_passes*
    follow-up batches with temperature jitter and input segmentation for
    long transcriptions.  Returns a list parallel to *transcriptions* with
    parsed payloads (or ``None`` for items that still failed).
    """
    total = len(transcriptions)
    payloads: list[dict | None] = [None] * total
    resolved_indices = set(pre_resolved_indices) if pre_resolved_indices else set()

    # Clean up stale batch jobs and orphan files from previous runs.
    print("Batch mode: cleaning up stale resources...")
    await asyncio.to_thread(cleanup_stale_batch_resources, client)

    # Build (index, transcription) items for non-empty inputs.
    items: list[tuple[int, str]] = []
    for idx, text in enumerate(transcriptions):
        if idx in resolved_indices:
            continue  # Skip items already completed from resume.
        if text.strip():
            items.append((idx, text))
        else:
            from common import build_empty_payload
            payloads[idx] = build_empty_payload()
            payloads[idx]["source_text"] = text

    if not items:
        return payloads

    pending_items = list(items)

    for pass_num in range(1, max_retry_passes + 2):  # pass 1 = initial, 2..N+1 = retries
        if not pending_items:
            break

        is_retry = pass_num > 1
        pass_label = "" if not is_retry else f" (retry {pass_num - 1})"

        # Apply temperature jitter on retries.
        pass_temperature = temperature
        if is_retry and retry_temperature_jitter > 0:
            pass_temperature = max(
                0.0,
                min(1.0, temperature + random.uniform(0.0, retry_temperature_jitter)),
            )
            print(f"Batch mode{pass_label}: temperature jitter applied: {pass_temperature:.3f}")

        # On retries, segment failed items by halving their input length.
        # Pass 1: use max_input_chars_per_call for very long inputs.
        # Retry 1: half the input length. Retry 2: quarter, etc.
        # batch_entries: list of (custom_key, original_idx, text_to_send, prev_ctx, next_ctx)
        batch_entries: list[tuple[str, int, str, str, str]] = []
        # segment_groups tracks which original items were split.
        segment_groups: dict[int, list[tuple[str, str]]] = {}

        for idx, transcription in pending_items:
            if is_retry:
                # Halve progressively: retry 1 = len/2, retry 2 = len/4, etc.
                retry_segment_limit = max(1, len(transcription) // (2 ** (pass_num - 1)))
            elif max_input_chars_per_call > 0 and len(transcription) > max_input_chars_per_call:
                retry_segment_limit = max_input_chars_per_call
            else:
                retry_segment_limit = 0

            should_segment = retry_segment_limit > 0 and len(transcription) > retry_segment_limit

            if should_segment:
                segments: list[str] = []
                start_offset = 0
                while start_offset < len(transcription):
                    seg, _ = take_next_transcription_segment_for_llm(
                        transcription, start_offset, retry_segment_limit,
                    )
                    if not seg:
                        seg = transcription[start_offset:start_offset + 1]
                    segments.append(seg)
                    start_offset += len(seg)

                if len(segments) > 1:
                    _CTX_SNIPPET_LEN = 50
                    seg_keys: list[tuple[str, str]] = []
                    for seg_num, seg_text in enumerate(segments):
                        custom_key = f"{idx}_s{seg_num}"
                        prev_ctx = segments[seg_num - 1][-_CTX_SNIPPET_LEN:] if seg_num > 0 else ""
                        next_ctx = segments[seg_num + 1][:_CTX_SNIPPET_LEN] if seg_num < len(segments) - 1 else ""
                        batch_entries.append((custom_key, idx, seg_text, prev_ctx, next_ctx))
                        seg_keys.append((custom_key, seg_text))
                    segment_groups[idx] = seg_keys
                    print(
                        f"  [{idx + 1}/{total}] Segmented into {len(segments)} parts "
                        f"(limit={retry_segment_limit}, input={len(transcription)} chars)"
                    )
                    continue

            # Single item (no segmentation).
            custom_key = str(idx)
            batch_entries.append((custom_key, idx, transcription, "", ""))

        # Build JSONL request lines with custom keys.
        all_jsonl_lines: list[str] = []
        for custom_key, _orig_idx, text, prev_ctx, next_ctx in batch_entries:
            prompt = build_patch_prompt_fn(
                prompt_template, text, chain_steps, locale=locale,
                prev_context=prev_ctx if prev_ctx else None,
                next_context=next_ctx if next_ctx else None,
            )
            request = {
                "custom_id": f"idx-{custom_key}",
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": deployment,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {
                        "type": "json_object",
                    },
                    "temperature": pass_temperature,
                    "top_p": top_p,
                },
            }
            all_jsonl_lines.append(json.dumps(request, ensure_ascii=False))

        # Submit in partitions and collect all results.
        all_raw_results: dict[str, str] = {}

        partition_starts = list(range(0, len(all_jsonl_lines), batch_size))
        num_partitions = len(partition_starts)

        print(
            f"Batch mode{pass_label}: {len(pending_items)} items "
            f"({len(batch_entries)} requests) in {num_partitions} partition(s) "
            f"(concurrency={concurrency})"
        )

        # Submit partitions with controlled concurrency.
        semaphore = asyncio.Semaphore(max(1, concurrency))
        results_lock = asyncio.Lock()

        async def _submit_partition(part_num: int, start: int) -> None:
            async with semaphore:
                batch_label = f" {part_num}/{num_partitions}{pass_label}"
                part_lines = all_jsonl_lines[start : start + batch_size]

                raw_results = await _submit_batch_and_wait_keyed(
                    client, part_lines, batch_label, poll_interval_seconds,
                )
                print(f"[batch{batch_label}] Received {len(raw_results)}/{len(part_lines)} results")
                async with results_lock:
                    all_raw_results.update(raw_results)

        await asyncio.gather(*[
            _submit_partition(part_num, start)
            for part_num, start in enumerate(partition_starts, 1)
        ])

        # Process non-segmented items.
        for custom_key, orig_idx, text, _prev_ctx, _next_ctx in batch_entries:
            if orig_idx in segment_groups:
                continue  # Handle below.

            raw_content = all_raw_results.get(custom_key)
            if raw_content is None:
                print(f"  [{orig_idx + 1}/{total}] No response from batch API; skipping.")
                continue

            payload, validation_error, _ = parse_validate_and_apply_text_fixes(
                raw_content,
                text,
                processing_id=f"{orig_idx + 1}/{total}",
                skip_first_token_casing_preservation=skip_first_token_casing_preservation_flags[orig_idx],
                active_step_keys=active_step_keys,
            )

            if payload is not None:
                corrected_text = payload.get("corrected_text")
                if isinstance(corrected_text, str) and corrected_text.strip():
                    payloads[orig_idx] = payload
                else:
                    print(f"  [{orig_idx + 1}/{total}] Empty corrected_text; skipping.")
            else:
                print(f"  [{orig_idx + 1}/{total}] Validation failed: {validation_error}")

        # Process segmented items: parse each segment, then merge.
        for orig_idx, seg_entries in segment_groups.items():
            seg_payloads: list[dict] = []
            seg_sources: list[str] = []
            all_segments_ok = True

            for custom_key, seg_source in seg_entries:
                raw_content = all_raw_results.get(custom_key)
                if raw_content is None:
                    print(
                        f"  [{orig_idx + 1}/{total}] Segment {custom_key}: "
                        "no response from batch API."
                    )
                    all_segments_ok = False
                    break

                seg_payload, validation_error, _ = parse_validate_and_apply_text_fixes(
                    raw_content,
                    seg_source,
                    processing_id=f"{orig_idx + 1}/{total} seg",
                    skip_first_token_casing_preservation=skip_first_token_casing_preservation_flags[orig_idx],
                    active_step_keys=active_step_keys,
                )

                if seg_payload is not None:
                    corrected = seg_payload.get("corrected_text")
                    if isinstance(corrected, str) and corrected.strip():
                        seg_payloads.append(seg_payload)
                        seg_sources.append(seg_source)
                    else:
                        print(
                            f"  [{orig_idx + 1}/{total}] Segment {custom_key}: "
                            "empty corrected_text."
                        )
                        all_segments_ok = False
                        break
                else:
                    print(
                        f"  [{orig_idx + 1}/{total}] Segment {custom_key}: "
                        f"validation failed: {validation_error}"
                    )
                    all_segments_ok = False
                    break

            if all_segments_ok and seg_payloads:
                merged = merge_segment_payloads(seg_payloads, seg_sources)
                if merged is not None:
                    payloads[orig_idx] = merged
                    print(f"  [{orig_idx + 1}/{total}] Merged {len(seg_payloads)} segments successfully.")
                else:
                    print(f"  [{orig_idx + 1}/{total}] Segment merge failed.")

        # Write progress snapshot so resume can pick up from here.
        if on_pass_complete is not None:
            try:
                on_pass_complete(payloads)
            except Exception as cb_err:
                print(f"Batch mode: progress snapshot error: {cb_err}")

        # Collect failed items for retry.
        failed_items = [(idx, text) for idx, text in pending_items if payloads[idx] is None]
        if not failed_items:
            break

        if pass_num > max_retry_passes:
            print(f"Batch mode: {len(failed_items)} items still failed after {max_retry_passes} retry pass(es).")
            break

        print(f"Batch mode: {len(failed_items)} failed items; scheduling retry pass {pass_num}...")
        pending_items = failed_items

    return payloads


async def _submit_batch_and_wait_keyed(
    client,
    jsonl_lines: list[str],
    batch_label: str = "",
    poll_interval_seconds: int = BATCH_POLL_INTERVAL_SECONDS,
) -> dict[str, str]:
    """Like submit_batch_and_wait but returns results keyed by the custom_id
    suffix (the part after 'idx-') as a string, supporting non-integer keys
    like '3_s0' for segmented items.
    """
    file_ids: list[str] = []
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
        ) as fh:
            fh.write("\n".join(jsonl_lines))
            tmp_path = fh.name

        print(f"[batch{batch_label}] Uploading {len(jsonl_lines)} requests \u2026")
        with open(tmp_path, "rb") as upload_fh:
            batch_file = _call_with_retry(
                client.files.create, file=upload_fh, purpose="batch",
            )
        file_ids.append(batch_file.id)

        batch_job = _call_with_retry(
            client.batches.create,
            input_file_id=batch_file.id,
            endpoint="/chat/completions",
            completion_window="24h",
        )
        print(f"[batch{batch_label}] Submitted job {batch_job.id}")

        while True:
            status = await asyncio.to_thread(
                _call_with_retry, client.batches.retrieve, batch_job.id,
            )
            print(f"[batch{batch_label}] Status: {status.status}")
            if status.status in ("completed", "failed", "cancelled", "expired"):
                break
            await asyncio.sleep(poll_interval_seconds)

        if status.output_file_id:
            file_ids.append(status.output_file_id)
        if status.error_file_id:
            file_ids.append(status.error_file_id)

        if status.status != "completed":
            print(f"[batch{batch_label}] Job ended with status: {status.status}")
            if status.error_file_id:
                try:
                    err_text = _call_with_retry(
                        client.files.content, status.error_file_id,
                    ).text
                    print(f"[batch{batch_label}] Error details:\n{err_text[:2000]}")
                except Exception:
                    pass
            return {}

        result_text = await asyncio.to_thread(
            lambda: _call_with_retry(client.files.content, status.output_file_id).text,
        )

        results: dict[str, str] = {}
        for line in result_text.strip().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            custom_id = row["custom_id"]
            key = custom_id.split("idx-", 1)[1] if "idx-" in custom_id else custom_id
            try:
                content = row["response"]["body"]["choices"][0]["message"]["content"]
                results[key] = content
            except (KeyError, IndexError, TypeError) as exc:
                print(f"[batch{batch_label}] Could not extract content for {key}: {exc}")
        return results

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        for fid in file_ids:
            try:
                _call_with_retry(client.files.delete, fid)
            except Exception:
                pass
