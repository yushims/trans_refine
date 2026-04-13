"""Cancel active Azure OpenAI batch jobs."""

import argparse
import datetime
import sys
from collections import defaultdict

from openai import AzureOpenAI
from dotenv import load_dotenv
from common import DEFAULT_AOAI_API_VERSION, DEFAULT_AOAI_ENDPOINT


ACTIVE_STATUSES = {"validating", "in_progress", "finalizing"}
ALL_STATUSES = {*ACTIVE_STATUSES, "completed", "failed", "cancelled", "expired", "cancelling"}


def _fmt_ts(epoch: int | None) -> str:
    if epoch is None:
        return "-"
    return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_counts(b) -> str:
    rc = b.request_counts
    if rc is None:
        return "counts=n/a"
    return f"total={rc.total} completed={rc.completed} failed={rc.failed}"


def _print_batch(b) -> None:
    print(f"  {b.id}")
    print(f"    status:    {b.status}")
    print(f"    endpoint:  {b.endpoint or '-'}")
    print(f"    created:   {_fmt_ts(b.created_at)}")
    if b.in_progress_at:
        print(f"    started:   {_fmt_ts(b.in_progress_at)}")
    if b.completed_at:
        print(f"    completed: {_fmt_ts(b.completed_at)}")
    if b.failed_at:
        print(f"    failed:    {_fmt_ts(b.failed_at)}")
    if b.expires_at:
        print(f"    expires:   {_fmt_ts(b.expires_at)}")
    print(f"    requests:  {_fmt_counts(b)}")
    if b.errors and b.errors.data:
        for err in b.errors.data[:3]:
            print(f"    error:     {err.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List and cancel active Azure OpenAI batch jobs.")
    parser.add_argument("--endpoint", default=DEFAULT_AOAI_ENDPOINT)
    parser.add_argument("--api-version", default=DEFAULT_AOAI_API_VERSION)
    parser.add_argument("--cancel", action="store_true", help="Actually cancel matched active jobs. Without this flag, only lists jobs (dry run).")
    parser.add_argument("--all", action="store_true", help="Show all batch jobs, not just active ones.")
    parser.add_argument("--user", dest="user_filter", help="Only match jobs whose metadata user contains this substring.")
    parser.add_argument("--input", dest="input_filter", help="Only match jobs whose metadata input_file contains this substring.")
    parser.add_argument("--status", dest="status_filter", help="Only match jobs with this status (e.g. cancelling, validating, in_progress).")
    parser.add_argument("--id", dest="batch_id", help="Cancel a specific batch job by ID.")
    parser.add_argument("--days", dest="days", type=float, default=None, help="Only match jobs created within the last N days (e.g. 1, 0.5, 7).")
    args = parser.parse_args()

    load_dotenv()
    client = AzureOpenAI(azure_endpoint=args.endpoint, api_version=args.api_version)

    # Cancel a specific batch by ID.
    if args.batch_id:
        try:
            b = client.batches.retrieve(args.batch_id)
            _print_batch(b)
            if not args.cancel:
                print("\nUse --cancel to actually cancel this job.")
                return
            if b.status in ACTIVE_STATUSES:
                client.batches.cancel(args.batch_id)
                print(f"\nCancelled {args.batch_id}")
            else:
                print(f"\nJob is already {b.status}, cannot cancel.")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
        return

    import time
    cutoff_ts = (time.time() - args.days * 86400) if args.days is not None else None

    if args.status_filter:
        target_statuses = {args.status_filter.lower()}
    elif args.all:
        target_statuses = ALL_STATUSES
    else:
        target_statuses = ACTIVE_STATUSES
    matched = []
    for batch in client.batches.list():
        if batch.status not in target_statuses:
            continue
        if cutoff_ts is not None and batch.created_at and batch.created_at < cutoff_ts:
            continue
        meta = batch.metadata if isinstance(batch.metadata, dict) else {}
        if args.input_filter:
            input_file = meta.get("input_file", "")
            if args.input_filter.lower() not in input_file.lower():
                continue
        if args.user_filter:
            user = meta.get("user", "")
            if args.user_filter.lower() not in user.lower():
                continue
        matched.append(batch)

    if not matched:
        print("No matching batch jobs found.")
        return

    label = args.status_filter if args.status_filter else ("all" if args.all else "active")

    # Group by (user, input_file) for readability.
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for b in matched:
        meta = b.metadata if isinstance(b.metadata, dict) else {}
        key = (meta.get("user", "<unknown>"), meta.get("input_file", "<unknown>"))
        groups[key].append(b)

    for (user, input_file), batches in groups.items():
        print(f"  user: {user}  |  input: {input_file}  ({len(batches)} job(s))")
        print(f"  {'─' * 60}")
        for b in batches:
            _print_batch(b)
            print()

    print(f"Found {len(matched)} {label} batch job(s).")
    print()

    if not args.cancel or args.all:
        return

    active = [b for b in matched if b.status in ACTIVE_STATUSES]
    if not active:
        print("No active jobs to cancel.")
        return

    for b in active:
        try:
            client.batches.cancel(b.id)
            print(f"  Cancelled {b.id}")
        except Exception as e:
            print(f"  Failed to cancel {b.id}: {e}", file=sys.stderr)

    print(f"\nDone. Cancelled {len(active)} job(s).")


if __name__ == "__main__":
    main()
