#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_tsv_value(value_str):
    if not isinstance(value_str, str):
        return str(value_str)

    value_str = value_str.replace('\n', ' ')
    value_str = value_str.replace('\r', ' ')
    value_str = value_str.replace('\t', ' ')
    value_str = ' '.join(value_str.split())
    return value_str.strip()


def parse_result_payload(result_value):
    if isinstance(result_value, dict):
        return result_value
    if isinstance(result_value, str):
        try:
            parsed = json.loads(result_value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def extract_response_text(response_value, llm_name=None):
    payload = parse_result_payload(response_value)
    if payload:
        if llm_name and llm_name in payload:
            return str(payload.get(llm_name, ''))

        for key, value in payload.items():
            if key == 'ter_sent_detail':
                continue
            if isinstance(value, (str, int, float, bool)):
                return str(value)

        return ''

    if isinstance(response_value, str):
        return response_value
    return str(response_value)


def extract_ter_metrics_from_sent_detail(sent_detail):
    if not isinstance(sent_detail, dict):
        sent_detail = {}

    ter_info = sent_detail.get('ter_info', {}) if isinstance(sent_detail, dict) else {}
    ter_category_info = sent_detail.get('ter_category_info', {}) if isinstance(sent_detail, dict) else {}
    ter_categories = ter_category_info.get('ter_categories', {}) if isinstance(ter_category_info, dict) else {}

    def category_edits(category_name):
        category_info = ter_categories.get(category_name, {}) if isinstance(ter_categories, dict) else {}
        if isinstance(category_info, dict):
            return category_info.get('number_of_edits', '')
        return ''

    return {
        'ter_info_number_of_tokens': ter_info.get('number_of_tokens', '') if isinstance(ter_info, dict) else '',
        'ter_info_number_of_edits': ter_info.get('number_of_edits', '') if isinstance(ter_info, dict) else '',
        'ter_info_display_ter': ter_info.get('display_ter', '') if isinstance(ter_info, dict) else '',
        'punc_number_of_edits': category_edits('punc'),
        'cap_number_of_edits': category_edits('cap'),
        'itn_number_of_edits': category_edits('itn'),
        'lexical_number_of_edits': category_edits('lexical'),
        'others_number_of_edits': category_edits('others'),
    }


def compute_batch_ter_report(batch_ids, batch_queries, batch_results, ter_locale, ter_type):
    script_path = Path(__file__).resolve().parent / 'TER' / 'getdfmetrics.py'
    if not script_path.exists():
        logger.warning('TER脚本不存在，跳过TER计算: %s', script_path)
        return {}, {}

    try:
        with tempfile.TemporaryDirectory(prefix='ter_extract_') as tmp_dir:
            input_tsv = os.path.join(tmp_dir, 'batch_ter_input.tsv')
            report_dir = os.path.join(tmp_dir, 'ter_report')

            with open(input_tsv, 'w', encoding='utf-8', newline='') as f:
                for row_id, query, result in zip(batch_ids, batch_queries, batch_results):
                    clean_id = clean_tsv_value(str(row_id))
                    clean_query = clean_tsv_value(str(query))
                    clean_result = clean_tsv_value(str(result))
                    f.write(f'{clean_id}\t{clean_query}\t{clean_result}\n')

            command = [
                sys.executable,
                str(script_path),
                'ter',
                '-i', input_tsv,
                '--utt_id_idx', '0',
                '--disp_trans_idx', '1',
                '--disp_reco_idx', '2',
                '--locale', ter_locale,
                '--ter_type', ter_type,
                '-o', report_dir,
            ]

            env = os.environ.copy()
            existing_pythonpath = env.get('PYTHONPATH', '')
            ter_root = str(script_path.parent)
            if existing_pythonpath:
                env['PYTHONPATH'] = f"{existing_pythonpath}:{ter_root}"
            else:
                env['PYTHONPATH'] = ter_root

            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            ter_json_path = os.path.join(report_dir, 'TERResponse.json')
            if not os.path.exists(ter_json_path):
                logger.warning('TER结果文件不存在: %s', ter_json_path)
                return {}, {}

            with open(ter_json_path, 'r', encoding='utf-8') as f:
                ter_report = json.load(f)

            sent_details = ter_report.get('sent_details', [])
            sent_detail_map = {str(item.get('utt_id')): item for item in sent_details}
            summary = ter_report.get('summary', {})
            return sent_detail_map, summary

    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or '').strip()
        stdout = (e.stdout or '').strip()
        logger.warning('TER计算失败，返回码=%s', e.returncode)
        if stderr:
            logger.warning('TER stderr: %s', stderr)
        if stdout:
            logger.warning('TER stdout: %s', stdout)
        return {}, {}
    except Exception as e:
        logger.warning('TER计算失败，跳过当前TER: %s', e)
        return {}, {}


def looks_like_header(first_row, id_col, query_col, response_col):
    lower = [c.strip().lower() for c in first_row]
    hints = {
        id_col.lower(),
        query_col.lower(),
        response_col.lower(),
        'id',
        'uttid',
        'utt_id',
        'query',
        'response',
    }
    if any(c in hints for c in lower):
        return True
    if any(c.endswith('_response') for c in lower):
        return True
    return False


def detect_response_column(headers, response_col):
    if response_col in headers:
        return response_col

    for header in headers:
        if header.endswith('_response'):
            return header

    if len(headers) >= 3:
        return headers[2]

    raise ValueError('Cannot detect response column from headers')


def read_input_tsv(input_tsv, id_col='uttid', query_col='query', response_col='response'):
    with open(input_tsv, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError('Input TSV is empty')

    first_line = lines[0].rstrip('\n')
    if not first_line.strip():
        raise ValueError('Input TSV first line is empty')

    first_row = first_line.split('\t')
    if len(first_row) < 3:
        raise ValueError(f'Input TSV requires at least 3 columns, got {len(first_row)}')

    has_header = looks_like_header(first_row, id_col, query_col, response_col)

    if has_header:
        headers = first_row
        data_start_idx = 1

        if id_col in headers:
            id_idx = headers.index(id_col)
        else:
            id_idx = 0
            logger.warning('ID列 %s 不在表头中，使用第1列: %s', id_col, headers[0])

        if query_col in headers:
            query_idx = headers.index(query_col)
        else:
            query_idx = 1
            logger.warning('Query列 %s 不在表头中，使用第2列: %s', query_col, headers[1])

        response_header = detect_response_column(headers, response_col)
        response_idx = headers.index(response_header)
    else:
        headers = [id_col, query_col, response_col]
        data_start_idx = 0
        id_idx, query_idx, response_idx = 0, 1, 2
        response_header = response_col

    rows = []
    for line_no in range(data_start_idx, len(lines)):
        line = lines[line_no].rstrip('\n')
        if not line.strip():
            continue

        parts = line.split('\t')
        if len(parts) < 3:
            logger.warning('Line %d has less than 3 columns, skipping', line_no + 1)
            continue

        max_idx = max(id_idx, query_idx, response_idx)
        if len(parts) <= max_idx:
            logger.warning('Line %d does not contain required indices, skipping', line_no + 1)
            continue

        uttid = parts[id_idx]
        query = parts[query_idx]
        response = parts[response_idx]
        rows.append((uttid, query, response))

    logger.info('Loaded %d rows from %s (has_header=%s)', len(rows), input_tsv, has_header)
    return rows, has_header, response_header


def save_tsv(rows, output_path, response_col, metrics_prefix, id_col='uttid', query_col='query'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    metric_cols = [
        f'{metrics_prefix}_ter_info_number_of_tokens',
        f'{metrics_prefix}_ter_info_number_of_edits',
        f'{metrics_prefix}_ter_info_display_ter',
        f'{metrics_prefix}_punc_number_of_edits',
        f'{metrics_prefix}_cap_number_of_edits',
        f'{metrics_prefix}_itn_number_of_edits',
        f'{metrics_prefix}_lexical_number_of_edits',
        f'{metrics_prefix}_others_number_of_edits',
    ]

    headers = [id_col, query_col, response_col] + metric_cols

    temp_path = output_path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8', newline='') as f:
        f.write('\t'.join(headers) + '\n')
        for row in rows:
            values = []
            for header in headers:
                value = row.get(header, '')
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                values.append(clean_tsv_value(str(value)))
            f.write('\t'.join(values) + '\n')

    if os.path.exists(output_path):
        backup_path = output_path + '.bak'
        os.rename(output_path, backup_path)
    os.rename(temp_path, output_path)

    backup_path = output_path + '.bak'
    if os.path.exists(backup_path):
        os.remove(backup_path)


def infer_output_response_col(response_header, llm_name):
    if response_header and response_header.endswith('_response'):
        return response_header

    model_col = (llm_name or '').strip() or 'model'
    if not model_col.endswith('_response'):
        model_col = f'{model_col}_response'
    return model_col


def response_prefix(response_col):
    if response_col.endswith('_response'):
        return response_col[:-len('_response')]
    return response_col


def main():
    parser = argparse.ArgumentParser(description='Calculate TER and extract puretext TSV from 3-column input')
    parser.add_argument('--input_tsv', required=True, help='Input TSV path (3 columns: uttid, query, response)')
    parser.add_argument('--output_tsv', required=False, help='Output puretext TSV path')
    parser.add_argument('--id_col', default='uttid', help='ID column name when header exists (default: uttid)')
    parser.add_argument('--query_col', default='query', help='Query column name when header exists (default: query)')
    parser.add_argument('--response_col', default='response', help='Response column name when header exists (default: response)')
    parser.add_argument('--llm_name', default='model', help='Used for output response column naming when input response column is generic')
    parser.add_argument('--ter_locale', default='en-us', help='TER locale (default: en-us)')
    parser.add_argument('--ter_type', default='verbatim', help='TER type (default: verbatim)')
    args = parser.parse_args()

    input_tsv = args.input_tsv
    output_tsv = args.output_tsv
    if not output_tsv:
        base, ext = os.path.splitext(input_tsv)
        if not ext:
            ext = '.tsv'
        output_tsv = f'{base}.puretext{ext}'

    rows, has_header, response_header = read_input_tsv(
        input_tsv=input_tsv,
        id_col=args.id_col,
        query_col=args.query_col,
        response_col=args.response_col,
    )

    output_response_col = infer_output_response_col(response_header, args.llm_name)
    metrics_prefix = response_prefix(output_response_col)

    batch_ids = [str(item[0]) for item in rows]
    batch_queries = [str(item[1]) for item in rows]
    batch_responses = [extract_response_text(item[2], llm_name=args.llm_name) for item in rows]

    ter_sent_map, summary = compute_batch_ter_report(
        batch_ids=batch_ids,
        batch_queries=batch_queries,
        batch_results=batch_responses,
        ter_locale=args.ter_locale,
        ter_type=args.ter_type,
    )

    out_rows = []
    for uttid, query, response in rows:
        response_text = extract_response_text(response, llm_name=args.llm_name)
        sent_detail = ter_sent_map.get(str(uttid), {})
        ter_metrics = extract_ter_metrics_from_sent_detail(sent_detail)

        row = {
            args.id_col: uttid,
            args.query_col: query,
            output_response_col: response_text,
        }
        for metric_name, metric_value in ter_metrics.items():
            row[f'{metrics_prefix}_{metric_name}'] = metric_value

        out_rows.append(row)

    save_tsv(
        rows=out_rows,
        output_path=output_tsv,
        response_col=output_response_col,
        metrics_prefix=metrics_prefix,
        id_col=args.id_col,
        query_col=args.query_col,
    )

    logger.info('Done. Output saved to: %s', output_tsv)
    if summary:
        logger.info('TER summary: %s', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
