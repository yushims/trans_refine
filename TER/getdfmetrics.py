#!/opt/conda/envs/ptca/bin/python3.10

import argparse
import json
import sys
from typing import List

strList = List[str]

class GetDFMetrics:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description = '''
                Display Formatting Utilities.
                Common commands are:
                getdfmetrics.py metrics
                getdfmetrics.py ftests
                getdfmetrics.py ter
                getdfmetrics.py analytics
            ''',
            formatter_class = argparse.RawTextHelpFormatter
        )

        subparsers = parser.add_subparsers(dest = 'mode', help = 'Available commands', required = True)

        # Set up subparsers for each mode
        self._setup_metrics_parser(subparsers)
        self._setup_ter_parser(subparsers)
        self._setup_ftests_parser(subparsers)
        self._setup_analytics_parser(subparsers)

        args = parser.parse_args()

        # Use dispatch pattern to invoke method with same name
        getattr(self, args.mode)(args)

    def _setup_metrics_parser(self, subparsers):
        # The 'metrics' command seems to have its own argument parsing,
        # so we pass the remaining arguments to it.
        parser_metrics = subparsers.add_parser('metrics', help = 'Generate dfmetrics.')
        parser_metrics.add_argument('metrics_args', nargs = argparse.REMAINDER, help = 'Arguments for metrics command')

    def _setup_ter_parser(self, subparsers):
        parser_ter = subparsers.add_parser('ter', help = 'Run TER computation.', description = 'Display TER Computation')

        # Group for TSV file input
        tsv_group = parser_ter.add_argument_group('TSV file input', 'Arguments for processing a TSV file.')
        tsv_group.add_argument('-i', '--input_file', help = 'The tsv file contains display format transcription and recognition columns, no header')
        tsv_group.add_argument('--utt_id_idx', help = 'The column index to the utterance unique id (for tsv input)')
        tsv_group.add_argument('--disp_trans_idx', help = 'The column index to the display format transcription (for tsv input)')
        tsv_group.add_argument('--disp_reco_idx', help = 'The column index to the display format recognition (for tsv input)')

        # Group for string input
        string_group = parser_ter.add_argument_group('String input', 'Arguments for processing transcription and recognition strings.')
        string_group.add_argument('--trans', help = 'The display format transcription string')
        string_group.add_argument('--reco', help = 'The display format recognition string')

        # Common arguments
        common_group = parser_ter.add_argument_group('Common options')
        common_group.add_argument('--locale', nargs='?', default = 'en-us', help = 'The locale to apply TER')
        common_group.add_argument('--ter_type', help = 'TER Type (verbatim/nondisfluency/nonverbatim)', choices = ['verbatim', 'nondisfluency', 'nonverbatim'], default='nondisfluency')
        common_group.add_argument('-o', '--output_ter_report', help = 'The final TER report, saved as a json file')

    def _setup_ftests_parser(self, subparsers):
        subparsers.add_parser('ftests', help = 'Run functional tests.')

    def _setup_analytics_parser(self, subparsers):
        parser_analytics = subparsers.add_parser('analytics', help = 'Run analytics computation.', description = 'Display Analytics Computation')

        # Group for TSV file input
        tsv_group = parser_analytics.add_argument_group('TSV file input', 'Arguments for processing a TSV file.')
        tsv_group.add_argument('-i', '--input_file', help = 'The tsv file contains display format transcription and recognition columns, no header')
        tsv_group.add_argument('--utt_id_idx', help = 'The column index to the utterance unique id (for tsv input)')
        tsv_group.add_argument('--disp_trans_idx', help = 'The column index to the display format transcription (for tsv input)')
        tsv_group.add_argument('--disp_reco_idx', help = 'The column index to the display format recognition (for tsv input)')

        # Group for string input
        string_group = parser_analytics.add_argument_group('String input', 'Arguments for processing transcription and recognition strings.')
        string_group.add_argument('--trans', help = 'The display format transcription string')
        string_group.add_argument('--reco', help = 'The display format recognition string')

        # Common arguments
        common_group = parser_analytics.add_argument_group('Common options')
        common_group.add_argument('-o', '--output_file', help = 'The final analytics report, saved as a json file')
        common_group.add_argument('--locale', nargs = '?', default = 'en-us', help = 'The locale to apply TER')
        common_group.add_argument('--ter_type', help = 'TER Type (verbatim/nondisfluency/nonverbatim)', choices = ['verbatim', 'nondisfluency', 'nonverbatim'], default='nondisfluency')
        common_group.add_argument('--analysis_type', help = 'Type of analysis to perform', choices = ['prf_scores'], default = 'prf_scores')

    def metrics(self, args):
        from dfmetrics.metrics import DFMetrics
        # The original DFMetrics class seems to handle its own argv parsing
        DFMetrics(args.metrics_args)

    def ter(self, ter_args):
        from dfmetrics.ter import TER
        ter = TER(locale=ter_args.locale.lower(), ter_type=ter_args.ter_type)

        if ter_args.input_file:
            if ter_args.trans or ter_args.reco:
                # Argparse can handle this with mutually exclusive groups, but for simplicity, we'll keep the manual check.
                sys.exit("Error: Cannot use --input_file with --trans or --reco.")

            ter.compute_ter_from_tsv(input_file = ter_args.input_file,
                utt_id_idx = ter_args.utt_id_idx,
                disp_trans_idx = ter_args.disp_trans_idx,
                disp_reco_idx = ter_args.disp_reco_idx,
                output_ter_report = ter_args.output_ter_report)

        elif ter_args.trans and ter_args.reco:
            response = ter.compute_ter_from_strings(
                transcription = ter_args.trans,
                recognition = ter_args.reco
            )

            if ter_args.output_ter_report:
                with open(ter_args.output_ter_report, 'w', encoding = 'utf-8') as f:
                    json.dump(response, f, indent = 4, ensure_ascii = False)
            else:
                print(json.dumps(response, indent = 4, ensure_ascii = False))
        else:
            sys.exit("Error: Either --input_file or both --trans and --reco are required.")

    def ftests(self, args):
        import unittest
        import dfmetrics.tests

        # Load tests directly from the dfmetrics.tests module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(dfmetrics.tests)
        runner = unittest.TextTestRunner()
        result = runner.run(suite)

        # Exit with a non-zero status code if any tests failed
        if not result.wasSuccessful():
            print("TESTS FAILED")
            sys.exit(1)
        else:
            print("TESTS PASSED")

    def analytics(self, ter_args):
        from dfmetrics.ter import TER
        response = {}
        if ter_args.analysis_type == 'prf_scores':
            ter = TER(locale = ter_args.locale.lower(), ter_type = ter_args.ter_type)

            if ter_args.input_file:
                if ter_args.trans or ter_args.reco:
                    sys.exit("Error: Cannot use --input_file with --trans or --reco.")

                response = ter.get_prf_scores_from_tsv(input_file = ter_args.input_file,
                    utt_id_idx = ter_args.utt_id_idx,
                    disp_trans_idx = ter_args.disp_trans_idx,
                    disp_reco_idx = ter_args.disp_reco_idx)

            elif ter_args.trans and ter_args.reco:
                response = ter.get_prf_scores_from_strings(
                    transcription = ter_args.trans,
                    recognition = ter_args.reco
                )
            else:
                sys.exit("Error: Either --input_file or both --trans and --reco are required.")

            if ter_args.output_file:
                with open(ter_args.output_file, 'w', encoding = 'utf-8') as f:
                    json.dump(response, f, indent = 4, ensure_ascii = False)
            else:
                print(json.dumps(response, indent = 4, ensure_ascii = False))

        return response

def launch():
    GetDFMetrics()

if __name__ == "__main__":
    launch()
