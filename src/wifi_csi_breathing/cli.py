"""Local-only entry points. No SSH, serial, network, or acquisition commands."""
import argparse
from pathlib import Path
import sys

from .analysis import analyze
from .data import MODES, import_archive, load_catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description='Wi-Fi CSI offline analysis and causal replay (research preview)')
    sub = parser.add_subparsers(dest='command', required=True)
    catalog = sub.add_parser('import-archive', help='Verify an existing six-session archive and create a local catalog')
    catalog.add_argument('archive', type=Path)
    catalog.add_argument('--output', type=Path, required=True)
    batch = sub.add_parser('analyze', help='Export same-processor offline waveforms and exploratory estimates')
    batch.add_argument('--catalog', type=Path, required=True)
    batch.add_argument('--output', type=Path, required=True, help='New directory; existing directories are never overwritten')
    batch.add_argument('--mode', action='append', choices=MODES, help='Repeat to compare modes; default is all three')
    for name, default in (('tx', 1), ('rx', 1), ('reference-rx', 2), ('subcarrier', 15)):
        batch.add_argument('--' + name, type=int, default=default)
    gui = sub.add_parser('gui', help='Open the desktop recorded-data replay viewer')
    gui.add_argument('--catalog', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'import-archive':
            count = import_archive(args.archive, args.output)
            print(f'Catalog created for {count} verified recordings; raw files unchanged.')
        elif args.command == 'analyze':
            report = analyze(load_catalog(args.catalog), args.output, args.mode or MODES,
                             args.tx, args.rx, args.reference_rx, args.subcarrier)
            print(f"Completed {len(report['results'])} dataset/feature analyses: {args.output / 'report.json'}")
        else:
            from .gui import launch
            return launch(args.catalog)
    except (ValueError, OSError, KeyError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Cancelled. An incomplete export has no report.json.', file=sys.stderr)
        return 130
    return 0
