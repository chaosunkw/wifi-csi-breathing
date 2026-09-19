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
    static = sub.add_parser('static', help='Compare six pilot recordings on shared axes with publication-style figures')
    static.add_argument('--catalog', type=Path, required=True)
    static.add_argument('--output', type=Path, required=True, help='New directory; never overwrite an existing run')
    static.add_argument('--seconds', type=float, default=120.0, help='Common interval starting at the first CSI packet (default: 120 s)')
    for name, default in (('tx', 1), ('rx', 1), ('reference-rx', 2), ('subcarrier', 15)):
        static.add_argument('--' + name, type=int, default=default)
    joint = sub.add_parser('joint', help='Plot a verified joint CSI / Marker transfer on its common host timeline')
    joint.add_argument('--input', type=Path, required=True, help='Transferred session directory containing transfer.json')
    joint.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite a run')
    joint.add_argument('--mode', action='append', choices=MODES, help='Repeat to compare modes; default is all three')
    for name, default in (('tx', 1), ('rx', 1), ('reference-rx', 2), ('subcarrier', 15)):
        joint.add_argument('--' + name, type=int, default=default)
    archive = sub.add_parser('archive-joint', help='Copy six verified joint transfers without altering source labels')
    archive.add_argument('--source-root', type=Path, required=True)
    archive.add_argument('--batch', required=True)
    archive.add_argument('--corrections', type=Path, required=True)
    archive.add_argument('--output', type=Path, required=True)
    diagnostic = sub.add_parser('diagnose-joint', help='Plot separate CSI matrix shapes, markers and stream states')
    diagnostic.add_argument('--archive', type=Path, required=True)
    diagnostic.add_argument('--output', type=Path, required=True)
    diagnostic.add_argument('--mode', action='append', choices=MODES)
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
        elif args.command == 'static':
            from .static_analysis import static_analyze
            report = static_analyze(load_catalog(args.catalog), args.output, args.seconds,
                                    args.tx, args.rx, args.reference_rx, args.subcarrier)
            print(f"Completed {len(report['metrics'])} comparisons: {args.output / 'index.html'}")
        elif args.command == 'joint':
            from .joint import joint_analyze
            report = joint_analyze(args.input, args.output, args.mode or MODES,
                                   args.tx, args.rx, args.reference_rx, args.subcarrier)
            print(f"Completed joint comparison with {report['window_markers']} manual events: {args.output / 'index.html'}")
        elif args.command == 'archive-joint':
            from .batch_diagnostics import archive_batch
            report = archive_batch(args.source_root, args.batch, args.corrections, args.output)
            print(f"Archived {len(report['sessions'])} verified sessions: {args.output / 'archive.json'}")
        elif args.command == 'diagnose-joint':
            from .batch_diagnostics import diagnose_archive
            report = diagnose_archive(args.archive, args.output, args.mode or MODES)
            print(f"Completed {len(report['summaries'])} session diagnostics: {args.output / 'index.html'}")
        else:
            from .gui import launch
            return launch(args.catalog)
    except (ValueError, OSError, KeyError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Cancelled. Check the completion manifest for the selected command; partial outputs are not complete runs.', file=sys.stderr)
        return 130
    return 0
