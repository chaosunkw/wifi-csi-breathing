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
    fixed = sub.add_parser('joint-fixed', help='Opt-in fixed 3x3/rate-flags analysis with bounded missing-sample gaps')
    fixed.add_argument('--input', type=Path, required=True)
    fixed.add_argument('--output', type=Path, required=True, help='New directory; never overwrite a run')
    fixed.add_argument('--rate-flags', type=int, required=True, help='Explicit observed raw rate code; not Mbps or packet rate')
    fixed.add_argument('--mode', action='append', choices=MODES)
    for name, default in (('tx', 1), ('rx', 1), ('reference-rx', 2), ('subcarrier', 15)):
        fixed.add_argument('--' + name, type=int, default=default)
    rate = sub.add_parser('joint-rate', help='Compare one 60 s log-ratio rate estimate with manual button counts')
    rate.add_argument('--input', type=Path, required=True)
    rate.add_argument('--output', type=Path, required=True, help='New directory; never overwrite a run')
    rate.add_argument('--start', type=float, default=10.0, help='Predefined window start from common T0 (default: 10 s)')
    rate.add_argument('--shape-policy', choices=('3x3', 'first-stream'), default='3x3',
                      help='3x3 baseline (default), or experimental stream 1 across 1x3/2x3/3x3')
    rate.add_argument('--gap-policy', choices=('reset', 'preserve-short'), default='reset',
                      help='Original gap resets, or bounded short-gap state preservation with visible gaps')
    for name, default in (('tx', 1), ('rx', 1), ('reference-rx', 2), ('subcarrier', 15)):
        rate.add_argument('--' + name, type=int, default=default)
    farsense = sub.add_parser('compare-farsense', help='Compare adapted FarSense core with existing FFT and ACF candidates')
    farsense.add_argument('--manifest', type=Path, required=True, help='Local JSON listing recordings and predefined minute starts')
    farsense.add_argument('--output', type=Path, required=True, help='New directory; never overwrite a run')
    wiresp = sub.add_parser('compare-wiresp', help='Compare adapted WiResP with FarSense and existing FFT/ACF candidates')
    wiresp.add_argument('--manifest', type=Path, required=True, help='Same predefined-minute manifest as compare-farsense')
    wiresp.add_argument('--output', type=Path, required=True, help='New directory; never overwrite a run')
    vitalcsi = sub.add_parser('compare-vitalcsi', help='Compare adapted VitalCSI with four existing baselines')
    vitalcsi.add_argument('--manifest', type=Path, required=True, help='Same predefined-minute comparison manifest')
    vitalcsi.add_argument('--output', type=Path, required=True, help='New directory; never overwrite a run')
    archive = sub.add_parser('archive-joint', help='Copy six verified joint transfers without altering source labels')
    archive.add_argument('--source-root', type=Path, required=True)
    archive.add_argument('--batch', required=True)
    archive.add_argument('--corrections', type=Path, required=True)
    archive.add_argument('--output', type=Path, required=True)
    diagnostic = sub.add_parser('diagnose-joint', help='Plot separate CSI matrix shapes, markers and stream states')
    diagnostic.add_argument('--archive', type=Path, required=True)
    diagnostic.add_argument('--output', type=Path, required=True)
    diagnostic.add_argument('--mode', action='append', choices=MODES)
    carriers = sub.add_parser('subcarriers', help='Plot all 30 subcarrier groups with separate CSI shapes')
    carriers.add_argument('--archive', type=Path, required=True)
    carriers.add_argument('--output', type=Path, required=True, help='New directory; never overwrite a run')
    carriers.add_argument('--mode', action='append', choices=MODES,
                          help='Repeat to compare modes; default: amplitude and phase_difference')
    for name, default in (('tx', 1), ('rx', 1), ('reference-rx', 2)):
        carriers.add_argument('--' + name, type=int, default=default)
    carriers.add_argument('--max-gap', type=float, default=.2, help='Break lines across gaps longer than this (seconds)')
    carriers.add_argument('--cell-seconds', type=float, default=.05, help='Maximum display width of a packet heatmap glyph')
    carriers.add_argument('--color-percentile', type=float, default=99., help='Centered nonphase color limit percentile, 90..100')
    carriers.add_argument('--groups', type=int, nargs='+', default=[1, 15, 30], help='One to four fixed example groups for traces')
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
        elif args.command == 'joint-fixed':
            from .fixed_configuration import analyze_fixed
            report = analyze_fixed(args.input, args.output, args.rate_flags, args.mode or MODES,
                                   args.tx, args.rx, args.reference_rx, args.subcarrier)
            print(f"Completed fixed-configuration exploration: {args.output / 'index.html'}")
        elif args.command == 'joint-rate':
            from .rate_comparison import analyze_rate
            report = analyze_rate(args.input, args.output, args.start,
                                  args.tx, args.rx, args.reference_rx, args.subcarrier, args.shape_policy,
                                  args.gap_policy)
            print(f"Completed 60 s rate comparison ({report['comparison']['status']}): {args.output / 'index.html'}")
        elif args.command == 'compare-farsense':
            from .farsense_comparison import compare_farsense
            report = compare_farsense(args.manifest, args.output)
            print(f"Completed {len(report['results'])} FarSense-core comparison minutes: {args.output / 'index.html'}")
        elif args.command == 'compare-wiresp':
            from .wiresp_comparison import compare_wiresp
            report = compare_wiresp(args.manifest, args.output)
            print(f"Completed {len(report['results'])} WiResP comparison minutes: {args.output / 'index.html'}")
        elif args.command == 'compare-vitalcsi':
            from .vitalcsi_comparison import compare_vitalcsi
            report = compare_vitalcsi(args.manifest, args.output)
            print(f"Completed {len(report['results'])} VitalCSI comparison minutes: {args.output / 'index.html'}")
        elif args.command == 'archive-joint':
            from .batch_diagnostics import archive_batch
            report = archive_batch(args.source_root, args.batch, args.corrections, args.output)
            print(f"Archived {len(report['sessions'])} verified sessions: {args.output / 'archive.json'}")
        elif args.command == 'diagnose-joint':
            from .batch_diagnostics import diagnose_archive
            report = diagnose_archive(args.archive, args.output, args.mode or MODES)
            print(f"Completed {len(report['summaries'])} session diagnostics: {args.output / 'index.html'}")
        elif args.command == 'subcarriers':
            from .subcarriers import analyze_subcarriers, DEFAULT_MODES
            report = analyze_subcarriers(args.archive, args.output, args.mode or DEFAULT_MODES,
                                        args.tx, args.rx, args.reference_rx, args.max_gap,
                                        args.cell_seconds, args.color_percentile, args.groups)
            print(f"Completed {len(report['summaries'])} session-feature views: {args.output / 'index.html'}")
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
