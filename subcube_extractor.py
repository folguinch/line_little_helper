"""Script to extract a subcube."""
from typing import Sequence
import argparse
import sys

import astropy.units as u
import toolkit.argparse_tools.actions as actions
import toolkit.argparse_tools.loaders as aploaders
import toolkit.argparse_tools.parents as apparents
import toolkit.astro_tools.cube_utils as cubeutils

from moving_moments import HelpFormatter, get_molecule
from parents import line_parents

def parent_parser() -> argparse.ArgumentParser:
    """Define the parent parser."""
    parents = [line_parents(['vlsr', 'molecule'])]
    parser = argparse.ArgumentParser(parents=parents)
    parser.add_argument('--put_rms', action='store_true',
                        help='Calculate and write input cube rms to subcube')
    parser.add_argument('--linefreq', nargs=2, action=actions.ReadQuantity,
                        default=None,
                        help='Line rest freq')
    group1 = parser.add_mutually_exclusive_group(required=False)
    group1.add_argument('--freq_range', metavar=('VAL0', 'VAL1', 'UNIT'),
                        nargs=3, action=actions.ReadQuantity,
                        help='Frequency range with unit')
    group1.add_argument('--vel_range', metavar=('VAL0', 'VAL1', 'UNIT'),
                        nargs=3, action=actions.ReadQuantity,
                        help='Velocity range with unit')
    group1.add_argument('--chan_range', metavar=('CHAN0', 'CHAN1'), nargs=2,
                        type=int,
                        help='Channel range')
    group1.add_argument('--win_halfwidth', nargs=1, type=int, default=[None],
                        help=('Channel window half width '
                              '(vlsr and line freq needed)'))
    group2 = parser.add_mutually_exclusive_group(required=False)
    group2.add_argument('--blc_trc', metavar=('BLCX', 'BLCY', 'TRCX', 'TRCY'),
                        nargs=4, type=int,
                        help='Position of BLC and TRC')
    group2.add_argument('--xy_ranges', metavar=('XLOW', 'XUP', 'YLOW', 'YUP'),
                        nargs=4, type=int,
                        help='Position x and y ranges')
    parser.add_argument('cubename', nargs=1, action=actions.CheckFile,
                        help='The input cube')

    return parser

def get_subcube(args: argparse.Namespace) -> None:
    """Extract the subcube."""
    args.log.info('Calculating subcube')
    args.subcube = cubeutils.get_subcube(args.cube,
                                         freq_range=args.freq_range,
                                         vel_range=args.vel_range,
                                         chan_range=args.chan_range,
                                         chan_halfwidth=args.win_halfwidth[0],
                                         blc_trc=args.blc_trc,
                                         xy_ranges=args.xy_ranges,
                                         vlsr=args.vlsr,
                                         linefreq=args.linefreq,
                                         put_rms=args.put_rms,
                                         log=args.log.info)

def check_line_freq(args: argparse.Namespace) -> None:
    """Copy molecule freq if `linefreq` is None."""
    if args.linefreq is None:
        args.cube = args.cube.with_spectral_unit(u.GHz)
        molec = get_molecule(args)
        if len(molec.transitions) > 1:
            raise ValueError('Too many transitions')
        args.linefreq = molec.transitions[0].restfreq

def _save_subcube(args: argparse.Namespace) -> None:
    """Save the subcube to disk."""
    args.subcube.write(args.output[0], overwrite=True)

def main(args: Sequence[str]):
    """Main program."""
    # Argument parser
    pipe = [aploaders.load_spectral_cube, check_line_freq, get_subcube,
            _save_subcube]
    args_parents = [parent_parser(),
                    apparents.logger('debug_subcube_extractor.log')]
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=HelpFormatter,
        parents=args_parents,
        conflict_handler='resolve')
    parser.add_argument('output', nargs=1, action=actions.NormalizePath,
                        help='The output cube')
    parser.set_defaults(cube=None, subcube=None)
    args = parser.parse_args(args)
    for task in pipe:
        task(args)

if __name__=='__main__':
    main(sys.argv[1:])
