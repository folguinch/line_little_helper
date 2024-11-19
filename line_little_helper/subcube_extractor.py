#!/bin/python3
"""Script to extract a subcube."""
from typing import Sequence, Optional
import argparse
import sys

from toolkit.argparse_tools import actions
import astropy.units as u
import toolkit.argparse_tools.loaders as aploaders
import toolkit.argparse_tools.parents as apparents

from line_little_helper.molecule import NoTransitionError
from line_little_helper.moving_moments import HelpFormatter
from line_little_helper.argparse_parents import line_parents, cube_parent
from line_little_helper.argparse_processing import get_subcube, load_molecule

def parent_parser() -> argparse.ArgumentParser:
    """Define the parent parser."""
    parents = [line_parents('vlsr', 'molecule', 'spectral_range',
                            'spatial_range'),
               cube_parent()]
    parser = argparse.ArgumentParser(parents=parents)
    parser.add_argument('--put_rms', action='store_true',
                        help='Calculate and write input cube rms to subcube')
    parser.add_argument('--put_linefreq', action='store_true',
                        help='Replace the rest freq with the line freq')
    parser.add_argument('--linefreq', nargs=2, action=actions.ReadQuantity,
                        default=None,
                        help='Line rest freq')

    return parser

def check_line_freq(args: argparse.Namespace) -> None:
    """Copy molecule freq if `linefreq` is None."""
    if args.linefreq is None and args.molecule is not None:
        args.cube = args.cube.with_spectral_unit(u.GHz)
        molec = load_molecule(args)
        args.log.info(f'{molec}')
        if len(molec.transitions) > 1:
            raise ValueError('Too many transitions')
        elif len(molec.transitions) == 0:
            #args.qns = None
            #args.log.info('No transition found')
            #molec = get_molecule(args)
            #args.log.info(f'Molecule:\n{molec}')
            #raise ValueError('No transitions')
            raise NoTransitionError(args.molecule[0], qns=args.qns)
        args.linefreq = molec.transitions[0].restfreq

def _convert_cube_spaxis(args: argparse.Namespace) -> None:
    """Convert the spectral axis to the requested type."""
    unit = args.subcube.spectral_axis.unit
    if args.spectral_axis == 'frequency' and not unit.is_equivalent(u.Hz):
        args.subcube = args.subcube.with_spectral_unit(
            u.Hz,
            velocity_convention='radio',
        )
    elif args.spectral_axis == 'velocity' and not unit.is_equivalent(u.m/u.s):
        args.subcube = args.subcube.with_spectral_unit(
            u.m/u.s,
            velocity_convention='radio',
        )
    else:
        pass

def _save_subcube(args: argparse.Namespace) -> None:
    """Save the subcube to disk."""
    args.subcube.write(args.output[0], overwrite=True)

def subcube_extractor(args: Optional[Sequence[str]] = None):
    """Main program."""
    # Argument parser
    pipe = [aploaders.load_spectral_cube, check_line_freq, get_subcube,
            _convert_cube_spaxis, _save_subcube]
    args_parents = [parent_parser(),
                    apparents.logger('debug_subcube_extractor.log')]
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=HelpFormatter,
        parents=args_parents,
        conflict_handler='resolve')
    parser.add_argument('--rms', metavar=('VAL', 'UNIT'), default=None,
                       action=actions.ReadQuantity,
                       help='Data rms value')
    parser.add_argument('--spectral_axis', default='original',
                        choices=['original', 'frequency', 'velocity'],
                        help='Output spectral axis type')
    parser.add_argument('output', nargs=1, action=actions.NormalizePath,
                        help='The output cube')
    parser.set_defaults(cube=None, subcube=None)
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)
    for task in pipe:
        task(args)

if __name__=='__main__':
    subcube_extractor(sys.argv[1:])
