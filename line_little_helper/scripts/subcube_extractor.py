#!/bin/python3
"""Script to extract a subcube."""
from typing import Sequence
import argparse
import sys

from toolkit.argparse_tools import actions
import astropy.units as u
import toolkit.argparse_tools.loaders as aploaders
import toolkit.argparse_tools.parents as apparents
import toolkit.astro_tools.cube_utils as cubeutils

from line_little_helper.molecule import NoTransitionError
from line_little_helper.scripts.moving_moments import HelpFormatter
from line_little_helper.scripts.argparse_parents import line_parents
from line_little_helper.scripts.argparse_processing import (get_subcube,
                                                            load_molecule)

def parent_parser() -> argparse.ArgumentParser:
    """Define the parent parser."""
    parents = [line_parents('vlsr', 'molecule', 'spectral_range',
                            'spatial_range')]
    parser = argparse.ArgumentParser(parents=parents)
    parser.add_argument('--put_rms', action='store_true',
                        help='Calculate and write input cube rms to subcube')
    parser.add_argument('--linefreq', nargs=2, action=actions.ReadQuantity,
                        default=None,
                        help='Line rest freq')
    parser.add_argument('cubename', nargs=1, action=actions.CheckFile,
                        help='The input cube')

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
