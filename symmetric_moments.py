#!/bin/python3
"""Compute moments in a window centered in a line."""
from typing import Sequence
import argparse
import sys

from toolkit.argparse_tools import actions
from toolkit.argparse_tools import parents
import toolkit.argparse_tools.loaders as aploaders
import toolkit.astro_tools.cube_utils as cubeutils

from line_little_helper.moving_moments import HelpFormatter
import line_little_helper.subcube_extractor as extractor

def _save_subcube(args: argparse.Namespace) -> None:
    """Save the subcube to disk."""
    outname = f'{args.output[0]}.subcube.fits'
    args.subcube.write(outname, overwrite=True)

def _get_moment(args: argparse.Namespace) -> Sequence[str]:
    """Calculate the moments and save."""
    filenames = []
    for mom in args.moments:
        # Calculate moment
        moment = cubeutils.get_moment(args.subcube,
                                      mom,
                                      linefreq=args.linefreq,
                                      lower_limit=args.fluxlimit,
                                      auto_rms=True,
                                      nsigma=args.nsigma[0],
                                      log=args.log.info)

        # Save
        outname = f'{args.output[0]}.subcube.moment{mom}.fits'
        moment.write(outname, overwrite=True)
        filenames.append(outname)

    return filenames

def symmetric_moments(args: Sequence[str]) -> Sequence[str]:
    """Calculate symmetric moments from commad line arguments.
    
    Args:
      args: command line arguments.

    Returns:
      A list with the moment file names.
    """
    # Argument parser
    pipe = [aploaders.load_spectral_cube, extractor.check_line_freq,
            extractor.get_subcube, _save_subcube, _get_moment]
    args_parents = [extractor.parent_parser(),
                    parents.logger('debug_symmetric_moments.log')]
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=HelpFormatter,
        parents=args_parents,
        conflict_handler='resolve')
    parser.add_argument('--fluxlimit', nargs=2, action=actions.ReadQuantity,
                        help='Flux lower limit with units')
    parser.add_argument('--nsigma', nargs=1, type=int, default=[5],
                        help='Number of rms levels for flux limit')
    parser.add_argument('output', nargs=1, action=actions.NormalizePath,
                        help='The output basename')
    parser.add_argument('moments', nargs='*', type=int,
                        help='Moments to calculate')
    parser.set_defaults(subcube=None, common_beam=True, put_rms=True)
    args = parser.parse_args(args)
    args.put_rms = True
    for i, step in enumerate(pipe):
        if i < len(pipe)-1:
            step(args)
        else:
            filenames = step(args)

    return filenames

if __name__=='__main__':
    symmetric_moments(sys.argv[1:])
