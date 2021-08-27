"""Compute moments in a window centered in a line."""
from typing import Sequence
import argparse
import sys

import toolkit.argparse_tools.actions as actions
import toolkit.argparse_tools.loaders as aploaders
import toolkit.argparse_tools.parents as parents
import toolkit.astro_tools.cube_utils as cubeutils

from moving_moments import HelpFormatter
import subcube_extractor as extractor

def _save_subcube(args: argparse.Namespace) -> None:
    """Save the subcube to disk."""
    outname = f'{args.output[0]}.subcube.fits'
    args.subcube.write(outname, overwrite=True)

def _get_moment(args: argparse.Namespace) -> None:
    """Calculate the moments and save."""
    for mom in args.moments:
        # Calculate moment
        moment = cubeutils.get_moment(args.subcube,
                                      mom,
                                      linefreq=args.linefreq,
                                      lower_limit=args.fluxlimit,
                                      auto_rms=True)

        # Save
        outname = f'{args.output[0]}.subcube.moment{mom}.fits'
        moment.write(outname, overwrite=True)

def main(args: Sequence[str]):
    """Main program."""
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
    parser.add_argument('output', nargs=1, action=actions.NormalizePath,
                        help='The output basename')
    parser.add_argument('moments', nargs='*', type=int,
                        help='Moments to calculate')
    parser.set_defaults(subcube=None)
    args = parser.parse_args(args)
    args.put_rms = True
    for step in pipe:
        step(args)

if __name__=='__main__':
    main(sys.argv[1:])
