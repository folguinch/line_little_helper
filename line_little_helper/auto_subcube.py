"""Extract sub-cubes from valid emission of a given representative transition.

An initial subcube of the representative transition is calculated from the input
parameters. From the line emission over a certain threshold a rectangular mask
is created encompasing the valid pixels. This mask is used to crop the initial
cube spatially. 

A padding parameter can be given to increase the size of the region as a factor
of the size of the rectangular mask axes. For example, if the mask box has 4 by
6 pixels and a padding of 0.5 the final mask will be a box of size 8 by 12 (if
not at the borders of the cube).
"""
from typing import Optional, Sequence
import argparse
import sys

from astropy.io import fits
from toolkit.argparse_tools import actions
from toolkit.argparse_tools import parents
from toolkit.astro_tools import cube_utils
from toolkit.astro_tools.masking import split_mask_structures
import numpy as np
import toolkit.argparse_tools.loaders as aploaders

from line_little_helper.argparse_parents import line_parents
from line_little_helper.argparse_processing import get_subcube, set_fluxlimit
from line_little_helper.moving_moments import HelpFormatter
import line_little_helper.subcube_extractor as extractor

def _process_data(args: argparse.Namespace):
    """Generate and save the mask for each connected structure."""
    # Initial mask
    args.log.info('-' * 80)
    args.log.info('Processing sub-cube:')
    args.log.info('Flux limit: %s', args.flux_limit[0])
    args.log.info('Sub-cube rms: %f %s', args.subcube.meta['RMS'],
                  args.subcube.unit)
    mask = np.any(args.subcube.unmasked_data[:] > args.flux_limit[0], axis=0)
    args.log.info('Pixels over flux limit: %i', np.sum(mask))

    # Find submasks
    mask, sub_masks = split_mask_structures(mask, min_area=args.min_area[0],
                                            padding=args.padding[0])
    if len(sub_masks) == 0:
        args.log.info('No structures over %s were identified', args.flux_limit)
        sys.exit()
    args.log.info('%i sub regions identified', len(sub_masks))

    # Generate header
    header = args.subcube.wcs.sub(2).to_header()

    # Save masks and cubes
    for i, sub_mask in enumerate(sub_masks):
        # Save mask
        hdu = fits.PrimaryHDU(data=sub_mask.astype(float), header=header)
        outname = f'{args.output[0]}.structure{i+1}.mask.fits'
        hdu.writeto(outname, overwrite=True)

        # Calculate subcube
        rms = args.flux_limit / args.nsigma
        masked = args.cube.with_mask(sub_mask)
        subcube = cube_utils.get_subcube(masked, shrink=True, rms=rms,
                                         log=args.log.info)
        cubename = f'{args.output[0]}.structure{i+1}.subcube.fits'
        subcube.write(cubename, overwrite=True)

def auto_subcube(args: Optional[Sequence[str]] = None) -> None:
    # Argument parser
    pipe = [aploaders.load_spectral_cube, extractor.check_line_freq,
            set_fluxlimit, get_subcube, _process_data]
    args_parents = [extractor.parent_parser(),
                    line_parents('flux'),
                    parents.logger('debug_auto_subcube.log')]
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=HelpFormatter,
        parents=args_parents,
        conflict_handler='resolve')
    parser.add_argument('--min_area', nargs=1, default=[12], type=float,
                        help='Minimum number of pixels per structure')
    parser.add_argument('--padding', nargs=1, default=[0.25], type=float,
                        help='Size increment in terms of axis fraction')
    parser.add_argument('output', nargs=1, action=actions.NormalizePath,
                        help='The output basename')
    #parser.add_argument('moments', nargs='*', type=int,
    #                    help='Moments to calculate')
    parser.set_defaults(subcube=None, put_rms=True)
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)
    for step in pipe:
        step(args)

if __name__ == '__main__':
    auto_subcube(sys.argv[1:])
