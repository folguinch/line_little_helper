"""Calculate moments for data over a fraction of the peak.

The script produces a velocity map at peak line emission and a moment map for
emission over a fraction of the peak.
"""
from typing import Sequence, Optional
import argparse
import sys

from astropy.io import fits
from toolkit.argparse_tools import actions
import numpy as np
import toolkit.argparse_tools.loaders as aploaders
import toolkit.argparse_tools.parents as apparents
import toolkit.astro_tools.cube_utils as cubeutils

from line_little_helper.argparse_parents import line_parents, cube_parent
from line_little_helper.argparse_processing import get_subcube, set_fluxlimit
from line_little_helper.moving_moments import HelpFormatter
import line_little_helper.subcube_extractor as extractor

def _save_subcube(args: argparse.Namespace) -> None:
    """Save the subcube to disk."""
    if args.savesteps:
        outname = f'{args.output[0]}.subcube.fits'
        args.subcube.write(outname, overwrite=True)

def _get_moment(args: argparse.Namespace) -> Sequence[str]:
    """Calculate the moments and save."""
    # Peak map
    peak_map = np.nanmax(args.subcube)
    header = args.subcube.wcs.sub(2).to_header()
    header['BUNIT'] = f'{peak_map.unit:FITS}'
    outname = f'{args.output[0]}.peakmap.fits'
    hdu = fits.PrimaryHDU(data=peak_map.value, header=header)
    hdu.writeto(outname, overwrite=True)

    # Moments over FWHM
    peakmask = np.greater(args.subcube.unmasked_data[:].value,
                          peak_map.value * args.fraction[0])
    fluxmask = args.subcube.unmasked_data[:] > args.flux_limit[0]
    totalmask = peakmask & fluxmask
    for mom in args.moments:
        # Calculate moment
        subcube = args.subcube.with_mask(totalmask)
        moment = cubeutils.get_moment(subcube,
                                      mom,
                                      linefreq=args.linefreq,
                                      auto_rms=False,
                                      log=args.log.info)

        # Save
        outname = f'{args.output[0]}.local_moment{mom}.fits'
        moment.writeto(outname, overwrite=True)

def local_moments(args: Optional[Sequence[str]] = None) -> None:
    """Calculate local moments from command line input."""
    pipe = [aploaders.load_spectral_cube, extractor.check_line_freq,
            set_fluxlimit, get_subcube, _save_subcube, _get_moment]
    args_parents = [apparents.logger('debug_local_moments.log'),
                    cube_parent(),
                    line_parents('vlsr',
                                 'molecule',
                                 'spectral_range',
                                 'flux',
                                 'spatial_range')]
    parser = argparse.ArgumentParser(
        add_help=True,
        #description=description,
        formatter_class=HelpFormatter,
        parents=args_parents,
        conflict_handler='resolve')
    parser.add_argument('--savesteps', action='store_true',
                        help='Save subcube and masks at each step')
    parser.add_argument('--moments', nargs='+', type=int, default=[1, 2],
                        help='Moments to calculate')
    parser.add_argument('--fraction', nargs='*', type=float, default=[0.5],
                        help='Fraction of the peak')
    parser.add_argument('output', nargs=1, action=actions.NormalizePath,
                        help='Output basename')
    parser.set_defaults(subcube=None, common_beam=True, put_rms=True)
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)
    for step in pipe:
        step(args)

if __name__ == '__main__':
    local_moments(sys.argv[1:])
