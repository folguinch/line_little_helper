#!/bin/python
"""Find the velocity gradient from a 1st moment map."""
from typing import Optional, Sequence
import argparse
import sys

from astropy.wcs import WCS
from radio_beam import Beam
from toolkit.argparse_tools import actions, parents
from toolkit.argparse_tools.functions import (source_properties,
                                              pixels_to_positions)
from toolkit.astro_tools.images import stats_at_position, intensity_gradient
import astropy.units as u
import numpy as np

from line_little_helper.moving_moments import HelpFormatter

#def map_gradient(args: argparse.Namespace
#                 ) -> Tuple[u.Quantity, u.Quantity, WCS]:
#    """Compute input map gradient."""
#    # Get first moment
#    img = args.moment
#    bunit = u.Unit(img.header['BUNIT'])
#    wcs = WCS(img, naxis=['longitude','latitude'])
#
#    # Gradient
#    args.log.info('Computing gradient')
#    pisize = np.sqrt(wcs.proj_plane_pixel_area()).to(u.arcsec)
#    grow, gcol = np.gradient(np.squeeze(inp.img) * bunit, pixsize)
#    modulus = np.sqrt(grow**2 + gcol**2)
#    direction = np.degrees(np.arctan2(grow, gcol))
#
#    # Save fits
#    args.log.info('Saving files')
#    outname = f'{args.output[0]}.gradient.modulus.fits'
#    header = wcs.to_header()
#    header = copy_header_keys(img.header, header)
#    header['BUNIT'] = f'{modulus.unit:FITS}'
#    hdu = fits.PrimaryHDU(modulus.value, header=header)
#    args.log.info('Writing modulus to: %s', outname)
#    hdu.writeto(outname, overwrite=True)
#    outname = f'{args.output[0]}.direction.modulus.fits'
#    header['BUNIT'] = f'{direction.unit:FITS}'
#    hdu = fits.PrimaryHDU(direction.value, header=header)
#    args.log.info('Writing direction to: %s', outname)
#    hdu.writeto(outname, overwrite=True)
#
#    return modulus, direction, wcs
#
#def map_stats(args: argparse.Namespace,
#              source_props: Dict,
#              modulus: u.Quantity,
#              direction: u.Quantity,
#              wcs: WCS):
#    """Compute statistics over maps."""
#
#    # Beam size
#    bmin, bmaj = inp.header['BMIN'], inp.header['BMAJ']
#    pixsize = np.sqrt(np.abs(inp.header['CDELT2'] * \
#            inp.header['CDELT1']))
#    beam = np.sqrt(bmin*bmaj)
#    lines += ['Beam size: %f arcsec' % (beam*3600.,)]
#    args.log.info(lines[-1])
#    beam = apstats.gaussian_fwhm_to_sigma*beam/pixsize
#    lines += ['Region radius: %f pix' % beam]
#    args.log.info(lines[-1])
#
#    # Read positions
#    args.log.info('Reading positions')
#
#    # Iterate over positions
#    for pos in args.pos:
#        lines += ['Position: %f, %f' % tuple(pos)]
#        args.log.info(lines[-1])
#
#        # Distance matrix
#        y,x = np.indices(drc.shape, dtype=float)
#        dist = np.sqrt((y-pos[1])**2 + (x-pos[0])**2)
#        mask = dist <= beam
#
#        # Statistics
#        avg_drc = np.nanmean(drc[mask])
#        std_drc = np.nanstd(drc[mask])
#        avg_drc = avg_drc - 90
#        if avg_drc<0:
#            avg_drc = 360. + avg_drc
#        lines += ['Average gradient direction: %f+/-%f' % (avg_drc, std_drc)]
#        args.log.info(lines[-1])
#
#        lines += ['='*80]
#        print(lines[-1])
#
#    # Write log
#    outfile = os.path.splitext(args.out)[0] + '.gradient.txt'
#    args.log.info('Writing results to: %s', outfile)
#    with open(outfile, 'w') as out:
#        out.write('\n'.join(lines))

def velocity_gradient(args: Optional[Sequence[str]] = None) -> None:
    """Calculate velocity gradient."""
    # Argument parser
    src_props = ['radius']
    args_parents = [parents.logger('debug_velocity_gradient.log'),
                    parents.source_position(required=True),
                    parents.source_properties(src_props)]
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=HelpFormatter,
        parents=args_parents,
        conflict_handler='resolve')
    parser.add_argument('moment', action=actions.LoadFITS,
                        help='First moment file')
    parser.add_argument('output', nargs=1, action=actions.NormalizePath,
                        help='Output basename')
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Gradient parameters
    modulus, direction = intensity_gradient(args.moment)
    outname = f'{args.output[0]}.gradient.modulus.fits'
    modulus.writeto(outname, overwrite=True)
    outname = f'{args.output[0]}.gradient.direction.fits'
    direction.writeto(outname, overwrite=True)

    # Check inputs for stats
    source_props = source_properties(args, src_props)
    if source_props.get('radius') is not None:
        pixels_to_positions(args, wcs=WCS(args.moment).sub(2))
    else:
        args.log.info('No radius specified for statistics')
        return None

    # Stats around source
    beam = np.sqrt(Beam.from_fits_header(args.moment.header).sr / np.pi)
    radius = source_props['radius'] + beam.to(u.arcsec)
    args.log.info('Stats over radius: %s', radius)
    drc_stats = stats_at_position(direction, args.coordinate,
                                  source_props['radius'],
                                  stats=(np.nanmean, np.nanstd, np.nanmedian))
    args.log.info('Direction mean: %s', drc_stats[0])
    args.log.info('Direction std. dev.: %s', drc_stats[1])
    args.log.info('Direction median: %s', drc_stats[2])

    return drc_stats

if __name__ == '__main__':
    velocity_gradient(sys.argv[1:])
