"""Analyze velocity gradients in first moment maps.

The program first determines the velocity gradient. Then it uses the input
information (zeroth moment, continuum maps or source positions and radius) to
analyze the gradients around each source.

The code generates cutouts of the data (input and generated) and plots them. It
also records `vlsr` values at any given source positions.
"""
from typing import List
import argparse
import sys

from astropy.io import fits
from astropy.table import QTable, vstack
from astropy.wcs import WCS
from toolkit.argparse_tools import actions, parents
from toolkit.astro_tools.masking import emission_mask, position_in_mask
from toolkit.astro_tools.images import (minimal_radius, emission_peaks,
                                        stats_in_beam, image_cutout,
                                        intensity_gradient)
from toolkit.converters import quantity_from_hdu, array_to_hdu
import numpy as np

def _proc(args: argparse.Namespace):
    """Process data."""
    # Load data
    moment1 = fits.open(args.moment[0])[0]
    if np.all(moment1.data == np.nan):
        args.log.warning('No valid moment 1 data')
        sys.exit()
    wcs = wcs.WCS(moment1)
    if args.continuum:
        continuum = fits.open(args.continuum[0])[0]
    else:
        continuum = None
    if args.moment_zero:
        moment_zero = fits.open(args.moment_zero[0])[0]
    else:
        moment_zero = None

    # Mask
    mask = ~np.isnan(moment1.data)
    if moment_zero is not None:
        mask = emission_mask(moment_zero, nsigma=args.nsigma,
                             initial_mask=mask, log=args.log.info)

    # Get positions
    if args.source:
        positions = [args.source.position]
        radii = [minimal_radius(moment1, positions[0])]
    elif continuum is not None:
        positions, radii = emission_peaks(continuum, min_area=1.5,
                                          log=args.log.info)
    elif moment_zero is not None:
        positions, radii = emission_peaks(moment_zero, min_area=1.5,
                                          log=args.log.info)
    else:
        raise ValueError('Could not identify any sources')

    # Iterate around positions
    table = []
    table_head = ['image', 'position', 'velocity', 'vel_std', 'radius',
                  'mean_gradient', 'mean_gradient_std', 'mean_gradient_beam',
                  'mean_gradient_beam_std', 'mean_direction',
                  'mean_direction_std', 'mean_direction_beam',
                  'mean_direction_beam_std']
    for i, position in enumerate(positions):
        # Check there is a velocity gradient at position
        if not position_in_mask(position, mask, wcs):
            args.log.info('No molecular emission at %s', position)
            table.append({
                'image': args.moment[0],
                'position': position,
            })
            continue

        # Estimate vlsr
        stats_mom1 = stats_in_beam(moment1, position, beam_radius_factor=1.5)
        args.log.info('Velocity at position: %s +/- %s', *tuple(stats_mom1))

        # Cutout around position
        args.log.info('Cutting map at %s (radius=%s)', position, radii[i])
        filename = args.moment[0].with_suffix(f'.cutout{i}.fits').name
        filename = args.outdir[0] / filename
        cutout = image_cutout(moment1, position, radii[i]*2,
                              filename=filename)
        args.log.info('Saving cutout at: %s', filename)

        # Velocity gradient
        grad, dirc = intensity_gradient(cutout)
        mean_grad = np.mean(quantity_from_hdu(grad))
        std_grad = np.std(quantity_from_hdu(grad))
        mean_dirc = np.mean(quantity_from_hdu(dirc))
        std_dirc = np.std(quantity_from_hdu(dirc))
        stats_grad = stats_in_beam(grad, position, beam_radius_factor=1.5)
        stats_dirc = stats_in_beam(dirc, position, beam_radius_factor=1.5)

        # Store in table
        table.append({
            'image': args.moment[0],
            'position': position,
            'velocity': stats_mom1[0],
            'vel_std': stats_mom1[1],
            'radius': radii[i],
            'mean_gradient': mean_grad,
            'mean_gradient_std': std_grad,
            'mean_gradient_beam': stats_grad[0],
            'mean_gradient_beam_std': stats_grad[1],
            'mean_direction': mean_dirc,
            'mean_direction_std': std_dirc,
            'mean_direction_beam': stats_dirc[0],
            'mean_direction_beam_std': stats_dirc[1],
        })

    # Save table
    table = QTable(rows=table, names=table_head)
    if args.table[0].exists():
        table_old = QTable.read(args.table[0], format='ascii.ecsv')
        table = vstack([table_old, table])
    table.write(args.table[0], format='ascii.ecsv')

def main(args: List):
    """Main program."""
    pipe = [_proc]
    args_parents = [parents.logger('debug_velocity_analyzer.log'),
                    parents.astro_source()]
    parser = argparse.ArgumentParser(
        add_help=True,
        #description=description,
        #formatter_class=HelpFormatter,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents)
    parser.add_argument('--nsigma', type=float, default=5,
                        help='Consider data over nsigma rms level.')
    parser.add_argument('--moment_zero', nargs=1, action=actions.CheckFile,
                        help='Zeroth moment file name.')
    parser.add_argument('--continuum', nargs=1, action=actions.CheckFile,
                        help='Continuum file name.')
    parser.add_argument('moment', nargs=1, action=actions.CheckFile,
                        help='First moment file name.')
    parser.add_argument('outdir', nargs=1, action=actions.MakePath,
                        help='Output directory.')
    parser.add_argument('table', nargs=1, action=actions.NormalizePath,
                        help='Output directory.')
    #parser.set_defaults(nsigma=5.)
    args = parser.parse_args(args)
    for step in pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
