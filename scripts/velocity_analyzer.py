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
from toolkit.astro_tools.masking import (emission_mask, position_in_mask,
                                         plot_mask)
from toolkit.astro_tools.images import (identify_structures, stats_in_beam,
                                        image_cutout, intensity_gradient,
                                        get_peak, positions_in_image)
from toolkit.converters import quantity_from_hdu
from toolkit.maths import quick_rms
import astropy.units as u
import numpy as np

from line_little_helper.plot_tools import plot_map

def _proc(args: argparse.Namespace):
    """Process data."""
    # Load data
    moment1 = fits.open(args.moment[0])[0]
    if np.all(moment1.data == np.nan):
        args.log.warning('No valid moment 1 data')
        sys.exit()
    wcs = WCS(moment1, naxis=['longitude', 'latitude'])
    if args.continuum:
        continuum = fits.open(args.continuum[0])[0]
    else:
        continuum = None
    if args.moment_zero:
        moment_zero = fits.open(args.moment_zero[0])[0]
    else:
        moment_zero = None

    # Identify regions with valid moment 1 data
    mask = ~np.isnan(moment1.data)
    figname = args.moment[0].with_suffix('.structures.png').name
    figname = args.outdir[0] / figname
    centroids, lengths = identify_structures(moment1, mask=mask, min_area=1.5,
                                             plot=figname, log=args.log.info)

    # Molecular emission mask
    if moment_zero is not None:
        mask = emission_mask(moment_zero, nsigma=args.nsigma,
                             initial_mask=mask, log=args.log.info)
        figname = args.moment[0].with_suffix('.mom0.mask.png').name
        figname = args.outdir[0] / figname
        fig, _ = plot_mask(mask, scatter=centroids, wcs=wcs)
        fig.savefig(figname)

    # Continuum emission
    if continuum is not None:
        sigma_cont = quick_rms(continuum.data)
    else:
        sigma_cont = None

    # Get positions
    if args.source:
        positions = [args.source[0].position]
    else:
        positions = None

    # Iterate around positions
    table = []
    table_head = ['image', 'centroid', 'lenx', 'leny', 'position', 'velocity',
                  'vel_std', 'mean_gradient', 'mean_gradient_std',
                  'mean_gradient_beam', 'mean_gradient_beam_std',
                  'mean_direction', 'mean_direction_std', 'mean_direction_beam',
                  'mean_direction_beam_std']
    for i, (centroid, length) in enumerate(zip(centroids, lengths)):
        # Check there is molecular line emission at centroid
        if not position_in_mask(centroid, mask, wcs):
            args.log.info('No molecular emission at %s', centroid)
            continue

        # Get true position
        args.log.info('Cutting map at %s (%s x %s)', centroid, *length)
        filename = args.moment[0].with_suffix(f'.cutout{i}.fits').name
        filename = args.outdir[0] / filename
        cutout = image_cutout(moment1, centroid, length, filename=filename)
        args.log.info('Cutout saved at: %s', filename)
        if continuum is not None:
            cutout_cont = image_cutout(continuum, centroid, length)
            # Check if there is continuum emission in cutout
            if not np.any(cutout_cont.data > args.nsigma * sigma_cont):
                args.log.info('No continuum emission at %s', centroid)
                continue
            position, _ = get_peak(cutout_cont)
            args.log.info('Continuum peak position: %s', position)
            pos_list = [position]
        elif positions is not None:
            pos_list = positions_in_image(positions, cutout)
            args.log.info('Input positions in cutout: %r', pos_list)
        elif moment_zero is not None:
            cutout_mol = image_cutout(moment_zero, centroid, length)
            position, _ = get_peak(cutout_mol)
            args.log.info('Zeroth moment peak position: %s', position)
            pos_list = [position]
        else:
            args.log.info('Using centroid as position')
            pos_list = [centroid]

        # Stats at each position
        for j, position in enumerate(pos_list):
            # Estimate vlsr
            stats_mom1 = stats_in_beam(moment1, position,
                                       beam_radius_factor=1.5)
            args.log.info('Velocity at position: %s +/- %s', *tuple(stats_mom1))

            # Velocity gradient
            grad, dirc = intensity_gradient(cutout)
            mean_grad = np.nanmean(quantity_from_hdu(grad))
            std_grad = np.nanstd(quantity_from_hdu(grad))
            mean_dirc = np.nanmean(quantity_from_hdu(dirc))
            std_dirc = np.nanstd(quantity_from_hdu(dirc))
            stats_grad = stats_in_beam(grad, position, beam_radius_factor=1.5)
            stats_dirc = stats_in_beam(dirc, position, beam_radius_factor=1.5)

            # Sve cutouts
            aux = filename.with_suffix(f'.pos{j}.gradient.fits')
            grad.writeto(aux, overwrite=True)
            aux = filename.with_suffix(f'.pos{j}.direction.fits')
            dirc.writeto(aux, overwrite=True)

            # Store in table
            table.append({
                'image': args.moment[0],
                'centroid': centroid,
                'lenx': length[1],
                'leny': length[0],
                'position': position,
                'velocity': stats_mom1[0],
                'vel_std': stats_mom1[1],
                'mean_gradient': mean_grad,
                'mean_gradient_std': std_grad,
                'mean_gradient_beam': stats_grad[0],
                'mean_gradient_beam_std': stats_grad[1],
                'mean_direction': mean_dirc,
                'mean_direction_std': std_dirc,
                'mean_direction_beam': stats_dirc[0],
                'mean_direction_beam_std': stats_dirc[1],
            })

            # Plot cutout
            figname = args.moment[0].with_suffix(f'.cutout{i}.pos{j}.png').name
            figname = args.outdir[0] / figname
            plotter, handler = plot_map(cutout, stats=table[-1],
                                        styles='bwr', xformat='hh:mm:ss.ss',
                                        yformat='dd:mm:ss.s', ticks_color='k',
                                        right='1.0', label_xpad='1.',
                                        label_ypad='-0.7', bname='Velocity')
            if continuum is not None:
                cont_unit = u.Unit(continuum.header['BUNIT'])
                handler.plot_contours(cutout_cont, rms=sigma_cont*cont_unit,
                                      ignore_units=True, colors='g', zorder=2)
            plotter.savefig(figname)

    # Save table
    if len(table) == 0:
        args.log.warn('No structure detected')
        return
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
                        help='Output table ecsv file.')
    #parser.set_defaults(nsigma=5.)
    args = parser.parse_args(args)
    for step in pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
