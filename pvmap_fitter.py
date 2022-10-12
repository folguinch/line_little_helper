"""Fit functions or models to pv maps.
"""
from typing import Optional, Sequence, Tuple
from pathlib import Path
import argparse
import sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from toolkit import array_utils
from toolkit.argparse_tools import actions, parents
from toolkit.astro_tools import images as img_tools
from tile_plotter.multi_plotter import OTFMultiPlotter
import astropy.units as u
import numpy as np

def fit_linear(pvmap: 'astropy.io.fits',
               rms: Optional[u.Quantity] = None,
               nsigma: float = 3) -> Tuple[fitting.LinearLSQFitter, u.Quantity]:
    """Fit a linear function to the data.

    If rms is given then filter out data below `nsigma*rms`.
    
    Note that for each x-axis value a single y-value is determined by a
    intensity weighted average.

    Args:
      pvmap: position-velocity map.
      rms: optional; rms of the data.
      nsigma: optional; sigma level for valid data.
    Returns:
      The fitted model.
      The coordinates of the points and errors.
    """
    # Get axes
    x, y = img_tools.get_coord_axes(pvmap)
    x = x.to(u.arcsec)
    y = y.to(u.km / u.s)
    yrep = np.tile(y[:, np.newaxis], (1, x.size)).value

    # Data as quantity
    data = pvmap.data * u.Unit(pvmap.header['BUNIT'])
    if rms is not None:
        yrep = np.ma.masked_where(data < rms * nsigma, yrep)
        data = np.ma.masked_where(data < rms * nsigma, data.value)
        average = np.ma.average
    else:
        data = data.value
        average = np.average

    # Average along axis
    yavg = average(yrep, axis=0, weights=np.abs(data))
    ystd = average((yrep - yavg)**2, axis=0,
                    weights=np.abs(data))
    nweights = np.count_nonzero(np.abs(data), axis=0)
    ystd = np.sqrt(ystd * nweights / (nweights - 1))
    x = x[~yavg.mask]
    ystd = ystd[~yavg.mask] * y.unit
    yavg = yavg[~yavg.mask] * y.unit

    # Fit function
    line_init = models.Linear1D()
    fit = fitting.LinearLSQFitter()
    try:
        fitted_line = fit(line_init, x, yavg, weights=1./ystd)
    except ValueError as exc:
        print(exc)
        fitted_line = fit(line_init, x, yavg)

    return fitted_line, x, yavg, ystd

def _fitter(args: argparse.Namespace):
    """Fit function to ov maps."""
    # Generate plot object
    plot = OTFMultiPlotter(nrows=f'{len(args.pvmaps)}',
                           right='1.5',
                           vertical_cbar='true',
                           styles='maps viridis',
                           vcbarpos='0')

    # Iterate over pv maps
    for i, pvmap in enumerate(args.pvmaps):
        # Open map
        pv = fits.open(pvmap)[0]

        # Fit by functions
        if args.function[0] == 'linear':
            model, *data = fit_linear(pv, rms=args.rms, nsigma=10)
            args.log.info('Modeling result: %r', model)

            # Build table
            names = (pv.header['CTYPE1'].lower(),
                        pv.header['CTYPE2'].lower(),
                        pv.header['CTYPE2'].lower() + '_err')
            args.log.info('Data names: %s', names)
            data_table, meta = array_utils.to_struct_array(*data,
                                                           names=names)

            # Save
            filename = args.outdir / f'{pvmap.stem}_averaged.dat'
            array_utils.save_struct_array(filename, data_table,
                                          meta['units'])
            args.log.info('Data saved at: %s', filename)

            # Save results
            model_results = args.outdir / f'{pvmap.stem}_linear_fit.log'
            lines = ['Model parameters:',
                     f'Slope: {model.slope.value} {model.slope.unit}',
                     (f'Intercept: {model.intercept.value}'
                      f'{model.intercept.unit}')]
            model_results.write_text('\n'.join(lines))

            # Plot
            x, y = img_tools.get_coord_axes(pv)
            xfn = np.linspace(x[0], x[-1], 100).to(u.arcsec)
            loc = (i, 0)
            handler = plot.gen_handler(
                loc,
                'pvmap',
                include_cbar=True,
                bunit=f'{args.bunit[0]}',
                xunit='arcsec',
                yunit='km/s',
                xname='Offset',
                yname='Velocity',
            )
            handler.plot_map(pv,
                             extent=(x[0], x[-1], y[0], y[-1]),
                             rms=args.rms,
                             self_contours=args.rms is not None,
                             contour_nsigma=5,
                             aspect='auto')
            handler.plot(data[0], data[1], 'ro')
            handler.plot(xfn, model(xfn), 'k-')
            plot.apply_config(loc, handler, 'pvmap', xlim=(x[0], x[-1]),
                              ylim=(y[0], y[-1]))
            if plot.has_cbar(loc):
                handler.plot_cbar(plot.fig,
                                  orientation=plot.axes[loc].cborientation)

        else:
            raise NotImplementedError((f'Function {args.function[0]} '
                                       'not implemented yet'))
    # Save plot
    plot.savefig(args.plotname)

def pvmap_fitter(args: Sequence):
    """Fit a function to input pv maps.

    Args:
      args: command line arguments.
    """
    pipe = [_fitter]
    args_parents = [parents.logger('debug_line_helper.log')]
    parser = argparse.ArgumentParser(
        add_help=True,
        parents=args_parents,
        conflict_handler='resolve',
    )
    parser.add_argument('--function', nargs=1, default=['linear'],
                        help='Function or model to fit')
    parser.add_argument('--rms', action=actions.ReadQuantity, default=None,
                        help='The rms of the data')
    parser.add_argument('--bunit', nargs=1, default=['Jy/beam'],
                        help='Intensity unit')
    parser.add_argument('--plotname', action=actions.NormalizePath,
                        default=Path('./pvmaps.png'),
                        help='Plot file name')
    parser.add_argument('--outdir', action=actions.NormalizePath,
                        default=Path('./'),
                        help='Output directory')
    parser.add_argument('pvmaps', action=actions.CheckFile, nargs='+',
                        help='Position-velocity files')
    parser.set_defaults()
    args = parser.parse_args()

    for step in pipe:
        step(args)


if __name__=='__main__':
    pvmap_fitter(sys.argv[1:])

