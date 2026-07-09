"""Fit functions or models to pv maps.
"""
from typing import Optional, Sequence, Tuple, Callable
from pathlib import Path
import argparse
import sys

from astropy.io import fits
from astropy.modeling import models, fitting
from toolkit import array_utils
from toolkit.argparse_tools import actions, parents
from toolkit.astro_tools import images as img_tools
from tile_plotter.multi_plotter import OTFMultiPlotter
import astropy.units as u
import numpy as np

def fit_linear(pvmap: 'astropy.io.fits',
               rms: Optional[u.Quantity] = None,
               nsigma: float = 3,
               xrange: Optional[Tuple[u.Quantity]] = None,
               yrange: Optional[Tuple[u.Quantity]] = None,
               ) -> Tuple[fitting.LinearLSQFitter, u.Quantity]:
    """Fit a linear function to the data.

    If rms is given then filter out data below `nsigma*rms`.
    
    Note that for each x-axis value a single y-value is determined by a
    intensity weighted average.

    Args:
      pvmap: Position-velocity map.
      rms: Optional. The rms of the data.
      nsigma: Optional. Sigma level for valid data.
      xrange: Optional. Range of valid x-axis values.
      yrange: Optional. Range of valid y-axis values.
    Returns:
      The fitted model.
      The coordinates of the points and errors.
    """
    # Get axes
    x, y = img_tools.get_coord_axes(pvmap)
    x = x.to(u.arcsec)
    y = y.to(u.km / u.s)
    yrep = np.tile(y[:, np.newaxis], (1, x.size)).value

    # Y-Axis range mask
    mask_yrange = np.zeros(yrep.shape, dtype=bool)
    if yrange is not None:
        ylow = np.min(yrange.to(y.unit).value)
        yup = np.max(yrange.to(y.unit).value)
        mask_yrange[yrep < ylow] = True
        mask_yrange[yrep > yup] = True

    # Data as quantity
    data = pvmap.data * u.Unit(pvmap.header['BUNIT'])
    if rms is not None:
        yrep = np.ma.masked_where((data < rms * nsigma) | mask_yrange,
                                  yrep)
        data = np.ma.masked_where((data < rms * nsigma) | mask_yrange,
                                  data.value)
        average = np.ma.average
    elif yrange is not None:
        yrep = np.ma.array(yrep, mask=mask_yrange)
        data = np.ma.array(data.value, mask=mask_yrange)
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

    # Filter if xrange is given before fit
    if xrange is not None:
        xlow = np.min(xrange.to(x.unit))
        xup = np.max(xrange.to(x.unit))
        mask_xrange = (x > xlow) & (x < xup)
        x = x[mask_xrange]
        ystd = ystd[mask_xrange]
        yavg = yavg[mask_xrange]

    # Fit function
    line_init = models.Linear1D()
    fit = fitting.LinearLSQFitter()
    try:
        fitted_line = fit(line_init, x, yavg, weights=1./ystd)
    except ValueError as exc:
        print(exc)
        fitted_line = fit(line_init, x, yavg)

    return fitted_line, x, yavg, ystd

def plot_pvmap(plot: OTFMultiPlotter,
               row: int,
               pvmap: fits.PrimaryHDU,
               data: Optional[Sequence] = None,
               model: Optional[Callable] = None,
               bunit: str = 'Jy/beam',
               rms: Optional[u.Quantity] = None):
    """Plot a pv map.

    Args:
      plot: on-the-fly multiplotter.
      row: row of the plot.
      pvmap: position-velocity map.
      data: optional; `xy` data to plot as dots.
      model: optional; model function to plot as line.
      bunit: optional; intensity scale.
      rms: optional; rms of the data for contours.
    """
    # Coodinates and location
    x, y = img_tools.get_coord_axes(pvmap)
    xfn = np.linspace(x[0], x[-1], 100).to(u.arcsec)
    loc = (row, 0)

    # Plot
    handler = plot.gen_handler(
        loc,
        'pvmap',
        include_cbar=True,
        name='Intensity',
        unit=bunit,
        xunit='arcsec',
        yunit='km/s',
        xname='Offset',
        yname='Velocity',
        xformat='{x:.1f}',
        yformat='{x:.1f}',
    )
    handler.plot_map(pvmap,
                     extent=(x[0], x[-1], y[0], y[-1]),
                     rms=rms,
                     self_contours=rms is not None,
                     contour_nsigma=5,
                     aspect='auto')

    # Additional plot
    if data is not None:
        handler.plot(data[0], data[1], 'ro')
    if model is not None:
        handler.plot(xfn, model(xfn), 'k-')

    # Configuration
    plot.apply_config(loc, handler, 'pvmap', xlim=(x[0], x[-1]),
                      ylim=(y[0], y[-1]))
    if plot.has_cbar(loc):
        handler.plot_cbar(plot.fig,
                          plot.axes[loc].cborientation)

def _fitter(args: argparse.Namespace):
    """Fit function to pv maps."""
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
        if args.rms is None and 'RMS' in pv.header:
            args.rms = float(pv.header['RMS']) * u.Unit(pv.header['BUNIT'])
            args.log.info('Using rms from header: %s', args.rms)

        # Fit by functions
        if args.function[0] == 'linear':
            model, *data = fit_linear(pv, rms=args.rms, nsigma=args.nsigma,
                                      xrange=args.xrange, yrange=args.yrange)
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
            plot_pvmap(plot, i, pv, data=data, model=model,
                       bunit=f'{args.bunit[0]}', rms=args.rms)
        elif args.function[0] == 'plot':
            plot_pvmap(plot, i, pv, bunit=f'{args.bunit[0]}', rms=args.rms)
        else:
            raise NotImplementedError((f'Function {args.function[0]} '
                                       'not implemented yet'))
    # Save plot
    plot.savefig(args.plotname)

def pvmap_fitter(args: Optional[Sequence] = None):
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
    parser.add_argument('--nsigma', type=float, default=10,
                        help='The number of sigma levels')
    parser.add_argument('--xrange', nargs=3, action=actions.ReadQuantity, default=None,
                        help='Range for x-axis points')
    parser.add_argument('--yrange', nargs=3, action=actions.ReadQuantity, default=None,
                        help='Range for y-axis points')
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
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    for step in pipe:
        step(args)

if __name__=='__main__':
    pvmap_fitter(sys.argv[1:])

