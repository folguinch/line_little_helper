"""Calculate a line peak map."""
from typing import Sequence
from itertools import product
import argparse
import sys

from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
from spectral_cube import SpectralCube
from toolkit.argparse_tools import actions
from toolkit.argparse_tools import parents
from toolkit.astro_tools import cube_utils
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

from line_little_helper.scripts import argparse_parents
from line_little_helper.scripts import argparse_processing as processing

def save_map(data: u.Quantity,
             cube: SpectralCube,
             cubename: 'pathlib.Path',
             outdir: 'pathlib.Path',
             suffix: str):
    """Save map to disk.

    Args:
      data: data to store.
      cube: data cube.
      cubename: cube filename.
      outdir: output directory.
      suffix: file suffix.
    """
    # Header
    header = cube.wcs.sub(['longitude', 'latitude']).to_header()
    if hasattr(data, 'unit'):
        header['BUNIT'] = f'{data.unit:FITS}'

    # Data
    hdu = fits.PrimaryHDU(data.value, header=header)

    # Filename
    filename = cubename.stem + '_' + suffix + '.fits'
    filename = outdir / filename
    hdu.writeto(filename, overwrite=True)

def fit_spec(xaxis: u.Quantity, flux: u.Quantity) -> models.Gaussian1D:
    """Fit input spectrum."""
    # Model
    aux = xaxis[flux > np.max(flux)/2]
    stddev = gaussian_fwhm_to_sigma * np.abs(aux[-1] - aux[0]) / 2
    model = models.Gaussian1D(amplitude=np.max(flux), mean=xaxis[len(xaxis)//2],
                              stddev=stddev)

    # Fitter
    fitter = fitting.LevMarLSQFitter()
    model_fit = fitter(model, xaxis, flux)

    return model_fit

def fit_cube(cube: SpectralCube, vel_axis: u.Quantity, indices: np.array,
             width: int, filename: 'pathlib.Path'):
    """Fit data in cube arround input channels.

    Args:
      cube: spectral cube.
      vel_axis: velocity axis.
      indices: position of the central channels.
      width: number of channels around the central channel.
      filename: base name for the result maps and plots.
    """
    ygrid = np.arange(indices.shape[-2])
    xgrid = np.arange(indices.shape[-1])
    vel_map = np.ones(indices.shape) * np.nan
    fwhm_map = np.ones(indices.shape) * np.nan
    peak_map = np.ones(indices.shape) * np.nan
    residual = np.ones(indices.shape) * np.nan
    i, page, fig, axs = 0, 1, None, None
    for x, y in product(xgrid, ygrid):
        # Channel range
        ind = indices[y, x]
        if np.isnan(ind):
            continue
        chmin = int(ind) - width
        chmax = int(ind) + width + 1

        # Get spectra
        spec = cube.unmasked_data[chmin:chmax, y, x]
        vel = vel_axis[chmin:chmax]

        # Fit spectrum
        model = fit_spec(vel, spec)
        vel_map[y, x] = model.mean.value
        fwhm_map[y, x] = model.stddev.value * gaussian_sigma_to_fwhm
        peak_map[y, x] = model.amplitude.value
        residual[y, x] = np.sum(np.abs(model(vel).to(spec.unit).value - \
                                       spec.value))

        # Plot
        if axs is None:
            fig, axs = plt.subplots(5, 3, figsize=(20,12))
        ax = axs.flatten()[i]
        aux = np.linspace(vel[0], vel[-1], 100)
        ax.set_xlim(vel[0].value, vel[-1].value)
        ax.set_ylim(-0.01, np.max(spec).value)
        ax.plot(vel, spec, 'go')
        ax.plot(aux, model(aux), 'b-')
        if i == 14:
            figname = filename.parent / f'{filename.stem}_fit_page{page}.png'
            fig.savefig(figname)
            page += 1
            i = 0
            fig, axs = None, None
            plt.close()
        else:
            i += 1

    save_map(vel_map * model.mean.unit, cube, filename, filename.parent,
             'fit_velocity')
    save_map(fwhm_map * model.stddev.unit, cube, filename, filename.parent,
             'fit_fwhm')
    save_map(peak_map * model.amplitude.unit, cube, filename, filename.parent,
             'fit_amplitude')
    save_map(residual * cube.unit, cube, filename, filename.parent,
             'fit_residual')

def _proc(args: argparse.Namespace):
    for transition in args.molec.transitions:
        # Store linefreq
        args.log.info('Setting frequency to transition rest: %s',
                      transition.restfreq)
        args.linefreq = transition.restfreq

        # Get channel range
        processing.get_channel_range(args)

        # Spectral axis
        equiv = u.doppler_radio(args.linefreq)
        vel_axis = args.cube.spectral_axis.to(u.km/u.s,
                                              equivalencies=equiv)

        # Select channels
        ind = np.squeeze(np.zeros(vel_axis.shape, dtype=bool))
        ind[args.chan_range[0]:args.chan_range[1] + 1] = True
        masked_cube = args.cube.mask_channels(ind)
        args.log.info('Cube size: %i', args.cube.size)
        args.log.info('Valid data after spectral range: %i',
                      np.sum(ind)*masked_cube.shape[-1]*masked_cube.shape[-2])

        # FWHM in channels
        data_max_map = np.nanmax(masked_cube.filled_data[:], axis=0)
        fwhm_mask = masked_cube.filled_data[:] > 0.5 * data_max_map
        fwhm_map_chan = np.sum(fwhm_mask, axis=0)
        fwhm_map_chan = fwhm_map_chan.astype(float)

        # Threshold mask
        args.log.info('Flux limit: %f %s', args.flux_limit[0].value,
                      args.flux_limit[0].unit)
        masked_cube = masked_cube.with_mask(masked_cube > args.flux_limit[0])
        args.log.info('Valid data after threshold mask: %i',
                      np.sum(masked_cube.mask.include()))

        # Max map
        mask_bad = np.all(np.isnan(masked_cube.filled_data[:]), axis=0)
        masked_cube = masked_cube.with_fill_value(-np.inf)
        ind_map = np.nanargmax(masked_cube.filled_data[:], axis=0)
        vel_map = vel_axis[ind_map]
        vel_map[mask_bad] = np.nan

        # FWHM in velocity
        med_chan = int(np.median(ind_map[~mask_bad]))
        args.log.info('Median channel: %i', med_chan)
        chan_width = np.abs(vel_axis[:-1] - vel_axis[1:])
        chan_width = chan_width[med_chan]
        args.log.info('Channel width: %f %s', chan_width.value, chan_width.unit)
        fwhm_map = fwhm_map_chan * chan_width

        # Apply mask
        fwhm_map_chan[mask_bad] = np.nan
        fwhm_map[mask_bad] = np.nan
        ind_map = ind_map.astype(float)
        ind_map[mask_bad] = np.nan

        # Save maps
        suffix = transition.generate_name()
        save_map(ind_map * u.Unit(1), args.cube, args.cubename[0],
                 args.outdir[0], f'{suffix}_peakchannel')
        save_map(vel_map, args.cube, args.cubename[0], args.outdir[0],
                 f'{suffix}_peakvel')
        save_map(fwhm_map, args.cube, args.cubename[0], args.outdir[0],
                 f'{suffix}_fwhm')
        save_map(fwhm_map_chan * u.Unit(1), args.cube, args.cubename[0],
                 args.outdir[0], f'{suffix}_fwhmchan')

        # Calculate moments
        if args.moments is not None:
            #fwhm_mask[mask_bad] = False
            args.log.info('FWHM mask valid data: %i', np.sum(fwhm_mask))
            moment_cube = masked_cube.with_mask(fwhm_mask)
            args.log.info('Valid data for moment: %i',
                          np.sum(moment_cube.mask.include()))
            masked_chans = np.any(fwhm_mask, axis=(1,2))
            ind = np.arange(vel_axis.size)[masked_chans]
            chmin, chmax = np.min(ind), np.max(ind)
            moment_cube = moment_cube[chmin:chmax+1]
            moment_cube = cube_utils.to_common_beam(moment_cube,
                                                    log=args.log.info)
            for moment in args.moments:
                filename = f'{args.cubename[0].stem}_{suffix}_moment{moment}.fits'
                filename = args.outdir[0] / filename
                cube_utils.get_moment(moment_cube, moment, filename=filename,
                                      log=args.log.info)

        if args.fit_halfwidth[0] is not None:
            ## Build mask
            #struc = np.zeros((3,)*3, dtype=bool)
            #struc[:, 1, 1] = True
            #m, n = ind_map.shape
            #I, J = np.ogrid[:m,:n]
            #prog_mask = np.zeros(args.cube.shape, dtype=bool)
            #prog_mask[ind_map, I, J] = True
            #prog_mask[args.cube < args.flux_limit] = False
            #prog_mask = ndimg.binary_dilation(prog_mask, structure=struc,
            #                                  iterations=args.fit_halfwidth[0])

            # Fit cube
            filename = f'{args.cubename[0].stem}_{suffix}.fits'
            filename = args.outdir[0] / filename
            fit_cube(args.cube, vel_axis, ind_map, args.fit_halfwidth[0],
                     filename)

def line_peak_map(args: Sequence[str]) -> None:
    """Calculate the line peak map from command line parameters."""
    pipe = [processing.load_cube, processing.load_molecule,
            processing.set_fluxlimit, _proc]
    args_parents = [
        argparse_parents.line_parents('vlsr', 'molecule', 'flux',
                                      'spectral_range'),
        argparse_parents.cube_parent(),
        parents.logger('debug_line_peak_map.log'),
    ]
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.HelpFormatter,
        parents=args_parents,
        conflict_handler='resolve')
    parser.add_argument('--fit_halfwidth', nargs=1, type=int, default=[None],
                        help='Half window channel size for fitting')
    parser.add_argument('--moments', nargs='*', type=int,
                        help='Moments to calculate')
    parser.add_argument('outdir', nargs=1, action=actions.NormalizePath,
                        help='The output directory')
    parser.set_defaults(molec=None)
    args = parser.parse_args(args)

    for step in pipe:
        step(args)

if __name__ == '__main__':
    line_peak_map(sys.argv[1:])
