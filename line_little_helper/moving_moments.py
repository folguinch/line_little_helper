#!/bin/python3
"""Moment 0/1 maps from incresing local window size.

The input `winwidth` value determines the working window. The input `molecule`
determines the central channel of the `winwidth` (correcting for the LSR
velocity).

The program has two separate modes:

1. Incremental moment 1: if `split` parameters are omitted, the program will
  calculate the velocity at the peak line emission for each pixel. Then will
  increase the number of channels around the peak intensity to calculate first
  moment maps, until the `winwidth` is covered.

2. Split mode: if `split` parameters are given, the program will run two steps:
  incremental and rolling step. In the incremental step, the program will
  calculate zeroth order moments in two windows at both sides of the line.
  The `WIDTH` parameter determines the channels around the line which are
  ignored (the line frequency is at the center of the `WIDTH` window). The
  initial size of the windows where the moments are calculated is given by the
  `WIN` paramter and will be increased by `INCR` until the `winwidth` is
  covered. In the rolling step, zeroth order moment maps are calculated each
  side of the line in window with `WIN` channels. The windows then roll further
  from the central channel by `ROLL` channels until the `winwidth` is covered.
"""
from pathlib import Path
from typing import TypeVar, Optional, List, Sequence
import argparse
import sys
import textwrap

from astropy.io import fits
from toolkit.argparse_tools import actions
from toolkit.astro_tools.cube_utils import (get_restfreq, get_cube_rms,
                                            to_common_beam)
import astropy.units as u
import toolkit.argparse_tools.loaders as aploaders
import toolkit.argparse_tools.parents as apparents
import numpy as np
import scipy.ndimage as ndimg

from line_little_helper.argparse_parents import line_parents
from line_little_helper.processing_tools import to_rest_freq
from line_little_helper.argparse_processing import load_molecule

Cube = TypeVar('Cube')
Logger = TypeVar('Logger')

class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    pass

def get_header(cube: Cube, bunit: u.Unit = None) -> fits.Header:
    """Extract 2-D header from cube header.

    Args:
      cube: spectral cube.
      bunit: flux unit.

    Return:
      A FITS header object.
    """
    # Header
    header = cube.wcs.sub(['longitude', 'latitude']).to_header()

    # Unit
    if bunit is not None:
        header['BUNIT'] = f'{bunit:FITS}'

    # Copy other cards
    cards = ['BMIN', 'BMAJ', 'BPA']
    for card in cards:
        if card in cube.header:
            header[card] = cube.header[card]

    return header

def save_mask(mask: np.array, header: fits.Header, filename: Path,
              dtype: Optional = int) -> None:
    """Save masks.

    Args:
      mask: mask array.
      header: FITS header.
      filename: file name.
      dtype: optional; data type.
    """
    hdu = fits.PrimaryHDU(mask.astype(dtype))
    hdu.header = header
    hdu.writeto(filename, overwrite=True)

def cube_to_marray(cube: Cube, unit: Optional[u.Unit] = None) -> np.ma.array:
    """Convert an `SpectralCube` to `np.ma.array`.

    Args:
      cube: spectral cube.
      unit: optional; units of the output data.

    Returns:
      A numpy masked array with the data of the cube.
    """
    # Check units
    if unit is None:
        unit = cube.unit

    # Convert to masked array
    data = cube.filled_data[:].to(unit).value
    data = np.ma.array(data, mask=np.isnan(data))

    return data

def max_vel_ind(data: np.array, headers: List[fits.Header],
                vel: u.Quantity, output: Path) -> np.array:
    """Calculate a line peak index and peak velocity maps.

    Args:
      data: an array with the data.
      headers: two headers, one for the index map and one for the velocity map.
      vel: velocity axis.
      output: output base name
    """
    mask = np.all(data.mask, axis=0)
    maxind = np.nanargmax(data, axis=0)
    maxvel = vel[maxind]
    maxvel[mask] = np.nan

    # Save
    out = str(output) + '_%s.fits'
    hdu = fits.PrimaryHDU(maxind)
    hdu.header = headers[0]
    hdu.writeto(f'{output}_spec_peak_index.fits', overwrite=True,
                output_verify='fix')
    hdu = fits.PrimaryHDU(maxvel.value)
    hdu.header = headers[1]
    hdu.writeto(f'{output}_spec_peak_vel.its', overwrite=True,
                output_verify='fix')

    return maxind

def full_split_moments(cube: Cube,
                       split_width: int,
                       split_win: int,
                       outdir: Path,
                       basename: str,
                       vlsr: u.Quantity,
                       incremental_step: Optional[int] = 2,
                       roll_step: Optional[int] = 2,
                       log: Optional[Logger] = None,) -> None:
    """Calculate moment 0 at mirrored windows arround line center.

    The spectral axis is divided in 2 bands (lower band `lb` and upper band
    `ub`). Two steps are performed in each band:

    1. Incremental step: the mirrored windows are expanded towards lower and
      higher channels in the `lb` and `ub`, respectively. The amount of
      increment is given by `incremental_step`.
    2. Rolling step: the mirrored windows are rolled (keeping the `split_win`
      size constant) towards lower and higher channels in the `lb` and `ub`,
      respectively. The amount of rolling is given by `roll_step`.

    Args:
      cube: spectral cube with spectral axis in systemic velocity units.
      split_width: size of the window around the line center to ignore (in
        channels).
      split_win: size of the window where the moment 0 are calculated.
      outdir: path to save the output images.
      basename: base name for the images.
      vlsr: LSR velocity.
      incremental_step: optional; amount of increment for the incremental step.
      roll_step: optional; number of channels to roll the split windows.
      log: optional; logger.
    """
    # Index of the minimum velocity
    vel = cube.spectral_axis - vlsr
    vel = vel.to(u.km / u.s)
    ind = np.nanargmin(np.abs(vel))

    # Half split width
    half_split_width = split_width // 2
    if log is not None:
        log.info(f'Spectral axis length: {vel.size}')
        log.info(f'Line central channel: {ind}')
        log.info(f'Half split width: {half_split_width}')

    # Central limits
    max_lb = ind - half_split_width
    min_ub = ind + half_split_width + 1

    # Colors
    if vel[max_lb] < vel[min_ub]:
        color_lb = 'blue'
        color_ub = 'red'
    else:
        color_lb = 'red'
        color_ub = 'blue'

    # Increasing windows size step
    range_min_lb = np.arange(max_lb - split_win, 0, -incremental_step)
    range_max_ub = np.arange(min_ub + split_win, len(vel), incremental_step)
    for min_lb, max_ub in zip(range_min_lb, range_max_ub):
        # Subcubes
        vel_rng_lb = (f'{vel[min_lb].value} -- '
                      f'{vel[max_lb - 1].value} {vel.unit}')
        vel_rng_ub = (f'{vel[min_ub].value} -- '
                      f'{vel[max_ub - 1].value} {vel.unit}')
        if log is not None:
            print('-' * 50)
            annimate = np.array(['=' if i != ind else '|'
                                 for i in range(len(vel))])
            annimate[min_lb:max_lb] = '#'
            annimate[min_ub:max_ub] = '#'
            log.info('Calculating moments for ranges:')
            log.info(f'Lower band: {min_lb} -- {max_lb - 1}')
            log.info(f'            {vel_rng_lb}')
            log.info(f'Upper band: {min_ub} -- {max_ub - 1}')
            log.info(f'            {vel_rng_ub}')
            log.info(''.join(annimate))
        aux_lb = cube[min_lb:max_lb, :, :]
        aux_ub = cube[min_ub:max_ub, :, :]

        # Moment 0
        aux_lb = aux_lb.moment(order=0).hdu
        aux_ub = aux_ub.moment(order=0).hdu

        # Store velocity range in header
        aux_lb.header['comment'] = f'Velocity range: {vel_rng_lb}'
        aux_ub.header['comment'] = f'Velocity range: {vel_rng_ub}'

        # Save
        filename = (f'{basename}_moment0_incremental_{color_lb}_'
                    f'{min_lb:02d}-{max_lb - 1:02d}.fits')
        aux_lb.writeto(outdir / filename, overwrite=True)
        filename = (f'{basename}_moment0_incremental_{color_ub}_'
                    f'{min_ub:02d}-{max_ub - 1:02d}.fits')
        aux_ub.writeto(outdir / filename, overwrite=True)

    # Rolling windows step
    range_min_lb = np.arange(max_lb - split_win, 0, -roll_step)
    range_max_ub = np.arange(min_ub + split_win, len(vel), roll_step)
    for min_lb, max_ub in zip(range_min_lb, range_max_ub):
        # Subcubes
        max_lb = min_lb + split_win
        min_ub = max_ub - split_win
        vel_rng_lb = (f'{vel[min_lb].value} -- '
                      f'{vel[max_lb - 1].value} {vel.unit}')
        vel_rng_ub = (f'{vel[min_ub].value} -- '
                      f'{vel[max_ub - 1].value} {vel.unit}')
        if log is not None:
            print('-' * 50)
            annimate = np.array(['=' if i != ind else '|'
                                 for i in range(len(vel))])
            annimate[min_lb:max_lb] = '#'
            annimate[min_ub:max_ub] = '#'
            log.info('Calculating moments for ranges:')
            log.info(f'Lower band: {min_lb} -- {max_lb - 1}')
            log.info(f'Upper band: {min_ub} -- {max_ub - 1}')
            log.info(''.join(annimate))
        aux_lb = cube[min_lb:max_lb, :, :]
        aux_ub = cube[min_ub:max_ub, :, :]

        # Moment 0
        aux_lb = aux_lb.moment(order=0).hdu
        aux_ub = aux_ub.moment(order=0).hdu

        # Store velocity range in header
        aux_lb.header['comment'] = f'Velocity range: {vel_rng_lb}'
        aux_ub.header['comment'] = f'Velocity range: {vel_rng_ub}'

        # Save
        filename = (f'{basename}_moment0_rolling_{color_lb}_'
                    f'{min_lb:02d}-{max_lb - 1:02d}.fits')
        aux_lb.writeto(outdir / filename, overwrite=True)
        filename = (f'{basename}_moment0_rolling_{color_ub}_'
                    f'{min_ub:02d}-{max_ub - 1:02d}.fits')
        aux_ub.writeto(outdir / filename, overwrite=True)

def full_incremental_moments(cube: Cube,
                             outdir: Path,
                             transition: str,
                             steps: Optional[Sequence[int]] = (1, 2, 3,
                                                               5, 8, 10),
                             log: Optional[Logger] = None,
                             save_masks: bool = False) -> None:
    """Calculate moments at increasing window sizes.

    The function locates the line peaks per pixel, and computes moment maps by
    delating along the spectral axis around the peak. The number of iterations
    for the dilation function are given in `steps`.

    Args:
      cube: spectral cube with spectral axis in velocity.
      outdir: output directory.
      transition: transition name.
      steps: optional; list of numbers of iterations for dilation.
      log: optional; logger.
      save_masks: optional; save masks at each step.
    """
    # Velocity axis
    if log is not None:
        log.info('Calculating peak velocity map')
    vel = cube.spectral_axis
    dvel = np.abs(vel[0] - vel[1])
    data = cube_to_marray(cube)

    # Max index along spectral axis
    headers = [get_header(cube), get_header(cube, bunit=vel.unit)]
    maxind = max_vel_ind(data, headers, vel, outdir / transition)

    # Binary structure for dilating only along spectral axis
    binstruc = np.zeros((3,)*3, dtype=bool)
    binstruc[:, 1, 1] = True

    # Grids to evaluate mask
    m, n = maxind.shape
    ii, jj = np.ogrid[:m,:n]

    # Auxiliary mask
    mask = np.zeros(data.shape, dtype=bool)

    # Set max values to True
    mask[maxind, ii, jj] = True
    mask[0][np.all(data.mask, axis=0)] = False
    if save_masks:
        if log is not None:
            log.info('Saving initial mask')
        filename = outdir / f'{transition}_moment1_initial_mask.fits'
        save_mask(mask, cube.header, filename)

    # Iterate over steps values
    for iterations in steps:
        # Dilate mask
        width = dvel * (iterations*2 + 1)
        if log is not None:
            log.info('-' * 50)
            log.info(f'Dilating {iterations} times')
            log.info(f'Width of window: {width.value} {width.unit}')
        aux = ndimg.binary_dilation(mask,
                                    structure=binstruc,
                                    iterations=iterations)

        # Those in the dialted mask with values over threshold
        aux = aux & ~data.mask

        # Save mask
        if save_masks:
            filename = f'{transition}_moment1_dilate{iterations}_mask.fits'
            save_mask(aux, cube.header, outdir / filename)

        # Subcube
        aux = cube.with_mask(aux)
        aux = aux.moment(order=1)
        filename = f'{transition}_moment1_dilate{iterations}.fits'
        aux.write(outdir / filename, overwrite=True)

def _preproc(args: argparse.Namespace) -> None:
    """Generate the necessary objects to process the data."""
    # Load cube
    args.log.info('Loading cube')
    args.cube(args, args.cubename)
    args.cube = args.cube.with_spectral_unit(u.GHz)

    # Load molecule
    args.mol = load_molecule(args)

def _proc(args: argparse.Namespace) -> None:
    """Process the cube."""
    # Spectral axis
    spectral_axis = args.cube.spectral_axis.to(u.GHz)

    # RMS
    if args.rms is not None:
        rms = args.rms
        args.log.info(f'Using rms: {rms.value} {rms.unit}')
    else:
        rms = get_cube_rms(args.cube, use_header=True,
                           sampled=args.sampled_rms, log=args.log.info)
        args.log.info(f'Cube rms: {rms.value} {rms.unit}')

    # Iterate over transitions
    for transition in args.mol.transitions:
        args.log.info('='*50)
        args.log.info((f'Working on transition: '
                       f'{transition.species} {transition.qns}'))
        args.log.info((f'Observed freq.: '
                       f'{transition.obsfreq.value} {transition.obsfreq.unit}'))

        # Get channel range
        diff = np.abs(spectral_axis - transition.obsfreq)
        ind = np.nanargmin(diff)
        chan1, chan2 = ind - args.winwidth[0]//2, ind + args.winwidth[0]//2
        chan1 = max(chan1, 0)
        chan2 = min(chan2, len(spectral_axis) - 1)
        args.log.info(f'Closest channel to line transition: {ind}')
        args.log.info(f'Channel range: {chan1} -- {chan2}')

        # Subcube
        subcube = args.cube[chan1:chan2+1,:,:]
        subcube = subcube.with_spectral_unit(u.km/u.s,
                                             velocity_convention='radio',
                                             rest_value=transition.restfreq)

        # Common beam
        subcube = to_common_beam(subcube, log=args.log.info)

        # Calculate results
        if args.split is not None:
            args.log.info('Split window set:')
            split_wind_width, split_win = args.split
            incremental_step, roll_step = args.split_steps
            args.log.info(f'Split width: {split_wind_width}')
            args.log.info(f'Split min window width: {split_win}')
            args.log.info(f'Split incremental step: {incremental_step}')
            args.log.info(f'Split rolling step: {roll_step}')
            full_split_moments(subcube, split_wind_width, split_win,
                               args.outdir[0], transition.generate_name(),
                               args.vlsr, incremental_step=incremental_step,
                               roll_step=roll_step, log=args.log)
        else:
            # Create cube mask
            args.log.info('Filtering out data < %i rms', args.nsigma)
            mask = subcube > rms * args.nsigma
            if not mask.any():
                args.log.info('No data over threshold, skipping')
                continue
            subcube = subcube.with_mask(mask)

            # Shrink cube
            if args.shrink:
                args.log.info('Shrinking subcube')
                subcube.allow_huge_operations = True
                subcube = subcube.minimal_subcube()

            # Calculate moments
            full_incremental_moments(subcube.to(rms.unit), args.outdir[0],
                                     transition.generate_name(), log=args.log,
                                     save_masks=args.savemasks)

def main(args: Optional[Sequence[str]] = None) -> None:
    """Moment 1 maps from different windows.

    Args:
      args: arguments for argparse.
    """
    # Parser
    description = textwrap.dedent("""\
        Moment 0/1 maps from incresing local window size. The input `winwidth`
        value determines the working window. The input `molecule` determines
        the central channel of the `winwidth` (correcting for the LSR velocity).

        The program has two separate modes:

        1. Incremental moment 1: if `split` parameters are omitted, the program
        will calculate the velocity at the peak line emission for each pixel.
        Then will increase the number of channels around the peak intensity to
        calculate first moment maps, until the `winwidth` is covered.

        2. Split mode: if `split` parameters are given, the program will
        run two steps: incremental and rolling step. In the incremental step,
        the program will calculate zeroth order moments in two windows at both
        sides of the line. The `WIDTH` parameter determines the channels around
        the line which are ignored (the line frequency is at the center of the
        `WIDTH` window). The initial size of the windows where the moments are
        calculated is given by the `WIN` paramter and will be increased by
        `INCR` until the `winwidth` is covered. In the rolling step, zeroth
        order moment maps are calculated each side of the line in window with
        `WIN` channels. The windows then roll further from the central channel 
        by `ROLL` channels until the `winwidth` is covered.""")
    pipe = [_preproc, _proc]
    args_parents = [apparents.logger('debug_moving_moments.log'),
                    line_parents('vlsr', 'molecule', 'flux')]
    parser = argparse.ArgumentParser(
        add_help=True,
        description=description,
        formatter_class=HelpFormatter, #argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents)
    parser.add_argument('--shrink', action='store_true',
                        help='Shrink to minimal cube.')
    parser.add_argument('--savemasks', action='store_true',
                        help='Save masks at each step.')
    parser.add_argument('--split', metavar=('WIDTH', 'WIN'), nargs=2, type=int,
                        help='Width of split window and moment 0 window.')
    parser.add_argument('--split_steps', metavar=('INCR', 'ROLL'), nargs=2,
                        type=int, default=[2, 2],
                        help='Steps for incremental and rolling steps.')
    parser.add_argument('winwidth', nargs=1, type=int,
                        help='Window width in channels.')
    #group1 = parser.add_mutually_exclusive_group(required=True)
    #parser.add_argument('lineconfig', nargs=1, action=actions.CheckFile,
    #                    help='Line configuration file')
    #parser.add_argument('section', nargs=1,
    #                    help='Configuration section')
    parser.add_argument('outdir', nargs=1, action=actions.MakePath,
                        help='Output directory.')
    parser.add_argument('cubenames', nargs='*', action=actions.CheckFile,
                        help='Cube file name(s).')
    parser.set_defaults(pipe=pipe, results=None, cubename=None,
                        cube=aploaders.load_spectral_cube, mol=None,
                        nsigma=5.)
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)
    for cube in args.cubenames:
        args.log.info(f'Working on cube: {cube}')
        args.cubename = cube
        for fn in args.pipe:
            fn(args)
        args.cube = aploaders.load_spectral_cube

if __name__=='__main__':
    main(sys.argv[1:])
