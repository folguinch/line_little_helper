#!/bin/python3
"""List basic cube information.

Data that will be printed:
  - Number of channels
  - Map size
  - Cube rms (if in header)
  - Type of beam
  - Beam size (if single beam)
  - First, last and common beam (if multi beam)
  - Rest frequency
  - Observed frequency range
  - Rest frequency range (if `vlsr` is provided)
"""
from typing import Optional, Sequence, TypeVar
import argparse
import sys

from toolkit.argparse_tools import actions
import astropy.units as u
import numpy as np
import toolkit.argparse_tools.loaders as aploaders
import toolkit.astro_tools.cube_utils as cbutils

from line_little_helper.moving_moments import HelpFormatter
from line_little_helper.argparse_parents import line_parents

SpectralCube = TypeVar('SpectralCube')

def _print_info(args):
    """Print information from cube in args."""
    info = extract_cube_info(args.cube, vlsr=args.vlsr, calc_rms=args.get_rms)
    print(info)

def extract_cube_info(cube: SpectralCube,
                      vlsr: Optional[u.Quantity] = None,
                      calc_rms: bool = False) -> str:
    """List cube basic information to print."""
    # Store everything in a list
    info = []

    # Extract shape information
    info.append(f'Number of channels: {cube.spectral_axis.size}')
    info.append(f'Map size: {cube.shape[-1]}x{cube.shape[-2]}')

    # Beams and rms
    if 'RMS' in cube.header:
        info.append(f"Cube rms: {cube.header['RMS']}")
    elif calc_rms:
        rms = get_cube_rms(cube, log=lambda x: f'{x}').to(u.mJy/u.beam)
        info.append(f'Cube rms: {rms}')
    try:
        smallest, largest = cube.beams.extrema_beams()
        common = cube.beams.common_beam()
        smallest = (f'{smallest.major.to(u.arcsec).value:.5f} x '
                    f'{smallest.minor.to(u.arcsec):.5f} '
                    f'PA={smallest.pa.to(u.deg):.1f} '
                    f'({smallest.to(u.arcsec**2)})')
        largest = (f'{largest.major.to(u.arcsec).value:.5f} x '
                   f'{largest.minor.to(u.arcsec):.5f} '
                   f'PA={largest.pa.to(u.deg):.1f} '
                   f'({largest.to(u.arcsec**2)})')
        common = (f'{common.major.to(u.arcsec).value:.5f} x '
                  f'{common.minor.to(u.arcsec):.5f} '
                  f'PA={common.pa.to(u.deg):.1f} '
                  f'({common.to(u.arcsec**2)})')
        info.append('Multi-beam cube')
        info.append(f'Smallest beam: {smallest}')
        info.append(f'Largest beam: {largest}')
        info.append(f'Common beam: {common}')
    except AttributeError:
        info.append('Single-beam cube')
        info.append(f'Beam size: {cube.beam}')

    # Spectral information
    restfreq = cbutils.get_restfreq(cube)
    restfreq = restfreq.to(u.GHz)
    info.append(f'Rest frequency: {restfreq.value} {restfreq.unit}')
    extrema = cube.spectral_extrema.to(u.GHz)
    info.append(('Observed frequency range: '
                 f'{extrema[0].value}-{extrema[1].value} {extrema[0].unit}'))
    axis = cube.spectral_axis
    freqwidth = np.median(np.abs(axis[1:] - axis[:-1])).to(u.kHz)
    axis = axis.to(u.km/u.s, equivalencies=u.doppler_radio(restfreq))
    velwidth = np.median(np.abs(axis[1:] - axis[:-1]))
    info.append(('Median channel width: '
                 f'{freqwidth.value:.3f} {freqwidth.unit} '
                 f'({velwidth.value:.2f} {velwidth.unit})'))
    if vlsr is not None:
        axis = axis - vlsr
        axis = axis.to(u.GHz, equivalencies=u.doppler_radio(restfreq))
        info.append(('Rest frequency range: '
                     f'{axis[0].value}-{axis[-1].value} {extrema[0].unit}'))

    return '\n'.join(info)

def cube_info(args: Optional[Sequence[str]] = None) -> None:
    """Main program.

    Args:
      args: arguments for argparse.
    """
    # Argument parser
    parents = [line_parents('vlsr')]
    pipe = [aploaders.load_spectral_cube, _print_info]
    parser = argparse.ArgumentParser(
        add_help=True,
        parents=parents,
        formatter_class=HelpFormatter,
        conflict_handler='resolve',
    )
    parser.add_argument('--get_rms', action='store_true',
                        help='Calculate cube rms')
    parser.add_argument('cubename', nargs=1, action=actions.CheckFile,
                        help='The input cube file name')
    parser.set_defaults(cube=None)
    args = parser.parse_args(args)
    for step in pipe:
        step(args)

if __name__ == '__main__':
    cube_info(sys.argv[1:])

