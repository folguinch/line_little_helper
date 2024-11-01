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
    info = extract_cube_info(args.cube, vlsr=args.vlsr)
    print(info)

def extract_cube_info(cube: SpectralCube,
                      vlsr: Optional[u.Quantity] = None) -> str:
    """List cube basic information to print."""
    # Store everything in a list
    info = []

    # Extract shape information
    info.append(f'Number of channels: {cube.spectral_axis.size}')
    info.append(f'Map size: {cube.shape[-1]}x{cube.shape[-2]}')

    # Beams and rms
    if 'RMS' in cube.header:
        info.append(f"Cube rms: {cube.header['RMS']}")
    try:
        smallest, largest = cube.beams.extrema_beams()
        info.append('Multi-beam cube')
        info.append(f'Smallest beam: {smallest}')
        info.append(f'Largest beam: {largest}')
        info.append(f'Common beam: {cube.beams.common_beam()}')
    except AttributeError:
        info.append('Single-beam cube')
        info.append(f'Beam size: {cube.beam}')

    # Spectral information
    restfreq = cbutils.get_restfreq(cube)
    restfreq = restfreq.to(u.GHz)
    info.append(f'Rest frequency: {restfreq.value} {restfreq.unit}')
    extrema = cube.spectral_extrema
    info.append(('Observed frequency range: '
                 f'{extrema[0].value}-{extrema[1].value} extrema[0].unit'))
    axis = cube.spectral_axis
    freqwidth = np.median(np.abs(axis[1:] - axis[:-1])).to(u.kHz)
    axis = axis.to(u.km/u.s, equivalencies=u.doppler_radio(restfreq))
    velwidth = np.median(np.abs(axis[1:] - axis[:-1]))
    info.append(('Median channel width: '
                 f'{.3:freqwidth.value} {freqwidth.unit} '
                 f'({.2:velwidth.value} {velwidth.unit})'))
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
    parser.add_argument('cubename', nargs=1, action=actions.CheckFile,
                        help='The input cube file name')
    parser.set_defaults(cube=None)
    args = parser.parse_args(args)
    for step in pipe:
        step(args)

if __name__ == '__main__':
    cube_info(sys.argv[1:])

