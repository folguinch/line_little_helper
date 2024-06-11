"""Commonly use `arparse` parents."""
from typing import Sequence, Union
import argparse

from toolkit.argparse_tools import actions

from .argparse_processing import get_freqrange

def cube_parent(nargs: Union[int, str] = 1) -> argparse.ArgumentParser:
    """Generate a parser to read cube filenames.

    It creates the `cubename` and the default `cube` attributes.
    If `nargs` is larger than 1 or in `[*, +]`, then a `cubenames` attibute
    is generated and the `cubename` attribute is set to `None`.
    If nargs is 0 then no cubename argument is generated, i.e. only `use_dask`
    and `common_beam` arguments are generated.
    """
    parser = argparse.ArgumentParser(add_help=False)

    if nargs in ['*', '+'] or nargs > 1:
        parser.add_argument('cubenames', nargs=nargs, action=actions.CheckFile,
                            help='Cube file name(s)')
        parser.set_defaults(cubename=None, cube=None)
    elif nargs == 1:
        parser.add_argument('cubename', nargs=nargs, action=actions.CheckFile,
                            help='Cube file name')
        parser.set_defaults(cube=None)
    elif nargs == 0:
        pass
    else:
        raise ValueError(f'nargs {nargs} not recognized')
    parser.add_argument('--use_dask', action='store_true',
                        help='Use dask for cube')
    parser.add_argument('--common_beam', action='store_true',
                        help='Convolve with common beam before results')
    #parser.add_argument('--shrink', action='store_true',
    #                    help='Reduce cube size to fit the FOV or mask')
    parser.add_argument('--mask', action=actions.CheckFile,
                        help='Cube mask')

    return parser

def line_parents(*parents: str) -> argparse.ArgumentParser:
    """Return an `arparse` parent with the resquested parents.

    Available parents:

    - `vlsr`: add `vlsr` attribute.
    - `molecule`: add `line_lists`, `molecule`, `save_molecule`, `qns` and
        `onlyj` attributes.
    - `flux`: add `rms`, `nsigma` and `flux_limit` attributes.
    - `spectral_range`: add `freq_range`, `vel_range`, `chan_range`,
      `win_halfwidth` attributes.
    - `spatial_range`: add `blc_trc` and `xy_ranges` attributes.

    Args:
      parents: list of parents to return.
    """
    options = {'vlsr': _attach_vlsr, 'molecule': _molecule_args,
               'flux': _flux_parameters, 'spectral_range': _spectral_range,
               'spatial_range': _spatial_range}

    parser = argparse.ArgumentParser(add_help=False)
    for val in parents:
        options[val](parser)

    return parser

def _attach_vlsr(parser: argparse.ArgumentParser) -> None:
    """Add vlsr to the parser."""
    parser.add_argument('--vlsr', metavar=('VEL', 'UNIT'),
                        default=None,#0 * u.km/u.s,
                        action=actions.ReadQuantity,
                        help='LSR velocity')

def _molecule_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments to configure molecules."""
    parser.add_argument('--line_lists', nargs='*', default=('CDMS', 'JPL'),
                        help='Filter line lists for online queries')
    parser.add_argument('--molecule', nargs=1, default=None,
                        help='Molecule name or formula')
    parser.add_argument('--qns', nargs=1, default=None,
                        help='Molecule qns')
    parser.add_argument('--onlyj', action='store_true',
                        help='Filter out F, K transitions')
    parser.add_argument('--save_molecule', nargs=1, default=[None],
                        action=actions.NormalizePath,
                        help='Save molecule to disk')
    parser.add_argument('--restore_molecule', nargs=1, default=[None],
                        action=actions.NormalizePath,
                        help='Restore molecule from disk')

def _flux_parameters(parser: argparse.ArgumentParser) -> None:
    """Add arguments to filter intensity."""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--rms', metavar=('VAL', 'UNIT'), default=None,
                       action=actions.ReadQuantity,
                       help='Data rms value')
    group.add_argument('--sampled_rms', action='store_true',
                       help='Calculate the rms from a sample of channels')
    group.add_argument('--flux_limit', metavar=('VAL', 'UNIT'), default=None,
                       action=actions.ReadQuantity,
                       help='Flux lower limit')
    parser.add_argument('--nsigma', nargs=1, type=float, default=[5],
                        help='Number of rms levels for flux limit')

def _spectral_range(parser: argparse.ArgumentParser) -> None:
    """Add arguments for spectral range selection."""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--freq_range', metavar=('VAL0', 'VAL1', 'UNIT'),
                       nargs=3, action=actions.ReadQuantity,
                       help='Frequency range with unit')
    group.add_argument('--vel_range', metavar=('VAL0', 'VAL1', 'UNIT'),
                       nargs=3, action=actions.ReadQuantity,
                       help='Velocity range with unit')
    group.add_argument('--chan_range', metavar=('CHAN0', 'CHAN1'), nargs=2,
                       type=int,
                       help='Channel range')
    group.add_argument('--win_halfwidth', nargs=1, type=int, default=[None],
                       help=('Channel window half width '
                             '(vlsr and line freq needed)'))

def _spatial_range(parser: argparse.ArgumentParser) -> None:
    """Add arguments for spatial range selection."""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--blc_trc', metavar=('BLCX', 'BLCY', 'TRCX', 'TRCY'),
                       nargs=4, type=int,
                       help='Position of BLC and TRC')
    group.add_argument('--xy_ranges', metavar=('XLOW', 'XUP', 'YLOW', 'YUP'),
                       nargs=4, type=int,
                       help='Position x and y ranges')
    group.add_argument('--shrink', action='store_true',
                       help='Shrink to minimal cube.')

def query_freqrange(required: bool = False) -> argparse.ArgumentParser:
    """Parent argparse to determine frequency range from cmd input.

    A default `freq_range` attribute is added to the parser with a `None`
    value.

    Args:
      required: optional; is this cmd input required?

    Returns:
      An argument parser object.
      A function to fill the value of the parser `freq_range` attribute.
    """
    parser = argparse.ArgumentParser(add_help=False)
    group1 = parser.add_mutually_exclusive_group(required=required)
    group1.add_argument('--freqrange', metavar=('LOW',  'UP', 'UNIT'), nargs=3,
                        default=None, action=actions.ReadQuantity,
                        help='Frequency range.')
    group1.add_argument('--alma', nargs=1, type=int, default=None,
                        help='ALMA band number.')
    parser.set_defaults(freq_range=None)

    return parser, get_freqrange
