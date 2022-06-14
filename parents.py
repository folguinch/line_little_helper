"""Commonly use `arparse` parents."""
from typing import Sequence
import argparse

#import astropy.units as u
from toolkit.argparse_tools import actions

def line_parents(parents: Sequence) -> argparse.ArgumentParser:
    """Return an `arparse` parent with the resquested parents.

    Available parents:

    - vlsr

    Args:
      parents: list of parents to return.
    """
    options = {'vlsr': _attach_vlsr, 'molecule': _molecule_args,
               'flux': _flux_parameters}

    parser = argparse.ArgumentParser(add_help=False)
    for val in parents:
        options[val](parser)

    return parser

def _attach_vlsr(parser: argparse.ArgumentParser) -> None:
    """Add vlsr to the parser."""
    parser.add_argument('--vlsr', metavar=('VEL', 'UNIT'),
                        default=None,#0 * u.km/u.s,
                        action=actions.ReadQuantity,
                        help='LSR velocity.')

def _molecule_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments to configure molecules."""
    parser.add_argument('--line_lists', nargs='*', default=('CDMS', 'JPL'),
                        help='Filter line lists for online queries.')
    parser.add_argument('--molecule', nargs=1, default=None,
                        help='Molecule name or formula.')
    parser.add_argument('--qns', nargs=1, default=None,
                        help='Molecule qns.')
    parser.add_argument('--onlyj', action='store_true',
                        help='Filter out F, K transitions.')

def _flux_parameters(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--rms', metavar=('VAL', 'UNIT'),
                       default=None,
                       action=actions.ReadQuantity,
                       help='Data rms value')
    group.add_argument('--flux_limit', metavar=('VAL', 'UNIT'),
                       default=None,
                       action=actions.ReadQuantity,
                       help='Flux lower limit')
