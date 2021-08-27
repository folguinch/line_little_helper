"""Commonly use `arparse` parents."""
from typing import Sequence
import argparse

#import astropy.units as u
import toolkit.argparse_tools.actions as actions

def line_parents(parents: Sequence) -> argparse.ArgumentParser:
    """Return an `arparse` parent with the resquested parents.

    Available parents:

    - vlsr

    Args:
      parents: list of parents to return.
    """
    options = {'vlsr': _attach_vlsr, 'molecule': _molecule_args}

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
