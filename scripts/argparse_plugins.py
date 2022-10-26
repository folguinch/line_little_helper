"""Parents and actions for argparse objects."""
import argparse

from toolkit.argparse_tools import actions

from .global_vars import ALMA_BANDS

# Args processing functions
def get_freqrange(args):
    """Process the arguments for freq. range.

    Args:
      args: argument parser.
    """
    if args.freqrange is not None:
        args.freq_range = (args.freqrange[0], args.freqrange[1])
    elif args.alma is not None:
        args.freq_range =  ALMA_BANDS[args.alma[0]]
    else:
        msg = 'No input frequency range.'
        try:
            args.log.info(msg)
        except AttributeError:
            print(msg)

# Parents
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

# Actions
