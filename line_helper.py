#!/bin/python3
import argparse
import sys

from toolkit.argparse_tools import actions, parents
import astropy.units as u

from .processing_tools import query_from_array

FREQ_COLS = ['freq_low', 'freq_up']

def _preproc(args: argparse.Namespace) -> None:
    """Prepare args inputs for analysis."""
    # Separate array and units
    if args.table is not None:
        args.array = args.table[0]
        args.units = args.table[1]
    else:
        dtype = [('n', int)] + zip(FREQ_COLS, [np.float, np.float])
        args.array = np.array([(1,) + tuple(args.freqrange.value)], 
                              dtype=dtype)
        args.units = {key: args.freqrange.unit for key in FREQ_COLS}
        args.units.update({'n': None})

    # Shift observed frequencies.
    if args.vlsr is not None and args.restfreq is not None:
        args.log.info('Shifting observed frequencies')
        for col in FREQ_COLS:
            # Convert to velocity
            freq_to_vel = u.doppler_radio(args.restfreq)
            freq = args.array[col] * args.units[col]
            vels = freq.to(args.vlsr.unit, equivalencies=freq_to_vel)

            # Shift and convert back
            vels = vels - args.vlsr
            freq = vels.to(freq.unit, equivalencies=freq_to_vel)

            # Replace
            args.array[col] = freq.value

def _proc1(args: argparse.Namespace) -> None:
    """First data processing.

    Use the information in the table to query splat and determine the lines in
    the given ranges. Store this information in a new table inside args.
    """
    # Query SPLAT
    args.results = query_from_array(args.array, args.units, freq_cols=FREQ_COLS)

def _proc2(args):
    """Analyse the results against the observations.

    Given the lines in each range, give information to determine which lines
    are more probable to be present. This function requires that an input
    spectrum is present in the args.
    """
    pass

def _post(args):
    """Post process the results.

    Plot the analysis in proc2 if needed. Save the tables.
    """
    pass

def main(args):
    """Main function.

    Args:
      args: arguments for argparse.
    """
    # Parser
    pipe = [_preproc, _proc1, _proc2, _post]
    args_parents = [parents.logger('debug_line_helper.log')]
    parser = argparse.ArgumentParser(add_help=True, parents=args_parents)
    parser.add_argument('--vlsr', action=actions.ReadQuantity, default=None,
                        help='Velocity shift for observed frequencies')
    parser.add_argument('--restfreq', action=actions.ReadQuantity,
                        default=None,
                        help='Rest frequency for observed frequencies')
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--table', nargs=1, default=None,
                        action=actions.LoadMixedStructArray,
                        help='Input table file name')
    group1.add_argument('--freqrange', nargs=3, default=None,
                        action=actions.ReadQuantity,
                        help='Frequency range')
    parser.set_defaults(pipe=pipe, array=None, units=None, results=None)
    args = parser.parse_args(args)
    
    # Run
    for step in args.pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
