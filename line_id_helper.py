#!/bin/python3
"""Program to help in the identification of lines."""
import argparse
import pathlib
import sys

from toolkit.argparse_tools import actions, parents
import astropy.units as u
import numpy as np

from line_little_helper.analysis_tools import simple_analysis, advanced_analysis
from line_little_helper.data_handler import ResultsHandler
from line_little_helper.processing_tools import observed_to_rest, get_spectral_equivalencies
from line_little_helper.spectrum import Spectra

FREQ_COLS = ['freq_low', 'freq_up']

def _preproc(args: argparse.Namespace) -> None:
    """Prepare args inputs for analysis."""
    # Equivalencies
    if args.restfreq:
        if args.table is not None and 'spw' in args.table[0].dtype.names:
            spws = np.unique(args.table[0]['spw'])
            args.equivalencies = get_spectral_equivalencies(args.restfreq,
                                                            keys=spws)
        else:
            args.equivalencies['all'] = u.doppler_radio(args.restfreq[0])

    # Define filename
    if args.filename is not None:
        args.filename = args.filename[0]
    else:
        args.filename = pathlib.Path('./results.ecsv').resolve()

def _proc1(args: argparse.Namespace) -> None:
    """First data processing.

    Use the information in the table to query splat and determine the lines in
    the given ranges. Store this information in a new table inside args.
    """
    # Separate array and units
    if args.table is not None:
        array = args.table[0]
        units = args.table[1]
    else:
        dtype = [('n', int)] + list(zip(FREQ_COLS, [np.float, np.float]))
        array = np.array([(1,) + tuple(args.freqrange.value)],
                              dtype=dtype)
        units = {key: args.freqrange.unit for key in FREQ_COLS}
        units.update({'n': None})

    # Shift observed frequencies.
    if args.vlsr is not None and args.restfreq is not None:
        args.log.info('Shifting observed frequencies')
        for col in FREQ_COLS:
            if 'spw' in array.dtype.names:
                spws_map = array['spw']
            else:
                spws_map = None
            freqs = observed_to_rest(array[col] * units[col], args.vlsr,
                                     args.equivalencies, spws_map=spws_map)
            array[col] = freqs.value

    # Query SPLAT
    args.results = ResultsHandler.from_struct_array(array, units,
                                                    freq_cols=FREQ_COLS)

def _proc2(args):
    """Analyse the results against the observations.

    Given the lines in each range, give information to determine which lines
    are more probable to be present. This function requires that an input
    spectrum is present in the args.
    """
    # Observed data
    if args.cubes is not None:
        if args.coord is None:
            args.log.warn('Coordinate needed for spectrum, skipping')
            return
        # Extract spectra
        args.log.info('Extracting spectra from cubes')
        args.spectra = Spectra.from_cubes(args.cubes, args.coord[0],
                                          vlsr=args.vlsr)
    elif args.spec is not None:
        args.log.info('Reading input spectra')
        args.spectra = Spectra.from_arrays(args.spec, args.restfreq,
                                           args.equivalencies, vlsr=args.vlsr,
                                           rms=args.rms)
    else:
        pass

    # If there are not any spectra then pass
    if len(args.spectra) == 0:
        args.log.info('Skipping spectrum analysis')
        return

    # Analysis
    simple_analysis(args.results, args.spectra)
    advanced_analysis(args.results, args.filename.with_suffix('.overall.dat'),
                      top=args.top[0])

    # Plot results
    args.results.plot(args.filename.with_suffix('.png'), spectra=args.spectra,
                      top=args.top[0])

def _post(args):
    """Post process the results.

    Plot the analysis in proc2 if needed. Save the tables.
    """
    # Save results
    args.log.info('Saving result tables')
    args.results.write(args.filename)

def main(args):
    """Main function.

    Args:
      args: arguments for argparse.
    """
    # Parser
    pipe = [_preproc, _proc1, _proc2, _post]
    args_parents = [parents.logger('debug_line_helper.log')]
    parser = argparse.ArgumentParser(add_help=True, parents=args_parents)
    parser.add_argument('--filename', action=actions.NormalizePath, nargs=1,
                        default=None,
                        help='Output table file name or filename base')
    parser.add_argument('--top', nargs=1, type=int, default=[None],
                        help='Consider only the best top results.')
    parser.add_argument('--vlsr', action=actions.ReadQuantity, default=None,
                        help='Velocity shift for observed frequencies')
    parser.add_argument('--restfreq', nargs='*', action=actions.ReadQuantity,
                        default=None, enforce_list=True,
                        help='Rest frequency for observed frequencies')
    parser.add_argument('--rms', nargs='*', action=actions.ReadQuantity,
                        default=None, enforce_list=True,
                        help='Noise level for input spectra.')
    parser.add_argument('--coord', action=actions.ReadSkyCoords, default=None,
                        help='Sky position required to get spectrum from cube')
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--table', default=None,
                        action=actions.LoadMixedStructArray,
                        help='Input table file name')
    group1.add_argument('--freqrange', nargs=3, default=None,
                        action=actions.ReadQuantity,
                        help='Frequency range')
    group2 = parser.add_mutually_exclusive_group(required=False)
    group2.add_argument('--cubes', action=actions.LoadCube, default=None,
                        nargs='*',
                        help='Cube file name to extract the spectrum')
    group2.add_argument('--spec', action=actions.LoadStructArray,
                        default=None, nargs='*',
                        help='Spectrum/spectra file name(s)')
    parser.set_defaults(pipe=pipe, equivalencies={}, results=None,
                        spectra=Spectra())
    args = parser.parse_args(args)

    # Run
    for step in args.pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
