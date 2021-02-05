#!/bin/python3
import argparse
import pathlib
import sys

from toolkit.argparse_tools import actions, parents
from toolkit.astro_tools import cube_utils
import astropy.units as u
import numpy as np

from analysis_tools import simple_analysis, advanced_analysis
from data_handler import ResultsHandler
from processing_tools import observed_to_rest
from spectrum import Spectrum, Spectra

FREQ_COLS = ['freq_low', 'freq_up']

def _preproc(args: argparse.Namespace) -> None:
    """Prepare args inputs for analysis."""
    # Equivalencies
    if args.restfreq:
        if args.table is not None and 'spw' in args.table[0].dtype.names:
            spws = np.unique(args.table[0]['spw'])
            if len(args.restfreq) == 1:
                args.equivalencies = {spw: u.doppler_radio(args.restfreq[0])
                                      for spw in spws}
            else:
                if len(spws) != len(args.restfreq):
                    raise ValueError('Size of spws and restfreqs do not match')
                aux = zip(spws, args.restfreq)
                args.equivalencies = {spw: u.doppler_radio(restfreq)
                                      for spw, restfreq in aux}
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
        for cube in args.cubes:
            # Observed frequencies shifted during subtraction
            spec = cube_utils.spectrum_at_position(cube, args.coord,
                                                   spectral_axis_unit=u.GHz,
                                                   vlsr=args.vlsr)
            spec = Spectrum(spec[0], spec[1].quantity,
                            restfreq=cube_utils.get_restfreq(cube), 
                            rms=cube_utils.get_cube_rms(cube, use_header=True))
            args.spectra.append(spec)
    elif args.spec is not None:
        freq_names = ['nu', 'freq', 'frequency', 'v', 'vel', 'velocity']
        int_names = ['F', 'f', 'Fnu', 'fnu', 'intensity', 'T', 'Tb']
        args.log.info('Reading input spectra')
        for spw, (data, units) in enumerate(args.spec):
            # Spectral axis
            freq_name = list(filter(lambda x: x in units, freq_names))[0]
            xaxis = data[freq_name] * units[freq_name]

            # Intensity axis
            int_name = list(filter(lambda x: x in units, int_names))[0]
            spec = data[int_name] * units[int_name]

            # Noise
            if args.rms is not None:
                rms = args.rms[i]
            else:
                rms = None

            # Shift spectral axis
            if 'all' in args.equivalencies:
                equivalency = args.equivalencies
            else:
                equivalency = {'all': args.equivalencies[spw]}
            if xaxis.unit.is_equivalent(u.Hz) and args.vlsr is not None:
                xaxis = observed_to_rest(xaxis, args.vlsr, equivalency)
                spec = Spectrum(xaxis, spec, restfreq=args.restfreq, rms=rms)
            elif xaxis.unit.is_equivalent(u.km/u.s):
                if freq_to_vel is None:
                    args.log.warn('Cannot convert spectral axis to GHz')
                    continue
                vels = xaxis - args.vlsr
                xaxis = vels.to(u.GHz, equivalencies=equivalency['all'])
                spec = Spectrum(xaxis, spec, restfreq=args.restfreq, rms=rms)
            else:
                spec = Spectrum(xaxis, spec, restfreq=args.restfreq, rms=rms)
            args.spectra.append(spec)
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
