#!/bin/python3
"""Script to display line information upon request.

For usage and command line options run:
```bash
python line_info.py --help
```
"""
from pathlib import Path
import argparse
import sys

from toolkit.argparse_tools import actions, parents
import spectral_cube

from lines import Molecule
from spectrum import Spectrum
from argparse_plugins import query_freqrange

def _preproc(args: argparse.ArgumentParser) -> None:
    """Pre-process the command arguments."""
    # Define molecule for query
    args.query_name = f' {args.molecule[0]} '

def _proc(args: argparse.ArgumentParser) -> None:
    """Process query and show results."""
    # Query
    args.log.info(f'Quering molecule: {args.query_name}')
    args.molec = Molecule.from_query(args.query_name, args.freq_range,
                                     vlsr=args.vlsr)

    # Print result
    args.log.info(f'Molecule information: {args.molec}')

def _post(args: argparse.ArgumentParser) -> None:
    """Additional tasks."""
    if args.cubes and (args.position or args.reference or args.coordinate):
        # Load cubes
        for cube_file in args.cubes:
            # Cube
            args.log.info(f'Loading cube: {cube_file}')
            cube = spectral_cube.SpectralCube.read(cube_file)
            wcs = cube.wcs.sub(['longitude', 'latitude'])

            # Positions
            args.position_fn(args, wcs=wcs)

            # Extract spectra
            for position in args.pos:
                args.log.info(f'Loading spectra at: {position}')
                spec = Spectrum.from_cube(cube, position, vlsr=args.vlsr)

                # Filter molecules
                molec = spec.filter(args.molec)

                # Frequency range
                if args.freqrange is not None:
                    xlim = args.freqrange
                    suffix = f'{xlim[0].value:.2f}_{xlim[1].value:.2f}_'
                else:
                    xlim = None
                    suffix = ''

                # Plot
                suffix = (f'_spectrum_{molec.name}_'
                          f'{suffix}'
                          f'{position[0]}_{position[1]}.png')
                output = cube_file.stem + suffix
                output = args.outdir[0] / output
                spec.plot(output, molecule=molec, xlim=xlim)

def main(args: list):
    """Search Splatalogue for line information.

    Args:
      args: command line arguments.
    """
    freq_range_parent, freq_range_fn = query_freqrange(required=True)
    pipe = [freq_range_fn, _preproc, _proc, _post]
    args_parents = [
        freq_range_parent,
        parents.logger('debug_line_info.log'),
        parents.source_position(),
        parents.verify_files('cubes',
                             cubes={'help': 'Cube file names', 'nargs': '*'}),
        parents.paths('outdir',
                      outdir={'help': 'Plot output directory',
                              'nargs': 1,
                              'default': [Path('./')]}),
    ]
    parser = argparse.ArgumentParser(
        add_help=True,
        description='Display line information upon request.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents)
    parser.add_argument('--vlsr', metavar=('VEL', 'UNIT'), default=None,
                        action=actions.ReadQuantity, 
                        help='Velocity shift for observed frequencies')
    parser.add_argument('molecule', nargs=1,
                        help='Molecule name or formula')
    parser.set_defaults(pipe=pipe, query_name=None, freq_range=None, molec=None)
    args = parser.parse_args(args)

    # Run
    for step in args.pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
