"""Extract spectra from cubes.

The script saves the spectra and plot them if requested.

If vlsr is given observed and rest frequencies are stored, else only observed
frequency is stored.
"""
from typing import Sequence, Tuple, Callable, Optional
from pathlib import Path
import argparse
import sys

from toolkit.argparse_tools import functions, parents, actions
import astropy.units as u
#import spectral_cube

from argparse_plugins import query_freqrange
from spectrum import IndexedSpectra

def spectra_loader(cubes: Sequence[Path],
                   positions: Sequence[Tuple[int]],
                   *,
                   vlsr: Optional[u.Quantity] = None,
                   log: Callable = print) -> dict:
    log('Extracting spectra')
    specs = IndexedSpectra.from_files(cubes, positions, vlsr=vlsr)
    log(specs)
    ## Load cubes
    #for cube_file in cubes:
    #    # Cube
    #    log(f'Loading cube: {cube_file}')
    #    cube = spectral_cube.SpectralCube.read(cube_file)
    #    wcs = cube.wcs.sub(['longitude', 'latitude'])
    #    specs[cube_file] = {}


    #    # Extract spectra
    #    specs[cube_file] = {}
    #    for position in args.pos:
    #        args.log.info(f'Loading spectra at: {position}')
    #        spec = Spectrum.from_cube(cube, position, vlsr=args.vlsr)

    #        # Filter molecules
    #        molec = spec.filter(args.molec)

    #        # Frequency range
    #        if args.freqrange is not None:
    #            xlim = args.freqrange
    #            suffix = f'{xlim[0].value:.2f}_{xlim[1].value:.2f}_'
    #        else:
    #            xlim = None
    #            suffix = ''

    #        # Plot
    #        suffix = (f'_spectrum_{molec.name}_'
    #                    f'{suffix}'
    #                    f'{position[0]}_{position[1]}.png')
    #        output = cube_file.stem + suffix
    #        output = args.outdir[0] / output
    #        spec.plot(output, molecule=molec, xlim=xlim)
    return specs

def _proc(args: argparse.ArgumentParser) -> None:
    """Main pipe."""
    # Positions
    functions.positions_to_pixels(args)
    args.log.info(f'Positions:\n{args.position}')

    # Extract spectra
    specs = spectra_loader(args.cubes, args.position, vlsr=args.vlsr,
                           log=args.log.info)

    # Save specs
    if args.outdir is not None:
        args.log.info(f'Saving to: {args.outdir[0]}')
        specs.write_to(args.outdir[0])

def main(args: list):
    """Search Splatalogue for line information.

    Args:
      args: command line arguments.
    """
    freq_range_parent, freq_range_fn = query_freqrange(required=False)
    pipe = [freq_range_fn, _proc]
    args_parents = [freq_range_parent,
                    parents.logger('debug_spectrum_extractor.log'),
                    parents.source_position(required=True),
                    parents.verify_files(
                        'cubes',
                        cubes={'help': 'Cube file names',
                               'nargs': '*'}),
                    ]
    parser = argparse.ArgumentParser(
        add_help=True,
        description='Extract spectra from cube(s)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents)
    parser.add_argument('--vlsr', metavar=('VALUE', 'UNIT'),
                        action=actions.ReadQuantity, default=None,
                        help='Velocity shift for observed frequencies')
    parser.add_argument('--outdir', action=actions.MakePath, default=None,
                        nargs=1,
                        help='Output directory')
    parser.set_defaults(pipe=pipe, query_name=None, freq_range=None, molec=None)
    args = parser.parse_args(args)

    # Run
    for step in args.pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
