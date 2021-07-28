"""Extract spectra from cubes.

The script saves the spectra and plot them if requested.

If vlsr is given observed and rest frequencies are stored, else only observed
frequency is stored.
"""

from toolkit.argparse_tools import functions

def spectrum_loader(cubes: Sequence[Path],
                    positions: Sequence[Tuple[int]], 
                    *,
                    log: Callable = print) -> dict:
    specs = {}
    # Load cubes
    for cube_file in cubes:
        # Cube
        log(f'Loading cube: {cube_file}')
        cube = spectral_cube.SpectralCube.read(cube_file)
        wcs = cube.wcs.sub(['longitude', 'latitude'])
        specs[cube_file] = {}


        # Extract spectra
        for position in args.pos:
            args.log.info(f'Loading spectra at: {position}')
        specs[cube_file] = {}
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


def _proc(args: argparse.ArgumentParser) -> None:
    """Main pipe."""
    # Positions
    functions.position_to_pixels(args)

def main(args: list):
    """Search Splatalogue for line information.

    Args:
      args: command line arguments.
    """
    freq_range_parent, freq_range_fn = query_freqrange(required=False)
    pipe = [freq_range_fn, _preproc, _proc]
    args_parents = [freq_range_parent,
                    parents.logger('debug_spectrum_extractor.log'),
                    parents.source_position(),
                    parents.verify_files(
                        'cubes',
                        cubes={'help': 'Cube file names', 'nargs': '*'}),
                    ]
    parser = argparse.ArgumentParser(
        add_help=True,
        description='Extract spectra from cube(s)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents)
    parser.add_argument('--vlsr', action=actions.ReadQuantity, default=None,
                        help='Velocity shift for observed frequencies')
    parser.set_defaults(pipe=pipe, query_name=None, freq_range=None, molec=None)
    args = parser.parse_args(args)

    # Run
    for step in args.pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
