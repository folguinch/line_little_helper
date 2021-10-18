"""Extract spectra from cubes.

The script saves the spectra and plot them if requested.

If vlsr is given observed and rest frequencies are stored, else only observed
frequency is stored.
"""
from typing import Sequence, Tuple, Callable, Optional
from pathlib import Path
import argparse
import sys

from astropy.io import fits
from toolkit.argparse_tools import functions, actions
from toolkit.maths import quick_rms
import astropy.units as u
import toolkit.argparse_tools.parents as apparents

from argparse_plugins import query_freqrange
from parents import line_parents
from spectrum import IndexedSpectra, on_the_fly_spectra_loader

def spectra_loader(cubes: Sequence[Path],
                   positions: Sequence[Tuple[int]],
                   *,
                   vlsr: Optional[u.Quantity] = None,
                   log: Callable = print) -> dict:
    """Load spectra from cube.

    Args:
      cubes: data cubes.
      positions: positions to extract the data from.
      vlsr: LSR velocity.
      log: logging function.
    """
    log('Extracting spectra')
    specs = IndexedSpectra.from_files(cubes, positions, vlsr=vlsr)
    log(specs)

    return specs

def _proc(args: argparse.ArgumentParser) -> None:
    """Main pipe."""
    # Extract spectra
    if args.position is not None:
        # Positions
        functions.positions_to_pixels(args)
        args.log.info(f'Positions:\n{args.position}')

        specs = spectra_loader(args.cubes, args.position, vlsr=args.vlsr,
                               log=args.log.info)
        # Save specs
        if args.outdir is not None:
            args.log.info(f'Saving to: {args.outdir[0]}')
            specs.write_to(args.outdir[0])
    else:
        # Check if there is a reference image for mask
        if args.mask_from is not None:
            args.log.info(f'Opening: {args.mask_from[0]}')
            img = fits.open(args.mask_from[0])[0]
            rms = quick_rms(img.data)
            args.log.info(f'Reference image rms: {rms}')
            mask = img.data > 5 * rms
            args.log.info('Extracting all spectrum in mask (5sigma)')
            if args.savemask[0] is not None:
                savemask = args.outdir[0] / args.savemask[0]
                args.log.info(f'Mask filename: {savemask}')
                args.log.info('Saving mask')
                header = img.header
                hdu = fits.PrimaryHDU(mask.astype(int), header=header)
                hdu.update_header()
                hdu.writeto(savemask, overwrite=True)
            exit()
        else:
            mask = None
            args.log.info('Extracting all spectrum over flux limit')
        frame = 'rest' if args.rest else 'observed'
        on_the_fly_spectra_loader(args.cubes,
                                  rms=args.rms,
                                  flux_limit=args.flux_limit,
                                  vlsr=args.vlsr,
                                  mask=mask,
                                  savedir=args.outdir[0],
                                  maskname=args.savemask[0],
                                  restframe=frame,
                                  log=args.log.info)

def main(args: list):
    """Search Splatalogue for line information.

    Args:
      args: command line arguments.
    """
    freq_range_parent, freq_range_fn = query_freqrange(required=False)
    pipe = [freq_range_fn, _proc]
    args_parents = [
        freq_range_parent,
        line_parents(['vlsr', 'flux']),
        apparents.logger('debug_spectrum_extractor.log'),
        apparents.source_position(required=False),
        apparents.verify_files(
            'cubes',
            'mask_from',
            cubes={'help': 'Cube file names', 'nargs': '*'},
            mask_from={'help': 'Image file name to build the mask from',
                       'nargs': 1},
        ),
    ]
    parser = argparse.ArgumentParser(
        add_help=True,
        description='Extract spectra from cube(s)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents)
    parser.add_argument('--rest', action='store_true',
                        help='Store in rest frame if vlsr is given')
    parser.add_argument('--savemask', nargs=1, type=Path, default=None,
                        help='Save the mask to `outdir.`')
    parser.add_argument('--outdir', action=actions.MakePath, default=None,
                        nargs=1,
                        help='Output directory, else first cube directory')
    parser.set_defaults(pipe=pipe, query_name=None, freq_range=None, molec=None)
    args = parser.parse_args(args)

    # Run
    for step in args.pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
