#!/bin/python3
"""Spectra extractor and analyzer.

The script loads and saves the spectra and plot them if requested.

If vlsr is given, then observed and rest frequencies are stored, else only
observed frequency is stored.

The `analyzer` looks for emission/absorption by performing sigma-clipping on
the spectra.
"""
from typing import Sequence, Tuple, Callable, Optional, Union
from pathlib import Path
import argparse
import sys

from astropy.io import fits
from astropy.table import QTable, vstack
from toolkit.argparse_tools import functions, actions
from toolkit.maths import quick_rms
import astropy.units as u
import toolkit.argparse_tools.parents as apparents

from line_little_helper.argparse_plugins import query_freqrange
from line_little_helper.parents import line_parents
from line_little_helper.spectrum import IndexedSpectra, on_the_fly_spectra_loader

def spectra_loader(filenames: Sequence[Path],
                   *,
                   positions: Optional[Sequence[Tuple[int]]] = None,
                   vlsr: Optional[u.Quantity] = None,
                   radius: Optional[u.Quantity] = None,
                   log: Callable = print) -> dict:
    """Load spectra from file names.

    Args:
      filenames: data file names.
      positions: optional; positions to extract the data from.
      vlsr: optional; LSR velocity.
      radius: optional; source radius for averaging.
      log: optional; logging function.
    """
    log('Extracting spectra')
    specs = IndexedSpectra.from_files(filenames, coords=positions, vlsr=vlsr,
                                      radius=radius)
    log(specs)

    return specs

def _build_table(results: dict, coords: Sequence, filename: Path,
                log: Callable = print):
    """Stores fitting results in Table."""
    table = []
    # Iterate over results
    for key, val in results.items():
        log(f'Storing results for file: {key}')
        # Iterate over results for each coordinate
        for coord, result in zip(coords, val):
            log(f'Storing results for coordinate: {coord}')

            # Iterate over model results
            for _, models in result:
                for slc, model in models:
                    table.append([coord,
                                  slc[0],
                                  slc[1],
                                  model.mean.quantity,
                                  model.amplitude.quantity,
                                  model.stddev.quantity])

    # Store table
    table = QTable(table,
                   names=('skycoord', 'start_freq', 'end_freq', 'mean',
                          'amplitude', 'std_dev'),
                   meta={'name': 'lines gaussian fit'})
    if filename.is_file():
        old_table = QTable.read(filename)
        table = vstack(old_table, table)
    table.sort(['skycoord'])
    log(f'Saving table to: {filename}')
    table.write(filename, overwrite=True)

    return table

def _extractor(args: argparse.ArgumentParser) -> None:
    """Extract spectra."""
    # Extract spectra
    functions.positions_to_pixels(args)
    args.specs = extractor(args.filenames,
                           position=args.position,
                           vlsr=args.vlsr,
                           rest=args.rest,
                           rms=args.rms,
                           radius=args.radius,
                           flux_limit=args.flux_limit,
                           mask_from=args.mask_from[0],
                           outdir=args.outdir[0],
                           savemask=args.savemask[0],
                           log=args.log.info)

def extractor(filenames: Sequence[Path],
              position: Optional = None,
              vlsr: Optional[u.Quantity] = None,
              rest: bool = False,
              radius: Optional[u.Quantity] = None,
              rms: Optional[u.Quantity] = None,
              flux_limit: Optional[u.Quantity] = None,
              mask_from: Optional[Path] = None,
              outdir: Optional[Path] = None,
              savemask: Optional[Path] = None,
              log: Callable = print) -> Union[IndexedSpectra, list]:
    """Extract spectra.

    If `position` is given, the spectra is extracted at that position in all
    the data (if they are data cubes). Else, spectra are loaded and saved at
    all position over a given `flux_limit` or 5`rms` levels.

    Args:
      filenames: filenames where to extract or load the spectra from.
      position: optional; coordinate where to get the spectra.
      vlsr: optional; source LSR velocity.
      rest: optional; store frequency in rest system.
      radius: optional; average all spectra within this radius.
      rms: optional; only save spectra with any value over 5sigma.
      flux_limit: optional; only save spectra with any value over limit.
      mask_from: optional; reference image to calculate a 5sigma mask.
      outdir: optional; where to save the extracted spectra.
      savemask: optional; save the mask.
      log: optional; logging function.

    Returns:
      An `IndexedSpectra` if position is given or empty `list` otherwise.
    """
    if position is not None:
        # Positions
        log(f'Position:\n{position}')

        # Specs
        specs = spectra_loader(filenames, positions=position, vlsr=vlsr,
                               radius=radius, log=log)
        # Save specs
        if outdir is not None:
            log(f'Saving to: {outdir}')
            specs.write_to(directory=outdir, fmt='dat')
    elif filenames[0].suffix.lower() == '.dat':
        specs = spectra_loader(filenames, log=log)
    else:
        # Check if there is a reference image for mask
        if mask_from is not None:
            log(f'Opening: {mask_from}')
            img = fits.open(mask_from)[0]
            rms = quick_rms(img.data)
            log(f'Reference image rms: {rms}')
            mask = img.data > 5 * rms
            log('Extracting all spectrum in mask (5sigma)')
            if savemask is not None:
                savemask = outdir / savemask
                log(f'Mask filename: {savemask}')
                log('Saving mask')
                header = img.header
                hdu = fits.PrimaryHDU(mask.astype(int), header=header)
                hdu.update_header()
                hdu.writeto(savemask, overwrite=True)
        else:
            mask = None
            log('Extracting all spectrum over flux limit')
        frame = 'rest' if rest else 'observed'
        on_the_fly_spectra_loader(filenames,
                                  rms=rms,
                                  flux_limit=flux_limit,
                                  vlsr=vlsr,
                                  mask=mask,
                                  savedir=outdir,
                                  maskname=savemask,
                                  restframe=frame,
                                  radius=radius,
                                  log=log)
        specs = []

    return specs

def _analyzer(args):
    """Analyze the spectra."""
    if not args.analyze:
        return

    # Search for lines
    args.log.info('Finding lines')
    line_results = args.specs.have_lines(dilate=5, slice_as_frequency=True,
                                         ax=args.ax)

    # Build a results table
    args.log.info('Processing results')
    filename = args.outdir[0] / 'line_gaussian_fit.ecsv'
    table = _build_table(line_results, args.position, filename,
                         log=args.log.info)

def _plotter(args):
    """Plot data and store figs/axes if analysis is requested."""
    if args.plot_separated:
        # Only for indexed spectra
        #figs, axs = None, None
        for key, specs in args.specs:
            for i, spec in enumerate(specs):
                fig, ax = spec.plot(ax=ax)

                if not args.analyze:
                    plotname = args.specs.generate_filename(
                        key,
                        index=i,
                        directory=args.outdir[0],
                    )
                    plotname = plotname.with_suffix('.png')
                    fig.savefig(plotname)

def main(args: list):
    """Search Splatalogue for line information.

    Args:
      args: command line arguments.
    """
    freq_range_parent, freq_range_fn = query_freqrange(required=False)
    pipe = [freq_range_fn, _extractor, _plotter]
    args_parents = [
        freq_range_parent,
        line_parents(['vlsr', 'flux']),
        apparents.logger('debug_spectrum_extractor.log'),
        apparents.source_position(required=False),
        apparents.verify_files(
            'filenames',
            '--mask_from',
            filenames={'help': 'File names (cubes or spectra)', 'nargs': '*'},
            mask_from={'help': 'Image file name to build the mask from',
                       'nargs': 1,
                       'default':[None]},
        ),
    ]
    parser = argparse.ArgumentParser(
        add_help=True,
        description='Spectra extractor and analyzer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents)
    group1 = parser.add_mutually_exclusive_group(required=False)
    group1.add_argument('--radius', metavar=('RADIUS', 'UNIT'),
                       action=actions.ReadQuantity, nargs=2,
                       help='Average pixels within radius')
    group1.add_argument('--size', metavar=('MAJ', 'MIN', 'UNIT'),
                       action=actions.ReadQuantity, nargs=3,
                       help='Get radius from size')
    group1.add_argument('--pix_area', metavar=('AREA'),
                       type=float, nargs=1,
                       help='Area of the source in pixels')
    group2 = parser.add_mutually_exclusive_group(required=False)
    group2.add_argument('--plot_separated', action='store_true',
                        help='Plot each spectra separated.')
    group2.add_argument('--plot_pos', nargs=1, action=actions.NormalizePath,
                        default=None,
                        help=('File name base to plot the spectra '
                              '(one plot per position).'))
    parser.add_argument('--rest', action='store_true',
                        help='Store in rest frame if vlsr is given')
    parser.add_argument('--savemask', nargs=1, type=Path, default=[None],
                        help='Save the mask to `outdir.`')
    parser.add_argument('--analyze', action='store_true',
                        help='Perform analysis of the spectrum.')
    parser.add_argument('--outdir', action=actions.MakePath, default=None,
                        nargs=1,
                        help='Output directory, else first file directory')
    parser.set_defaults(specs=None, ax=None, fig=None)
    args = parser.parse_args(args)

    # Run
    for step in pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
