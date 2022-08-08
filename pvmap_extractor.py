#!/bin/python3
"""Calculate a position-velocity (pv) map."""
from typing import Callable, Optional, Sequence, Iterable, List
from pathlib import Path
import argparse
import sys

from astro_source.source import LoadSources
from astropy.io import fits
from astropy.units.equivalencies import doppler_radio
from pvextractor import PathFromCenter, extract_pv_slice
from pvextractor import Path as pvPath
from regions import Regions
from toolkit.argparse_tools import actions, parents, functions
from toolkit.astro_tools import cube_utils
import astropy.units as u
import numpy as np

def swap_axes(hdu: 'PrimaryHDU'):
    """Swap the axis of an HDU."""
    # swap data
    aux = np.swapaxes(hdu.data, -1, -2)

    # Fill header
    header = fits.Header(cards=hdu.header.keys())
    for key, val in hdu.header.items():
        if '1' in key:
            header[key.replace('1', '2')] = val
        elif '2' in key:
            header[key.replace('2', '1')] = val
        else:
            header[key] = val

    return fits.PrimaryHDU(aux, header=header)

def get_pvmap_from_slit(cube: 'SpectralCube',
                        position: 'SkyCoord',
                        length: u.Quantity,
                        width: u.Quantity,
                        angle: u.Quantity,
                        invert: bool = False,
                        filename: Optional[Path] = None,
                        log: Callable = print) -> 'PrimaryHDU':
    """Calculate a position velocity map from a slit.

    Args:
      cube: spectral cube.
      position: central position of the slit.
      length: length of the slit.
      width: width of the slit.
      angle: position angle of the slit.
      invert: optional; invert the velocity/position axes
      filename: optional; output file name.
      log: optional; logging function.
    Returns:
      A `PrimaryHDU` containing the pv map.
    """
    # Define slit path
    path = PathFromCenter(center=position, length=length, angle=angle,
                          width=width)

    # PV map
    pv_map = extract_pv_slice(cube, path)

    # Invert velocity/position axes
    if invert:
        pv_map = swap_axes(pv_map)

    # Copy BUNIT
    try:
        pv_map.header['BUNIT'] = cube.header['BUNIT']
    except KeyError:
        log('Could not find BUNIT')

    if filename is not None:
        log(f'Saving file: {filename}')
        pv_map.writeto(filename, overwrite=False)

    return pv_map

def get_pvmap_from_region(cube: 'SpectralCube',
                          region: Path,
                          width: u.Quantity,
                          invert: bool = False,
                          filename: Optional[Path] = None,
                          log: Callable = print) -> 'PrimaryHDU':
    """Calculate a position velocity map from a CASA poly region.

    Args:
      cube: spectral cube.
      region: the `crtf` region filename.
      width: width of the slit.
      invert: optional; invert the velocity/position axes
      filename: optional; output file name.
      log: optional; logging function.
    Returns:
      A `PrimaryHDU` containing the pv map.
    """
    # Read region
    reg = Regions.read(region, format='crtf').pop()

    # Path
    pv_path = pvPath(reg.vertices, width=width)

    # PV map
    pv_map = extract_pv_slice(cube, pv_path)

    # Invert velocity/position axes
    if invert:
        pv_map = swap_axes(pv_map)

    # Copy BUNIT
    try:
        pv_map.header['BUNIT'] = cube.header['BUNIT']
    except KeyError:
        log('Could not find BUNIT')

    if filename is not None:
        log(f'Saving file: {filename}')
        pv_map.writeto(filename, overwrite=False)

    return pv_map

def get_parent_parser() -> argparse.ArgumentParser:
    """Base parent parser for pv maps."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--invert',
        action='store_true',
        help='Invert axes')
    parent_parser.add_argument(
        '--pvconfig',
        action=actions.LoadConfig,
        help='Configuration of the PV maps (position and slit)')
    parent_parser.add_argument(
        '--config_section',
        dest='sections',
        nargs='*',
        help='Use only this section(s) from the config file')
    parent_parser.add_argument(
        '--line_freq',
        action=actions.ReadQuantity,
        help='Line (or slab center) frequency')
    parent_parser.add_argument(
        '--dfreq',
        action=actions.ReadQuantity,
        help='Frequency slab width')
    parent_parser.add_argument(
        '--vlsr',
        action=actions.ReadQuantity,
        help='LSR velocity in km/s')
    parent_parser.add_argument(
        '--pa',
        nargs='*',
        action=actions.ReadQuantity,
        help='Position angles in degrees')
    parent_parser.add_argument(
        '--length',
        action=actions.ReadQuantity,
        help='Length of the slit')
    parent_parser.add_argument(
        '--width',
        action=actions.ReadQuantity,
        help='Width of the slit in arcsec')
    parent_parser.add_argument(
        '--paths',
        nargs='*',
        action=actions.NormalizePath,
        help='Read a casa poly region as pv path')
    parent_parser.add_argument(
        '--output',
        action=actions.NormalizePath,
        help='File output')

    return parent_parser

def get_spectral_slab(cube: 'SpectralCube',
                      line_freq: u.Quantity,
                      delta_freq: u.Quantity,
                      rest_freq: Optional[u.Quantity] = None,
                      vlsr: u.Quantity = 0 * u.km/u.s) -> 'SpectralCube':
    """Make a subcube around the line frequency.

    Args:
      cube: input spectral cube.
      line_freq: central frequency of the slab.
      delta_freq: half the size of the slab.
      rest_freq: optional; rest frequency for velocities.
      vlsr: optional; LSR velocity.
    Returns:
      A new `SpectralCube`.
    """
    if rest_freq is None:
        rest_freq = cube_utils.get_restfreq(cube)

    dfreq = np.array([line_freq.to(u.GHz).value - delta_freq.to(u.GHz).value,
                      line_freq.to(u.GHz).value + delta_freq.to(u.GHz).value])
    dfreq = dfreq * u.GHz
    dvel = dfreq.to(u.km/u.s, equivalencies=doppler_radio(rest_freq))
    dvel = dvel + vlsr
    dfreq = dvel.to(u.GHz, equivalencies=doppler_radio(rest_freq))
    try:
        return cube.spectral_slab(*dfreq)
    except u.core.UnitsError:
        return cube.spectral_slab(*dvel)

def _minimal_set(args: argparse.Namespace) -> None:
    """Determine wether input cmd parameters define the slice."""
    if (args.cube and args.pa and args.length and
        args.width and args.pvconfig is None):
        functions.pixels_to_positions(args,
                                      wcs=args.cube.wcs.sub(['longitude',
                                                             'latitude']))
        args.sections = [args.cube]
    elif args.cube and args.width and args.path and args.pvconfig is None:
        args.sections = [args.cube]
    else:
        args.sections = args.sections or args.pvconfig.sections()
        args.log.info('PV config file selected sections: %r', args.sections)

def _iter_sections(args: argparse.Namespace) -> None:
    """Iterate over the sections in configuration."""
    # Compute pv maps
    for section in args.sections:
        args.log.info('PV map for: %s', section)

        # Load moment 0 map or use source position
        source_section = None
        if args.pvconfig and 'moment0' in args.pvconfig[section]:
            raise NotImplementedError
            #moment0 = Image(args.pvconfig.get(section, 'moment0'))
            #xmax, ymax = moment0.max_pix()
            #rest_freq = moment0.header['RESTFRQ'] * u.Hz
        elif args.pvconfig and args.source is not None:
            source_section = args.pvconfig.get(section,
                                               'source_section',
                                               fallback=section)
            # This assumes the cubes are the same for all sources
            args.cube = args.source[0][source_section]
            functions.pixels_to_positions(args,
                                          wcs=args.cube.wcs.sub(['longitude',
                                                                 'latitude']))
        else:
            raise NotImplementedError('Only pvconfig for now')
        rest_freq = cube_utils.get_restfreq(args.cube)

        # Line frequency
        args.log.info('Rest frequency: %s', rest_freq.to(u.GHz))
        if args.line_freq:
            args.log.info('Line frequency: %s', args.line_freq.to(u.GHz))
        elif args.pvconfig and 'line_freq' in args.pvconfig[section]:
            args.line_freq = args.pvconfig.getquantity(section, 'line_freq')
            args.log.info('Line frequency: %s', args.line_freq.to(u.GHz))
        else:
            args.log.warning('No line frequency given, using whole range')
            args.line_freq = None

        # vlsr
        if args.vlsr is not None:
            args.log.info('Input LSR velocity: %s', args.vlsr)
        elif args.pvconfig and 'v_lsr' in args.pvconfig[section]:
            args.vlsr = args.pvconfig.getquantity(section, 'v_lsr')
            args.log.info('LSR velocity from config: %s', args.vlsr)
        else:
            vlsr = 0. * u.km / u.s
            args.log.warning('LSR velocity: %s', vlsr)

        # Get frequency/velocity slab
        subcube = args.cube.with_spectral_unit(u.GHz,
                                               velocity_convention='radio')
        cdelt = np.abs(subcube.spectral_axis[0] - subcube.spectral_axis[1])
        args.log.info('Cube spectral axis step: %s', cdelt.to(u.MHz))
        if args.line_freq is not None:
            if args.dfreq:
                dfreq2 = args.dfreq / 2.
                args.log.info('Frequency slab width: %s', args.dfreq.to(u.MHz))
            elif args.pvconfig and 'freq_slab' in args.pvconfig[section]:
                dfreq2 = args.pvconfig.getquantity(section, 'freq_slab') / 2.
                args.log.info('Frequency slab width: %s', dfreq2.to(u.MHz)*2.)
            elif args.pvconfig and 'chan_slab' in args.pvconfig[section]:
                chan0, chan1 = args.pvconfig.getintlist(section, 'chan_slab')
                args.log.info('Channel slab: %i, %i', chan0, chan1)
                dfreq2 = None
                subcube = subcube.data[chan0:chan1+1]
                args.log.info('New cube shape: %r', subcube.shape)
            else:
                args.log.warning('No frequency slab width, using whole range')
                dfreq2 = None

            # Get the spectral slab
            if dfreq2 is not None:
                subcube = get_spectral_slab(subcube, args.line_freq, dfreq2,
                                            rest_freq=rest_freq,
                                            vlsr=args.vlsr)
                cdelt = np.abs(subcube.spectral_axis[0] - \
                               subcube.spectral_axis[1])
                args.log.info('New cube shape: %r', subcube.shape)
                args.log.info('New cube spectral axis step: %s',
                              cdelt.to(u.MHz))
        else:
            pass

        # Change units to velocity
        # The zero velocity is in the line center
        subcube = subcube.with_spectral_unit(u.km/u.s,
                                             velocity_convention='radio',
                                             rest_value=args.line_freq)

        # PV map options
        pvmap_kwargs = {'output': args.output,
                        'file_fmt': args.pvconfig.get(section, 'file_fmt',
                                                      fallback=None),
                        'sources': args.source,
                        'source_section': source_section,
                        'section': section,
                        }
        if args.paths or (args.pvconfig and 'paths' in args.pvconfig[section]):
            pvmap_kwargs['paths'] = args.path
            if args.path is None:
                aux = args.pvconfig.get(section, 'paths').split(',')
                pvmap_kwargs['paths'] = (Path(path) for path in aux)
        else:
            # Positions and other options
            pvmap_kwargs['positions'] = args.position

            # PA
            if args.pa is not None:
                pvmap_kwargs['pas'] = args.pa
            elif args.pvconfig and 'PAs' in args.pvconfig[section]:
                pvmap_kwargs['pas'] = args.pvconfig.getquantity(section, 'PAs')
            else:
                args.log.warning('No PAs given, skipping %s', section)
                continue

            # Length
            if args.length:
                pvmap_kwargs['length'] = args.length
            elif args.pvconfig and 'length' in args.pvconfig[section]:
                pvmap_kwargs['length'] = args.pvconfig.getquantity(section,
                                                                   'length')
            else:
                raise ValueError('No slit length given')
            args.log.info(
                f'Slit length: {args.length.value} {args.length.unit}')

        # Width is common
        if args.width:
            pvmap_kwargs['width'] = args.width
        elif args.pvconfig and 'width' in args.pvconfig[section]:
            pvmap_kwargs['width'] = args.pvconfig.getquantity(section, 'width')
        else:
            raise ValueError('No slit width given')
        args.log.info('Slit width: {args.width.value} {args.width.unit}')

        # Get pv maps:
        _calculate_pv_maps(subcube, invert=args.invert, log=args.log.warning,
                           **pvmap_kwargs)

def _calculate_pv_maps(cube, invert: bool = False, log: Callable = print,
                       **kwargs):
    """Calculate pv maps based on input.

    Args:
      cube: data cube.
      invert: optional; invert the velocity/position axes.
      log: optional; logging function.
      kwargs: pv map input parameters.
    """
    width = kwargs.pop('width')
    if kwargs.get('paths') is not None:
        paths = kwargs.pop('paths')
        _pv_maps_from_region(cube, paths, width, invert=invert, log=log,
                             **kwargs)
    else:
        pas = kwargs.pop('pas')
        positions = kwargs.pop('positions')
        length = kwargs.pop('length')
        _pv_maps_from_region(cube, pas, positions, length, width,
                             invert=invert, log=log, **kwargs)

def _pv_maps_from_region(cube: 'SpectralCube',
                         paths: Iterable[Path],
                         width: u.Quantity,
                         invert: bool = False,
                         output: Optional[Path] = None,
                         file_fmt: Optional[str] = None,
                         log: Callable = print,
                         **kwargs):
    """Iterate over paths to get pv maps.

    Args:
      cube: data cube.
      paths: `crtf` region files.
      width: slit width.
      invert: optional; invert the velocity/position axes.
      output: optional; output filename.
      file_fmt: optional; filename format.
      log: optional; logging function.
      kwargs: ignored keyword parameters.
    """
    # Iterate paths
    for i, path in enumerate(paths):
        log(f'Path = {path}')
        suffix_fmt = '.path{ind}'
        filename = _generate_filename(suffix_fmt, output=output,
                                      file_fmt=file_fmt,
                                      log=log, ind=i)
        get_pvmap_from_region(cube, path, width, filename=filename,
                              invert=invert)

def _pv_maps_from_slit(cube: 'SpectralCube',
                       pas: Iterable[u.Quantity],
                       positions: Iterable['SkyCoord'],
                       length: u.Quantity,
                       width: u.Quantity,
                       invert: bool = False,
                       output: Optional[Path] = None,
                       file_fmt: Optional[str] = None,
                       sources: Optional[List['AstroSource']] = None,
                       source_section: Optional[str] = None,
                       section: Optional[str] = None,
                       log: Callable = print):
    """Iterate over PAs and sources to get pv maps.

    Args:
      cube: data cube.
      pas: position angles.
      positions: slit central positions.
      length: slit length.
      width: slit width.
      invert: optional; invert the velocity/position axes.
      output: optional; output filename.
      file_fmt: optional; filename format.
      sources: optional; astro sources.
      source_section: optional; data section of the source.
      section: optional; pv map section.
      log: optional; logging function.
    """
    # Iterate position angles
    for pa in pas:
        log(f'Computing pv map for PA={pa}')
        for i, position in enumerate(positions):
            log(f'Position = {position}')
            suffix_fmt = '.ra{ra:.3f}_dec{dec:.3f}.PA{pa}'
            filename = _generate_filename(suffix_fmt, output=output,
                                          file_fmt=file_fmt, source=sources[i],
                                          source_section=source_section,
                                          log=log,
                                          ra=position.ra.deg,
                                          dec=position.dec.deg,
                                          pa=int(pa.value), section=section)

            # Get pv map
            get_pvmap_from_slit(cube, position, length, width, pa,
                                invert=invert, filename=filename, log=log)

def _generate_filename(suffix_fmt: str,
                       output: Optional[Path] = None,
                       file_fmt: Optional[Path] = None,
                       source: Optional['AstroSource'] = None,
                       source_section: Optional[str] = None,
                       log: Callable = print,
                       **kwargs):
    """Generate the pv map file name.

    At leas one of the following input need to be specified to generate the
    file name (in order of priority): `output`, `file_fmt` or
    (`source`, `source_section`)

    Args:
      suffix_fmt: format of the suffix.
      output: optional; output filename.
      file_fmt: optional; filename format.
      sources: optional; astro sources.
      source_section: optional; data section of the source.
      kwargs: keyword arguments for the suffix format.
    """
    # Get filename
    if output is not None:
        suffix = suffix_fmt.format(**kwargs) + output.suffix
        #suffix = output.suffix
        #suffix = (f'.ra{position.ra.deg:.3f}'
        #          f'_dec{position.ra.deg:.3f}'
        #          f'.PA{int(pa.value)}'
        #          f'{suffix}')
        filename = output.with_suffix(suffix)
    elif file_fmt is not None:
        filename = file_fmt.format(**kwargs)
        filename = Path(filename).expanduser()
    elif (source is not None and source_section is not None):
        cubename = source.config.getpath(source_section, 'file')
        if 'section' in kwargs:
            suffix = f".pvmap.{kwargs['section']}"
        else:
            suffix = '.pvmap'
        suffix = suffix + suffix_fmt.format(**kwargs) + cubename.suffix
        #suffix = cubename.suffix
        #suffix = (f'.pvmap.{section}'
        #          f'.ra{position.ra.deg}_dec{position.ra.deg}'
        #          f'.PA{pa.value:d}'
        #          f'{suffix}')
        filename = cubename.with_suffix(suffix)
    else:
        log('No output file!')
        filename = None

    return filename

def main(args: Sequence):
    """Extract the pv maps based on command line input.

    Args:
      args: command line arguments.
    """
    # Command line options
    pipe = [_minimal_set, _iter_sections]
    args_parents = [parents.logger('debug_line_helper.log'),
                    parents.source_position(required=False),
                    get_parent_parser()]
    parser = argparse.ArgumentParser(
        add_help=True,
        parents=args_parents,
        conflict_handler='resolve',
    )
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--source', action=LoadSources,
            help='Source configuration file')
    group1.add_argument('--cube', action=actions.LoadCube,
            help='Cube file name')
    parser.set_defaults()
    args = parser.parse_args()

    for step in pipe:
        step(args)


if __name__=='__main__':
    main(sys.argv[1:])