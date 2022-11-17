#!/bin/python3
"""Calculate a position-velocity (pv) map."""
from typing import Callable, Optional, Sequence, Iterable, List
from pathlib import Path
import argparse
import sys

from astro_source.source import LoadSource
from astropy.io import fits
from astropy.units.equivalencies import doppler_radio
from pvextractor import PathFromCenter, extract_pv_slice
from pvextractor import Path as pvPath
from regions import Regions
from spectral_cube import SpectralCube
from toolkit.argparse_tools import actions, parents, functions
from toolkit.astro_tools import cube_utils
import astropy.units as u
import numpy as np

from line_little_helper.scripts.argparse_parents import line_parents
from line_little_helper.molecule import get_molecule

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
                        rms: Optional[u.Quantity] = None,
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
      rms: optional; pv map rms.
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

    # RMS
    if rms is not None:
        pv_map.header['RMS'] = rms.to(cube.unit).value

    if filename is not None:
        log(f'Saving file: {filename}')
        pv_map.writeto(filename, overwrite=False)

    return pv_map

def get_pvmap_from_region(cube: 'SpectralCube',
                          region: Path,
                          width: u.Quantity,
                          invert: bool = False,
                          rms: Optional[u.Quantity] = None,
                          filename: Optional[Path] = None,
                          log: Callable = print) -> 'PrimaryHDU':
    """Calculate a position velocity map from a CASA poly region.

    Args:
      cube: spectral cube.
      region: the `crtf` region filename.
      width: width of the slit.
      invert: optional; invert the velocity/position axes
      rms: optional; pv map rms.
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

    # RMS
    if rms is not None:
        pv_map.header['RMS'] = rms.to(cube.unit).value

    if filename is not None:
        log(f'Saving file: {filename}')
        pv_map.writeto(filename, overwrite=False)

    return pv_map

def get_parent_parser() -> argparse.ArgumentParser:
    """Base parent parser for pv maps."""
    parents = [line_parents('vlsr', 'molecule')]
    parent_parser = argparse.ArgumentParser(add_help=False, parents=parents)
    parent_parser.add_argument(
        '--invert',
        action='store_true',
        help='Invert axes')
    parent_parser.add_argument(
        '--estimate_error',
        action='store_true',
        help='Estimate uncertainties by propagating errors')
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
        help='Read a CASA poly region as pv path')
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
        args.sections = ['cube']
    elif args.cube and args.width and args.path and args.pvconfig is None:
        args.sections = ['cube']
    elif args.pvconfig is not None:
        args.sections = args.sections or args.pvconfig.sections()
        args.log.info('PV config file selected sections: %r', args.sections)
    else:
        raise ValueError('Not enough information for pv map')

def _load_cube(args, section):
    """Load the cube depending on the input data."""
    if args.cube is not None:
        args._cube = args.cube
        return None
    elif args.pvconfig is not None:
        if args.source is not None:
            source_section = args.pvconfig.get(section, 'source_section',
                                               fallback=section)
            args._cube = args.source[source_section]
            return source_section
        elif 'cube' in args.pvconfig[section]:
            args._cube = SpectralCube.read(args.pvconfig[section]['cube'])
            return None
        else:
            raise ValueError('Cannot load the input cube')
    else:
        raise ValueError('Cannot load the input cube')

def _iter_sections(args: argparse.Namespace) -> None:
    """Iterate over the sections in configuration."""
    # Compute pv maps
    for section in args.sections:
        args.log.info('PV map for: %s', section)

        # PV map options
        has_paths = (args.paths or
                     (args.pvconfig and 'paths' in args.pvconfig[section]))
        pvmap_kwargs = {'output': args.output,
                        'file_fmt': args.pvconfig.get(section, 'file_fmt',
                                                      fallback=None),
                        'source': args.source,
                        'section': section,
                        }
        if args.pvconfig:
            pvmap_kwargs['file_fmt'] = args.pvconfig.get(section, 'file_fmt',
                                                         fallback=None)

        # Load data
        source_section = _load_cube(args, section)
        pvmap_kwargs['source_section'] = source_section
        rest_freq = cube_utils.get_restfreq(args._cube)

        # Estimate rms
        if args.estimate_error:
            if (args.source is not None and
                'rms' in args.source.config[source_section]):
                cube_rms = args.source.get_quantity('rms',
                                                    section=source_section)
                cube_rms = cube_rms.to(args._cube.unit)
            else:
                cube_rms = cube_utils.get_cube_rms(args._cube, log=args.log.info)
            args.log.info('Cube rms: %s', cube_rms)

        # Load position or path
        if args.pvconfig and 'moment0' in args.pvconfig[section]:
            raise NotImplementedError
            #moment0 = Image(args.pvconfig.get(section, 'moment0'))
            #xmax, ymax = moment0.max_pix()
            #rest_freq = moment0.header['RESTFRQ'] * u.Hz
        elif args.paths is not None or args.pvconfig is not None:
            if args.paths is not None:
                pvmap_kwargs['paths'] = args.paths
                args.log.info('Using input paths')
            elif args.pvconfig and 'paths' in args.pvconfig[section]:
                aux = args.pvconfig.get(section, 'paths').split(',')
                pvmap_kwargs['paths'] = (Path(path.strip()) for path in aux)
                args.log.info('Using config paths')
            elif args.source is not None:
                # This assumes the cubes are the same for all sources
                functions.pixels_to_positions(args,
                                              wcs=args._cube.wcs.sub(['longitude',
                                                                     'latitude']))
                pvmap_kwargs['positions'] = args.position
                args.log.info('Using source positions for slit')
            else:
                raise ValueError('Cannot determine the slit or region')
        else:
            raise NotImplementedError('Only pvconfig for now')

        # vlsr
        if args.vlsr is not None:
            args.log.info('Input LSR velocity: %s', args.vlsr)
        elif args.pvconfig and 'v_lsr' in args.pvconfig[section]:
            args.vlsr = args.pvconfig.getquantity(section, 'v_lsr')
            args.log.info('LSR velocity from config: %s', args.vlsr)
        elif args.source is not None and args.source.vlsr is not None:
            args.vlsr = args.source.vlsr
            args.log.info('LSR velocity from source: %s', args.vlsr)
        else:
            vlsr = 0. * u.km / u.s
            args.log.warning('LSR velocity: %s', vlsr)

        # Line frequency
        args.log.info('Rest frequency: %s', rest_freq.to(u.GHz))
        if args.line_freq:
            args.log.info('Line frequency: %s', args.line_freq.to(u.GHz))
        elif args.molecule and args.qns:
            mol = get_molecule(args.molecule[0], args._cube, qns=args.qns,
                               onlyj=args.onlyj, line_lists=args.line_lists,
                               vlsr=args.vlsr)
            if len(mol.transitions) != 1:
                raise ValueError(('Number of transitions: '
                                  f'{len(mol.transitions)}'))
            args.line_freq = mol.transitions[0].restfreq
        elif args.pvconfig is not None:
            if 'line_freq' in args.pvconfig[section]:
                args.line_freq = args.pvconfig.getquantity(section, 'line_freq')
                args.log.info('Line frequency: %s', args.line_freq.to(u.GHz))
            elif ('molecule' in args.pvconfig[section] and
                  'qns' in args.pvconfig[section]):
                molecule = args.pvconfig[section]['molecule']
                qns = args.pvconfig[section]['qns']
                line_lists = args.line_lists
                if (line_lists is None and
                    'line_lists' in args.pvconfig[section]):
                    line_lists = args.pvconfig[section]['line_lists'].split(',')
                    line_lists = [x.strip() for x in line_lists]
                mol = get_molecule(molecule, args._cube, qns=qns,
                                   onlyj=args.onlyj,
                                   line_lists=line_lists,
                                   vlsr=args.vlsr)
                if len(mol.transitions) != 1:
                    raise ValueError(('Number of transitions: '
                                      f'{len(mol.transitions)}'))
                args.line_freq = mol.transitions[0].restfreq
        else:
            args.log.warning('No line frequency given, using whole range')
            args.line_freq = None

        # Get frequency/velocity slab
        subcube = args._cube.with_spectral_unit(u.GHz,
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

        # Check input for positions
        if 'positions' in pvmap_kwargs:
            # PA
            if args.pa is not None:
                pvmap_kwargs['pas'] = args.pa
            elif args.pvconfig and 'PAs' in args.pvconfig[section]:
                pvmap_kwargs['pas'] = args.pvconfig.getquantity(section, 'PAs')
                try:
                    pvmap_kwargs['pas'] = list(pvmap_kwargs['pas'])
                except TypeError:
                    pvmap_kwargs['pas'] = [pvmap_kwargs['pas']]
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
            args.log.info('Slit length: %f %s', pvmap_kwargs['length'].value,
                          pvmap_kwargs['length'].unit)

        # Width is common
        if args.width:
            pvmap_kwargs['width'] = args.width
        elif args.pvconfig and 'width' in args.pvconfig[section]:
            pvmap_kwargs['width'] = args.pvconfig.getquantity(section, 'width')
        else:
            raise ValueError('No slit width given')
        args.log.info('Slit width: %f %s', pvmap_kwargs['width'].value,
                      pvmap_kwargs['width'].unit)

        # Estimate error
        if args.estimate_error:
            pixsize = subcube.wcs.sub(['longitude', 'latitude'])
            pixsize = pixsize.proj_plane_pixel_area()
            pixsize = np.sqrt(pixsize).to(pvmap_kwargs['width'].unit)
            npix = pvmap_kwargs['width'] / pixsize
            args.log.info('Pixels in slit: %f', npix)
            cube_rms = cube_rms / np.sqrt(npix)
            args.log.info('Estimated pvmap rms: %s', cube_rms)
        else:
            cube_rms = None

        # Get pv maps:
        args.filenames = _calculate_pv_maps(subcube, invert=args.invert,
                                            rms=cube_rms, log=args.log.warning,
                                            **pvmap_kwargs)

def _calculate_pv_maps(cube, invert: bool = False, log: Callable = print,
                       rms: Optional[u.Quantity] = None,
                       **kwargs) -> List[Path]:
    """Calculate pv maps based on input.

    Args:
      cube: data cube.
      invert: optional; invert the velocity/position axes.
      rms: optional; pv map rms.
      log: optional; logging function.
      kwargs: pv map input parameters.
    """
    # Common beam
    cube = cube_utils.to_common_beam(cube, log=log)

    # Compute
    width = kwargs.pop('width')
    if kwargs.get('paths') is not None:
        paths = kwargs.pop('paths')
        filenames = _pv_maps_from_region(cube, paths, width, invert=invert,
                                         rms=rms, log=log, **kwargs)
    else:
        pas = kwargs.pop('pas')
        positions = kwargs.pop('positions')
        length = kwargs.pop('length')
        filenames = _pv_maps_from_slit(cube, pas, positions, length, width,
                                       invert=invert, rms=rms, log=log,
                                       **kwargs)

    return filenames

def _pv_maps_from_region(cube: 'SpectralCube',
                         paths: Iterable[Path],
                         width: u.Quantity,
                         invert: bool = False,
                         rms: Optional[u.Quantity] = None,
                         output: Optional[Path] = None,
                         file_fmt: Optional[str] = None,
                         log: Callable = print,
                         **kwargs) -> List[Path]:
    """Iterate over paths to get pv maps.

    Args:
      cube: data cube.
      paths: `crtf` region files.
      width: slit width.
      invert: optional; invert the velocity/position axes.
      rms: optional; pv map rms.
      output: optional; output filename.
      file_fmt: optional; filename format.
      log: optional; logging function.
      kwargs: ignored keyword parameters.
    """
    # Iterate paths
    filenames = []
    for i, path in enumerate(paths):
        log(f'Path = {path}')
        #suffix_fmt = '.path{ind}'
        #filename = _generate_filename(suffix_fmt, output=output,
        #                              file_fmt=file_fmt,
        #                              log=log, ind=i)
        filename = output.parent / f'{path.stem}.fits'
        get_pvmap_from_region(cube, path, width, filename=filename,
                              invert=invert, rms=rms)
        filenames.append(filename)

    return filenames

def _pv_maps_from_slit(cube: 'SpectralCube',
                       pas: Iterable[u.Quantity],
                       positions: Iterable['SkyCoord'],
                       length: u.Quantity,
                       width: u.Quantity,
                       invert: bool = False,
                       rms: Optional[u.Quantity] = None,
                       output: Optional[Path] = None,
                       file_fmt: Optional[str] = None,
                       source: Optional['AstroSource'] = None,
                       source_section: Optional[str] = None,
                       section: Optional[str] = None,
                       log: Callable = print) -> List[Path]:
    """Iterate over PAs and sources to get pv maps.

    Args:
      cube: data cube.
      pas: position angles.
      positions: slit central positions.
      length: slit length.
      width: slit width.
      invert: optional; invert the velocity/position axes.
      rms: optional; pv map rms.
      output: optional; output filename.
      file_fmt: optional; filename format.
      source: optional; astro source.
      source_section: optional; data section of the source.
      section: optional; pv map section.
      log: optional; logging function.
    """
    # Iterate position angles
    filenames = []
    for pa in pas:
        log(f'Computing pv map for PA={pa}')
        for i, position in enumerate(positions):
            log(f'Position = {position}')
            if section is not None:
                suffix_fmt = '.{section}.ra{ra:.5f}_dec{dec:.5f}.PA{pa}'
            else:
                suffix_fmt = '.ra{ra:.5f}_dec{dec:.5f}.PA{pa}'
            filename = _generate_filename(suffix_fmt, output=output,
                                          file_fmt=file_fmt, source=source,
                                          source_section=source_section,
                                          log=log,
                                          ra=position.ra.deg,
                                          dec=position.dec.deg,
                                          pa=int(pa.value), section=section)

            # Get pv map
            get_pvmap_from_slit(cube, position, length, width, pa,
                                invert=invert, rms=rms, filename=filename,
                                log=log)
            filenames.append(filename)

    return filenames

def _generate_filename(suffix_fmt: str,
                       output: Optional[Path] = None,
                       file_fmt: Optional[Path] = None,
                       source: Optional['AstroSource'] = None,
                       source_section: Optional[str] = None,
                       log: Callable = print,
                       **kwargs):
    """Generate the pv map file name.

    At least one of the following input need to be specified to generate the
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

def pvmap_extractor(args: Sequence):
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
    group1.add_argument('--source', action=LoadSource,
                        help='Source configuration file')
    group1.add_argument('--cube', action=actions.LoadCube,
                        help='Cube file name')
    parser.set_defaults(rms=None, filenames=None, _cube=None)
    args = parser.parse_args(args)

    for step in pipe:
        step(args)

    return args.filenames

if __name__=='__main__':
    pvmap_extractor(sys.argv[1:])
