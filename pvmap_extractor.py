"""Calculate the position-velocity (pv) map.


"""
from typing import Callable, Optional, Sequence
from pathlib import Path
import argparse
import sys

from astro_source.source import LoadSources
from astropy.io import fits
from astropy.units.equivalencies import doppler_radio
from pvextractor import PathFromCenter, extract_pv_slice
from toolkit.argparse_tools import actions, parents, functions
from toolkit.astro_tools import cube_utils
#from myutils.image_utils import lin_vel_gradient
#from myutils.file_utils import write_txt
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

def get_pvmap(cube: 'SpectralCube',
              position: 'SkyCoord',
              length: u.Quantity,
              width: u.Quantity,
              angle: u.Quantity,
              invert: bool = False,
              filename: Optional[Path] = None,
              log: Callable = print) -> 'PrimaryHDU':
    """Calculate a position velocity map.

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

def get_parent_parser() -> argparse.ArgumentParser:
    """Base parent parser for pv maps."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--stats',
        action='store_true',
        help='Compute statistics (e.g. linear velocity gradient)')
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
        '--output',
        action=actions.NormalizePath,
        help='File output')

    return parent_parser

def save_stats(stats, file_fmt):
    # Save fit
    text = 'Linear velocity gradient regression:\n'
    labels = ['Slope', 'Slope error', 'Intercept']
    for pair in zip(labels, stats):
        text += '%s = %s\n' % pair
    write_txt(text.strip(), file_fmt % '_lin_vel_fit.txt')

    # Save distribution
    fmt = lambda x: '%10.4f\t%10.4f' % x
    text = '#%10s\t%10s\n' % ('v', 'offset')
    text += '#%10s\t%10s\n' % (stats[-2].to(u.km/u.s).unit, 
            stats[-1].to(u.arcsec).unit)
    text += '\n'.join(map(fmt, zip(stats[-2].to(u.km/u.s).value, 
        stats[-1].to(u.arcsec).value)))
    write_txt(text, file_fmt % '_lin_vel.dat')

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
    else:
        args.sections = args.sections or args.pvconfig.sections()
        args.log.info('PV config file selected sections: %r', args.sections)

def _iter_sections(args: argparse.Namespace) -> None:
    """Iterate over the sections in configuration."""
    # Compute pv maps
    source_data = None
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
            args.log.warn('No line frequency given, using whole range')
            args.line_freq = None

        # vlsr
        if args.vlsr is not None:
            args.log.info('Input LSR velocity: %s', args.vlsr)
        elif args.pvconfig and 'v_lsr' in args.pvconfig[section]:
            args.vlsr = args.pvconfig.getquantity(section, 'v_lsr')
            args.log.info('LSR velocity from config: %s', args.vlsr)
        else:
            vlsr = 0. * u.km / u.s
            args.log.warn('LSR velocity: %s', vlsr)

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
                args.log.warn('No frequency slab width, using whole range')
                dfreq2 = None

            # Get the spectral slab
            if dfreq2 is not None:
                subcube = get_spectral_slab(subcube, args.line_freq, dfreq2,
                                            rest_freq=rest_freq,
                                            vlsr=args.vlsr)
                cdelt = np.abs(subcube.spectral_axis[0] - \
                               subcube.spectral_axis[1])
                args.log.info('New cube shape: %r', subcube.shape)
                args.log.info('New cube spectral axis step: %s', cdelt.to(u.MHz))
        else:
            pass

        # Change units to velocity
        # The zero velocity is in the line center
        subcube = subcube.with_spectral_unit(u.km/u.s,
                                             velocity_convention='radio',
                                             rest_value=args.line_freq)
        
        # PA
        if args.pa is not None:
            pass
        elif args.pvconfig and 'PAs' in args.pvconfig[section]:
            args.pa = args.pvconfig.getquantity(section, 'PAs')
        else:
            args.log.warn('No PAs given, skipping %s', section)
            continue

        # Path
        if args.length:
            pass
        elif args.pvconfig and 'length' in args.pvconfig[section]:
            args.length = args.pvconfig.getquantity(section, 'length')
        else:
            raise ValueError('No slit length given')
        args.log.info('Slit length: {0.value} {0.unit}'.format(args.length))
        if args.width:
            pass
        elif args.pvconfig and 'width' in args.pvconfig[section]:
            args.width = args.pvconfig.getquantity(section, 'width')
        else:
            raise ValueError('No slit width given')
        args.log.info('Slit width: {0.value} {0.unit}'.format(args.width))

        
        for pa in args.pa:
            args.log.info('Computing pv map for PA=%s', pa)
            for i, position in enumerate(args.position):
                args.log.info('Position = %s', position)

                # Get filename
                if args.output is not None:
                    suffix = args.output.suffix
                    suffix = (f'.ra{position.ra.deg:.3f}'
                              f'_dec{position.ra.deg:.3f}'
                              f'.PA{int(pa.value)}'
                              f'{suffix}')
                    filename = args.output.with_suffix(suffix)
                elif args.pvconfig and 'file_fmt' in args.pvconfig[section]:
                    filename = args.pvconfig.get(section, 'file_fmt')
                    filename = filename.format(pa=pa.value,
                                               ra=position.ra.deg,
                                               dec=position.dec.deg)
                    filename = Path(filename).expanduser()
                elif args.pvconfig and source_section is not None:
                    cubename = args.source[i].config.getpath(source_section,
                                                             'file')
                    suffix = cubename.suffix
                    suffix = (f'.pvmap.{section}'
                              f'.ra{position.ra.deg}_dec{position.ra.deg}'
                              f'.PA{pa.value:d}'
                              f'{suffix}')
                    filename = cubename.with_suffix(suffix)
                else:
                    args.log.warn('No output file')
                    filename = None

                # Get pv map
                pv_map = get_pvmap(subcube, position, args.length, args.width,
                                   pa, filename=filename, log=args.log.info)
    
                ## Statistics
                #if args.stats and args.pvconfig:
                #    # Linear velocity gradient
                #    rms = args.pvconfig.getquantity(section, 'rms')
                #    nsigma = args.pvconfig.getfloat(section, 'nsigma')
                #    if 'xpixsize' in args.pvconfig[section] and \
                #            'ypixsize' in args.pvconfig[section]:
                #        logger.info('Using configuration file pixsizes:')
                #        pixsizes = (args.pvconfig.getquantity(section, 'xpixsize'),
                #                args.pvconfig.getquantity(section, 'ypixsize'))
                #        logger.info('\tx-axis pixel size: %s', pixsizes[0])
                #        logger.info('\ty-axis pixel size: %s', pixsizes[1])
                #    else:
                #        pixsizes = (None, None)
                #    if 'xlim%i' % pa.value in args.pvconfig[section]:
                #        logger.info('Filtering x-axis pixels')
                #        filterx = args.pvconfig.getfloatlist(section, 
                #                'xlim%i' % pa.value)
                #    else:
                #        filterx=None
                #    stats = lin_vel_gradient(pv_map, sigma=rms.value, nsigma=nsigma,
                #            pixsizes=pixsizes, filterx=filterx)
                #
                #    # Save stats
                #    if file_name:
                #        save_stats(stats, file_name % (pa.value, '%s'))

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
