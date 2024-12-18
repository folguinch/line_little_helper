"""Functions for processing `argparse` inputs."""
import argparse

from toolkit.astro_tools import cube_utils
from toolkit.argparse_tools import loaders

from .global_vars import ALMA_BANDS
from .molecule import get_molecule

def load_cube(args: argparse.Namespace) -> None:
    """Load a spectral cube from `args`."""
    loaders.load_spectral_cube(args, use_dask=args.use_dask)

def load_molecule(args: argparse.Namespace
                  ) -> 'line_little_helper.molecule.Molecule':
    """Generate a `Molecule` from argparse.

    Requires the argparse to have `cube` and `log` attributes. The `vlsr`
    attribute is also needed but does not need to be initialized.
    """
    molecule = get_molecule(args.molecule[0],
                            args.cube,
                            qns=args.qns[0],
                            onlyj=args.onlyj,
                            line_lists=args.line_lists,
                            vlsr=args.vlsr,
                            save_molecule=args.save_molecule[0],
                            restore_molecule=args.restore_molecule[0],
                            log=args.log.info)
    if 'molec' in vars(args):
        args.molec = molecule

    return molecule

def get_freqrange(args: argparse.Namespace) -> None:
    """Process the arguments for freq. range.

    Args:
      args: argument parser.
    """
    if args.freqrange is not None:
        args.freq_range = (args.freqrange[0], args.freqrange[1])
    elif args.alma is not None:
        args.freq_range =  ALMA_BANDS[args.alma[0]]
    else:
        msg = 'No input frequency range.'
        try:
            args.log.info(msg)
        except AttributeError:
            print(msg)

def get_channel_range(args: argparse.Namespace,
                      #allow_all: bool = False
                      ) -> None:
    """Convert spectral ranges to channel range."""
    # Filter args
    kwargs = {}
    optional = ['freq_range', 'vel_range', 'chan_range', 'win_halfwidth',
                'vlsr', 'linefreq', 'log']
    for key, val in vars(args).items():
        if key == 'win_halfwidth':
            kwargs[key] = val[0]
        elif key == 'log':
            kwargs[key] = val.info
        elif key in optional:
            kwargs[key] = val
        else:
            continue

    # Get range
    args.chan_range = cube_utils.limits_to_chan_range(args.cube,
                                                      #allow_all=allow_all,
                                                      **kwargs)

def get_subcube(args: argparse.Namespace) -> None:
    """Extract the subcube."""
    # Filter args
    kwargs = {}
    optional = ['freq_range', 'vel_range', 'chan_range', 'win_halfwidth',
                'blc_trc', 'xy_ranges', 'vlsr', 'linefreq', 'put_rms',
                'put_linefreq', 'rms', 'common_beam', 'shrink', 'log']
    for key, val in vars(args).items():
        if key == 'win_halfwidth':
            kwargs[key] = val[0]
        elif key == 'log':
            kwargs[key] = val.info
        elif key in optional:
            kwargs[key] = val
        else:
            continue

    # Get subcube
    args.subcube = cube_utils.get_subcube(args.cube, **kwargs)

def set_fluxlimit(args: argparse.Namespace, get_rms: bool = True) -> None:
    """Set the `flux_limit` attribute of the parser.

    If `get_rms` is `True` and `args.flux_limit` or `args.rms` are `None`, then
    the `args.cube` (if any) is used to estimate the rms.
    """
    if args.flux_limit is not None:
        return
    elif args.rms is not None:
        args.flux_limit = args.rms * args.nsigma
    elif 'cube' in vars(args) and args.cube is not None and get_rms:
        args.rms = cube_utils.get_cube_rms(args.cube, sampled=args.sampled_rms,
                                           log=args.log.info)
        args.log.info(f'Cube rms = {args.rms.value} {args.rms.unit}')
        args.flux_limit = args.rms * args.nsigma
    else:
        args.log.warning('Cannot set flux limit')
