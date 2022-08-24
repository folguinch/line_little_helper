"""Calculate requested moments for all trasitions of molecule."""
from typing import Sequence
import argparse
import sys

from toolkit.argparse_tools import actions
from toolkit.argparse_tools import parents
import numpy as np
import toolkit.argparse_tools.loaders as aploaders
import toolkit.astro_tools.cube_utils as cubeutils

import line_little_helper.moving_moments as mov_moments
import line_little_helper.subcube_extractor as extractor

def _proc(args: argparse.Namespace) -> None:
    """Process the data."""
    # Get molecule
    molec = mov_moments.get_molecule(args)

    # Compute cube rms
    rms = cubeutils.get_cube_rms(args.cube)
    args.log.info(f'Cube rms: {rms}')

    # Iterate over transitions
    table = []
    table_head = ['cube', 'moment0', 'moment1', 'cube_rms', 'molecule',
                  'qns', 'rest_freq', 'list']
    for transition in molec.transitions:
        args.log.info(f'Working ontransition:\n{transition}')
        # Get subcube
        args.linefreq = transition.restfreq
        extractor.get_subcube(args)

        # Transition name
        trans_name = transition.generate_name()

        # Table entry
        table_entry = {
            'cube': str(args.cubename[0]),
            'cube_rms': rms,
            'molecule': molec.name,
            'qns': transition.qns,
            'rest_freq': transition.restfreq,
            'list': f'{args.line_lists}',
        }

        # Calculate moments
        for nmoment in args.moments:
            moment = cubeutils.get_moment(args.subcube,
                                          nmoment,
                                          linefreq=args.linefreq,
                                          rms=rms,
                                          lower_limit=args.fluxlimit,
                                          skip_beam_error=True,
                                          log=args.log.info)

            # Check it has data
            if moment is not None and np.all(moment.data == np.nan):
                args.log.warn('No valid data for moment')
                continue

            # Save
            if moment is not None:
                outname = f'{args.output[0]}.{trans_name}.moment{nmoment}.fits'
                args.log.info(f'Writing: {outname}')
                moment.write(outname, overwrite=True)
                if nmoment in [0, 1]:
                    table_entry[f'moment{nmoment}'] = outname

        # Append to table
        table.append(table_entry)

    # Save table
    if args.table is not None:
        table = QTable(rows=table, names=table_head)
        if args.table[0].exists():
            table_old = QTable.read(args.table[0], format='ascii.ecsv')
            table = vstack([table_old, table])
        table.write(args.table[0], format='ascii.ecsv')

def main(args: Sequence[str]):
    """Main program."""
    # Argument parser
    pipe = [aploaders.load_spectral_cube, _proc]
    args_parents = [extractor.parent_parser(),
                    parents.logger('debug_molecule_moments.log')]
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=mov_moments.HelpFormatter,
        parents=args_parents,
        conflict_handler='resolve')
    parser.add_argument('--fluxlimit', nargs=2, action=actions.ReadQuantity,
                        help='Flux lower limit with units')
    parser.add_argument('--table', nargs=1, action=actions.NormalizePath,
                        default=None,
                        help='Output table ecsv file.')
    parser.add_argument('output', nargs=1, action=actions.NormalizePath,
                        help='The output basename')
    parser.add_argument('moments', nargs='*', type=int,
                        help='Moments to calculate')
    parser.set_defaults(subcube=None)
    args = parser.parse_args(args)
    for step in pipe:
        step(args)

if __name__=='__main__':
    main(sys.argv[1:])
