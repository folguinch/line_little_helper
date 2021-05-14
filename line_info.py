#!/bin/python3
"""Display line information upon request.

For usage and command line options run:
```bash
python line_info.py --help
```
"""
import argparse
import sys

from toolkit.argparse_tools import actions, parents

from lines import Molecule
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
    print(args.molec)

def main(args: list):
    """Search Splatalogue for line information.

    Args:
      args: command arguments.
    """
    freq_range_parent, freq_range_fn = query_freqrange(required=True)
    pipe = [freq_range_fn, _preproc, _proc]
    args_parents = [freq_range_parent,
                    parents.logger('debug_line_info.log')]
    parser = argparse.ArgumentParser(
        add_help=True,
        description='Display line information upon request.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents)
    parser.add_argument('--vlsr', action=actions.ReadQuantity, default=None,
                        help='Velocity shift for observed frequencies')
    parser.add_argument('molecule', nargs=1,
                        help='Molecule name or formula.')
    parser.set_defaults(pipe=pipe, query_name=None, freq_range=None, molec=None)
    args = parser.parse_args(args)

    # Run
    for step in args.pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
