from typing import List, Optional, Tuple

from toolkit.array_utils import load_mixed_struct_array
import astropy.units as u
import astroquery.splatalogue as splat
import numpy as np

# Types
Pair = Tuple[u.Unit, u.Unit]

def query_lines(freq_range: Pair) -> astropy.table.Table:
    """Query splatalogue to get lines in range.

    Args:
      freq_range: frequency range.
    """
    columns = ('Species', 'Chemical Name', 'Resolved QNs', 'Freq-GHz',
               'Meas Freq-GHz', 'Log<sub>10</sub> (A<sub>ij</sub>)',
               'E_U (K)')
    query = splat.Splatalogue.query_lines(*freq_range,
                                          only_NRAO_recommended=True)[columns]
    query.rename_column('Log<sub>10</sub> (A<sub>ij</sub>)', 'log10Aij')
    query.rename_column('E_U (K)', 'EU/K')
    query.rename_column('Resolved QNs', 'QNs')
    query.rename_column('Freq-GHz', 'Freq/GHz')
    query.rename_column('Meas Freq-GHz', 'MeasFreq/GHz')

    return query

def query_from_array(array: np.array,
                     units: dict,
                     freq_cols: List[str, str] = ['freq_low', 'freq_up'],
                     name_cols: Optional[List[str]] = None) -> dict:
    """Iterate over the lines of input file and obtain lines.
    
    To determine the output dictionary keys (in order of priority):
      - Use the values from name_cols. Example: name_cols=['spw','n'] will
        be converter to a key 'spw<spw value>_n<n value>'. 
      - A name column in the input array.
      - If spw and n columns are present, create a 
        key='spw<spw value>_<n value>'.
      - The number of the row as string.

    Args:
      array: structured array with the columns needed.
      units: the units of each array column.
      freq_cols: optional; name of the frequency columns.
      name_cols: optional; columns used to determine the output keys.
    Returns:
      Dictionary with the results where the keys are given by the rows names.
    """
    # Iterate
    results = {}
    for i, row in enumerate(array):
        # Frequency range
        freq_range = (row[freq_cols[0]] * units[freq_cols[0]],
                      row[freq_cols[1]] * units[freq_cols[1]])
        
        # Key value
        if name_cols is not None:
            key = []
            for colname in name_cols:
                key.append(f'{colname}{row[colname]}')
            key = '_'.join(key)
        elif 'name' in array.dtype.fields:
            key = array['name']
        elif 'spw' in array.dtype.fields and 'n' in array.dtype.fields:
            key = f"spw{row['spw']}_{row['n']}"
        else:
            key = f'{i}'

        # Query
        results[key] = query_lines(freq_range)

    return results
