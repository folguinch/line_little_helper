from typing import List, Optional, Tuple, TypeVar, Union

from toolkit.array_utils import load_mixed_struct_array
import astropy.units as u
import astropy.table as aptable
import astroquery.splatalogue as splat
import numpy as np

from common_types import QPair, Table

Equivalency = TypeVar('Equivalency')

def to_rest_freq(freqs: u.Quantity, vlsr: u.Quantity, 
                 equivalencies: Equivalency) -> u.Quantity:
    """Convert input observed frequencies to rest frequencies.

    Args:
      freqs: observed frequencies.
      vlsr: LSR velocity.
      equivalencies: frequency to velocity equivalency.
    """
    # Convert to velocity
    vels = freqs.to(vlsr.unit, equivalencies=equivalencies)

    # Shift and convert back
    vels = vels - vlsr
    freqs = vels.to(freqs.unit, equivalencies=equivalencies)
    
    return freqs

def observed_to_rest(freqs: u.Quantity, vlsr: u.Quantity, equivalencies: dict,
                     spws_map: Optional[np.array] = None) -> u.Quantity:
    """Convert observed frequencies to rest frequencies.
        
    If the same equivalency is used for all frequencies, the equivalencies
    dictionary has to have an 'all' key with the equivalency.

    Args:
      freqs: observed frequencies.
      vlsr: LSR velocity.
      equivalencies: dictionary with the equivalencies to convert frequency to
        velocity.
      spws_map: optional; an array indicating the key in equivalencies for each
        frequency.
    Returns:
      An array with the rest frequencies.
    Raises:
      ValueError if an equivalency is not found.
    """
    if spws_map is None:
        if 'all' in equivalencies:
            return to_rest_freq(freqs, vlsr, equivalencies['all'])
        else:
            raise ValueError('Could not convert to rest frequency')
    else:
        for spw in np.unique(spws_map):
            # Equivalency
            if 'all' in equivalencies:
                equivalency = equivalencies['all']
            else:
                equivalency = equivalencies[spw]
            
            # Mask
            mask = spws_map == spw

            # Convert
            freqs[mask] = to_rest_freq(freqs[mask], vlsr, equivalency)
        return freqs

def query_lines(freq_range: QPair) -> Table:
    """Query splatalogue to get lines in range.

    Args:
      freq_range: frequency range.
    """
    columns = ('Species', 'Chemical Name', 'Resolved QNs', 
               'Freq-GHz(rest frame,redshifted)',
               'Meas Freq-GHz(rest frame,redshifted)', 
               'Log<sub>10</sub> (A<sub>ij</sub>)',
               'E_U (K)')
    query = splat.Splatalogue.query_lines(*freq_range,
                                          only_NRAO_recommended=True)[columns]
    query.rename_column('Chemical Name', 'Name')
    query.rename_column('Log<sub>10</sub> (A<sub>ij</sub>)', 'log10Aij')
    query.rename_column('E_U (K)', 'Eu')
    query.rename_column('Resolved QNs', 'QNs')
    query.rename_column('Freq-GHz(rest frame,redshifted)', 'Freq')
    query.rename_column('Meas Freq-GHz(rest frame,redshifted)', 'MeasFreq')
    query.sort(['Name', 'Species', 'Eu'])
    query['Eu'].info.unit = u.K
    query['Freq'].info.unit = u.GHz
    query['MeasFreq'].info.unit = u.GHz

    return query

def query_from_array(array: np.array,
                     units: dict,
                     freq_cols: List[str] = ['freq_low', 'freq_up'],
                     name_cols: Optional[List[str]] = None) -> dict:
    """Iterate over the frequency ranges in input array and obtain lines.
    
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

def combine_columns(table: Table, 
                    cols: list) -> Union[np.array, u.Quantity]:
    """Combine 2 columns in table replacing elements."""
    # Check
    if len(cols) != 2:
        raise IndexError('cols must have 2 values')
    
    # Initial value
    val = table[cols[0]]
    try:
        # Replace masked elements
        mask = val.mask
        val[mask].mask = False
        val[mask] = table[cols[1]][mask]
    except AttributeError:
        pass

    return val
