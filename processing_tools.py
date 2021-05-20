"""Processing data from input."""
from typing import List, Optional, TypeVar, Union

from toolkit.astro_tools import cube_utils
import astropy.units as u
import astroquery.splatalogue as splat
import numpy as np

from common_types import QPair, Table
from spectrum import Spectra, Spectrum

Equivalency = TypeVar('Equivalency')

def get_spectral_equivalencies(restfreqs: list,
                               keys: Optional[list] = None) -> dict:
    """Get spectral equivalencies from rest frequencies.

    Args:
      restfreqs: rest frequencies.
      keys: optional; keys for the output dictionary.
    Returns:
      A dictionary with the astropy equivalency functions.
    """
    # Define keys
    if keys is None:
        keys = range(len(restfreqs))

    # Process restfreqs
    if len(restfreqs) == 1:
        equiv = {key: u.doppler_radio(restfreqs[0]) for key in keys}
    else:
        if len(keys) != len(restfreqs):
            raise ValueError('Size of keys and restfreqs do not match')
        aux = zip(keys, restfreqs)
        equiv = {key: u.doppler_radio(restfreq) for key, restfreq in aux}

    return equiv

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

def query_lines(freq_range: QPair, 
                line_lists: List[str] = ['CDMS', 'JPL'],
                **kwargs) -> Table:
    """Query splatalogue to get lines in range.

    Args:
      freq_range: frequency range.
      line_lists: optional; only use data from these databases.
      kwargs: optional; additional filters for `astroquery`.
    """
    columns = ('Species', 'Chemical Name', 'Resolved QNs',
               'Freq-GHz(rest frame,redshifted)',
               'Meas Freq-GHz(rest frame,redshifted)',
               'Log<sub>10</sub> (A<sub>ij</sub>)',
               'E_U (K)')
    query = splat.Splatalogue.query_lines(*freq_range,
                                          #only_NRAO_recommended=True,
                                          line_lists=line_lists,
                                          **kwargs)[columns]
    query.rename_column('Chemical Name', 'Name')
    query.rename_column('Log<sub>10</sub> (A<sub>ij</sub>)', 'log10Aij')
    query.rename_column('E_U (K)', 'Eu')
    query.rename_column('Resolved QNs', 'QNs')
    query.rename_column('Freq-GHz(rest frame,redshifted)', 'Freq')
    query.rename_column('Meas Freq-GHz(rest frame,redshifted)', 'MeasFreq')
    query.sort(['Name', 'Species', 'MeasFreq', 'Eu'])
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

      - Use the values from `name_cols`. Example: `name_cols=['spw','n']` will
        be converter to a key `'spw<spw value>_n<n value>'`.
      - A name column in the input array.
      - If spw and n columns are present, create a
        `key='spw<spw value>_<n value>'`.
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
    """Combine 2 columns in table replacing elements.
    
    Args:
      table: input `astropy.Table`.
      cols: columns to merge.
    """
    # Check
    if len(cols) != 2:
        raise IndexError('cols must have 2 values')

    # Initial value
    val = table[cols[0]]
    try:
        # Replace masked elements
        mask = val.mask
        val[mask].mask = False
        val.dtype = float
        val[mask] = table[cols[1]][mask]
    except AttributeError:
        pass

    return val

def zip_columns(table: Table, cols: List[str]):
    """Generates a zip of columns in a table.

    Args:
      table: input `astropy.Table`.
      cols: columns to zip.
    """
    aux = []
    for col in cols:
        try:
            aux.append(table[col].quantity)
        except TypeError:
            aux.append(table[col])

    return zip(*aux)

def refine_query(table: Table,
                 cols: List[str],
                 exclude_qns: Sequence[str] = (), 
                 bin_freq: Optional[u.Quantity] = None) -> :
    """Apply standard filters to a line query table.

    Args:
    """
    pass
    

def get_spectra(spectra: Spectra,
                cubes: Optional[list] = None,
                coords: Optional[list] = None,
                specs: Optional[list] = None,
                vlsr: Optional[u.Quantity] = None,
                rms: Optional[u.Quantity] = None,
                equivalencies: Optional[dict] = None,
                log = None) -> Spectra:
    # Observed data
    if cubes is not None:
        # Validate input
        if coords is None:
            raise ValueError('Coordinate needed for spectrum')

        # Extract spectra
        log.info('Extracting spectra from cubes')
        for cube in cubes:
            # Observed frequencies shifted during subtraction
            for coord in coords:
                spec = cube_utils.spectrum_at_position(cube, coord,
                                                       spectral_axis_unit=u.GHz,
                                                       vlsr=vlsr)
                spec = Spectrum(spec[0], spec[1].quantity,
                                restfreq=cube_utils.get_restfreq(cube),
                                rms=cube_utils.get_cube_rms(cube,
                                                            use_header=True))
                spectra.append(spec)
    elif specs is not None:
        freq_names = ['nu', 'freq', 'frequency', 'v', 'vel', 'velocity']
        int_names = ['F', 'f', 'Fnu', 'fnu', 'intensity', 'T', 'Tb']
        log.info('Reading input spectra')
        for key, (data, units) in enumerate(specs):
            # Spectral axis
            freq_name = list(filter(lambda x, unt=units: x in unt,
                                    freq_names))[0]
            xaxis = data[freq_name] * units[freq_name]

            # Intensity axis
            int_name = list(filter(lambda x, unt=units: x in unt,
                                   int_names))[0]
            spec = data[int_name] * units[int_name]

            # Noise
            if rms is not None:
                rms = rms[key]
            else:
                rms = None

            # Equivalencies
            if 'all' in equivalencies:
                equivalency = equivalencies
            else:
                equivalency = {'all': equivalencies[key]}
            restfreq = (0. * u.km/u.s).to(u.GHz,
                                          equivalencies=equivalency['all'])

            # Shifted spectrum
            if xaxis.unit.is_equivalent(u.Hz) and vlsr is not None:
                xaxis = observed_to_rest(xaxis, vlsr, equivalency)
                spec = Spectrum(xaxis, spec, restfreq=restfreq, rms=rms)
            elif xaxis.unit.is_equivalent(u.km/u.s):
                #if freq_to_vel is None:
                #    log.warn('Cannot convert spectral axis to GHz')
                #    continue
                vels = xaxis - vlsr
                xaxis = vels.to(u.GHz, equivalencies=equivalency['all'])
                spec = Spectrum(xaxis, spec, restfreq=restfreq, rms=rms)
            else:
                spec = Spectrum(xaxis, spec, restfreq=restfreq, rms=rms)

            # Store
            spectra.append(spec)
    else:
        pass

    return spectra
