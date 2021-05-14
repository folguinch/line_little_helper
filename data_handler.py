"""Data handler classes for result and statistics tables."""
from typing import List, Optional

import astropy.units as u
import numpy as np

from common_types import QPair, Table, Path
from plot_tools import Plot, plot_spectrum
from processing_tools import query_lines, combine_columns
from spectrum import Spectra, Spectrum

class ResultHandler:
    """Class for handling the results from queries and postprocessing.

    Attributes:
      index: index number.
      table: results table.
      info: additional information (e.g. spw).
      spectrum: observed spectrum.
    """

    def __init__(self,
                 index: Optional[int] = None,
                 table: Optional[Table] = None,
                 info: Optional[dict] = None,
                 spectrum: Optional[Spectrum] = None) -> None:
        """Initialize a ResultHandler object."""
        self.index = index
        self.table = table
        self.info = info
        self.spectrum = spectrum

    @property
    def freq_low(self):
        return self.info.get('freq_low')

    @property
    def freq_up(self):
        return self.info.get('freq_up')

    @property
    def freq_range(self):
        return self.freq_low, self.freq_up

    @classmethod
    def from_query(cls, index: int, freq_range: QPair, info: dict):
        """Create a result handler from a query.

        The info dictionary is updated with the value of the input frequency
        range for future reference.

        Args:
          index: result index.
          freq_range: frequency range.
          infor: additional information dictionary.
        """
        # Query
        table = query_lines(freq_range)

        # Update info
        info['freq_low'] = freq_range[0]
        info['freq_up'] = freq_range[1]

        return cls(index=index, table=table, info=info)

    @staticmethod
    def generate_name(index: int,
                      fields: Optional[List[str]] = None,
                      values: Optional[np.array] = None,
                      index_key: str = 'n',
                      name_field: str = 'name') -> str:
        """Generate name for the handler from input.

        To determine the output name (in order of priority):
          - Use the values from fields. Example: fields=['spw','n'] will
            be converted to a key 'spw<spw value>_n<n value>'.
          - A name field in the input array.
          - If spw is values dtype fields:
              - If index key field is in values:
                name='spw<spw value>_<index_key value>'.
              - else: name='spw<spw value>_<index>'.
          - The index as string.

        Args:
          index: index of the result.
          fields: optional; fields used to determine the name.
          values: optional; a structured with values for the fields.
          index_key: optional; replace index with the value of index_key in the
            values array.
          name_field: optional; the name field to use as name.
        Returns:
          A name string.
        """
        # Key value
        if fields is not None and values is not None:
            key = []
            for field in fields:
                key.append(f'{field}{values[field]}')
            key = '_'.join(key)
        elif values is not None and name_field in values.dtype.names:
            key = f'{values[name_field]}'
        elif values is not None and 'spw' in values.dtype.names:
            if index_key in values.dtype.fields:
                key = f"spw{values['spw']}_{values[index_key]}"
            else:
                key = f"spw{values['spw']}_{index}"
        else:
            key = f'{index}'

        return key

    def get_peak_frequency(self) -> u.Quantity:
        """Calculate the peak line frequency from stored spectrum."""
        if self.spectrum is None:
            return None

        return self.spectrum.peak_frequency(self.freq_low, self.freq_up)

    def get_centroid(self) -> u.Quantity:
        """Calculate the line centroid from stored spectrum."""
        if self.spectrum is None:
            return None

        return self.spectrum.centroid(self.freq_low, self.freq_up)

    def set_spectrum(self, spectrum: Spectrum) -> bool:
        """Validate and set the spectrum of the result set."""
        # Spectrum exists
        if spectrum is not None:
            return False

        # Input spectrum
        if spectrum.is_in(self.central_freq()):
            self.spectrum = spectrum
            return True
        else:
            return False

    def spec_from_spectra(self, spectra: Spectra) -> None:
        """Set a validated spectrum from spectra."""
        # Set spectrum
        if self.spectrum is None:
            self.spectrum = spectra.get_spectrum(self.central_freq())

    def central_freq(self) -> u.Quantity:
        """Return the central frequency in the spectral range."""
        return (self.freq_low + self.freq_up) / 2

    def distance_to(self, freq: u.Quantity, name: str, sort: bool = False,
                    extra_sort_keys: list = []) -> None:
        """Calculate a new column with the distance to the input frequency.

        Args:
          freq: reference frequency.
          name: name of the new column.
          sort: optional; sort results based on the new column.
          extra_sort_keys: optional; additional keys to sort the data by
            (after name).
        """
        # Use Freq column
        linefreqs = combine_columns(self.table, ['Freq', 'MeasFreq'])
        dist = np.abs(freq - linefreqs)

        # Append new column
        self.table[name] = dist.to(u.MHz)

        # Sort
        if sort:
            self.table.sort([name] + extra_sort_keys)

    def plot(self, filename: Path, spectra: Optional[Spectra] = None,
             top: Optional[int] = None) -> Plot:
        """Plot results.

        Args:
          filename: plot file name.
          spectra: optional; store spectrum if not yet stored.
          top: optional; only plot the top n values from table.
        Returns:
          A tuple with the figure and axes objects.
        """
        # Check spectrum is set
        if spectra is not None:
            self.spec_from_spectra(spectra)

        # Plot
        if 'distance_cen' in self.table.colnames:
            color_key = 'distance_cen'
        else:
            color_key = None
        plt = plot_spectrum(self.spectrum, self.freq_range, table=self.table,
                            key='log10Aij', color_key=color_key, top=top)
        plt[0].savefig(filename, bbox_inches='tight')

        return plt

class ResultsHandler(dict):
    """Class for handling ResultHandler objects."""

    @classmethod
    def from_struct_array(cls,
                          array: np.array,
                          units: dict,
                          freq_cols: List[str] = ['freq_low', 'freq_up'],
                          name_cols: Optional[List[str]] = None,
                          info_keys: List[str] = ['spw'],
                          index_key: str = 'n') -> dict:
        """Create a results handler from structured array data.

        Args:
          array: numpy structured array with frequency range data.
          units: units of the columns in array.
          freq_cols: optional; name of the frequency columns in array.
          name_cols: optional; fields to use for naming the results.
          info_keys: optional; fields to store in the result info dict.
          index_key: optional; field to use as index.
        """
        keys = []
        results = []
        for i, row in enumerate(array):
            # Frequency range
            freq_range = (row[freq_cols[0]] * units[freq_cols[0]],
                          row[freq_cols[1]] * units[freq_cols[1]])

            # Result name
            name = ResultHandler.generate_name(i, fields=name_cols, values=row,
                                               index_key=index_key)
            keys.append(name)

            # Info
            names = row.dtype.names
            info = {key: row[key] for key in info_keys if key in names}

            # Query
            results.append(ResultHandler.from_query(i, freq_range, info))

        return cls(zip(keys, results))

    def write(self, filename: Path) -> None:
        """Write results to disk."""
        for key, result in self.items():
            if len(self) == 1:
                newfile = filename
            else:
                newfile = filename.with_suffix(f'.{key}.ecsv')
            result.table.write(newfile, format='ascii.ecsv')

    def plot(self, filename: Path, spectra: Optional[Spectra] = None,
             top: Optional[int] = None) -> None:
        """Plot all results.

        Args:
          filename: plot file name.
          spectra: optional; store spectrum if not yet stored.
          top: optional; only plot the top n values from table.
        """
        for key, val in self.items():
            if len(self) == 1:
                newfile = filename
            else:
                suff = filename.suffix
                newfile = filename.with_suffix(f'.{key}{suff}')
            val.plot(newfile, spectra=spectra, top=top)

class StatsHandler(dict):
    """Class for handling results and statistics."""

    def update(self, key: str, val: u.Quantity) -> None:
        """Update the dictionary and the values stored."""
        aux = self.get(key)
        if aux is None:
            try:
                val = np.array([val.value]) * val.unit
            except AttributeError:
                val = np.array([val]) * u.Unit('')
            super().__setitem__(key, val)
        else:
            self[key] = self[key].insert(-1, val)

    def stats_per_key(self) -> List[dict]:
        """Compute statistics per key."""
        aux = []
        for key, val in self.items():
            stats = {'name': key, 'mean': np.mean(val), 'stdev': np.std(val),
                     'median': np.median(val)}
            aux.append(stats)

        return aux

