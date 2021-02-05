from typing import Optional

import numpy as np

from common_types import Path
from data_handler import ResultHandler, ResultsHandler, StatsHandler
from spectrum import Spectra

def _single_analysis(result: ResultHandler):
    """Perform simple analysis of a single set of results."""
    # Get peak frequency
    peak = result.get_peak_frequency()

    # Get centroid frequency
    centroid = result.get_centroid()

    # Calculate distances
    result.distance_to(peak, 'distance_peak')
    result.distance_to(centroid, 'distance_cen', sort=True, 
                       extra_sort_keys=['log10Aij'])

def simple_analysis(results: ResultsHandler, spectra: Spectra):
    """Perform simple analysis of individual results."""
    # Iterate over results
    for key, val in results.items():
        # Set spectrum
        val.spec_from_spectra(spectra)

        # Perform analysis
        _single_analysis(val)

def _get_values(result: ResultHandler, stats: StatsHandler, 
                key: str, top: Optional[int] = None) -> StatsHandler:
    """Store values in distance key per species in results.
    
    Args:
      result: result handler with the tables.
      stats: stats handler to store the handlers.
      key: distance key.
      top: optional; only use the first top results.
    Return:
      An updated stats handler.
    """
    # Filter top
    if top:
        species = result.table['Species'][:top]
        results = result.table[key][:top]
    else:
        species = result.table['Species']
        results = result.table[key]

    # Get values
    sps, ind = np.unique(species, return_inverse=True)
    for i, sp in enumerate(sps):
        mask = ind == i
        value = np.nanmin(results[mask]) * results.unit
        stats.update(sp, value)
    
    return stats

def results_per_line(results: ResultsHandler, top: Optional[int] = None,
                     distance_key: str = 'distance_cen') -> StatsHandler:
    """Collect results per line in all results."""
    # Initialize stats
    stats = StatsHandler()

    # Separate by species
    for result in results.values():
        stats = _get_values(result, stats, distance_key)

    return stats

def advanced_analysis(results: ResultsHandler, filename: Path, 
                      top: Optional[int] = None,
                      distance_key: str = 'distance_cen') -> None:
    """Analyse the results and save them."""
    # Claculate stats per species
    stats = results_per_line(results, distance_key=distance_key, top=top)
    stats_per_line = stats.stats_per_key()

    # Save stats
    fmt = '{name}\t{mean.value}\t{stdev.value}\t{median.value}'
    units = [f'{val.unit}' 
             for key, val in stats_per_line[0].items() if key != 'name']
    header = ['#Species\tmean\tstddev\tmedian', '#\t' + '\t'.join(units)]
    lines = [fmt.format(**res) for res in stats_per_line]
    with filename.open('w') as out:
        out.write('\n'.join(header + lines))
