"""Use the results from Cassis to construct maps.

It uses the file name default structure of `spectrum_extractor.py` to obtain
the coordinates and build the maps.
"""
from pathlib import Path
from typing import Optional, Dict, Sequence, Tuple
import argparse
import sys

from astropy.io import fits
from toolkit.argparse_tools import actions
import astropy.units as u
import numpy as np

class CassisResult():
    """Store results from Cassis runs.

    Attributes:
      species: species name.
      obs_spec: observed spectral data file name.
      mod_spec: model spectral data file name.
      stats: dictionary with the results.
    """
    PROPS = ['nmol', 'tex', 'fwhm', 'size', 'vlsr']
    UNITS = {'nmol': 1/u.cm**2, 'tex': u.K, 'fwhm': u.km/u.s, 
             'size': u.sr, 'vlsr': u.km/u.s}

    def __init__(self,
                 species: str,
                 obs_spec: Optional[Path] = None,
                 mod_spec: Optional[Path] = None,
                 **stats) -> None:
        """Initialize new object."""
        self.species = species.strip()
        self.obs_spec = obs_spec
        self.mod_spec = mod_spec
        self.stats = stats

    @classmethod
    def from_file(cls, filename: Path, component: int = 1):
        """Load results from a result file."""
        mod_spec = filename.with_suffix('.lis')
        stats = {f'{key}_{component}': None
                 for key in cls.PROPS}
        data = filename.read_text().split('\n')
        for field in data:
            if len(field) == 0 or field[0] in ['=', '*', '-']:
                continue
            elif '=' in field:
                try:
                    key, val = field.split('=')
                except ValueError:
                    key, *val = field.split('=')
                    val = '='.join(val)
                if key.strip() == 'species':
                    species = val.strip()
                elif key.strip() == 'inputFile':
                    obs_spec = Path(val.strip())
                elif key.strip() == 'chi2Min':
                    stats['chi2'] = [float(val.strip()), 0, 0]
                elif key.strip() == 'Chi2MinReduced':
                    stats['redchi2'] = [float(val.strip()), 0, 0]
                else:
                    pass
            else:
                key, *vals = field.split()
                if key in stats:
                    stats[key] = list(map(lambda x: float(x)*cls.UNITS[key[:-2]],
                                          vals))

        return cls(species, obs_spec=obs_spec, mod_spec=mod_spec, **stats)

class CassisResults(dict):
    """Stores the results from Cassis in a dictionary.

    The keys of the dictionary indicates the position in the map.
    """

    def __init__(self, pairs: Sequence[Tuple[Tuple[int,int], CassisResult]],
                 shape: Tuple[int,int], header: Dict):
        self._shape = shape
        self._header = header
        super().__init__(pairs)

    @classmethod
    def with_mask(cls,
                  mask: Path,
                  directory: Path,
                  fmt: str = 'spec_x{col:04d}_y{row:04d}.{ext}') -> Dict:
        """Load Cassis results for points in a mask.

        Args:
          mask: FITS file with 1 for valid points.
          directory: path for the results.
          fmt: optional; format of the file name
        """
        # Load mask
        img = fits.open(mask)[0]
        header = img.header
        img = img.data.astype(bool)
        rows, cols = np.indices(img.shape)

        # Iterate over valid results
        keys = []
        vals = []
        for row, col in zip(rows.flatten(), cols.flatten()):
            # Skip False
            if not img[row, col]:
                continue

            # Result file name
            filename = directory / fmt.format(row=row, col=col, ext='txt')
            if not filename.exists():
                continue
            keys.append((row, col))
            vals.append(CassisResult.from_file(filename))

        return cls(zip(keys, vals), shape=img.shape, header=header)

    def generate_map(self,
                     key: str,
                     filename: Optional[Path] = None,
                     *,
                     error_map: bool = False,
                     median_map: bool = False) -> np.array:
        """Build the maps from the results.

        Args:
          key: physical quantity to map.
          error_map: optional; map of the standard deviation values?
          median_map: optional; map of the median values?
          filename: optional; filename to save the maps as FITS.
        """
        # Determine type of map
        if error_map:
            ind = 2
        elif median_map:
            ind = 1
        else:
            ind = 0

        # Initial map
        data = np.zeros(self._shape)
        data[:] = np.nan

        # Fill data
        data_unit = None
        for pos, val in self.items():
            data[pos] = val.stats[key][ind].value
            if data_unit is None:
                data_unit = val.stats[key][ind].unit

        # Save
        if filename is not None:
            hdu = fits.PrimaryHDU(data, header=self._header)
            hdu.update_header()
            hdu.header['BUNIT'] = f'{data_unit:FITS}'
            hdu.writeto(filename, overwrite=True)

        return data

def _proc(args: argparse.ArgumentParser) -> None:
    """Process inputs."""
    model = CassisResults.with_mask(args.maskfile[0], args.indir[0])
    for key in args.keys:
        key_comp = f'{key}_{args.component[0]}'
        filename = args.indir[0] / f'{key_comp}_map.fits'
        print(filename)
        model.generate_map(key_comp, filename=filename)

def main(args: list) -> None:
    """Build the maps from Cassis results.

    Args:
      args: command line arguments.
    """
    pipe = [_proc]
    parser = argparse.ArgumentParser(
        add_help=True,
        description='Extract spectra from cube(s)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-c', '--component', nargs=1, default=[1],
                        help='Model component number')
    parser.add_argument('-k', '--keys', nargs='+', default=CassisResult.PROPS,
                        help='Physical quantities')
    parser.add_argument('maskfile', action=actions.CheckFile, nargs=1,
                        help='Mask file name')
    parser.add_argument('indir', action=actions.NormalizePath, nargs=1,
                        help='Directory with model files')
    args = parser.parse_args(args)

    # Run
    for step in pipe:
        step(args)

if __name__ == '__main__':
    main(sys.argv[1:])
