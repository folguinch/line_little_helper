#!/bin/python3
"""Use the results from Cassis to construct quantity maps.

It uses the file name default structure of `spectrum_extractor.py` to obtain
the coordinates and build the maps.
"""
from pathlib import Path
from typing import Optional, Dict, Sequence, Tuple
import argparse
import math
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
    PROPS = ('nmol', 'tex', 'fwhm', 'size', 'vlsr')
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

    @staticmethod
    def data_from_lam(lam_file: Path, component: int = 1):
        """Extract the best model information from best model file."""
        # Read parameters from file
        data = lam_file.read_text().split('\n')
        comp = f'Comp{component+1}'
        valid_keys = {f'{comp}Mol1NSp': 'nmol',
                      f'{comp}Mol1Tex': 'tex',
                      f'{comp}Mol1FWHM': 'fwhm',
                      f'{comp}Mol1Size': 'size',
                      f'{comp}Vlsr': 'vlsr'}
        stats = {}
        for field in data:
            # Skip comments
            if field.startswith('#'):
                continue
            
            # Search for values
            try:
                key, val = field.split('=')
            except ValueError:
                key, *val = field.split('=')
                val = '='.join(val)
            if key in valid_keys:
                vkey = valid_keys[key]
                value = float(val) * CassisResult.UNITS[vkey]
                stats[f'{vkey}_{component}'] = [value, None, None]
            elif key == f'{comp}Mol1Species':
                species = val.strip()
            elif key == 'nameData':
                obs_spec = Path(val.strip())

        return stats, species, obs_spec

    @staticmethod
    def values_from_txt(txt_file: Path, keys: Sequence[str]) -> Dict:
        """Read the requested values of the best model."""
        data = txt_file.read_text().split('\n')
        vals = {}
        for field in data:
            try:
                key, val = tuple(map(lambda x: x.strip(), field.split('=')))
            except ValueError:
                key, *val = tuple(map(lambda x: x.strip(), field.split('=')))
                val = '='.join(val)
            if key not in keys:
                continue
            elif key == 'Chi2MinReduced':
                vals[key] = float(val) * u.Unit(1)
            elif key == 'inputFile':
                vals[key] = Path(val)
            else:
                vals[key] = val

        return vals

    @classmethod
    def from_best_file(cls,
                       filename: Path,
                       component: int = 1,
                       stddev: int = 100,
                       best: Optional[int] = None,
                       species: Optional[str] = None):
        """Load results from a result file.
        
        If `best` is given, then the best model will be computed from the
        average of the `best` models sorted by `chi2`. In this case the value
        of `stdev` is replaced by `best`. 

        Args:
          filename: any of the files produced by CASSIS.
          component: optional; number of the component to load.
          stddev: optional; number of models for calculating the std dev of
            parameters.
          best: optional; number of model to average.
          species: optional; name of the molecule.
        """
        # Files
        mod_spec = filename.with_suffix('.lis')
        log_file = filename.with_suffix('.log')
        lam_file = filename.with_suffix('.lam')
        txt_file = filename.with_suffix('.txt')

        # Model list
        dtype = {'names': ('n',) + cls.PROPS + ('chi2', 'rate'),
                 'formats':(np.int32,) + (np.float128,)*7}
        mod_list = np.loadtxt(log_file, skiprows=2, dtype=dtype)
        mod_list.sort(order='chi2')

        # Extract information
        obs_spec = None
        if lam_file.is_file():
            stats, species, obs_spec = CassisResult.data_from_lam(
                lam_file, component=component)
            from_txt = ('Chi2MinReduced',)
        else:
            stats = {f'{key}_{component}': None for key in cls.PROPS}
            from_txt = ('Chi2MinReduced', 'species', 'inputFile')
        if txt_file.is_file():
            vals_txt = CassisResult.values_from_txt(txt_file, from_txt)
            redchi2 = vals_txt['Chi2MinReduced']
            if len(vals_txt) == 3:
                species = vals_txt['species']
                obs_spec = vals_txt['inputFile']
        else:
            redchi2 = 0*u.Unit(1)

        # Store other stats
        for prop in cls.PROPS:
            # Read Value
            stat = f'{prop}_{component}'
            if best is not None:
                val = [np.mean(mod_list[prop][:best]) * cls.UNITS[prop], None,
                       None]
                stddev = best
            else:
                val = stats[f'{prop}_{component}']
                if val is None:
                    val = [mod_list[prop][0] * cls.UNITS[prop], None, None]
            
            # Median
            val[1] = np.median(mod_list[prop][:stddev]) * cls.UNITS[prop]

            # Std dev
            val[2] = np.std(mod_list[prop][:stddev]) * cls.UNITS[prop]

            stats[f'{prop}_{component}'] = val

        # Store chi2
        stats['chi2'] = [mod_list['chi2'][0]*u.Unit(1), 0, 0]
        stats['redchi2'] = [redchi2, 0, 0]

        return cls(species, obs_spec=obs_spec, mod_spec=mod_spec, **stats)

    @classmethod
    def from_avg_file(cls, filename: Path, component: int = 1):
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
                    chi2 = float(val.strip())
                    if math.isnan(chi2):
                        print(f'WARNING: file {filename} has no data')
                    stats['chi2'] = [chi2*u.Unit(1), 0, 0]
                elif key.strip() == 'Chi2MinReduced':
                    stats['redchi2'] = [float(val.strip())*u.Unit(1), 0, 0]
                else:
                    pass
            else:
                key, *vals = field.split()
                if key in stats:
                    stats[key] = list(float(x)*cls.UNITS[key[:-2]]
                                      for x in vals)

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
                  fmt: str = 'spec_x{col:04d}_y{row:04d}.{ext}',
                  remove_tex_below: Optional[u.Quantity] = None) -> Dict:
        """Load Cassis results for points in a mask.

        Args:
          mask: FITS file with 1 for valid points.
          directory: path for the results.
          fmt: optional; format of the file name.
          remove_tex_below: optional; remove data with temperature below value.
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
            result = CassisResult.from_best_file(filename)
            if (remove_tex_below is not None and
                result.stats['tex'][0] < remove_tex_below):
                for suffix in ['.lam', '.log', '.txt', '.lis']:
                    aux = filename.with_suffix(suffix)
                    aux.unlink()
                continue
            vals.append(result)

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
    model = CassisResults.with_mask(args.maskfile[0], args.indir[0],
                                    remove_tex_below=args.remove_cold)
    for key in args.keys:
        key_comp = f'{key}_{args.component[0]}'
        if args.error:
            filename = args.indir[0] / f'{key_comp}_map_error.fits'
        else:
            filename = args.indir[0] / f'{key_comp}_map.fits'
        print(filename)
        model.generate_map(key_comp, filename=filename, error_map=args.error)

    # Chi2 does not have errors
    if args.chi:
        for key in ['chi2', 'redchi2']:
            filename = args.indir[0] / f'{key}_map.fits'
            model.generate_map(key, filename=filename)

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
    parser.add_argument('-e', '--error', action='store_true',
                        help='Save error maps only?')
    parser.add_argument('-x', '--chi', action='store_true',
                        help='Save chi2 and reduced chi2 map?')
    parser.add_argument('--remove_cold', action=actions.ReadQuantity, nargs=2,
                        help='Remove fits with temperature below this value')
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
