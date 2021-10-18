"""Objects for managing spectral data."""
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypeVar, Union, Callable

from astropy.io import fits
from radio_beam import Beam
from spectral_cube import SpectralCube
from toolkit.astro_tools import cube_utils
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from processing_tools import observed_to_rest
from lines import Molecule

Coordinate = TypeVar('Coordinate')

class Spectrum:
    """Class for storing a single spectrum.

    A spectrum has a spectral axis and an intensity axis, and optionaly a rest
    frequency and an rms (noise) value. The rest frequency is used to convert
    the spectral axis units.

    Attributes:
      spectral_axis: frequency or velocity axis.
      intensity: intensity axis.
      restfreq: rest frequency.
      rms: rms value.
      beam: beam size(s).
      vlsr: LSR velocity.
      _frame: spectral axis reference frame.
    """
    restfreq = None
    rms = None
    beam = None

    def __init__(self, spectral_axis: u.Quantity, intensity: u.Quantity,
                 restfreq: Optional[u.Quantity] = None,
                 restframe: str = 'observed',
                 vlsr: Optional[u.Quantity] = None,
                 rms: Optional[u.Quantity] = None,
                 beam: Optional[Beam] = None) -> None:
        """Initialize a spectrum object."""
        self.spectral_axis = spectral_axis
        self.intensity = intensity
        self.restfreq = restfreq
        self.rms = rms
        self.beam = beam
        self.vlsr = vlsr
        self._frame = restframe.lower()

    def __repr__(self):
        restfreq = self.restfreq.to(self.spectral_axis.unit)
        lines = [f'Spectrum length: {self.length}',
                 f'Rest freq.: {restfreq.value:.3f} {restfreq.unit}',
                 f'Spectrum rms: {self.rms.value:.3f} {self.rms.unit}']

        return '\n'.join(lines)

    def __add__(self, spec):
        if self._frame != spec._frame:
            # This should be adapted in the future
            raise ValueError('Cannot append spectra in different frames')
        spectral_axis = np.append(self.spectral_axis, spec.spectral_axis)
        ind = np.argsort(spectral_axis)
        intensity = np.append(self.intensity, spec.intensity)
        if self.beam is not None:
            beam = np.append(self.beam, spec.beam)
            beam = beam.take(ind)
        else:
            beam = None
        return Spectrum(spectral_axis=spectral_axis[ind],
                        intensity=intensity[ind],
                        restfreq=self.restfreq,
                        restframe=self._frame,
                        vlsr=self.vlsr,
                        rms=self.rms,
                        beam=beam)

    @property
    def length(self):
        return len(self.intensity)

    @classmethod
    def from_cube(cls,
                  cube: SpectralCube,
                  coord: Coordinate,
                  spectral_axis_unit: u.Unit = u.GHz,
                  vlsr: Optional[u.Quantity] = None,
                  rms: Optional[u.Quantity] = None,
                  restframe: str = 'observed'):
        """Generate a Spectrum from a cube.

        Args:
          cubes: spectral cube.
          coord: coordinate where the spectra are extracted.
          spectral_axis_unit: optional; units of the spectral axis.
          vlsr: optional; LSR velocity.
          rms: optional; cube rms.
          rest_frame: optional; spectral frame (observed or rest).
        """
        if restframe == 'rest' and vlsr is None:
            raise ValueError('Cannot change to rest frame: vlsr is None')
        spec = cube_utils.spectrum_at_position(
            cube,
            coord,
            spectral_axis_unit=spectral_axis_unit,
            vlsr=vlsr if restframe == 'rest' else None,
        )
        try:
            beam = cube.beam
        except AttributeError:
            beam = cube.beams

        # rms
        if rms is None:
            rms = cube_utils.get_cube_rms(cube, use_header=True, sampled=True)

        return cls(spec[0], spec[1].quantity,
                   restfreq=cube_utils.get_restfreq(cube),
                   restframe=restframe,
                   vlsr=vlsr,
                   rms=rms,
                   beam=beam)

    @property
    def velocity_axis(self):
        """Obtain spectral axis en velocity units."""
        if self.spectral_axis.unit.is_equivalent(u.km / u.s):
            return self.spectral_axis
        else:
            equiv = u.doppler_radio(self.restfreq)
            return self.spectral_axis.to(u.km / u.s, equivalencies=equiv)

    def to_temperature(self,
                       bmaj: Optional[u.Quantity] = None,
                       bmin: Optional[u.Quantity] = None) -> u.Quantity:
        """Obtain the intensity axis in temperature units.

        Args:
          bmaj: optional; beam major axis.
          bmin: optional; beam minor axis.

        Returns:
          The intensity axis in temperature units.
        """
        # Beam
        if bmaj is not None and bmin is not None:
            beam = Beam(bmaj, bmin)
        elif self.beam is not None:
            beam = self.beam
        else:
            raise ValueError('Cannot convert to temperature units')

        # Convert
        equiv = u.brightness_temperature(self.spectral_axis, beam_area=beam)
        return self.intensity.to(u.K, equivalencies=equiv)

    def _as_cassis(self) -> str:
        """Generate an string with the spectrum in CASSIS format."""
        fmt_str = ('%10s\t'*7).strip()
        lines = [f'// number of line : {self.length - 1}']
        if self.vlsr is not None and self._frame == 'observed':
            lines.append(f'// vlsr : {-self.vlsr.value:.1f}')
        else:
            lines.append('// vlsr : 0.0')
        lines.append(fmt_str % ('FreqLsb', 'VeloLsb', 'FreqUsb', 'VeloUsb',
                                'Intensity', 'DeltaF', 'DeltaV'))

        fmt_tab = ('%10.3f\t' + '%10.4f\t'*6).strip()
        delta_nu = np.abs(np.diff(self.spectral_axis.to(u.MHz).value))
        delta_vel = np.abs(np.diff(self.velocity_axis.to(u.m/u.s).value))
        zeros = np.zeros(delta_nu.shape)
        data = (self.spectral_axis.to(u.MHz).value,
                self.velocity_axis.to(u.m/u.s).value,
                zeros,
                zeros,
                self.to_temperature().value,
                delta_nu,
                delta_vel)
        lines += [fmt_tab % d for d in zip(*data)]

        return '\n'.join(lines)

    def saveas(self, filename: Path, fmt='cassis') -> None:
        """Save spectrum to disk.

        Args:
          filename: output path.
          fmt: optional; output format.
        """
        fmt_avail = {'cassis': self._as_cassis}

        # Check format
        if fmt not in fmt_avail:
            raise NotImplementedError('Output format not available')

        # Write data
        filename.write_text(fmt_avail[fmt]())

    def range_mask(self,
                   low: Optional[u.Quantity] = None,
                   up: Optional[u.Quantity] = None) -> np.array:
        """Creates a mask from spectral axis limits."""
        mask = np.ones(self.spectral_axis.shape, dtype=bool)
        if low is not None:
            mask = self.spectral_axis >= low
        if up is not None:
            mask = mask & (self.spectral_axis <= up)

        return mask

    def intensity_mask(self, nsigma: float = 5) -> np.array:
        """Create a mask based on the intensity over the number of rms."""
        if self.rms is not None:
            mask = self.intensity >= nsigma*self.rms
        else:
            mask = np.ones(self.intensity.shape, dtype=bool)

        return mask

    def combined_mask(self,
                      low: Optional[u.Quantity] = None,
                      up: Optional[u.Quantity] = None,
                      nsigma: float = 5) -> np.array:
        """Combine range and intensity masks."""
        mask = self.range_mask(low=low, up=up)
        mask = mask & self.intensity_mask(nsigma=nsigma)

        return mask

    def peak_frequency(self,
                       low: Optional[u.Quantity] = None,
                       up: Optional[u.Quantity] = None) -> u.Quantity:
        """Return the spectral axis value at the intensity peak.

        Regions of the spectral axis where to find the peak can be limited with
        the low and up parameters.

        Args:
          low: optional; lower spectral axis limit.
          upper: optional; upper spectral axis limit.
        Returns:
          The spectral axis value where intensity is max.
        """
        mask = self.range_mask(low=low, up=up)
        ind = np.nanargmax(self.intensity[mask])

        return self.spectral_axis[mask][ind]

    def centroid(self,
                 low: Optional[u.Quantity] = None,
                 up: Optional[u.Quantity] = None,
                 nsigma: float = 5) -> u.Quantity:
        """Determine the spectral axis value center of mass.

        Args:
          low: optional; lower spectral axis limit.
          upper: optional; upper spectral axis limit.
          nsigma: optional; lower limit for the emission.
        Returns:
          The spectral axis value weighted by intensity.
        """
        mask = self.combined_mask(low=low, up=up, nsigma=nsigma)
        cm = np.sum(self.spectral_axis[mask] * self.intensity[mask])
        cm = cm / np.sum(self.intensity[mask])

        return cm.to(self.spectral_axis.unit)

    def extrema(self) -> Tuple[u.Quantity]:
        """Returns the extremes of the spectral axis."""
        return np.min(self.spectral_axis), np.max(self.spectral_axis)

    def is_in(self, freq) -> bool:
        """Is the input frequency in the spectral axis range?"""
        low, up = self.extrema()
        return low <= freq <= up

    def filter(self, molecule: Molecule) -> Molecule:
        """Create a new `Molecule` with transitions in the spectra."""
        transitions = []
        for transition in molecule.transitions:
            if self.is_in(transition.restfreq):
                transitions.append(transition)

        return Molecule(molecule.name, transitions)

    def plot(self, output: Optional[Path] = None,
             ax: Optional['Axis'] = None,
             molecule: Optional[Molecule] = None,
             xlim: Optional[Sequence[u.Quantity]] = None
             ) -> Tuple['Axis','Figure']:
        """Plot spectra and overplot line transitions.

        Args:
          output: figure path.
          ax: optional; axis object.
          molecule: optional; transitions to overplot.
          xlim: optional; x-axis limits.
        Returns:
          A tuple with the figure and axis objects.
        """
        # Figure
        if ax is None:
            plt.close()
            fig = plt.figure(figsize=(15, 6))
            ax = fig.add_subplot(111)

        # Plot
        xunit = self.spectral_axis.unit
        if xlim is None:
            xlim = self.extrema()
        ax.plot(self.spectral_axis, self.intensity, 'b-')
        ax.set_xlim(xlim[0].to(xunit).value, xlim[1].to(xunit).value)
        ax.set_ylim(np.min(self.intensity.value),
                    1.1 * np.max(self.intensity.value))
        ax.set_ylabel(f'Intensity ({self.intensity.unit:latex_inline})')
        ax.set_xlabel((f'{self._frame.capitalize()} frequency'
                       f'({xunit:latex_inline})'))

        # Plot transitions
        if molecule is not None:
            self.plot_transitions(ax, molecule)

        # Save
        try:
            if output:
                fig.savefig(output)
            return fig, ax
        except NameError:
            return None, ax

        return fig, ax

    def plot_transitions(self, ax: 'Axis', molecule: Molecule) -> None:
        """Plot transitions from molecule.

        Args:
          ax: matplotlib axis.
          molecule: molecule with transitions.
        """
        ylocs = 1.1 * np.max(self.intensity.value) * np.linspace(0.9, 1.0, 6)
        for i, transition in enumerate(molecule.transitions):
            # Line
            if self._frame == 'rest':
                freq = transition.restfreq.to(self.spectral_axis.unit).value
            else:
                freq = transition.obsfreq.to(self.spectral_axis.unit).value
            ax.axvline(freq, color='c', linestyle='--')

            # Label
            xy = freq, ylocs[i%6]
            ax.annotate(transition.qns, xy, xytext=xy, verticalalignment='top',
                        horizontalalignment='right')

class Spectra(list):
    """Class to store Spectrum objects."""

    @classmethod
    def from_cubes(cls,
                   cubes: Sequence[Union[SpectralCube, Path]],
                   coord: Coordinate,
                   spectral_axis_unit: u.Unit = u.GHz,
                   vlsr: Optional[u.Quantity] = None) -> List:
        """Generate an Spectra object from input cubes.

        Args:
          cubes: list of cubes or filenames.
          coord: coordinate where the spectra are extracted.
          spectral_axis_unit: optional; units of the spectral axis.
          vlsr: optional; LSR velocity.
        """
        specs = []
        for cube in cubes:
            if not hasattr(cube, 'spectral_axis'):
                aux = SpectralCube.read(cube)
            else:
                aux = cube

            # Observed frequencies shifted during subtraction
            spec = Spectrum.from_cube(aux, coord,
                                      spectral_axis_unit=spectral_axis_unit,
                                      vlsr=vlsr)
            specs.append(spec)

        return cls(specs)

    @classmethod
    def from_arrays(
        cls,
        arrays: List,
        restfreq: u.Quantity,
        equivalencies: dict,
        vlsr: Optional[u.Quantity] = None,
        rms: Optional[List[u.Quantity]] = None,
        freq_names: Sequence[str] = ('nu', 'freq', 'frequency',
                                     'v', 'vel', 'velocity'),
        flux_names: Sequence[str] = ('F', 'f', 'Fnu', 'fnu',
                                     'intensity', 'T', 'Tb'),
    ):
        """Creates spectra object from a list structured array.

        The values in `arrays` are pairs consisting of an structured `np.array`
        and a dictionary with the units for each column.
        """
        specs = []
        for spw, (data, units) in enumerate(arrays):
            # Spectral axis
            freq_name = list(filter(lambda x, un=units: x in un, freq_names))[0]
            xaxis = data[freq_name] * units[freq_name]

            # Intensity axis
            int_name = list(filter(lambda x, un=units: x in un, flux_names))[0]
            spec = data[int_name] * units[int_name]

            # Noise
            if rms is not None:
                noise = rms[spw]
            else:
                noise = None

            # Shift spectral axis
            if 'all' in equivalencies:
                equivalency = equivalencies
            else:
                equivalency = {'all': equivalencies[spw]}
            if xaxis.unit.is_equivalent(u.Hz) and vlsr is not None:
                xaxis = observed_to_rest(xaxis, vlsr, equivalency)
                spec = Spectrum(xaxis, spec, restfreq=restfreq, rms=noise)
            elif xaxis.unit.is_equivalent(u.km/u.s):
                vels = xaxis - vlsr
                xaxis = vels.to(u.GHz, equivalencies=equivalency['all'])
                spec = Spectrum(xaxis, spec, restfreq=restfreq, rms=noise)
            else:
                spec = Spectrum(xaxis, spec, restfreq=restfreq, rms=noise)

            specs.append(spec)

        return cls(specs)

    def get_spectrum(self, freq: u.Quantity):
        """Get the first spectrum where freq is in the spectral axis."""
        for sp in self:
            if sp.is_in(freq):
                return sp

        return None

class IndexedSpectra(dict):
    """Class to store Spectra objects indexed by key."""

    def __repr__(self):
        strval = []
        for key, items in self.items():
            strval.append(f'Cube: {key}')
            for item in items:
                strval.append(repr(item))
        return '\n'.join(strval)

    @classmethod
    def from_files(cls,
                   cubenames: Sequence[Path],
                   coords: Sequence[Coordinate],
                   index: Union[str, Sequence] = 'cubenames',
                   spectral_axis_unit: u.Unit = u.GHz,
                   vlsr: Optional[u.Quantity] = None) -> dict:
        """Store spectra in a dictionary indexed by `index`.

        If `index` is a sequence of values, it length is trimmed to the length
        of cubenames or viceversa by `zip`.

        Args:
          cubenames: list of file names.
          coords: positions or coordinates where the spectra are extracted.
          index: optional; index the dictionary by `cubenames` or `coords` or
            list of keys.
          spectral_axis_unit: optional; units of the spectral axis.
          vlsr: optional; LSR velocity.
        """
        # Dictionary index
        vals = None
        if index == 'coords':
            keys = coords

            # Load spectra
            vals = []
            for coord in coords:
                vals.append(Spectra.from_cubes(cubenames, coord,
                                               spectral_axis_unit, vlsr))
        elif index == 'cubenames':
            keys = cubenames
        else:
            keys = index

        # Load spectra for other cases
        if vals is None:
            vals = []
            for cube in cubenames:
                aux = SpectralCube.read(cube)

                # Iter over coordinates
                specs = Spectra()
                for coord in coords:
                    # Observed frequencies shifted during subtraction
                    spec = Spectrum.from_cube(
                        aux,
                        coord,
                        spectral_axis_unit=spectral_axis_unit,
                        vlsr=vlsr,
                    )
                    specs.append(spec)

                vals.append(specs)

        return cls(zip(keys, vals))

    def write_to(self, directory: Path = Path('./'), fmt: str = 'cassis'):
        """Write spectra to disk.

        Filename is generated from the key names.

        Args:
          directory: optional; directory to save the files to.
          fmt: optional; format of the output files.
        """
        # Iterate over values
        for key, val in self.items():
            aux = Path(key)
            for i, spec in enumerate(val):
                filename = aux.name.replace(aux.suffix, f'_spec{i}.dat')
                filename = directory / filename
                spec.saveas(filename, fmt=fmt)

#    def overplot(self, output: Optional['Path'] = None,
#                 molecule: Optional[Molecule] = None)

def on_the_fly_spectra_loader(cubenames: Sequence[Path],
                              rms: Optional[u.Quantity] = None,
                              nsigma: int = 5,
                              flux_limit: Optional[u.Quantity] = None,
                              spectral_axis_unit: u.Unit = u.GHz,
                              vlsr: Optional[u.Quantity] = None,
                              savedir: Optional[Path] = None,
                              maskname: Optional[Path] = None,
                              mask: Optional[np.array] = None,
                              restframe: Optional[str] = 'observed',
                              fmt: str = 'cassis',
                              log: Callable = print) -> np.array:
    """Loads and saves spectra without storing them in memory.

    It creates a mask from the first cube on the list, so it saves the spectra
    at the same positions from all the cubes.

    Args:
      cubenames: list of file names.
      rms: optional; common rms value for all cubes.
      nsigma: optional; level over rms to filter data out.
      flux_limit: optional; flux limit to filter data out.
      spectral_axis_unit: optional; units of the spectral axis.
      vlsr: optional; LSR velocity.
      savedir: optional; saving directory. Defaults to cube directory.
      maskname: optional; mask file name.
      mask: optional; true where the spectra will be extracted.
      restframe: optional; wether to use `observed` or `rest` frame.
      fmt: optional; output format.
      log: optional; logging function.
    """
    # Load 1st cube
    log(f'Loading 1st cube for initial mask: {cubenames[0].name}')
    aux = SpectralCube.read(cubenames[0])

    # Saving basic setup
    if savedir is None:
        savedir = Path(cubenames[0].parent)
    filename = cubenames[0].stem
    log(f'Saving directory: {savedir}')

    # Create mask
    savemask = None
    if maskname is not None and mask is None:
        savemask = savedir / maskname
        log(f'Mask filename: {savemask}')
        if savemask.exists():
            log('Loading mask')
            mask = fits.open(savemask)[0].data
            mask = mask.astype(bool)
    if mask is None:
        # Flux limit
        if flux_limit is not None:
            low_lim = flux_limit
        elif rms is not None:
            low_lim = nsigma * rms
        else:
            rms = cube_utils.get_cube_rms(aux,
                                          use_header=True,
                                          sampled=True)
            low_lim = nsigma * rms
        log(f'Flux limit: {low_lim.value} {low_lim.unit}')

        # Create flux limit mask from the first cube
        low_lim = low_lim.to(aux.unit).value
        mask = np.any(np.squeeze(aux.unmasked_data[:].value) > low_lim,
                      axis=0)
    else:
        if aux.shape[-2:] != mask.shape:
            raise ValueError((f'mask shape (mask.shape), inconsistent with'
                              f'cube shape (aux.shape)'))
    log(f'Initial number of points: {np.sum(mask)}')

    # Iterate over coordinates
    rows, cols = np.indices(mask.shape)
    cubes = {cubenames[0]: aux}
    log('Extracting spectra')
    for row, col in zip(rows.flatten(), cols.flatten()):
        # Load spectrum
        if not mask[row, col]:
            continue
        spec = None
        for cube in cubenames:
            # Open cube
            aux = cubes.get(cube)
            if aux is None:
                cubes[cube] = SpectralCube.read(cube)
                aux = cubes[cube]
            aux2 = Spectrum.from_cube(
                aux,
                [col, row],
                spectral_axis_unit=spectral_axis_unit,
                vlsr=vlsr,
                rms=rms.to(aux.unit),
                restframe=restframe,
            )
            if spec is None:
                spec = aux2
            else:
                spec = spec + aux2

        # Save
        if np.any(np.isnan(spec.intensity)):
            #log(f'Removing spectrum x={col} y={row}')
            mask[row, col] = False
            continue
        #log(f'Saving spectrum x={col} y={row}')
        fname = f'spec_x{col:04d}_y{row:04d}.dat'
        spec.saveas(savedir / fname, fmt=fmt)

    # Save mask
    log(f'Final number of points: {np.sum(mask)}')
    if savemask is not None:
        log('Saving mask')
        header = aux.wcs.sub(['longitude', 'latitude']).to_header()
        hdu = fits.PrimaryHDU(mask.astype(int), header=header)
        hdu.update_header()
        hdu.writeto(savemask, overwrite=True)
