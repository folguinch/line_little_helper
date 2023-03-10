"""Objects for managing spectral data."""
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypeVar, Union, Callable
import io

from astropy.io import fits
from astropy.modeling import models, fitting
from radio_beam import Beam, Beams
from scipy import ndimage, signal
from spectral_cube import SpectralCube
from toolkit.astro_tools import cube_utils
from toolkit.logger import LoggedObject
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .processing_tools import observed_to_rest
from .molecule import Molecule

Coordinate = TypeVar('Coordinate')

def _value_from_header(text: str):
    """Separate value and unit from spectrum header."""
    value = text.strip().split('=')[1].split()
    if value[0] == 'none':
        return None
    else:
        return float(value[0]) * u.Unit(value[1], format='cds')

class Spectrum(LoggedObject):
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
                 beam: Optional[Union[Beam, Beams]] = None) -> None:
        """Initialize a spectrum object."""
        super().__init__(__name__)
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
        """Length of the spectrum array."""
        return len(self.intensity)

    @property
    def velocity_equiv(self):
        """Frequency-velocity equivalency for current rest frequency."""
        return u.doppler_radio(self.restfreq)

    @property
    def velocity_axis(self):
        """Obtain spectral axis in velocity units."""
        if self.spectral_axis.unit.is_equivalent(u.km / u.s):
            return self.spectral_axis
        else:
            return self.spectral_axis.to(u.km / u.s,
                                         equivalencies=self.velocity_equiv)

    @property
    def frequency_axis(self):
        """Obtain spectral axis in frequency units."""
        if self.spectral_axis.unit.is_equivalent(u.Hz):
            return self.spectral_axis
        else:
            equiv = u.doppler_radio(self.restfreq)
            return self.spectral_axis.to(u.MHz,
                                         equivalencies=self.velocity_equiv)

    @classmethod
    def from_cube(cls,
                  cube: SpectralCube,
                  coord: Coordinate,
                  spectral_axis_unit: u.Unit = u.GHz,
                  vlsr: Optional[u.Quantity] = None,
                  rms: Optional[u.Quantity] = None,
                  sampled_rms: bool = True,
                  radius: Optional[u.Quantity] = None,
                  area_pix: Optional[float] = None,
                  restframe: str = 'observed'):
        """Generate a Spectrum from a cube.

        Args:
          cubes: spectral cube.
          coord: coordinate where the spectra are extracted.
          spectral_axis_unit: optional; units of the spectral axis.
          vlsr: optional; LSR velocity.
          rms: optional; cube rms.
          sampled_rms: optional; calculate the rms from a sample of channels?
          radius: optional; source radius.
          area_pix: optional; source area in pixels.
          restframe: optional; spectral frame (observed or rest).
        """
        if restframe == 'rest' and vlsr is None:
            raise ValueError('Cannot change to rest frame: vlsr is None')
        spec = cube_utils.spectrum_at_position(
            cube,
            coord,
            spectral_axis_unit=spectral_axis_unit,
            vlsr=vlsr if restframe == 'rest' else None,
            radius=radius,
            area_pix=area_pix,
            log=cls.log.info,
        )
        try:
            beam = cube.beam
        except AttributeError:
            beam = cube.beams

        # rms
        if rms is None:
            rms = cube_utils.get_cube_rms(cube, use_header=True,
                                          sampled=sampled_rms)

        return cls(spec[0], spec[1].quantity,
                   restfreq=cube_utils.get_restfreq(cube),
                   restframe=restframe,
                   vlsr=vlsr,
                   rms=rms,
                   beam=beam)

    @classmethod
    def from_file(cls, filename: Path,
                  spectral_axis_unit: u.Unit = u.GHz) -> None:
        """Load spectrum from `dat` file."""
        # Read file
        text = filename.read_text()
        text = text.split('#')[1:]

        # Get header values
        restfreq = _value_from_header(text[0])
        rms = _value_from_header(text[1])
        vlsr = _value_from_header(text[2])
        beam = text[3].split('=')[1]

        # Frequency frame and units
        restframe = 'rest' if 'Rest' in text[4].split()[0] else 'observed'
        units = text[5].split()
        units = tuple(map(lambda x: u.Unit(x, format='cds'), units))

        # Get data
        data = io.StringIO(text[6])
        if beam == 'none':
            spectral_axis, intensity = np.loadtxt(data, usecols=(0,2),
                                                  unpack=True)
            beam = None
        else:
            cols = (0, 2, 3, 4, 5)
            spectral_axis, intensity, bmaj, bmin, pa = np.loadtxt(data,
                                                                  usecols=cols,
                                                                  unpack=True)

            # Beam
            beam = Beams(bmaj*units[3], bmin*units[4], pa*units[5])

        return cls((spectral_axis * units[0]).to(spectral_axis_unit),
                   intensity * units[2], restfreq=restfreq,
                   restframe=restframe, vlsr=vlsr, rms=rms, beam=beam)

    @classmethod
    def from_cassis(cls, filename: Path) -> None:
        """Load spectrum from CASSIS `lis` file."""
        # Read file
        text = filename.read_text()
        text = io.StringIO('\n'.join(text.split('\n')[3:]))

        # Data
        freq, spec = np.loadtxt(text, usecols=(0,4), unpack=True)
        freq = freq * u.MHz
        spec = spec * u.K

        return cls(freq, spec)

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

    def _as_dat(self) -> str:
        """Generate an string with the spectrum in tab-separated format."""
        # Header
        if self.restfreq is None:
            header = '#restfreq=none\n'
        else:
            header = f'#restfreq={self.restfreq.value} {self.restfreq.unit}\n'
        if self.rms is None:
            header += '#rms=none\n'
        else:
            header += f'#rms={self.rms.value} {self.rms.unit}\n'
        if self.vlsr is None:
            header += '#vlsr=none\n'
        else:
            header += f'#vlsr={self.vlsr.value} {self.vlsr.unit}\n'
        freq = 'RestFreq' if self._frame == 'rest' else 'ObsFreq'

        # Data and units
        data = (self.frequency_axis.value,
                self.velocity_axis.value,
                self.intensity.value)
        units = (f'#{self.frequency_axis.unit:cds}'
                 f' {self.velocity_axis.unit:cds}'
                 f' {self.intensity.unit:cds}')
        if self.beam is None:
            header += '#beam=none\n'
            header += f'#{freq} Vel Flux'
        else:
            header += '#beam=true\n'
            header += f'#{freq} Vel Flux bmaj bmin pa'
            try:
                if len(self.beam) == self.length:
                    data += (self.beam.major.value,
                             self.beam.minor.value,
                             self.beam.pa.value)
                else:
                    raise ValueError('Beam array with wrong size')
            except TypeError:
                data += (np.repeat(self.beam.major.value, self.length),
                         np.repeat(self.beam.minor.value, self.length),
                         np.repeat(self.beam.pa.value, self.length))
            units += (f' {self.beam.major.unit:cds}'
                      f' {self.beam.minor.unit:cds}'
                      f' {self.beam.pa.unit:cds}')
        fmt_tab = '%f\t' * len(data)
        fmt_tab = fmt_tab.strip()
        units += '#'
        lines = [fmt_tab % d for d in zip(*data)]

        return '\n'.join([header, units] + lines)

    def saveas(self, filename: Path, fmt='dat') -> None:
        """Save spectrum to disk.

        Args:
          filename: output path.
          fmt: optional; output format.
        """
        fmt_avail = {'cassis': self._as_cassis, 'dat': self._as_dat}

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

    def intensity_mask(self, nsigma: float = 5) -> npt.ArrayLike:
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

    def find_peaks(self, **kwargs) -> Tuple[npt.ArrayLike]:
        """Find the peaks in the spectra and width properties.

        Args:
          kwargs: additional arguments for `signal.peak_widths`.

        Return:
          A tuple with the peaks channel positions, width, level at width,
          lower and upper limit.
        """
        # Find peaks
        peaks, _ = signal.find_peaks(self.intensity)
        
        # Widths
        kwargs.setdefault('rel_height', 1)
        widths = signal.peak_widths(self.intensity, peaks, **kwargs)

        return (peaks,) + widths

    def has_lines(self, min_width: int = 5,
                  dilate: Optional[int] = None) -> list:
        """Find ranges with likely line emission.

        Args:
          min_width: optional; minimum number of channels to be considered a
            line.
          dilate: optional; number of channels to add around the lines.

        Returns:
          A list of slices.
        """
        # Generate mask
        mask = self.intensity_mask()

        # No lines
        if np.sum(mask) == 0:
            return []

        # First pass
        # Labels and objects
        labs, nlabs = ndimage.label(mask)
        objs = ndimage.find_objects(labs)
        # Filter slices
        for obj in objs:
            slc = obj[0]
            delta = slc.stop - slc.start
            if delta > min_width:
                continue
            else:
                mask[slc] = False

        # Erode mask
        if dilate is not None:
            mask = ndimage.binary_dilation(mask, iterations=dilate)

        # Final objects
        labs, nlabs = ndimage.label(mask)
        objs = ndimage.find_objects(labs)

        return objs

    def fit_line(self, spec_range: Optional[Sequence[u.Quantity]] = None,
                 slice_range: Optional[slice] = None) -> models.Gaussian1D:
        """Fit the spectrum with a Gaussian funtion.

        Args:
          spec_range: optional; upper and lower limits of the spectral axis.
          slice_range: optional; slice object of the data to fit.

        Returns:
          A Gaussian model with the fitted parameters.
        """
        # Select data
        if spec_range is not None:
            mask = self.range_mask(*spec_range)
            x = self.spectral_axis[mask]
            y = self.intensity[mask]
        elif slice_range is not None:
            x = self.spectral_axis[slice_range]
            y = self.intensity[slice_range]
        else:
            x = self.spectral_axis
            y = self.intensity

        # Initial conditions
        aux = x[y > np.max(y)/2]
        stddev = np.abs(aux[-1] - aux[0]) / 2
        model = models.Gaussian1D(amplitude=np.max(y), mean=x[len(x)//2],
                                  stddev=stddev)

        # Fitter
        fitter = fitting.LevMarLSQFitter()
        model_fit = fitter(model, x, y)

        return model_fit

    def fit_lines(self, min_width: int = 5,
                  dilate: Optional[int] = None,
                  slice_as_freq: bool = False,
                  ) -> Tuple[List[slice], List[models.Gaussian1D]]:
        """Fit all the potential lines.

        Args:
          min_width: optional; minimum number of channels to be considered a
            line.
          dilate: optional; number of channels to add around the lines.
          slice_as_frequency: optional; convert the index slices to frequency
            range?

        Returns:
          The slices where the lines are fit.
          The model results.
        """
        # Find slices with lines
        slices = self.has_lines(min_width=min_width, dilate=dilate)

        # Fit each slice
        results = []
        final_slices = []
        for slc in slices:
            results.append(self.fit_line(slice_range=slc))
            if slice_as_freq:
                final_slices = [(self.frequency_axis[slc.start],
                                 self.frequency_axis[slc.stop-1])]

        # Check slices
        if len(final_slices) == 0:
            final_slices = slices

        return final_slices, results

    def fit_transition(self, transition: 'Transition',
                       velocity_range: Optional[u.Quantity] = None,
                       channel_range: Optional[int] = None
                       ) -> models.Gaussian1D:
        """Fit emission around a reference molecule.

        The transition frequency is only used as a reference to fit the line,
        but it must be within the velocity or channel range.

        The `channel_range` corresponds to the number of channels of the window
        with the line channel in the center. The `velocity_range` corresponds
        to the velocity furthest from the central line.

        Args:
          transition: `Transition` with the line information.
          velocity_range: optional; velocity range around the line.
          channel_range: optional; number of channels of the window around line.
        """
        if velocity_range is not None:
            if transition.obsfreq is None:
                transition.set_obsfreq(self.vlsr)
            self.restfreq = transition.obsfreq
            vel_axis = self.velocity_axis
            ind = np.indices(vel_axis.shape)
            mask = (vel_axis >= -velocity_range) & (vel_axis <= velocity_range)
            slc = slice(min(ind[mask]), max(ind[mask]) + 1)
        elif channel_range is not None:
            axis = self.spectral_axis
            if self._frame == 'observed':
                if transition.obsfreq is None:
                    transition.set_obsfreq(self.vlsr)
                reffreq = transition.obsfreq
            else:
                reffreq = transition.restfreq
            ind = np.nanargmin(np.abs(self.frequency_axis - reffreq))
            slc = slice(ind - channel_range//2, ind + channel_range//2 + 1)
        else:
            raise ValueError('Could not determine spectral axis range')

        return self.fit_line(slice_range=slc)

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

        return self.frequency_axis[mask][ind]

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
             xlim: Optional[Sequence[u.Quantity]] = None,
             use_velocity: bool = False
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
        if xlim is None:
            xlim = self.extrema()
        if use_velocity:
            ax.plot(self.velocity_axis, self.intensity, 'b-')
            xunit = self.velocity_axis.unit
        else:
            ax.plot(self.frequency_axis, self.intensity, 'b-')
            xunit = self.frequency_axis.unit
        ax.set_xlim(xlim[0].to(xunit, equivalencies=self.velocity_equiv).value,
                    xlim[1].to(xunit, equivalencies=self.velocity_equiv).value)
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

    def plot_model(self, ax: 'Axis', result_fn: Callable):
        """
        """
        ax.plot(x, y)

class Spectra(list):
    """Class to store Spectrum objects."""

    @classmethod
    def from_cubes(cls,
                   cubes: Sequence[Union[SpectralCube, Path]],
                   coord: Coordinate,
                   spectral_axis_unit: u.Unit = u.GHz,
                   vlsr: Optional[u.Quantity] = None,
                   radius: Optional[u.Quantity] = None) -> List:
        """Generate an Spectra object from input cubes.

        Args:
          cubes: list of cubes or filenames.
          coord: coordinate where the spectra are extracted.
          spectral_axis_unit: optional; units of the spectral axis.
          vlsr: optional; LSR velocity.
          radius: optional; average pixels inside this radius.
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
                                      vlsr=vlsr,
                                      radius=radius)
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

    def find_peaks(self) -> List[Tuple[npt.ArrayLike]]:
        """Find peaks and their widths for each spectrum."""
        vals = []
        for spec in self:
            vals.append(spec.find_peaks())

        return vals

class CassisModelSpectra(Spectra):

    @classmethod
    def read(cls, filename: Path):
        """Load model spectra from CASSIS `.lis` file."""
        # Read file
        text = filename.read_text()
        text = io.StringIO('\n'.join(text.split('\n')[3:]))

        # Data
        freq, vel, spec = np.loadtxt(text, usecols=(0, 1, 4), unpack=True)

        # Split using velocity
        ind = np.where(vel[:-1] < vel[1:])[0] + 1
        ind = np.hstack(([0], ind, vel.size))
        specs = []
        for i, j in zip(ind[:-1], ind[1:]):
            spectral_axis = freq[i:j] * u.MHz
            intensity = spec[i:j] * u.K
            specs.append(Spectrum(spectral_axis, intensity))

        return cls(specs)

class IndexedSpectra(dict):
    """Class to store Spectra objects indexed by key."""

    def __repr__(self):
        strval = []
        for key, items in self.items():
            strval.append(f'Spectrum from: {key}')
            for item in items:
                strval.append(repr(item))
        return '\n'.join(strval)

    @classmethod
    def from_files(cls,
                   filenames: Sequence[Path],
                   coords: Optional[Sequence[Coordinate]] = None,
                   index: Union[str, Sequence] = 'filenames',
                   spectral_axis_unit: u.Unit = u.GHz,
                   vlsr: Optional[u.Quantity] = None,
                   radius: Optional[u.Quantity] = None) -> dict:
        """Store spectra in a dictionary indexed by `index`.

        If `filenames` are from previously saved spectra, all other keywords
        are ignored (except for `coords` when `index=coords`, and
        `spectral_axis_unit`).

        If `index` is a sequence of values, it length is trimmed to the length
        of cubenames or viceversa by `zip`.

        Args:
          filenames: list of file names (FITS or `Spectrum` generated `dat`).
          coords: positions or coordinates where the spectra are extracted.
          index: optional; index the dictionary by `filenames` or `coords` or
            list of keys.
          spectral_axis_unit: optional; units of the spectral axis.
          vlsr: optional; LSR velocity.
          radius: optional; average pixels inside this radius.
        """
        # Determine the extension of the data
        filetype = filenames[0].suffix.lower()
        if filetype not in ['.fits', '.dat']:
            raise TypeError(f'File type {filetype} not recognized')

        # Dictionary index
        vals = None
        if index == 'coords':
            # Keys
            keys = coords

            # Load spectra
            vals = []
            for coord in coords:
                if filetype == '.dat':
                    raise NotImplementedError('dat file loader not implemented')
                else:
                    vals.append(Spectra.from_cubes(filenames, coord,
                                                   spectral_axis_unit, vlsr,
                                                   radius=radius))
        elif index == 'filenames':
            keys = filenames
        else:
            keys = index

        # Load spectra for other cases
        if vals is None:
            vals = []
            for filename in filenames:
                if filetype == '.dat':
                    # Load spectrum
                    spec = Spectrum.from_file(
                        filename,
                        spectral_axis_unit=spectral_axis_unit,
                    )
                else:
                    aux = SpectralCube.read(filename)

                    # Iter over coordinates
                    specs = Spectra()
                    for coord in coords:
                        # Extract spectra
                        spec = Spectrum.from_cube(
                            aux,
                            coord,
                            spectral_axis_unit=spectral_axis_unit,
                            vlsr=vlsr,
                            radius=radius,
                        )
                        specs.append(spec)

                vals.append(specs)

        return cls(zip(keys, vals))

    @property
    def nspecs(self) -> List[int]:
        """The number of spectra per key."""
        nspec = []
        for key in self:
            nspec.append(len(self[key]))

        return nspec

    def generate_filename(self, key: str,
                          index: int = 0,
                          base_name: str = None,
                          directory: Path = Path('./')) -> Path:
        """Generate a standard file name to store indexed spectra.

        Args:
          key: the key of the spectra indexed.
          index: optional; spectrum within the given name.
          base_name: optional; a base file name without extension.
          directory: optional; directory of the final filename.

        Returns:
          A filename `Path`:

            - if the number of stored spectra in `key` is 1:
                ```{directory}/{base_name or key.name}.dat```
            - else:
                ```{directory}/{base_name or key.name}_spec{index}.dat```
        """
        # Parameters
        nspecs = len(self[key])
        name = Path(key)

        if nspecs == 1:
            return directory / f'{base_name or name.stem}.dat'
        else:
            return directory / f'{base_name or name.stem}_spec{index}.dat'

    def write_to(self, base_name: str = None, directory: Path = Path('./'),
                 fmt: str = 'cassis') -> None:
        """Write spectra to disk.

        Filename is generated from the key names.

        Args:
          base_name: optional; a base file name without extension.
          directory: optional; directory to save the files to.
          fmt: optional; format of the output files.
        """
        # Iterate over values
        for key, val in self.items():
            #aux = Path(key)
            for i, spec in enumerate(val):
                #filename = aux.name.replace(aux.suffix, f'_spec{i}.dat')
                #filename = directory / filename
                filename = self.generate_filename(key, index=i,
                                                  base_name=base_name,
                                                  directory=directory)
                spec.saveas(filename, fmt=fmt)

    def have_lines(self, min_width: int = 5, dilate: Optional[int] = None,
                   slice_as_frequency: bool = False):
        """Check if spectra have emission lines.

        Args:
          min_width: optional; minimum number of channels to be considered a
            line.
          dilate: optional; number of channels to add around the lines.
          slice_as_frequency: optional; convert the index slices to frequency
            range?

        Returns:
          A dictionary with the results for each spectrum.
        """
        results = {}
        for key, specs in self.items():
            result = []
            for spec in specs:
                result.append(
                    spec.fit_lines(min_width=min_width,
                                   dilate=dilate,
                                   slice_as_frequency=slice_as_frequency)
                )
            results[key] = result

        return results

#    def overplot(self, output: Optional['Path'] = None,
#                 molecule: Optional[Molecule] = None)

def generate_mask(cube: SpectralCube,
                  rms: Optional[u.Quantity] = None,
                  nsigma: int = 5,
                  flux_limit: Optional[u.Quantity] = None,
                  sampled_rms: bool = False,
                  savedir: Path = Path('./'),
                  maskname: Optional[Path] = None,
                  mask: Optional[np.array] = None,
                  log: Callable = print) -> Tuple[np.array, Union[Path, None]]:
    """Generate a mask and optionally its filename.

    Args:
      cube: the spectral cube.
      rms: optional; common rms value for all cubes.
      nsigma: optional; level over rms to filter data out.
      flux_limit: optional; flux limit to filter data out.
      sampled_rms: optional; calculate rms from a sample of channels?
      savedir: optional; saving directory. Defaults to cube directory.
      maskname: optional; mask file name.
      mask: optional; true where the spectra will be extracted.
      log: optional; logging function.

    Returns:
      The boolean mask array (`True` for valid points).
      The mask filename if `maskname` given, else `None`.
    """
    # Determine mask filename
    if maskname is not None and mask is None:
        savemask = savedir / maskname
        log(f'Mask filename: {savemask}')
        if savemask.exists():
            log('Loading mask')
            mask = fits.open(savemask)[0].data
            mask = mask.astype(bool)
    else:
        savemask = None

    # Generate and check mask
    if mask is None:
        # Flux limit
        if flux_limit is not None:
            low_lim = flux_limit
        elif rms is not None:
            low_lim = nsigma * rms
        else:
            rms = cube_utils.get_cube_rms(cube,
                                          use_header=True,
                                          sampled=sampled_rms)
            low_lim = nsigma * rms
        log(f'Flux limit: {low_lim.value} {low_lim.unit}')

        # Create flux limit mask from the first cube
        low_lim = low_lim.to(cube.unit).value
        mask = np.any(np.squeeze(cube.unmasked_data[:].value) > low_lim,
                      axis=0)
    else:
        if cube.shape[-2:] != mask.shape:
            raise ValueError((f'Mask shape {mask.shape}, inconsistent with'
                              f'cube shape {cube.shape[-2:]}'))
    log(f'Number of valid pixels in data: {np.sum(mask)}')

    return mask, savemask

def on_the_fly_spectra_loader(cubenames: Sequence[Path],
                              rms: Optional[u.Quantity] = None,
                              nsigma: int = 5,
                              flux_limit: Optional[u.Quantity] = None,
                              common_rms: Callable[[u.Quantity],
                                                   u.Quantity] = np.max,
                              sampled_rms: bool = False,
                              spectral_axis_unit: u.Unit = u.GHz,
                              vlsr: Optional[u.Quantity] = None,
                              savedir: Path = Path('./'),
                              maskname: Optional[Path] = None,
                              mask: Optional[np.array] = None,
                              restframe: str = 'observed',
                              fmt: str = 'cassis',
                              radius: Optional[u.Quantity] = None,
                              area_pix: Optional[float] = None,
                              log: Callable = print) -> np.array:
    """Loads and saves spectra without storing them in memory.

    It creates a mask from the first cube on the list, so it saves the spectra
    at the same positions from all the cubes.

    An average spectrum can be created if `radius` or `area_pix` are given.

    Args:
      cubenames: list of file names.
      rms: optional; common rms value for all cubes.
      nsigma: optional; level over rms to filter data out.
      flux_limit: optional; flux limit to filter data out.
      common_rms: optional; function to the determine the common rms.
      sampled_rms: optional; calculate rms from a sample of channels?
      spectral_axis_unit: optional; units of the spectral axis.
      vlsr: optional; LSR velocity.
      savedir: optional; saving directory. Defaults to cube directory.
      maskname: optional; mask file name.
      mask: optional; true where the spectra will be extracted.
      restframe: optional; wether to use `observed` or `rest` frame.
      fmt: optional; output format.
      radius: optional; source radius.
      area_pix: optional; source area in pixels.
      log: optional; logging function.
    """
    # Load 1st cube
    log(f'Loading 1st cube for initial mask: {cubenames[0].name}')
    aux = SpectralCube.read(cubenames[0])

    # Saving basic setup
    if savedir is None:
        savedir = Path(cubenames[0].parent)
    log(f'Saving directory: {savedir}')

    # Check rms: adding spectra imply that the rms is common between spectra
    # This may not be important in some cases
    cubes = {cubenames[0]: aux}
    if rms is None:
        rms_vals = np.array([]) * aux.unit
        for cube in cubenames:
            aux = cubes.get(cube)
            if aux is None:
                cubes[cube] = SpectralCube.read(cube)
                aux = cubes[cube]
            rms_val = cube_utils.get_cube_rms(aux, use_header=True,
                                              sampled=sampled_rms)
            rms_vals = np.append(rms_vals, [rms_val])
        rms = common_rms(rms_vals)
        log(f'Common rms: {rms}')

    # Create mask
    mask, savemask = generate_mask(aux, rms=rms, nsigma=nsigma,
                                   flux_limit=flux_limit, savedir=savedir,
                                   maskname=maskname, mask=mask, log=log)

    # Iterate over coordinates
    rows, cols = np.indices(mask.shape)
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
                rms=rms,
                restframe=restframe,
                radius=radius,
                area_pix=area_pix,
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

def cube_fitter(cube: SpectralCube,
                rms: Optional[u.Quantity] = None,
                nsigma: int = 5,
                flux_limit: Optional[u.Quantity] = None,
                spectral_axis_unit: u.Unit = u.GHz,
                vlsr: Optional[u.Quantity] = None,
                savedir: Path = Path('./'),
                maskname: Optional[Path] = None,
                mask: Optional[np.array] = None,
                restframe: str = 'observed',
                fmt: str = 'cassis',
                radius: Optional[u.Quantity] = None,
                area_pix: Optional[float] = None,
                log: Callable = print) -> np.array:
    """Fit all the 
    """
    # Check input
    if rms is not None:
        rms = rms.to(cube.unit)

    # Create mask
    mask, savemask = generate_mask(cube, rms=rms, nsigma=nsigma,
                                   flux_limit=flux_limit, savedir=savedir,
                                   maskname=maskname, mask=mask, log=log)

    # Iterate over coordinates
    rows, cols = np.indices(mask.shape)
    log('Extracting spectra')
    for row, col in zip(rows.flatten(), cols.flatten()):
        # Load spectrum
        if not mask[row, col]:
            continue
        spectrum = Spectrum.from_cube(cube, [col, row],
                                      spectral_axis_unit=spectral_axis_unit,
                                      vlsr=vlsr, rms=rms, restframe=restframe,
                                      radius=radius, area_pix=area_pix)

        # Save
        if np.any(np.isnan(spec.intensity)):
            #log(f'Removing spectrum x={col} y={row}')
            mask[row, col] = False
            continue
        #log(f'Saving spectrum x={col} y={row}')
        fname = f'spec_x{col:04d}_y{row:04d}.dat'
        spec.saveas(savedir / fname, fmt=fmt)

