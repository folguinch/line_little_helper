"""Objects for managing spectral data."""
from typing import List, Optional, Sequence, Tuple, TypeVar

from toolkit.astro_tools import cube_utils
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from processing_tools import observed_to_rest
from lines import Molecule

Coordinate = TypeVar('Coordinate')
SpectralCube = TypeVar('SpectralCube')

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
    """
    restfreq = None
    rms = None

    def __init__(self, spectral_axis: u.Quantity, intensity: u.Quantity,
                 restfreq: Optional[u.Quantity] = None,
                 rms: Optional[u.Quantity] = None) -> None:
        """Initialize a spectrum object."""
        self.spectral_axis = spectral_axis
        self.intensity = intensity
        self.restfreq = restfreq
        self.rms = rms

    @classmethod
    def from_cube(cls,
                  cube: SpectralCube,
                  coord: Coordinate,
                  spectral_axis_unit: u.Unit = u.GHz,
                  vlsr: Optional[u.Quantity] = None):
        """Generate a Spectrum from a cube.

        Args:
          cubes: spectral cube.
          coord: coordinate where the spectra are extracted.
          spectral_axis_unit: optional; units of the spectral axis.
          vlsr: optional; LSR velocity.
        """
        spec = cube_utils.spectrum_at_position(
            cube,
            coord,
            spectral_axis_unit=spectral_axis_unit,
            vlsr=vlsr,
        )

        return cls(spec[0], spec[1].quantity,
                   restfreq=cube_utils.get_restfreq(cube),
                   rms=cube_utils.get_cube_rms(cube, 
                                               use_header=True,
                                               sampled=True))

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

    def plot(self, output: 'Path',
             ax: Optional['Axis'] = None,
             molecule: Optional[Molecule] = None,
             xlim: Optional[Sequence[u.Quantity]] = None) -> None:
        """Plot spectra and overplot line transitions.

        Args:
          output: figure path.
          ax: optional; axis object.
          molecule: optional; transitions to overplot.
          xlim: optional; x-axis limits.
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
        ax.set_xlabel(('Rest frequency'
                       f'({self.spectral_axis.unit:latex_inline})'))

        # Plot transitions
        ylocs = 1.1 * np.max(self.intensity.value) * np.linspace(0.9, 1.0, 6)
        for i, transition in enumerate(molecule.transitions):
            # Line
            restfreq = transition.restfreq.to(xunit).value
            ax.axvline(restfreq, color='c', linestyle='--')

            # Label
            xy = restfreq, ylocs[i%6]
            ax.annotate(transition.qns, xy, xytext=xy, verticalalignment='top',
                        horizontalalignment='right')

        # Save
        try:
            fig.savefig(output)
        except NameError:
            pass

class Spectra(list):
    """Class to store Spectrum objects."""

    @classmethod
    def from_cubes(cls,
                   cubes: Sequence[SpectralCube],
                   coord: Coordinate,
                   spectral_axis_unit: u.Unit = u.GHz,
                   vlsr: Optional[u.Quantity] = None) -> List:
        """Generate an Spectra object from input cubes.

        Args:
          cubes: list of file names.
          coord: coordinate where the spectra are extracted.
          spectral_axis_unit: optional; units of the spectral axis.
          vlsr: optional; LSR velocity.
        """
        specs = []
        for cube in cubes:
            # Observed frequencies shifted during subtraction
            spec = Spectrum.from_cube(cube, coord,
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
