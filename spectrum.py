from typing import Optional, Tuple

import astropy.units as u
import numpy as np

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
        return freq >= low and freq <= up

class Spectra(list):
    """Class to store Spectrum objects."""

    def get_spectrum(self, freq: u.Quantity):
        """Get the first spectrum where freq is in the spectral axis."""
        for sp in self:
            if sp.is_in(freq):
                return sp

        return None
