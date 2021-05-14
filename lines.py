"""Store information of molecular lines.
"""
from typing import List, Optional, TypeVar

import astropy.units as u

from processing_tools import to_rest_freq, query_lines, zip_columns, combine_columns
from common_types import QPair

# Addtitonal types
Config = TypeVar('ConfigParser')

class Transition:
    """Class to store line information.

    Attributes:
      species: chemical formula.
      qns: quantum numbers.
      restfreq: rest frequency.
      obsfreq: observed frequency.
      eup: upper level energy.
      logaij: log of the Aij coeficient.
    """

    def __init__(self, species: str, qns: str, restfreq: u.Quantity,
                 obsfreq: Optional[u.Quantity] = None,
                 vlsr: Optional[u.Quantity] = None,
                 eup: Optional[u.Quantity] = None,
                 logaij: Optional[u.Quantity] = None) -> None:
        """Initiate a transition object.

        Args:
          species: chemical formula.
          qns: quantum numbers.
          restfreq: transition rest frequency.
          obsfreq: optional; observed frequency.
          vlsr: optional; used to determine `obsfreq` if not given.
          eup: optional; upper level energy.
          logaij: optional; log of the Aij coeficient.
        """
        self.species = species
        self.qns = qns
        self.restfreq = restfreq.to(u.GHz)
        self.obsfreq = obsfreq
        if self.obsfreq is None and vlsr is not None:
            equiv = u.doppler_radio(self.restfreq)
            self.obsfreq = to_rest_freq(self.restfreq, -vlsr, equiv)
        try:
            self.obsfreq = self.obsfreq.to(u.GHz)
        except AttributeError:
            pass
        self.eup = eup
        self.logaij = logaij

    def __str__(self):
        lines = [f'Species: {self.species}', f'QNs: {self.qns}']
        lines.append(f'Rest Freq = {self.restfreq.value} {self.restfreq.unit}')
        if self.obsfreq is not None:
            lines.append(f'Obs Freq = {self.obsfreq.value} {self.obsfreq.unit}')
        if self.eup is not None:
            lines.append(f'Eup = {self.eup.value} {self.eup.unit}')
        if self.logaij:
            lines.append(f'log(Aij) = {self.logaij}')
        return '\n'.join(lines)

    @classmethod
    def from_config(cls, config:Config,
                    vlsr: Optional[u.Quantity] = None):
        """Create a new transition from config parser proxy.

        Args:
          config: `configparseradv` proxy.
          vlsr: optional; LSR velocity to obtain observed frequency.
        """
        species = config.get('species', fallback=None)
        qns = config.get('qns', fallback=None)
        restfreq = config.getquantity('restfreq', fallback=None)
        obsfreq = config.getquantity('obsfreq', fallback=None)
        if obsfreq is None and vlsr is None:
            vlsr = config.getquantity('vlsr', fallback=None)
        eup = config.getquantity('eup', fallback=None)
        logaij = config.getfloat('logaij', fallback=None)

        return cls(species, qns, restfreq, obsfreq=obsfreq, eup=eup, vlsr=vlsr,
                   logaij=logaij)

    def generate_name(self,
                      include: Optional[List[str]] = ['species', 'qns']) -> str:
        """Generate a name string from requested keys.

        The default is to use the species and QNs for the name. It replaces any
        brackets and comas with underscores.

        Args:
          include: optional; which values include in the name.
        """
        name = []
        if 'species' in include:
            name.append(f'{self.species}'.replace('=', ''))
        if 'qns' in include:
            qns = f'{self.qns}'
            qns = qns.replace('-', '_').replace('(', '').replace(')', '')
            qns = qns.replace(',', '_').replace('=','_').replace('/', '_')
            name.append(qns)
        return '_'.join(name)

class Molecule:
    """Class for storing molecule information.

    Attributes:
      name: molecule chemical name.
      transitions: list of transitions of interest.
    """

    def __init__(self, name: str, transitions: List[Transition]) -> None:
        """Store the molecule information.

        Args:
          name: molecule name identifier.
          transitions: transitions to store.
        """
        self.name = name
        self.transitions = transitions

    def __str__(self):
        lines = [f'Molecule name: {self.name}']
        joint = '\n' + '-' * 40 + '\n'
        for transition in self.transitions:
            lines.append(str(transition))
        return joint.join(lines)

    @classmethod
    def from_config(cls, name: str, config: Config):
        """Create a new molecule instance from a config parser.

        The section titles are ignored at the moment.

        Args:
          name: molecule name.
          config: `configparseradv` object.
        """
        transitions = []
        for section in config.sections():
            transitions.append(Transition.from_config(config[section]))
        return cls(name, transitions)

    @classmethod
    def from_query(cls, name: str, freq_range: QPair,
                   vlsr: Optional[u.Quantity] = None,
                   filter_out: Optional[list] = None,
                   **kwargs):
        """Obtain information about molecule from splat.

        Args:
          name: species name following
            [astroquery](https://astroquery.readthedocs.io/en/v0.1-0/splatalogue.html).
          freq_range: frequency range to look for transitions.
          vlsr: optional; LSR velocity to obtain observed frequency.
          filter_out: optional; filter out transitions with given QNs.
          kwargs: optional additional query constraints.
        """
        table = query_lines(freq_range, chemical_name=name, **kwargs)
        table['Freq'] = combine_columns(table, ['Freq', 'MeasFreq'])
        transitions = []
        cols = ['Species', 'QNs', 'Freq', 'Eu', 'log10Aij']
        for data in zip_columns(table, cols):
            # Filter transitions
            if filter_out is not None:
                aux = False
                for fil in filter_out:
                    if fil in data[1]:
                        aux = True
                        break
                if aux:
                    continue

            # Do not use unresolved QNs
            if str(data[1]) == '--':
                continue

            transitions.append(
                Transition(data[0], data[1], data[2],
                           vlsr=vlsr, eup=data[3], logaij=data[4]))

        return cls(name.strip(), transitions)

class Molecules(list):
    """Store molecular information."""

    @classmethod
    def from_config(cls, configs: dict):
        molecules = cls()
        for molecule, config in configs.items():
            molecules.append(Molecule.from_config(molecule, config))
        return molecules
