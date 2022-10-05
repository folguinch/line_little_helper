"""Store information of molecular lines.
"""
from typing import List, Optional, TypeVar, Sequence

import astropy.units as u

from .processing_tools import to_rest_freq, query_lines, zip_columns, combine_columns
from .common_types import QPair

# Addtitonal types
ConfigParser = TypeVar('ConfigParser')

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
    def from_config(cls, config:ConfigParser,
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

    def generate_name(
        self,
        include: Optional[Sequence[str]] = ('species', 'qns'),
    ) -> str:
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

    @property
    def qns_list(self):
        """List of stored molecule QNs."""
        return [tra.qns for tra in self.transitions]

    @classmethod
    def from_config(cls, name: str, config: ConfigParser):
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
                   qns: Optional[str] = None,
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
        transitions = _query_to_transition(name, freq_range, vlsr=vlsr,
                                           filter_out=filter_out, qns=qns,
                                           **kwargs)
        return cls(name.strip(), transitions)

    def reduce_qns(self):
        """Delete repeated QNs."""
        qns = self.qns_list
        qns_set = set(qns)
        if len(qns) != len(qns_set):
            new_tra = []
            for qn, tra in zip(qns, self.transitions):
                if qn in qns_set:
                    new_tra.append(tra)
                    qns_set.remove(qn)
            self.transitions = new_tra

class Molecules(list):
    """Store molecular information."""

    @classmethod
    def from_config(cls, configs: dict):
        molecules = cls()
        for molecule, config in configs.items():
            molecules.append(Molecule.from_config(molecule, config))
        return molecules

class NoTransitionError(Exception):
    """Exception for `Molecule` without transitions.

    Attributes:
      message: output message.
    """

    def __init__(self,
                 molecule: str,
                 qns: Optional[str] = None,
                 spectral_range: Optional[str] = None,
                 message: Optional[str] = None):
        # Message
        if qns is not None and spectral_range is None:
            if message is None:
                message = 'Transition not found'
            self.message = f'{molecule} ({qns}): {message}'
        elif qns is not None and spectral_range is not None:
            if message is None:
                message = 'Transition not found'
            self.message = (f'{molecule} ({qns} in {spectral_range[0].value}-'
                            f'{spectral_range[0].value} {spectral_range.unit})'
                            f': {message}')
        else:
            if message is None:
                message = 'No transition found'
            self.message = f'{molecule} -> {message}'

        super().__init__(self.message)

def _query_to_transition(name: str,
                         freq_range: QPair,
                         vlsr: Optional[u.Quantity] = None,
                         filter_out: Optional[list] = None,
                         qns: Optional[str] = None,
                         #interactive: bool = False,
                         #ntransitions: Optional[int] = None,
                         **kwargs) -> List[Transition]:
    """Perform a query, filter results and create transitions."""
    # Query
    table = query_lines(freq_range, chemical_name=name, **kwargs)
    table['Freq'] = combine_columns(table, ['Freq', 'MeasFreq'])

    # Filter QNs
    if qns is not None:
        ind = table['QNs'] == qns
        try:
            table = table[ind]
        except IndexError as exc:
            #print('Could not find QNs')
            #print('Available QNs:')
            #print(table['QNs'])
            #raise ValueError('No QNs detected') from exc
            raise NoTransitionError(name, qns=qns) from exc

    # Create transitions
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

    return transitions

def get_molecule(molecule: str,
                 cube: 'spectral_cube.SpectralCube',
                 qns: Optional[str] = None,
                 onlyj: bool = False,
                 line_lists: Sequence[str] = ['CDMS', 'JPL'],
                 vlsr: Optional[u.Quantity] = None,
                 log: Callable = print) -> u.Quantity
    # Frequency ranges
    spectral_axis = cube.spectral_axis
    obs_freq_range = spectral_axis[[0,-1]]
    log((f'Observed freq. range: {obs_freq_range[0].value} '
         f'{obs_freq_range[1].value} {obs_freq_range[1].unit}'))
    if vlsr is not None:
        equiv = u.doppler_radio(get_restfreq(cube))
        rest_freq_range = to_rest_freq(obs_freq_range, vlsr, equiv)
        log((f'Rest freq. range: {rest_freq_range[0].value} '
             f'{rest_freq_range[1].value} {rest_freq_range[1].unit}'))
    else:
        log('Could not determine rest frequency, using observed')
        rest_freq_range = obs_freq_range

    # Create molecule
    if onlyj:
        filter_out = ['F', 'K']
    else:
        filter_out = None
    mol = Molecule.from_query(f' {molecule} ', rest_freq_range,
                              vlsr=vlsr, filter_out=filter_out,
                              line_lists=line_lists, qns=qns)
    mol.reduce_qns()
    log(f'Number of transitions: {len(mol.transitions)}')

    return mol
