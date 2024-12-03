"""Store information of molecular lines.
"""
from typing import Callable, List, Optional, TypeVar, Sequence, Dict
from dataclasses import dataclass, InitVar
import json

import astropy.units as u
from astropy.io.ascii.core import InconsistentTableError
from toolkit.astro_tools import cube_utils

from .processing_tools import to_rest_freq, query_lines, zip_columns, combine_columns
from .common_types import QPair
from .utils import normalize_qns

# Addtitonal types
ConfigParser = TypeVar('ConfigParser')

@dataclass
class Transition:
    """Molecular line trasition information."""
    species: str
    """Species name."""
    qns: str
    """Quantum numbers."""
    restfreq: u.Quantity[u.GHz]
    """Rest frequency."""
    obsfreq: Optional[u.Quantity[u.GHz]] = None
    """Observed frequency."""
    eup: Optional[u.Quantity[u.K]] = None
    """Upper level energy."""
    logaij: Optional[float] = None
    """Logarithm of the Aij coeficient."""
    vlsr: InitVar[Optional[u.Quantity[u.km/u.s]]] = None
    """LSR velocity to determine `obsfreq` if not given."""

    def __post_init__(self, vlsr):
        # Set default restfreq unit to GHz
        self.restfreq = self.restfreq.to(u.GHz)

        # Calculate obsfreq
        if self.obsfreq is None and vlsr is not None:
            self.set_obsfreq(vlsr)
        if self.obsfreq is not None:
            self.obsfreq = self.obsfreq.to(self.restfreq.unit)

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
                    vlsr: Optional[u.Quantity[u.km/u.s]] = None):
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

    @classmethod
    def from_json(cls,
                  filename: Optional['pathlib.Path'] = None,
                  info: Optional[Dict] = None,
                  vlsr: Optional[u.Quantity[u.km/u.s]] = None):
        """Create a new transition from JSON file or JSON dictionary.

        Args:
          filename: Optional; JSON file name.
          info: Optional; Dictionary from a JSON loaded transition.
        """
        if filename is not None:
            info = json.loads(filename.read_text())
        species = info['species']
        qns = info['qns']
        restfreq = info['restfreq'] * u.Unit(info['restfreq_unit'])
        obsfreq = info['obsfreq']
        if obsfreq is not None:
            obsfreq = obsfreq * u.Unit(info['obsfreq_unit'])
        eup = info['eup']
        if eup is not None:
            eup = eup * u.Unit(info['eup_unit'])
        logaij = info['logaij']

        return cls(species, qns, restfreq, obsfreq=obsfreq, eup=eup,
                   logaij=logaij, vlsr=vlsr)

    def to_json_dict(self, unset_obsfreq: bool = True) -> Dict:
        """Convert to a JSON compatible dictionary."""
        info = {'species': self.species,
                'qns': self.qns,
                'restfreq': self.restfreq.value,
                'restfreq_unit': f'{self.restfreq.unit}',
                'logaij': self.logaij}
        if self.obsfreq is not None and not unset_obsfreq:
            info['obsfreq'] = self.obsfreq.value
            info['obsfreq_unit'] = f'{self.obsfreq.unit}'
        else:
            info['obsfreq'] = None
        if self.eup is not None:
            info['eup'] = self.eup.value
            info['eup_unit'] = f'{self.eup.unit}'
        else:
            info['eup'] = None

        return info

    @u.quantity_input
    def set_obsfreq(self, vlsr: u.Quantity[u.km/u.s]):
        """Store the observed frequency of the transition."""
        equiv = u.doppler_radio(self.restfreq)
        self.obsfreq = to_rest_freq(self.restfreq, -vlsr, equiv)

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
            name.append(f'{self.species}'.replace('=', '').replace(' ', '_'))
        if 'qns' in include:
            qns = normalize_qns(self.qns)
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
          name: species name following [astroquery]
            (https://astroquery.readthedocs.io/en/v0.1-0/splatalogue.html).
          freq_range: frequency range to look for transitions.
          vlsr: optional; LSR velocity to obtain observed frequency.
          filter_out: optional; filter out transitions with given QNs.
          kwargs: optional additional query constraints.
        """
        transitions = _query_to_transition(name, freq_range, vlsr=vlsr,
                                           filter_out=filter_out, qns=qns,
                                           **kwargs)
        return cls(name.strip(), transitions)

    @classmethod
    def from_json(cls, filename: 'pathlib.Path',
                  vlsr: Optional[u.Quantity[u.km/u.s]] = None):
        """Load `Molecule` from JSON file."""
        info = json.loads(filename.read_text())
        name = info['name']
        transitions = [Transition.from_json(info=trinfo, vlsr=vlsr)
                       for trinfo in info['transitions']]

        return cls(name, transitions)

    def to_json(self, filename: 'pathlib.Path'):
        """Save molecule as JSON."""
        info = {'name': self.name}
        info['transitions'] = [transition.to_json_dict()
                               for transition in self.transitions]
        filename.write_text(json.dumps(info, indent=4))

    def merge_with(self, other,
                   filename: Optional['pathlib.Path'] = None):
        """Merge molecule transition lists."""
        if other.name != self.name:
            raise ValueError(('Different molecules to merge: '
                              f'{self.name} - {other.name}'))
        qns = self.qns_list
        for transition in other.transitions:
            if transition.qns not in qns:
                self.transitions.append(transition)

        if filename is not None:
            self.to_json(filename)

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

    @u.quantity_input
    def filter_range(self, rest_freq_range: u.Quantity[u.GHz],
                     qns: Optional[str] = None):
        """Create a new molecule with transitions in range.
        
        Args:
          rest_freq_range: Rest frequency range.
          qns: Optional; Filter quantum number in range.

        Returns:
          A new molecule with transitions (and QNs) in given range.
        """
        filtered_transitions = []
        low = min(rest_freq_range)
        high = max(rest_freq_range)
        for transition in self.transitions:
            if low <= transition.restfreq <= high:
                if qns is not None and qns != transition.qns:
                    continue
                filtered_transitions.append(transition)

        # Check not empty
        if len(filtered_transitions) == 0:
            raise NoTransitionError(self.name)

        return Molecule(self.name, filtered_transitions)

    def transition_info(self, qns: str) -> Transition:
        """Get transition information."""
        for transition in self.transitions:
            if qns == transition.qns:
                return transition

        return None

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
                 line_lists: Sequence[str] = ('CDMS', 'JPL'),
                 vlsr: Optional[u.Quantity[u.km/u.s]] = None,
                 save_molecule: Optional['pathlib.Path'] = None,
                 restore_molecule: Optional['pathlib.Path'] = None,
                 log: Callable = print) -> Molecule:
    """Get a `Molecule` from  a splatalogue query.

    Args:
      molecule: Name of the molecule.
      cube: Spectral cube where to search the molecule.
      qns: Optional; Quantum number.
      onlyj: Optional; Ingore F and K quantum numbers.
      line_lists: Optional; Line lists.
      vlsr: Optional; Systemic velocity of the source.
      save_molecule: Optional; Save molecule to JSON file.
      restore_molecule: Optional; Read stored molecule from JSON.
      log: Optional; Logging function
    """
    # Frequency ranges
    spectral_axis = cube.spectral_axis
    obs_freq_range = spectral_axis[[0,-1]]
    log((f'Observed freq. range: {obs_freq_range[0].value} '
         f'{obs_freq_range[1].value} {obs_freq_range[1].unit}'))
    if vlsr is not None:
        equiv = u.doppler_radio(cube_utils.get_restfreq(cube))
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
    if restore_molecule is not None and not restore_molecule.is_file():
        save_molecule = restore_molecule
    if restore_molecule is not None and restore_molecule.is_file():
        log(f'Restoring molecule: {restore_molecule}')
        mol = Molecule.from_json(restore_molecule, vlsr=vlsr)
        try:
            mol = mol.filter_range(rest_freq_range, qns=qns)
        except NoTransitionError:
            #log(f'No trasition {qns} in rage {rest_freq_range}')
            #log(f'Transition info: {mol.transition_info(qns)}')
            log('Transition not in saved molecule, seaching on splatalogue')
            try:
                aux = Molecule.from_query(f' {molecule} ', rest_freq_range,
                                          vlsr=vlsr, filter_out=filter_out,
                                          line_lists=line_lists, qns=qns)
            except InconsistentTableError as exc:
                log('Splatalogue not working')
                raise NoTransitionError(molecule) from exc
            aux.reduce_qns()
            mol.merge_with(aux, filename=restore_molecule)
            mol = aux
    else:
        mol = Molecule.from_query(f' {molecule} ', rest_freq_range,
                                  vlsr=vlsr, filter_out=filter_out,
                                  line_lists=line_lists, qns=qns)
        mol.reduce_qns()
        if save_molecule is not None:
            mol.to_json(save_molecule)
    log(f'Number of transitions: {len(mol.transitions)}')

    return mol
