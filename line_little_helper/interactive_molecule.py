"""Create a `Molecule` from command line interactively."""
from pathlib import Path

from line_little_helper.molecule import Molecule, Transition
import astropy.units as u

def ask_transition() -> Transition:
    """Ask for transition information."""
    species = input('Species: ')
    qns = input('Quantum number: ')
    restfreq = input('Rest frequency (GHz): ')
    eup = input('Optional: Eup (K): ')
    logaij = input('Optional: log(Aij): ')
    if eup == '':
        eup = None
    else:
        eup = float(eup) * u.K
    if logaij == '':
        logaij = None
    else:
        logaij = float(logaij)

    return Transition(species.strip(), qns.strip(), float(restfreq)*u.GHz,
                      eup=eup, logaij=logaij)

if __name__ == '__main__':
    ans = input('[1] Create new molecule or [2] load from disk? ')
    if ans == '1':
        print("Let's create a new molecule")
        name = input('Molecule chemical name: ')
        transitions = []
        output = None
        initial = None
    else:
        output = Path(input('Molecule file name: '))
        initial = Molecule.from_json(output)
        print('Loaded molecule:')
        print(initial)

    print('Inserting transitions:')
    print('----------------------')
    while True:
        transitions.append(ask_transition())
        ans = input('Add more transitions? [y, n]: ')
        if ans.lower() in ['n', 'no']:
            break
    print('----------------------')

    print('Molecule:')
    molecule = Molecule(name=name, transitions=transitions)
    if initial is not None:
        molecule.merge_with(initial)
    print(molecule)
    ans = input('Save to disk? [y, n]: ')
    if ans.lower() in ['y', 'yes']:
        if output is None:
            output = Path(input('Output file: '))
        molecule.to_json(output)
