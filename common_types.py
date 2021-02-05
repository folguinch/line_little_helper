from typing import List, Optional, Tuple, TypeVar, NewType
import pathlib

import astropy
import matplotlib.pyplot as plt

"""Commoly used type aliases."""
Path = TypeVar('Path', pathlib.Path, str)
Plot = NewType('Plot', Tuple[plt.Figure, plt.Axes])
QPair = Tuple[astropy.units.Quantity, astropy.units.Quantity]
Table = NewType('Table', astropy.table.Table)
