"""Commoly used type aliases."""
from typing import Tuple, TypeVar, NewType
import pathlib

from astropy import table
import astropy.units as u
import matplotlib.pyplot as plt

Path = TypeVar('Path', pathlib.Path, str)
Plot = NewType('Plot', Tuple[plt.Figure, plt.Axes])
QPair = Tuple[u.Quantity, u.Quantity]
Table = NewType('Table', table.Table)
