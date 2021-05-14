"""Useful global values."""
import astropy.units as u

ALMA_BANDS = {
    3: (84*u.GHz, 116*u.GHz),
    4: (125*u.GHz, 163*u.GHz),
    5: (163*u.GHz, 211*u.GHz),
    6: (211*u.GHz, 275*u.GHz),
    7: (275*u.GHz, 373*u.GHz),
    8: (385*u.GHz, 500*u.GHz),
    9: (602*u.GHz, 720*u.GHz),
    10: (787*u.GHz, 950*u.GHz),
}

