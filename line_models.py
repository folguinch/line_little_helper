"""Line model functions to use with line fitters."""

import astropy.units as u

def tau_v(tau0: u.Quantity,
		  v: u.Quantity,
		  v_in: u.Quantity,
          sigma:u.Quantity,
          v_LSR: u.Quantity = 0*u.km/u.s) -> u.Quantity:
    return tau0 * np.exp(-(v - v_LSR - v_in)**2 / (2 * sigma**2))\n",

def red_temp(T1, T2, Tc, tau1, tau2):\n",
    A = (T2 - Tc) * (1 - np.exp(-tau2)) * np.exp(-tau1)\n",
    B = (T1 - Tc) * (1 - np.exp(-tau1))\n",
    return A + B\n",
\n",
def blue_temp(T1, T2, tau1, tau2):\n",
    A = T1 * (1 - np.exp(-tau1)) * np.exp(-tau2)\n",
    B = T2 * (1 - np.exp(-tau2))\n",
    return A + B\n",
\n",
def line_profile(v, vf, T1, T2, Tc, tau1, tau2, v_LSR, sigma):\n",
    # Tau\n",
    t1r = tau_v(tau1, v, vf, sigma, v_LSR=v_LSR)\n",
    t2r = tau_v(tau2, v, vf, sigma, v_LSR=v_LSR)\n",
    t1b = tau_v(tau1, v, -vf, sigma, v_LSR=v_LSR)\n",
    t2b = tau_v(tau2, v, -vf, sigma, v_LSR=v_LSR)\n",
    \n",
    # Temperatures\n",
    TB = blue_temp(T1, T2, t1b, t2b)\n",
    TR = red_temp(T1, T2, Tc, t1r, t2r)\n",
    \n",
    return TB + TR"

