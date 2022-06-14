"""Line model functions to use with line fitters."""
import astropy.units as u
import numpy as np

def tau_v(tau0: u.Quantity,
		  v: u.Quantity,
		  v_in: u.Quantity,
          sigma:u.Quantity,
          v_lsr: u.Quantity = 0*u.km/u.s) -> u.Quantity:
    return tau0 * np.exp(-(v - v_lsr - v_in)**2 / (2 * sigma**2))

def red_temp(temp1, temp2, tempc, tau1, tau2):
    aux1 = (temp2 - tempc) * (1 - np.exp(-tau2)) * np.exp(-tau1)
    aux2 = (temp1 - tempc) * (1 - np.exp(-tau1))
    return aux1 + aux2

def blue_temp(temp1, temp2, tau1, tau2):
    aux1 = temp1 * (1 - np.exp(-tau1)) * np.exp(-tau2)
    aux2 = temp2 * (1 - np.exp(-tau2))
    return aux1 + aux2

def line_profile(v, vf, temp1, temp2, tempc, tau1, tau2, v_lsr, sigma):
    # Tau
    t1r = tau_v(tau1, v, vf, sigma, v_lsr=v_lsr)
    t2r = tau_v(tau2, v, vf, sigma, v_lsr=v_lsr)
    t1b = tau_v(tau1, v, -vf, sigma, v_lsr=v_lsr)
    t2b = tau_v(tau2, v, -vf, sigma, v_lsr=v_lsr)

    # Temperatures
    temp_blue = blue_temp(temp1, temp2, t1b, t2b)
    temp_red = red_temp(temp1, temp2, tempc, t1r, t2r)

    return temp_blue + temp_red

