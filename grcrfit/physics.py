# "fundamental" physical constants, conversions, etc. that will be frequently used

import numpy as np
import re

# speed of light
C_SI = 299792458.

# element masses and charges (# nucleons, proton charge)
M_DICT = {'h': 1.0072, '1h': 1., 'he': 4.0026, 'c': 12., 'n': 14.0067,
         'o': 15.999, 'ne': 20.1797, 'mg': 24.305, 'si': 28.0855,
         's': 32.065, 'fe': 55.845}
Z_DICT = {'h': 1., '1h': 1., 'he': 2., 'c': 6., 'n': 7.,
         'o': 8., 'ne': 10., 'mg': 12., 'si': 14.,
         's': 16., 'fe': 26.}

# previous best-fit LIS's for initializing sampler
LIS_DICT = {'s': [1.57423715677e-11,-2.7102],
            'mg': [7.93411178821e-11,-2.6272],
            'ne': [3.23379880315e-11,-2.6513],
            'h': [5.539805949e-12,-2.86],
            'si': [8.34896554956e-11,-2.6589],
            'o': [1.296202054131e-10,-2.6199],
            'he': [3.10830363936e-11,-2.7796],
            'c': [8.678671340299998e-12,-2.8783],
            'n': [3.0749133394999997e-12,-2.8892]}

# Ek in MeV --> momentum in MeV/c
# M = number of nucleons
def E_to_p(Ek,M):
    Em=931.49*M #mass energy
    
    Etot = Ek + Em #total energy
    radical=Etot**2 - Em**2
    if np.any(radical<0):
        return -np.inf
    
    p = (1/C_SI)*np.sqrt(radical)
    return p

# p in MeV/c, M in # nucleons --> Etot in MeV
def p_to_Etot(p,M):
    p=np.array(p)
    Em=931.49*M #mass energy
    
    radical=(p*C_SI)**2. + Em**2.
    Etot = np.sqrt(radical)
    return Etot

# Ekin in MeV, M in # nucleons --> Etot in MeV
def get_Etot(Ek, M):
    Em=931.49*M #mass energy
    return Ek + Em

# calculate each element's momentum at Ek = 1 GeV
p1_DICT = {}
for key in M_DICT:
    p1_DICT[key] = E_to_p(1*1000, M_DICT[key])

    
# MODULATION PHYSICS
# E(_IS) in MeV, Z = charge, M = mass in # nucleons,
# phi = modulation parameter (Gleeson & Axford 1968) in MV
# _IS indicates interstellar (unmodulated) quantities

def demod_energy(E,phi,Z):
    E_IS = E + abs(Z)*phi
    return E_IS #MeV Ekin

def demod_flux(flux,E,E_IS,M):
    Em=931.49*M #mass energy
    
    E=E + Em #MeV Ekin to MeV Etot
    E_IS=E_IS + Em #MeV Ekin to MeV Etot
    flux_IS = flux*(E_IS**2 - Em**2)/(E**2 - Em**2)
    if np.any(flux_IS<0):
        return -np.inf
    return flux_IS

def mod_energy(E,phi,Z):
    E_IS = E - abs(Z)*phi
    if np.any(E_IS<0): return -np.inf
    return E_IS #MeV Ekin

def mod_flux(flux_IS,E,E_IS,M):
    Em=931.49*M #mass energy
    
    E=E + Em #MeV Ekin to MeV Etot
    E_IS=E_IS + Em #MeV Ekin to MeV Etot
    flux = flux_IS*(E**2 - Em**2)/(E_IS**2 - Em**2)
    if np.any(flux<0):
        return -np.inf
    return flux

