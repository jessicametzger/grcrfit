# physics, constants, and conversions that will be frequently used

import numpy as np
import re

# PHYSICS CONSTANTS

C_SI = 299792458.

M_DICT = {'h': 1.0072, '1h': 1., 'he': 4.0026, 'c': 12., 'n': 14.0067,
         'o': 15.999, 'ne': 20.1797, 'mg': 24.305, 'si': 28.0855,
         's': 32.065, 'fe': 55.845}
Z_DICT = {'h': 1., '1h': 1., 'he': 2., 'c': 6., 'n': 7.,
         'o': 8., 'ne': 10., 'mg': 12., 'si': 14.,
         's': 16., 'fe': 26.}

# Ek in MeV, Z = charge, N = nuc
def E_to_p(Ek,M):
    Em=931.49*M #mass energy
    
    Etot = Ek + Em #total energy
    radical=Etot**2 - Em**2
    
    p = (1/C_SI)*np.sqrt(radical)
    return p

# return total energy in MeV from p in MeV/c
def p_to_Etot(p,M):
    p=np.array(p)
    Em=931.49*M #mass energy
    
    radical=(p*C_SI)**2. + Em**2.
    Etot = np.sqrt(radical)
    return Etot

# given Ekin in MeV, returns total energy in MeV
def get_Etot(Ek, M):
    Em=931.49*M #mass energy
    return Ek + Em

# momentum at Ek = 1 GeV
p1_DICT = {}
for key in M_DICT:
    p1_DICT[key] = E_to_p(1*1000, M_DICT[key])

# MODULATION PHYSICS
# E in MeV, Z = charge, N = nuc

def demod_energy(E,phi,Z):
    E_IS = E + abs(Z)*phi
    return E_IS #MeV Ekin

def demod_flux(flux,E,E_IS,M):
    Em=931.49*M #mass energy
    
    E=E + Em #MeV Ekin to MeV Etot
    E_IS=E_IS + Em #MeV Ekin to MeV Etot
    flux_IS = flux*(E_IS**2 - Em**2)/(E**2 - Em**2)
    return flux_IS

def mod_energy(E,phi,Z):
    E_IS = E - abs(Z)*phi
    return E_IS #MeV Ekin

def mod_flux(flux_IS,E,E_IS,M):
    Em=931.49*M #mass energy
    
    E=E + Em #MeV Ekin to MeV Etot
    E_IS=E_IS + Em #MeV Ekin to MeV Etot
    flux = flux_IS*(E**2 - Em**2)/(E_IS**2 - Em**2)
    return flux

