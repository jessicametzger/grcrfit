# function that interpolates the gammads.dat file to get
#  the ds/dE contributions of p-p interactions for a range of
#  Tp (proton energy) values, and desired Eg (gamma energy)
#  values, and calculates the weighted sum given the proton fluxes
#  at those Tp values.
# 
# gammads.dat file is calculated using the cparamlib package
#  described in Kamae +06.

import numpy as np
from scipy import interpolate
import os

from . import helpers as h
from . import physics as ph

# should start in repo directory
path = os.getcwd()+'/'

# open gamma-ray cross-section (ds/dE) contribution table
gammads = h.lstoarr(h.open_stdf(path+'data/gammads.dat'),',').astype(np.float)
Tps = gammads[1:,0] # incoming proton energy
Egs = gammads[0,1:] # outgoing gamma energy
logEgs = np.log10(Egs)
ds = gammads[1:,1:]*Egs/(4*np.pi) #convert to right emissivity units

widths=np.empty(Tps.shape)
for j in range(Tps.shape[0]):
    widths[j] = Tps[j]*.348; #MeV
    if (j < 38):
        widths[j] = 1*widths[j]
    elif (j == 38):
        widths[j] = .75*widths[j]
    elif (j > 38):
        widths[j] = .5*widths[j]

# proton momenta at Tps
Pps = ph.E_to_p(Tps, ph.M_DICT['h'])

# interpolater assumes ~linear relationship between ds/dE and log Eg
interps = [interpolate.interp1d(logEgs, ds[i,:], kind='linear', fill_value='extrapolate') for i in range(ds.shape[0])]

# only from p-p interactions; will be enhanced
def get_fluxes_pp(LIS_params_pp, GRlogEgs, crfunc):
    
    #get proton CR flux at Tps, cancel out sr^-1, convert area^-1 units to mbarn^-1
    Jps = crfunc(LIS_params_pp, Pps, ph.M_DICT['h'])*4*np.pi*1e-34
    
    GRfluxes = []
    for i in range(len(GRlogEgs)):
        GRfluxes += [np.zeros(GRlogEgs[i].shape)]
        
        for j in range(Tps.shape[0]):
            
            # interpolate at desired GR energies for the current Tp
            # weight by proton flux at current Tp, dTp (bin width)
            GRfluxes[i] += interps[j](GRlogEgs[i])*Jps[j]*widths[j]
    
    return GRfluxes
