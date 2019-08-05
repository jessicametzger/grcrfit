# Interpolate the Orlando +18 electron bremsstrahlung flux table,
#  using the PDDE framework, at the desired GR energies.
# Assuming ~linear relationship between log Eg and ebr flux,
#  between the given points.

import numpy as np
from scipy import interpolate
import os

from . import helpers as h

# should start in repo directory
path = os.getcwd()+'/'

# open e-bremss data (PDDE model from Orlando 2018)
ebremss = h.lstoarr(h.open_stdf(path+'data/ebremss_PDDE.dat'),None).astype(np.float)
logEes = np.log10(ebremss[:,0])

interp = interpolate.interp1d(logEes, ebremss[:,1], kind='linear', fill_value='extrapolate')

def get_fluxes(GRlogEgs):
    
    ebrfluxes = []
    for i in range(len(GRlogEgs)):
        ebrfluxes+=[interp(GRlogEgs[i])]
    
    return ebrfluxes