import numpy as np
import subprocess
from scipy import interpolate

from . import helpers as h
from . import physics as ph

path = os.getcwd()+'/'

gammads = h.open_stdf(path+'/grcrfit/gammads.dat','r')
gammads = np.array([x.split(',') for x in gammads]).astype(np.float)
Tps = gammads[1:,0]
Egs = gammads[0,1:]
ds = gammads[1:,1:]

Pps = E_to_p(Tps, M_DICT['h'])

interps = [interpolate.interp1d(Egs, ds[i,:], kind='linear', fill_value='extrapolate') for i in range(ds.shape[0])]

# only from p-p interactions; will be enhanced
def get_fluxes_pp(LIS_params_pp, GRdata, crfunc):
    
    Jps = crfunc(LIS_params_pp, Pps, M_DICT['h'])*4*np.pi*1e-34 #get proton CR flux, convert units
    
    GRfluxes = []
    for i in range(len(GRdata)):
        GRfluxes += [np.zeros(GRdata[i][:,0].shape)]
        
        for j in range(Tps.shape[0]):
            
            width = Tps[j]*.348; #MeV
            if (j < 38):
                width = 1*width
            elif (j == 38):
                width = .75*width
            elif (j > 38):
                width = .5*width
            
            # interpolate at desired GR energies for the current Tp
            GRfluxes[i] += interps[j](GRdata[i][:,0])*Jps[j]*width
    
    return GRfluxes