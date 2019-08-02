import numpy as np
import os
from scipy import interpolate

from . import helpers as h
from . import physics as ph

path = os.getcwd()+'/'

# get enhancement factors corresponding to the GR energies provided

enh_els={'h': ['h'],
         'he': ['he'],
         'cno': ['c','n','o'],
         'mgsi': ['mg','si','ne','s'],
         'fe': ['fe']}
enh_els_ls=['h','he','cno','mgsi','fe']

# DEFAULT FALLBACK LIS PARAMETERS FROM HONDA 2004
LIS_params={'h': [2.74,14900,2.15,0.21],
            'he': [2.64,600,1.25,0.14],
            'cno': [2.60,33.2,0.97,0.01],
            'mgsi': [2.79,34.2,2.14,0.01],
            'fe': [2.68,4.45,3.07,0.41]}

# LIS_params must be 1D array or list
def Honda_LIS(LIS_params, E):
    return np.array(LIS_params[1]*(E/1000. + LIS_params[2]*np.exp(-LIS_params[3]*np.sqrt(E/1000.)))**(-LIS_params[0]))

# get Kachelriess+14 multiplication factors
mults=h.open_stdf(path+"data/enh_f_2014.dat","r")

# make table of multiplication factors
mult_alphas = np.array([2., 2.2, 2.4, 2.6, 2.8, 3.])
mult_Es = np.array([10,100,1000])*1000. #MeV
tables=[mults[5:15],mults[16:26],mults[27:37],
        mults[38:48],mults[49:59],mults[60:70]]
mults=np.array([[x.split('\t')[2:-1] for x in tab] for tab in tables]).astype(np.float)

# make interpolator for each interaction & each framework
interps = [[None]*mults.shape[1],[None]*mults.shape[1]]
for j in range(mults.shape[1]):
    interps[0][j] = interpolate.interp2d(np.log10(mult_Es), mult_alphas, mults[0*3:3*(0+1),j,:].transpose())
    interps[1][j] = interpolate.interp2d(np.log10(mult_Es), mult_alphas, mults[1*3:3*(1+1),j,:].transpose())

# construct abundance ratios
abund_rats=np.array([1,0.096,1.38e-3,2.11e-4,3.25e-5])

# interpolate mult. table for enh factors
# sum along interaction axis

# enhtype is 0 (QGSjet) or 1 (LHC)
# enh_fs =  empty array of the right shape for enhancement factors (same as GRdata, corresponding energy arrays)
# GRdata should be same as everywhere else (see construction in model.py)
# CRfluxes is dict of 5 lists (length=len(GRdata)) of flux arrs at GRdata energies
# CRinds is dict of 5 alphas (positive)
# CRfluxes/inds keys are same as enh_els
def enh(enhtype, enh_fs, GRdata, CRfluxes, CRinds):
    intplr=interps[enhtype]
    
    for k in range(len(GRdata)):
        E=np.copy(GRdata[k][:,0])
        logE = np.log10(E)
        
        for i in range(len(enh_els_ls)): #loop over projectile species
            current_proj_flux=np.copy(np.array(CRfluxes[enh_els_ls[i]][k]))
            
            for j in range(2): #loop over target species
                current_intplr = intplr[j*5 + i]
                mult_f = current_intplr(logE, CRinds[enh_els_ls[i]])
                
                if i==0 and j==0:
                    mult_ratio = np.repeat(1.,logE.shape[0])
                    flux_ratio = np.repeat(1.,logE.shape[0])
                    mult_f_pp = mult_f
                    CR_flux_p = current_proj_flux
                else:
                    mult_ratio = mult_f/mult_f_pp
                    flux_ratio = current_proj_flux/CR_flux_p
                
                # sum over interactions
                contribution = abund_rats[j]*mult_ratio*flux_ratio
                enh_fs[k] += contribution
    
    return enh_fs
    