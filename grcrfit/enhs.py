# function that calculates the gamma ray flux enhancement
#  factor by interpolating the table of multiplication factors
#  from Kachelriess +14, and calculating their weighted sum
#  as described in Kachelriess +14 using given CR fluxes.

import numpy as np
import os
from scipy import interpolate

from . import helpers as h
from . import physics as ph

path = os.getcwd()+'/'

# get enhancement factors corresponding to the GR energies provided

# enhancement factor element bins
enh_els={'h': ['h'],
         'he': ['he'],
         'cno': ['c','n','o'],
         'mgsi': ['mg','si','ne','s'],
         'fe': ['fe']}
enh_els_ls=['h','he','cno','mgsi','fe']


# CONVERSION FACTORS TO GO FROM GAMMA E TO CR E
# linear interpolation in log-log space
factors=h.lstoarr(h.open_stdf(path+"data/Tp_Eg_convert.dat"),",").astype(np.float)
fac_interp=interpolate.interp1d(np.log10(factors[:,0]), np.log10(factors[:,1]), kind='linear', fill_value='extrapolate')

# DEFAULT FALLBACK LIS PARAMETERS FROM HONDA 2004
LIS_params={'h': [2.74,14900,2.15,0.21],
            'he': [2.64,600,1.25,0.14],
            'cno': [2.60,33.2,0.97,0.01],
            'mgsi': [2.79,34.2,2.14,0.01],
            'fe': [2.68,4.45,3.07,0.41]}

# LIS_params must be 1D array or list (index (alpha), normalization (K), b, c)
# E = total Ekin in MeV
def Honda_LIS(LIS_params, E):
    return np.array(LIS_params[1]*(E/1000. + LIS_params[2]*np.exp(-LIS_params[3]*np.sqrt(E/1000.)))**(-LIS_params[0]))


# get Kachelriess+14 multiplication factors
mults=h.open_stdf(path+"data/enh_f_2014.dat")

# make table of multiplication factors
mult_alphas = np.array([2., 2.2, 2.4, 2.6, 2.8, 3.])
mult_Es = np.array([10,100,1000])*1000. #MeV
tables=[mults[5:15],mults[16:26],mults[27:37],
        mults[38:48],mults[49:59],mults[60:70]]
mults=np.array([[x.split('\t')[2:-1] for x in tab] for tab in tables]).astype(np.float)

# make interpolator for each framework, & each interaction
# interp2d can't extrapolate, so fill_value is default (nearest)
interps = [[None]*mults.shape[1],[None]*mults.shape[1]]
for j in range(mults.shape[1]):
    interps[0][j] = interpolate.interp2d(np.log10(mult_Es), mult_alphas, mults[0*3:3*(0+1),j,:].transpose())
    interps[1][j] = interpolate.interp2d(np.log10(mult_Es), mult_alphas, mults[1*3:3*(1+1),j,:].transpose())

# construct abundance ratios
abund_rats=np.array([1,0.096,1.38e-3,2.11e-4,3.25e-5])

# interpolate mult. table for enh factors
# sum along interaction axis

# enhtype is 0 (QGSjet) or 1 (LHC)
# ext = True (explicit calculation assuing a single PL with index in highE) or False (extrapolate below 10 GeV) for enhancement factor calculation
# enh_fs = zeros array of the right shape for enhancement factors (same as GRdata, corresponding energy arrays)
# GRdata should be same as everywhere else (see construction in model.py)
# CRfluxes is dict of 5 lists (length=len(GRdata)) of flux arrs at GRdata energies
# CRinds is dict of 5 alphas (positive)
# CRfluxes/inds keys are same as enh_els
def enh(enhtype, ext, enh_fs, GREs, CRfluxes, CRinds):
    intplr=interps[enhtype]
    
    for k in range(len(GREs)):
        E=np.copy(GREs[k])
        logE = np.log10(E)
        
        # assume a constant enhancement factor below the 10 GeV (10e3 MeV)
        if not ext:
            first_ind = np.where(logE >= 4)[0][0]
            logE = logE[first_ind:]
        else:
            first_ind = 0
        
        for i in range(len(enh_els_ls)): #loop over projectile species
            current_proj_flux=np.copy(np.array(CRfluxes[enh_els_ls[i]][k]))[first_ind:]
            
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
                
                # add contribution of current projectile-target interaction
                contribution = abund_rats[j]*mult_ratio*flux_ratio
                enh_fs[k][first_ind:] += contribution
                
                # fill in extrapolated values
                enh_fs[k][0:first_ind] = enh_fs[k][first_ind]
    
    return enh_fs

