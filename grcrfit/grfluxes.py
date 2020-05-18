# function that interpolates the gammads.dat file to get
#  the ds/dE contributions of hadronic interactions for a range of
#  Tp (proton kinetic energy, or kinetic energy per nucleon for alpha) values, 
#  and desired Eg (gamma energy)
#  values, and calculates the weighted sum given the proton (or alpha) fluxes
#  at those Tp values.
# 
# gammads.dat file is calculated using the cparamlib package
#  described in Kamae +06.
#
# gammads_hybrid.dat is calculated using the cparmlib package (Kamae +06) and
# the AAfrag package (Kahelriess +14) 

import numpy as np
from scipy import interpolate
import os

from . import helpers as h
from . import physics as ph

# should start in repo directory
path = os.getcwd()+'/'

# open gamma-ray cross-section (ds/dE) contribution table
gammads = h.lstoarr(h.open_stdf(path+'data/gammads.dat'),',').astype(np.float) # Kamae +06
Tps = gammads[1:,0] # incoming proton energy
Egs = gammads[0,1:] # outgoing gamma energy
logEgs = np.log10(Egs)
ds = gammads[1:,1:]*Egs/(4*np.pi) # convert to right emissivity units (MeV*mb/MeV => mb/MeV/sr)

gammads_hybrid = h.lstoarr(h.open_stdf(path+'data/gammads_hybrid.dat'),',').astype(np.float) # hybrid (Kamae +06 (non-diffractive) below 10 GeV and AAfrag above that)
Tps_hybrid = gammads_hybrid[1:,0] # incoming proton energy
Egs_hybrid = gammads_hybrid[0,1:] # outgoing gamma energy
logEgs_hybrid = np.log10(Egs_hybrid)
ds_hybrid = gammads_hybrid[1:,1:]*Egs_hybrid/(4*np.pi) # convert to right emissivity units (MeV*mb/MeV => mb/MeV/sr)

gammads_pHe_hybrid = h.lstoarr(h.open_stdf(path+'data/gammads_pHe_hybrid.dat'),',').astype(np.float) # hybrid (Kamae +06 (non-diffractive) below 10 GeV and AAfrag above that)
Tps_pHe_hybrid = gammads_pHe_hybrid[1:,0] # incoming proton kinetic energy
Egs_pHe_hybrid = gammads_pHe_hybrid[0,1:] # outgoing gamma energy
logEgs_pHe_hybrid = np.log10(Egs_pHe_hybrid)
ds_pHe_hybrid = gammads_pHe_hybrid[1:,1:]*Egs_pHe_hybrid/(4*np.pi) # convert to right emissivity units (MeV*mb/MeV => mb/MeV/sr)

gammads_Hep_hybrid = h.lstoarr(h.open_stdf(path+'data/gammads_Hep_hybrid.dat'),',').astype(np.float) # hybrid (Kamae +06 (non-diffractive) below 10 GeV and AAfrag above that)
Tps_Hep_hybrid = gammads_Hep_hybrid[1:,0] # incoming kinetic energy per nucleon of alpha
Tks_Hep_hybrid = gammads_Hep_hybrid[1:,0]*4.0 # itotal kinetic energy of alpha
Egs_Hep_hybrid = gammads_Hep_hybrid[0,1:] # outgoing gamma energy
logEgs_Hep_hybrid = np.log10(Egs_Hep_hybrid)
ds_Hep_hybrid = gammads_Hep_hybrid[1:,1:]*Egs_Hep_hybrid/(4*np.pi) # convert to right emissivity units (MeV*mb/MeV => mb/MeV/sr)

gammads_HeHe_hybrid = h.lstoarr(h.open_stdf(path+'data/gammads_HeHe_hybrid.dat'),',').astype(np.float) # hybrid (Kamae +06 (non-diffractive) below 10 GeV and AAfrag above that)
Tps_HeHe_hybrid = gammads_HeHe_hybrid[1:,0] # incoming kinetic energy per nucleon of alpha
Tks_HeHe_hybrid = gammads_HeHe_hybrid[1:,0]*4.0 # total kinetic energy of alpha
Egs_HeHe_hybrid = gammads_HeHe_hybrid[0,1:] # outgoing gamma energy
logEgs_HeHe_hybrid = np.log10(Egs_HeHe_hybrid)
ds_HeHe_hybrid = gammads_HeHe_hybrid[1:,1:]*Egs_HeHe_hybrid/(4*np.pi) # convert to right emissivity units (MeV*mb/MeV => mb/MeV/sr)

widths=np.empty(Tps.shape)
for j in range(Tps.shape[0]):
    widths[j] = Tps[j]*.348; #MeV
    if (j < 38):
        widths[j] = 1*widths[j]
    elif (j == 38):
        widths[j] = .75*widths[j]
    elif (j > 38):
        widths[j] = .5*widths[j]

widths_hybrid=np.empty(Tps_hybrid.shape)
for j in range(Tps_hybrid.shape[0]):
    widths_hybrid[j] = Tps_hybrid[j]*.348; #MeV
    if (j < 38):
        widths_hybrid[j] = 1*widths_hybrid[j]
    elif (j == 38):
        widths_hybrid[j] = .75*widths_hybrid[j]
    elif (j > 38):
        widths_hybrid[j] = .5*widths_hybrid[j]

widths_pHe_hybrid=np.empty(Tps_pHe_hybrid.shape)
for j in range(Tps_pHe_hybrid.shape[0]):
    widths_pHe_hybrid[j] = Tps_pHe_hybrid[j]*.348; #MeV
    if (j < 38):
        widths_pHe_hybrid[j] = 1*widths_pHe_hybrid[j]
    elif (j == 38):
        widths_pHe_hybrid[j] = .75*widths_pHe_hybrid[j]
    elif (j > 38):
        widths_pHe_hybrid[j] = .5*widths_pHe_hybrid[j]

widths_Hep_hybrid=np.empty(Tps_Hep_hybrid.shape)
for j in range(Tps_Hep_hybrid.shape[0]):
    widths_Hep_hybrid[j] = Tps_Hep_hybrid[j]*.348; #MeV
    if (j < 38):
        widths_Hep_hybrid[j] = 1*widths_Hep_hybrid[j]
    elif (j == 38):
        widths_Hep_hybrid[j] = .75*widths_Hep_hybrid[j]
    elif (j > 38):
        widths_Hep_hybrid[j] = .5*widths_Hep_hybrid[j]

widths_HeHe_hybrid=np.empty(Tps_HeHe_hybrid.shape)
for j in range(Tps_HeHe_hybrid.shape[0]):
    widths_HeHe_hybrid[j] = Tps_HeHe_hybrid[j]*.348; #MeV
    if (j < 38):
        widths_HeHe_hybrid[j] = 1*widths_HeHe_hybrid[j]
    elif (j == 38):
        widths_HeHe_hybrid[j] = .75*widths_HeHe_hybrid[j]
    elif (j > 38):
        widths_HeHe_hybrid[j] = .5*widths_HeHe_hybrid[j]

# proton momenta at Tps. NB ph.E_to_p converts total kinetic energy to momentum
Pps = ph.E_to_p(Tps, ph.M_DICT['h'])
Pps_hybrid = ph.E_to_p(Tps_hybrid, ph.M_DICT['h'])
Pps_pHe_hybrid = ph.E_to_p(Tps_pHe_hybrid, ph.M_DICT['h'])
Pps_Hep_hybrid = ph.E_to_p(Tks_Hep_hybrid, ph.M_DICT['he'])
Pps_HeHe_hybrid = ph.E_to_p(Tks_HeHe_hybrid, ph.M_DICT['he'])

# interpolater assumes ~linear relationship between ds/dE and log Eg
interps = [interpolate.interp1d(logEgs, ds[i,:], kind='linear', fill_value='extrapolate') for i in range(ds.shape[0])]
interps_hybrid = [interpolate.interp1d(logEgs_hybrid, ds_hybrid[i,:], kind='linear', fill_value='extrapolate') for i in range(ds_hybrid.shape[0])]
interps_pHe_hybrid = [interpolate.interp1d(logEgs_pHe_hybrid, ds_pHe_hybrid[i,:], kind='linear', fill_value='extrapolate') for i in range(ds_pHe_hybrid.shape[0])]
interps_Hep_hybrid = [interpolate.interp1d(logEgs_Hep_hybrid, ds_Hep_hybrid[i,:], kind='linear', fill_value='extrapolate') for i in range(ds_Hep_hybrid.shape[0])]
interps_HeHe_hybrid = [interpolate.interp1d(logEgs_HeHe_hybrid, ds_HeHe_hybrid[i,:], kind='linear', fill_value='extrapolate') for i in range(ds_HeHe_hybrid.shape[0])]

# only from p-p interactions; will be enhanced
def get_fluxes_pp(LIS_params_pp, GRlogEgs, crfunc):
    
    # get proton CR flux at Tps, cancel out sr^-1, convert area^-1 units to mbarn^-1
    Jps = crfunc(LIS_params_pp, Pps, 'h')*4*np.pi*1e-34
    
    GRfluxes = []
    for i in range(len(GRlogEgs)):
        GRfluxes += [np.zeros(GRlogEgs[i].shape)]
        
        for j in range(Tps.shape[0]):
            
            # interpolate at desired GR energies for the current Tp
            # weight by proton flux at current Tp, dTp (bin width)
            try: GRfluxes[i] += interps[j](GRlogEgs[i])*Jps[j]*widths[j]
            except: return -np.inf
    
    return GRfluxes

def get_fluxes_pp_hybrid(LIS_params_pp, GRlogEgs, crfunc):
    
    # get proton CR flux at Tps, cancel out sr^-1, convert area^-1 units to mbarn^-1
    Jps_hybrid = crfunc(LIS_params_pp, Pps_hybrid, 'h')*4*np.pi*1e-34
    
    GRfluxes_hybrid = []
    for i in range(len(GRlogEgs)):
        GRfluxes_hybrid += [np.zeros(GRlogEgs[i].shape)]
        
        for j in range(Tps_hybrid.shape[0]):
            
            # interpolate at desired GR energies for the current Tp
            # weight by proton flux at current Tp, dTp (bin width)
            try: GRfluxes_hybrid[i] += interps_hybrid[j](GRlogEgs[i])*Jps_hybrid[j]*widths_hybrid[j]
            except: return -np.inf
    
    return GRfluxes_hybrid

def get_fluxes_pHe_hybrid(LIS_params_pHe, GRlogEgs, crfunc):
    
    # get proton CR flux at Tps, cancel out sr^-1, convert area^-1 units to mbarn^-1
    Jps_pHe_hybrid = crfunc(LIS_params_pHe, Pps_pHe_hybrid, 'h')*4*np.pi*1e-34
    
    GRfluxes_pHe_hybrid = []
    for i in range(len(GRlogEgs)):
        GRfluxes_pHe_hybrid += [np.zeros(GRlogEgs[i].shape)]
        
        for j in range(Tps_pHe_hybrid.shape[0]):
            
            # interpolate at desired GR energies for the current Tp
            # weight by proton flux at current Tp, dTp (bin width)
            try: GRfluxes_pHe_hybrid[i] += interps_pHe_hybrid[j](GRlogEgs[i])*Jps_pHe_hybrid[j]*widths_pHe_hybrid[j]
            except: return -np.inf
    
    return GRfluxes_pHe_hybrid

def get_fluxes_Hep_hybrid(LIS_params_Hep, GRlogEgs, crfunc):
    
    # get alpha CR flux at Tps, cancel out sr^-1, convert area^-1 units to mbarn^-1
    Jps_Hep_hybrid = crfunc(LIS_params_Hep, Pps_Hep_hybrid, 'he')*4*np.pi*1e-34
    
    GRfluxes_Hep_hybrid = []
    for i in range(len(GRlogEgs)):
        GRfluxes_Hep_hybrid += [np.zeros(GRlogEgs[i].shape)]
        
        for j in range(Tps_Hep_hybrid.shape[0]):
            
            # interpolate at desired GR energies for the current Tp
            # weight by proton flux at current Tp, dTp (bin width)
            try: GRfluxes_Hep_hybrid[i] += interps_Hep_hybrid[j](GRlogEgs[i])*Jps_Hep_hybrid[j]*widths_Hep_hybrid[j]
            except: return -np.inf
    
    return GRfluxes_Hep_hybrid

def get_fluxes_HeHe_hybrid(LIS_params_HeHe, GRlogEgs, crfunc):
    
    #get alpha CR flux at Tps, cancel out sr^-1, convert area^-1 units to mbarn^-1
    Jps_HeHe_hybrid = crfunc(LIS_params_HeHe, Pps_HeHe_hybrid, 'he')*4*np.pi*1e-34
    
    GRfluxes_HeHe_hybrid = []
    for i in range(len(GRlogEgs)):
        GRfluxes_HeHe_hybrid += [np.zeros(GRlogEgs[i].shape)]
        
        for j in range(Tps_HeHe_hybrid.shape[0]):
            
            # interpolate at desired GR energies for the current Tp
            # weight by proton flux at current Tp, dTp (bin width)
            try: GRfluxes_HeHe_hybrid[i] += interps_HeHe_hybrid[j](GRlogEgs[i])*Jps_HeHe_hybrid[j]*widths_HeHe_hybrid[j]
            except: return -np.inf
    
    return GRfluxes_HeHe_hybrid

