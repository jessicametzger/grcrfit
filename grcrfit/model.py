# configuration of the Model class, which is typically created within run.py.
# Each Model object initializes its own lnlike, lnprior, and lnprob functions
#  which are used by the emcee sampler.
# Much of the code below consists of re-formatting the data & parameter lists
#  so the MCMC helper functions run as efficiently as possible.

import numpy as np
import copy
import sys

from . import helpers as h
from . import physics as ph
from . import crfluxes as crf
from . import grfluxes as grf
from . import ebrfluxes as ebrf
from . import enhs as enf

# each run will create its own custom "Model" object
# containing the helper functions for the MCMC

# modflags:
# - 'pl' = 's' or 'b', determines whether to assume power-law with just momentum ('s')
#    or with momentum and velocity, each getting its own index ('b')
# - 'enh' = 0 or 1, determines which enhancement framework to use out of the two given
#    in Kachelriess +14 (0 for QGS and 1 for LHC)
# - 'ppmodel' = 0 or 1 or 2, determined which pp-interaction model (0: Kamae +06, 1: hybrid (Kamae +06 and AAfrag), 2: hybrid and pHe, Hep, HeHe in addition to pp)
# - 'weights' = 3-item list giving relative weights of CR, Voyager, and GR log-like.
#    Note that priors aren't weighted, so may need to tune weight normalization to get
#    correct relative weighting between likelihood & prior. If weights==None, no weighting
# - 'priors' = 0 or 1. If 0, gaussian priors on scaling (0.05/0.1 for cr/gr; hard coded). If 1, flat prior on scaling. 
#    NB param limits for cr-scaling are set by scaleRanges (default=0.15), and gr-scaling from 0.5 to 2.0 (hard coded). 
#    Also NB phis are limited within +/- 0.15*phi0 where phi0 is the value by Usoskin+17.
#    (NB the effect on fit is minor, since the number of data points is large and hence data will dominate the fit.)
# - 'vphi_err': Voyager phi error (1sigma) can be set and used if 'priors'=0.
# - 'cr/grscaling' = True or False, whether or not to scale all CR experiments  except AMS-02 
#    or all GR experiments to correct for systematic errors
# - 'enhext' = True or False. If True we calculate the enhancement factor explicitly assuming single PL with index of highE. If False we extrapolate below 10 GeV.
# - 'priorlimits' = True or False, whether or not to constrain the parameter in specified ranges
# - 'fixd': If "None" the delta is treated as a free parameter. If some number is given, deita is fixed to that value.
# - 'one_d' = True or False, whether we use single value or multiple values for delta (sharpness of the break)
# - 'fix_vphi': If "None" the voyager phi is treated as a free parameter ('vphi_err' work).
# -  If some number is given, it is virtually fixed to that value. ('vphi_err' is ignored)

class Model():
    
    # initialize the Model object
    def __init__(self, modflags, data):
        self.modflags = modflags
        self.data = data
        
        # if parallel, these will need to be pickled
        global lnlike, lnprior, lnprob
        
        
        # PARAMETER CONSOLIDATION
        
        # count number of points for weighting
        self.nCRpoints=0
        self.nVRpoints=0
        self.nGRpoints=0
        
        # - get Usoskin+ reference phi array from data for priors
        # - get IDs, labels, etc. all in same order
        # - make list of each dataset's atomic number & mass
        # - get various parameter counts to help llike, lprob 
        self.phis=[]
        self.phierrs=[]
        self.scaleRanges=[]
        self.CRels=[]
        self.CRexps=[]
        self.CRexps_phi=[] ### duplicate based on phi will be removed
        self.CRdata=[]
        self.CRZs=[]
        self.CRMs=[]
        self.CRtimes=[]
        self.GRexps=[]
        self.GRdata=[]
        self.VRinds=[]
        
        # voyager phi and error for (virtually) fixing
        self.vphi0 = 0.1 # will be used if 'fix_vphi'==None.
        self.vphi_err0 = 0.1 # we want to limit vphi close to vphi0. not used if 'fix_vphi'==None 
        if not self.modflags['fix_vphi']==None: # if voyager phis are fixed
            self.vphi0 = self.modflags['fix_vphi']

        elements = list(self.data.keys())
        elements.sort()
        for elkey in elements: #loop thru elements
            
            experiments = list(self.data[elkey].keys())
            experiments.sort()
            
            for expkey in experiments: #loop thru experiments
                
                # separate GR and CR data
                if 'gamma' in elkey.lower():
                    self.GRexps+=[expkey]
                    self.GRdata+=[self.data[elkey][expkey][0]]
                    
                    self.nGRpoints+=self.data[elkey][expkey][0].shape[0]
                else:
                    self.phis+=[self.data[elkey][expkey][2]]
                    self.phierrs+=[self.data[elkey][expkey][3]]
                    self.CRels+=[elkey.lower()]
                    self.CRexps+=[expkey]
                    self.CRexps_phi+=[expkey]
                    self.CRdata+=[self.data[elkey][expkey][0]]
                    self.CRZs+=[ph.Z_DICT[elkey.lower()]]
                    self.CRMs+=[ph.M_DICT[elkey.lower()]]
                    self.CRtimes+=[self.data[elkey][expkey][1]]
                    if self.modflags['scaleRange']==None:
                        self.scaleRanges+=[0.15]
                    else:
                        if not (elkey in modflags['scaleRange']):
                            self.scaleRanges+=[0.15]
                        else:
                            dict_scaleRanges = {k.lower():v for k, v in modflags['scaleRange'][elkey].items()}
                            if expkey in dict_scaleRanges:
                                self.scaleRanges+=[dict_scaleRanges[expkey]]
                            else:
                                self.scaleRanges+=[0.15]
                    
                    # will need to distinguish voyager data for likelihood weighting
                    if 'voyager1' in expkey.lower() and '2012' in expkey:
                        self.nVRpoints+=self.data[elkey][expkey][0].shape[0]
                        self.VRinds+=[len(self.CRdata)-1]
                    else:
                        self.nCRpoints+=self.data[elkey][expkey][0].shape[0]
        
        # make everything a np array
        self.phis=np.array(self.phis).astype(np.float)
        self.phierrs=np.array(self.phierrs).astype(np.float)
        self.scaleRanges=np.array(self.scaleRanges).astype(np.float)
        self.CRZs=np.array(self.CRZs).astype(np.float)
        self.CRMs=np.array(self.CRMs).astype(np.float)
        self.CRels=np.array(self.CRels)
        self.CRexps=np.array(self.CRexps)
        self.CRexps_phi=np.array(self.CRexps_phi)
        self.CRtimes=np.array(self.CRtimes)
        self.phitimes=np.copy(self.CRtimes)
        self.GRexps=np.array(self.GRexps)
        
        # remove duplicate phi's based on matching time, exp.
        # remove duplicates from self.phis, self.phierrs
        # & create list of lists corresponding to self.CRexps inds
        # of same phi.
        self.match_inds = []
        self.remove_inds = []
        for i in range(self.CRtimes.shape[0]):
            
            # skip if already accounted for
            if i in self.remove_inds:
                continue
            
            # don't count Voyager H and He as duplicates
            duplicates = np.where((self.CRtimes == self.CRtimes[i]) & (self.CRexps == self.CRexps[i]) &\
                         ([self.CRels[j]==self.CRels[i] or 'voyager1' not in self.CRexps[j] for j in range(self.CRels.shape[0])]))[0]
            duplicates=list(duplicates)
            duplicates.sort()
            
            # for scaling
            if self.modflags['crscaling'] and 'ams02' in self.CRexps[i].lower():
                self.AMS02_inds = duplicates
            
            if len(duplicates)>1:
                self.remove_inds+=duplicates[1:]
            
            # which CR data indices to assign the phi at this position in self.match_inds
            self.match_inds+=[duplicates]
        
        # remove duplicates:
        print ("## self.phis=", self.phis)
        print ("## self.phierrs=", self.phierrs)
        print ("## self.CRexps", self.CRexps)
        print ("## self.CRexps_phi", self.CRexps_phi)
        print ("## self.scaleRanges", self.scaleRanges)
        self.phis = np.delete(self.phis, self.remove_inds)
        self.phierrs = np.delete(self.phierrs, self.remove_inds)
        self.phitimes = np.delete(self.phitimes, self.remove_inds)
        self.CRexps_phi = np.delete(self.CRexps_phi, self.remove_inds)
        self.scaleRanges = np.delete(self.scaleRanges, self.remove_inds)
        # dump phi and phierrs for diagnostics
        print ("### self.phis=", self.phis)
        print ("### self.phierrs=", self.phierrs)
        print ("### self.CRexps", self.CRexps)
        print ("### self.CRexps_phi", self.CRexps_phi)
        print ("### self.scaleRanges", self.scaleRanges)
        # if voyager phi is fixed, update the initial values
        if not self.modflags['fix_vphi']==None: # if voyager phis are fixed
            for i in range(len(self.CRexps_phi)):
                if 'voyager1' in self.CRexps_phi[i]:
                    self.phis[i] = self.vphi0
                    self.phierrs[i] = self.vphi_err0
                    # dump phi and phierrs for diagnostics
                    print ("## self.phis=", self.phis)
                    print ("## self.phierrs=", self.phierrs)
        
        # also create list of ref. scaling factors & errors (all 1, 0.03)
        # corresponding with the list of phi's, for prior calculation.
        # (So proton and He share the common scaling for the same phi)
        # Don't ignore AMS-02, scaling will be set to 1 anyways.
        # If 'crscaling'=True, they will be used in prior calculation (lp_scale). Also will be used as hard limit
        if self.modflags['crscaling']:
            self.scales = np.repeat(1, self.phis.shape[0])
            ### self.scaleerrs = np.repeat(0.03, self.phis.shape[0]) ## we will use self.scaleRanges instead
        
        
        # determine order of the elements' LIS parameters
        self.LISorder=np.unique(self.CRels)
        if self.LISorder.shape[0]==0: # if no CR data, only fit hydrogen spectrum
            self.LISorder = np.array(['h']) 
        self.LISdict={}
        for i in range(self.LISorder.shape[0]):
            self.LISdict[self.LISorder[i]] = i
            
        self.nels = self.LISorder.shape[0]
        self.nphis=self.phis.shape[0]
        self.npoints = self.nCRpoints + self.nVRpoints + self.nGRpoints
        
        # for indexing purposes
        if self.modflags['crscaling']:
            self.ncrparams = self.nphis*2
        else:
            self.ncrparams = self.nphis
        
        # whether each el's LIS params will have just norm, alpha1 ('s'),
        # norm, alpha1, and alpha ('b')
        # or norm, alpha1, alpha2, pc_br ('br').
        if self.modflags['pl']=='s':
            self.nLISparams=2
        elif self.modflags['pl']=='b':
            self.nLISparams=3
        elif self.modflags['pl']=='br':
            if self.modflags['one_d']: self.nLISparams=4
            else: self.nLISparams=5
        elif self.modflags['pl']=='dbr':
            # independent among elements: LIS_norm, alpha1(HE), alpha3(LE), pc_br2(M-L), delta2(M-L)
            # common among elements: alpha2(ME), pc_br1(H-M), delta1(H-M)
            self.nLISparams=5
        elif self.modflags['pl']=='dbr2':
            # independent among elements: LIS_norm, alpha1(HE), alpha3(LE), rig_br2(M-L), delta2(M-L)
            # common among elements: alpha2(ME), rig_br1(H-M), delta1(H-M)
            self.nLISparams=5
        else:
            print('Must give valid CR model flag')
            sys.exit()
            
            
        # DATA CONSOLIDATION
        
        # save old copies of un-consolidated data (for analysis purposes)
        # columns: Emean, Elow, Ehigh, value, stat-, stat+, sys-, sys+
        self.CRdata_old = [np.copy(x) for x in self.CRdata]
        self.GRdata_old = [np.copy(x) for x in self.GRdata]
        
        # create list of bare-minimum CR data arrays (only E bin average, single err)
        for i in range(len(self.CRdata)):
            
            # errorbars: rms of sys/stat, then avg of upper/lower
            errs = (np.sqrt(self.CRdata[i][:,-4]**2 + self.CRdata[i][:,-2]**2) + \
                    np.sqrt(self.CRdata[i][:,-3]**2 + self.CRdata[i][:,-1]**2))/2.
            
            # remove all but most impt cols: E [MeV], flux, err
            self.CRdata[i] = self.CRdata[i][:,np.array([0,3])]
            self.CRdata[i] = np.append(self.CRdata[i], np.array([errs]).transpose(), axis=1)
            
        # create list of bare-minimum GR data arrays (only E bin average, single err)
        for i in range(len(self.GRdata)):
            
            # errorbars: rms of sys/stat, then avg of upper/lower
            errs = (np.sqrt(self.GRdata[i][:,-4]**2 + self.GRdata[i][:,-2]**2) + \
                    np.sqrt(self.GRdata[i][:,-3]**2 + self.GRdata[i][:,-1]**2))/2.
            
            # remove all but most impt cols: E [MeV], emissivity, err
            self.GRdata[i] = self.GRdata[i][:,np.array([0,3])]
            self.GRdata[i] = np.append(self.GRdata[i], np.array([errs]).transpose(), axis=1)
            
            # change y-axis from emissivity*1e24 to emissivity
            self.GRdata[i][:,1:] = self.GRdata[i][:,1:]*1e-24
        
        # data energies for comparison - can edit this variable if want cr fluxes at other energies
        self.CREs = []
        for i in range(len(self.CRdata)):
            self.CREs += [self.CRdata[i][:,0]]
        
        # data energies for comparison - can edit this variable if want gr fluxes at other energies
        self.GREs = []
        for i in range(len(self.GRdata)):
            self.GREs += [self.GRdata[i][:,0]]
        
        # CR momenta at GR energies (for enhancement calculation)
        # shape: 5-item dict (corresponding to 5 categories of elements for enh routine), 
        # each containing a dict which gives GRdata-shaped list at each entry corresponding to that element
        self.CRp_atGRE = {}
        for key in enf.enh_els:
            self.CRp_atGRE[key]={}
            for subkey in enf.enh_els[key]:
                current_el=[]
                for j in range(len(self.GREs)):
                    
                    # old code: GR energy is about 10 times smaller than CR p energy (take from Mori 1997)
                    # factors=np.repeat(10.,self.GREs[i].shape[0])
                    # new code: Kachelriess +14 refers to CR flux at Ek=Eg, so we use factor=1 instead. So CRp_atGRE literary means
                    #  CR momentum at Ek=Eg
                    factors=np.repeat(1.,self.GREs[i].shape[0])
                    current_el+=[ph.E_to_p(self.GREs[i]*ph.M_DICT[subkey]*factors, ph.M_DICT[subkey])]
                    
                self.CRp_atGRE[key][subkey] = current_el
        
        # HELPER FUNCTIONS FOR LNLIKE
        # (initialize before since they only have to be initialized once)
        
        # CR FLUXES
        
        # choose a CR flux formula based on the flagged one
        if self.modflags['pl']=='s':
            self.crformula = crf.flux_spl
            self.crformula_IS = crf.flux_spl_IS
        elif self.modflags['pl']=='b':
            self.crformula = crf.flux_bpl
            self.crformula_IS = crf.flux_bpl_IS
        elif self.modflags['pl']=='br':
            self.crformula = crf.flux_brpl
            self.crformula_IS = crf.flux_brpl_IS
        elif self.modflags['pl']=='dbr':
            self.crformula = crf.flux_dbrpl
            self.crformula_IS = crf.flux_dbrpl_IS
        elif self.modflags['pl']=='dbr2':
            self.crformula = crf.flux_dbr2pl
            self.crformula_IS = crf.flux_dbr2pl_IS
        
        # create list of model interstellar CR fluxes, same order as data ones
        # only used in plotting
        def crfunc_IS(theta):
            crflux_ls=[None] * self.CRels.shape[0]
            for i in range(self.nphis):
                
                # loop thru experiments that share that phi
                for index in self.match_inds[i]:
                    
                    # use that element's LIS parameters, and that experiment's phi, to get CR flux
                    LIS_params=list(theta[self.ncrparams + int(self.modflags['grscaling']) +\
                                          self.LISdict[self.CRels[index]]*self.nLISparams:\
                                          self.ncrparams + int(self.modflags['grscaling']) +\
                                          (self.LISdict[self.CRels[index]] + 1)*self.nLISparams])
                    
                    if self.modflags['pl']=='br' and self.modflags['one_d']: # add universal delta parameter
                        if self.modflags['fixd']==None:
                            LIS_params+=[theta[-1]]
                        else:
                            LIS_params+=[self.modflags['fixd']]
                        
                    if self.modflags['pl']=='dbr': # add common alpha2(ME), pc_br1(H-M), delta1(H-M)
                        LIS_params+=[theta[-3], theta[-2], theta[-1]]
                        
                    if self.modflags['pl']=='dbr2': # add common alpha2(ME), rig_br1(H-M), delta1(H-M)
                        LIS_params+=[theta[-3], theta[-2], theta[-1]]
                        
                    crflux_ls[index]=self.crformula_IS(LIS_params, ph.E_to_p(self.CREs[index], self.CRMs[index]), 
                                                    self.CRels[index])
            
            return crflux_ls
        
        # create list of model CR fluxes, same order as data ones
        def crfunc(theta):
            crflux_ls=[None] * self.CRels.shape[0]
            for i in range(self.nphis):
                
                # loop thru experiments that share that phi
                for index in self.match_inds[i]:
                    
                    # use that element's LIS parameters, and that experiment's phi, to get CR flux
                    LIS_params=list(theta[self.ncrparams + int(self.modflags['grscaling']) +\
                                          self.LISdict[self.CRels[index]]*self.nLISparams:\
                                          self.ncrparams + int(self.modflags['grscaling']) +\
                                          (self.LISdict[self.CRels[index]] + 1)*self.nLISparams])
                    
                    if self.modflags['pl']=='br' and self.modflags['one_d']: #add universal delta parameter
                        if self.modflags['fixd']==None:
                            LIS_params+=[theta[-1]]
                        else:
                            LIS_params+=[self.modflags['fixd']]
                        
                    if self.modflags['pl']=='dbr': # add common alpha2(ME), pc_br1(H-M), delta1(H-M)
                        LIS_params+=[theta[-3], theta[-2], theta[-1]]

                    if self.modflags['pl']=='dbr2': # add common alpha2(ME), rig_br1(H-M), delta1(H-M)
                        LIS_params+=[theta[-3], theta[-2], theta[-1]]

                    # account for scaling parameter if needed
                    if self.modflags['crscaling']:
                        current_phi = theta[2*i]
                        current_scale = theta[2*i+1]
                    else:
                        current_phi = theta[i]
                        current_scale = 1

                    crflux_ls[index]=self.crformula(LIS_params, current_phi, self.CREs[index], 
                                                   self.CRels[index])*current_scale
            
            return crflux_ls
        
        
        # ENHANCEMENT FACTOR
            
        # compile old LIS fluxes, inds (if not included in model) for enhancement factor calculation (Honda +04)
        self.CRfluxes={'h': None, 'he': None, 'cno': None, 'mgsi': None, 'fe': None}
        self.CRinds={'h': None, 'he': None, 'cno': None, 'mgsi': None, 'fe': None}
        self.fit_els=[x.lower() for x in list(self.LISorder)]
        for key in enf.enh_els_ls:
            
            # if some particle types are not fit, we need to use old LIS data (Honda +04)
            if not all(x in self.fit_els for x in enf.enh_els[key]):
            # if 1: # test to see the enhancemen factor calculation based on the LIS of Honda +04
                
                # add old spectral index
                self.CRinds[key] = enf.LIS_params[key][0]
                
                # add old fluxes
                self.CRfluxes[key] = []
                for i in range(len(self.GRdata)):
                    mean_mass = np.mean([ph.M_DICT[x] for x in enf.enh_els[key]])
                    
                    # old code: GR energy is about 10 times smaller than CR p energy (take from Mori 1997)
                    # factors=np.repeat(10.,self.GREs[i].shape[0])
                    # new code: Kachelriess +14 refers to CR flux at Ek=Eg, so we use factor=1 instead. So CRp_atGRE literary means
                    #  CR momentum at Ek=Eg
                    factors=np.repeat(1.,self.GREs[i].shape[0])
                    self.CRfluxes[key] += [enf.Honda_LIS(enf.LIS_params[key], self.GREs[i]*factors)]
        
        # empty flux array (same shame as in GRdata)
        self.empty_fluxes=[np.zeros(self.GREs[i].shape) for i in range(len(self.GREs))]
        
        # get enhancement factors (all/pp and all_other_than_He/pp) at GR energies
        def enhfuncs(theta):
            
            CRfluxes_theta = copy.deepcopy(self.CRfluxes)
            CRinds_theta = copy.deepcopy(self.CRinds)
            
            # add theta values to CRfluxes/inds
            for key in CRfluxes_theta:
                # if these CR LIS's are included in model (not retrieving old values)
                if CRfluxes_theta[key] == None:
                    
                    if key not in ['cno', 'mgsi']: #if the bin contains only 1 element
                        LIS_params=list(theta[self.ncrparams + int(self.modflags['grscaling']) +\
                                              self.LISdict[key]*self.nLISparams:\
                                              self.ncrparams + int(self.modflags['grscaling']) +\
                                              (self.LISdict[key] + 1)*self.nLISparams])
                        
                        if self.modflags['pl']=='br' and self.modflags['one_d']: #add universal delta parameter
                            if self.modflags['fixd']==None:
                                LIS_params+=[theta[-1]]
                            else:
                                LIS_params+=[self.modflags['fixd']]
                        
                        if self.modflags['pl']=='dbr': # add common alpha2(ME), pc_br1(H-M), delta1(H-M)
                            LIS_params+=[theta[-3], theta[-2], theta[-1]]

                        if self.modflags['pl']=='dbr2': # add common alpha2(ME), rig_br1(H-M), delta1(H-M)
                            LIS_params+=[theta[-3], theta[-2], theta[-1]]

                        # enhancement factors given w.r.t. positive spectral ind.
                        # if beta pl, this will be momentum (higher-energy) index
                        # if broken pl, this will be first (higher-energy) index
                        CRinds_theta[key] = LIS_params[1]
                        
                        CRfluxes_theta[key] = []
                        for i in range(len(self.GRdata)):
                            try: 
                                addition = self.crformula_IS(LIS_params, self.CRp_atGRE[key][key][i], key)
                                addition = addition.reshape(self.GREs[i].shape)
                            except: return -np.inf
                            CRfluxes_theta[key] += [addition]
                            
                    else: #if the bin contains multiple elements
                        
                        # alpha will be avg weighted by flux at 10 GeV/n; fluxes will be sum of fluxes
                        weighted_sum = 0
                        flux_sum = 0
                        fluxes = [np.copy(x) for x in self.empty_fluxes] #must copy np arrays individually
                        for el in enf.enh_els[key]:
                            LIS_params=list(theta[self.ncrparams + int(self.modflags['grscaling']) + \
                                                  self.LISdict[el]*self.nLISparams:\
                                                  self.ncrparams + int(self.modflags['grscaling']) + \
                                                  (self.LISdict[el] + 1)*self.nLISparams])
                            
                            if self.modflags['pl']=='br' and self.modflags['one_d']: #add universal delta parameter
                                if self.modflags['fixd']==None:
                                    LIS_params+=[theta[-1]]
                                else:
                                    LIS_params+=[self.modflags['fixd']]
                            
                            if self.modflags['pl']=='dbr': # add common alpha2(ME), pc_br1(H-M), delta1(H-M)
                                LIS_params+=[theta[-3], theta[-2], theta[-1]]

                            if self.modflags['pl']=='dbr': # add common alpha2(ME), rig_br1(H-M), delta1(H-M)
                                LIS_params+=[theta[-3], theta[-2], theta[-1]]

                            # add to weighted sum
                            flux_10GeVn=self.crformula_IS(LIS_params, ph.p10_DICT[el], el)
                            weighted_sum += LIS_params[1]*flux_10GeVn #enhancement factors given w.r.t. positive spectral ind.
                            flux_sum += flux_10GeVn
                            
                            for i in range(len(self.GREs)):
                                try: 
                                    addition = self.crformula_IS(LIS_params, self.CRp_atGRE[key][el][i],\
                                                             el).reshape(self.GREs[i].shape)
                                except: return -np.inf
                                fluxes[i] += addition
                        
                        CRfluxes_theta[key] = fluxes
                        CRinds_theta[key] = weighted_sum/flux_sum
            
            enh1_fs = [np.copy(x) for x in self.empty_fluxes]
            enh1_fs = enf.enh1(self.modflags['enh'], self.modflags['enhext'], enh1_fs, self.GREs, CRfluxes_theta, CRinds_theta)
            enh2_fs = [np.copy(x) for x in self.empty_fluxes]
            enh2_fs = enf.enh2(self.modflags['enh'], self.modflags['enhext'], enh2_fs, self.GREs, CRfluxes_theta, CRinds_theta)
            
            # print for diagnostics
            # print ("### CRfluxes_theta=", CRfluxes_theta)
            # print ("### CRinds_theta=", CRinds_theta)
            # print ("### self.GREs=", self.GREs)
            # print ("### enh_fs=", enh_fs)
            return {'enh1':enh1_fs, 'enh2':enh2_fs}
            # return enh_fs
            
        # get GR fluxes due to pp interaction at log10 of GR energies
        # can also consider some modflag options here
        self.GRlogEgs=[np.log10(self.GREs[i]) for i in range(len(self.GREs))]
        def grfunc_pp(theta):
            LIS_params_pp = list(theta[self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*self.LISdict['h']:\
                                       self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*(self.LISdict['h']+1)])
            
            # add universal delta parameter
            if self.modflags['pl']=='br' and self.modflags['one_d']:
                if self.modflags['fixd']==None:
                    LIS_params_pp+=[theta[-1]]
                else:
                    LIS_params_pp+=[self.modflags['fixd']]
                
            # add common alpha2(ME), pc_br1(H-M), delta1(H-M)
            if self.modflags['pl']=='dbr':
                LIS_params_pp+=[theta[-3], theta[-2], theta[-1]]

            # add common alpha2(ME), rig_br1(H-M), delta1(H-M)
            if self.modflags['pl']=='dbr2':
                LIS_params_pp+=[theta[-3], theta[-2], theta[-1]]

            if self.modflags['ppmodel']==0:
                return grf.get_fluxes_pp(LIS_params_pp, self.GRlogEgs, self.crformula_IS)
            elif self.modflags['ppmodel']==1:
                return grf.get_fluxes_pp_hybrid(LIS_params_pp, self.GRlogEgs, self.crformula_IS)
            elif self.modflags['ppmodel']==2:
                # If ppmodel=2, we will calculate gamma-ray fluxes for pp, pHe, Hep, and HeHe interactions. Here we obtain the flux for pp interactions.
                return grf.get_fluxes_pp_hybrid(LIS_params_pp, self.GRlogEgs, self.crformula_IS)
        
        # get GR fluxes due to pp, pHe, Hep, and HeHe interactions at log10 of GR energies
        # can also consider some modflag options here
        self.GRlogEgs=[np.log10(self.GREs[i]) for i in range(len(self.GREs))]
        def grfuncs(theta):
            LIS_params_pp = list(theta[self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*self.LISdict['h']:\
                                       self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*(self.LISdict['h']+1)])
            LIS_params_pHe = list(theta[self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*self.LISdict['h']:\
                                       self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*(self.LISdict['h']+1)])
            LIS_params_Hep = list(theta[self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*self.LISdict['he']:\
                                       self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*(self.LISdict['he']+1)])
            LIS_params_HeHe = list(theta[self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*self.LISdict['he']:\
                                       self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*(self.LISdict['he']+1)])
            
            # add universal delta parameter
            if self.modflags['pl']=='br' and self.modflags['one_d']:
                if self.modflags['fixd']==None:
                    LIS_params_pp+=[theta[-1]]
                    LIS_params_pHe+=[theta[-1]]
                    LIS_params_Hep+=[theta[-1]]
                    LIS_params_HeHe+=[theta[-1]]
                else:
                    LIS_params_pp+=[self.modflags['fixd']]
                    LIS_params_pHe+=[self.modflags['fixd']]
                    LIS_params_Hep+=[self.modflags['fixd']]
                    LIS_params_HeHe+=[self.modflags['fixd']]
                
            # add common alpha2(ME), pc_br1(H-M), delta1(H-M)
            if self.modflags['pl']=='dbr':
                LIS_params_pp+=[theta[-3], theta[-2], theta[-1]]
                LIS_params_pHe+=[theta[-3], theta[-2], theta[-1]]
                LIS_params_Hep+=[theta[-3], theta[-2], theta[-1]]
                LIS_params_HeHe+=[theta[-3], theta[-2], theta[-1]]

            # add common alpha2(ME), rig_br1(H-M), delta1(H-M)
            if self.modflags['pl']=='dbr2':
                LIS_params_pp+=[theta[-3], theta[-2], theta[-1]]
                LIS_params_pHe+=[theta[-3], theta[-2], theta[-1]]
                LIS_params_Hep+=[theta[-3], theta[-2], theta[-1]]
                LIS_params_HeHe+=[theta[-3], theta[-2], theta[-1]]

            # return fluxes only when ppmodel==2
            if self.modflags['ppmodel']==2:
                # If ppmodel=2, we will calculate gamma-ray fluxes for pp, pHe, Hep, and HeHe interactions. Here we obtain the flux for pp interactions.
                return {'grfluxes_pp': grf.get_fluxes_pp_hybrid(LIS_params_pp, self.GRlogEgs, self.crformula_IS), \
                        'grfluxes_pHe': grf.get_fluxes_pHe_hybrid(LIS_params_pHe, self.GRlogEgs, self.crformula_IS), \
                        'grfluxes_Hep': grf.get_fluxes_Hep_hybrid(LIS_params_Hep, self.GRlogEgs, self.crformula_IS), \
                        'grfluxes_HeHe': grf.get_fluxes_HeHe_hybrid(LIS_params_HeHe, self.GRlogEgs, self.crformula_IS)}
                #
                # for test. cross section of p-He interaction is ~4 times larger than that of pp. 
                # So this returns similar gamma-ray flux to that for "ppmodel=1"
                # return [0.25*v for v in grf.get_fluxes_pHe_hybrid(LIS_params_pHe, self.GRlogEgs, self.crformula_IS)] 
                #
                # for test. cross section of He-p interaction is ~4 times larger than that of pp, and the He flux is ~18 times smaller than that of p (at the same Ekin/N).
                # So this returns similar gamma-ray flux to that of "ppmodel=1"
                # return [4.5*v for v in grf.get_fluxes_Hep_hybrid(LIS_params_Hep, self.GRlogEgs, self.crformula_IS)]
                #
                # for test. cross section of He-He interaction is ~15 times larger than that of pp, and the He flux is ~18 times smaller than that of p (at the same Ekin/N).
                # So this returns similar gamma-ray flux to that of "ppmodel=1"
                # return [1.25*v for v in grf.get_fluxes_HeHe_hybrid(LIS_params_HeHe, self.GRlogEgs, self.crformula_IS)]
        
        # retrieve e-bremss values at desired energies
        def ebrfunc():
            return ebrf.get_fluxes(self.GRlogEgs)
        
        # e-bremss value doesn't depend on model, so calculate it only once here
        ebr_fluxes = ebrfunc()
        
        
        # CREATE LNLIKE FUNCTION
        
        def lnlike(theta):
            
            # set ams02 scaling to 1
            # (keep this for compatibility. NB it does not work since the param of walker not affected.
            # instead, we limit AMSscaling in lnprior)
            if self.modflags['crscaling']:
                theta=np.array(theta)
                #theta[2*np.array(self.AMS02_inds) + 1] = 1.  # this will override h_scale
                theta[2*np.array(self.AMS02_inds[0]) + 1] = 1.
                theta=list(theta)
            
            crlike=0
            vrlike=0
            grlike=0
            
            # create CR fluxes to match all datasets
            if self.nCRpoints!=0:
                cr_fluxes = crfunc(theta)
                try:
                    for i in range(len(cr_fluxes)):
                        if not np.all(np.isfinite(cr_fluxes[i])):
                            return -np.inf
                except:
                    return -np.inf

                # compare CR fluxes to data
                # do Voyager separately for weighting
                if self.nCRpoints!=0:
                    crlike = -.5*np.sum(np.array([np.sum(((cr_fluxes[i] - self.CRdata[i][:,1])/self.CRdata[i][:,2])**2.) \
                              for i in range(len(cr_fluxes)) if i not in self.VRinds]))
                else:
                    print("Must include CR data")
                    sys.exit()

                if self.nVRpoints!=0:
                    vrlike = -.5*np.sum(np.array([np.sum(((cr_fluxes[i] - self.CRdata[i][:,1])/self.CRdata[i][:,2])**2.) \
                              for i in range(len(cr_fluxes)) if i in self.VRinds]))
            
            # add gamma-ray contribution
            if self.nGRpoints!=0:
                
                # get enhancement factors
                enh1_f = enhfuncs(theta)['enh1']
                enh2_f = enhfuncs(theta)['enh2']
                # enh_f = enhfunc(theta)

                # make sure they are list of finite numbers
                try:
                    for i in range(len(enh1_f)):
                        if not (np.all(np.isfinite(enh1_f[i]))):
                            return -np.inf
                except: return -np.inf
                try:
                    for i in range(len(enh2_f)):
                        if not (np.all(np.isfinite(enh2_f[i]))):
                            return -np.inf
                except: return -np.inf

                # get p-p GR fluxes at gamma data's energies
                if (self.modflags['ppmodel']==0 or self.modflags['ppmodel']==1):
                    gr_fluxes_pp = grfunc_pp(theta)
#                    gr_fluxes = grfunc_pp(theta)
                if (self.modflags['ppmodel']==2):
                    gr_fluxes_pp = grfuncs(theta)['grfluxes_pp']
                    gr_fluxes_pHe = grfuncs(theta)['grfluxes_pHe']
                    gr_fluxes_Hep = grfuncs(theta)['grfluxes_Hep']
                    gr_fluxes_HeHe = grfuncs(theta)['grfluxes_HeHe']
                
                # make sure they are list of finite numbers
                try:
                    for i in range(len(gr_fluxes_pp)):
                        if not (np.all(np.isfinite(gr_fluxes_pp[i]))):
                            return -np.inf
                except: return -np.inf
                if (self.modflags['ppmodel']==2):                
                    try:
                        for i in range(len(gr_fluxes_pHe)):
                            if not (np.all(np.isfinite(gr_fluxes_pHe[i]))):
                                return -np.inf
                    except: return -np.inf
                    try:
                        for i in range(len(gr_fluxes_Hep)):
                            if not (np.all(np.isfinite(gr_fluxes_Hep[i]))):
                                return -np.inf
                    except: return -np.inf
                    try:
                        for i in range(len(gr_fluxes_HeHe)):
                            if not (np.all(np.isfinite(gr_fluxes_HeHe[i]))):
                                return -np.inf
                    except: return -np.inf

                # enhance p-p GR fluxes & add e-bremss data
                gr_fluxes = copy.deepcopy(gr_fluxes_pp)
#                gr_fluxes = [np.zeros(len(gr_fluxes_pp))] # a bit faster
                for i in range(len(self.GRdata)):
                    if (self.modflags['ppmodel']==0 or self.modflags['ppmodel']==1):
                        gr_fluxes[i] = enh1_f[i]*gr_fluxes_pp[i] + ebr_fluxes[i]
                    elif (self.modflags['ppmodel']==2):
                        gr_fluxes[i] = enh2_f[i]*gr_fluxes_pp[i] + gr_fluxes_pHe[i]*enf.abund_rats_dict['he'] + gr_fluxes_Hep[i] + gr_fluxes_HeHe[i]*enf.abund_rats_dict['he'] + ebr_fluxes[i]
                    
                    # add scaling parameter if gr scaling
                    if self.modflags['grscaling']:
                        scale = theta[self.ncrparams]
                    else:
                        scale = 1
                        
                    gr_fluxes[i] = gr_fluxes[i]*scale

                # compare GR fluxes to data
                grlike = -.5*np.sum(np.array([np.sum(((gr_fluxes[i] - self.GRdata[i][:,1])/self.GRdata[i][:,2])**2.) \
                          for i in range(len(gr_fluxes))]))
            
            # add weights
            if self.modflags['weights']!=None:
                crlike = crlike*modflags['weights'][0]/self.nCRpoints
                vrlike = vrlike*modflags['weights'][1]/self.nVRpoints
                grlike = grlike*modflags['weights'][2]/self.nGRpoints
            
            # sum up weighted contributions
            return crlike + vrlike + grlike
        
        
        # CREATE LNPRIOR FUNCTION
        
        # apply gaussian priors
        # force spectral indices to be negative, normalizations to be positive
        if self.modflags['priors']==0:
            
            def lp_scale(theta): 
                lp=0
                
                # priors on scaling factors.
                if self.modflags['crscaling']:
                    ##lp += -.5*np.sum(((theta[1:self.ncrparams:2] - self.scales)/self.scaleerrs)**2.)
                    lp += -.5*np.sum(((theta[1:self.ncrparams:2] - self.scales)/(0.05))**2.) # hard coded
                if self.modflags['grscaling']:
                    lp += -.5*np.sum(((theta[self.ncrparams] - 1)/(0.1))**2.) # hard coded, allow larger deviation (from 1) than cr_scale
                    
                return lp
            
        # no gaussian priors
        elif self.modflags['priors']==1:
            
            def lp_scale(theta):
                return 0
            
        else:
            print("Must give valid 'priors' entry in modflags.")
            sys.exit()
        
        
        # set hard limits on parameters
        # also add gaussian priors
        def lnprior(theta):
            
            theta_slice = theta[0:self.ncrparams:(int(self.modflags['crscaling'])+1)]
            # print for diagnostics
            # print ("### thetha=", theta)
            # print ("### thetha(slice)=", theta_slice)
            # print ("## vphi0/vphi_err0=", self.vphi0, self.vphi_err0)
            # print ("## CRexps=", self.CRexps, len(self.CRexps))
            # print ("## CRexps_phi=", self.CRexps_phi, len(self.CRexps_phi))
            ### limit phi so that the fit converge well
            for kk in range(len(theta_slice)):
                phi = theta_slice[kk]
                phi0 = self.phis[kk]
                err0 = self.phierrs[kk]
                if (phi<phi0-err0 or phi>=phi0+err0):
                    return -np.inf
            ### limit CRscale to be 1 +/- scaleRange(default 0.15) so that the fit converge well.
            ### NB AMS scaling is always 1. (startpos=1.0 with 0 error)
            if self.modflags['crscaling']:
                theta_slice2 = theta[1:self.ncrparams:(int(self.modflags['crscaling'])+1)]
                for kk in range(len(theta_slice2)):
                    CRscale = theta_slice2[kk]
                    scale0 = self.scales[kk]
                    ##err0 = self.scaleerrs[kk]
                    err0 = self.scaleRanges[kk]
                    if (CRscale<scale0-err0 or CRscale>scale0+err0):
                        return -np.inf
                ## set AMS scaling very close to 1. Not used anymore since it is now always 1 (startpos=1.0 with 0 error)
                #AMSscale = theta_slice2[self.AMS02_inds[0]]
                #if AMSscale<=1-0.005 or AMSscale>=1+0.005:
                #    return -np.inf
            ### limit GR scaling from 0.5 to 2.0 so that the fit converge well
            if self.modflags['grscaling']:
                GRscale = theta[self.ncrparams]
                if (GRscale<0.5 or GRscale>2.0):
                    return -np.inf
            #### see if the parameter is voyager phi
            #### (not used anymore since vphi_err0 is already used for err0 above) 
            #if not self.modflags['fix_vphi']==None: # if voyager phis are fixed (some variation allowed for plotting)
            #    for i in range(len(self.CRexps_phi)):
            #        if 'voyager1' in self.CRexps_phi[i]:
            #            vphi = theta_slice[i]
            #            # print ("# vphi=", vphi)
            #            if vphi<=self.vphi0-self.vphi_err0 or vphi>self.vphi0+self.vphi_err0:
            #                return -np.inf

            ### positive phis # now allow negative values
            #if np.any(theta[0:self.ncrparams:(int(self.modflags['crscaling'])+1)] <= 0):
            #    return -np.inf
            
            for i in range(self.nels):
                # positive (high-energy, momentum) index
                if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1] < 0:
                    return -np.inf
                
                # positive LIS norm
                if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams] < 0:
                    return -np.inf
                
                if self.modflags['pl']=='br':
                    
                    # positive low-energy index above 0
                    if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 2] <= 0:
                        return -np.inf
                    
                    # high-energy index higher than low-energy one
                    if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1] <= \
                       theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 2]:
                        return -np.inf
                    
                    # break momentume between 0 and 300 GeV
                    # We may want to share the break among spices in rigidity (rather than momentum). So rig_br also calculated 
                    massNumber = ph.M_DICT[self.LISorder[i].lower()]
                    atomNumber = ph.Z_DICT[self.LISorder[i].lower()]
                    # old code (for reference): E_br->p_br->pc_br
                    # E_br = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3]
                    # p_br = ph.E_to_p(E_br, massNumber)
                    # pc_br = p_br*ph.C_SI # in MeV
                    pc_br = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3]
                    rig_br = pc_br/atomNumber # in MV
                    if pc_br <= 0 or pc_br >= 3e5:
                        return -np.inf
                    
                    # delta above 0
                    if self.modflags['fixd']==None:
                        if self.modflags['one_d']:
                            if theta[-1] <= 0:
                                return -np.inf
                        else:
                            if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 4] <= 0:
                                return -np.inf
                        
                if self.modflags['pl']=='dbr':
                    # NB theta[-3], theta[-2], theta[-1] = alpha2(ME), pc_br1(H-M), delta1(H-M)
                    alpha2, pc_br1, delta1 = theta[-3], theta[-2], theta[-1]
                    # alpha2(ME) should be positive (negative slope)
                    # if theta[-3] <=0:
                    if alpha2 <= 0:
                        return -np.inf
                    
                    # alpha3(LE) should be negative (positive slope)
                    alpha3 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 2]
                    # if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 2] >=0:
                    if alpha3 >=0:
                        return -np.inf
                    
                    # alpha1(HE) should be larger (steeper) than alpha2(ME)
                    alpha1 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1]
                    # if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1] <= theta[-3]:
                    if alpha1 <= alpha2:
                        return -np.inf
                    
                    # break momentum 1 (H-M) should be larger than break momentum 2 (M-L)
                    # We may want to share the break among spices in rigidity (rather than momentum). So rig_br also calculated 
                    pc_br2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3]
                    massNumber = ph.M_DICT[self.LISorder[i].lower()]
                    atomNumber = ph.Z_DICT[self.LISorder[i].lower()]
                    rig_br1 = pc_br1/atomNumber # in MV
                    rig_br2 = pc_br2/atomNumber # in MV
                    if pc_br1 < pc_br2:
                        return -np.inf
                    
                    # delta1(H-M) and delta2(M-L) above 0
                    delta2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 4]
                    if delta1 <= 0:
                        return -np.inf
                    if delta2 <=0:
                        return -np.inf

                if self.modflags['pl']=='dbr2':
                    # NB theta[-3], theta[-2], theta[-1] = alpha2(ME), rig_br1(H-M), delta1(H-M)
                    alpha2, rig_br1, delta1 = theta[-3], theta[-2], theta[-1]
                    # alpha2(ME) should be positive (negative slope)
                    # if theta[-3] <=0:
                    if alpha2 <= 0:
                        return -np.inf
                    
                    # alpha3(LE) should be negative (positive slope)
                    alpha3 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 2]
                    # if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 2] >=0:
                    if alpha3 >=0:
                        return -np.inf
                    
                    # alpha1(HE) should be larger (steeper) than alpha2(ME)
                    alpha1 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1]
                    # if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1] <= theta[-3]:
                    if alpha1 <= alpha2:
                        return -np.inf
                    
                    # break rigidity 1 (H-M) should be larger than break rigidity 2 (M-L)
                    rig_br2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3]
                    if rig_br1 < rig_br2:
                        return -np.inf
                    
                    # delta1(H-M) and delta2(M-L) above 0
                    delta2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 4]
                    if delta1 <= 0:
                        return -np.inf
                    if delta2 <=0:
                        return -np.inf


                # force params to be within limits from Strong 2015 for broken power-law model.
                #if self.modflags['priorlimits'] and self.modflags['pl']=='br' and self.LISorder[i].lower()=='h':
                if self.modflags['priorlimits'] and self.modflags['pl']=='br':
                    
                    # limit for normalization, only for hydrogen. 
                    # c/4pi n_ref,100GeV/n between ~1e-9 and ~20e-9 (# /cm^2 /s /sr /MeV). This translates into 5-100 at 10 GeV/n in (#/s/m2/sr/GeV) for index=2.7.
                    # => 22.5 to 27.5 (close to best-fit value by Honda+04)
                    if (self.LISorder[i].lower()=='h'):
                        norm = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams]
                        if norm < 22.5 or norm  > 27.5:
                            return -np.inf
                    
                    # alpha1 between 3.5 and 2.6 => 2.7 to 3.0 (close to best-fit value by Honda+04)
                    alpha1 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1] 
                    if alpha1 < 2.7 or alpha1> 3.0:
                        return -np.inf
                    
                    # alpha2 between 2.7 and 2.2
                    alpha2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 2] 
                    if alpha2 < 2.2 or alpha2 > 2.7:
                        return -np.inf
                    
                    # alpha2 smaller than alpha1 (i.e., flatter)
                    if alpha2 > alpha1:
                        return -np.inf

                    # break momentum between 1e3 and 1e4 MeV
                    # We may want to share the break among spices in rigidity (rather than momentum). So rig_br also calculated 
                    pc_br = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3]
                    massNumber = ph.M_DICT[self.LISorder[i].lower()]
                    atomNumber = ph.Z_DICT[self.LISorder[i].lower()]
                    rig_br = pc_br/atomNumber # in MV
#                    if (self.LISorder[i].lower() == 'he'):
#                      print ("###", self.LISorder[i].lower(), E_br, p_br, pc_br, rig_br)
                    if pc_br < 1e3 or pc_br > 1e4:
                        return -np.inf
                    
                    # delta between 0.05 and 2.0
                    if not self.modflags['one_d']:
                        delta = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 4] 
                        if delta < 0.05 or delta > 2.0:
                            return -np.inf
                    elif self.modflags['fixd']==None:
                        delta = theta[-1]
                        if delta < 0.05 or delta > 2.0: return -np.inf

                # force params to be within limits from Strong 2015 for double-broken power-law model.
                if self.modflags['priorlimits'] and self.modflags['pl']=='dbr':
                    
                    # limit for normalization, only for hydrogen. 
                    # c/4pi n_ref,100GeV/n between ~1e-9 and ~20e-9 (# /cm^2 /s /sr /MeV). This translates into 5-100 at 10 GeV/n in (#/s/m2/sr/GeV) for index=2.7.
                    # => 22.5 to 27.5 (close to best-fit value by Honda+04)
                    if (self.LISorder[i].lower()=='h'):
                        norm = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams]
                        if norm < 22.5 or norm  > 27.5:
                            return -np.inf
                    
                    # alpha1(HE) between 3.5 and 2.6 => 2.7 to 3.0 (close to best-fit value by Honda+04)
                    alpha1 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1] 
                    if alpha1 < 2.7 or alpha1> 3.0:
                        return -np.inf
                    
                    # alpha2(ME) between 2.7 and 2.2
                    alpha2 = theta[-3]
                    if alpha2 < 2.2 or alpha2 > 2.7:
                        return -np.inf
                    
                    # break momentum 1 (H-M) between pc_th(1e3) to 1e4 MeV, and
                    # break momentum 2 (M-L) between 1e2 MeV to rig_th(1e3 MeV)
                    # We may want to share the break among spices in rigidity (rather than momentum). So rig_br also calculated 
                    pc_br1 = theta[-2]
                    pc_br2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3]
                    massNumber = ph.M_DICT[self.LISorder[i].lower()]
                    atomNumber = ph.Z_DICT[self.LISorder[i].lower()]
                    rig_br1 = pc_br1/atomNumber # in MV
                    rig_br2 = pc_br2/atomNumber # in MV
                    pc_th = 1e3
                    if pc_br1 < pc_th or pc_br1 > 1e4:
                        return -np.inf
                    if pc_br2 < 1e2 or pc_br2 > pc_th:
                        return -np.inf
                    
                    # delta1(H-M) and delta2(M-L) between 0.05 and 2.0
                    delta1 = theta[-1]
                    delta2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 4]
                    if  delta1 < 0.05 or delta1 > 2.0:
                        return -np.inf
                    if  delta2 < 0.05 or delta2 > 2.0:
                        return -np.inf
                    
                # force params to be within limits from Strong 2015 for double-broken power-law model.
                if self.modflags['priorlimits'] and self.modflags['pl']=='dbr2':
                    
                    # limit for normalization, only for hydrogen. 
                    # c/4pi n_ref,100GeV/n between ~1e-9 and ~20e-9 (# /cm^2 /s /sr /MeV). This translates into 5-100 at 10 GeV/n in (#/s/m2/sr/GeV) for index=2.7.
                    # => 22.5 to 27.5 (close to best-fit value by Honda+04)
                    if (self.LISorder[i].lower()=='h'):
                        norm = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams]
                        if norm < 22.5 or norm  > 27.5:
                            return -np.inf
                    
                    # alpha1(HE) between 3.5 and 2.6 => 2.7 to 3.0 (close to best-fit value by Honda+04)
                    alpha1 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1] 
                    if alpha1 < 2.7 or alpha1> 3.0:
                        return -np.inf
                    
                    # alpha2(ME) between 2.7 and 2.2
                    alpha2 = theta[-3]
                    if alpha2 < 2.2 or alpha2 > 2.7:
                        return -np.inf
                    
                    # break rigidity 1 (H-M) between rig_th(1e3) to 1e4 MV, and
                    # break rigidity 2 (M-L) between 1e2 MV to righ_th(1e3 MV)
                    rig_br1 = theta[-2]
                    rig_br2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3]
                    rig_th = 2e3
                    if rig_br1 < rig_th or rig_br1 > 1e4:
                        return -np.inf
                    if rig_br2 < 1e2 or rig_br2 > rig_th:
                        return -np.inf
                    
                    # delta1(H-M) and delta2(M-L) between 0.05 and 2.0
                    delta1 = theta[-1]
                    delta2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 4]
                    if  delta1 < 0.05 or delta1 > 2.0:
                        return -np.inf
                    if  delta2 < 0.05 or delta2 > 2.0:
                        return -np.inf
                    
            return lp_scale(theta)
        
        
        # CREATE LNPROB FUNCTION
        
        def lnprob(theta):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf

            ll=lnlike(theta)
            if not np.isfinite(ll):
                return -np.inf
            
            return lp+ll
            
        # to be directly used by Model instances
        self.lnlike=lnlike
        self.lnprior=lnprior
        self.lnprob=lnprob
        self.crfunc=crfunc
        self.crfunc_IS=crfunc_IS
        self.grfunc_pp=grfunc_pp
        self.grfuncs=grfuncs
        self.enhfuncs=enhfuncs
        # self.enhfunc=enhfunc
        self.ebrfunc=ebrfunc
        return
        
        
    # default start positions for the MCMC sampler
    def get_startpos(self):
        
        # add phis & scaling parameters, interlaced
        NormScale = 0.001 # fractional variation, equivalent to 1+np.random.normal(scale=0.001)
        if self.modflags['crscaling']:
            
            startpos = np.empty((self.phis.size + self.scales.size,), dtype=self.phis.dtype)
            startpos_err = np.empty((self.phis.size + self.scales.size,), dtype=self.phis.dtype)
            startpos[0::2] = self.phis
            startpos[1::2] = self.scales
            startpos_err[0::2] = self.phis*NormScale
            #startpos_err[1::2] = self.scales*NormScale
            startpos_err[1::2] = self.scaleRanges*NormScale
            
            startpos = list(startpos)
            startpos_err = list(startpos_err)
            
        else: #or just phis if no scaling
            startpos=list(self.phis)
            startpos_err=list(self.phis*NormScale)
        
        # Voyager phi may be set to 0, so we give absolute value for variation (if fractional variation is given, the value does not change during the fit)
        for i in range(self.nphis):
            if 'voyager' in self.CRexps[self.match_inds[i][0]] and '2012' in self.CRexps[self.match_inds[i][0]]:
                if self.modflags['fix_vphi']==None: # if voyager phis are treated as free parameter
                    if self.modflags['crscaling']:
                        startpos[2*i] = 0.
                        startpos_err[2*i] = 0.001 # absolute value
                    else:
                        startpos[i] = 0.
                        startpos_err[i] = 0.001 # absolute value
                else: # if voyager phi is fixed to a specific common value
                    if self.modflags['crscaling']:
                        startpos[2*i] = self.vphi0
                        startpos_err[2*i] = 0.001 # absolute value (lnprior will virtually fix the param)
                    else:
                        startpos[i] = self.vphi0
                        startpos_err[i] = 0.001 # absolute value (lnpriro will virtually fix the param)

        # add gr scaling factor
        if self.modflags['grscaling']:
            startpos += [1.]
            startpos_err += [1.*NormScale]
            # startpos += [1.25] # nominal value for a hybrid model (Kamae +06 and AAgrag)

        # add LIS parameters
        for i in range(self.LISorder.shape[0]):
            startpos += ph.LIS_DICT[self.LISorder[i]] #initialize at former best-fit LIS norm, index
            v = ph.LIS_DICT[self.LISorder[i]]
            for k in range(len(v)):
                startpos_err += [v[k]*NormScale]

            if self.modflags['pl']=='b':                
                # add ~best-fit value from previous fit
                startpos += [2.5]
                startpos_err += [2.5*NormScale]
                
            elif self.modflags['pl']=='br':                
                # add ~best-fit proton values from Strong 2015 (ICRC)
                ## old code (for reference) where E_br was used instaed of pc_br
                ## startpos+=[2.37, ph.p_to_E(5870./ph.C_SI, 1)]
                startpos += [2.37, 5870]
                startpos_err += [2.37*NormScale, 5870*NormScale]
                # add delta if each experiment gets their own
                if not self.modflags['one_d']:
                    startpos += [0.5]
                    startpos_err += [0.5*NormScale]

            elif self.modflags['pl']=='dbr':
                # alpha3(ML) = -3.0, pc_br2 = 0.5 GeV, delta2(M-L) = 0.5
                startpos += [-3.0, 500., 0.5]
                startpos_err += [-3.0*NormScale, 500.*NormScale, 0.5*NormScale]
        
            elif self.modflags['pl']=='dbr2':
                # alpha3(ML) = -3.0, rig_br2 = 0.5 GeV, delta2(M-L) = 0.5
                startpos += [-3.0, 500., 0.5]
                startpos_err += [-3.0*NormScale, 500.*NormScale, 0.5*NormScale]
        
        # add universal delta (Strong 2015 ICRC)
        if self.modflags['pl']=='br' and self.modflags['one_d']:
            if self.modflags['fixd']==None:
                startpos += [0.5]
                startpos_err += [0.5*NormScale]

        # add common alpha2(M)=2.37, pc_br1(H-M)=5870 MeV, delta1 = 0.5
        if self.modflags['pl']=='dbr':
            startpos += [2.37, 5870., 0.5]
            startpos_err += [2.37*NormScale, 5870.*NormScale, 0.5*NormScale]
        
        # add common alpha2(M)=2.37, pc_br1(H-M)=5870 MeV, delta1 = 0.5
        if self.modflags['pl']=='dbr2':
            startpos += [2.37, 5870., 0.5]
            startpos_err += [2.37*NormScale, 5870.*NormScale, 0.5*NormScale]
        
        if self.modflags['crscaling']:
            startpos_err[self.AMS02_inds[0]*2+1] = 0.0 # to make AMS02 scaling always to 1.0
        print ("## startpos, startpos_err =", startpos, startpos_err)
        return np.array(startpos), np.array(startpos_err)
        
        
    # default parameter names
    def get_paramnames(self):
        LISparams={'s': ['norm','alpha1'],
                   'b': ['norm','alpha1','alpha'],
                   'br': ['norm','alpha1','alpha2', 'Pbr'],
                   'dbr': ['norm', 'alpha1', 'alpha3', 'Pbr2', 'delta2'],
                   'dbr2': ['norm', 'alpha1', 'alpha3', 'Rbr2', 'delta2']}
        
        paramnames=[]
        
        # phis, scaling
        for i in range(self.nphis):
            paramnames+=[self.CRels[self.match_inds[i][0]]+'_'+self.CRexps[self.match_inds[i][0]]+'_phi']
            if self.modflags['crscaling']:
                paramnames+=[self.CRels[self.match_inds[i][0]]+'_'+self.CRexps[self.match_inds[i][0]]+'_cr_scale']
            
        if self.modflags['grscaling']:
            paramnames+=['gr_scale']
            
        # LIS params
        for i in range(self.LISorder.shape[0]):
            paramnames+=[self.LISorder[i]+'_'+x for x in LISparams[self.modflags['pl']]]
            
            if self.modflags['pl']=='br' and not self.modflags['one_d']:
                paramnames+=[self.LISorder[i]+'_delta']
        
        if self.modflags['pl']=='br' and self.modflags['one_d']:
            if self.modflags['fixd']==None:
                paramnames+=['delta']

        if self.modflags['pl']=='dbr':
            paramnames+=['alpha2', 'Pbr1', 'delta1']

        if self.modflags['pl']=='dbr2':
            paramnames+=['alpha2', 'Rbr1', 'delta1']

        # print ("## paramnames=", paramnames)
        return np.array(paramnames)
    
