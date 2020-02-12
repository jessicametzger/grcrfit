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
# - 'weights' = 3-item list giving relative weights of CR, Voyager, and GR log-like.
#    Note that priors aren't weighted, so may need to tune weight normalization to get
#    correct relative weighting between likelihood & prior. If weights==None, no weighting
# - 'priors' = 0 or 1, for gaussian (0) or flag (1) priors on phis
# - 'cr/grscaling' = True or False, whether or not to scale all CR experiments  except AMS-02 
#    or all GR experiments to correct for systematic errors
# - 'priorlimits' = True or False, whether or not to constrain the parameter in specified ranges
# - 'one_d' = True or False, whether we use single value or multiple values for delta (sharpness of the break)
# - 'fixd': If "None" the delta is treated as a free parameter. If some number is given, deita is fixed to that value.

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
        self.CRels=[]
        self.CRexps=[]
        self.CRdata=[]
        self.CRZs=[]
        self.CRMs=[]
        self.CRtimes=[]
        self.GRexps=[]
        self.GRdata=[]
        self.VRinds=[]
        
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
                    self.CRdata+=[self.data[elkey][expkey][0]]
                    self.CRZs+=[ph.Z_DICT[elkey.lower()]]
                    self.CRMs+=[ph.M_DICT[elkey.lower()]]
                    self.CRtimes+=[self.data[elkey][expkey][1]]
                    
                    # will need to distinguish voyager data for likelihood weighting
                    if 'voyager1' in expkey.lower() and '2012' in expkey:
                        self.nVRpoints+=self.data[elkey][expkey][0].shape[0]
                        self.VRinds+=[len(self.CRdata)-1]
                    else:
                        self.nCRpoints+=self.data[elkey][expkey][0].shape[0]
        
        # make everything a np array
        self.phis=np.array(self.phis).astype(np.float)
        self.phierrs=np.array(self.phierrs).astype(np.float)
        self.CRZs=np.array(self.CRZs).astype(np.float)
        self.CRMs=np.array(self.CRMs).astype(np.float)
        self.CRels=np.array(self.CRels)
        self.CRexps=np.array(self.CRexps)
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
            self.match_inds += [duplicates]
        
        # remove duplicates:
        self.phis = np.delete(self.phis, self.remove_inds)
        self.phierrs = np.delete(self.phierrs, self.remove_inds)
        self.phitimes = np.delete(self.phitimes, self.remove_inds)
        
        # also create list of ref. scaling factors & errors (all 1, .1)
        # corresponding with the list of phi's, for prior calculation.
        # Don't ignore AMS-02, scaling will be set to 1 anyways.
        if self.modflags['crscaling']:
            self.scales = np.repeat(1, self.phis.shape[0])
            self.scaleerrs = np.repeat(.1, self.phis.shape[0])
        
        
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
        # or norm, alpha1, alpha2, p_br ('br').
        if self.modflags['pl']=='s':
            self.nLISparams=2
        elif self.modflags['pl']=='b':
            self.nLISparams=3
        elif self.modflags['pl']=='br':
            if self.modflags['one_d']: self.nLISparams=4
            else: self.nLISparams=5
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
                    
                    # GR energy is some factor smaller than CR energy; take from Mori 1997
                    factors=np.repeat(10,self.GREs[i].shape[0])
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
                    
                    if self.modflags['pl']=='br' and self.modflags['one_d']: #add universal delta parameter
                        if self.modflags['fixd']==None:
                            LIS_params+=[theta[-1]]
                        else:
                            LIS_params+=[self.modflags['fixd']]
                        
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
            
        # compile old LIS fluxes, inds (if not included in model) for enhancement factor calculation
        self.CRfluxes={'h': None, 'he': None, 'cno': None, 'mgsi': None, 'fe': None}
        self.CRinds={'h': None, 'he': None, 'cno': None, 'mgsi': None, 'fe': None}
        self.fit_els=[x.lower() for x in list(self.LISorder)]
        for key in enf.enh_els_ls:
            
            # if need to use old LIS data
            if not all(x in self.fit_els for x in enf.enh_els[key]):
                
                # add old spectral index
                self.CRinds[key] = enf.LIS_params[key][0]
                
                # add old fluxes
                self.CRfluxes[key] = []
                for i in range(len(self.GRdata)):
                    mean_mass = np.mean([ph.M_DICT[x] for x in enf.enh_els[key]])
                    
                    factors=np.repeat(10.,self.GREs[i].shape[0])
                    self.CRfluxes[key] += [enf.Honda_LIS(enf.LIS_params[key], self.GREs[i]*factors)]
        
        # empty flux array (same shame as in GRdata)
        self.empty_fluxes=[np.zeros(self.GREs[i].shape) for i in range(len(self.GREs))]
        
        # get enhancement factor at GR energies
        def enhfunc(theta):
            
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
            
            enh_fs = [np.copy(x) for x in self.empty_fluxes]
            enh_fs = enf.enh(self.modflags['enh'], self.modflags['enhext'], enh_fs, self.GREs, CRfluxes_theta, CRinds_theta)
            
            return enh_fs
            
        # get GR fluxes at log10 of GR energies
        # can also consider some modflag options here
        self.GRlogEgs=[np.log10(self.GREs[i]) for i in range(len(self.GREs))]
        def grfunc_pp(theta):
            LIS_params_pp = list(theta[self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*self.LISdict['h']:\
                                       self.ncrparams + int(self.modflags['grscaling']) + self.nLISparams*(self.LISdict['h']+1)])
            
            #add universal delta parameter
            if self.modflags['pl']=='br' and self.modflags['one_d']:
                if self.modflags['fixd']==None:
                    LIS_params_pp+=[theta[-1]]
                else:
                    LIS_params_pp+=[self.modflags['fixd']]
                
            return grf.get_fluxes_pp(LIS_params_pp, self.GRlogEgs, self.crformula_IS)
        
        # retrieve e-bremss values at desired energies
        def ebrfunc():
            return ebrf.get_fluxes(self.GRlogEgs)
        
        # e-bremss value doesn't depend on model, so calculate it only once here
        ebr_fluxes = ebrfunc()
        
        
        # CREATE LNLIKE FUNCTION
        
        def lnlike(theta):
            
            # set ams02 scaling to 1
            if self.modflags['crscaling']:
                theta=np.array(theta)
                theta[2*np.array(self.AMS02_inds) + 1] = 1.
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
                enh_f = enhfunc(theta)

                # get p-p GR fluxes at gamma data's energies
                gr_fluxes = grfunc_pp(theta)
                
                # make sure it's a list of finite numbers
                try:
                    for i in range(len(gr_fluxes)):
                        if not (np.all(np.isfinite(gr_fluxes[i])) and np.all(np.isfinite(enh_f[i]))):
                            return -np.inf
                except: return -np.inf

                # enhance p-p GR fluxes & add e-bremss data
                for i in range(len(self.GRdata)):
                    gr_fluxes[i] = enh_f[i]*gr_fluxes[i] + ebr_fluxes[i]
                    
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
            
            def lp_phi(theta):
                lp=0
                
                # priors on phis; must account for index interlacing if crscaling
                lp += -.5*np.sum(((theta[0:self.ncrparams:(int(self.modflags['crscaling'])+1)] - self.phis)/self.phierrs)**2.)
                    
                #scaling factors
                if self.modflags['crscaling']:
                    lp += -.5*np.sum(((theta[1:self.ncrparams:2] - self.scales)/self.scaleerrs)**2.)
                if self.modflags['grscaling']:
                    lp += -.5*np.sum(((theta[self.ncrparams] - 1)/0.05)**2.)
                    
                return lp
            
        # no gaussian priors
        elif self.modflags['priors']==1:
            
            def lp_phi(theta):
                return 0
            
        else:
            print("Must give valid 'priors' entry in modflags.")
            sys.exit()
        
        
        # set hard limits on parameters
        # also add gaussian priors
        def lnprior(theta):
            
            # positive phis
            if np.any(theta[0:self.ncrparams:(int(self.modflags['crscaling'])+1)] <= 0):
                return -np.inf
            
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
                    
                    # break energy between 0 and 300 GeV
                    if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3] <= 0 or\
                       theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3] >= 3e5:
                        return -np.inf
                    
                    # delta above 0
                    if self.modflags['fixd']==None:
                        if self.modflags['one_d']:
                            if theta[-1] <= 0:
                                return -np.inf
                        else:
                            if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 4] <= 0:
                                return -np.inf
                        

                # force params to be within limits from Strong 2015 for broken power-law model.
#                if self.modflags['priorlimits'] and self.modflags['pl']=='br' and self.LISorder[i].lower()=='h':
                if self.modflags['priorlimits'] and self.modflags['pl']=='br':
                    
                    # limit for normalization, only for hydrogen. 
                    # c/4pi n_ref,100GeV/n between ~1e-9 and ~20e-9 (# /cm^2 /s /sr /MeV). This translates into 5-100 at 10 GeV/n in (#/s/m2/sr/GeV) for index=2.7.
                    if (self.LISorder[i].lower()=='h'):
                        norm = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams]
                        if norm < 5 or norm  > 100:
                            return -np.inf
                    
                    # alpha1 between 3.5 and 2.6
                    alpha1 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 1] 
                    if alpha1 < 2.6 or alpha1> 3.5:
                        return -np.inf
                    
                    # alpha2 between 2.7 and 2.2
                    alpha2 = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 2] 
                    if alpha2 < 2.2 or alpha2 > 2.7:
                        return -np.inf
                    # alpha2 smaller than alpha1 (i.e., flatter)
                    if alpha2 > alpha1:
                        return -np.inf
                    
                    # break momentum between 1e3 and 1e5 MeV
                    # We may want to share the break among spices in rigidity (rather than momentum). So rig_br also calculated 
                    E_br = theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 3]
                    massNumber = ph.M_DICT[self.LISorder[i].lower()]
                    atomNumber = ph.Z_DICT[self.LISorder[i].lower()]
                    p_br = ph.E_to_p(E_br, massNumber)
                    pc_br = p_br*ph.C_SI # in MeV
                    rig_br = pc_br/atomNumber # in MV
#                    if (self.LISorder[i].lower() == 'he'):
#                      print ("###", self.LISorder[i].lower(), E_br, p_br, pc_br, rig_br)
                    if pc_br < 1e3 or pc_br > 1e5:
                        return -np.inf
                    
                    # delta between 0.05 and 1.0
                    if not self.modflags['one_d']:
                        if theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 4] < 0.05 or\
                           theta[self.ncrparams + int(self.modflags['grscaling']) + i*self.nLISparams + 4] > 1.0:
                            return -np.inf
                    elif self.modflags['fixd']==None:
                        if theta[-1] < 0.05 or theta[-1] > 1.0: return -np.inf
                    
            return lp_phi(theta)
        
        
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
        self.enhfunc=enhfunc
        self.ebrfunc=ebrfunc
        return
        
        
    # default start positions for the MCMC sampler
    def get_startpos(self):
        
        # add phis & scaling parameters, interlaced
        if self.modflags['crscaling']:
            
            startpos = np.empty((self.phis.size + self.scales.size,), dtype=self.phis.dtype)
            startpos[0::2] = self.phis
            startpos[1::2] = self.scales
            
            startpos = list(startpos)
            
        else: #or just phis if no scaling
            startpos=list(self.phis)
        
        # prevent Voyager phis from starting at very small region around 0
        for i in range(self.nphis):
            if 'voyager' in self.CRexps[self.match_inds[i][0]] and '2012' in self.CRexps[self.match_inds[i][0]]:
                if self.modflags['crscaling']:
                    startpos[2*i] = 10.
                else:
                    startpos[i] = 10.
        
        # add gr scaling factor
        if self.modflags['grscaling']:
            startpos += [1.]
        
        # add LIS parameters
        for i in range(self.LISorder.shape[0]):
            startpos+=ph.LIS_DICT[self.LISorder[i]] #initialize at former best-fit LIS norm, index

            if self.modflags['pl']=='b':
                
                # add ~best-fit value from previous fit
                startpos+=[2.5]
                
            elif self.modflags['pl']=='br':
                
                # add ~best-fit proton values from Strong 2015 (ICRC)
                startpos+=[2.37, ph.p_to_E(5870./ph.C_SI, 1)]
                
                # add delta if each experiment gets their own
                if not self.modflags['one_d']:
                    startpos+=[0.5]
        
        # add universal delta (Strong 2015 ICRC)
        if self.modflags['pl']=='br' and self.modflags['one_d']:
            if self.modflags['fixd']==None:
                startpos+=[0.5]
        
        return np.array(startpos)
        
        
    # default parameter names
    def get_paramnames(self):
        LISparams={'s': ['norm','alpha1'],
                   'b': ['norm','alpha1','alpha'],
                   'br': ['norm','alpha1','alpha2', 'Ebr']}
        
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
        
        return np.array(paramnames)
    
