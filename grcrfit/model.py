# configuration of the Model class, which is typically created within run.py.
# Each Model object initializes its own lnlike, lnprior, and lnprob functions
#  which are used by the emcee sampler.
# Much of the code below consists of re-formatting the data & parameter lists
#  so the MCMC helper functions run as efficiently as possible.

import numpy as np
import copy

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
# - 'scaling' = True or False, whether or not to scale all CR experiments  except AMS-02 
#    to correct for systematic errors
class Model():
    
    # initialize the Model object
    def __init__(self, modflags, data):
        self.modflags = modflags
        self.data = data
        
        try: test=self.modflags['scaling']
        except KeyError: self.modflags['scaling']=False
        
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
            
            duplicates = list(np.where((self.CRtimes == self.CRtimes[i]) & (self.CRexps == self.CRexps[i]))[0])
            duplicates.sort()
            
            # for scaling
            if self.modflags['scaling'] and 'ams02' in self.CRexps[i].lower():
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
        if self.modflags['scaling']:
            self.scales = np.repeat(1, self.phis.shape[0])
            self.scaleerrs = np.repeat(.1, self.phis.shape[0])
        
        
        # determine order of the elements' LIS parameters
        self.LISorder=np.unique(self.CRels)
        self.LISdict={}
        for i in range(self.LISorder.shape[0]):
            self.LISdict[self.LISorder[i]] = i
            
        self.nels = self.LISorder.shape[0]
        self.nphis=self.phis.shape[0]
        self.npoints = self.nCRpoints + self.nVRpoints + self.nGRpoints
        
        # for indexing purposes
        if self.modflags['scaling']:
            self.nexpparams = self.nphis*2
        else:
            self.nexpparams = self.nphis
        
        # whether each el's LIS params will have norm, alpha1, and alpha 2
        # or norm, alpha1, alpha3, p_br
        # or just norm, alpha1
        if self.modflags['pl']=='s':
            self.nLISparams=2
        elif self.modflags['pl']=='b':
            self.nLISparams=3
        elif self.modflags['pl']=='br':
            self.nLISparams=4
        else:
            print('Must give valid CR model flag')
            sys.exit()
            
            
        # DATA CONSOLIDATION
        
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
#                     factors = 10**enf.fac_interp(np.log10(self.GREs[i]))
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
        
        # create list of model CR fluxes, same order as data ones
        def crfunc(theta):
            crflux_ls=[None] * self.CRels.shape[0]
            for i in range(self.nphis):
                
                # loop thru experiments that share that phi
                for index in self.match_inds[i]:
                    
                    # use that element's LIS parameters, and that experiment's phi, to get CR flux
                    LIS_params=list(theta[self.nexpparams + self.LISdict[self.CRels[index]]*self.nLISparams:\
                                          self.nexpparams + (self.LISdict[self.CRels[index]] + 1)*self.nLISparams])
                    if self.modflags['pl']=='br': #add universal delta parameter
                        LIS_params+=[theta[-1]]
                        
                    # account for scaling parameter if needed
                    if self.modflags['scaling']:
                        current_phi = theta[2*i]
                        current_scale = theta[2*i+1]
                    else:
                        current_phi = theta[i]
                        current_scale = 1
                        
                    crflux_ls[index]=self.crformula(LIS_params, current_phi, self.CREs[index], 
                                                   self.CRZs[index], self.CRMs[index])*current_scale
            
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
                        LIS_params=list(theta[self.nexpparams + self.LISdict[key]*self.nLISparams:\
                                              self.nexpparams + (self.LISdict[key] + 1)*self.nLISparams])
                        if self.modflags['pl']=='br': #add universal delta parameter
                            LIS_params+=[theta[-1]]
                        
                        # enhancement factors given w.r.t. positive spectral ind.
                        # if beta pl, this will be momentum (higher-energy) index
                        # if broken pl, this will be first (higher-energy) index
                        CRinds_theta[key] = -LIS_params[1]
                        
                        CRfluxes_theta[key] = []
                        for i in range(len(self.GRdata)):
                            CRfluxes_theta[key] += [self.crformula_IS(LIS_params, self.CRp_atGRE[key][key][i], \
                                                    ph.M_DICT[key]).reshape(self.GREs[i].shape)]
                            
                    else: #if the bin contains multiple elements
                        
                        # alpha will be avg weighted by flux at 1 GeV; fluxes will be sum of fluxes
                        weighted_sum = 0
                        flux_sum = 0
                        fluxes = [np.copy(x) for x in self.empty_fluxes] #must copy np arrays individually
                        for el in enf.enh_els[key]:
                            LIS_params=list(theta[self.nexpparams + self.LISdict[el]*self.nLISparams:\
                                                  self.nexpparams + (self.LISdict[el] + 1)*self.nLISparams])
                            if self.modflags['pl']=='br': #add universal delta parameter
                                LIS_params+=[theta[-1]]
                            
                            flux_1GeV=self.crformula_IS(LIS_params, ph.p1_DICT[el], ph.M_DICT[el])
                            weighted_sum += (-LIS_params[1]*flux_1GeV) #enhancement factors given w.r.t. positive spectral ind.
                            flux_sum += flux_1GeV
                            
                            for i in range(len(self.GREs)):
                                addition = self.crformula_IS(LIS_params, self.CRp_atGRE[key][el][i], ph.M_DICT[el])
                                fluxes[i] += addition.reshape(self.GREs[i].shape)
                        
                        CRfluxes_theta[key] = fluxes
                        CRinds_theta[key] = weighted_sum/flux_sum
            
            enh_fs = [np.copy(x) for x in self.empty_fluxes]
            enh_fs = enf.enh(self.modflags['enh'], enh_fs, self.GREs, CRfluxes_theta, CRinds_theta)
            return enh_fs
            
        # get GR fluxes at log10 of GR energies
        # can also consider some modflag options here
        self.GRlogEgs=[np.log10(self.GREs[i]) for i in range(len(self.GREs))]
        def grfunc_pp(theta):
            LIS_params_pp = list(theta[self.nexpparams + self.nLISparams*self.LISdict['h']:\
                                       self.nexpparams + self.nLISparams*(self.LISdict['h']+1)])
            if self.modflags['pl']=='br': #add universal delta parameter
                LIS_params_pp+=[theta[-1]]
            return grf.get_fluxes_pp(LIS_params_pp, self.GRlogEgs, self.crformula_IS)
        
        # retrieve e-bremss values at desired energies
        def ebrfunc():
            return ebrf.get_fluxes(self.GRlogEgs)
        
        # e-bremss value doesn't depend on model, so calculate it only once here
        ebr_fluxes = ebrfunc()
        
        
        # CREATE LNLIKE FUNCTION
        
        def lnlike(theta):
            
            # set ams02 scaling to 1
            if self.modflags['scaling']:
                theta=np.array(theta)
                theta[2*np.array(self.AMS02_inds) + 1] = 1.
                theta=list(theta)
            
            crlike=0
            vrlike=0
            grlike=0
            
            # create CR fluxes to match all datasets
            cr_fluxes = crfunc(theta)
            for i in range(len(cr_fluxes)):
                if not np.all(np.isfinite(cr_fluxes[i])):
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

                # enhance p-p GR fluxes & add e-bremss data
                for i in range(len(self.GRdata)):
                    gr_fluxes[i] = enh_f[i]*gr_fluxes[i] + ebr_fluxes[i]

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
        
        # force spectral indices to be negative, normalizations to be positive
        if self.modflags['priors']==0: #gaussian priors on phis
            
            if self.modflags['scaling']: #scaling factors
                def lp_phi(theta):
                    lp = -.5*np.sum(((theta[0:self.nexpparams:2] - self.phis)/self.phierrs)**2.)
                    lp+= -.5*np.sum(((theta[1:self.nexpparams:2] - self.scales)/self.scaleerrs)**2.)
                    return lp
            else: #no scaling factors
                def lp_phi(theta):
                    lp= -.5*np.sum(((theta[0:self.nphis] - self.phis)/self.phierrs)**2.)
                    return lp
            
        elif self.modflags['priors']==1: #flat priors on phis
            
            def lp_phi(theta):
                return 0
            
        else:
            print("Must give valid 'priors' entry in modflags.")
            sys.exit()
        
        
        def lnprior(theta):
            
            for i in range(self.nels):
                
                # negative (high-energy, momentum) index, positive LIS norm
                if theta[self.nexpparams + i*self.nLISparams + 1] >= 0 or\
                   theta[self.nexpparams + i*self.nLISparams] < 0:
                    return -np.inf
                
                # positive break energy
                if self.modflags['pl']=='br':
                    if theta[self.nexpparams + i*self.nLISparams + 3] <= 0:
                        return -np.inf
                
            # negative delta
            if self.modflags['pl']=='br':
                if theta[-1]>=0: return -np.inf
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
        self.grfunc_pp=grfunc_pp
        self.enhfunc=enhfunc
        self.ebrfunc=ebrfunc
        return
        
        
    # default start positions for the MCMC sampler
    def get_startpos(self):
        
        # add phis & scaling parameters, interlaced
        if self.modflags['scaling']:
            
            startpos = np.empty((self.phis.size + self.scales.size,), dtype=self.phis.dtype)
            startpos[0::2] = self.phis
            startpos[1::2] = self.scales
            
            startpos = list(startpos)
            
        else: #or just phis if no scaling
            startpos=list(self.phis)
        
        # prevent Voyager phis from starting at very small region around 0
        for i in range(self.nphis):
            if 'voyager' in self.CRexps[self.match_inds[i][0]] and '2012' in self.CRexps[self.match_inds[i][0]]:
                if self.modflags['scaling']:
                    startpos[2*i] = 10.
                else:
                    startpos[i] = 10.
        
        # add LIS parameters
        for i in range(self.LISorder.shape[0]):
            startpos+=ph.LIS_DICT[self.LISorder[i]] #initialize at former best-fit LIS norm, index

            if self.modflags['pl']=='b':
                
                # add ~best-fit value from previous fit
                startpos+=[2.5]
                
            elif self.modflags['pl']=='br':
                
                # add best-fit proton values from Strong 2015 (ICRC)
                startpos+=[-2.37,5870.]
        
        # add delta (Strong 2015 ICRC)
        if self.modflags['pl']=='br':
            startpos+=[-.5]
        
        return np.array(startpos)
        
        
    # default parameter names
    def get_paramnames(self):
        LISparams={'s': ['norm','alpha1'],
                   'b': ['norm','alpha1','alpha2'],
                   'br': ['norm','alpha1','alpha3', 'Ebr']}
        
        paramnames=[]
        
        # phis, scaling
        for i in range(self.nphis):
            paramnames+=[self.CRels[self.match_inds[i][0]]+'_'+self.CRexps[self.match_inds[i][0]]+'_phi']
            if self.modflags['scaling']:
                paramnames+=[self.CRels[self.match_inds[i][0]]+'_'+self.CRexps[self.match_inds[i][0]]+'_scale']
            
        # LIS params
        for i in range(self.LISorder.shape[0]):
            paramnames+=[self.LISorder[i]+'_'+x for x in LISparams[self.modflags['pl']]]
        
        if self.modflags['pl']=='br':
            paramnames+=['delta']
        
        return np.array(paramnames)
    
