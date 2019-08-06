# configuration of the Model class, which is typically created within run.py.
# Each Model object initializes its own lnlike, lnprior, and lnprob functions
#  which are used by the emcee sampler.
# Much of the code below consists of re-formatting the data & parameter lists
#  so the MCMC helper functions run as efficiently as possible.

import numpy as np

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
        self.GRexps=[]
        self.GRdata=[]
        self.VRinds=[]
        for elkey in self.data: #loop thru elements
            for expkey in self.data[elkey]: #loop thru experiments
                
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
        self.GRexps=np.array(self.GRexps)
        
        # whether LIS params will have norm, alpha1, and alpha 2
        # or just norm, alpha1
        if self.modflags['pl']=='s':
            self.nLISparams=2
        elif self.modflags['pl']=='b':
            self.nLISparams=3
        else:
            print('Must give valid CR model flag')
            sys.exit()
            
        self.nphis=self.phis.shape[0]
        self.npoints = self.nCRpoints + self.nVRpoints + self.nGRpoints
        
        # determine order of the elements' LIS parameters
        self.LISorder=np.unique(self.CRels)
        self.LISdict={}
        for i in range(self.LISorder.shape[0]):
            self.LISdict[self.LISorder[i]] = i
            
        self.nels = self.LISorder.shape[0]
            
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
                    # should the extra *ph.M_DICT[subkey] be here??
                    current_el+=[ph.E_to_p(self.GREs[i]*ph.M_DICT[subkey], ph.M_DICT[subkey])]
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
        
        # create list of model CR fluxes, same order as data ones
        def crfunc(theta):
            crflux_ls=[]
            for i in range(self.nphis):
                
                # use that element's LIS parameters, and that experiment's phi, to get CR flux
                LIS_params=theta[self.nphis + self.LISdict[self.CRels[i]]*self.nLISparams:\
                                 self.nphis + (self.LISdict[self.CRels[i]] + 1)*self.nLISparams]
                crflux_ls+=[self.crformula(LIS_params, theta[i], self.CREs[i], self.CRZs[i], self.CRMs[i])]
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
                    self.CRfluxes[key] += [enf.Honda_LIS(enf.LIS_params[key], self.GREs[i])]
        
        # empty flux array (same shame as in GRdata)
        self.empty_fluxes=[np.zeros(self.GREs[i].shape) for i in range(len(self.GREs))]
        
        # get enhancement factor at GR energies
        def enhfunc(theta):
            
            CRfluxes_theta = self.CRfluxes.copy()
            CRinds_theta = self.CRinds.copy()
            
            # add theta values to CRfluxes/inds
            for key in CRfluxes_theta:
                
                # if these CR LIS's are included in model (not retrieving old values)
                if CRfluxes_theta[key] == None:
                    
                    if key not in ['cno', 'mgsi']: #if the bin contains only 1 element
                        LIS_params=theta[self.nphis + self.LISdict[key]*self.nLISparams:\
                                         self.nphis + (self.LISdict[key] + 1)*self.nLISparams]
                        CRinds_theta[key] = -LIS_params[1] #enhancement factors given w.r.t. positive spectral ind.
                        
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
                            LIS_params=theta[self.nphis + self.LISdict[el]*self.nLISparams:\
                                             self.nphis + (self.LISdict[el] + 1)*self.nLISparams]
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
            LIS_params_pp = theta[self.nphis + self.nLISparams*self.LISdict['h']:\
                                  self.nphis + self.nLISparams*(self.LISdict['h']+1)]
            return grf.get_fluxes_pp(LIS_params_pp, self.GRlogEgs, self.crformula_IS)
        
        # retrieve e-bremss values at desired energies
        def ebrfunc():
            return ebrf.get_fluxes(self.GRlogEgs)
        
        # e-bremss value doesn't depend on model, so calculate it only once here
        ebr_fluxes = ebrfunc()
        
        
        # CREATE LNLIKE FUNCTION
        
        def lnlike(theta):
            
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
            if modflags['weights']!=None:
                crlike = crlike*modflags['weights'][0]/self.nCRpoints
                vrlike = vrlike*modflags['weights'][1]/self.nVRpoints
                grlike = grlike*modflags['weights'][2]/self.nGRpoints
            
            # sum up weighted contributions
            return crlike + vrlike + grlike
        
        
        # CREATE LNPRIOR FUNCTION
        
        # force spectral indices to be negative, normalizations to be positive
        if self.modflags['priors']==0: #gaussian priors on phis
            
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
                if theta[self.nphis + i*self.nLISparams + 1] >= 0 or\
                   theta[self.nphis + i*self.nLISparams] < 0:
                    return -np.inf
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
        startpos=list(self.phis+np.random.normal(scale=1e-4))
        
        # prevent Voyager data from starting at very small region around 0
        for i in range(len(startpos)):
            if abs(startpos[i])<1: startpos[i]=startpos[i]/abs(startpos[i])
        
        for i in range(self.LISorder.shape[0]):
            startpos+=ph.LIS_DICT[self.LISorder[i]] #initialize at former best-fit LIS norm, index

            if self.modflags['pl']=='b':
                startpos+=[2.5]
        
        return np.array(startpos)
        
        
    # default parameter names
    def get_paramnames(self):
        LISparams=['norm','alpha1','alpha2']
        
        paramnames=[]
        for i in range(self.CRexps.shape[0]):
            paramnames+=[self.CRels[i]+'_'+self.CRexps[i]+'_phi']
        for i in range(self.LISorder.shape[0]):
            for j in range(self.nLISparams):
                paramnames+=[self.LISorder[i]+'_'+LISparams[j]]
        
        return np.array(paramnames)
    