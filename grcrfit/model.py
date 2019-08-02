import numpy as np
import time

from . import helpers as h
from . import physics as ph
from . import crfluxes as crf
from . import grfluxes as grf
from . import enhs as enf

# each run will create its own custom "Model" object
# containing the helper functions for the MCMC

class Model():
    
    # modflags is a dictionary. 'pl' key returns power-law type ('s' for single or 'b' for beta), 
    # 'enh' key returns which enhancement framework to use (1 for QGS and 2 for LHC)
    # 'weights' key returns 3-item list of weights (CR, Voyager, GR); only relative weight matters
    def __init__(self, modflags, data, priors = {'phi': 0, 'vphi': 0}):
        self.modflags = modflags
        self.data = data
        self.priors = priors
        
        # if parallel, these will need to be pickled
        global lnlike, lnprior, lnprob
        
        # count number of points for weighting
        self.nCRpoints=0
        self.nVRpoints=0
        self.nGRpoints=0
        
        # Basically, compile all the custom data for MCMC helper functions.
        # - get Usoskin+ reference phi array from data for priors
        # - get IDs, labels, etc. all in same order
        # - make list of each dataset's atomic number & mass
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
        for elkey in self.data:
            for expkey in self.data[elkey]:
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
                    
                    if 'voyager1' in expkey.lower() and '2012' in expkey:
                        self.nVRpoints+=self.data[elkey][expkey][0].shape[0]
                        self.VRinds+=[len(self.CRdata)-1]
                    else:
                        self.nCRpoints+=self.data[elkey][expkey][0].shape[0]
                
        self.phis=np.array(self.phis).astype(np.float)
        self.phierrs=np.array(self.phierrs).astype(np.float)
        self.CRZs=np.array(self.CRZs).astype(np.float)
        self.CRMs=np.array(self.CRMs).astype(np.float)
        self.CRels=np.array(self.CRels)
        self.CRexps=np.array(self.CRexps)
        self.GRexps=np.array(self.GRexps)
        
        self.phi_count=self.phis.shape[0]
        
        self.npoints = self.nCRpoints + self.nVRpoints + self.nGRpoints
        
        # determine order of the elements' LIS parameters
        self.LISorder=np.unique(self.CRels)
        self.LISdict={}
        for i in range(self.LISorder.shape[0]):
            self.LISdict[self.LISorder[i]] = i
        
        # create list of bare-minimum CR data arrays (only bin average, single err)
        for i in range(len(self.CRdata)):
            
            # errorbars: rms of sys/stat, then avg of upper/lower
            errs = (np.sqrt(self.CRdata[i][:,-4]**2 + self.CRdata[i][:,-2]**2) + \
                    np.sqrt(self.CRdata[i][:,-3]**2 + self.CRdata[i][:,-1]**2))/2.
            
            # remove all but most impt cols: E [MeV], flux, err
            self.CRdata[i] = self.CRdata[i][:,np.array([0,3])]
            self.CRdata[i] = np.append(self.CRdata[i], np.array([errs]).transpose(), axis=1)
            
        # create list of bare-minimum GR data arrays (only bin average, single err)
        for i in range(len(self.GRdata)):
            
            # errorbars: rms of sys/stat, then avg of upper/lower
            errs = (np.sqrt(self.GRdata[i][:,-4]**2 + self.GRdata[i][:,-2]**2) + \
                    np.sqrt(self.GRdata[i][:,-3]**2 + self.GRdata[i][:,-1]**2))/2.
            
            # remove all but most impt cols: E [MeV], emissivity, err
            self.GRdata[i] = self.GRdata[i][:,np.array([0,3])]
            self.GRdata[i] = np.append(self.GRdata[i], np.array([errs]).transpose(), axis=1)
            
            # change y-axis to emissivity (from emissivity*1e24)
            self.GRdata[i][:,1:] = self.GRdata[i][:,1:]*1e-24
        
        
        # CR FLUXES
        
        # choose a CR flux formula based on the flagged one
        if self.modflags['pl']=='s':
            self.crformula = crf.flux_spl
            self.crformula_IS = crf.flux_spl_IS
            self.LISparamcount=2
        elif self.modflags['pl']=='b':
            self.crformula = crf.flux_bpl
            self.crformula_IS = crf.flux_bpl_IS
            self.LISparamcount=3
        else:
            print('Must give valid CR model flag')
            sys.exit()
        
        # create list of model CR fluxes, same order as data ones
        def crfunc(theta):
            crflux_ls=[]
            for i in range(self.phi_count):
                
                LIS_params=theta[self.phi_count + self.LISdict[self.CRels[i]]*self.LISparamcount:\
                                 self.phi_count + (self.LISdict[self.CRels[i]] + 1)*self.LISparamcount]
                
                # use that element's LIS parameters, and that experiment's phi, to get CR flux
                crflux_ls+=[self.crformula(LIS_params, theta[i], self.CRdata[i][:,0], self.CRZs[i], self.CRMs[i])]
            return crflux_ls
        
        
        # ENHANCEMENT FACTOR
        
        # CR momenta at GR energies (for enhancement calculation)
        # shape: 5-item dict (corresponding to 5 categories of elements for enh routine), 
        # each containing a dict which gives GRdata-shaped list at each entry corresponding to that element
        self.CRp_atGRE = {}
        for key in enf.enh_els:
            self.CRp_atGRE[key]={}
            for subkey in enf.enh_els[key]:
                current_el=[]
                for j in range(len(self.GRdata)):
                    current_el+=[ph.E_to_p(self.GRdata[j][:,0], ph.M_DICT[subkey])]
                self.CRp_atGRE[key][subkey] = current_el
            
        # compile old LIS fluxes, inds
        self.CRfluxes={'h': None, 'he': None, 'cno': None, 'mgsi': None, 'fe': None}
        self.CRinds={'h': None, 'he': None, 'cno': None, 'mgsi': None, 'fe': None}
        self.fit_els=[x.lower() for x in list(self.LISorder)]
        for key in self.CRfluxes:
            
            # if need to use old LIS data
            if not all(x in self.fit_els for x in enf.enh_els[key]):
                self.CRinds[key] = enf.LIS_params[key][0]
                
                self.CRfluxes[key] = []
                for i in range(len(self.GRdata)):
                    self.CRfluxes[key] += [enf.Honda_LIS(enf.LIS_params[key], self.GRdata[i][:,0])]
        
        self.empty_fluxes=[np.zeros(self.GRdata[i][:,0].shape) for i in range(len(self.GRdata))]
        self.flux_sum = {'cno': ph.p1_DICT['c'] + ph.p1_DICT['n'] + ph.p1_DICT['o'],
                        'mgsi': ph.p1_DICT['mg'] + ph.p1_DICT['ne'] + ph.p1_DICT['s'] + ph.p1_DICT['si']}
        
        # get enhancement factor at GR energies
        def enhfunc(theta):
            
            CRfluxes_theta = self.CRfluxes.copy()
            CRinds_theta = self.CRinds.copy()
            
            # add theta values to CRfluxes/inds
            for key in CRfluxes_theta:
                if CRfluxes_theta[key] == None:
                    if key not in ['cno', 'mgsi']:
                        LIS_params=theta[self.phi_count + self.LISdict[key]*self.LISparamcount:\
                                         self.phi_count + (self.LISdict[key] + 1)*self.LISparamcount]
                        CRinds_theta[key] = -LIS_params[1]
                        
                        CRfluxes_theta[key] = []
                        for i in range(len(self.GRdata)):
                            CRfluxes_theta[key] += [self.crformula_IS(LIS_params, self.CRp_atGRE[key][key][i], \
                                                    ph.M_DICT[key]).reshape(self.GRdata[i][:,0].shape)]
                            
                    else: #if the bin contains multiple elements
                        
                        # alpha will be avg weighted by flux at 1 GeV; fluxes will be sum of fluxes
                        weighted_sum = 0
                        fluxes = [np.copy(x) for x in self.empty_fluxes]
                        for el in enf.enh_els[key]:
                            LIS_params=theta[self.phi_count + self.LISdict[el]*self.LISparamcount:\
                                          self.phi_count + (self.LISdict[el] + 1)*self.LISparamcount]
                            weighted_sum += ph.p1_DICT[el]*LIS_params[1]
                            
                            for i in range(len(self.GRdata)):
                                addition = self.crformula_IS(LIS_params, self.CRp_atGRE[key][el][i], ph.M_DICT[el])
                                fluxes[i] += addition.reshape(self.GRdata[i][:,0].shape)
                        
                        CRfluxes_theta[key] = fluxes
                        CRinds_theta[key] = weighted_sum/self.flux_sum[key]
            
            enh_fs = [np.copy(x) for x in self.empty_fluxes]
            return enf.enh(self.modflags['enh'], enh_fs, self.GRdata, CRfluxes_theta, CRinds_theta)
            
        # get GR fluxes at GR energies
        # can also consider some modflag options here
        def grfunc(theta):
            LIS_params_pp = theta[self.phi_count + self.LISparamcount*self.LISdict['h']:\
                                  self.phi_count + self.LISparamcount*(self.LISdict['h']+1)]
            return grf.get_fluxes_pp(LIS_params_pp, self.GRdata, self.crformula_IS)
        
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
                          for i in range(len(cr_fluxes)) if i not in self.VRinds]))*modflags['weights'][0]/self.nCRpoints
            if self.nVRpoints!=0:
                vrlike = -.5*np.sum(np.array([np.sum(((cr_fluxes[i] - self.CRdata[i][:,1])/self.CRdata[i][:,2])**2.) \
                          for i in range(len(cr_fluxes)) if i in self.VRinds]))*modflags['weights'][1]/self.nVRpoints
            
            # get enhancement factors
            if self.nGRpoints!=0:
                enh_f = enhfunc(theta)

                # get p-p GR fluxes at gamma data's energies
                gr_fluxes = grfunc(theta)

                # enhance p-p GR fluxes
                for i in range(len(self.GRdata)):
                    gr_fluxes[i] = enh_f[i]*gr_fluxes[i] #something like this

                # compare GR fluxes to data
                grlike = -.5*np.sum(np.array([np.sum(((gr_fluxes[i] - self.GRdata[i][:,1])/self.GRdata[i][:,2])**2.) \
                          for i in range(len(gr_fluxes))]))*modflags['weights'][2]/self.nGRpoints
            
            # sum up weighted contributions
            return crlike + vrlike + grlike
        
        # CREATE LNPRIOR FUNCTION
        
        if priors['phi']==0: #priors on phis
            def lp_phi(theta):
                lp= -.5*np.sum(((theta[0:self.phi_count] - self.phis)/self.phierrs)**2.)
                return lp
        elif priors['phi']==1: #free phis
            def lp_phi(theta):
                return 0
        
        def lnprior(theta):
            lp=lp_phi(theta)
            return lp
        
        # CREATE LNPROB FUNCTION
        
        def lnprob(theta):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf

            ll=lnlike(theta)
            if not np.isfinite(ll):
                return -np.inf
            
            return lp+ll
            
        self.lnlike=lnlike
        self.lnprior=lnprior
        return
        
    # default start positions for the MCMC sampler
    def get_startpos(self):
        startpos=list(self.phis+np.random.normal(scale=1e-4))
        
        for i in range(self.LISorder.shape[0]):
            startpos+=[5e-12, -2.8]

            if self.modflags['pl']=='b':
                startpos+=[1.]
        
        return np.array(startpos)
        
    # default start positions for the MCMC sampler
    def get_paramnames(self):
        LISparams=['norm','alpha1','alpha2']
        
        paramnames=[]
        for i in range(self.CRexps.shape[0]):
            paramnames+=[self.CRels[i]+'_'+self.CRexps[i]+'_phi']
        for i in range(self.LISorder.shape[0]):
            for j in range(self.LISparamcount):
                paramnames+=[self.LISorder[i]+'_'+LISparams[j]]
        
        return np.array(paramnames)
    