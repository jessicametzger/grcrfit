import os
import sys
import numpy as np
import emcee
from multiprocessing import Pool
from tqdm import tqdm

from . import helpers as h
from . import physics as ph
from .model import Model


# technical stuff - configuration of the fitting routine
class Fitter:

    def __init__(self, data, nsteps=5000, nwalkers=None, PT=True, ntemps=10, parallel=True,\
                 priors = {'phi': 0, 'vphi': 0}, modflags = {'pl': 's', 'enh': 0}):
        self.data=data
        self.nsteps=nsteps
        self.nwalkers=nwalkers
        self.PT=PT
        self.ntemps=ntemps
        self.parallel=parallel
        self.modflags = modflags
        self.priors=priors
        
        # create Model object w/MCMC helper functions
        myModel = Model(self.modflags, self.data, priors=self.priors)
        
        # whether or not to run in parallel
        if self.parallel:
            self.pool=Pool()
        else:
            self.pool=None
        
        self.startpos = myModel.get_startpos()
        self.ndim = self.startpos.shape[0]
            
        # figure out how many walkers
        if self.nwalkers==None or self.nwalkers < self.ndim/2. or self.nwalkers%2 != 0:
            self.nwalkers = max((self.ndim + 1)*2,100)
        
        # duplicate startpos for all temps & walkers
        self.startpos = np.array([[self.startpos for x in range(self.nwalkers)] for y in range(self.ntemps)])
        
        # create sampler, either parallel tempering or not
        if self.PT:
            self.sampler = emcee.PTSampler(self.ntemps, self.nwalkers, self.ndim, myModel.lnlike,\
                                           myModel.lnprior, pool=self.pool)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, myModel.lnprob, pool=self.pool)
        
        
    def execute_fit(self):
        
        # run for nsteps, printing progressbar
        for result in tqdm(self.sampler.sample(self.startpos, iterations=self.nsteps, storechain=True)):
            pass
        
        
    def get_chain(self):
        return self.sampler.chain



# logistical stuff - configuration of the run
class Run:

    # flag will be appended to all output filenames.
    # flist is a list of [crfiles, grfiles] to use, where
    # c/grfiles is a dict with fname as key returning
    # list of expnames to use
    def __init__(self, flag, fdict, rerun=False, priors = {'phi': 0, 'vphi': 0}, modflags={'pl': 's', 'enh': 0}):
        
        self.flag = flag
        self.fdict = fdict
        self.priors = priors
        self.modflags = modflags
        self.rerun = rerun
        
        # make sure everything is new
        if not rerun and os.path.exists(self.flag):
            print("Must give new fit a unique flag")
            sys.exit()
        
        # get & format data from fdict
        # each element must have its own file!
        self.data = {}
        for dtype in self.fdict:
            for fkey in self.fdict[dtype]:
                
                # open complete USINE file as arr of strings
                el_data=h.lstoarr(h.open_stdf(fkey,'r'),None)
                
                # create entry for the current element
                self.data[el_data[0,0].lower()]={}
                
                # add entry for each included experiment of that element
                for exp in self.fdict[dtype][fkey]:
                    
                    # select & parse USINE columns to include
                    # all energy x-axis converted to MeV!!
                    entry = el_data[np.where(el_data[:,1] == exp)[0],:]
                    
                    # check x-axis units
                    if np.any(el_data[:,2]!='EKN') and el_data[0,0].lower() != 'gamma':
                          print('Invalid energy units for '+el_data[0,0]+', '+exp+': must be in kinetic energy/nucleon')
                          sys.exit()
                    if el_data[0,0].lower() == 'gamma' and np.any(el_data[:,2]!='EK'):
                          print('Invalid energy units for '+el_data[0,0]+', '+exp+': must be in emissivity*1e24')
                          sys.exit()
                    
                    # columns: Emean, Elow, Ehigh, value, stat-, stat+, sys-, sys+, 
                    entry = el_data[:,np.array([3,4,5,6,7,8,9,10,12,13,14,15])]
                    
                    # calculate dates
                    entry[:,-2] = np.array([str(h.Udate_to_JD(x)) for x in entry[:,-2]])
                    
                    # remove upper limits
                    entry = entry[np.where(entry[:,-1].astype(np.int) != 1)[0],:-1]
                    if entry.shape[0]==0: continue
                    
                    # no valid date for CR experiment?
                    entry=entry.astype(np.float)
                    if np.any(entry[:,-1]<0) and el_data[0,0].lower() != 'gamma':
                        print('Invalid date for '+el_data[0,0]+', '+exp)
                        sys.exit()
                    
                    # change energy units from GeV/n to MeV
                    if 'gamma' not in el_data[0,0]:
                        entry[:,0:3] = entry[:,0:3]*(1000.*ph.M_DICT[el_data[0,0].lower()])
                    else:
                        entry[:,0:3] = entry[:,0:3]#*1000. #already in MeV
                    
                    date = entry[0,-1]
                    phi = entry[0,-3]
                    phi_err = 26.
                    dist = entry[0,-2]
                    
                    # set Voyager phi to 0+-65
                    if 'voyager' in exp.lower() and '2012' in exp.lower():
                        phi = 0.
                        phi_err = 65.
                    
                    # data array, date, phi, phi_err, distance
                    self.data[el_data[0,0].lower()][exp] = [entry[:,:-3], entry[0,-1], entry[0,-3], phi_err, entry[0,-2]]
        
        return
    
    def get_data(self):
        return self.data
    
    def execute_run(self, nsteps=5000, nwalkers=None, PT=True, ntemps=10, parallel=True):
        
        self.nsteps=nsteps
        self.nwalkers=nwalkers
        self.PT=PT
        self.ntemps=ntemps
        self.parallel=parallel
        
        # create the Fitter object with the given configuration
        self.myFitter = Fitter(self.data, nsteps=self.nsteps, nwalkers=self.nwalkers, PT=self.PT, ntemps=self.ntemps,
                          parallel=self.parallel, modflags = self.modflags)
        
        # execute the fit
        self.myFitter.execute_fit()

    def create_files(self):
        
        os.mkdir(self.flag)
        
        # paths for walker & metadata files
        self.wkfilep=self.flag+'/walkers.dat'
        self.mdfilep=self.flag+'/metadata.txt'

        open(self.wkfilep,'a').close()
        open(self.mdfilep,'a').close()
        
    
    def log_output(self):
        
        # write fit specs
        self.mf=open(self.mdfilep,'a')
        
        self.mf.close()
        
        # write samples
        self.wf=open(self.wkfilep,'a')
        self.myChain = self.myFitter.get_chain()
        # write...
        self.wf.close()


