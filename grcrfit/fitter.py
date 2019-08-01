import emcee
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

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

