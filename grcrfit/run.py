import os
import sys
import numpy as np
import emcee
from multiprocessing import Pool
from tqdm import tqdm
import time
import json

from . import helpers as h
from . import physics as ph
from .model import Model


# technical stuff - configuration of the fitting routine
class Fitter:
    
    # note: if rerun, nwalkers will be forced to match original number
    def __init__(self, data, nsteps=5000, nwalkers=None, PT=True, ntemps=10, parallel=True, rerun=False, flag=None,\
                 priors = {'phi': 0, 'vphi': 0}, modflags = {'pl': 's', 'enh': 0, 'weights': [.33,.33,.33]}):
        
        self.data=data
        self.nsteps=nsteps
        self.nwalkers=nwalkers
        self.PT=PT
        self.ntemps=ntemps
        self.parallel=parallel
        self.modflags = modflags
        self.priors=priors
        self.rerun=rerun
        self.flag=flag
        
        # create Model object w/MCMC helper functions
        myModel = Model(self.modflags, self.data, priors=self.priors)
        
        # whether or not to run in parallel
        if self.parallel:
            self.pool=Pool()
        else:
            self.pool=None
        
        if not self.rerun:
            self.startpos = myModel.get_startpos()
            self.ndim = self.startpos.shape[0]
            
            # figure out how many walkers
            if self.nwalkers==None or self.nwalkers < self.ndim/2. or self.nwalkers%2 != 0:
                self.nwalkers = max((self.ndim + 1)*2,100)
                
            self.startpos=np.array([self.startpos*(1 + np.random.normal(scale=1e-4)) for x in range(self.nwalkers)])
            
        else:
            walkerfile=self.flag+'/walkers.dat'
            walkers=h.lstoarr(h.open_stdf(walkerfile,'r'),',').astype(np.float)
            walkers=walkers[np.where(walkers[:,1]==0.)[0][-1]:,2:] #take only last step
            
            self.nwalkers=walkers.shape[0]
            self.ndim=walkers.shape[1]
            
            self.startpos=walkers
            
        self.pnames = myModel.get_paramnames()
        
        # duplicate startpos for all temps. Shape = [ntemps, nwalkers, ndim]
        if self.PT:
            self.startpos = np.array([self.startpos for y in range(self.ntemps)])
        
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
    def __init__(self, flag, fdict, rerun=False, priors = {'phi': 0, 'vphi': 0}, \
                 modflags={'pl': 's', 'enh': 0, 'weights': [.33,.33,.33]}):
        
        self.metadata={}
        
        self.metadata['flag'] = flag
        self.metadata['fdict'] = fdict
        self.metadata['priors'] = priors
        self.metadata['modflags'] = modflags
        self.metadata['rerun'] = rerun
        
        # paths for walker & metadata files
        self.wkfilep=self.metadata['flag']+'/walkers.dat'
        self.mdfilep=self.metadata['flag']+'/metadata.json'
        
        # make sure everything is new
        if not self.metadata['rerun'] and os.path.exists(self.metadata['flag']):
            print("Must give new fit a unique flag")
            sys.exit()
        
        # get & format data from fdict
        # each element must have its own file!
        self.data = {}
        for dtype in self.metadata['fdict']:
            for fkey in self.metadata['fdict'][dtype]:
                
                # open complete USINE file as arr of strings
                el_data=h.lstoarr([x.lower() for x in h.open_stdf(fkey,'r')],None)
                
                # create entry for the current element
                self.data[el_data[0,0].lower()]={}
                
                # add entry for each included experiment of that element
                for exp in self.metadata['fdict'][dtype][fkey]:
                    exp=exp.lower()
                    
                    # select & parse USINE columns to include
                    # all energy x-axis converted to MeV!!
                    entry = el_data[np.where(el_data[:,1] == exp)[0],:]
                    
                    # check x-axis units
                    if np.any(entry[:,2]!='ekn') and entry[0,0].lower() != 'gamma':
                          print('Invalid energy units for '+el_data[0,0]+', '+exp+': must be in kinetic energy/nucleon')
                          sys.exit()
                    if entry[0,0].lower() == 'gamma' and np.any(entry[:,2]!='ek'):
                          print('Invalid energy units for '+el_data[0,0]+', '+exp+': must be in emissivity*1e24')
                          sys.exit()
                    
                    # columns: Emean, Elow, Ehigh, value, stat-, stat+, sys-, sys+, phi, dist, date, is upper limit
                    entry = entry[:,np.array([3,4,5,6,7,8,9,10,12,13,14,15])]
                    
                    # calculate dates
                    entry[:,-2] = np.array([str(h.Udate_to_JD(x)) for x in entry[:,-2]])
                    
                    # remove upper limits
                    entry = entry[np.where(entry[:,-1].astype(np.int) != 1)[0],:-1] #remove last col too
                    if entry.shape[0]==0: continue
                    
                    # no valid date for CR experiment?
                    entry=entry.astype(np.float)
                    if np.any(entry[:,-1]<0) and el_data[0,0].lower() != 'gamma':
                        print('Invalid date for '+el_data[0,0]+', '+exp)
                        sys.exit()
                    
                    # check errorbars
                    if np.any(np.array([np.all(entry[i,4:8]==0) for i in range(entry.shape[0])])):
                        print('Point with no errorbars for '+el_data[0,0]+', '+exp)
                        for i in range(entry.shape[0]):
                            print(entry[i,4:8],np.all(entry[i,4:8]==0))
                        sys.exit()
                    entry[:,4:8]=np.abs(entry[:,4:8])
                    
                    # change energy units from GeV/n to MeV
                    if 'gamma' not in el_data[0,0]:
                        entry[:,0:3] = entry[:,0:3]*(1000.*ph.M_DICT[el_data[0,0].lower()])
                    else:
                        entry[:,0:3] = entry[:,0:3] #already in MeV
                    
                    date = entry[0,-1]
                    phi = entry[0,-3]
                    phi_err = 26.
                    dist = entry[0,-2]
                    
                    # set Voyager phi to 0+-65
                    if 'voyager1' in exp.lower() and '2012' in exp:
                        phi = 0.
                        phi_err = 65.
                    
                    # data array, date, phi, phi_err, distance
                    self.data[el_data[0,0].lower()][exp.lower()] = [entry[:,:-3], entry[0,-1], entry[0,-3], phi_err, entry[0,-2]]
        return
    
    def execute_run(self, nsteps=5000, nwalkers=None, PT=True, ntemps=10, parallel=True):
        
        self.metadata['nsteps']=nsteps
        self.metadata['nwalkers']=nwalkers
        self.metadata['PT']=PT
        self.metadata['ntemps']=ntemps
        self.metadata['parallel']=parallel
        
        # create the Fitter object with the given configuration
        self.myFitter = Fitter(self.data, nsteps=self.metadata['nsteps'], nwalkers=self.metadata['nwalkers'],\
                               PT=self.metadata['PT'], ntemps=self.metadata['ntemps'],flag=self.metadata['flag'], 
                               rerun=self.metadata['rerun'],parallel=self.metadata['parallel'], modflags=self.metadata['modflags'])
        
        # execute the fit
        self.myFitter.execute_fit()

    def create_files(self):
        
        os.mkdir(self.metadata['flag'])

        open(self.wkfilep,'a').close()
        open(self.mdfilep,'a').close()
        
    
    def log_output(self):
        
        # write fit specs
        if self.metadata['rerun']:
            mf=open(self.mdfilep,'r')
            old_mf=mf.read()
            mf.close()
            
            # each run has its own entry
            old_metadata=json.loads(old_mf)
            runs=len(old_metadata)
            
        else:
            runs=0
            old_metadata={}
        
        # add new run's metadata to new entry & save to metadata file
        old_metadata['run'+str(runs)]=self.metadata
        mf=open(self.mdfilep,'w')
        json.dump(old_metadata,mf)
        mf.close()
        
        # write samples
        wf=open(self.wkfilep,'a')
        myChain = self.myFitter.get_chain()
        
        # only write lowest temperature
        if self.metadata['PT']:
            myChain=myChain[0,...]
        
        # which steps to record?
        # - if rerun, all.
        # - if not rerun, min(nsteps, 1000).
        if self.metadata['rerun']:
            start_ind=0
        else:
            start_ind=max(myChain.shape[1]-1000, 0)
        
        wf.write('#step, walker, ')
        for i in range(self.myFitter.pnames.shape[0]):
            wf.write(self.myFitter.pnames[i]+', ')
        
        for i in range(myChain.shape[1]-start_ind): #steps
            for j in range(myChain.shape[0]): #walkers
                wf.write('\n'+str(i)+', '+str(j))
                for k in range(myChain.shape[2]):
                    wf.write(', '+str(myChain[j,i,k]))
                
        wf.close()

        return
