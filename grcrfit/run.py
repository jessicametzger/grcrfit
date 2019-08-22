# Run & Fitter classes
# The Run class handles logistical details of a run (e.g. parameters, output files, etc.)
#  and is typically created in the run_fit() function of main.py.
# The Fitter class handles technical details of a run (e.g. walkers, temperatures, etc.)
#  and is typically created in the execute_run() function of the Run class.

import os
import sys
import numpy as np
import emcee
from tqdm import tqdm
import json
from multiprocessing import Pool

from . import helpers as h
from . import physics as ph
from .model import Model

path = os.getcwd()+'/'

# open & read walker file for plotting, interpretation
# returns names too
def get_walker_data(flag, cutoff=0):
    
    f=open(path+flag+'/walkers.dat','r')
    data=f.readlines()
    f.close()
    
    # param names
    names=np.array(data[0].split(','))
    
    data=[x for x in data if x[0]!='#']
    data=data[cutoff:]
    data=np.array([x.split(',') for x in data]).astype(np.float)
    
    # set the AMS02 scale to 1 as was done in fitting
    AMS02_scale = [i for i in range(names.shape[0]) if 'ams02' in names[i].lower() and 'scale' in names[i].lower()]
    if len(AMS02_scale)!=0:
        data[:,AMS02_scale[0]] = 1.
    
    return [names[2:],data]

# open & read metadata file
# return metadata needed for reconstructing model
def get_metadata(flag):
    
    f=open(path+flag+'/metadata.json','r')
    metadata=f.read()
    f.close()
    
    metadata=json.loads(metadata)
    
    # only take one key; they should all be same
    try: metadata=metadata['run0']
    except KeyError: return {}
    
    try: test=metadata['modflags']['crscaling']
    except KeyError: metadata['modflags']['crscaling']=False
    try: test=metadata['modflags']['grscaling']
    except KeyError: metadata['modflags']['grscaling']=False
    try: test=metadata['modflags']['fixd']
    except KeyError: metadata['modflags']['fixd']=None
    try: test=metadata['modflags']['enhext']
    except KeyError: metadata['modflags']['enhext']=False

    return metadata


# get & format data from fdict, into "standard" format for Model class argument
# each element must have its own file!
def get_data(fdict):
    
    dtypes = list(fdict.keys())
    dtypes.sort()
    
    data = {}
    for dtype in dtypes:
        
        files=list(fdict[dtype].keys())
        files.sort()
        
        for fkey in files:

            # open complete USINE file as arr of strings
            el_data=h.lstoarr([x.lower() for x in h.open_stdf(fkey)],None)

            # create entry for the current element
            data[el_data[0,0].lower()]={}

            # add entry for each included experiment of that element
            for exp in fdict[dtype][fkey]:
                exp=exp.lower()

                # select & parse USINE columns to include
                # all energy x-axis converted to MeV!!
                entry = el_data[np.where(el_data[:,1] == exp)[0],:]
                if entry.shape[0]==0:
                    print('Exp not found: '+el_data[0,0]+', '+exp)
                    continue
                    
                # make sure only one isotope (the one with more data) is included
                isotopes = np.unique(entry[:,0])
                if isotopes.shape[0]>1:
                    counts=[]
                    for j in range(isotopes.shape[0]):
                        counts+=[len(np.where(entry[:,0]==isotopes[j])[0])]
                    counts=np.array(counts)
                    pref_isotope=isotopes[np.where(counts==np.amax(counts))]
                    entry = entry[np.where(entry[:,0]==pref_isotope)[0],:]

                # check x-axis units
                if np.any(entry[:,2]!='ekn') and entry[0,0].lower() != 'gamma':
                    print('Invalid energy units: '+el_data[0,0]+', '+exp+' (must be in kinetic energy/nucleon)')
                    continue
                if entry[0,0].lower() == 'gamma' and np.any(entry[:,2]!='ek'):
                    print('Invalid energy units: '+el_data[0,0]+', '+exp+' (must be in kinetic energy)')
                    continue

                # columns: Emean, Elow, Ehigh, value, stat-, stat+, sys-, sys+, phi, dist, date, is upper limit
                entry = entry[:,np.array([3,4,5,6,7,8,9,10,12,13,14,15])]

                # calculate dates
                entry[:,-2] = np.array([str(round(h.Udate_to_JD(x),1)) for x in entry[:,-2]])

                # remove upper limits
                entry = entry[np.where(entry[:,-1].astype(np.int) != 1)[0],:-1] #remove u.l. flag col too
                if entry.shape[0]==0:
                    print('All points upper limits: '+el_data[0,0]+', '+exp)
                    continue

                entry=entry.astype(np.float)

                # no valid date for CR experiment?
                if np.any(entry[:,-1]<0) and el_data[0,0].lower() != 'gamma':
                    print('Invalid date: '+el_data[0,0]+', '+exp)
                    continue

                # check errorbars - make sure there's at least one nonzero; make them positive
                entry = entry[np.where(np.array([not np.all(entry[i,4:8]==0) for i in range(entry.shape[0])]))]
                if entry.shape[0]==0: continue
                entry[:,4:8]=np.abs(entry[:,4:8])

                # change energy units from GeV/n to MeV total
                if 'gamma' not in el_data[0,0]:

                    # remove data higher than 300 GeV/n
                    entry = entry[np.where(entry[:,0] < 300)[0],:]
                    if entry.shape[0]==0:
                        print('No low-energy data: '+el_data[0,0]+', '+exp)
                        continue

                    entry[:,0:3] = entry[:,0:3]*(1000.*ph.M_DICT[el_data[0,0].lower()])
                else:
                    entry[:,0:3] = entry[:,0:3]*1000.

                date = entry[0,-1]
                phi = entry[0,-3]
                phi_err = 26.
                dist = entry[0,-2]

                # set Voyager phi to 0+-65
                if 'voyager1' in exp.lower() and '2012' in exp:
                    phi = 0.
                    phi_err = 65.

                # data array, date, phi, phi_err, distance
                data[el_data[0,0].lower()][exp.lower()] = [entry[:,:-3], date, phi, phi_err, dist]
    
    return data


# technical stuff - configuration of the fitting routine
# data = dictionary of CR & GR data
# nsteps = now many steps to run the sampler for
# nwalkers = how many walkers (default is max(100, 2*(ndim + 1)); forced to be even and at least 2*ndim)
# PT = whether or not to use parallel tempering
# ntemps = how many temperatures for parallel tempering, if PT
# processes = number of threads to run in parallel (default = 1)
# rerun = whether or not this is a rerun of a previous run.
#  If yes, then flag must be provided.
# flag = the name of the run
# modflags = dictionary of model specifications (see model.py)
class Fitter:
    
    # Initialize the Fitter object
    def __init__(self, data, nsteps=5000, nwalkers=None, PT=True, ntemps=10, processes=None, rerun=False, flag=None,\
                 modflags = {'pl': 's', 'enh': 0, 'weights': None, 'priors': 0, 'crscaling': False,
                             'grscaling': False, 'fixd': None, 'enhext': False}):
        
        self.data=data
        self.nsteps=nsteps
        self.nwalkers=nwalkers
        self.PT=PT
        self.ntemps=ntemps
        self.modflags = modflags
        self.rerun=rerun
        self.flag=flag
        self.processes=processes
        
        # fill in missing modflag keys with defaults
        try: test=self.modflags['pl']
        except KeyError: self.modflags['pl'] = 's'
        try: test=self.modflags['enh']
        except KeyError: self.modflags['enh'] = 0
        try: test=self.modflags['weights']
        except KeyError: self.modflags['weights'] = None
        try: test=self.modflags['priors']
        except KeyError: self.modflags['priors'] = 0
        try: test=self.modflags['crscaling']
        except KeyError: self.modflags['crscaling'] = False
        try: test=self.modflags['grscaling']
        except KeyError: self.modflags['grscaling'] = False
        try: test=self.modflags['fixd']
        except KeyError: self.modflags['fixd'] = None
        try: test=self.modflags['enhext']
        except KeyError: self.modflags['enhext'] = False
        
        # create Model object w/MCMC helper functions
        self.myModel = Model(self.modflags, self.data)
        
        if not self.rerun:
            self.startpos = self.myModel.get_startpos()
            self.ndim = self.startpos.shape[0]
            
            # figure out how many walkers
            if self.nwalkers==None or self.nwalkers < (self.ndim + 1)*2 or self.nwalkers%2 != 0:
                self.nwalkers = max((self.ndim + 1)*2,100)
                
            self.startpos=np.array([[self.startpos[i]*(1 + np.random.normal(scale=1e-4)) for i in range(self.startpos.shape[0])]\
                                    for x in range(self.nwalkers)])
            
        else:
            if flag==None:
                print("Must provide flag for rerun.")
                sys.exit()
            
            walkerfile=self.flag+'/walkers.dat'
            metadatafile=self.flag+'/metadata.json'
            try:
                walkers=h.lstoarr(h.open_stdf(walkerfile),',').astype(np.float)
                metadata=h.open_stdf(metadatafile)
            except FileNotFoundError:
                print("Rerun must have existing walker and metadata files")
                sys.exit()

            walkers=walkers[np.where(walkers[:,1]==0.)[0][-1]:,2:] #take only last step
            
            self.nwalkers=walkers.shape[0]
            self.ndim=walkers.shape[1]
            
            self.startpos=walkers
            
        self.pnames = self.myModel.get_paramnames()
        
        # if parallel or not
        if processes in [None,1]:
            self.pool=None
        elif processes>1:
            self.pool=Pool(processes)
        else:
            print("Must give valid number of processes.")
            sys.exit()
        
        # duplicate startpos for all temps. Shape = [ntemps, nwalkers, ndim]
        if self.PT:
            self.startpos = np.array([self.startpos for y in range(self.ntemps)])
        
        # create sampler, either parallel tempering or not
        if self.PT:
            self.sampler = emcee.PTSampler(self.ntemps, self.nwalkers, self.ndim, self.myModel.lnlike,\
                                           self.myModel.lnprior, pool=self.pool)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.myModel.lnprob, pool=self.pool)
            
        return
        
        
    def execute_fit(self):
        
        # run for nsteps, printing progressbar
        for result in tqdm(self.sampler.sample(self.startpos, iterations=self.nsteps, storechain=True)):
            pass
        return

    def get_chain(self):
        return self.sampler.chain


# logistical stuff - configuration of the run, including creation of a Model object.
# flag = name of the run (will be title of run folder, & appended to all output filenames)
# fdict = dictionary of which experiments from which files to use
# modflags = specifications for the model (see model.py)
class Run:
    
    # Initialize Run object
    def __init__(self, flag, fdict, rerun=False, nwalkers=None,
                 modflags = {'pl': 's', 'enh': 0, 'weights': None, 'priors': 0, 'crscaling': False,
                             'grscaling': False, 'fixd': None, 'enhext': False}):
        
        self.metadata={}
        self.metadata['rerun'] = rerun
        self.metadata['flag'] = flag
        
        # paths for walker & metadata files. They will be in a run directory in the repo folder
        self.wkfilep=path + self.metadata['flag']+'/walkers.dat'
        self.mdfilep=path + self.metadata['flag']+'/metadata.json'
        
        # make sure everything is new
        if not self.metadata['rerun'] and os.path.exists(self.metadata['flag']):
            print("Must give new fit a unique flag")
            sys.exit()
        
        # make sure everything is new
        if self.metadata['rerun'] and not os.path.exists(self.metadata['flag']):
            print("Must give rerun fit an existing flag")
            sys.exit()
        
        # copy all entries that must be same as old run
        if self.metadata['rerun']:
            old_metadata = get_metadata(self.metadata['flag'])
            for key in old_metadata:
                self.metadata[key] = old_metadata[key]
            self.metadata['rerun']=True
        else:
            self.metadata['fdict'] = fdict
            self.metadata['nwalkers'] = nwalkers
            self.metadata['modflags'] = modflags
        
        self.data = get_data(self.metadata['fdict'])

        return
    
    
    # execute the run which has been configured in __init__.
    # Create a Fitter object with the right parameters.
    def execute_run(self, nsteps=5000, PT=True, ntemps=10, processes=None):
        
        # new entries unique to this run
        self.metadata['nsteps']=nsteps
        self.metadata['PT']=PT
        self.metadata['ntemps']=ntemps
        self.metadata['processes']=processes
        
        # create the Fitter object with the given configuration
        self.myFitter = Fitter(self.data, nsteps=self.metadata['nsteps'], nwalkers=self.metadata['nwalkers'],\
                               PT=self.metadata['PT'], ntemps=self.metadata['ntemps'],flag=self.metadata['flag'], 
                               rerun=self.metadata['rerun'],processes=self.metadata['processes'],
                               modflags=self.metadata['modflags'])
        
        # execute the fit
        self.myFitter.execute_fit()
        
        return
    
    def get_chain(self):
        return self.myFitter.get_chain()
    
    def get_pnames(self):
        return self.myFitter.pnames
    
    # create run folder & output files for the run
    def create_files(self):
        
        os.mkdir(self.metadata['flag'])

        open(self.wkfilep,'a').close()
        open(self.mdfilep,'a').close()
        
        return
        
    
    # log walker positions to the walkers.dat file
    # and log run metadata to the metadata.json file
    def log_output(self, save_steps=2000):
        
        # WRITE METADATA DICTIONARY TO METADATA FILE
        
        if self.metadata['rerun']:
            mf=open(self.mdfilep,'r')
            old_mf=mf.read()
            mf.close()
            
            # each run has its own entry, key = "run0", "run1", etc.
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
        
        
        # WRITE WALKER POSITIONS TO WALKER FILE
        myChain = self.myFitter.get_chain()
        
        # only write lowest temperature
        if self.metadata['PT']:
            myChain=myChain[0,...]
        
        # if rerun, add old steps to beginning of chain
        if self.metadata['rerun']:
            
            names,old_data = get_walker_data(self.metadata['flag'])

            # walker IDs
            walkers=np.unique(old_data[:,1])

            # collate data by walker. shape is (walkers,steps,dim), as in sampler chain
            new_arrs=[]
            for i in range(walkers.shape[0]):
                new_arrs+=[old_data[np.where(old_data[:,1]==walkers[i])[0],:]]
            old_data=np.stack(new_arrs)
            old_data=data[:,:,2:]
            
            # add the old chain to the new chain
            myChain = np.concatenate((old_data, myChain), axis=1)
        
        
        # which steps to record?
        # - last min(nsteps, save_steps)
        start_ind=int(max(myChain.shape[1]-save_steps, 0))
        
        wf=open(self.wkfilep,'w')

        # write header
        wf.write('#step, walker')
        for i in range(self.myFitter.pnames.shape[0]):
            wf.write(', '+self.myFitter.pnames[i])
        
        # write walker positions
        for i in range(int(myChain.shape[1]-start_ind)): #steps
            for j in range(myChain.shape[0]): #walkers
                wf.write('\n'+str(i)+', '+str(j))
                for k in range(myChain.shape[2]): #dimensions
                    wf.write(', '+str(myChain[j,i+start_ind,k]))
                
        wf.close()

        return
