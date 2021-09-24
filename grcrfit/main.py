
# the main fitting function that will be run by the user

from .run import Run

# creates a Run object, executes it, logs output, & returns it
# "flag" gives the name of the run
# "fdict" is dictionary (key = USINE db filepath - one per element, entry = list of its exps to use)
# rerun is bool specifying whether this is a rerun of existing run of same flag (its folder, metadata.json, & walkers.dat must exist)
# nsteps, nwalkers, PT, ntemps, processes - see Fitter, Model objects
# modflags - see Model object
# savesteps is how many steps to save in walkers.dat (if larger than nsteps, will be nsteps)
def run_fit(flag, fdict, rerun=False, nsteps=5000, nwalkers=None, PT=True, ntemps=10, processes=None,
            modflags={'pl': 's', 'enh': 0, 'weights': None, 'priors': 0, 'crscaling': False,
                      'grscaling': False, 'enhext': False, 'priorlimits': False, 'vphi_err': 100.,
                      'fixd': None, 'one_d': True, 'fix_vphi': None, 'scaleRange': None},
            save_steps=1000.):
    
    # initialize run
    myRun=Run(flag, fdict, rerun=rerun, modflags=modflags, nwalkers=nwalkers)
    
    # execute run
    myRun.execute_run(nsteps=nsteps, PT=PT, ntemps=ntemps, processes=processes)
    
    # create the output files
    if not rerun:
        myRun.create_files()
    
    # log output (walkers, metadata) to output files
    myRun.log_output(save_steps=save_steps)

    return myRun
