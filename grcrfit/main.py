
# the main fitting function that will be run by the user

from .run import Run

# creates a Run object, executes it, logs output, & returns it
def run_fit(flag, fdict, nsteps=5000, nwalkers=None, rerun=False, PT=True, ntemps=10, processes=None,
        modflags={'pl': 's', 'enh': 0, 'weights': None, 'priors': 0}, save_steps=2000.):
    
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
