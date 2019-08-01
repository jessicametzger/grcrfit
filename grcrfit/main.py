from .run import Run

# creates a run & executes its main tasks
def run_fit(flag, fdict, nsteps=5000, nwalkers=None, rerun=False, PT=True, ntemps=10, parallel=True,
        priors=None, modflags={'pl': 's', 'enh': 0}):
    
    # initialize run
    myRun=Run(flag, fdict, rerun=rerun, priors=priors, modflags=modflags)
    
    # execute run
    myRun.execute_run(nsteps=nsteps, nwalkers=nwalkers, PT=PT, ntemps=ntemps, parallel=parallel)
    
    # create the output files
    if not rerun:
        myRun.create_files()
    
    # log output to output files
    myRun.log_output()

    return
