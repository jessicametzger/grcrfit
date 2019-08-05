# the main functions that will be run by the user

from .run import Run, Fitter

# creates a Run object, executes it, & logs output
def run_fit(flag, fdict, nsteps=5000, nwalkers=None, rerun=False, PT=True, ntemps=10, processes=None,
        priors=None, modflags={'pl': 's', 'enh': 0, 'weights': [.33,.33,.33]}):
    
    # initialize run
    myRun=Run(flag, fdict, rerun=rerun, priors=priors, modflags=modflags)
    
    # execute run
    myRun.execute_run(nsteps=nsteps, nwalkers=nwalkers, PT=PT, ntemps=ntemps, processes=processes)
    
    # create the output files
    if not rerun:
        myRun.create_files()
    
    # log output (walkers, metadata) to output files
    myRun.log_output()

    return

# create walker plots (all params)
def walker_plots():
    return

# create corner plots (all phis vs. everything else, individually)
def corner_plots():
    return

# plot CR data w/best-fit LIS's
def plot_crfits():
    return

# plot GR data w/best-fit spectrum
def plot_grfits():
    return

# plot enhancement factor contributions vs. GR energy, over range considered
def plot_enhfs():
    return
