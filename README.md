
# grcrfit

Fit local cosmic ray & glamma ray fluxes, all in one go. We use MCMC (through Python's Emcee package) to constrain the cosmic ray spectrum parameters, and the solar modulation values, through comparison with cosmic ray (CR) and gamma ray (GR) data. The model has 5 basic steps, which are carried out by each walker at each iteration assuming a set of LIS parameters and solar modulation values:

1. We assume the local interstellar CR flux is a single- or broken-power-law with respect to CR momentum (or, optionally, a single power law with both CR momentum and velocity).

2. We assume solar modulation proceeds according to the simplified force-field model of Gleeson & Axford 1968. The modulated data is compared to the CR data.

2. We use Kamae +06's gamma ray production framework to get the proton-proton-produced gamma ray flux.

3. We interpolate Kachelriess +14's tables of multiplication factors to calculate the enhancement factor to get the total CR-produced gamma ray spectrum from the proton-proton-produced one.

4. We interpolate Orlando 18's electron brehmsstrahlung-produced gamma ray fluxes assuming the PDDE framework, and add this to the CR-produced spectrum to get the final spectrum, which we compare to the GR data.


# Requirements

Python 3 & everything that comes with it, numpy, emcee, tqdm, scipy, corner.


# Basic run_fit instructions

Runs are performed as demonstrated in test_spl.py, test_bpl.py, and test_brpl.py. Typically, the user will work in the grcrfit package directory and will execute in Python:

    import grcrfit
    grcrfit.run_fit("flag", filedict, kwargs=...)

where "flag" (str) is the name of the run, filedict is a dictionary of .USINE data files & sub-experiments to use, and kwargs specify details of the run (see below, along with run.py and model.py, for more information). "flag" must contain no slashes, etc. as it will be the name of the run's directory. In filedict, each keyword must be a database's full file path and the corresponding entry must be the list of sub-experiments to include for that database (from the second column of the .USINE file). Each cosmic ray species must have its own .USINE database. Gamma ray data must also be in .USINE format, for convenience. In grcrfit/data/data_conversion/, there is a script (dat_to_usine.py) that can be modified to convert data files to .USINE format.


# modflags

The "modflags" kwarg of run_fit is a dictionary specifying details of the model.

- "pl" entry: "s" (single power-law), "b" (beta power-law), or "br" (broken power-law); specifies which CR model to use. test_spl.py assumes the "single power-law" model, where the CR flux is a power law with only the momentum. test_bpl.py assumes the "beta power-law" model, where the CR flux is a power law with both the momentum and the velocity, each getting their own index. test_brpl.py assumes the "broken power-law" model, where the CR flux is a broken power-law with respect to momentum, using the formula from Strong 2015 (ICRC) with a free break momentum for each element and two free indices for each elements, and one free softening parameter $\delta$ for all elements' LISs. So the beta power-law has one extra parameter per element, and the broken power-law has two extra parameters per element and one more extra parameter in total. Default is "s" (single power-law).

- "enh" entry: 0 or 1, specifies which set of multiplication factors to use (Kachelriess +14 provides two tables, calculated using two different frameworks). Default is 0 (QGSJET results).

- "weights" entry: 3-item list of floats, specifying the weights to apply to the cosmic ray, voyager, and gamma ray likelihood values. The log-probability is set as:

        logprob = (weights[0] * CRchisqu/nCRpoints) + (weights[1] * Voyagerchisqu/nVoyagerpoints) + (weights[2] * GRchisqu/nGRpoints) + logprior

    So, note that the absolute weighting of the CR, VR, and GR contributions does matter, if you aren't using a flat prior. If you set "weights" to None, the logprob will be calculated without weighting:
    
        logprob = CRchisqu + Voyagerchisqu + GRchisqu + logprior
    
    Note that for most fits, there is so much data that the prior will have hardly any influence at all with no weighting. The default for this flag is None.

- "priors" entry: 0 or 1, specifying whether to apply a gaussian (0) or flat (1) prior to the solar modulation (and, if included, scaling factors). The Earth-based phi priors are taken from Usoskin +11 (given in the crdb data exports), and the Voyager phi prior is 0+-65 MV. The scaling factor priors are all 1+-0.1. Default is 0 (gaussian priors).

- "scaling" entry: True or False, specifying whether or not to add a free scaling factor to all CR experiments except AMS-02. The scaling factors are meant to account for systematic errors in each experimental setup, and experiments of the same setup and data-taking times share a scaling factor. If the "priors" entry is 0, a prior of 1+-0.1 is applied to each factor. Default is False (no scaling factors).


# Results

After running a fit, the results are saved in these files in the run's folder named after its flag:

- walkers.dat: a table of all walkers from the lowest-temperature chain's last N steps (N specified given in the save_steps kwarg).

- metadata.json: a dictionary of all kwargs, etc. needed to reproduce the run.

You can use these methods in analysis.py to interpret the results (all output is saved in the run's folder):

- walker_plot() creates plots of the last M steps (M specified by the user in the "cutoff" kwarg, or default all steps in walkers.dat).

- corner_plot() creates corner plots of all solar modulation parameters (and, if applicable, scaling parameters) together with the LIS parameter of the same respective element. The user can again choose to only include the last M steps by passing M to the "cutoff" kwarg; otherwise all steps in walkers.dat will be used).

- bestfit_plot() plots the best-fit models with the data (cosmic ray and gamma ray), again allowing the user to specify a sample step cutoff for calculating median parameters for the best-fit. It also creates a plot of the best-fit enhancement factors.

You can see how these are run in the test_spl.py, etc. scripts. The resulting plots are included in their respective folders (however, the walkers.dat file has been excluded from github since it is too large).


# Citations

CR data: http://lpsc.in2p3.fr/crdb/ & references therein

GR data: https://ui.adsabs.harvard.edu/abs/2015ApJ...806..240C/abstract

Enhancement factors: https://ui.adsabs.harvard.edu/abs/2014ApJ...789..136K/abstract

Electron-bremsstrahlung data: https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.2724O/abstract

Solar modulation values: https://ui.adsabs.harvard.edu/abs/2011JGRA..116.2104U/abstract

Solar modulation model: https://ui.adsabs.harvard.edu/abs/1968ApJ...154.1011G/abstract

Reference LIS values: https://ui.adsabs.harvard.edu/abs/2004PhRvD..70d3008H/abstract
