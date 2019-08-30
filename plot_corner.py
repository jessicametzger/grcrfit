

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import re
import sys
import json
import time

from grcrfit import run
from grcrfit import model
from grcrfit import physics as ph
from grcrfit import enhs as enf
from grcrfit import helpers as h

path = os.getcwd()+'/'



# create corner plots (all phis vs. respective LIS parameters, individually)
# saves plots in the run's directory (given by its flag), filename = "param#_corner.png"
def corner_plot(flag, cutoff=0):
    
    # get walker data
    names,data = run.get_walker_data(flag, cutoff=cutoff)
    data=data[:,2:]
    
    # get param indices
    phi_params = [i for i in range(names.shape[0]) if 'phi' in names[i].lower()]
    LIS_params = [i for i in range(names.shape[0]) if 'phi' not in names[i].lower() and\
                 'cr_scale' not in names[i].lower()]
    
    # corner plot of each phi vs. all its corresponding LIS params:
    for i in range(len(phi_params)):
        
        if 'h_voyager' not in names[phi_params[i]].lower():
            continue
        
        # get inds of parameters to use
        phi_name=names[phi_params[i]].strip()
        
        # LIS params for that element
        el_inds=[LIS_params[j] for j in range(len(LIS_params)) if\
                 'he'==names[LIS_params[j]].strip()[0:2] or\
                 ('delta' in names[LIS_params[j]].lower())]
        
        # all indices (LIS, phi(, scale)) for that element/exp; don't plot ams02 filler steps
        current_inds = el_inds + [phi_params[i]]
            
        current_inds=np.array(current_inds)

        plot_data = np.copy(data[:,current_inds])
        plot_labels = np.copy(names[current_inds])
        
        plot_labels = [re.sub(r'\(.+?\)', '', plot_labels[j]) for j in range(plot_labels.shape[0])]
        
        # plot & save figure
        corner.corner(plot_data, labels=plot_labels, show_titles=True, quantiles=[.16,.5,.84])
        plt.savefig(path+flag+'/voyager_he_corner.png')
        plt.clf()
        plt.close()
    return

corner_plot('test_brpl',cutoff=-50000)
