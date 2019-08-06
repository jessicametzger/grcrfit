# the main analysis functions, which will be run by the user

import os
import numpy as np
import matplotlib.pyplot as plt
import corner

from . import run
from . import model
from . import physics as ph

path = os.getcwd()+'/'

# open & read walker file
def get_walker_data(flag, cutoff=0):
    
    f=open(path+flag+'/walkers.dat','r')
    data=f.readlines()
    f.close()
    
    # param names
    names=np.array(data[0].split(','))
    
    data=[x for x in data if x[0]!='#']
    data=data[cutoff:]
    data=np.array([x.split(',') for x in data]).astype(np.float)
    
    return [names[2:],data]


# given 2d array of parameter samples (indexed by step, paramID),
# return the median of each parameter
def get_best_fit(data):
    return np.median(data,axis=0)


# create walker plots (all params), given a run's flag
# will plot last 1000 steps, and only saved temperature
# saves plots in the run's directory (given by its flag)
def walker_plot(flag,cutoff=0):
    
    names,data = get_walker_data(flag,cutoff=cutoff)
    
    # walker IDs
    walkers=np.unique(data[:,1])
    
    # collate data by walker
    new_arrs=[]
    for i in range(walkers.shape[0]):
        new_arrs+=[data[np.where(data[:,1]==walkers[i])[0],:]]
    data=np.stack(new_arrs)
    data=data[:,:,2:]
    
    # plot
    for i in range(data.shape[-1]):
        for j in range(data.shape[0]):
            plt.plot(range(data[j,:,i].shape[0]),data[j,:,i],lw=.2)
        plt.title(names[i])
        plt.savefig(path+flag+'/param'+str(i)+'_walkers.png')
        plt.clf()
    
    return


# create corner plots (all phis vs. everything else, individually)
# saves plots in the run's directory (given by its flag)
def corner_plot(flag, cutoff=0):
    
    names,data = get_walker_data(flag, cutoff=cutoff)
    data=data[:,2:]
    
    phi_params = [i for i in range(names.shape[0]) if 'phi' in names[i].lower()]
    LIS_params = [i for i in range(names.shape[0]) if i not in phi_params]
    
    # corner plot of each phi vs. all LIS params
    for i in range(len(phi_params)):
        phi_name=names[i].strip()
        el_inds=[LIS_params[j] for j in range(len(LIS_params)) if\
                 phi_name[0:2]==names[LIS_params[j]].strip()[0:2]]

        current_inds=np.array(el_inds + [i])

        # scale up LIS norm so it doesn't just say "0.0"
        plot_data = np.copy(data[:,current_inds])
        plot_data[:,0] = plot_data[:,0]*1e9
        plot_labels = np.copy(names[current_inds])
        plot_labels[0] = plot_labels[0]+'*1e9'

        corner.corner(plot_data, labels=plot_labels, show_titles=True, quantiles=[.16,.5,.84])
        plt.savefig(path+flag+'/param'+str(i)+'_corner.png')
        plt.clf()
        plt.close()
    
    return


# reconstruct Model object for given run
def get_model(flag):
    
    # get run info for model configuration
    metadata = run.get_metadata(flag)
    data = run.get_data(metadata['fdict'])
    
    # re-create model
    return model.Model(metadata['modflags'], data)
    

# plot all data w/best-fit models (CR, GR, enhancement)
# saves plots in the run's directory (given by its flag)
def bestfit_plot(flag, cutoff=0):
    
    # get walker data for best-fit params
    names,wdata = get_walker_data(flag, cutoff=cutoff)
    nwalkers=np.unique(wdata[:,1]).shape[0]
    params=get_best_fit(wdata[nwalkers*cutoff:, 2:])
    
    # reconstruct model
    myModel = get_model(flag)
    crfluxes = myModel.crfunc(params)
    
    # get enhancement factors
    enh_f = myModel.enhfunc(params)
    
    if len(myModel.GRdata)!=0:

        # get p-p GR fluxes at gamma data's energies
        grfluxes_pp = myModel.grfunc_pp(params)
        grfluxes = [np.copy(x) for x in grfluxes_pp.copy()]
        
        # get e-bremss. fluxes
        ebrfluxes = myModel.ebrfunc()

        # enhance p-p GR fluxes & add e-bremss data
        for i in range(len(myModel.GRdata)):
            grfluxes[i] = enh_f[i]*grfluxes_pp[i] + ebrfluxes[i]
            
    
    # CR PLOTS
    for i in range(len(crfluxes)):
        x_axis=myModel.CRdata[i][:,0]
        
        plt.plot(x_axis*1e-3/ph.M_DICT[myModel.CRels[i].lower()], 
                 crfluxes[i]*(x_axis**2.), color='blue', label='model',ls='--',marker='o')
        plt.errorbar(x_axis*1e-3/ph.M_DICT[myModel.CRels[i].lower()], 
                     myModel.CRdata[i][:,1]*(x_axis**2.), yerr=myModel.CRdata[i][:,2]*(x_axis**2.),\
                     color='black', label='data',marker='o',ls='')
        plt.title(myModel.CRels[i]+'_'+myModel.CRexps[i])
        plt.xlabel('E [GeV/n]')
        plt.ylabel('flux * E$^2$')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.savefig(path+flag+'/param'+str(i)+'_CR.png')
        plt.clf()
        plt.close()
    
    # GR PLOTS
    for i in range(len(grfluxes)):
        x_axis=myModel.GRdata[i][:,0]
        
        plt.plot(x_axis*1e-3, enh_f[i]*grfluxes_pp[i], color='red', label='model_CR',ls='--',marker='o')
        plt.plot(x_axis*1e-3, ebrfluxes[i], color='green', label='model_ebr',ls='--',marker='o')
        plt.plot(x_axis*1e-3, grfluxes[i], color='blue', label='model',ls='--',marker='o')
        plt.errorbar(x_axis*1e-3, myModel.GRdata[i][:,1], yerr=myModel.GRdata[i][:,2],\
                     color='black', label='data',marker='o',ls='')
        plt.title('gamma_'+myModel.GRexps[i])
        plt.xlabel('E [GeV]')
        plt.ylabel('emissivity')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.ylim(bottom=1e-25)
        plt.savefig(path+flag+'/exp'+str(i)+'_GR.png')
        plt.clf()
        plt.close()
        
    # ENH_F PLOTS
    for i in range(len(grfluxes)):
        x_axis=myModel.GRdata[i][:,0]
        
        plt.plot(x_axis*1e-3, enh_f[i], ls='--',marker='o')
        plt.title('gamma_'+myModel.GRexps[i]+'_enhancement')
        plt.xlabel('E [GeV]')
        plt.ylabel('enhancement')
        plt.xscale('log')
        plt.grid()
        plt.savefig(path+flag+'/exp'+str(i)+'_enh.png')
        plt.clf()
        plt.close()
    
    
    return

