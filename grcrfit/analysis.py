# the main analysis functions, which will be run by the user

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import re
import sys

from . import run
from . import model
from . import physics as ph
from . import enhs as enf
from . import helpers as h

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
    
    return [names[2:],data]


# given 2d array of parameter samples (indexed by step, paramID),
# return the mean & stddev of each parameter
def get_best_fit(data):
    return [np.median(data,axis=0), np.std(data,axis=0)]


# create walker plots (all params), given a run's flag.
# will remove first "cutoff" steps.
# saves plots in the run's directory (given by its flag), filename = "param#_walkers.png"
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
            plt.plot(range(data[j,:,i].shape[0]),data[j,:,i],ls='',marker=',',ms=.1)
        plt.title(names[i].replace('_',', ').upper())
        plt.savefig(path+flag+'/param'+str(i)+'_walkers.png')
        plt.clf()
    
    return


# create corner plots (all phis vs. respective LIS parameters, individually)
# saves plots in the run's directory (given by its flag), filename = "param#_corner.png"
def corner_plot(flag, cutoff=0):
    
    # get walker data
    names,data = get_walker_data(flag, cutoff=cutoff)
    data=data[:,2:]
    
    # get param names
    phi_params = [i for i in range(names.shape[0]) if 'phi' in names[i].lower()]
    LIS_params = [i for i in range(names.shape[0]) if i not in phi_params]
    
    # corner plot of each phi vs. all its corresponding LIS params
    for i in range(len(phi_params)):
        
        # get inds of parameters to use
        phi_name=names[i].strip()
        el_inds=[LIS_params[j] for j in range(len(LIS_params)) if\
                 phi_name[0:2]==names[LIS_params[j]].strip()[0:2]]
        current_inds=np.array(el_inds + [i])

        # scale up LIS norm so it doesn't just say "0.0"
        plot_data = np.copy(data[:,current_inds])
        plot_data[:,0] = plot_data[:,0]*1e9
        plot_labels = np.copy(names[current_inds])
        plot_labels[0] = plot_labels[0]+'*1e9'
        plot_labels = [re.sub(r'\(.+?\)', '', plot_labels[j]) for j in range(plot_labels.shape[0])]
        
        # plot & save figure
        corner.corner(plot_data, labels=plot_labels, show_titles=True, quantiles=[.16,.5,.84])
        plt.savefig(path+flag+'/param'+str(i)+'_corner.png')
        plt.clf()
        plt.close()
    return


# reconstruct Model object for given run
def get_model(flag):
    
    # get run info necessary for model configuration
    metadata = run.get_metadata(flag)
    datadict = run.get_data(metadata['fdict'])
    
    # re-initialize model & return it
    return model.Model(metadata['modflags'], datadict)


# plot all data w/best-fit models (CR, GR, enhancement model, phi best-fit w/Usoskin values)
# saves plots in the run's directory (given by its flag) - file format "exp#_.png" (# is kind of meaningless)
# also print total chi squared
# cutoff specifies how many burn-in steps to exclude when getting median parameters
def bestfit_plot(flag, cutoff=0):
    
    # get walker data
    names,wdata = get_walker_data(flag, cutoff=cutoff)
    nwalkers=np.unique(wdata[:,1]).shape[0]
    
    # get best-fit params from walker data
    params,paramerrs=get_best_fit(wdata[nwalkers*cutoff:, 2:])
    
    # reconstruct model
    myModel = get_model(flag)
    
    # Get fluxes at data energies for chi squared calculation
    
    # CR model
    crfluxes_d = myModel.crfunc(params)
    
    # enhancement factors
    enh_f_d = myModel.enhfunc(params)
    
    # GR model
    if len(myModel.GRdata)!=0:

        # get p-p GR fluxes at gamma data's energies
        grfluxes_pp_d = myModel.grfunc_pp(params)
        grfluxes_d = [np.copy(x) for x in grfluxes_pp_d]
        
        # get e-bremss. fluxes
        ebrfluxes_d = myModel.ebrfunc()

        # enhance p-p GR fluxes & add e-bremss data
        for i in range(len(myModel.GRdata)):
            grfluxes_d[i] = enh_f_d[i]*grfluxes_pp_d[i] + ebrfluxes_d[i]
    
    
    # calculate chi squared
    cr_chisqu=[]
    cr_chisqu_r=[]
    for i in range(len(myModel.CRdata)):
        cr_chisqu += [np.sum(((crfluxes_d[i] - myModel.CRdata[i][:,1])/myModel.CRdata[i][:,2])**2.)]
        
        # each dataset, considered independently, has ~nLISparams + 1 free params (nLISparams + solar modulation)
        cr_chisqu_r += [np.sum(((crfluxes_d[i] - myModel.CRdata[i][:,1])/myModel.CRdata[i][:,2])**2.)/
                        max(myModel.CRdata[i].shape[0]-myModel.nLISparams-1,1)]
    
    gr_chisqu=[]
    gr_chisqu_r=[]
    for i in range(len(myModel.GRdata)):
        gr_chisqu += [np.sum(((grfluxes_d[i] - myModel.GRdata[i][:,1])/myModel.GRdata[i][:,2])**2.)]
        
        # gamma ray data, considered independently, has ~nLISparams*nels free params
        gr_chisqu_r += [np.sum(((grfluxes_d[i] - myModel.GRdata[i][:,1])/myModel.GRdata[i][:,2])**2.)/
                        max(myModel.GRdata[i].shape[0]-myModel.nLISparams*myModel.nels,1)]
    
    cr_chisqu=np.array(cr_chisqu)
    gr_chisqu=np.array(gr_chisqu)
    
    # total (reduced) chisqu
    total_chisqu=np.sum(cr_chisqu) + np.sum(gr_chisqu)
    dof=myModel.nCRpoints + myModel.nVRpoints + myModel.nGRpoints - params.shape[0]
    total_chisqu_r=total_chisqu/dof
    print("total chisqu: ",total_chisqu)
    print("total reduced chisqu: ",total_chisqu_r)
    
    # --------------------------------------------------------------------------------------------------------------
    # make x-axis finer for plotting
    # have to re-create lots of the variables - should be neater way to do this
    # this is all directly from model.py
    for i in range(len(myModel.CREs)):
        myModel.CREs[i] = np.logspace(np.log10(np.amin(myModel.CREs[i])), np.log10(np.amax(myModel.CREs[i])), num=100)
    for i in range(len(myModel.GREs)):
        myModel.GREs[i] = np.logspace(np.log10(np.amin(myModel.GREs[i])), np.log10(np.amax(myModel.GREs[i])), num=100)
    myModel.CRp_atGRE = {}
    for key in enf.enh_els:
        myModel.CRp_atGRE[key]={}
        for subkey in enf.enh_els[key]:
            current_el=[]
            for j in range(len(myModel.GREs)):
                # GR energy is some factor smaller than CR energy; take from Mori 1997
                factors=10**enf.fac_interp(np.log10(myModel.GREs[j]))
                current_el+=[ph.E_to_p(myModel.GREs[j]*ph.M_DICT[subkey]*factors, ph.M_DICT[subkey])]
            myModel.CRp_atGRE[key][subkey] = current_el
    myModel.CRfluxes={'h': None, 'he': None, 'cno': None, 'mgsi': None, 'fe': None}
    for key in enf.enh_els_ls:
        if not all(x in myModel.fit_els for x in enf.enh_els[key]):
            myModel.CRfluxes[key] = []
            for i in range(len(myModel.GREs)):
                mean_mass = np.mean([ph.M_DICT[x] for x in enf.enh_els[key]])
                
                factors=10**enf.fac_interp(np.log10(myModel.GREs[i]))
                myModel.CRfluxes[key] += [enf.Honda_LIS(enf.LIS_params[key], myModel.GREs[i]*mean_mass*factors)]
    myModel.empty_fluxes=[np.zeros(myModel.GREs[i].shape) for i in range(len(myModel.GREs))]
    myModel.GRlogEgs=[np.log10(myModel.GREs[i]) for i in range(len(myModel.GREs))]
    myModel.ebr_fluxes = myModel.ebrfunc()
    # --------------------------------------------------------------------------------------------------------------
    
    # get new, finer lists of model values
    crfluxes = myModel.crfunc(params)
    enh_f = myModel.enhfunc(params)
    
    # get p-p GR fluxes at gamma data's energies
    grfluxes_pp = myModel.grfunc_pp(params)
    grfluxes = [np.copy(x) for x in grfluxes_pp]

    # get e-bremss. fluxes
    ebrfluxes = myModel.ebrfunc()

    # enhance p-p GR fluxes & add e-bremss data
    for i in range(len(myModel.GRdata)):
        grfluxes[i] = enh_f[i]*grfluxes_pp[i] + ebrfluxes[i]
            
    
    # CR PLOTS
    for i in range(len(crfluxes_d)):
        x_axis=myModel.CRdata[i][:,0]
        
        # plot
        plt.plot(myModel.CREs[i]*1e-3/ph.M_DICT[myModel.CRels[i].lower()], 
                 crfluxes[i]*(myModel.CREs[i]**2.), color='blue', 
                 label=r'model, $\chi^2$ = '+str(round(cr_chisqu[i],2)),lw=1)
        plt.errorbar(x_axis*1e-3/ph.M_DICT[myModel.CRels[i].lower()], 
                     myModel.CRdata[i][:,1]*(x_axis**2.), yerr=myModel.CRdata[i][:,2]*(x_axis**2.),\
                     color='black', label=r'data',marker='o',ls='',ms=4,zorder=3)
        
        # element name, first letter capitalized
        proper_name=myModel.CRels[i].upper()
        if len(proper_name)>1:
            proper_name=proper_name[0] + proper_name[1:].lower()
        
        # plot paraphernalia
        plt.title(proper_name+', '+myModel.CRexps[i].upper().replace('_',' '))
        plt.xlabel('E [GeV/n]')
        plt.ylabel('(flux) '+r' (E$^2$)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.savefig(path+flag+'/exp'+str(i)+'_CR.png')
        plt.clf()
        plt.close()
    
    # GR PLOTS
    for i in range(len(grfluxes_d)):
        x_axis=myModel.GRdata[i][:,0]
        
        # plot data/all model components
        plt.plot(myModel.GREs[i]*1e-3, enh_f[i]*grfluxes_pp[i], color='red', label=r'model, CR',lw=1)
        plt.plot(myModel.GREs[i]*1e-3, ebrfluxes[i], color='green', label=r'model, e-br',lw=1)
        plt.plot(myModel.GREs[i]*1e-3, grfluxes[i], color='blue', 
                 label=r'model, $\chi^2$ = '+str(round(gr_chisqu[i],2)),lw=1)
        plt.errorbar(x_axis*1e-3, myModel.GRdata[i][:,1], yerr=myModel.GRdata[i][:,2],\
                     color='black', label=r'data',marker='o',ls='',ms=4,zorder=3)
        
        # plot paraphernalia
        plt.title(myModel.GRexps[i].upper().replace('_',' '))
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
        x_axis=myModel.GREs[i]
        
        # plot model
        plt.plot(x_axis*1e-3, enh_f[i],lw=1)
        plt.title('$\t{'+myModel.GRexps[i].upper().replace('_',' ')+', Enhancement}$')
        plt.xlabel('E [GeV]')
        plt.ylabel('enhancement factor')
        plt.xscale('log')
        plt.grid()
        plt.savefig(path+flag+'/exp'+str(i)+'_enh.png')
        plt.clf()
        plt.close()
    
    # PHI PLOTS
    inds=myModel.phitimes.argsort()
    
    plt.errorbar(myModel.phitimes[inds], myModel.phis[inds], yerr=myModel.phierrs[inds], color='black',
                 marker='o', markersize=6, label='Usoskin +11')
    plt.errorbar(myModel.phitimes[inds], params[0:myModel.phis.shape[0]][inds], yerr=paramerrs[0:myModel.phis.shape[0]][inds],
                 color='blue', marker='o', markersize=6, label='best-fit')
    
    plt.ylim(bottom=min(np.amin(myModel.phis),np.amin(params[0:myModel.phis.shape[0]]))*1.1)
    
    plt.legend()
    times=np.arange(np.amin(myModel.phitimes), np.amax(myModel.phitimes),
                   (np.amax(myModel.phitimes) - np.amin(myModel.phitimes))/10.)
    tlabels=['/'.join([str(round(y)) for y in h.JD_to_cal(x)]) for x in times]
    plt.xticks(times, tlabels, rotation=45.)
    plt.ylabel('$\phi$')
    plt.title('Solar Modulation')
    plt.savefig(path+flag+'/phi.png')
    
    return

