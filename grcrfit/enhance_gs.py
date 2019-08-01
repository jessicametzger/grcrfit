import os,sys
sys.path=["/users/jmetzger/local/"]+sys.path
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

c=299792458. #m/s

def E_to_p(E,m):
    Ekin=E*m*1000 #GeV/n to MeV
    Etot = Ekin + 939*m #total energy
    radical=Etot**2 - (939*m)**2
    if np.any(radical)<0: print(E,m)
    
    p = (1/c)*np.sqrt(radical)
    return p

fname=raw_input('File must be comma, space, or tab separated\n'+\
    'First column should be E [GeV], second should be gamma ray flux\n'+\
    'File headers/comments must start with "#"\n'+\
    'Enter file name: ')

try:
    f=open('./'+fname,'r')
except:
    print('file not found')
    sys.exit()

data=f.readlines()
f.close()

data=[x for x in data if '#' not in x]
if ',' in data[0]:
    data=np.array([x.split(',') for x in data]).astype(np.float)
elif '\t' in data[0]:
    data=np.array([x.split('\t') for x in data]).astype(np.float)
else:
    try:
        data=np.array([x.split() for x in data]).astype(np.float)
    except:
	print('bad file format')
	sys.exit()

# get multiplication factors
f=open("./enh_f_2014.txt","r")
Zs=f.readlines()
f.close()

# make table of multiplication factors
Z_alphas = np.array([2, 2.2, 2.4, 2.6, 2.8, 3])
Z_Es = np.array([10,100,1000])
tables=[Zs[5:15],Zs[16:26],Zs[27:37],Zs[38:48],Zs[49:59],Zs[60:70]]
Zs=np.array([[x.split('\t')[2:-1] for x in tab] for tab in tables]).astype(np.float)

# construct abundance ratios
path='/users/jmetzger/LIS_fitting/'
abund_rats=np.array([1,0.096,1.38e-3,2.11e-4,3.25e-5])

# construct LISs, fluxes
E=data[:,0] #given data
fluxes=np.empty((5,E.shape[0]))

# Hydrogen
m=1
p=E_to_p(E,m)
f=path+'H_fluxes/fits/results_5capsingle_LIS.txt'
LIS=open(f,'r').readlines()[1].split(',')[3:5]
LIS=[float(x) for x in LIS]
LISH=LIS
fluxes[0]=(LIS[1]*1e-20)*(p**(LIS[0]))

# Helium
m=4
p=E_to_p(E,m)
f=path+'He_fluxes/fits/results_single_LIS.txt'
LIS=open(f,'r').readlines()[1].split(',')[3:5]
LIS=[float(x) for x in LIS]
LISHe=LIS
fluxes[1]=(LIS[1]*1e-20)*(p**(LIS[0]))

# CNO
m=14
p=E_to_p(E,m)
fileC=path+'C_fluxes/fits/results_single_LIS.txt'
LISC=open(fileC,'r').readlines()[1].split(',')[3:5]
LISC=[float(x) for x in LISC]
fileN=path+'N_fluxes/fits/results_single_LIS.txt'
LISN=open(fileN,'r').readlines()[1].split(',')[3:5]
LISN=[float(x) for x in LISN]
fileO=path+'O_fluxes/fits/results_single_LIS.txt'
LISO=open(fileO,'r').readlines()[1].split(',')[3:5]
LISO=[float(x) for x in LISO]
fluxes[2]=(LISC[1]*1e-20)*(p**(LISC[0]))+(LISN[1]*1e-20)*(p**(LISN[0]))+(LISO[1]*1e-20)*(p**(LISO[0]))

# weighted mean alpha for CNO
p10=E_to_p(10,m)
fluxes10=[(LISC[1]*1e-20)*(p10**(LISC[0])),(LISN[1]*1e-20)*(p10**(LISN[0])),(LISO[1]*1e-20)*(p10**(LISO[0]))]
CNO_alpha=(LISC[0]*fluxes10[0]+LISN[0]*fluxes10[1]+\
        LISO[0]*fluxes10[2])/(fluxes10[0]+fluxes10[1]+fluxes10[2])

# Honda et al. 2004 for heavier elements
LIS_params=np.array([[2.74,14900,2.15,0.21],
                     [2.64,600,1.25,0.14],
                     [2.60,33.2,0.97,0.01],
                     [2.79,34.2,2.14,0.01],
                     [2.68,4.45,3.07,0.41]])

old_fluxes=np.array([LIS_params[:,1]*(x + LIS_params[:,2]*np.exp(-LIS_params[:,3]*np.sqrt(x)))**(-LIS_params[:,0]) \
                     for x in E])

## plot old & new LIS's
#elements=['H','He','CNO']
#for i in range(3):
#    plt.plot(E,fluxes[i]*E**2,label='new, '+elements[i])
#    plt.plot(E,old_fluxes[:,i]*E**2,label='old, '+elements[i])

#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel('E [GeV/n]')
#plt.ylabel('flux E^2')
#plt.grid()
#plt.title('LIS Comparison')
#plt.savefig('/home/jmetzger/CR_modeling/LIS_comparison.png')
#plt.clf()

# use old fluxes for heavier elements
fluxes[3]=old_fluxes[:,3]
fluxes[4]=old_fluxes[:,4]

# create array of indices
alphas=np.array([LISH[0],LISHe[0],CNO_alpha,-LIS_params[3,0],-LIS_params[4,0]])

enh_fs=[np.empty((10,E.shape[0])),np.empty((10,E.shape[0]))]

# QGSJET-II-04 (using Zs[0:3], filling enh_fs[0]), then EPOS-LHC (using Zs[3:], filling enh_fs[1])
for k in range(2):
    for i in range(alphas.shape[0]): #loop over projectile species

        # get p-p Zs for ratio
        current_Zs=Zs[k*3:3*(k+1),0,:]
        fs=[interpolate.interp1d(Z_alphas,y,kind='linear',fill_value='extrapolate') for y in current_Zs]
        alph_interp_Zs_pp=np.array([f(-alphas[i]) for f in fs]) # get at all 3 energies at interpolated index
        f=interpolate.interp1d(np.log10(Z_Es),alph_interp_Zs_pp,kind='linear',fill_value='extrapolate')
        interp_Zs_pp=f(np.log10(E)) #interpolate at desired energies assuming Z vs. log(E) is ~linear

        for j in range(2): #loop over target species
            current_Zs=Zs[k*3:3*(k+1),5*j+i,:]
            fs=[interpolate.interp1d(Z_alphas,y,kind='linear',fill_value='extrapolate') for y in current_Zs]
            alph_interp_Zs=np.array([f(-alphas[i]) for f in fs]) # get at all 3 energies at interpolated index
            f=interpolate.interp1d(np.log10(Z_Es),alph_interp_Zs,kind='linear',fill_value='extrapolate')
            interp_Zs=f(np.log10(E)) #interpolate at desired energies assuming Z vs. log(E) is ~linear

            enh_fs[k][5*j+i,:]=(abund_rats[j]/abund_rats[0])*(fluxes[i]/fluxes[0])*(interp_Zs/interp_Zs_pp)

# sum along interaction axis
enh_fs[0]=np.sum(enh_fs[0],axis=0)
enh_fs[1]=np.sum(enh_fs[1],axis=0)

# enhance
enh_data=np.empty((data.shape[0],3))
enh_data[:,0]=data[:,0]
enh_data[:,1]=data[:,1]*enh_fs[0]
enh_data[:,2]=data[:,1]*enh_fs[1]

for i in range(enh_data.shape[0]):
    print(enh_data[i,0],enh_fs[1][i],enh_data[i,2]/data[i,1])

# write enhanced data to csv file
f=open(fname[0:fname.index('.')]+'_enhanced.csv','w')

# columns: E [GeV/n], first set of enhanced fluxes (input units), second set (input units)
f.write('# E [GeV/n], QGSJET-II-04 enhanced, EPOS-LHC enhanced\n')

for i in range(enh_data.shape[0]):
    f.write(str(enh_data[i,0])+', '+str(enh_data[i,1])+', '+str(enh_data[i,2])+'\n')

f.close()

