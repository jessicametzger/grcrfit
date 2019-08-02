import os
import numpy as np
import datetime

f=open(os.getcwd()+'/gamma_fluxes.dat','r')
data=[x for x in f.readlines() if x[0]!='#']
f.close()

data=np.array([x.split(',') for x in data]).astype(np.float)
data[:,0:2] = data[:,0:2]/1000. #to GeV
print(data)
d=datetime.datetime.today()
print(type(d.year))

header= '# Date: '+str(d.year)+'/'+str(d.month)+'/'+str(d.day)+'\n'+\
    '#   Format: USINE code'+'\n'+\
    '#   Col.1  -  QUANTITY NAME (case insensitive)'+'\n'+\
    '#   Col.2  -  SUB-EXP NAME (case insensitive, no space)'+'\n'+\
    '#   Col.3  -  EAXIS TYPE: EKN, EK, R, or ETOT'+'\n'+\
    '#   Col.4  -  <E>: mean value bin [GeV/n, GeV, GV, or GeV]'+'\n'+\
    '#   Col.5  -  EBIN_LOW'+'\n'+\
    '#   Col.6  -  EBIN_HIGH'+'\n'+\
    '#   Col.7  -  QUANTITY VALUE: [#/sr/s/m2/EAxis] if flux , no unit if ratio, 1e24 emissivity if gamma flux'+'\n'+\
    '#   Col.8  -  ERR_STAT-'+'\n'+\
    '#   Col.9  -  ERR_STAT+'+'\n'+\
    '#   Col.10 -  ERR_SYST-'+'\n'+\
    '#   Col.11 -  ERR_SYST+'+'\n'+\
    '#   Col.12 -  ADS URL FOR PAPER REF (no space)'+'\n'+\
    '#   Col.13 -  phi [MV]'+'\n'+\
    '#   Col.14 -  DISTANCE EXP IN SOLAR SYSTEM [AU]'+'\n'+\
    '#   Col.15 -  DATIMES: format = yyyy/mm/dd-hhmmss:yyyy/mm/dd-hhmmss;...'+'\n'+\
    '#   Col.16 -  IS UPPER LIMIT: format = 0 or 1'+'\n'

info={'quantity':'gamma',
    'subexp': 'FERMI-LAT_LOCAL_HI',
    'Eax': 'EK', 
    'ADS': '2015ApJ...806..240C',
    'phi': str(-1),
    'dist': str(-1),
    'datimes': str(-1),
    'isul': str(0)}

newf=open(os.getcwd()+'/gammadatabase.USINE','w')
newf.write(header)
for i in range(data.shape[0]):
    # choose log mean of E bin as mean E
    newf.write(info['quantity']+'  '+info['subexp']+'  '+info['Eax']+'  '+\
        str(10.**((np.log10(data[i,0])+np.log10(data[i,1]))/2.))+'  '+\
        str(data[i,0])+'  '+str(data[i,1])+'  '+str(data[i,2])+'  '+\
        str(data[i,3])+'  '+str(data[i,3])+'  '+str(data[i,4])+'  '+str(data[i,4])+'  '+\
        info['ADS']+'  '+info['phi']+'  '+info['dist']+'  '+info['datimes']+'  '+info['isul']+'\n')

newf.close()
