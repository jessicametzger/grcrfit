import numpy as np
import matplotlib.pyplot as plt

f=open('./Tp_Eg_convert.dat','r')
data=[x for x in f.readlines() if x[0]!='#']
f.close()

data=np.array([x.split(',') for x in data]).astype(np.float)

plt.plot(data[:,0],data[:,1])
plt.xscale('log')
plt.yscale('log')
plt.savefig('./find_factor.png')
plt.clf()