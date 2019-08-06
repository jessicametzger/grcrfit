import numpy as np
import matplotlib.pyplot as plt

start_ind=2000

f=open('./test_spl/walkers.dat','r')
data=f.readlines()
f.close()

names=data[0].split(',')
data=np.array([x.split(',') for x in data if x[0]!='#']).astype(np.float)
walkers=np.unique(data[:,1])
new_arrs=[]
for i in range(walkers.shape[0]):
    new_arrs+=[data[np.where(data[:,1]==walkers[i])[0],:]]

data=np.stack(new_arrs)
for i in range(2,data.shape[-1]):
    for j in range(2,data.shape[-1]):
        if i==j: continue
        if np.all(data[:,:,i]==data[:,:,j]):
            print(names[i],names[j])

for i in range(2,data.shape[-1]):
    for j in range(data.shape[0]):
        plt.plot(range(data[j,start_ind:,i].shape[0]),data[j,start_ind:,i],lw=.2)
    plt.title(names[i])
    plt.savefig('./test_spl/param'+str(i)+'_walkers.png')
    plt.clf()
