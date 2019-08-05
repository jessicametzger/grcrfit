import grcrfit.physics as ph
import grcrfit.crfluxes as crf
import numpy as np
import matplotlib.pyplot as plt

E=np.logspace(-1,2,num=50)
print(ph.LIS_DICT)

for key in ph.LIS_DICT:
    if key=='1h': continue
    plt.plot(E,E*E*crf.flux_spl_IS(ph.LIS_DICT[key], 
            ph.E_to_p(E*1000.*ph.M_DICT[key], ph.M_DICT[key]), ph.M_DICT[key]),label=key)
plt.xlabel("Ek/N [GeV]")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('./test_LIS.png')
plt.clf()
