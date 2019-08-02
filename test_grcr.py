import numpy as np
import grcrfit,os

cdata=os.getcwd()+'/cdatabase.USINE'
hdata=os.getcwd()+'/hdatabase.USINE'

gdata=os.getcwd()+'/gammadatabase.USINE'

# f=open(hdata,'r')
# data=[x for x in f.readlines()]
# f.close()

# # remove commented lines & divide by columns
# data = [x[:x.find('#')] for x in data]
# data = np.array([x.split() for x in data if len(x)>0])

# names=np.unique(data[:,1])

cnames=['ACE-CRIS(1997/08-1998/04)','PAMELA(2006/07-2008/03)', 'TRACER06(2006/07)',
 'Voyager1(2012/09-2012/12)']
hnames=['AMS01(1998/06)', 'AMS02(2011/05-2013/11)', 'ATIC02(2003/01)',
 'BESS-PolarI(2004/12)','PAMELA(2006/07-2006/07)',
 'PAMELA(2006/07-2006/08)', 'PAMELA(2006/07-2008/12)']
gnames=['FERMI-LAT_LOCAL_HI']

# create dictionary identifying which data to use
fdict={'cr': {cdata: cnames,
              hdata: hnames},
       'gr': {gdata: gnames}}


grcrfit.run_fit('testrun',fdict,nsteps=10,rerun=True,parallel=True, modflags={'pl': 'b', 'enh': 1, 'weights': [.33,.33,.33]})

