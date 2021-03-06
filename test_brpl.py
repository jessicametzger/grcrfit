import os
import grcrfit
from grcrfit import analysis

path = os.getcwd()+'/'

# create USINE database filenames
hdata=path+'data/hdatabase.USINE'
hedata=path+'data/hedatabase.USINE'
cdata=path+'data/cdatabase.USINE'
ndata=path+'data/ndatabase.USINE'
odata=path+'data/odatabase.USINE'
gdata=path+'data/gammadatabase.USINE'

# create lists of experiment names for each element
hnames=['AMS01(1998/06)', 'AMS02(2011/05-2013/11)', 'BESS-TeV(2002/08)',
        'BESS-PolarI(2004/12)', 'BESS-PolarII(2007/12-2008/01)', 
        'PAMELA(2006/07-2006/07)', 'PAMELA(2008/03-2008/04)', 'PAMELA(2010/01-2010/01)',
        'Voyager1(2012/10-2012/12)']
henames=['AMS01(1998/06)', 'AMS02(2011/05-2013/11)', 'BESS-TeV(2002/08)',
         'BESS-PolarI(2004/12)', 'BESS-PolarII(2007/12-2008/01)',
         'Voyager1(2012/10-2012/12)']
cnames=['PAMELA(2006/07-2008/03)', 'ACE-CRIS(1998/01-1999/01)', 
        'ACE-CRIS(2001/05-2003/09)', 'ACE-CRIS(1997/08-1998/04)',
        'Voyager1(2012/09-2012/12)', 'HEAO3-C2(1979/10-1980/06)']
nnames=['HEAO3-C2(1979/10-1980/06)', 'ACE-CRIS(1997/08-1998/04)', 
        'ACE-CRIS(2001/05-2003/09)', 'ACE-CRIS(2009/03-2010/01)',
        'IMP5(1969/06-1970/06)', 'IMP7(1973/05-1973/12)']
onames=['HEAO3-C2(1979/10-1980/06)', 'ACE-CRIS(1997/08-1998/04)', 
        'ACE-CRIS(2001/05-2003/09)', 'ACE-CRIS(2009/03-2010/01)',
        'Voyager1(2012/09-2012/12)', 'IMP5(1969/06-1970/06)',
        'IMP7(1973/05-1973/12)']
gnames=['FERMI-LAT_LOCAL_HI']

# create dictionary identifying which data to use; only H and He for now
fdict={'cr': {hdata: hnames,
              hedata: henames},
       'gr': {gdata: gnames}}

# testRun = grcrfit.run_fit('test_brpl',fdict,nsteps=4000,rerun=False,processes=32,ntemps=10,
#                 modflags={'pl': 'br'})

# # plot the last 1000 walkers
# analysis.walker_plot('test_brpl',cutoff=-100000)

# # create corner plots
# analysis.corner_plot('test_brpl',cutoff=-100000)

# create CR plots
analysis.bestfit_plot('test_brpl',cutoff=-100000)

