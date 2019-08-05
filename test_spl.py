import os
import grcrfit
from multiprocessing import Pool

path = os.getcwd()+'/'

hdata=path+'data/hdatabase.USINE'
hedata=path+'data/hedatabase.USINE'
cdata=path+'data/cdatabase.USINE'
ndata=path+'data/ndatabase.USINE'
odata=path+'data/odatabase.USINE'
gdata=path+'data/gammadatabase.USINE'

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

# create dictionary identifying which data to use
fdict={'cr': {hdata: hnames,
              hedata: henames,
              cdata: cnames,
              ndata: nnames,
              odata: onames},
       'gr': {gdata: gnames}}

grcrfit.run_fit('test_spl',fdict,nsteps=2000,rerun=True,processes=4,
                modflags={'pl': 's', 'enh': 0, 'weights': [1,.3,1], 'priors': 0})
