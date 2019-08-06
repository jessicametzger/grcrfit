f=open('./test_spl/walkers.dat','r')
data=f.readlines()
f.close()

headers=[x for x in data if x[0]=='#']
data=data[-200000:]

f=open('./test_spl/walkers1.dat','w')
f.write(headers[0])
for line in data:
    f.write(line)
f.close()
