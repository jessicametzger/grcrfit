import json,os

path=os.getcwd()+'/'
folders=['test_spl','test_bpl','test_brpl','noweight_spl','noweight_bpl','noweight_brpl']

for i in range(len(folders)):
    mf=open(path+folders[i]+'/metadata.json','r')
    md=mf.read()
    mf.close()
    
    md=json.loads(md)
    for key in md:
        try: test=md[key]['run']
        except KeyError: md[key]['run']=int(key.replace('run','0'))
        try: test=md[key]['modflags']['fixd']
        except KeyError: md[key]['modflags']['fixd']=None
        try: test=md[key]['modflags']['crscaling']
        except KeyError: md[key]['modflags']['crscaling']=False
        try: test=md[key]['modflags']['grscaling']
        except KeyError: md[key]['modflags']['grscaling']=False
        try: test=md[key]['modflags']['enhext']
        except KeyError: md[key]['modflags']['enhext']=True
    
    mf=open(path+folders[i]+'/metadata.json','w')
    json.dump(md,mf)
    mf.close()
