import os
import sys
import re
import numpy as np

# open file, read lines, remove comments
def open_stdf(fname,flag):
    f=open(fname,flag)
    fdata=f.readlines()
    f.close()
    
    fdata = [x[:x.find('#')] for x in fdata]
    return [x for x in fdata if len(x)>0]
    
# turn list of strings, each delimited by delimiter d,
# into np array of strings
def lstoarr(datlist,d):
    fdata=[x.split(d) for x in datlist]
    return np.array(fdata)


# DATE CONVERSION

def cal_to_JD(date_ls):
    Y,M,D=date_ls
    JD=0
    JD+=(1461*(Y + 4800 + (M - 14) // 12)) // 4
    JD+=(367*(M - 2 - 12*((M - 14) // 12))) // 12
    JD-=((3*((Y + 4900 + (M - 14) // 12) // 100)) // 4 - D + 32075)
    return JD

def JD_to_cal(JD):
    L= JD+68569
    N= 4*L//146097
    L= L-(146097*N+3)//4
    I= 4000*(L+1)//1461001
    L= L-1461*I//4+31
    J= 80*L//2447
    K= L-2447*J//80
    L= J//11
    J= J+2-12*L
    I= 100*(N-49)+I+L

    Y= I
    M= J
    D= K
    
    return [Y, M, D]

def time_to_days(time_ls):
    while len(time_ls)<3:
        time_ls+=[0]
    
    hh, mm, ss = time_ls
    days = hh/24. + mm/(24*60) + ss/(24*60*60)
    return days

# go from USINE-formatted date ranges (yyyy/mm/dd-hhmmss:yyyy/mm/dd-hhmmss;...)
# to the averate JD of the date ranges
def Udate_to_JD(udate):
    udate_ls = re.split('[:;]',udate.strip())
    
    # list of date lists (yyyy, mm, dd)
    date_ls = [x.split('-')[0] for x in udate_ls]
    try:
        date_ls = [[float(z) for z in y.split('/')] for y in date_ls]
    except ValueError:
        return -1
    
    # list of time lists (hh, mm, ss)
    time_ls = [x.split('-')[1] for x in udate_ls]
    try:
        time_ls = [[float(y[2*i:2*(i+1)]) for i in range(int(len(time_ls)/2))] for y in time_ls]
    except ValueError:
        time_ls = [[0,0,0] for y in time_ls]
    
    assert len(date_ls) == len(time_ls), "Incorrect date format"
    
    # take avg of list of JDs
    JDs = np.array([cal_to_JD(date_ls[i]) + time_to_days(time_ls[i]) for i in range(len(date_ls))])
    return np.mean(JDs)
