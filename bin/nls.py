#!/usr/bin/python3

import sys
import os
import numpy as np
from scipy.special import erfc
from scipy.optimize import curve_fit
from scipy.integrate import quad

def exp_abs(x,beta):
    return 0.5*beta*np.exp(-1.0*beta*np.abs(x))
def qua_gau(x,bet):
    r = np.exp(-0.25*bet*np.power(x,2)) * np.abs(x/4.0/bet)
    r += (np.ones(x.shape) / 2.0 / bet - np.power(x, 2.0) / 4.0) * erfc(np.sqrt(bet) * np.abs(x) / 2.0) * 0.5 * np.sqrt(np.pi/bet)
    r = r * np.exp(-bet*np.power(x,2.0)/4.0) * bet**2
    return r

ss = "start"
with open("SVS",'r') as f:
    ss = f.readline()
    NSite = int(ss.split()[3])
    CrlSize = NSite*2-2
    CurBond = 0
    SVDifLis = []
    while ss:
        ss = f.readline()    
        if len(ss)<10 and len(ss)>0:
            CurBond += 1
            NumSVDif = max(int(ss)-2,0)
            if CurBond<=CrlSize:
                Dif = []
                for i in range(NumSVDif):
                    ss = f.readline()
                    Dif.append(float(ss))
                if CurBond < NSite:
                    SVDifLis.append(Dif)
                else:
                    SVDifLis[CrlSize-CurBond] = SVDifLis[CrlSize-CurBond] + Dif
            else:
                yuBond = CurBond % CrlSize
                if yuBond > NSite - 1:
                    yuBond = CrlSize - yuBond
                for i in range(NumSVDif):
                    ss = f.readline()
                    try:
                        SVDifLis[yuBond].append(float(ss))
                    except:
                        SVDifLis[yuBond-1].append(float(ss))
f.close()
fot = open("beta",'w')
for betas in SVDifLis:
    if len(betas)>2:
        ni,bini = np.histogram(betas,50)
        Res = curve_fit(qua_gau,bini[:-1],ni/len(betas),bounds=(0,np.inf))
        fot.writelines("{:21.15f}\n".format(Res[0][0]))
    else:
        fot.writelines("{:21.15f}\n".format(1.0))
fot.close()
