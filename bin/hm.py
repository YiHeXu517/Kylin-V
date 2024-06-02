#!/home/xyh/Apps/miniconda3/bin/python3

import sys
import os
import numpy as np
import math

''' read input '''

fi = open("{}".format(sys.argv[1]),'r')

ss = "start"

Nel,ww,coups,Hel = 0,0,0,0

while ss:

    ss = fi.readline()

    if "electron" in ss:

        ss = fi.readline()

        lis = ss.split()

        Nel,NTerm = int(lis[0]),int(lis[1])

        Hel = np.zeros((Nel,Nel))

        for tm in range(NTerm):

            ss = fi.readline()

            lis = ss.split()

            Hel[int(lis[0])-1,int(lis[1])-1] = float(lis[2])

    if "vibration" in ss:

        ss = fi.readline()

        lis = ss.split()

        Nvib = int(lis[0])

        ww = np.zeros((Nvib))

        for tm in range(Nvib):

            ss = fi.readline()

            lis = ss.split()

            ww[int(lis[0])-1] = float(lis[1])

    if "1st" in ss:

        ss = fi.readline()

        lis = ss.split()

        NTerm = int(lis[0])

        coups = np.zeros((Nvib,Nel*Nel))

        for tm in range(NTerm):

            ss = fi.readline()

            lis = ss.split()

            coups[int(lis[0])-1,(int(lis[1])-1)*Nel+int(lis[2])-1] = float(lis[3])

print("{} electronic states and {} vibrational modes.".format(Nel,Nvib))

fi.close()

''' start HM '''

epss = 1e-5

U,s,Vt = np.linalg.svd(coups.T,full_matrices=False)

#print(s)


# number of direct modes
Neff = 6

# number of INDIRECT blocks
Nbk = 14

def equ_prod (w, ky):

    res = np.zeros(ky.shape)

    for k in range(ky.shape[1]):

        res[:,k] += w*ky[:,k]

    return res

def lanczos_block ( Nel, Neff ):
    V0 = Vt[:Neff,:].T
    Krys = [V0]
    ''' lanczos '''
    for k in range(Nbk):
        w = equ_prod(ww,Krys[k])
        for j in range(k+1):
            mu = Krys[j].T@w
            w -= Krys[j]@mu
        Q,R = np.linalg.qr(w)
        Krys.append(Q)

    Xi = np.hstack(Krys)
    Xi , R = np.linalg.qr(Xi,mode="reduced")
    xHx = Xi.T@equ_prod(ww,Xi)

    G = coups.T@Xi[:,:Neff]
    return xHx,G

xHx,G = lanczos_block(Nel,Neff)

#print(G)

# maximal occupation
Nocc = 16

''' output part  '''

fo = open("map.input",'w')

fo.writelines("{}  {}  {}  NumTerm\n".format(Nel,xHx.shape[0],Nocc))

for i in range(Nel):

    for j in range(Nel):

        if abs(Hel[i,j]) > epss:

            fo.writelines("{}  {}  0  0  {}\n".format(i+1,j+1,Hel[i,j]))


for i in range(xHx.shape[0]):

    for j in range(xHx.shape[0]):

        if abs(xHx[i,j]) > epss:

            fo.writelines("{}  {}  {}  {}  {}\n".format(i*2+1+Nel,i*2+2+Nel,j*2+1+Nel,j*2+2+Nel,xHx[i,j]))

cur = 0

for i in range(Nel):

    for j in range(Nel):

        for k in range(Neff):

            if abs(G[cur,k]) > epss:

                fo.writelines("{}  {}  {}  {}  {}\n".format(i+1,j+1,k*2+1+Nel,k*2+2+Nel,G[cur,k]))

                if i!=j:

                    fo.writelines("{}  {}  {}  {}  {}\n".format(j+1,i+1,k*2+1+Nel,k*2+2+Nel,G[cur,k]))

        cur += 1

fo.close()

print("Remember to change the NumTerm to number of HM terms in the input file!")