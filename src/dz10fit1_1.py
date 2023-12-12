# By Karl Sallmén

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import scipy.optimize as op
from numba import jit
import pymc3 as pm
import arviz as az
import autograd.numpy as agnp
from autograd import grad
import pandas as pd
import matplotlib
import seaborn as sns
import scipy.stats as stats

from utilterms import *
from utilloops import *
#All model terms now self-contained in utilterms
# If you want to add any new terms you can add them below or in utilterms.py

#---------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------
#Main function for fitter

#@jit(nopython=True)
def minimizefunctionDZ10(p):
    print('called',p)
    global BEA
    global Nl
    global Zl
    global pnl
    
    DZ10l = []
    for i in range(len(BEA)):
        temp = DZ10(Nl[i],Zl[i],p)   
        DZ10l.append(temp)
    BEa=np.array(BEA)

    DZ10a=np.array(DZ10l)
    rms = np.sqrt(np.mean((BEa - DZ10a)**2))
    print(rms,'rms')
    return rms

#@jit(nopython=True)
def DZ10(Nli,Zli,p):
    

    N = Nli
    Z = Zli

    r = rho(N,Z)

    A = N + Z
    T = abs(N-Z)/2
    t = abs(N-Z)
 
    loopvecZ = configloop(Z)
    loopvecN = configloop(N)

 
  
 
    M = Master(np.array(loopvecN[0]),np.array(loopvecZ[0]),np.array(loopvecN[6]),np.array(loopvecZ[6]),r)
    Mdef =Master(np.array(loopvecN[1]),np.array(loopvecZ[1]),np.array(loopvecN[7]),np.array(loopvecZ[7]),r)
    S = SpinOrbit(np.array(loopvecN[0]),np.array(loopvecZ[0]),np.array(loopvecN[6]),np.array(loopvecZ[6]),np.array(loopvecN[12]),np.array(loopvecZ[12]),np.array(loopvecN[2]),np.array(loopvecN[3]),np.array(loopvecZ[3]),np.array(loopvecZ[2]))
    Sdef =SpinOrbit(np.array(loopvecN[1]),np.array(loopvecZ[1]),np.array(loopvecN[7]),np.array(loopvecZ[7]),np.array(loopvecN[13]),np.array(loopvecZ[13]),np.array(loopvecN[4]),np.array(loopvecN[5]),np.array(loopvecZ[5]),np.array(loopvecZ[4]))  
    C = Coulomb(Z,A,t)
    T = Symmetry1(r,t,A)
    TS = Symmetry2(r,t,A) 
    P = Pairing(N,Z,r) 
    S1 = Spherical3(loopvecN[8],loopvecN[9],loopvecN[10],loopvecZ[8],loopvecZ[9],loopvecZ[10],r) 
    S2 = (1/r)*Spherical3(loopvecN[8],loopvecN[9],loopvecN[10],loopvecZ[8],loopvecZ[9],loopvecZ[10],r)
    S3 = Spherical4(loopvecN[8],loopvecN[9],loopvecN[10],loopvecZ[8],loopvecZ[9],loopvecZ[10],loopvecN[11],loopvecZ[11],r) 
    D = Deformed(np.array(loopvecN[5])[-2],np.array(loopvecN[4])[-1],loopvecN[14],np.array(loopvecZ[5])[-2],np.array(loopvecZ[4])[-1],loopvecZ[14],r) 
    
 

    DZvalsph= p[0]*(M+S/r) - (p[1]/r)*M + p[2]*C + p[3]*T +p[4]*TS +p[5]*P + p[6]*S1 + p[7]*S2 +p[8]*S3 + p[9]*0
    DZvaldef =  p[0]*(Mdef+Sdef/r) - (p[1]/r)*Mdef + p[2]*C + p[3]*T +p[4]*TS +p[5]*P + p[6]*0 + p[7]*0 + p[8]*0 +p[9]*D
    diff = DZvaldef - DZvalsph
    if diff<0 or Z<50:
        DZval = DZvalsph
    else:
        DZval = DZvaldef    
    #DZval = a1*(M+S/r) - (a2/r)*M + a3*C + a4*T +a5*TS +a6*P + a7*S1 + a8*S2 + a9*S3# +a10*D
    #print(DZval)
    return DZval

#---------------------------------------------------------------------------------------------------------


#Main

"""
#Data handling
with open('bindData/bind2020.dat') as f:
   data = [line.split()[:] for line in f]  

Nl=[]
Zl=[]
Al=[]
BE=[]
Name=[]
Uncertainty = []
BEgvar=[]
for i in range(len(data)):
    if float(data[i][5])*int(data[i][2]) < 100 and int(data[i][0])>=28 and int(data[i][1])>=28:
        Nl.append(int(data[i][0]))
        Zl.append(int(data[i][1]))
        Al.append(int(data[i][2]))
        Name.append(data[i][3])
        BE.append(float(data[i][4])/1000)
        Uncertainty.append(float(data[i][5])/1000)
BEA = np.multiply(BE,Al)
    


NZ= np.column_stack((Nl,Zl))
"""



#params = [17.36375251, 16.48000526,  0.7373936, 36.2612322,  53.0753743,   6,0.5,2,0.02,41]#,0,2,0]#,41]

#params = [17.738, 16.203, 0.705, 148.429/4, 203.749/4, 5.406, 0.465, 2.113, 0.021, 41.448]


#params = [17.818077733032688,  16.4826757097638,  0.7091918021982203,  37.50233239829876, 0.010338645522967208, 52.776271663439104, 6.01840256187061, 0.437535230559547, -1.9654419379618902, 0.021216109151616678, 41.022227286329695]

#params = [17.765663698730954, 16.29351394700684, 0.7070629907584295, 37.18743044048488, 51.26674129059006, 5.50687538812146, 0.374668278239347, -1.684510336003523, 0.02465263029156557, 0.39880670506207166, 41.022227286329695]

#params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

"""
resultmin = op.minimize(minimizefunctionDZ10, params, method='L-BFGS-B', options={'ftol':3e-9})

ftol = 3e-9
tmp_i = np.zeros(len(resultmin.x))
for i in range(len(resultmin.x)):       # getting uncertainty (taken from https://stackoverflow.com/questions/43593592/errors-to-fit-parameters-of-scipy-optimize)
    tmp_i[i] = 1.0
    hess_inv_i = resultmin.hess_inv(tmp_i)[i]
    uncertainty_i = np.sqrt(max(1, abs(resultmin.fun)) * ftol * hess_inv_i)
    tmp_i[i] = 0.0
    print(i, resultmin.x[i], uncertainty_i)


#factr = op.fmin_l_bfgs_b()   # ???
#print(factr * np.finfo(float).eps)

print(minimizefunctionDZ10(params))

"""