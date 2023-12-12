# By Karl Sallmén

from numba import jit
import numpy as np
from math import sqrt
import inspect



def rho(N,Z):
    A = N+Z
    t = abs(N-Z)/2
    return (A**(1/3))*((1-(1*((t/A)**2)))**2)
    
#Coulomb term p['aC'] (C)
@jit(nopython=True)
def Coulomb(z,A,T):
    return (((-z*(z-1)) + 0.76*((z*(z-1))**(2/3))))/((A**(1/3))*(1-0.25*((T/A)**2)))

#Check vectors in spin orbit terms

#spin orbit term p['aSO'] (M+S)
@jit(nopython=True)
def SpinOrbit(npa,zpa,Dnpa,Dzpa,pn,pz,njp,nrp,zrp,zjp):                                                                                                                                                           #SO=SpinOrbit(npa,zpa,Dnpa,Dzpa,pna,pza,njpa,nrpa,zrpa,zjpa,rho) SpinOrbit(npa,zpa,Dnpa,Dzpa,pn,pz,njp,nrp,zrp,zjp,R):
    return ( (np.sum((((pn*njp)/2)-nrp)*(((1+(npa/(Dnpa**0.5)))*(pn**2/(Dnpa**(3/2)))) \
            + (1-(npa/(Dnpa**0.5)))*((4*pn-5)/(Dnpa**(3/2)))))) \
                + (np.sum((((pz*zjp)/2)-zrp)*(((1+(zpa/(Dzpa**0.5)))*(pz**2/(Dzpa**(3/2)))) \
                    + (1-(zpa/(Dzpa**0.5)))*((4*pz-5)/(Dzpa**(3/2)))))))


# master surface term -p['aMSurf'] (M)
@jit(nopython=True)
def Master(npa,zpa,Dnpa,Dzpa,rho):
    return (1/rho)*((np.sum(npa/(Dnpa**0.5))**2) + (np.sum(zpa/(Dzpa**0.5))**2))


#need to fix symmetry terms (T + TS)
# symmetry term p['aSym1'] p['aSym2']
@jit(nopython=True)
def Symmetry1(R,t,A):
    #return ((4*T*(T+1))/(R*(A**(2/3)))) 
    #print((t*(t+2)))
    #print((A**(2/3))
    return -(t*(t+2))/(A**(2/3))*(1/R)
@jit(nopython=True)
def Symmetry2(R,T,A):
    #return -((4*T*(T+1))/((A**(2/3))*(R**2)) + (T*(1-T))/(A*(R**3)))
    return ((1/R)*(T*(T+2))/(A**(2/3)))*(1/R)
'''
def Symmetry1(R,T,A):
    return ((T*(T+1))/(R*(A**(1)))) 

def Symmetry2(R,T,A):
    return -(T*(T+1))/((A**(4/3))*(R**2))
'''
#Spherical terms
@jit(nopython=True)
def Spherical3(n,nbar,Dn,z,zbar,Dz,rho):
    return (1/rho)*((n*nbar*(n-nbar))/Dn + (z*zbar*(z-zbar))/Dz)
@jit(nopython=True)
def Spherical4(n,nbar,Dn,z,zbar,Dz,pn,pz,rho):
    #return (1/rho)*((((2**(sqrt(pn)+sqrt(pz)))*(n*nbar))/Dn)*(((1)*z*(zbar))/Dz))
    #return (1/rho)*((2**(sqrt(pn)+sqrt(pz)))*((n*nbar)/Dn)*((z*zbar)/Dz)) 
    return (1/rho)*(((2**sqrt(pn))*n*nbar)/Dn)*(((2**sqrt(pz))*z*zbar)/Dz)
'''
def Spherical4HO():

def Spherical41(n,nbar,z,zbar,Nn,Nz):
    return ((n*nbar)/(sqrt(Nn)))*((z*zbar)/(sqrt(Nz)))

def Spherical42(n,nbar,z,zbar,Nn,Nz):
    return ((n*nbar)/(Nn))*((z*zbar)/(Nz))
def Spherical41p():


'''



#p['aSph1']*(1/rho)*S3
#p['aSph2']*(1/(rho**2))*S3
#p['aSph3']*(1/rho)*S4


#Deformed and pairing term need to be added, Chong had an idea for a modification to these from the original DZ
@jit(nopython=True)
def Deformed(n,nbar,Dn,z,zbar,Dz,rho):
    #npri = n - 4 # 4 particles being promoted
    #nbarpri = nbar + 4 
    #zpri = z - 4
    #zbarpri = z +4

    #return (1/rho)*((npri*nbarpri)/(Dn**(3/2)))*((zpri*zbarpri)/(Dz**(3/2)))
    return (1/rho)*(((n*(Dn-n-4)/(Dn**(3/2))))*((z*(Dz-z-4))/(Dz**(3/2))))
'''
def DeformedC1(n,nbar,z,zbar,Nn,Nz):

def DeformedC2(n,nbar,z,zbar,Dn,Dz):





'''


@jit(nopython=True)
def Pairing(N,Z,rho):
    A = N+Z
    
    if (N % 2) == 0:
        Neven = True
        Nodd = False
    else:
        Nodd = True
        Neven = False
    if (Z % 2) == 0:
        Zeven = True
        Zodd = False
    else:
        Zodd = True
        Zeven = True

    if Nodd == True and Zodd == True:
        return 0
    elif Nodd == True and Zeven == True:
        return 1/rho  
    elif Neven == True and Zodd == True:
        return 1/rho   
    elif Neven == True and Zeven == True:
        return 2/rho
    else:
        print('Something went wrong with pairing')
        return 0                      


#print(inspect.getargspec(Pairing).args[2])
