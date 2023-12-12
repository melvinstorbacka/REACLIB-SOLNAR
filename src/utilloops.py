# By Karl Sallmén

from numba import jit
import numpy as np


#@jit(nopython=True)
def configloop(part):
    Dpl = [np.int_(x) for x in range(0)]
    Dpldef = [np.int_(x) for x in range(0)]
    pvec = [np.int_(x) for x in range(0)]
    pvecdef = [np.int_(x) for x in range(0)]
    zjpl = [np.int_(x) for x in range(0)]
    zrpl = [np.int_(x) for x in range(0)]
    zjpldef = [np.int_(x) for x in range(0)]
    zrpldef = [np.int_(x) for x in range(0)]  
    p = 0
    #print(p,'p')
    safe = part
    #print(part)
    while part > 0:
        
        temp = 2*(p+1) + p*(p+1)
        
        #print(temp)
        if part >= temp:
            zjpl.append(2*(p+1))
            zrpl.append(p*(p+1))    
            part = part-temp
            
            p = p+1
        elif part < temp and part != 0:
            if part <= 2*(p+1):
                zjpl.append(part)
                part = part - part
                
            elif part > 2*(p+1) and part < temp:
                zjpl.append(2*(p+1))
                part = part - 2*(p+1)
                zrpl.append(part)
                part = part-part  
                
            else:
                print('something wrong')
                print(part,'part1')
                break     
        if len(zjpl)-len(zrpl) == 1:
            zrpl.append(0)
    
    for i in zjpl:
        zjpldef.append(i)
    for j in zrpl:
        zrpldef.append(j)
    #zrpldef = zrpl
    zrpldef[-1] = zrpldef[-1] - 4
    zjpldef.append(4)
    if len(zjpldef) - len(zrpldef) ==1:
        zrpldef.append(0)
    '''
    if zrpl[-1] != 0:
        zrpl[-1] = zrpl[-1] - 4
        zjpl.append(4)
    elif zrpl[-1] == 0:
        zjpl[-1] = zjpl[-1] - 4
    '''    
    if safe >= 3:
        if zrpl[-1] == 0:
            val = zjpl[-1] + zrpl[-2]
            degval = 2*((len(zjpl)-1)+1) + (len(zrpl)-2)*((len(zrpl)-2)+1)
            valh = degval-val
            pp=len(zjpl)-1
        else:
            val = zrpl[-1]
            degval = 2*((len(zjpl))+1) + (len(zrpl)-1)*((len(zrpl)-1)+1)   
            valh = degval-val
            pp=len(zjpl)-1 
    else:
        val = zjpl[-1]
        degval =  2
        valh = degval - val
        pp = len(zjpl)-1
    for i in range(len(zjpl)):
        pvec.append(i)
        Dpl.append((i+1)*(i+2))
    zpl = [sum(x) for x in zip(zrpl, zjpl)]
    zpldef= [sum(x) for x in zip(zrpldef, zjpldef)]            
    for i in range(len(zjpldef)):
        pvecdef.append(i)
        Dpldef.append((i+1)*(i+2))
    degvaldef = 2*((len(zjpldef)-1)+1) + (len(zrpldef)-2)*((len(zrpldef)-2)+1)

    return [zpl,zpldef,zjpl,zrpl,zjpldef,zrpldef,Dpl,Dpldef,val, valh, degval,pp,pvec,pvecdef, degvaldef]
