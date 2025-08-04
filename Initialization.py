
from sympy import symbols, nsolve,sin,cos
import numpy as np

edd,eqd,id,iq,delta,w=symbols(['e_dd','e_qd','i_d','i_q','delta','omega'])
def find_initial(machine_param,input_vector):
    Xd,Xq,Xdd,Xqd,Tdod,Tqod,H,D = machine_param
    Tm,Ef,Vg,ws = input_vector
    
    f1 = eqd - Vg*cos(delta)-id*Xdd
    f2 = edd + Xqd*iq - Vg*sin(delta)
    f3 = Tm - (edd*id + eqd*iq + (Xqd - Xdd)*id*iq)
    f4 = edd - (Xq-Xqd)*iq
    f5 = eqd + (Xd-Xdd)*id - Ef
    f6 = w-ws

    xsol = nsolve((f1, f2,f3,f4,f5,f6), (id,iq,delta,edd,eqd,w), (1,1,1,1,1,1))
    id0,iq0,delta0,edd0,eqd0,w0 = xsol.values()
    x0 = np.array([delta0,w0,edd0,eqd0]).astype('float')
    alg0 = np.array([id0,iq0]).astype('float')
    print("Steady state value: delta0,w0,edd0,eqd0:",x0)
    print("Steady state algebraic solution:", alg0)
    return alg0,x0