from sympy import symbols, Matrix, sin,cos,zeros,diff
import numpy as np
from scipy.integrate import solve_ivp

def get_matrix(der,stat):
    mat = zeros(len(der),len(stat))
    for i in range(len(der)):
        for j in range(len(stat)):
            mat[i,j] = diff(der[i],stat[j])
    return mat


delta,w,edd,eqd = symbols(['delta','omega','e_dd','e_qd'])
H,D,Tdod,Tqod,Xdd,Xqd, Xd,Xq = symbols(['H','D','T_dod','T_qod','X_dd','X_qd','X_d','X_q'])
Tm,Ef,Vg,ws = symbols(['T_m','E_f','V_g','omega_s'])
id,iq = symbols(['i_d','i_q'])

def get_differential_equation():
    delta_dot = w-ws
    w_dot = (1/(2*H*ws))*((Tm - (edd*id + eqd*iq + (Xqd - Xdd)*id*iq)) - D*(w-ws))
    edd_dot = (1/Tqod)*(-edd + (Xq-Xqd)*iq)
    eqd_dot = (1/Tdod)*(Ef - eqd - (Xd - Xdd)*id)

    return Matrix([delta_dot,w_dot,edd_dot,eqd_dot])

def get_algebraic_equation():
    a1 = (-(Vg*cos(delta)-eqd)/Xdd) - id
    a2 = ((Vg*sin(delta)-edd)/Xqd) - iq
    return Matrix([a1,a2])

F = get_differential_equation()
G = get_algebraic_equation()
X = Matrix([delta,w,edd,eqd])
U = Matrix([Tm,Ef,Vg,ws])
Y = Matrix([id,iq])

def find_state_matrix():
    A,B,E = get_matrix(F,X), get_matrix(F,Y), get_matrix(F,U)
    C,D,H = get_matrix(G,X), get_matrix(G,Y), get_matrix(G,U)
    Aeff = A - B*D.inv()*C
    Beff = E - B*D.inv()*H
    return Aeff,Beff

def get_state_matrix(X0, machine_param,input_vector_init,alg_vector):
    delta0,w0,edd0,eqd0 = X0
    Xd0,Xq0,Xdd0,Xqd0,Tdod0,Tqod0,H0,D0 = machine_param
    Tm0,Ef0,Vg0,ws0 = input_vector_init
    id0,iq0 = alg_vector 

    A,B= find_state_matrix()
    Amm = A.subs([(delta,delta0),(w,w0),(edd,edd0),(eqd,eqd0),
              (H,H0),(Xq,Xq0),(Xqd,Xqd0),(Xdd,Xdd0),(Xd,Xd0),
              (Xqd,Xqd0),(ws,ws0),(Tdod,Tdod0),(Tqod,Tqod0),(Vg,Vg0),(D,D0),(id,id0),(iq,iq0)])

    Bmm = B.subs([(ws,ws0),(Ef,Ef0),(Vg,Vg0),(Tm,Tm0),(id,id0),(iq,iq0)])
    Bmm = Bmm.subs([(delta,delta0),(w,w0),(edd,edd0),(eqd,eqd0),
                (H,H0),(Xq,Xq0),(Xqd,Xqd0),(Xdd,Xdd0),(Xd,Xd0),
                (Xqd,Xqd0),(Tdod,Tdod0),(Tqod,Tqod0),(D,D0)])
    
   
    Am = np.array(Amm).astype(np.float64)
    Bm = np.array(Bmm).astype(np.float64)
   
    return Am,Bm

def get_linear_solution(A,B,x0,input_vector,t0,t):
    dt = 0.0001
    t_eval = np.arange(t0,t,dt)
    print("Initial Condition", x0)
    def o_model(t,x):
        x_dot = A @ x   + B @ input_vector
        return x_dot

    sol = solve_ivp(o_model,[t0,t],y0 = x0,t_eval=t_eval)
    print("Steady State Value:", sol.y.T[-1])
    return sol.t, sol.y