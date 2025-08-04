from numpy import sin, cos, array,arange
from scipy.integrate import solve_ivp

def get_nonlinear_solution(x0,machine_param,input_vector,t0,t,):
    Xd,Xq,Xdd,Xqd,Tdod,Tqod,H,D = machine_param
    Tm,Ef,Vg,ws = input_vector
    dt = 0.0001
    t_eval = arange(t0,t,dt)
    print("Machine parameters:", machine_param)
    print("Inpute Vector:", input_vector)
    print("Initial Condition", x0)
    def system_of_equations(t, x):
    #state variables
        delta,w,edd,eqd = x
        id,iq = -(Vg*cos(delta)-eqd)/Xdd, (Vg*sin(delta)-edd)/Xqd
        Te = edd*id + eqd*iq + (Xqd - Xdd)*id*iq    

        delta_dot = w-ws
        w_dot = (1/(2*H*ws))*(((Tm - Te))-D*(w-ws))
        edd_dot = (1/Tqod)*(-edd + (Xq-Xqd)*iq)
        eqd_dot = (1/Tdod)*(Ef - eqd - (Xd - Xdd)*id)
        
        return array([delta_dot,w_dot,edd_dot,eqd_dot])
    
    sol = solve_ivp(system_of_equations,[t0,t],y0 = x0,t_eval=t_eval)
    print("Steady State Value:", sol.y.T[-1])
    return sol.t, sol.y