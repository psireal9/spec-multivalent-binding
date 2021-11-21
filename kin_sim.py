import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint, solve_ivp

# bivalent case
cRTotal = 0.071
cLTotal = 0.7
t = 2
kon1 = 136.4185
koff1 = 2.3716
KD1 = koff1/kon1
koff2 =18.5925
kon2 = 33.4088
numOfSpecies = 4

def cR(t, cRTotal=cRTotal):
    return cRTotal - cLR1(t) - cLR2(t)

def cL(t, cLTotal=cLTotal):
    return cLTotal - cLR1(t) - cLR2(t)

def cLR1(t, cRTotal=cRTotal, KD1=KD1):
    return (cRTotal*cL(t)-cLR2(t)*cL(t))/(KD1+cLR1(t))

def cLR2(t, KD1=KD1, koff2=koff2, kon2=kon2, cRTotal=cRTotal):
    alpha = (kon2*cL(t) + KD1*koff2 + koff2*cL(t))/(KD1+cL(t))
    beta = kon2*cRTotal*cL(t)/(KD1+cL(t))
    return (beta/alpha)*(1-np.exp(-alpha*t))

# solve the system dy/dt = f(y,t)
def f(y, t):
    L, R, LR1, LR2 = y[0], y[1], y[2], y[3]
    # the model equations 

    f0 = -kon1*L*R + koff1*LR1
    f1 = -kon1*L*R + koff1*LR1
    f2 = kon1*L*R - koff1*LR1 - kon2*LR1 + koff2*LR2
    f3 = kon2*LR1 - koff2*LR2
    
    return [f0,f2,f2,f3]

def f1(t, state, kon1, koff1, kon2, koff2):
    L, R, LR1, LR2 = state
    # the model equations 

    stateL = -kon1*L*R + koff1*LR1
    strateR = -kon1*L*R + koff1*LR1
    stateLR1 = kon1*L*R - koff1*LR1 - kon2*LR1 + koff2*LR2
    stateLR2 = kon2*LR1 - koff2*LR2
    
    return [stateL,strateR,stateLR1,stateLR2]

# initial conditions - erste Injektion
L0 = 0.0485
R0 = 0.030588
LR1_0 = 0.00031
LR2_0 = 0.00043 - LR1_0
y0 = [L0,R0,LR1_0,LR2_0]
param = (kon1, koff1, kon2, koff2)
t  = np.linspace(0, 40, 1000)

# solve the DEs with FORTRAN `lsoda`
# 'LSODA': Adams/BDF method with automatic stiffness detection and
#              switching [7]_, [8]_. This is a wrapper of the Fortran solver
#              from ODEPACK.
soln = odeint(f,y0,t)
L, R, LR1, LR2 = (soln[:,i] for i in range(numOfSpecies))

# solve the DEs with Runge-Kutta (5th order)
# 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
#               The error is controlled assuming accuracy of the fourth-order
#               method, but steps are taken using the fifth-order accurate
#               formula (local extrapolation is done). A quartic interpolation
#               polynomial is used for the dense output [2]_. Can be applied in
#               the complex domain.
t_span= (0,40)
soln = solve_ivp(f1,t_span, y0, args=param)
L, R, LR1, LR2 = (soln.y[i,:] for i in range(numOfSpecies))


plt.figure()
plt.plot( L, label='L')
plt.plot(R, label='R')
plt.plot(LR1, label='LR1')
plt.plot(LR2, label='LR2')
plt.legend()
plt.show()

