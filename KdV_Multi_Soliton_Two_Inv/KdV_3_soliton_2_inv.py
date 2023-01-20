#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.optimize import fsolve,root,minimize,brute,fmin,fmin_slsqp

#-----------#
from RKSchemes import ImEx_schemes


# In[ ]:


xL = -130 # left end point of the domain 
xR = 130 # right end point of the domain 
L = xR-xL # length of the domain
N = 1536 # number of grid points
# 3-soliton solution parameter
b1 = 0.4; b2 = 0.7; b3 = 1
    
xplot = np.linspace(xL, xR, N+1)
x = xplot[0:-1] 
dx = x[1] - x[0]
xi = np.fft.fftfreq(N) * N * 2*np.pi / L

# required matrix
I = np.eye(N)

def F_matrix(N):
    F = np.zeros((N,N),dtype=complex)
    for j in range(N):
        v = np.zeros(N)
        v[j] = 1.
        F[:,j] = np.fft.fft(v)
    return F

dft_mat = F_matrix(N)
inv_dft_mat = np.linalg.inv(dft_mat)
xi3 = np.tile((-1j)*xi*xi*xi,(N,1)).T
M = np.dot(inv_dft_mat,np.multiply(xi3,dft_mat))

def CompositeSimposonInt(x,f):
    dx = (x[1]-x[0])
    approx = 1/3*dx*(f[0]+4*np.sum(f[1::2])+2*sum(f[2::2][:-1])+f[-1])
    return approx

def eta0(w):
    return np.sum(w) * dx

def eta1(w):
    return 0.5 * np.sum(w*w) * dx

def eta2(w):
    f1 = np.append(w,w[0])**3
    what = np.fft.fft(w)
    wx = np.real(np.fft.ifft(1j*xi*what))
    f2 = np.append(wx,wx[0])**2
    int_f1 = CompositeSimposonInt(xplot,f1)
    int_f2 = CompositeSimposonInt(xplot,f2)
    return 2*int_f1 - int_f2

def kdV_stiff_rhs(u):
    uhat = np.fft.fft(u)
    return -np.real(np.fft.ifft(-1j*xi*xi*xi*uhat))

def kdV_non_stiff_rhs(u): # Spectral and SBP
    uhat = np.fft.fft(u)
    u2hat = np.fft.fft(u*u)
    return -6*(u * np.real(np.fft.ifft(1j*xi*uhat)) + np.real(np.fft.ifft(1j*xi*u2hat))) / 3.

def analytical_sol(x,t,b1,b2,b3):
    # 3-soliton solution
    X1 = np.sqrt(b1/2)*(x - 2*b1*t); X2 = np.sqrt(b2/2)*(x - 2*b2*t); X3 = np.sqrt(b3/2)*(x - 2*b3*t)  
    b1sqrt = np.sqrt(b1); b2sqrt = np.sqrt(b2); b3sqrt = np.sqrt(b3)
    sech2X1 = 1/np.cosh(X1)**2; sech2X3 = 1/np.cosh(X3)**2; cosech2X2 = 1/np.sinh(X2)**2;
    tanhX1 = np.tanh(X1); tanhX3 = np.tanh(X3); cothX2 = np.cosh(X2)/np.sinh(X2);
    Num1 = 2*(b3-b1)*(b3*sech2X3 - b1*sech2X1) / (np.sqrt(2)*b3sqrt*tanhX3 - np.sqrt(2)*b1sqrt*tanhX1)**2
    Num2 = 2*(b1-b2)*(b2*cosech2X2 + b1*sech2X1) / (np.sqrt(2)*b1sqrt*tanhX1 - np.sqrt(2)*b2sqrt*cothX2)**2
    Num = 2*(b2 - b3)*(Num1 - Num2)
    Den1 = 2*(b1 - b2) / (np.sqrt(2)*b1sqrt*tanhX1 - np.sqrt(2)*b2sqrt*cothX2)
    Den2 = 2*(b3-b1) / (np.sqrt(2)*b3sqrt*tanhX3 - np.sqrt(2)*b1sqrt*tanhX1)
    Den = (Den1 - Den2)**2
    return b1*sech2X1 - Num/Den


# In[ ]:


def Compute_Sol_Without_Relaxation(Mthdname,rkim, rkex, c, b, bhat, dt, f_stiff, f_non_stiff, T, u0, t0):    
    tt = np.zeros(1) 
    t = t0; tt[0] = t
    
    uu = np.zeros((1,np.size(u0))) 
    uu[0,:] = u0.copy()
    
    s = len(rkim)
    Rim = np.zeros((s,len(u0)))
    Rex = np.zeros((s,len(u0))) 
    steps = 0
    while t < T and not np.isclose(t, T):
        clear_output(wait=True)
        if t + dt > T:
            dt = T - t
        for i in range(s):
            rhs = uu[-1].copy()
            if i>0:
                for j in range(i):
                    rhs += dt*(rkim[i,j]*Rim[j,:] + rkex[i,j]*Rex[j,:])

            Mat = I + dt*rkim[i,i]*M
            g_j = np.linalg.solve(Mat, rhs)
            Rim[i,:] = f_stiff(g_j)
            Rex[i,:] = f_non_stiff(g_j)

        inc = dt*sum([ b[j]*(Rim[j]+Rex[j]) for j in range(s)])    
        unew = uu[-1]+inc; t+= dt
        tt = np.append(tt, t)
        steps += 1
        uu = np.append(uu, np.reshape(unew.copy(), (1,len(unew))), axis=0)  
        
        print("Method = Baseline %s: Step number = %d (time = %1.2f)"%(Mthdname,steps,tt[-1]))
        
    return tt, uu


# In[ ]:


def rgam(gammas,u,inc1,E1_old,inc2,E2_old):
    gamma1, gamma2 = gammas
    uprop = u + gamma1*inc1 + gamma2*inc2  
    E1 = eta1(uprop); E2 = eta2(uprop)
    return np.array([E1-E1_old,E2-E2_old])

def norm_rgam(gammas,u,inc1,E1_old,inc2,E2_old):
    gamma1,gamma2 = gammas
    uprop = u + gamma1*inc1 + gamma2*inc2   
    E1 = eta1(uprop); E2 = eta2(uprop)
    return np.linalg.norm(np.array([E1-E1_old,E2-E2_old]))

def compute_sol_multi_relaxation(Mthdname, rkim, rkex, c, b, bhat, dt, f_stiff, f_non_stiff, T, u0, t0):
    tt = np.zeros(1) 
    t = t0; tt[0] = t

    uu = np.zeros((1,np.size(u0)))
    uu[0,:] = u0.copy()

    s = len(rkim)
    Rim = np.zeros((s,len(u0))) 
    Rex = np.zeros((s,len(u0))) 
    G1 = np.array([]); G2 = np.array([]);
    no_inv = 2; gamma0 = np.zeros(no_inv)

    steps = 0; no_ier_five = 0; no_ier_one = 0; no_ier_else = 0

    while t < T and not np.isclose(t, T):
        clear_output(wait=True)
        if t + dt > T:
            dt = T - t
        for i in range(s):
            rhs = uu[-1].copy()
            if i>0:
                for j in range(i):
                    rhs += dt*(rkim[i,j]*Rim[j,:] + rkex[i,j]*Rex[j,:] )

            Mat = I + dt*rkim[i,i]*M
            g_j = np.linalg.solve(Mat, rhs)
            Rim[i,:] = f_stiff(g_j)
            Rex[i,:] = f_non_stiff(g_j)
        
        inc1 = dt*sum([ b[i]*(Rim[i]+Rex[i]) for i in range(s)])  
        inc2 = dt*sum([ bhat[i]*(Rim[i]+Rex[i]) for i in range(s)]) 
        unew = uu[-1]+inc1; E1_old = eta1(uu[-1]); E2_old = eta2(uu[-1])
        
        # fsolve
        ga_fsolve, info, ier, mesg = fsolve(rgam,gamma0,args=(unew,inc1,E1_old,inc2,E2_old),full_output=True) 
                                     
        if ier == 1:
            gamma1, gamma2 = ga_fsolve; gamma0 = ga_fsolve
        else: 
            # brute followed by fmin
            rranges = (slice(0, 2, 0.1), slice(0, 2, 0.1))
            ga_brute = brute(norm_rgam, rranges, args=(unew,inc1,E1_old,inc2,E2_old), full_output=True,
                               finish=fmin)
            brute_fval = norm_rgam(ga_brute[0],unew,inc1,E1_old,inc2,E2_old)
            fsolve_fval = norm_rgam(ga_fsolve,unew,inc1,E1_old,inc2,E2_old)
            if brute_fval<fsolve_fval:
                gamma1, gamma2 = ga_brute[0]; gamma0 = ga_brute[0]
            else:
                gamma1, gamma2 = ga_fsolve; gamma0 = ga_fsolve
                
        steps += 1
        if ier == 1:
            no_ier_one += 1
        elif ier == 5:
            no_ier_five += 1
        else:
            no_ier_else += 1

        unew = unew + gamma1*inc1 + gamma2*inc2; t+=(1+gamma1+gamma2)*dt
        tt = np.append(tt, t)
        uu = np.append(uu, np.reshape(unew.copy(), (1,len(unew))), axis=0)  
        G1 = np.append(G1, gamma1); G2 = np.append(G2, gamma2)       
        print("Method = Relaxation %s: At step number = %d (time = %1.2f), integer flag for fsolve = %d and γ1+γ2 = %f"%(Mthdname,steps,tt[-1],ier,gamma1+gamma2))

    return tt, uu, G1, G2, no_ier_one, no_ier_five, no_ier_else


# In[ ]:


method_names = ["ImEx32_2","ImEx43"]; DT = [0.01, 0.01]
S = [4,6]; P = [3,4]; em_P = [2,3]; Sch_No = [3,4]

eqn = 'KdV_%d_sol_%d_inv'%(3,2)

# Inputs to solve the system of ODEs
t0 = -50; u0 = analytical_sol(x,t0,b1,b2,b3)
f_stiff = kdV_stiff_rhs; f_non_stiff = kdV_non_stiff_rhs; T = 50; 

data = {'Method': method_names,
        'B: dt': DT,
        'R: dt': DT,
        'Domain':'[%d,%d]'%(xL,xR),
        'N': N,
        't0': t0,
        'tf': T,
        'b1,b2,b3':'%1.1f, %1.1f, %1.1f'%(b1,b2,b3)}
df = pd.DataFrame(data)
df['R: ier = 1'] = np.nan; df['R: ier = 5'] = np.nan; df['R: ier = else'] = np.nan

b_tt = []; b_uu = []; r_tt = []; r_uu = [];  
for idx in range(len(method_names)):
    dt = DT[idx]; rkim, rkex, c, b, bhat = ImEx_schemes(S[idx],P[idx],em_P[idx],Sch_No[idx])
    tt, uu, G1,G2,IF_1,IF_5,IF_else = compute_sol_multi_relaxation(method_names[idx],rkim, rkex, c, b, bhat, dt, f_stiff, f_non_stiff, T, u0, t0)
    df.at[idx,'R: ier = 1'] = int(IF_1); df.at[idx,'R: ier = 5'] = int(IF_5); df.at[idx,'R: ier = else'] = int(IF_else)
    r_tt.append(tt); r_uu.append(uu)
    
for idx in range(len(method_names)):
    dt = DT[idx]; rkim, rkex, c, b, bhat = ImEx_schemes(S[idx],P[idx],em_P[idx],Sch_No[idx])
    tt, uu = Compute_Sol_Without_Relaxation(method_names[idx], rkim, rkex, c, b, bhat,dt, f_stiff, f_non_stiff, T, u0, t0)
    b_tt.append(tt); b_uu.append(uu)


# In[ ]:


def analytical_u_KdV(tvec,u0):
    true_u = np.zeros((len(tvec), len(u0))) 
    for idx in range(len(tvec)):
        true_u[idx,:] = analytical_sol(x,tvec[idx],b1,b2,b3) 
    return true_u

# Computing reference solution corresponding to methods without relaxation
b_UTrue = [];
for idx in range(len(method_names)):
    b_tvec = b_tt[idx]
    b_utrue = analytical_u_KdV(b_tvec,u0)
    b_UTrue.append(b_utrue)
    
# Computing reference solution corresponding to methods with relaxation
r_UTrue = []
for idx in range(len(method_names)):
    r_tvec = r_tt[idx]
    r_utrue = analytical_u_KdV(r_tvec,u0)
    r_UTrue.append(r_utrue)


# In[ ]:


import os
path = '%s'%('Data/%s'%eqn)

import os
if not os.path.exists(path):
   os.makedirs(path)


# In[ ]:


from pathlib import Path  
filepath = Path("./Data/%s/MethodsData_N%d_T%d.csv"%(eqn,N,T),index = False)    
df.to_csv(filepath) 
# Numerical Solution
np.save("./Data/%s/Baseline_Time_N%d_T%d.npy"%(eqn,N,T), b_tt)
np.save("./Data/%s/Baseline_NumSol_N%d_T%d.npy"%(eqn,N,T), b_uu)
np.save("./Data/%s/Relaxation_Time_N%d_T%d.npy"%(eqn,N,T), r_tt)
np.save("./Data/%s/Relaxation_NumSol_N%d_T%d.npy"%(eqn,N,T), r_uu)
# Reference Solution
np.save("./Data/%s/TrueSol_BaselineTime_N%d_T%d.npy"%(eqn,N,T), b_UTrue)
np.save("./Data/%s/TrueSol_RelaxationTime_N%d_T%d.npy"%(eqn,N,T), r_UTrue)


# In[ ]:




