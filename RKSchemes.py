#!/usr/bin/env python
# coding: utf-8

import numpy as np
from nodepy import rk

# Second order SSPRK(2,2) method with a first order embedding
ssp22 = rk.loadRKM('SSP22').__num__()
ssp22.bhat = np.array([1/3, 2/3])

# Third order Heun(3,3) method with a second order embedding
heun33 = rk.loadRKM('Heun33').__num__()
heun33.bhat = np.array([0.006419303047187, 0.487161393905626, 0.506419303047187])

# Third order SSPRK(3,3) method with second order embeddings, one 2nd order embedding (with bhat) is given
ssp33 = rk.loadRKM('SSP33').__num__()
ssp33.b3 = np.array([0.395011932394815, 0.395011932394815, 0.209976135210371])

# Fourth order classical RK(4,4) method with a second order embedding. 3rd order embedding is not possible
rk44 = rk.loadRKM('RK44').__num__()
rk44.bhat = np.array([0.25,0.25,0.25,0.25])

# Fourth order Fehlberg(6,4) method with third order embeddings. The default Fehlberg method is 5th order
# with given b and 4th order with bhat. Here we use the 4th order method with bhat as our base method and 
# provide other 3rd order embedding methods with bhat and b3.
fehlberg45 = rk.loadRKM("Fehlberg45").__num__()
fehlberg45.b = fehlberg45.bhat
fehlberg45.bhat = np.array([0.122702088570621, 0.000000000000003, 0.251243531398616, -0.072328563385151, 0.246714063515406, 0.451668879900505])
fehlberg45.b3 = np.array([0.150593325320835, 0.000000000000003, 0.275657325006399, 0.414789231909538, -0.131467847351019, 0.290427965114243])

# Fifth order DP(7,5) method with a fourth order embedding (bhat), which is already given and (A,b3) is also 
# 4th order accurate
dp75 = rk.loadRKM('DP5').__num__()
dp75.b3 = np.array([0.159422044716717, 0.000000000000009, 0.310936711045800, 0.444052776789396, 0.307005319740028, -0.230738637667449, 0.009321785375499])

# Imex scheme
def ImEx_schemes(s,p,emp,sch_no):
    #3rd order ImEx with b and 2nd order ImEx with bhat. This method is taken from Implicit-explicit 
    # Runge-Kutta methods for time-dependent partial differential equations by Ascher, Steven Ruuth and Spiteri.
    if s == 4 and p == 3 and emp == 2 and sch_no == 2:
        rkim = np.array([ [ 0,              0,             0,             0],
                          [ 0,   0.4358665215,             0,             0],
                          [ 0,   0.2820667392,  0.4358665215,             0],
                          [ 0,   1.2084966490, -0.6443631710,  0.4358665215] ])
        rkex = np.array([ [ 0,                       0,             0,             0],
                          [ 0.4358665215,            0,             0,             0],
                          [ 0.3212788860, 0.3966543747,             0,             0],
                          [-0.1058582960, 0.5529291479,  0.5529291479,             0] ])
        c = sum(rkex.T)
        b = np.array([0,   1.2084966490, -0.6443631710,  0.4358665215])
        bhat = np.array([0,0.886315063820486,0 ,0.113684936179514])
    #3rd order ImEx with b and 2nd order ImEx with bhat. This method is taken from Additive Runge–Kutta schemes 
    #for convection–diffusion–reaction equations by Kennedy and Carpenter.
    if s == 4 and p == 3 and emp == 2 and sch_no == 3:
        rkim = np.array([ [ 0,              0,             0,             0],
                  [1767732205903/4055673282236, 1767732205903/4055673282236, 0, 0],
                  [2746238789719/10658868560708, -640167445237/6845629431997, 1767732205903/4055673282236, 0],              
                  [1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821,                 1767732205903/4055673282236] ])

        rkex = np.array([ [0,                            0,         0,             0],
                          [1767732205903/2027836641118,  0,         0,             0],
                          [5535828885825/10492691773637, 788022342437/10882634858940, 0, 0],
                          [6485989280629/16251701735622, -4246266847089/9704473918619, 10755448449292/10357097424841, 0] ])

        c = np.array([0, 1767732205903/2027836641118, 3/5, 1])
        b = np.array([1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821, 1767732205903/4055673282236])
        bhat = np.array([2756255671327/12835298489170, -10771552573575/22201958757719, 9247589265047/10645013368117, 2193209047091/5459859503100])

    #4th order ImEx with b and 3rd order ImEx with bhat. This method is taken from Additive Runge–Kutta schemes 
    #for convection–diffusion–reaction equations by Kennedy and Carpenter.
    elif s == 6 and p == 4 and emp == 3 and sch_no == 4:
        rkex = np.array([ [0, 0, 0, 0, 0, 0],
                  [1/2, 0, 0, 0, 0, 0],
                  [13861/62500, 6889/62500, 0, 0, 0, 0], 
                  [-116923316275/2393684061468, -2731218467317/15368042101831, 9408046702089/11113171139209, 0, 0, 0], 
                  [-451086348788/2902428689909, -2682348792572/7519795681897, 12662868775082/11960479115383, 3355817975965/11060851509271, 0, 0], 
                  [647845179188/3216320057751, 73281519250/8382639484533, 552539513391/3454668386233, 3354512671639/8306763924573, 4040/17871, 0] 
                ])

        rkim = np.array([ [0, 0, 0, 0, 0, 0],
                          [1/4, 1/4, 0, 0, 0, 0],
                          [8611/62500, -1743/31250, 1/4, 0, 0, 0],
                          [5012029/34652500, -654441/2922500, 174375/388108, 1/4, 0, 0],
                          [15267082809/155376265600, -71443401/120774400, 730878875/902184768, 2285395/8070912, 1/4, 0],
                          [82889/524892, 0, 15625/83664, 69875/102672, -2260/8211, 1/4]
                        ])

        c = np.array([0, 1/2, 83/250, 31/50, 17/20, 1])
        b = np.array([82889/524892, 0, 15625/83664, 69875/102672, -2260/8211, 1/4])
        bhat = np.array([4586570599/29645900160, 0, 178811875/945068544, 814220225/1159782912, -3700637/11593932, 61727/225920])
        
    return rkim, rkex, c, b, bhat 