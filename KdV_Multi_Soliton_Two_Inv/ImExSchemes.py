#!/usr/bin/env python
# coding: utf-8
import numpy as np

# Imex scheme
def ImEx_schemes(s,p,emp,sch_no):
    #2nd order ImEx with b and 1st order ImEx with bhat. This method is taken from Implicit-explicit 
    # Runge-Kutta methods for time-dependent partial differential equations by Ascher, Steven Ruuth and Spiteri.
    if s==2 and p==2 and emp == 1 and sch_no == 1:
        rkim = np.array([ [0,   0],
                          [0, 1/2] ])
        rkex = np.array([ [  0,    0],
                          [1/2,    0] ])
        c = sum(rkex.T);
        b = np.array([0, 1])
        bhat = np.array([1/2, 1/2])
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
    #5th order ImEx with b and 4th order ImEx with bhat. This method is taken from Additive Runge–Kutta schemes 
    #for convection–diffusion–reaction equations by Kennedy and Carpenter.       
    elif s == 8 and p == 5 and emp == 4 and sch_no == 5: 
        rkim = np.array([   [                            0,                            0,                             0,                              0,                            0,                              0,                             0,                            0],
                            [                       41/200,                       41/200,                             0,                              0,                            0,                              0,                             0,                            0],
                            [                       41/400, -567603406766/11931857230679,                        41/200,                              0,                            0,                              0,                             0,                            0],
                            [   683785636431/9252920307686,                            0,   -110385047103/1367015193373,                         41/200,                            0,                              0,                             0,                            0],
                            [ 3016520224154/10081342136671,                            0, 30586259806659/12414158314087, -22760509404356/11113319521817,                       41/200,                              0,                             0,                            0],
                            [   218866479029/1489978393911,                            0,    638256894668/5436446318841,   -1179710474555/5321154724896,   -60928119172/8023461067671,                         41/200,                             0,                            0],
                            [  1020004230633/5715676835656,                            0, 25762820946817/25263940353407,   -2161375909145/9755907335909,  -211217309593/5846859502534,   -4269925059573/7827059040749,                        41/200,                            0],
                            [  -872700587467/9133579230613,                            0,                             0,   22348218063261/9555858737531, -1143369518992/8141816002931, -39379526789629/19018526304540, 32727382324388/42900044865799,                       41/200]
                        ])

        rkex = np.array([   [                             0,                          0,                             0,                              0,                            0,                              0,                             0,                            0],
                            [                        41/100,                          0,                             0,                              0,                            0,                              0,                             0,                            0],
                            [    367902744464/2072280473677, 677623207551/8224143866563,                             0,                              0,                            0,                              0,                             0,                            0],
                            [  1268023523408/10340822734521,                          0,  1029933939417/13636558850479,                              0,                            0,                              0,                             0,                            0],
                            [  14463281900351/6315353703477,                          0,  66114435211212/5879490589093,  -54053170152839/4284798021562,                            0,                              0,                             0,                            0],
                            [ 14090043504691/34967701212078,                          0, 15191511035443/11219624916014, -18461159152457/12425892160975,  -281667163811/9011619295870,                              0,                             0,                            0],
                            [ 19230459214898/13134317526959,                          0,  21275331358303/2942455364971,  -38145345988419/4862620318723,                         -1/8,                           -1/8,                             0,                            0],
                            [-19977161125411/11928030595625,                          0, -40795976796054/6384907823539, 177454434618887/12078138498510,   782672205425/8267701900261,  -69563011059811/9646580694205,   7356628210526/4942186776405,                            0]
                        ])

        c = np.array([0, 41/100, 2935347310677/11292855782101, 1426016391358/7196633302097, 92/100, 24/100, 3/5, 1])
        b = np.array([   -872700587467/9133579230613,                          0,                             0,   22348218063261/9555858737531, -1143369518992/8141816002931, -39379526789629/19018526304540, 32727382324388/42900044865799,                       41/200])
        bhat = np.array([-975461918565/9796059967033,                          0,                             0,  78070527104295/32432590147079,  -548382580838/3424219808633, -33438840321285/15594753105479,   3629800801594/4656183773603, 4035322873751/18575991585200])
        
    return rkim, rkex, c, b, bhat 
