# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 18:12:39 2022

@author: lidel
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interpolatescipy
import imageio                    #for the gif
import os                         #for the gif
from fenics import *
from dolfin import *

#control verbosity of solve function 
#https://fenicsproject.org/qa/810/how-to-disable-message-solving-linear-variational-problem/
set_log_level(21)





#FUNCTIONS

def get_ab(rho,rho_ice,phi_snow,ab_phi_lim,nglen,K):
    
    rhoh = rho/Constant(rho_ice) # normalized density (rho hat)
    rhohsnow, rhohcrit = Constant(phi_snow), Constant(ab_phi_lim)

    f_a0 = lambda rhoh: (1+2/3*(1-rhoh))*rhoh**(-2*nglen/(nglen+1))
    f_b0 = lambda rhoh: 3/4*((1/nglen*(1-rhoh)**(1/nglen))/(1-(1-rhoh)**(1/nglen)))**(2*nglen/(nglen+1))

    gamma_mu = 20*1
    mu = lambda rhoh: 1/(1+exp(-gamma_mu*(rhohcrit*1-rhoh))) # step function (approximated by logistics function)

    gamma_a = lambda k: (ln(k)-ln(f_a0(rhohcrit)))/(rhohcrit-rhohsnow)
    gamma_b = lambda k: (ln(k)-ln(f_b0(rhohcrit)))/(rhohcrit-rhohsnow)
    f_a1 = lambda rhoh,k: k*exp(-gamma_a(k)*(rhoh-rhohsnow))
    f_b1 = lambda rhoh,k: k*exp(-gamma_b(k)*(rhoh-rhohsnow))

    f_a = lambda rhoh,k: f_a0(rhoh) + mu(rhoh)*f_a1(rhoh,k)
    f_b = lambda rhoh,k: f_b0(rhoh) + mu(rhoh)*f_b1(rhoh,k)
    
    a = f_a(rhoh,K)
    b = f_b(rhoh,K)
    
    return a, b

def get_sigma(v,a,b,Aglen,nglen):
    
    eps_dot=v.dx(0)                   #sym(grad(v))
    J1=v.dx(0)                #tr(eps(v))
    J2=eps_dot*eps_dot           #inner(eps_dot,eps_dot)
    eps_E2=1/a*(J2-J1**2/3) + (3/2)*(1/b)*J1**2
    viscosity = (1/2)**((1-nglen)/(2*nglen)) * Aglen**(-1/nglen) * (eps_E2)**((1-nglen)/(2*nglen))    
    sigma = viscosity * (1/a*(eps_dot-(J1/3))+(3/2)*(1/b)*J1)
    
    return sigma

def convert_to_np(U,ufl_array):
    
    zcoord=U.tabulate_dof_coordinates()[:,0] #koordinatuak gorde, ordenatu gabe!
    np_array=ufl_array.vector()              #interesatzen zaigun aldagaiaren balioak gorde, ordenatu gabe!
    I=np.argsort(zcoord)                     #z-ren arabera ordenatzeko indexen ordena gorde 
    sorted_coors=zcoord[I]                   #koordinatuak gorde, ordenatuta!
    sorted_np=np_array[I]                    #interesatzen zaigun aldagaiaren balioak gorde, ordenatuta!

    return sorted_coors,sorted_np

def acc_from_iceeqyr_to_snoweqs(acc,rho_obs):
    
    accum_watereqyr = acc # water equiv. accum, m/yr
    accum_iceeqyr = 1000/rho_ice * accum_watereqyr
    accum_iceeqs = accum_iceeqyr/SecPerYear # ice accum, m/s 
    rho_snow = np.amin(rho_obs) 
    acc_rate = rho_ice/rho_snow * accum_iceeqs
    
    print('adot [metre snow/yr] = %e'%(acc_rate*SecPerYear))
    
    return acc_rate

def get_rho_rawdata(H,site):

    obs= np.loadtxt(r'../icecores/density-raw/dens_%s.txt'%(site))#, header=0, dtype={'a': np.float32, 'b': np.float32}, sep='\s+')
    # df = pd.read_csv(r'Nicholas fit/icecores/density-fits/dens_%s_fit.txt'%(site), header=0, dtype={'a': np.float32, 'b': np.float32}, sep='\s+')
    # obs = df.to_numpy()
    z_obs = H - (obs[:,0] - 0*obs[0,0])
    I_obs = np.argsort(z_obs) # must be sorted for interpolation below
    z_obs = z_obs[I_obs]
    rho_obs = obs[I_obs,1] # to kg/m^3

    I_rm = np.argwhere(z_obs < 0)
    rho_obs = rho_obs[z_obs >= 0]
    z_obs   = z_obs[z_obs >= 0]
    
    return rho_obs, z_obs

def get_rho_data(H,site):

    obs= np.loadtxt(r'./icecores/density-fits/dens_%s_fit.txt'%(site))#, header=0, dtype={'a': np.float32, 'b': np.float32}, sep='\s+')
    # df = pd.read_csv(r'Nicholas fit/icecores/density-fits/dens_%s_fit.txt'%(site), header=0, dtype={'a': np.float32, 'b': np.float32}, sep='\s+')
    # obs = df.to_numpy()
    z_obs = H - (obs[:,0] - 0*obs[0,0])
    I_obs = np.argsort(z_obs) # must be sorted for interpolation below
    z_obs = z_obs[I_obs]
    rho_obs = 1e3*obs[I_obs,1] # to kg/m^3

    I_rm = np.argwhere(z_obs < 0)
    rho_obs = rho_obs[z_obs >= 0]
    z_obs   = z_obs[z_obs >= 0]
    
    return rho_obs, z_obs

def solve_forward_withT(H,Hres,acc_rate,K,n,Tsite,Flux,rho_obsfit,rho_ice=910,phi_snow=0.4,ab_phi_lim=0.81):

    #COUPLED PROBLEM
    
    #------------------------------------------INITIAL DENSITY PROFILE
    
    rho_surf = np.amin(rho_obsfit) #BAKOITZARI BEREAAAAAAA
    rho_snow=rho_surf
    
    T_sfc = Tsite #-28.8 + 273 # Surface temperature in K
    T_bed = Tsite+2 # Bed temperature in K
    
    rho_nicholas = Expression("rho_surf-(rho_surf-rho_ice)*pow((H-x[0])/H,0.35)", H=H, rho_ice=rho_ice, rho_surf=rho_surf, degree=2) 
    Tprev_expr=Expression("T_surf-(T_surf-T_bot)*pow((H-x[0])/H,0.1)", H=H, T_bot=T_bed, T_surf=T_sfc, degree=2)
    
    rho_prev=rho_nicholas
    T_prev = Tprev_expr
    
    #-----------------------------------------INITIALIZE ARRAYS AND VARIABLES
    
    tstep=0
    
    #-----------------------------------------DEFINE PARAMETERS FOR THE STEADY STATE 
    
    dHdt=100      #So that it enters the while
    dHdt_tol=0.01 #Change to consider that we have reached the steady state
    
    
    while dHdt>=dHdt_tol:
    #for tstep in tqdm.tqdm(range(0,int(Nt))):
    
        #-------------------------------------------MESH-----------------------------------------------------#
        
        mesh=IntervalMesh(Hres,0,H)
        # mesh = createMesh(sfc.xvec, zbot, sfc.zvec, Lres, Hres)
        
        #--------------------------------------BOUNDARY SUBDOMAINS-------------------------------------------#
        
        def bottom_boundary(x, on_boundary):
            return on_boundary and near(x[0], 0) # Note "x = x[0]" (first spatial dimension)
        
        def surface_boundary(x, on_boundary):
            return on_boundary and near(x[0], H)
        
        #--------------------------------------FUNCTION SPACE------------------------------------------------#
        
        #Define function space for density
        deg=2  #1 #Polinomial degree 
        U=FunctionSpace(mesh, 'Lagrange', deg); # Polynomial function space of order "deg" (=1 is linear, ...)
    
        rho=Function(U) # the unknown function
        wr=TestFunction(U)  # the weight function
    
        #Define function space for velocity
        deg=2
        V=FunctionSpace(mesh, "CG", deg)
    
        v=Function(V) # the unknown function
        wv=TestFunction(V)  # the weight function
        
        #Define function space for temperature
        deg=2
        Q=FunctionSpace(mesh,'Lagrange',deg)
        
        T=Function(Q)
        wt=TestFunction(Q)
        
        #--------------------------------------BOUNDARY CONDITIONS--------------------------------------------#
        
        #-----------------------------------TOP
        bc_rho_s=DirichletBC(U,rho_surf,surface_boundary) #Density at the surface
        #bc_v_s=DirichletBC(V,-acc_rate,surface_boundary) #Velocity at the surface 
        bc_T_s=DirichletBC(Q, Constant(T_sfc), surface_boundary)
        #-----------------------------------BOTTOM
        #bc_rho_b=DirichletBC(U,rho_ice_softened,bottom_boundary) #Density at the bottom
        bc_v_b=DirichletBC(V,-acc_rate*(rho_surf/rho_ice),bottom_boundary) #Velocity at the bottom
        #bc_T_b=DirichletBC(Q, Constant(T_bed), boundary_subdomains, 2)
    
        bcs_rho=[bc_rho_s] 
        bcs_v=[bc_v_b]
        bcs_T=[bc_T_s]#,bc_T_b]
        
        #--------------------------------------INITIAL CONDITION--------------------------------------------#
        
        if tstep==0:
            
            r_init = Expression("rho0  - (rho0-rhoi)*pow((H-x[0])/H,1/3.0)", H=H, rhoi=rho_ice, rho0=rho_surf, degree=2)
            u_init = Expression("uztop - (uztop-uzbot)*pow((H-x[0])/H,0.35)", H=H, uztop=-acc_rate, uzbot=-acc_rate*(rho_surf/rho_ice),  degree=2)
            t_init = Expression("T_surf-(T_surf-T_bot)*pow((H-x[0])/H,0.1)", H=H, T_bot=T_bed, T_surf=T_sfc, degree=2)
            rho_init, v_init, T_init = interpolate(r_init, U), interpolate(u_init, V), interpolate(t_init,Q)
            rho.assign(rho_init)
            v.assign(v_init)
            T.assign(T_init)
            
        else:
            
            rho_prev.set_allow_extrapolation(True)       
            rho_init = interpolate(rho_prev,U)
            v_sol.set_allow_extrapolation(True)
            v_init = interpolate(v_sol,V)
            T_prev.set_allow_extrapolation(True)
            T_init = interpolate(T_prev,Q)
            rho.assign(rho_init)
            v.assign(v_init)
            T.assign(T_init)
            
        #--------------------------------------INTERPOLATE RHO------------------------------------------------#
        
        if tstep >0: 
            
            #rho_prev=project(rho_prev,U)  hau ezin dezakegu egin orain U beste espazio berri (motzago) bat delako
                                            #horretara interpolatzen saiatzen ari gara
            
            rho_old=rho_prev.copy()
            rho_old.set_allow_extrapolation(True)
    
            rho_new=Function(U)
            LagrangeInterpolator.interpolate(rho_new,rho_old)
    
            rho_prev.assign(rho_new)
            rho_prev.set_allow_extrapolation(True)  
            
            T_old=T_prev.copy()
            T_old.set_allow_extrapolation(True)
    
            T_new=Function(Q)
            LagrangeInterpolator.interpolate(T_new,T_old)
    
            T_prev.assign(T_new)
            T_prev.set_allow_extrapolation(True) 
            
            #................................KEEP IT BELOW RHO_ICE...........................#
            
            rhovec = rho_prev.vector()[:]
            rhovec[rhovec > rho_ice_softened] = rho_ice_softened
            rho_prev.vector()[:] = rhovec
        
        rho_prev=project(rho_prev,U)
        T_prev = project(T_prev, Q) # Make sure Tprev from old mesh is projected to new mesh.
    
        #-----------------------------------------------------------------------------------------------------#
        #--------------------------------------SOLVE FEM PROBLEM----------------------------------------------#
        #-----------------------------------------------------------------------------------------------------# 
        
        #--------------------------------GET a, b, VARIABLES AND SIGMA-----------------------------------------
        a_,b_=get_ab(rho_prev,rho_ice,phi_snow,ab_phi_lim,n,K)
        Aglen = Constant(A0)*exp(-Q1/(R*T_prev)) #T_prev
        
        # viscosity=get_viscosity(v,a_,b_,Aglen,n)
        sigma=get_sigma(v,a_,b_,Aglen,n)
        
        #-----------------------------------SOLVE MOMENTUM EQUATION--------------------------------------------
        a_v =sigma*wv.dx(0)*dx           #a_v = (Lambda_constant(a_,b_)*D(v)*wv.dx(0)+2*etha_constant()*D(v)*wv.dx(0))*dx  
        L_v = rho_prev*g*wv*dx 
        
        F_v = a_v - L_v
        
        tol, relax, maxiter = 1e-2, 0.35, 50
        solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
        solve(F_v==0, v, bcs_v, solver_parameters=solparams)
        
        v_sol=project(v,V)
        
        #---------------------------------SOLVE MASS BALANCE EQUATION------------------------------------------
        a_rho = Constant(1/dt)*rho*wr*dx + rho*v_sol.dx(0)*wr*dx + v_sol*rho.dx(0)*wr*dx
        L_rho = wr* Constant(1/dt)*rho_prev *dx 
        
        F_rho = a_rho - L_rho
    
        tol, relax, maxiter = 1e-2, 0.35, 50
        solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
        solve(F_rho==0, rho, bcs_rho, solver_parameters=solparams)
        
        rho_prev.assign(rho)  #<-------------UPDATE RHO PROFILE
        
         #---------------------------------SOLVE HEAT EQUATION------------------------------------------
        a_,b_=get_ab(rho_prev,rho_ice,phi_snow,ab_phi_lim,n,K)
        sigma=get_sigma(v_sol,a_,b_,Aglen,n)
        
        Ssol = Function(V)    
        ssol = sigma*v_sol.dx(0)
        
        c=c0+c1*(T-T0)
        k=(kf0-kf1*rho_prev+kf2*rho_prev**2)/(kf0-kf1*rho_ice+kf2*rho_ice**2) * k0*exp(-gammaT*T)
        
        a_T = T*wt*dx + dt*v_sol*T.dx(0)*wt*dx + dt*k/(rho_prev*c)*T.dx(0)*wt.dx(0)*dx - dt/(rho_prev*c)*ssol*wt*dx - dt/(rho_prev*c)*k.dx(0)*T.dx(0)*wt*dx
        L_T = T_prev*wt*dx
        
        L_T += wt * Constant(Flux) * ds
        
        F_T = a_T - L_T
        
        tol, relax, maxiter = 1e-2, 0.35, 50
        solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
        solve(F_T==0, T, bcs_T, solver_parameters=solparams)
        T_prev.assign(T)
        
        #-----------------------------------------------------------------------------------------------------#
        #----------------------------------------CALCULATE NEW H----------------------------------------------#
        #-----------------------------------------------------------------------------------------------------#
    
        zs_v,v_np=convert_to_np(V,v_sol)
        deltaH=(v_np[-1]+acc_rate)*dt  
        H+=deltaH #deltaH>0 means that accumulation is bigger than the downwards Stokes'flow, therefore, the ice/firn column gets thicker
        
        dHdt = abs(deltaH)*yr2s/dt
        print(dHdt)
        tstep+=1
    
        
    #Get nodal coordinates
    z_rho = U.tabulate_dof_coordinates()[:,0] #velocity node coordinates
    z_v  = V.tabulate_dof_coordinates()[:,0] #density node coordinates
    I_rho, I_v = np.argsort(z_rho), np.argsort(z_v)
    z_rho, z_v = z_rho[I_rho], z_v[I_v] # sorted list of coordinates
    
    v_sol, rho_sol = project(v_sol, V), project(rho_prev, U) # ... but this does
    v, rho = v_sol.vector()[I_v], rho_prev.vector()[I_rho] # solution on nodes, correctly sorted for plotting 
    
    return rho, z_rho

#SIMULATION PARAMETERS

SecPerYear = 31556926;
s2yr = 1/SecPerYear
ms2myr = 31556926
ms2kyr = ms2myr/1000

day2s=24*60*60
yr2s=365.25*day2s

n=3

H0=200 #Depth, Length
Hres=20 #Nodal resolution 

rho_ice= 910
rho_ice_softened= rho_ice - 0.2

phi_snow=0.4
ab_phi_lim=0.81

# Thermal parameters of pure ice
c0=2127.5
c1=7.253
T0=273.16

k0=9.828
gammaT=0.0057
kf0=0.138
kf1=1.010e-3
kf2=3.233e-6

A0,Q1,R = 3.985e-13, 60e3, 8.314 # Canonical values, same as Zwinger et al. (2007)

Flux=0.079 #W/m2

g=Constant(-9.82)

sites=['dye3','grip','neem','ngrip','site_2','siteA_crete']
# sites=['site_2','siteA_crete']#['dye3','grip','neem','ngrip','site2','siteA_crete']
Ks=[500,1000]


Hs={'dye3':H0, 'grip':H0, 'neem':H0-15, 'ngrip':H0-10, 'site_2':H0, 'siteA_crete':H0}
acc_sites={'dye3':0.50,    'grip':0.21,   'neem':0.20,   'ngrip':0.175,   'site_2':0.36,     'siteA_crete':0.282}
T_sites={'dye3':-21,    'grip':-31.7,   'neem':-28.8,   'ngrip':-31.5,   'site_2':-24,     'siteA_crete':-29.5}

# site='neem'
# H=Hs[site]
# rho_obsfit, z_obsfit=get_rho_data(H,site)
# acc_rate=acc_from_iceeqyr_to_snoweqs(acc_sites[site],rho_obsfit)#1.54/yr2s #0.2/yr2s
# Tsite=T_sites[site]+273

fac=0.2
dt=0.25*yr2s/fac
Nt=15*100*fac


for site in sites:
    
    H=Hs[site]
    rho_obsfit, z_obsfit=get_rho_data(H,site)
    acc_rate=acc_from_iceeqyr_to_snoweqs(acc_sites[site],rho_obsfit)#1.54/yr2s #0.2/yr2s
    Tsite=T_sites[site]+273
    
    for K in Ks:
            
        print(site)
        print(K)
        
        rho,z=solve_forward_withT(H,Hres,acc_rate,K,n,Tsite,Flux,rho_obsfit,rho_ice=910,phi_snow=0.4,ab_phi_lim=0.81)

        np.save('./Tresults/rho_1D_'+str(site)+'_withT_K='+str(K)+'.npy',rho)
        np.save('./Tresults/z_1D_'+str(site)+'_withT_K='+str(K)+'.npy',z)