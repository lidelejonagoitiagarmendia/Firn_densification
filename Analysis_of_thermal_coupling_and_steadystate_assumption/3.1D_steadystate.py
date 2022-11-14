# import pandas as pd
import copy, code # code.interact(local=locals())
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from dolfin import *
from dolfin_adjoint import * #pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0
import moola

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

def acc_from_iceeqyr_to_snoweqs(acc,rho_obs):
    
    accum_watereqyr = acc # water equiv. accum, m/yr
    accum_iceeqyr = 1000/rho_ice * accum_watereqyr
    accum_iceeqs = accum_iceeqyr/SecPerYear # ice accum, m/s 
    rho_snow = np.amin(rho_obs) 
    acc_rate = rho_ice/rho_snow * accum_iceeqs
    
    print('adot [metre snow/yr] = %e'%(acc_rate*SecPerYear))
    
    return acc_rate


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
    
    eps_dot=v.dx(0) #sym(grad(v))                      #sym(grad(v))
    J1=eps_dot #tr(eps_dot)                            #tr(eps(v))
    J2=eps_dot*eps_dot #inner(eps_dot,eps_dot)
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


#Define boundaries of the domain
class bottom_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],0)

class surface_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],H)  #horrela definitu dezakegu gero eremua definitiuko ditugulako zatika


def solve_forward_problem(H,Hres,acc_rate,K,n,Aglen,rho_obsfit,rho_ice=910,phi_snow=0.4,ab_phi_lim=0.81):
    
    print(H)
    #Define acc and rho_surf for each site!!!!!!!!!!!!!!!!!
    rho_surf = np.amin(rho_obsfit) #BAKOITZARI BEREAAAAAAA
    rho_snow=rho_surf
    
    #-------------------------------------------MESH-----------------------------------------------------#

    mesh = IntervalMesh(Hres-1, 0, H) # Hres nodes on x=[0,H]

    #-----------------------------------MIXED FUNCTION SPACE---------------------------------------------#

    Uele = FiniteElement("CG", mesh.ufl_cell(), 1) #Density
    Vele = FiniteElement("CG", mesh.ufl_cell(), 2) #Velocity
    U, V = FunctionSpace(mesh, Uele), FunctionSpace(mesh, Vele) 
    MixedEle = MixedElement([Uele, Vele]) 
    W = FunctionSpace(mesh, MixedEle)

    (wr,wv) = TestFunctions(W) #weight functions (Galerkin method)
    w = Function(W) #container for solution (rho,v)
    (rho,v) = split(w)

    #Get nodal coordinates
    z_rho = U.tabulate_dof_coordinates()[:,0] #velocity node coordinates
    z_v  = V.tabulate_dof_coordinates()[:,0] #density node coordinates
    I_rho, I_v = np.argsort(z_rho), np.argsort(z_v)
    z_rho, z_v = z_rho[I_rho], z_v[I_v] # sorted list of coordinates

    #--------------------------------------BOUNDARY SUBDOMAINS--------------------------------------------#

    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)

    bottom=bottom_boundary()
    bottom.mark(boundary_subdomains, 1)
    surface=surface_boundary()
    surface.mark(boundary_subdomains, 2)

    #--------------------------------------BOUNDARY CONDITIONS--------------------------------------------#
    v_surf=-acc_rate
    v_bot=-acc_rate*(rho_surf/rho_ice)

    #-----------------------------------TOP
    bc_rho_s=DirichletBC(W.sub(0),rho_surf,boundary_subdomains,2) #Density at the surface
    bc_v_s=DirichletBC(W.sub(1),v_surf,boundary_subdomains,2) #Velocity at the surface

    #-----------------------------------BOTTOM
    bc_v_b=DirichletBC(W.sub(1),v_bot,boundary_subdomains,1) #Velocity at the bottom

    bcs=[bc_rho_s,bc_v_s,bc_v_b]      

    #----------------------------------------INITIAL STATE--------------------------------------------#

    assigner = FunctionAssigner(W, [U, V]) # used to assign values into components of mixed function space

    r_init = Expression("rho0  - (rho0-rhoi)*pow((H-x[0])/H,1/3.0)", H=H, rhoi=rho_ice, rho0=rho_snow, degree=2)
    u_init = Expression("uztop - (uztop-uzbot)*pow((H-x[0])/H,1/3.0)", H=H, uztop=-acc_rate, uzbot=v_bot,  degree=2)

    rho_init, v_init = interpolate(r_init, U), interpolate(u_init, V)
    w_init = Function(W) # container for solution (uz,rho)
    assigner.assign(w_init, [rho_init,v_init]) # assign initial guesses
    w.assign(w_init)

    #--------------------------------------SOLVE FEM PROBLEM----------------------------------------------#

    #---------------------------STOKES

    #Get Cs, a, b, and sigma
    Aglen = interpolate(Constant(Aglen), U)  
    a_,b_=get_ab(rho,rho_ice,phi_snow,ab_phi_lim,n,K)
    sigma=get_sigma(v,a_,b_,Aglen,n)

    a_sig = sigma * wv.dx(0) * dx # viscous stress divergence
    L_sig = rho*Constant(g)*wv*dx # Body force (gravity)

    #---------------------------DENSITY

    a_rho = (rho*v).dx(0) * wr * dx # d(rho)/dt + div(rho*u) = 0, for d/dt = 0 
    L_rho = 0

    #---------------------------NON-LINEAR WEAK FORM

    F  = a_sig - L_sig # stokes problem 
    F += a_rho - L_rho # density problem

    #---------------------------SOLVE MIXED PROBLEM

    tol, relax, maxiter = 1e-2, 0.35, 50
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F==0, w, bcs, solver_parameters=solparams)
    rho_sol, v_sol = w.split()
    
    v_sol, rho_sol = project(v_sol, V), project(rho_sol, U) # ... but this does
    v, rho = v_sol.vector()[I_v], rho_sol.vector()[I_rho] # solution on nodes, correctly sorted for plotting 
    
    return rho, z_rho

#Define parameters

#Unit conversion
SecPerYear = 31556926;
s2yr = 1/SecPerYear
ms2myr = 31556926
ms2kyr = ms2myr/1000

#Physical constants
g = -9.82  # gravitational accel
rho_ice = 910 # density of glacier ice
ab_phi_lim = 0.81 # critical relative density (\hat{\rho}) at which coefficient function changes from a0 to a1, and b0 to b1
phi_snow = 0.4#rho_snow/rho_ice # relative density of snow, used for calibrating a(rhoh_snow) = b(rhoh_snow) = k 
n = 3 # flow law exponent

A0,Q1,R = 3.985e-13, 60e3, 8.314

#Site related parameters
H0=200
Hres=180

sites=['dye3','grip','neem','ngrip','site_2','siteA_crete']
#sites=['neem']#['dye3','grip','neem','ngrip','site2','siteA_crete']
Ks=[100,500,1000]
#Ks=[1000]
Hs={'dye3':H0, 'grip':H0, 'neem':H0-15, 'ngrip':H0-10, 'site_2':H0, 'siteA_crete':H0}
acc_sites={'dye3':0.50,    'grip':0.21,   'neem':0.20,   'ngrip':0.175,   'site_2':0.36,     'siteA_crete':0.282}
T_sites={'dye3':-21,    'grip':-31.7,   'neem':-28.8,   'ngrip':-31.5,   'site_2':-24,     'siteA_crete':-29.5}

for site in sites:
    
    H=Hs[site]
    rho_obsfit, z_obsfit=get_rho_data(H,site)
    acc_rate=acc_from_iceeqyr_to_snoweqs(acc_sites[site],rho_obsfit)#1.54/yr2s #0.2/yr2s
    Tsite=T_sites[site]+273
    Aglen = Constant(A0)*exp(-Q1/(R*Tsite))
    
    for K in Ks:
              
        rho,z=solve_forward_problem(H,Hres,acc_rate,K,n,Aglen,rho_obsfit,rho_ice=910,phi_snow=0.4,ab_phi_lim=0.81)

        np.save('./SSresults/rho_1D_'+str(site)+'_withoutT_K='+str(K)+'.npy',rho)
        np.save('./SSresults/z_1D_'+str(site)+'_withoutT_K='+str(K)+'.npy',z)




