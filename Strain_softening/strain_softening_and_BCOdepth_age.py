import math
from tqdm import tqdm 
from pymatreader import read_mat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate as interpolatescipy
import imageio                    #for the gif
import os                         #for the gif
from fenics import *
from dolfin import *
import openpyxl
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

#control verbosity of solve function 
#https://fenicsproject.org/qa/810/how-to-disable-message-solving-linear-variational-problem/
set_log_level(21)


#FUNCTIONS:


def acc_from_iceeqyr_to_snoweqs(acc,rho_obs):
    
    # accum_watereqyr = acc # water equiv. accum, m/yr
    # accum_iceeqyr = 1000/rho_ice * accum_watereqyr
    accum_iceeqs = acc/SecPerYear #accum_iceeqyr/SecPerYear # ice accum, m/s 
    rho_snow = rho_obs
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

def get_sigma(v,a,b,Aglen,nglen,epsl,epss):
    
    eps_dot=sym(grad(v))                      #sym(grad(v))
    J1=tr(eps_dot) #+ epsl                   #tr(eps(v))
    J2=inner(eps_dot,eps_dot)+2*epss**2#+epsl**2              #inner(eps_dot,eps_dot)
    eps_E2=1/a*(J2-J1**2/3) + (3/2)*(1/b)*J1**2
    viscosity = (1/2)**((1-nglen)/(2*nglen)) * Aglen**(-1/nglen) * (eps_E2)**((1-nglen)/(2*nglen))    
    sigma = viscosity * (1/a*(eps_dot-(J1/3)*Identity(2))+(3/2)*(1/b)*J1*Identity(2))
    
    return sigma

def inizialize():
      
    x = np.linspace(0, L, Lres)
    tops = H*np.ones(len(x))
    bots = np.zeros(len(x))
    
    deltax0=3000+2500
    x0=2000+deltax0 #7500ean jartzeko zentrua
    
    #Create bump in the bedrock
    ones=np.heaviside(x-1200-deltax0, 0)*np.heaviside(-(x-2800-deltax0), 0)
    sqrtzeatzeko=ones*(-x**2 + 2*(x0)*x +  (1000**2 - x0**2))
    kentzeko=ones*(-600)
    bedbump=np.sqrt(sqrtzeatzeko)+ kentzeko
    bedbump=0*0.5*bedbump #Apur bat txikitzeko
    
    return x, bots, tops, bedbump

#Surface evolution
class Surface:
    
    def __init__(self, f_z0):
        
        # init profile
        self.RES_L = Lres #-----------------------------------KONTUZ!!! Eskuz aldatu behar da
        self.xvec = np.linspace(0,L, self.RES_L)
        self.zvec = f_z0(self.xvec) # init profile
        
        # mesh 
        self.smesh = IntervalMesh(self.RES_L-1, 0, L)
        self.xs = self.smesh.coordinates()[:, 0] # x coords of mesh *vertices* 
        
        # func spaces 
        self.S  = FunctionSpace(self.smesh, "CG", 1)#, constrained_domain=pbc_1D)
        scoords = self.S.tabulate_dof_coordinates()
        self.xs_dof = scoords[:,0] # x coords of func space NODES
        self.s_numdofs = len(self.xs_dof)
        self.IS = np.argsort(self.xs_dof)    
        
        #gurea:
        self.deltaH = self.zvec #will be changed after evolving surface
        self.usz = Function(self.S)
        
        self.accumulation=True
        if self.accumulation:
            
            self.accf=Function(self.S)
            
            for ii in range(self.s_numdofs):
            
                xii = self.xs_dof[ii] # x coord of DOF (node)
                self.accf.vector()[ii]=acc_rate
                
            acc_proj=project(self.accf,self.S)
            
            fig, ax = plt.subplots(figsize=(15,5))
            plot(acc_proj*yr2s)
            ax.set_aspect(aspect=1000)
        
        
    def _extendPeriodicVector(self, vec):
        # make the last point the first value too, which is exluded in a Vector()
        return np.append([vec[-1]],vec) 
        
    def evolve(self, u, dt):
        
        # "u" is the full 2D velocity field solution
        
        u.set_allow_extrapolation(True) 
        
        zvec_intp = interpolatescipy.interp1d(self.xvec, self.zvec, kind='linear') 
        
        # x and z vel. values at surface
        usx_arr, usz_arr = np.zeros(self.s_numdofs), np.zeros(self.s_numdofs)
        
        
        for ii in np.arange(self.s_numdofs): # loop over surface mesh DOFs (nodes)
            xii = self.xs_dof[ii] # x coord of DOF (node)
            zii = zvec_intp(xii) # the surface height of DOF (node)
            usx_arr[ii], usz_arr[ii] = u(xii, zii) # surface x and z velocity components
        
        # Save the sfc vel. values in "S" function spaces for solving the sfc evo problem below
        usx, usz = Function(self.S), Function(self.S) 
        usx.vector()[:] = usx_arr[:]
        usz.vector()[:] = usz_arr[:]
        
        # Function spaces
        v       = TestFunction(self.S) # weight funcs
        strial  = TrialFunction(self.S) # new (unknown) solution
        s0      = Function(self.S) # prev. solution
        s0.vector()[self.IS] = np.copy(self.zvec)#[1:]) # -1 because of periodic domain #HAU KONTUAN hartu ere kentzen baditugu momenturen batean
        
        # Solve surface equation
        dt_ = Constant(dt)
        a  = strial*v*dx + dt_*usx*strial.dx(0)*v*dx # *.dx(0) = d/d_{x_0} = "d/d_x"  #hemen oraindik ez dugu jarri akumulaziorik
        L  = s0*v*dx + dt_*(usz+project(self.accf,self.S))*v*dx
        s = Function(self.S)
        solve(a == L, s, [])
     
        # Update surface profile
        
        self.deltaH = s.vector()[self.IS] - self.zvec
        #self.deltaH = self._extendPeriodicVector(s.vector()[self.IS]) - self.zvec #This is for PBC
        
        #print('OLD------',self.zvec)
        #print('DELTA---',self.deltaH)
        #self.zvec = self._extendPeriodicVector(s.vector()[self.IS]) #Hau mugalde baldintza periodikoentzako, bestela begiratu Muller
        self.zvec = s.vector()[self.IS]
        #print('NEW------',self.zvec)
        self.usz=usz

#Create mesh
def createMesh(xvec, bottomsurf, topsurf, Nx, Nz):
    
    hl = Nx-1 # Number of horizontal layers
    vl = Nz-1 # Number of vertical layers
    
    # generate mesh on unitsquare
    mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), hl, vl)
    
    # Extract x and y coordinates from mesh
    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]
    
    x0 = min(xvec)
    x1 = max(xvec)
    
    # Map coordinates on unit square to the computational domain
    xnew = x0 + x*(x1-x0)
    zs = np.interp(xnew, xvec, topsurf)
    zb = np.interp(xnew, xvec, bottomsurf)
    ynew = zb + y*(zs-zb)
    
    xynewcoor = np.array([xnew, ynew]).transpose()
    
    mesh.coordinates()[:] = xynewcoor
    
    return mesh

# Define boundaries of the domain
class bottom_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class top_boundary(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and x[1]>=50   #horrela definitu dezakegu gero eremua definitiuko ditugulako zatika

class right_boundary(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and near(x[0],L)

class left_boundary(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and near(x[0],0)  


def convert_to_np(U,ufl_array):
    
    zcoord=U.tabulate_dof_coordinates()[:,0] #koordinatuak gorde, ordenatu gabe!
    np_array=ufl_array.vector()              #interesatzen zaigun aldagaiaren balioak gorde, ordenatu gabe!
    I=np.argsort(zcoord)                     #z-ren arabera ordenatzeko indexen ordena gorde 
    sorted_coors=zcoord[I]                   #koordinatuak gorde, ordenatuta!
    sorted_np=np_array[I]                    #interesatzen zaigun aldagaiaren balioak gorde, ordenatuta!

    return sorted_coors,sorted_np


#SIMULATION PARAMETERS

n=3

deltaL=5000 #buffer on the sides
H,L=200,37440 +2*deltaL #Depth, Length
Hres,Lres=50,150 #Nodal resolution 

rho_surf=275
rho_ice= 910
rho_ice_softened= rho_ice - 0.2

phi_snow=0.4
ab_phi_lim=0.81

K=500
A0,Q,R,T = 3.985e-13, 60e3, 8.314,-28+273.15
Aglen=A0*exp(-Q/(R*T))

day2s=24*60*60
yr2s=365.25*day2s
SecPerYear=yr2s

acc_meas=0.18 #Measured acc in m of ice equivalent per year
acc_rate=acc_from_iceeqyr_to_snoweqs(acc_meas,rho_surf)
g=Constant((0.0,-9.82))

fac=0.2
dt=0.25*yr2s/fac
Nt=15*100*fac

#INITIAL DENSITY PROFILES:

#Possible initial density profiles

rho_allice=Expression('rho_bottom',rho_bottom=rho_ice_softened,degree=1)
rho_kte=Expression('rho',rho=500,degree=1)
rho_linear=Expression('rho_bot+((rho_top-rho_bot)/h)*x[1]',rho_top=rho_surf,rho_bot=rho_ice_softened,h=H,degree=1)
rho_zwinger=Expression('rho_ice*(1-surface_porosity*exp(-compaction*(h-x[1])))',rho_ice=rho_ice,surface_porosity=0.55,compaction=0.038,h=H,degree=1)                                                                                                                        #degree?
rho_init = Expression("rho_surf-(rho_surf-rho_ice)*pow((H-x[1])/H,0.35)", H=H, rho_ice=rho_ice, rho_surf=rho_surf, degree=2)

rho_prev=rho_init   #chose from the above!


# ### ADDITIONAL STRAIN RATE VALUES IN THE MIDDLE ZONE

epsyy=np.load("NEGISepsyy.npy")
epsxy=np.load("NEGISepsxy.npy")

xs=np.linspace(deltaL,L-deltaL,len(epsyy))
plt.figure()
plt.plot(xs,epsyy/yr2s)
plt.plot(xs,epsxy/yr2s)
plt.show()


print(len(epsyy))
eps_dens= len(epsyy)/(L-2*deltaL)
print(eps_dens)

Neps_deltaL=int(deltaL*eps_dens)
print(Neps_deltaL)

extension=np.zeros(Neps_deltaL)

epsyy= np.concatenate((extension,epsyy,extension))
epsxy= np.concatenate((extension,epsxy,extension))

print(epsyy)
print(len(epsyy))
xs=np.linspace(0,L,len(epsyy))
plt.figure()
plt.plot(xs,epsyy/yr2s)
plt.plot(xs,epsxy/yr2s)
plt.show()
    
#Make the strain rate smoother

window_length=5#int(dx_model/dx_rawdata)
polyorder=3

epsyy_smooth=savgol_filter(epsyy/yr2s,window_length,polyorder=polyorder)
epsxy_smooth=savgol_filter(epsxy/yr2s,window_length,polyorder=polyorder)


fig, ax = plt.subplots(figsize=(15,5))
plt.plot(epsyy_smooth*yr2s,label='epsyy')
plt.plot(epsxy_smooth*yr2s,label='epsxy')
plt.ylabel('smoothed strain rates $\ \ \ \  (yr^{\ -1})$')
plt.xlabel('X (m)')
plt.legend()
plt.show()


#Create mesh
mesh=RectangleMesh(Point(0,0),Point(L,H),Lres,Hres)

#Define function space for density
deg=2 #Polinomial degree 
U=FunctionSpace(mesh, 'Lagrange', deg); # Polynomial function space of order "deg" (=1 is linear, ...)


#------------to properly order the nodes before setting the values--------------------------
f_xs = mesh.coordinates()[:, 0] # x coords of mesh *vertices* 
f_ys = mesh.coordinates()[0, :] # y coords of mesh *vertices* 


# func spaces 
scoords = U.tabulate_dof_coordinates()
xs_dof = scoords[:,0] # x coords of func space NODES
s_numdofs = len(xs_dof)
ISx = np.argsort(xs_dof)  

fepsyy_smooth = interp1d(np.linspace(0,L,len(epsyy_smooth)), epsyy_smooth, kind='cubic',fill_value='extrapolate')
fepsxy_smooth = interp1d(np.linspace(0,L,len(epsxy_smooth)), epsxy_smooth, kind='cubic',fill_value='extrapolate')

strainl=Function(U)
strains=Function(U)


for ii in range(s_numdofs):
            
    xii = xs_dof[ii] # x coord of DOF (node)
    strainl.vector()[ii]=fepsyy_smooth(xii)
    strains.vector()[ii]=fepsxy_smooth(xii)
    
strainl_p=project(strainl,U)
strains_p=project(strains,U)


fig, ax = plt.subplots(figsize=(15,5))

rhoplot=plot(strainl_p)

ax.set_ylabel('Z(m)')
ax.set_xlabel('X (m)')
clb=plt.colorbar(rhoplot,orientation="vertical",label='Strain rate (1/s)')

ax.set_aspect(aspect=50)


fig, ax = plt.subplots(figsize=(15,5))

rhoplot=plot(strains_p)

ax.set_ylabel('Z(m)')
ax.set_xlabel('X (m)')
clb=plt.colorbar(rhoplot,orientation="vertical",label='Strain rate (1/s)')

ax.set_aspect(aspect=50)

#COUPLED PROBLEM: MAIN LOOP

#Initialize mesh
xvec, bots, ztop, zbot = inizialize() 

#Define initial surface  (flat surface) 
z0 = H/2
f_z0 = lambda x: H +x*0
sfc = Surface(f_z0)


mesh=createMesh(xvec, zbot, ztop, Lres, Hres)

#------------------------------------------INITIAL DENSITY PROFILE

rho_prev=rho_init

#-----------------------------------------INITIALIZE ARRAYS AND VARIABLES

tstep=0

#-----------------------------------------DEFINE PARAMETERS FOR THE STEADY STATE 

dHdt=100      #So that it enters the while
dHdt_tol=0.015# 0.01 #Change to consider that we have reached the steady state


while dHdt>=dHdt_tol:

    q_degree = 3
    dx = dx(metadata={'quadrature_degree': q_degree})

    #-------------------------------------------MESH-----------------------------------------------------#
    
    mesh = createMesh(sfc.xvec, zbot, sfc.zvec, Lres, Hres)
    
    #--------------------------------------BOUNDARY SUBDOMAINS-------------------------------------------#
    
    #Give a number to each different boundary
    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
        
    bottom=bottom_boundary()
    bottom.mark(boundary_subdomains, 2)
    top=top_boundary()
    top.mark(boundary_subdomains, 1)
    left=left_boundary()
    left.mark(boundary_subdomains, 3)  #berresleitzen ditu aldeeakoak, nahiz ta hasieran horien zati bat topean egon
    right=right_boundary()
    right.mark(boundary_subdomains, 4)
    
    #--------------------------------------FUNCTION SPACE------------------------------------------------#
    
    #Define function space for density
    deg=1  #1 #Polinomial degree 
    U=FunctionSpace(mesh, 'Lagrange', deg); # Polynomial function space of order "deg" (=1 is linear, ...)

    rho=Function(U) # the unknown function
    wr=TestFunction(U)  # the weight function

    #Define function space for velocity
    deg=2
    V=VectorFunctionSpace(mesh, "CG", 2)

    v=Function(V) # the unknown function
    wv=TestFunction(V)  # the weight function
    
    #--------------------------------------BOUNDARY CONDITIONS--------------------------------------------#
    
    #-----------------------------------TOP
    bc_rho_s=DirichletBC(U,rho_surf,boundary_subdomains,1) #Density at the surface
    #bc_v_s=DirichletBC(V,-acc_rate,surface_boundary) #Velocity at the surface 
    #-----------------------------------BOTTOM
    #bc_rho_b=DirichletBC(U,rho_ice_softened,bottom_boundary) #Density at the bottom
    bc_v_b=DirichletBC(V,(0.0,-acc_rate*(rho_surf/rho_ice)),boundary_subdomains,2) #Velocity at the bottom
    #-----------------------------------LEFT
    #bc_rho_b=DirichletBC(U,rho_ice_softened,bottom_boundary) #Density at the bottom
    bc_v_l=DirichletBC(V.sub(0),0.0,boundary_subdomains,3) #Velocity at the left boundary
    #-----------------------------------RIGHT
    #bc_rho_b=DirichletBC(U,rho_ice_softened,bottom_boundary) #Density at the bottom
    bc_v_r=DirichletBC(V.sub(0),0.0,boundary_subdomains,4) #Velocity at the right boundary

    bcs_rho=[bc_rho_s] 
    bcs_v=[bc_v_b,bc_v_l,bc_v_r]
    
    #--------------------------------------INITIAL CONDITION--------------------------------------------#
    
    if tstep==0:
        
        r_init = Expression("rho0  - (rho0-rhoi)*pow((H-x[1])/H,1/3.0)", H=H, rhoi=rho_ice, rho0=rho_surf, degree=2)
        u_init = Expression(('vx',"uztop - (uztop-uzbot)*pow((H-x[1])/H,0.35)"),vx=Constant(0.0), H=H, uztop=-acc_rate, uzbot=-acc_rate*(rho_surf/rho_ice),  degree=2)
        rho_init, v_init = interpolate(r_init, U), interpolate(u_init, V)
        rho.assign(rho_init)
        v.assign(v_init)
        
    else:
        
        rho_prev.set_allow_extrapolation(True)       
        rho_init = interpolate(rho_prev,U)
        v_init = interpolate(v_sol,V)
        rho.assign(rho_init)
        v.assign(v_init)
        
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
        
        strainl.set_allow_extrapolation(True)       
        strains.set_allow_extrapolation(True)       
        
        #................................KEEP IT BELOW RHO_ICE...........................#
        
        rhovec = rho_prev.vector()[:]
        rhovec[rhovec > rho_ice_softened] = rho_ice_softened
        rho_prev.vector()[:] = rhovec
    
    rho_prev=project(rho_prev,U)
    
    
    #plotting progress------------------
    
    if tstep%10==0:
        
        fig, ax=plt.subplots(figsize=(15,5))
        rhoplot=plot(rho_prev,cmap='PuBu')
        #plt.ylim([100,180])
        plot(rho_prev,mode='contour',levels=[350,830,915],cmap='Pastel2')
        clb=plt.colorbar(rhoplot,orientation="horizontal",label='Density (kg/m3)')  

        # labels = [item.get_text() for item in ax.get_yticklabels()]
        # labels = [90, 70, 50, 30, 10]

        # ax.set_yticklabels(labels)
        ax.set_aspect(aspect=100)
        ax.set_ylabel('Depht (m)')
        plt.title(tstep)
        plt.show()
    
    if tstep==0:
        
        plt.figure()
        rhoplot=plot(rho_prev)
        clb=plt.colorbar(rhoplot,orientation="vertical",label='Density (kg/m3)')    
        plt.show()
      
    #----------------------------------CHECK CONSERVATION OF MASS-----------------------------------------#
    
    mass.append(assemble(rho_prev*dx))
    
    #-----------------------------------------------------------------------------------------------------#
    #--------------------------------------SOLVE FEM PROBLEM----------------------------------------------#
    #-----------------------------------------------------------------------------------------------------# 
    
    #--------------------------------GET a, b, VARIABLES AND SIGMA-----------------------------------------
    a_,b_=get_ab(rho_prev,rho_ice,phi_snow,ab_phi_lim,n,K)
    
    rho_max=rho_ice*0.99
    new_strain_value=Constant(0.0)
    
    strainl_add=project(strainl,U)
    strains_add=project(strains,U)
    csub_array1 = strainl_add.vector().get_local()
    csub_array2 = strains_add.vector().get_local()
    rho_prev_arr = rho_prev.vector().get_local()
    
    np.place(csub_array1, rho_prev_arr>rho_max, new_strain_value)
    strainl_add.vector()[:] = csub_array1
    np.place(csub_array2, rho_prev_arr>rho_max, new_strain_value)
    strains_add.vector()[:] = csub_array2
    
    sigma=get_sigma(v,a_,b_,Aglen,n,strainl_add,strains_add)
    
    #-----------------------------------SOLVE MOMENTUM EQUATION--------------------------------------------
    a_v = inner(sigma,grad(wv))*dx           #a_v = (Lambda_constant(a_,b_)*D(v)*wv.dx(0)+2*etha_constant()*D(v)*wv.dx(0))*dx  
    L_v = rho_prev*inner(g,wv)*dx 
    
    F_v = a_v - L_v
    
    tol, relax, maxiter = 1e-2, 0.35, 50
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_v==0, v, bcs_v, solver_parameters=solparams)
    
    v_sol=project(v,V)
    
    #---------------------------------SOLVE MASS BALANCE EQUATION------------------------------------------
    a_rho = Constant(1/dt)*rho*wr*dx + rho*div(v_sol)*wr*dx + dot(v_sol,grad(rho))*wr*dx
    L_rho = wr* Constant(1/dt)*rho_prev *dx 
    
    F_rho = a_rho - L_rho

    tol, relax, maxiter = 1e-2, 0.35, 50
    solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
    solve(F_rho==0, rho, bcs_rho, solver_parameters=solparams)
    
    rho_prev.assign(rho)  #<-------------UPDATE RHO PROFILE
    
    if tstep==0:
        
        v_init=project(v_sol,V)
        zcoord_initial_np,v_initial_np=convert_to_np(V,v_init)

    #-----------------------------------------------------------------------------------------------------#
    #----------------------------------------SURFACE EVOLUTION----------------------------------------------#
    #-----------------------------------------------------------------------------------------------------#
   
    #Surface evolution
    sfc.evolve(v_sol, dt)
    dHdt = max(np.abs(sfc.deltaH))*yr2s/dt
    print(dHdt)
    
    tstep+=1



fig, ax=plt.subplots(figsize=(15,5))
rhoplot=plot(rho_prev,cmap='PuBu')
plt.ylim([93,190])
plt.xlim([deltaL,L-deltaL])
plot(rho_prev,mode='contour',levels=[280,830,915],cmap='Pastel2')
clb=plt.colorbar(rhoplot,orientation="vertical",label=r'Density (kg/m$^3$)')  

plt.xlabel('Offset along line (km)')


positions = (9520, 14520,19520,24520,29520,34520,39520)
labels = (15,20,25,30,35,40,45)
plt.xticks(positions, labels)

positions = (173, 163,153,143,133,123,113,103,93)
labels = (0,10,20,30,40,50,60,70,80)

plt.yticks(positions, labels)

ax.set_aspect(aspect=125)
ax.set_ylabel('Depht (m)')
plt.savefig('strain_softening_FINAL_'+str(K)+'K_'+str(dHdt_tol)+'convergLIM_'+str(Lres)+'Lres_'+str(window_length)+'smoothingwindow.png',format='png',dpi=200)


#CALCULATE BCO DEPTH AND THE AGE OF THE FIRN AT THAT POINT

x_v  = V.tabulate_dof_coordinates()[:,0]
z_v  = V.tabulate_dof_coordinates()[:,1]

non_repeat_xs = list(set(x_v))

order_x=np.argsort(non_repeat_xs)

xs_nodes=np.zeros(len(non_repeat_xs))

for i in range(len(order_x)):
    xs_nodes[i]=non_repeat_xs[order_x[i]]                        

# len(z_v[np.where(x_v==xs_nodes[i])][::2])=99

zs_nodes=np.zeros((len(xs_nodes),99))

for i in range(len(xs_nodes)):
    
    trial=z_v[np.where(x_v==xs_nodes[i])][::2]
    order_z=np.argsort(trial)
    zs_nodes[i,:]=trial[order_z]

vs_nodes=np.zeros((len(xs_nodes),np.shape(zs_nodes)[1]))

for i in range(len(xs_nodes)):
    for j in range(np.shape(zs_nodes)[1]):
        vs_nodes[i,j]=v(int(xs_nodes[i]),int(zs_nodes[i,j]))[1]

      
#Get the density values at the nodes
rho_nodes=np.zeros((len(xs_nodes),np.shape(zs_nodes)[1]))

for i in range(len(xs_nodes)):
    for j in range(np.shape(zs_nodes)[1]):
        rho_nodes[i,j]=rho(int(xs_nodes[i]),int(zs_nodes[i,j]))

#Get the BCO depth
zsBCO=np.zeros(len(xs_nodes))
    
for i in range(len(xs_nodes)):
    f_zBCO=interp1d(rho_nodes[i,:],zs_nodes[i,:])
    zsBCO[i]=f_zBCO(830)


plt.figure()
plt.plot(xs_nodes,zsBCO)


Nyears=500
dt=1/2
ts=np.linspace(0,Nyears,int(Nyears/dt))*yr2s

position_evol=np.zeros((len(xs_nodes),len(ts)))

for j in tqdm(range(len(xs_nodes))):
    f_vz=interp1d(zs_nodes[j,:],vs_nodes[j,:],fill_value='extrapolate')
    for i in range(len(ts)):
    
        
        if i==0:
            z0=zs_nodes[j,-1]
        else:
            z0=z1
            
        
        vz=f_vz(z0)

        z1=z0+vz*dt*yr2s
        
        position_evol[j,i]=z1


fig, ax=plt.subplots(figsize=(20,15))
zplot=ax.imshow(position_evol.transpose())
ax.set_aspect(aspect=0.1)
ax.set_ylabel('Time (yr)')
plt.show()


t_BCO=np.zeros(len(xs_nodes))

for i in range(len(xs_nodes)):
    f_tBCO=interp1d(position_evol[i,:],ts)
    t_BCO[i]=f_tBCO(zsBCO[i])


matplotlib.rcParams.update({'font.size': 16})


plt.figure(figsize=(20,5))
plt.plot(np.linspace(10480-deltaL,47920+deltaL,len(zs_nodes[:,-1]))/1000,t_BCO/yr2s,color='red')
plt.xlim([10480/1000,47920/1000])
plt.xlabel('Offset along line (km)')
plt.ylabel(r'$\rm{Age}_{\rm{transition}}$ (yr)')
plt.savefig('NEGIS_iceage_at_transition.png',dpi=150,format='png')
plt.show()


plt.figure(figsize=(20,5))
plt.plot(np.linspace(10480-deltaL,47920+deltaL,len(zs_nodes[:,-1]))/1000,-(zs_nodes[:,-1]-max(zs_nodes[:,-1])),color='blue',label='Model output surface')
plt.gca().fill_between(np.linspace(10480-deltaL,47920+deltaL,len(zs_nodes[:,-1]))/1000,-(zs_nodes[:,-1]-max(zs_nodes[:,-1])),-5,facecolor='tab:blue', alpha=0.5)
plt.xlim([10480/1000,47920/1000])
plt.ylim([-1,14])
plt.gca().invert_yaxis()
plt.legend(loc='lower right')
plt.xlabel('Offset along line (m)')
plt.ylabel('Depth (m)')
plt.savefig('NEGIS_surface_modeloutput.png',dpi=150,format='png')
plt.show()

