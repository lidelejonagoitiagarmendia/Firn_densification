
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from create_mesh import inflow_marker, outflow_marker, top_marker, bottom_marker, obstacle_marker, c_x, c_y, L, H


try:
    
    import h5py 
    """H5PY JOAN BEHAR DA ESPLIZITUKI FENICS INPORTATU BAINO  LEHEN"""
    import gmsh
    import meshio
    
    
    #Set shortcut for calling geometry functions
    geom = gmsh.model.geo
    
    
    print('before----FENICS',h5py.__version__)
    print('before----FENICS',gmsh.__version__)
    print('before----FENICS',meshio.__version__)

except ImportError:
    print("meshio and/or gmsh not installed. Requires the non-python libraries:\n",
          "- libglu1\n - libxcursor-dev\n - libxinerama1\n And Python libraries:\n"
          " - h5py",
          " (pip3 install --no-cache-dir --no-binary=h5py h5py)\n",
          "- gmsh \n - meshio")
    exit(1)



from dolfin import *
from dolfin_adjoint import *
set_log_level(LogLevel.ERROR)


#------------some parameters


rho_trench= 550 #denser firn after cutting the trench out. measured
deltax_trench=0.0#distance from the trench to linearly smooth the density 
deltaz_trench=0.0
L00= 5 #maximum mesh width before optimization
h00=5 #maximum mesh height before optimization

tolerance=1.60585e-3 #optimization process tolerance <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Niter_opt=100 #100<<<<<<<<<<<<<<KONTUZ, SOILIK MUGATU HAU ARAZOAK BADITUGU

alpha = 0.1#0-1 #implicitly divided by gamma=1
beta = 0  #implicitly divided by gamma=1

omega = 2 #implicitly divided by gamma=1

Htot=0 
Hfloor=8 


Hdldh_prefix='_Hfloor'+str(Hfloor)+'_'+str(h00)+'h0_'+str(L00)+'L0_dl'+str(deltax_trench)+'_dh'+str(deltaz_trench)+'_'
res_folder='./results/' #folder to save the output


# Next, we load the facet marker values used in the mesh, as well as some
# geometrical quantities mesh-generator file.

# The initial (unperturbed) mesh and corresponding facet function from their respective
# xdmf-files.

mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)

mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("mf.xdmf") as infile:
    infile.read(mvc, "name_to_read")
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    
    
# We compute the initial volume of the obstacle

one = Constant(1)
Vol0 = L * H - assemble(one * dx(domain=mesh))


# We create a Boundary-mesh and function space for our control h

b_mesh = BoundaryMesh(mesh, "exterior")
S_b = VectorFunctionSpace(b_mesh, "CG", 1)
h = Function(S_b, name="Design")

zero = Constant([0] * mesh.geometric_dimension())

# We create a corresponding function space on :math:`\Omega`, and
# transfer the corresponding boundary values to the function
# :math:`h_V`. This call is needed to be able to represent
# :math:`h` in the variational form of :math:`s`.

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Mesh perturbation field")
h_V = transfer_from_boundary(h, mesh)
h_V.rename("Volume extension of h", "")


# We can now transfer our mesh according to :eq:`deformation`.

def mesh_deformation(h):
    # Compute variable :math:`\mu`
    print('mesh_deformation1---',h)
    
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)

    a = -inner(grad(u), grad(v)) * dx
    l = Constant(0) * v * dx

    mu_max = Constant(500, name="mu_min")     #Trukatu ditugu!
    mu_min = Constant(1, name="mu_max")
    bcs = []
    for marker in [inflow_marker, outflow_marker, top_marker, bottom_marker]:
        bcs.append(DirichletBC(V, mu_min, mf, marker))
    bcs.append(DirichletBC(V, mu_min, mf, obstacle_marker))

    mu = Function(V, name="mesh deformation mu")
    solve(a == l, mu, bcs=bcs)

    # Compute the mesh deformation
    S = VectorFunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(S), TestFunction(S)
    dObstacle = Measure("ds", subdomain_data=mf, subdomain_id=obstacle_marker)

    def epsilon(u):
        return sym(grad(u))

    def sigma(u, mu=500, lmb=0):
        return 2 * mu * epsilon(u) + lmb * tr(epsilon(u)) * Identity(2)

    a = inner(sigma(u, mu=mu), grad(v)) * dx
    L = inner(h, v) * dObstacle

    bcs = []
            
    class non_elastic(SubDomain):
        def inside(self,x,on_boundary):
            return x[1]<=Htot - Hfloor + 0.05 
        
    # bcs.append(DirichletBC(S.sub(1),0.0,non_elastic())) #method='pointwise'))
    bcs.append(DirichletBC(S,zero,non_elastic()))
    
    for marker in [inflow_marker, outflow_marker, top_marker, bottom_marker]:
        bcs.append(DirichletBC(S, zero, mf, marker))

    s = Function(S, name="mesh deformation")
    solve(a == L, s, bcs=bcs)
    
    return s

# We compute the mesh deformation with the volume extension of the control
# variable :math:`h` and move the domain.


s = mesh_deformation(h_V)
ALE.move(mesh, s)

###################################################################################
###################################################################################

#--------------------- FUNCTIONS FOR FORWARD PROBLEM------------------------------

    
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
    
    eps_dot=sym(grad(v))                      
    J1=tr(eps_dot) 
    J2=inner(eps_dot,eps_dot)
    eps_E2=1/a*(J2-J1**2/3) + (3/2)*(1/b)*J1**2
    viscosity = (1/2)**((1-nglen)/(2*nglen)) * Aglen**(-1/nglen) * (eps_E2)**((1-nglen)/(2*nglen))    
    sigma = viscosity * (1/a*(eps_dot-(J1/3)*Identity(2))+(3/2)*(1/b)*J1*Identity(2))
    
    return sigma

def A_glen_temperature(T):
    
    Q=60e3 #66.4e3 #Higashi, 1964
    R=8.314
    A0=3.985e-13 #zwinger
    
    return A0*exp(-Q/(R*T)) 


#---------------------PARAMETERS FOR FORWARD PROBLEM------------------------------

n=3
Tsite=-28.8 +273.15 #NEEM, hori txekeatu nahi dugulako

rho_surf=315#.3717043 #300
rho_ice= 910
rho_ice_softened= rho_ice - 0.2




phi_snow=0.4
ab_phi_lim=0.81

K=500 

Aglen=A_glen_temperature(Tsite)
print('-----------------------------Aglen=',Aglen)

day2s=24*60*60
yr2s=365.25*day2s

acc_rate= 1.55/yr2s
g=Constant((0.0,-9.82))

factor=3
dt=(0.025/factor*yr2s)

#---------------------SOLVE FORWARD PROBLEM------------------------------

q_degree = 3
dx = dx(metadata={'quadrature_degree': q_degree})  

#--------------------------------------FUNCTION SPACE------------------------------------------------#
    
#Define function space for density
deg=2  #1 #Polinomial degree 
U=FunctionSpace(mesh, 'Lagrange', deg); # Polynomial function space of order "deg" (=1 is linear, ...)

rho=Function(U) # the unknown function
wr=TestFunction(U)  # the weight function

#Define function space for velocity
deg=2
V=VectorFunctionSpace(mesh, "CG", 2)

v=Function(V) # the unknown function
wv=TestFunction(V)  # the weight function 

################################### INITIAL CONDITIONS #################################################

#first of all, identify the dimensions of the whole automatically
#not necessary when we create the hole by hand
#but it will be once adjoint starts to optimize the shape

#Get coordinates and velocities of the points at the obstacle's surface
mesh.init(2,1)
dofs = []
cell_to_facets = mesh.topology()(2,1)
for cell in cells(mesh):
    facets = cell_to_facets(cell.index())
    for facet in facets:
        if mf[facet] == obstacle_marker: #We have given the number 5 to the subdomain of the obstacle
            dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
            for dof in dofs_:
                dofs.append(dof)

unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
boundary_coords = V.tabulate_dof_coordinates()[unique_dofs] #surfaceko puntuen koordenatuak
hole_zmin=np.min(boundary_coords[:,1])
print('--:::HOLE-zmin=',hole_zmin)
hole_xmin=np.min(boundary_coords[:,0])
hole_xmax=np.max(boundary_coords[:,0])
print('--:::HOLE-xlims=(',hole_xmin,',',hole_xmax,')')

##############################################################3Initial density profile (from 1D problem)

rho_init_neem=np.load('rho_NEEM_n3_H0_180_fittedA(T)_K100.npy')
z_init_neem=np.load('z_NEEM_n3_H0_180_fittedA(T)_K100.npy')

z_init_neem_adjusted = z_init_neem - (np.max(z_init_neem) )#- H ) #adjusted to the new mesh height

# func spaces 
# self.S  = FunctionSpace(self.smesh, "CG", 1, constrained_domain=pbc_1D)
scoords_r0 = U.tabulate_dof_coordinates()
xs_dof_r0 = scoords_r0[:,0] # x coords of func space NODES
zs_dof_r0 = scoords_r0[:,1] # z coords of func space NODES
s_numdofs_r0 = len(zs_dof_r0)
ISz_r0 = np.argsort(zs_dof_r0)  

f_rho_neem = interp1d(z_init_neem_adjusted, rho_init_neem, kind='cubic',fill_value='extrapolate')

r_init=Function(U)


for ii in range(s_numdofs_r0):
    
            
    xii = xs_dof_r0[ii]# x coord of DOF (node)
    zii = zs_dof_r0[ii] # z coord of DOF (node)
    deltarho_trench= rho_trench - f_rho_neem(zii) #to smooth out the transition
    
    if ((xii>= (hole_xmin-deltax_trench) ) and (xii<= (hole_xmax+deltax_trench) ) and (zii>=hole_zmin-deltaz_trench)):
        
        r_init.vector()[ii]= rho_trench
    
    else:
        r_init.vector()[ii]=f_rho_neem(zii)
    
rho_init=project(r_init,U)
rho.assign(rho_init)
rho_prev=Function(U)
rho_prev.assign(rho_init) 
rho_prev=project(rho_prev,U)

u_init = Expression(('vx',"uztop - (uztop-uzbot)*pow((H-x[1])/H0,0.35)"),vx=Constant(0.0), H=H, H0=180, uztop=-acc_rate, uzbot=-acc_rate*(rho_surf/rho_ice),  degree=2)
v_init = interpolate(u_init, V)

v.assign(v_init)


#----------------------------------plotting initial conditions
plt.figure()
rhoplot=plot(rho_prev)
clb = plt.colorbar(rhoplot, orientation="vertical",
                    label='Density (kg/m3)')
plt.savefig(res_folder+'initial_hole_and_densities'+Hdldh_prefix+'.png',format='png',dpi=150)


plt.figure()
rhoplot=plot(rho_prev)
clb = plt.colorbar(rhoplot, orientation="vertical",
                    label='Density (kg/m3)')
plt.ylabel('z (m)')
plt.xlim(-5,5)
plt.ylim(-10,0)
plt.savefig(res_folder+'initial_hole_and_densities_ZOOM'+Hdldh_prefix+'.png',format='png',dpi=150)
# plt.show()


#########################################################Boundary conditions 

bc_v_b = DirichletBC(V,(0.0,0.0),mf,bottom_marker)
bc_v_l = DirichletBC(V.sub(0),0.0,mf,inflow_marker)
bc_v_r = DirichletBC(V.sub(0),0.0,mf,outflow_marker)

bcs_v=[bc_v_b,bc_v_l,bc_v_r]

#-----------------------------------------------------------------------------------------------------#
#--------------------------------------SOLVE FEM PROBLEM----------------------------------------------#
#-----------------------------------------------------------------------------------------------------# 

#--------------------------------GET a, b, VARIABLES AND SIGMA-----------------------------------------
a_,b_=get_ab(rho_prev,rho_ice,phi_snow,ab_phi_lim,n,K)
sigma=get_sigma(v,a_,b_,Aglen,n)

#-----------------------------------SOLVE MOMENTUM EQUATION--------------------------------------------
a_v = inner(sigma,grad(wv))*dx           #a_v = (Lambda_constant(a_,b_)*D(v)*wv.dx(0)+2*etha_constant()*D(v)*wv.dx(0))*dx  
L_v = rho_prev*inner(g,wv)*dx 

F_v = a_v - L_v

tol, relax, maxiter = 1e-2, 0.35, 50
solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
solve(F_v==0, v, bcs_v, solver_parameters=solparams)

v_sol=project(v,V)

# We compute the normal velocity integral around the obstacle,
# :math:`\int_{\Omega(s)} \sum_{i,j=1}^2 \left(\frac{\partial u_i}{\partial x_j}\right)^2~\mathrm{d} x`

ds = Measure("ds", domain=mesh, subdomain_data=mf)
dObstacle = ds(obstacle_marker)
normal = FacetNormal(mesh)
#tangent = as_vector([normal[1], -normal[0]])

#if we define alpha as alpha_hat/gamma, numbers are too small for the optimization program
#we will change the units of the collapse rate to m2/yr so that the magnitude is more appropiate

gamma=1 #it was 1e7 in m2/s  (1yr=3.154e7s)

Jcompression=assemble(dot(v, normal)*dObstacle)*yr2s #<<<<<<<<<<<<<<<<<<<<<<<<watch out, in m2/yr already
J = (gamma*assemble(dot(v, normal)*dObstacle)*yr2s )/gamma #redundant but just for the sake of clarity


# Then, we add a weak enforcement of the volume contraint,
# :math:`\alpha\big(\mathrm{Vol}(\Omega(s))-\mathrm{Vol}(\Omega_0)\big)^2`.

# alpha = 1.0/gamma #1e4
Vol = assemble(one * dx(domain=mesh))
Jvol= (L * H - Vol) - Vol0
J += alpha * ((L * H - Vol) - Vol0)**2


# Similarly, we add a weak enforcement of the barycenter contraint,
# :math:`\beta\big(\mathrm{Bc}_j(\Omega(s))-\mathrm{Bc}_j(\Omega_0)\big)^2`.


(x, y) = SpatialCoordinate(mesh)
Bc1 = (L**2 * H / 2 - assemble(x * dx(domain=mesh))) / (L * H - Vol)
Bc2 = (L * H**2 / 2 - assemble(y * dx(domain=mesh))) / (L * H - Vol)


(x, y) = SpatialCoordinate(mesh)

# beta = 0/gamma #1e3
Jcenter= (Bc1 - c_x)**2 + (Bc2 - c_y)**2
J += beta * ((Bc1 - c_x)**2 + (Bc2 - c_y)**2)


Jcurvature = omega*assemble((div(grad((x-c_x)**2))+div(grad((y-c_y)**2)))*dObstacle)
J += Jcurvature


file=open('Jdata.txt','a')
file.write('\n')
file.write('Normal velocity integral' + str(gamma*assemble(dot(v, normal)*dObstacle))+'\n')
file.write('Volume constraint'+str(alpha * ((L * H - Vol) - Vol0)**2)+'\n')
file.write('Barycenter constraint'+str(beta * ((Bc1 - c_x)**2 + (Bc2 - c_y)**2))+'\n')
file.close

# We define the reduced functional, where :math:`h` is the design parameter# and use scipy to minimize the objective.
 
def iter_cb(x): #callback function to store results from the minimization process

    #-------------------------------------------------scrapy but working-------------
    
    """#just to enter the respective callback function and save the value.
    #it doesn't change anything if the maximum iteration is 0, does it? check
    #agian gorde Jtotal honekin eta hau gabe eta ikusi identikoak diren"""
    
    _= minimize(Jhat_compression, tol=1e6, options={"gtol": 1e6, "maxiter": 0, "disp": True}) ;
    _= minimize(Jhat_volume, tol=1e6, options={"gtol": 1e6, "maxiter": 0, "disp": True}) ;
    _= minimize(Jhat_center, tol=1e6, options={"gtol": 1e6, "maxiter": 0, "disp": True}) ;
    # _= minimize(Jhat_curvature, tol=1e6, options={"gtol": 1e6, "maxiter": 0, "disp": True}) ;
    


# Reduced functional Callback called at each iteration
total_obj_list = []
def eval_cb(j, a):
    print('\n\n                                              JJJJJJ__TOTAL__JJJJJJ_=',j)
        
        #Get coordinates and velocities of the points at the obstacle's surface
    mesh.init(2,1)
    dofs = []
    cell_to_facets = mesh.topology()(2,1)
    for cell in cells(mesh):
        facets = cell_to_facets(cell.index())
        for facet in facets:
            if mf[facet] == obstacle_marker: #We have given the number 5 to the subdomain of the obstacle
                dofs_ = S.dofmap().entity_closure_dofs(mesh, 1, [facet])
                for dof in dofs_:
                    dofs.append(dof)
                    
    unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
    boundary_coords = S.tabulate_dof_coordinates()[unique_dofs] #surfaceko puntuen koordenatuak
    # for i, dof in enumerate(unique_dofs):
    #     print(boundary_coords[i], s.vector()[dof])
    
    print('coorssum',np.sum(boundary_coords[:][0]))
    print('ssum',np.sum(s.vector().get_local()))
    
    # print('aaaaaaaaaaaa',a) #deribatua, ufl objektua
    total_obj_list.append(j)


Jcompression_list = []
def eval_Jcompression_cb(j, a):
    print(' \n\n                                             jjjjj__COMPRESSION=',j)
    # print('aaaaaaaaaaaa',a) #deribatua, ufl objektua
    Jcompression_list.append(j)
    
Jvol_list = []
def eval_Jvol_cb(j, a):
    print('  \n\n                                                   jjjjj__VOLUME=',j)
    # print('aaaaaaaaaaaa',a) #deribatua, ufl objektua
    Jvol_list.append(j)

Jcenter_list = []
def eval_Jcenter_cb(j, a):
    print('  \n\n                                                    jjjjj__CENTER=',j)
    # print('aaaaaaaaaaaa',a) #deribatua, ufl objektua
    Jcenter_list.append(j)

Jcurvature_list = []
def eval_Jcurvature_cb(j, a):
    print('  \n\n                                                    jjjjj__CENTER=',j)
    # print('aaaaaaaaaaaa',a) #deribatua, ufl objektua
    Jcurvature_list.append(j)

Jhat = ReducedFunctional(J, Control(h),eval_cb_post=eval_cb)

#________________CHECK THAT IT WORKS AS INTENDED___________________________
#just to evaluate the functions. we have not found a more elegant way
Jhat_compression = ReducedFunctional(Jcompression, Control(h),eval_cb_post=eval_Jcompression_cb)
Jhat_volume = ReducedFunctional(Jvol, Control(h),eval_cb_post=eval_Jvol_cb)
Jhat_center = ReducedFunctional(Jcenter, Control(h),eval_cb_post=eval_Jcenter_cb)
Jhat_curvature = ReducedFunctional(Jcurvature, Control(h),eval_cb_post=eval_Jcurvature_cb)


s_opt = minimize(Jhat, tol=tolerance, options={"gtol": tolerance, "maxiter": Niter_opt, "disp": True}, callback=iter_cb)
#this minimize is scipy's. Documentation about the callback function can be found there

Vol = assemble(one * dx(domain=mesh))

#-----------------------------------saving the results-------------------

alpha_str='{:.0e}'.format(alpha)
beta_str='{:.0e}'.format(beta)
gamma_str='{:.0e}'.format(gamma)
omega_str='{:.0e}'.format(omega)
tol_str='{:.0e}'.format(tolerance)
print(alpha_str,beta_str,gamma_str)


iteration_suffix= Hdldh_prefix + str(Niter_opt)+'iter_'+tol_str+'tol_'
constants_suffix= '(K'+str(K)+'alpha'+alpha_str+'beta'+beta_str+'gamma'+gamma_str+'(m2yr-1)_omega'+omega_str+')'

np.save(res_folder+'Jcompressionrate'+iteration_suffix+constants_suffix+'.npy',np.array(Jcompression_list))
np.save(res_folder+'Jvolumechange'+iteration_suffix+constants_suffix+'.npy',np.array(Jvol_list))
np.save(res_folder+'Jcenterchange'+iteration_suffix+constants_suffix+'.npy', np.sqrt(np.array(Jcenter_list)))
np.save(res_folder+'Jcurvaturechange'+iteration_suffix+constants_suffix+'.npy', np.array(Jcurvature_list))
np.save(res_folder+'Jtotal'+iteration_suffix+constants_suffix+'.npy', np.array(total_obj_list))


#----------------------------------------------------plotting--------------------

plt.figure()
plt.plot(np.array(Jcompression_list),lw=1)
plt.ylabel('Initial collapse rate (m2/yr)')
plt.title(iteration_suffix+constants_suffix)
plt.xlabel('Iterations')
plt.savefig(res_folder+'Jcompressionrate'+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)


plt.figure()
plt.plot(np.array(Jvol_list),lw=1)
plt.ylabel('Initial proposal volume change (relative to V0) (m2)')
plt.xlabel('Iterations')
plt.title(iteration_suffix+constants_suffix)
plt.savefig(res_folder+'Jvolumechange'+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)


plt.figure()
plt.plot((np.array(Jvol_list)+Vol0)/Vol0,lw=1)
plt.ylabel('Initial proposal volume (relative to V0) (m2)')
plt.xlabel('Iterations')
plt.title(iteration_suffix+constants_suffix)
plt.savefig(res_folder+'JvolumeRELATIVE'+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)


plt.figure()
plt.plot(np.sqrt(np.array(Jcenter_list)),lw=1)
plt.ylabel('Baricenter distance to original center (m)')
plt.xlabel('Iterations')
plt.title(iteration_suffix+constants_suffix)
plt.savefig(res_folder+'Jcenterchange'+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)

plt.close()

plt.figure()
plt.plot(np.array(Jcurvature_list),lw=1)
plt.ylabel('Curvature')
plt.xlabel('Iterations')
plt.title(iteration_suffix+constants_suffix)
plt.savefig(res_folder+'Jcurvaturechange'+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)

plt.close()


#triple plot----------------------------------------------

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)
twin1 = ax.twinx()
twin2 = ax.twinx()
# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
twin2.spines.right.set_position(("axes", 1.2))

p1, = ax.plot(np.array(Jcompression_list), c='k', label="-dV/dt (t=0)")
p2, = twin1.plot((np.array(Jvol_list)+Vol0)/Vol0, c='tab:orange', label="V/V0")
p3, = twin2.plot(np.sqrt(np.array(Jcenter_list)),c='tab:green', label="Δr_center")


ax.set_xlabel("Iterations")
ax.set_ylabel("Initial collapse rate (m2/yr)")
twin1.set_ylabel("Initial proposal volume (relative to V0) (m2)")
twin2.set_ylabel("Baricenter distance to original center (m)")

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2, p3])

plt.savefig(res_folder+'Jtriple'+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)


plt.close(fig)

#---
fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)
twin1 = ax.twinx()
twin2 = ax.twinx()
# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
twin2.spines.right.set_position(("axes", 1.2))

twin2.axhline(1,ls='--',c='tab:orange',lw=0.25)
p2, = twin1.plot((np.array(Jvol_list)+Vol0)/Vol0, c='tab:orange', label="V/V0")
p3, = twin2.plot(np.sqrt(np.array(Jcenter_list)),c='tab:green', label="Δr_center")
p1, = ax.plot(np.array(Jcompression_list)/Vol0, c='k',lw=3, label="-(dV/dt)/V0 (t=0)")

# ax.set_xlim(0, 2)
ax.set_ylim(0, 1)
twin1.set_ylim(0.95, 1.025)
twin2.set_ylim(0, 1)

ax.set_xlabel("Iterations")
ax.set_ylabel("Initial collapse rate (Fraction of volume lost in the first year)",fontsize=8)
twin1.set_ylabel("Initial proposal volume (relative to V0) (m2)")
twin2.set_ylabel("Baricenter distance to original center (m)")

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2, p3])

plt.savefig(res_folder+'Jtriple_FINAL'+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)




#-------------------------------------------------------



plt.figure()
plt.plot(np.array(total_obj_list),lw=1.5,c='k',label='Jtotal')
plt.plot(alpha * (np.array(Jvol_list)**2),lw=1.0,label='Jvolume (alpha='+alpha_str+')')
plt.plot(beta * np.array(Jcenter_list),lw=1.0,c='tab:green',label='Jcenter (beta='+beta_str+')')
plt.plot(gamma*np.array(Jcompression_list),lw=1.0,label='Jcollapserate (gamma='+gamma_str+')')
plt.ylabel('cost functions')
plt.xlabel('Iterations')
plt.title(iteration_suffix+constants_suffix)
plt.legend()
plt.ylim(0,np.max(np.array(total_obj_list)[0:25]))
plt.savefig(res_folder+'Jtotal'+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)


plt.figure()
initial, _ = plot(mesh, color="gray", linewidth=0.25, label="Initial mesh")
plt.ylabel('z (m)')
plt.xlim(-5,5)
plt.ylim(-10,0)
plt.savefig(res_folder+"mesh_INITIAL_ZOOM_"+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)




plt.figure()
initial, _ = plot(mesh, color="gray", linewidth=0.25, label="Initial mesh")
Jhat(s_opt)
print(Jhat(s_opt))
print(s_opt)
optimal, _ = plot(mesh, color="r", linewidth=0.175, label="Optimal mesh")
plt.ylabel('z (m)')
plt.xlim(-5,5)
plt.ylim(-10,0)
# plt.axis("off")
plt.savefig(res_folder+"meshes_ZOOM_"+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)



plt.figure()
# initial, _ = plot(mesh, color="b", linewidth=0.25, label="Initial mesh")
Jhat(s_opt)
print(Jhat(s_opt))
print(s_opt)
optimal, _ = plot(mesh, color="r", linewidth=0.25, label="Optimal mesh")
plt.savefig(res_folder+"meshes"+iteration_suffix+constants_suffix+".png", dpi=200, bbox_inches="tight", pad_inches=0)



#and saving the mesh--------------------------------------------------

Jhat(s_opt) #setting the mesh to be the last annotated mesh

meshname='optimal_mesh_squarecircle_'+iteration_suffix+constants_suffix
print(meshname)

# #hau da h5 moduan gordetzeko,, baina hau da funtzioak gordetzeko eta ez meshak. hoenk ez du balio
# fFile=HDF5File(MPI.comm_world,meshname+'.h5','w')
# fFile.write(v_sol,"/f")
# fFile.close()

# with XDMFFile(mpi_comm_world(), meshname+'.xdmf') as xdmf:
#     xdmf.write(mesh)
    
with XDMFFile(MPI.comm_world, res_folder+meshname+'.xdmf') as xdmf:
    xdmf.write(mesh)
    
    
#------------------ proper minimization magnitude evolution graphs-----------------------


Jcomp=np.load(res_folder+'Jcompressionrate'+iteration_suffix+constants_suffix+'.npy')
Jvol=np.load(res_folder+'Jvolumechange'+iteration_suffix+constants_suffix+'.npy')
Jcent=np.load(res_folder+'Jcenterchange'+iteration_suffix+constants_suffix+'.npy')  #watch out! this is saved as np.sqr(Jcent)
Jtotal_ext=np.load(res_folder+'Jtotal'+iteration_suffix+constants_suffix+'.npy')




Next=len(Jtotal_ext)
Nsmall=len(Jvol)


Jcomp_ext=np.zeros(Next)
Jvol_ext=np.zeros(Next)
Jcent_ext=np.zeros(Next)


print('\n\n\n\n <<<<<<<<<<<<<<<<FINISHED>>>>>>>>>>>>>>>>>\n\n\n')
print('number of optimization iterations accepted =', Next)
