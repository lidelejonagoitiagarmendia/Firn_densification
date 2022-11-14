
from math import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interpolatescipy
from scipy.interpolate import interp1d
import imageio #for the gif
import os #for the gif
import os.path
import traceback

""" GOIKO HORIEK EZ DUTE BERTSIO ARAZORIK EMATEN"""


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



from fenics import *   
from dolfin import *

#control verbosity of solve function 
set_log_level(21)

#################################################
    
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



"""
Define boundaries of the domain
In order to define the inner irregular boundary, we define it first as the whole
domain of boundaries, and start subtracting the rest of the well defined boundaries 
by properly defining them
"""

class obstacle_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class bottom_boundary(SubDomain):
    def inside(self, x, on_boundary):
        # return on_boundary and near(x[1],0)
        return on_boundary and (x[1]<1-H)  

class top_boundary(SubDomain):  
    
    def inside(self,x,on_boundary):
        
        dz=0.25 #how many meters from the 
        return on_boundary and (x[1]> (zmin-dz))   #horrela definitu dezakegu gero eremua definitiuko ditugulako zatika
        # return on_boundary and (x[1]> (H-dz)) #soilik momentuz, lehenengo iteraziorako
    
    
class right_boundary(SubDomain):
    def inside(self,x,on_boundary):
        # return on_boundary and near(x[0],L)
        return on_boundary and (x[0]>L-1-L/2)  

class left_boundary(SubDomain):
    def inside(self,x,on_boundary):
        # return on_boundary and near(x[0],0)  
        return on_boundary and (x[0]<1-L/2)  



"""-----------------------------------------------------REMESHING-------------"""

def msh2xdmf(filename,outname="mesh_temp.xdmf"):
    
    msh = meshio.read(filename)
    
    line_cells = []
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif cell.type == "line":
            if len(line_cells) == 0:
                line_cells = cell.data
            else:
                line_cells = np.vstack([line_cells, cell.data])
    
    line_data = []
    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "line":
            if len(line_data) == 0:
                line_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                line_data = np.vstack([line_data, msh.cell_data_dict["gmsh:physical"][key]])
        elif key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    
    triangle_mesh = meshio.Mesh(points=msh.points[:, :2], cells={"triangle": triangle_cells},
                                cell_data={"name_to_read": [triangle_data]})
    
    line_mesh = meshio.Mesh(points=msh.points[:, :2], cells=[("line", line_cells)],
                            cell_data={"name_to_read": [line_data]})
    meshio.write(outname, triangle_mesh)
    #meshio.write("mf_ir.xdmf", line_mesh)

def angle(x,z,xc,zc): 
    
    if x>=xc:  
        alpha= np.arctan((z-zc)/(x-xc))
    else:
        alpha= np.pi + np.arctan((z-zc)/(x-xc))
        
    if alpha>2*np.pi:
        alpha-=2*np.pi
        
    elif alpha<0:
        alpha+=2*np.pi
        
    return alpha


def dilation(xs,zs,factor=0.95):
    
    Npoints=len(xs)
    
    xdil=np.zeros(Npoints)
    zdil=np.zeros(Npoints)
    
    xc=np.mean(xs)
    zc=np.mean(zs)
    
    for i in range(Npoints):
        
        r=np.sqrt((xc-xs[i])**2 + (zc-zs[i])**2)
        alpha = angle(xs[i],zs[i],xc,zc)
        
        r_new= factor * r
        
        xdil[i]= xc + r_new*np.cos(alpha)
        zdil[i]= zc + r_new*np.sin(alpha)
        
    return xdil,zdil


    
def sort_hole_fenics_v2(hole_coords,ndmin=4): 
    
        """arguments: 
        scoords = U.tabulate_dof_coordinates()
        
        ndmin=how many nearest neighbors will be checked
        
        We do not care where the points start, we just need them to follow a direction around
        the hole and back to the start
        
        Much more complicated to sort,since it is a closed loop and we can have 
        a lot of identical xs and zs
        
        we were thinking of dividing them in quadrants but it might not be enough
        
        We will try to define a direction and sort them by distance to the last point

        """
        

        
        xs_hole_xsorted, zs_hole_xsorted, s_numdofs = sort_fenics(hole_coords,axis=0) #just for the algorithm to be quicker
        s_numdofs=len(xs_hole_xsorted)
        
        xdil,zdil =  dilation(xs_hole_xsorted, zs_hole_xsorted,factor=0.94)
        
        #points sorted to follow the loop
        xs_hole=[]
        zs_hole=[]
        
        #arbitrary initial point. (away from spikes)
        
        isafe=int(s_numdofs/2)
        
        xs_hole.append(xs_hole_xsorted[isafe])
        zs_hole.append(zs_hole_xsorted[isafe])
        
        x_rn = xs_hole_xsorted[isafe] #coordinates of the point we are calculating the distance from rn
        z_rn = zs_hole_xsorted[isafe]
        
        xd_rn = xdil[isafe]
        zd_rn = zdil[isafe]
        
        #delete it from the list of points to be sorted
        xs_hole_xsorted= np.delete(xs_hole_xsorted,isafe)
        zs_hole_xsorted= np.delete(zs_hole_xsorted,isafe)
        
        xdil= np.delete(xdil,isafe)
        zdil= np.delete(zdil,isafe)
        
        #calculate closest point and follow the loop from there
        #the direction will be random but not important
        #we can maybe improve the cost of this function by searching just in the closest half
        
        for ii in range(s_numdofs-1):
            
            ndmin=min(ndmin,len(xs_hole_xsorted))
            
            dist= (xs_hole_xsorted - x_rn)**2 + (zs_hole_xsorted - z_rn)**2
            i_mins=np.argsort(dist)[:ndmin] #indexes of the ndmin minimum distances
            

            ref_angle=angle(x_rn,z_rn,xd_rn,zd_rn)
            
            #angle with respect to inner close point. OXY frame of reference
            alpha_mins=np.zeros(ndmin)
            #angle with respect to inner close point. POINT_rn frame of reference
            alpha_mins_ref=np.zeros(ndmin)
            
            for i in range(ndmin):
                
                alpha_mins[i]=angle(xs_hole_xsorted[i_mins[i]],zs_hole_xsorted[i_mins[i]],xd_rn,zd_rn)
                
                dif = alpha_mins[i] - ref_angle
                if dif > 0:
                    alpha_mins_ref[i] = dif
                else:
                    alpha_mins_ref[i] = dif + 2*np.pi
                 
                
            
            i_next= i_mins[np.argmin(alpha_mins_ref)]
            
            
            #append
            xs_hole.append(xs_hole_xsorted[i_next])
            zs_hole.append(zs_hole_xsorted[i_next])
            
            x_rn = xs_hole_xsorted[i_next] #coordinates of the point we are calculating the distance from rn
            z_rn = zs_hole_xsorted[i_next]
            
            xd_rn = xdil[i_next]
            zd_rn = zdil[i_next]
            
            #delete it from the list of points to be sorted
            xs_hole_xsorted= np.delete(xs_hole_xsorted,i_next)
            zs_hole_xsorted= np.delete(zs_hole_xsorted,i_next)
            
            xdil= np.delete(xdil,i_next)
            zdil= np.delete(zdil,i_next)
            
        
        return xs_hole, zs_hole
        
        


#Set shortcut for calling geometry functions
geom = gmsh.model.geo



def remesh(xs_hole,zs_hole,xs_surf,zs_surf,L,tstep, maxN=75, mode='linear',outname='mesh_temp.msh'): #jarri txukunago gero
    
    """mode can be linear or spline,but spline smoothes the shape too much"""
    
    #Define size of mesh for different places
    tm = 2.0    #Exterior
    tmsurf =0.2  #surface
    tmr = 0.25 #Obstacle
    
    #Initialize mesh
    gmsh.initialize()
    
    # gmsh.option.general.AbortOnError(3) #honek ez du funtzionatzen, 3-ra aldatu behar dut baina ez dakit nola
    # #aldatu behar da baina ez dakit zela aldatzen den python API-tik General.AbortOnError
    #http://gmsh.info/dev/doc/texinfo/gmsh.pdf
    
    
    # print(gmsh.option.getNumber('General.AbortOnError'))
    # gmsh.option.setNumber('General.AbortOnError',4)    #hala ere hoenk ez du mozten
    # print(gmsh.option.getNumber('General.AbortOnError'))


    gmsh.option.setNumber('General.Verbosity',2) 
    
    ######### Create obstacle curve using spline ###########
    
    N_hole=len(xs_hole)
    
    """The resolution of each point does not take into account how many 
    points it already has around, so there is a positive feedback loop
    that makes the resolution go crazy from a certain resolution to
    point density ratio. 
    need to limit the maximum number of points by hand"""
    
    # maxN=150 #to be adjusted. assemble() glitches at around 2000
    portion= (N_hole//maxN) + 1 #portion of the points to keep
    
    print('$$$$$$$$$$$$$$$$$$$$$N_hole=',N_hole,'$$$$$$$$$$$$$portion=',portion)
    xs_hole = xs_hole[::portion]
    zs_hole = zs_hole[::portion]
    N_hole=len(xs_hole)
    print('                  >>>>>>>>>>>>>>>new N_hole=',N_hole,'<<<<<<<<<<<<<<<<<<<')
            
    
    ps_hole=[]
    
    for i in range(N_hole):
        
        ps_hole.append(geom.addPoint(xs_hole[i],zs_hole[i], 0,tmr))
    
    ps_hole.append(1) #First and last points (tags) must be the same to have a close boundary!

    #Create 'line' by interpolating around the give points
    
    if mode=='spline':
        
        curve_hole = geom.addBSpline(ps_hole,-1)
        #Create the curve loop of the hole
        hole = geom.addCurveLoop([curve_hole])
        
    elif mode=='linear':
        
        curve_hole=[]
        for i in range(N_hole):
            curve_hole.append(geom.addLine(ps_hole[i],ps_hole[i+1]));

        #Create the curve loop of the hole
        hole = geom.addCurveLoop(curve_hole);
    
    
    ######### Create exterior boundary using spline ###########

    H_left=zs_surf[np.argmin(xs_surf)]
    H_right=zs_surf[np.argmax(xs_surf)]

    #Irregular surface (left to right)
    xs1=np.concatenate(([0-L/2],xs_surf,[L-L/2]))
    ys1=np.concatenate(([H_left],zs_surf,[H_right]))
    

    #Add all the surface points to mesh
    ps_surf=[]
    
    for i in range(len(xs1)):
        ps_surf.append(geom.addPoint(xs1[i],ys1[i], 0,tmsurf))
           
    p1=geom.addPoint(L-L/2,0-H,0,tm)
    p2=geom.addPoint(0-L/2,0-H,0,tm)
    
    l1=geom.addBSpline(ps_surf,-1)
    l2=geom.addLine(N_hole+len(xs1),p1)
    l3=geom.addLine(p1,p2)
    l4=geom.addLine(p2,N_hole+1)
    
    ext=geom.addCurveLoop([l1,l2,l3,l4])

    ############ Generate the mesh itself ##############
    
    #Create surface between exterior and hole
    s = geom.addPlaneSurface([ext,hole])
    
    # Create the physical surface 
    gmsh.model.addPhysicalGroup(2, [s], tag=tstep)
    gmsh.model.setPhysicalName(2, tstep, "Firn/ice")
    
    #Generate mesh and save
    geom.synchronize()
    gmsh.model.mesh.generate(2)
    
    gmsh.option.setNumber('Mesh.SurfaceFaces', 1)
    gmsh.option.setNumber('Mesh.Points', 1)
    
    
    gmsh.write(outname) #write msh file, but we ned xdmf to work with fenics
    msh2xdmf(outname,outname="mesh_temp_500.xdmf") #rewrite the file into an xdmf to be read
    
    gmsh.finalize() #important to close gmsh!!

def sort_fenics(scoords,axis):
    
        """arguments: 
        
        scoords = U.tabulate_dof_coordinates()
        axis---with respect to which axis should it be sorted---(xs=0,zs=1)
        
        """
        step=1
        
        axis_dof= scoords[:,axis] #axis coords of func space NODES (xs=0,zs=1)
        IS = np.argsort(axis_dof)
        s_numdofs= len(axis_dof) 
        
        scoords_x=scoords[:,0][IS] #sorted
        scoords_z=scoords[:,1][IS]
        
        
        if (np.abs(s_numdofs-2*len(np.unique(axis_dof)))<2):
            
            print('doubled')
            step=2
        
        return scoords_x[::step],scoords_z[::step], int(s_numdofs/2)
    


def smooth_initial_mesh(xs_hole0,zs_hole0,H,L,Hfloor,h00,L00,deltax_trench,outname='mesh_original'): #jarri txukunago gero
    
    """mode can be linear or spline,but spline smoothes the shape too much"""
    
    #Define size of mesh for different places
    tm = 1   #Exterior
    tmsurf =0.25  #surface
    tmr = 0.25 #Obstacle
    
    #.................................................handle the irregularities.....................
    
    #identify max horizontal extent above epsh height (just to avoid defaulting to the fixed floor)
    
    epsh= 0- Hfloor + 0.40*h00
    
    imax= np.argmax(xs_hole0[zs_hole0>epsh])
    
    x_tangent=xs_hole0[zs_hole0>epsh][imax]
    z_tangent=zs_hole0[zs_hole0>epsh][imax]
    
    print(x_tangent)
    print(z_tangent)
    
    xs_top= xs_hole0[zs_hole0>=z_tangent]
    zs_top= zs_hole0[zs_hole0>=z_tangent]
    
    
    #-----------------------------------------check that points are not doubled, and delete them if so
    
    print('\n\n len(zceilings) before------->',len(zs_top))
    
    
    xunique=[] #the one where sure there won't be any duplicates
    zunique=[]
    
    
    for i in range(len(xs_top)):
        
        xzref= (xs_top[i],zs_top[i])
        inside=False
        
        for j in range(len(xunique)):
            
            xzcomp = (xunique[j],zunique[j])
            
            if xzref==xzcomp:
                inside=True
                
        if (not inside):
            
            xunique.append(xs_top[i])
            zunique.append(zs_top[i])
 
            
    #save back unique values
    xs_top=np.copy(np.array(xunique))
    zs_top=np.copy(np.array(zunique))
    
    
    print('\n\n len(zceilings) after------->',len(zs_top))
    
    #------------------------------------------------------------
    
    Lhalf_prima= (max(xs_top))
    
    plt.figure()
    plt.scatter(xs_hole0,zs_hole0,s=2)
    plt.scatter(xs_top,zs_top,s=2)
    plt.axvline( Lhalf_prima)
    plt.axvline(- Lhalf_prima)
    plt.savefig(output_folder+'rawmesh_irakurrita.png',format='png',dpi=150)
    plt.close()
    # plt.show()
    
    
    L00=Lhalf_prima*2
    h00= np.max(zs_top) - (H-Hfloor)
    print(h00)
    
    # itxi()
    
    #------------------------------------------------------
    #and finally save the top part to interpolate it. SORT IT FIRST (we will choose left to right)
    
    Ixs=np.argsort(xs_top)
    
    xs_hole=xs_top[Ixs] #watch out, in this case onlythe top part will be saved here
    zs_hole=zs_top[Ixs]
    
    
    #..............................................................................................
    #..............................................................................................
    #......................................SAVE THE SMOOTHED MESH-FILE TO BE READ................
    #..............................................................................................
    #..............................................................................................
    
    #Initialize mesh
    gmsh.initialize()


    N_hole=len(xs_hole)
    print('                  >>>>>>>>>>>>>>>new N_hole=',N_hole,'<<<<<<<<<<<<<<<<<<<')
            
    
    ps_hole=[]
    
    # plt.figure()
    
    for i in range(N_hole):
        
        ps_hole.append(geom.addPoint(xs_hole[i],zs_hole[i], 0,tmr))
        # plt.scatter(xs_hole[i],zs_hole[i])

    
    # ps_hole.append(1) #commented because it's not a closed loop anymore
    
    r_corner=0.9 #m to smooth the lower corners
    
    
    p1h=geom.addPoint(L/2 + L00/2 - L/2, H-Hfloor + r_corner -H,0,tmr)
    p2h=geom.addPoint(L/2 + L00/2 - r_corner - L/2,H-Hfloor  -H,0,tmr)
    p3h=geom.addPoint(L/2 - L00/2 + r_corner- L/2 ,H-Hfloor -H ,0,tmr)
    p4h=geom.addPoint(L/2 - L00/2 - L/2,H-Hfloor + r_corner -H ,0,tmr)
    
    pc1h=geom.addPoint(L/2 + L00/2 - r_corner -L/2, H-Hfloor + r_corner -H,0,tmr)
    pc2h=geom.addPoint(L/2 - L00/2 + r_corner-L/2, H-Hfloor + r_corner -H,0,tmr)
    
    
    #create a close loop of lines. cloclwise because top will be left to right
    #--------------------Create 'line' by interpolating around the give points
    
    curve_hole=[]
    for i in range(N_hole-1):
        curve_hole.append(geom.addLine(ps_hole[i],ps_hole[i+1]));
        
    curve_hole.append(geom.addLine(ps_hole[N_hole-1],p1h));

    curve_hole.append(geom.addCircleArc(p1h,pc1h,p2h));
    curve_hole.append(geom.addLine(p2h,p3h));

    curve_hole.append(geom.addCircleArc(p3h,pc2h,p4h));
    curve_hole.append(geom.addLine(p4h,ps_hole[0]));
   

    #Create the curve loop of the hole
    hole = geom.addCurveLoop(curve_hole);
    
    
    ######### Create exterior boundary  ###########

     
    p1=geom.addPoint(L-L/2,0-H,0,tm)
    p2=geom.addPoint(0-L/2,0-H,0,tm)
    p3=geom.addPoint(0-L/2,H-H,0,tm)
    p4=geom.addPoint(L-L/2,H-H,0,tm)
    
    l1=geom.addLine(p1,p2)
    l2=geom.addLine(p2,p3)
    l3=geom.addLine(p3,p4)
    l4=geom.addLine(p4,p1)
    
    ext=geom.addCurveLoop([l1,l2,l3,l4])
    
    
    print('checkpoint1')


    ############ Generate the mesh itself ##############
    
    #Create surface between exterior and hole
    s = geom.addPlaneSurface([ext,hole])
    
    print('checkpoint2')
    
    # Create the physical surface
    gmsh.model.addPhysicalGroup(2, [s], tag=101)
    gmsh.model.setPhysicalName(2, tstep, "Firn/ice")
    
    
    print('checkpoint3')
    #Generate mesh and save
    
    geom.synchronize()
    
    print('checkpoint4')
    
    gmsh.model.mesh.generate(2)
    
    print('mesh created successfully---')
    
    gmsh.option.setNumber('Mesh.SurfaceFaces', 1)
    gmsh.option.setNumber('Mesh.Points', 1)
    
    
    gmsh.write(outname+'.msh') #write msh file, but we ned xdmf to work with fenics
    msh2xdmf(outname+'.msh',outname=outname+'.xdmf') #rewrite the file into an xdmf to be read
    
    gmsh.finalize() #important to close gmsh!!
    
    
    return h00,L00,deltax_trench

    
def A_glen_temperature(T):
    
    Q=60e3 #66.4e3 #Higashi, 1964
    R=8.314
    A0=3.985e-13 #zwinger
    
    return A0*exp(-Q/(R*T)) 


############################################################################################################
#################################### READ ALL THE MESH FILENAMES FROM THE RESULT FOLDER #################

#current directory

parent=os.getcwd()
print(parent)

#change directory

os.chdir('./results')
resdir=os.getcwd()
print(resdir)


#files with pattern matching 'useful for getting all the meshes only

allfiles=os.listdir()

all_xdmf=[]
all_h5=[]


for filename in allfiles:
    if ('.xdmf' in filename): all_xdmf.append(filename)
    if ('.h5' in filename): all_h5.append(filename)
    

#mesh.h5

print(*all_xdmf,sep='\n')


#just to be safe....................................................................................
if ( len(all_xdmf) != len(all_h5) ):   
    #checking-------------------------
    for i in range(len(all_xdmf)):
        print(all_xdmf[i])
        print(all_h5[i])
        
    print(len(all_xdmf),len(all_h5))
    raise Exception('MISMATCH between xdmf and h5 mesh files. check files read')
#.................................................................................................


#read paramters from filename

Hfloors=[]
h00s=[]
L00s=[]
dh0s=[]
dL0s=[]
tol0s=[]

    
for filename in all_xdmf:
    
    """for filenames of the following form
    optimal_mesh_squarecircle__Hfloor7_6h0_8L0_dl0_dh0_100iter_5e-03tol_(K500alpha1e+00beta0e+00gamma1e+00(m2yr-1)).xdmf
    """
    
    q=filename.split('_')

    Hfloors.append(float(q[4].replace('Hfloor','')))
    h00s.append(float(q[5].replace('h0','')))
    L00s.append(float(q[6].replace('L0','')))
    dh0s.append(float(q[8].replace('dh','')))
    dL0s.append(float(q[7].replace('dl','')))
    tol0s.append(float(q[10].replace('tol','')))


    if (dL0s[-1]>3 or dh0s[-1]>3): raise Exception('Check dl and dh values reading. too big, probably a coma is missing in the filename')

#back to parent directory
os.chdir('./..')

#check if a file with solved cases exists already and, if not, create an empty one 
solved_arr_filename='already_solved.txt'

if os.path.isfile(solved_arr_filename):
    
    print ("Solved solution file exist already")

    with open(solved_arr_filename, 'r') as f: #loading the data
        mystring = f.read()
    already_solved= mystring.split("\n")
    
    print('\n\n\n                <<<<<<<<<<<<<<ALREADY EVOLVED>>>>>>>>>>>>>')
    print(*already_solved,sep='\n')
    
    
else:
    print ("File does not exist")
    already_solved=[]



#------------------problematic runs
ignored_arr_filename='already_ignored.txt'

if os.path.isfile(ignored_arr_filename):
    
    print ("Problem file exist already")

    with open(ignored_arr_filename, 'r') as f: #loading the data
        mystring = f.read()
    problems= mystring.split("\n")
    
    print('\n\n\n                <<<<<<<<<<<<<<TO BE IGNORED>>>>>>>>>>>>>')
    print(*problems,sep='\n')

    
else:
    print ("problem File does not exist")
    problems=[]


#################################################################################################
######################################################################################################

for jj in range(len(all_xdmf)):
    
    plt.show()
    
    print('\n\n\n            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n\n')
    print('-------------------jj',jj,'--------',all_xdmf[jj],'\n\n\n')
    print('\n\n\n            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n\n')

    
    if ((all_xdmf[jj] in already_solved) or (all_xdmf[jj] in problems)):
        print(all_xdmf[jj],' has already been evolved')
        continue
        
    
    try:
        
    
        Hfloor=Hfloors[jj]
        h00=h00s[jj]
        L00=L00s[jj]
        deltax_trench=dL0s[jj]
        deltaz_trench=dh0s[jj]
        tol0=tol0s[jj]
        
        
        Hdldh_prefix='_Hfloor'+str(Hfloor)+'_'+str(h00)+'h0_'+str(L00)+'L0_dl'+str(deltax_trench)+'_dh'+str(deltaz_trench)+'_'
    
        res_folder='./results/' #folder where the optimization results are located. NOT THE OUTPUT FOLDER
        meshfilename = res_folder + all_xdmf[jj]  #with the '.xdmf'  included
    
        #create the time evolution output folder if it does not exist already
        output_folder='./results_TIMEevolution/' + Hdldh_prefix +'/'
        
        try:
            os.mkdir(output_folder)
        except OSError as error:
            print(error) #the directory already exists
            
        #-----------------------------
    
        #Define parameters
        n=3
        Tsite=-28.8 +273.15 #NEEM
        
        rho_surf=315#.3717043 #300
        rho_ice= 910
        rho_ice_softened= rho_ice - 0.2
        
        rho_trench= 550 #denser firn after cutting the trench out. measured
        
        
        phi_snow=0.4
        ab_phi_lim=0.81
        
        K=500 
        
        Aglen=A_glen_temperature(Tsite)
        print('-----------------------------Aglen=',Aglen)
             
        
        day2s=24*60*60
        yr2s=365.25*day2s
        
        acc_rate= 1.55/yr2s #ez dago acc-rik MOVE erabiltzen badugu #txiki bat momentuz    #1.55/yr2s #0.2/yr2s
        g=Constant((0.0,-9.82))
        
        factor=3
        dt=(0.025/factor*yr2s)
        
        #Parameters for mesh creation
        
        """
        #Hauek momentuz horrela utziko ditugu 
        #MOMENTUZ, ZIURTATU FUNTZIONATZEN DUELA
        """
        
        L = 20  # Length 
        H = 30  # height 


        #-----------------------------------------INITIALIZE ARRAYS AND VARIABLES
        
        tstep=0
        zmin=0
        
        #-----------------------------------------DEFINE PARAMETERS FOR THE STEADY STATE 
        
        dHdt=100      #So that it enters the while
        dHdt_tol=0.01 #Change to consider that we have reached the steady state
        
        igifs=0
        filenames_rho = []
        filenames_vel = []
        filenames_mesh = []
        filenames_mesh_final = []
        
        #-------------------------------------------MESH-----------------------------------------------------#
        """
        
        
        >>>>>>>>>>>>>>>>>>>>>>>>CHOOSE MESH FILE TO READ<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        tweak individual parmeters by running the mesh creating files  (e.g.  __just_create....py files)
        
        mesh_ir.xdmf ---------------full circle
        semicircle.xdmf-----------------semicircle. problems at the corners, up and down too close from each other and cut right away
        semicircle_SMOOTH.xdmf -------------truncated at the sides
        
        
        """
        
        mesh = Mesh()
        with XDMFFile(meshfilename) as infile:  #READ FILE<<<<<<<<<<<<<<<<<<<<jarri izena eskuz hor
            infile.read(mesh)
            
            
        ####check mesh 
        plt.figure()
        plot(mesh)
        plt.savefig(output_folder+'mesh_initial.png',format='png',dpi=150)
        plt.close()
        # itxi()
        

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #-------------------------------------------------------------------------------
        
        nyears_ref=7#10 
        
        still_not_shown_ref=True
        
        
        heights=np.array([])
        widths=np.array([])
        volumes=np.array([])
        times=np.array([])
        
        nyears= nyears_ref + 0.01  #5 etik aldatuta
        
        lastyear=-1
        
        while tstep<=(160*factor*(nyears/4)):
                 
            q_degree = 3
            dx = dx(metadata={'quadrature_degree': q_degree})  
            
            #-------------------------------------remesh
            
            if (tstep>0 and tstep%1==0):
                
                print('::::::::::::::REMESHING:::::::::::::::::::::::::::')
                
                #Get coordinates and velocities of the points at the obstacle's surface
                mesh.init(2,1)
                dofs = []
                cell_to_facets = mesh.topology()(2,1)
                for cell in cells(mesh):
                    facets = cell_to_facets(cell.index())
                    for facet in facets:
                        if boundary_subdomains[facet] == 5: #We have given the number 5 to the subdomain of the obstacle
                            dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                            for dof in dofs_:
                                dofs.append(dof)
                
                unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
                hole_coords = V.tabulate_dof_coordinates()[unique_dofs] #surfaceko puntuen koordenatuak
                
            
                #Get coordinates and velocities of the points at th surface
                mesh.init(2,1)
                dofs = []
                cell_to_facets = mesh.topology()(2,1)
                for cell in cells(mesh):
                    facets = cell_to_facets(cell.index())
                    for facet in facets:
                        if boundary_subdomains[facet] == 2: #We have given the number 2 to the subdomain of the surface
                            dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                            for dof in dofs_:
                                dofs.append(dof)
                
                unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
                surface_coords = V.tabulate_dof_coordinates()[unique_dofs] #surfaceko puntuen koordenatuak
            
                
                """we need them sorted for gmsh!!"""
                
                surface_xs, surface_zs,_ = sort_fenics(surface_coords,axis=0)
                hole_xs, hole_zs = sort_hole_fenics_v2(hole_coords,ndmin=4)

                #save hole shape throughout the simulation
                
                if  int(tstep*dt/yr2s) != lastyear : #if not saved yet for this year
                    
                
                    year=int(tstep*dt/yr2s)
                    prefixes='OPTIMALhole_'+Hdldh_prefix+'K'+str(K)+'_'+str(year)+'yr'
                    
                    
                    np.save(output_folder+'hole_xs_'+prefixes+'.npy',hole_xs)
                    np.save(output_folder+'hole_zs_'+prefixes+'.npy',hole_zs)
                    np.save(output_folder+'surface_xs_'+prefixes+'.npy',surface_xs)
                    np.save(output_folder+'surface_zs_'+prefixes+'.npy',surface_zs)
                    
                    
                    np.save(output_folder+'widths.npy',widths)
                    np.save(output_folder+'heights.npy',heights)
                    np.save(output_folder+'volumes.npy',volumes)
                    np.save(output_folder+'times.npy',times)
                    
                    
                    lastyear=year
                
                #create new mesh file
                temp_meshfile='mesh_temp_500'
                remesh(hole_xs,hole_zs,surface_xs,surface_zs,L,tstep, mode='linear',outname=temp_meshfile+'.msh')
                
                #read new mesh file
                mesh = Mesh()
                with XDMFFile(temp_meshfile+'.xdmf') as infile:  #READ FILE<<<<<<<<<<<<<<<<<<<<jarri izena eskuz hor
                    infile.read(mesh)
                    
                    
                #-----compute and save tunnel volume over time------
                
                #compute bulk + hole volume:
                
                vbulk=0
   
                """this is implemented for surface_zs <0 """
                
                for j in range(len(surface_xs)+1):
                      
                    if j==0:  
                        vbulk += (H + surface_zs[j]) *( surface_xs[j] -0+L/2)
                    
                    elif j==(len(surface_xs)):
                        vbulk += (H + surface_zs[j-1]) * (L/2 -surface_xs[j-1])
                        
                    else:
                        vbulk += (H + surface_zs[j])*( surface_xs[j]- surface_xs[j-1])
                
                #-subract the tunnel volume to the whole 'bulk' volume
                
                one = Constant(1)
                Vol = vbulk - assemble(one * dx(domain=mesh))
                              
                
                #-----------compute the height and width at the center and at 3m from ground
                
                xcm=L/2-L/2
                zcm=np.mean(hole_zs)
                
                cdown=(0-L/2,0-H)
                cup=(0-L/2,0-H)
                cleft=(0-L/2,0-H)
                cright=(0-L/2,0-H)
                
                for i in range(len(hole_zs)):
                    
                    current=(hole_xs[i],hole_zs[i])
                    
                    if (np.abs(cdown[0]-(L/2-L/2))>np.abs(current[0]-(L/2-L/2)) and current[1]<zcm): cdown=current
                    if (np.abs(cup[0]-(L/2-L/2))>np.abs(current[0]-(L/2-L/2)) and current[1]>zcm): cup=current
                    
                #now we know cdown[1]
                
                for i in range(len(hole_xs)):
                    
                    current=(hole_xs[i],hole_zs[i])
                    
                    if (np.abs(cleft[1]-(cdown[1]+3))>np.abs(current[1]-(cdown[1]+3)) and current[0]<L/2-L/2): cleft=current
                    if (np.abs(cright[1]-(cdown[1]+3))>np.abs(current[1]-(cdown[1]+3)) and current[0]>L/2-L/2): cright=current
                    
                print('\n\n .........DISTANCES')
                print([cdown[0],cup[0],cleft[0],cright[0]])
                print([cdown[1],cup[1],cleft[1],cright[1]])
                ####check points
                plt.figure()
                plt.scatter(hole_xs,hole_zs,s=2)
                plt.scatter([cdown[0],cup[0],cleft[0],cright[0]],[cdown[1],cup[1],cleft[1],cright[1]],c='r')
                plt.xlim(5-L/2,15-L/2)
                plt.ylim(20-H,30-H)
                plt.savefig(output_folder+'mesh_distances.png',format='png',dpi=150)
                plt.close()

                
                widths=np.append(widths,cright[0]-cleft[0])
                heights=np.append(heights,cup[1]-cdown[1])
                volumes=np.append(volumes,Vol)
                times=np.append(times,tstep*dt/yr2s)

            #--------------------------------------BOUNDARY SUBDOMAINS-------------------------------------------#
            
            #Give a number to each different boundary
            boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
            boundary_subdomains.set_all(0)
            
            
            obstacle=obstacle_boundary()
            obstacle.mark(boundary_subdomains, 5)
            bottom=bottom_boundary()
            bottom.mark(boundary_subdomains, 1)
            top=top_boundary()
            top.mark(boundary_subdomains, 2)
            left=left_boundary()
            left.mark(boundary_subdomains, 3)  #berresleitzen ditu aldeeakoak, nahiz ta hasieran horien zati bat topean egon
            right=right_boundary()
            right.mark(boundary_subdomains, 4)
            
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
            
            
            #--------------------------------------BOUNDARY CONDITIONS--------------------------------------------#
            
            #--------------------------------------INITIAL CONDITION--------------------------------------------#
            
            if tstep==0:
                
                ######_--------------------first of all we need to smooth our initial mesh----------
                
                #-------------------------------------------SMOOTHING OUR OPTIMIZATION PROBLEM SOLUTION-------------------
                #----------------------------------------------(the initial mesh)-----------------------------------------
                
                
                
                print('::::::::::::::SMOOTHING:::::::::::::::::::::::::::')
                
                #Get coordinates and velocities of the points at the obstacle's surface
                mesh.init(2,1)
                dofs = []
                cell_to_facets = mesh.topology()(2,1)
                for cell in cells(mesh):
                    facets = cell_to_facets(cell.index())
                    for facet in facets:
                        if boundary_subdomains[facet] == 5: #We have given the number 5 to the subdomain of the obstacle
                            dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                            for dof in dofs_:
                                dofs.append(dof)
                
                unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
                hole_coords = V.tabulate_dof_coordinates()[unique_dofs] #surfaceko puntuen koordenatuak
                
                
                """but we need them sorted for gmsh!!"""
                
                
                hole_xs0, hole_zs0 = sort_hole_fenics_v2(hole_coords,ndmin=4)
                
                
                # #save the RAW initial hole shape
                
                year=int(tstep*dt/yr2s)
                prefixes='OPTIMALhole_'+Hdldh_prefix+'K'+str(K)+'_'+str(year)+'yr'
                
                np.save(output_folder+'hole_xs_'+prefixes+'RAW.npy',hole_xs0)
                np.save(output_folder+'hole_zs_'+prefixes+'RAW.npy',hole_zs0)

            

                print('--------------------zenbat nodo zuloan---------',len(hole_xs0))
                
                #create new mesh file
                temp_meshfile='mesh_initialSMOOTHED'  #without the .msh (added inside)
                h00,L00,deltax_trench =smooth_initial_mesh(np.array(hole_xs0),np.array(hole_zs0),H,L,Hfloor,h00,L00,deltax_trench,outname=temp_meshfile)
                
                print('-->>>>>>>>>>>>>>>>>>>>>>>>>>hole succesfully smoothed<<<<<<<<<<<')
                
                
                #read new mesh file
                mesh = Mesh()
                with XDMFFile(temp_meshfile+'.xdmf') as infile:  #READ FILE<<<<<<<<<<<<<<<<<<<<jarri izena eskuz hor
                    infile.read(mesh)
                    
                ####check mesh 
                plt.figure()
                plot(mesh,lw=0.5)
                plt.savefig(output_folder+'smoothed_mesh.png',format='png',dpi=150)
                plt.close()
   
                
                #------------------------------------------------restart domains 
     
                """
                it seems like the mesh potentially changes too much for the previous definitions
                they need to be restarted
                """
                
                #--------------------------------------BOUNDARY SUBDOMAINS-------------------------------------------#
                
                #Give a number to each different boundary
                boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
                boundary_subdomains.set_all(0)
                

                obstacle=obstacle_boundary()
                obstacle.mark(boundary_subdomains, 5)
                bottom=bottom_boundary()
                bottom.mark(boundary_subdomains, 1)
                top=top_boundary()
                top.mark(boundary_subdomains, 2)
                left=left_boundary()
                left.mark(boundary_subdomains, 3)  
                right=right_boundary()
                right.mark(boundary_subdomains, 4)
                
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
                

                #--------------------------------------------------------------------------------
                
                #Then, identify the dimensions of the whole automatically
                #not necessary when we create the hole by hand
                #but it will be once adjoint starts to optimize the shape
                
                #Get coordinates and velocities of the points at the obstacle's surface
                mesh.init(2,1)
                dofs = []
                cell_to_facets = mesh.topology()(2,1)
                for cell in cells(mesh):
                    facets = cell_to_facets(cell.index())
                    for facet in facets:
                        if boundary_subdomains[facet] == 5: #We have given the number 5 to the subdomain of the obstacle
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
                
                iorder= np.argsort(boundary_coords[:,0])
                initial_hole_xs = boundary_coords[:,0][iorder] #for plotting it afterwards
                initial_hole_zs = boundary_coords[:,1][iorder] #for plotting it afterwards
                
                

                #----------INITIAL CONDITIONS------to properly order the nodes before setting the values--------------------------
                #-----------------------------------initial density profile--solution of 1d problem---
                #compacted already
                
                rho_init_neem=np.load('rho_NEEM_n3_H0_180_fittedA(T)_K100.npy')
                z_init_neem=np.load('z_NEEM_n3_H0_180_fittedA(T)_K100.npy')
                
                z_init_neem_adjusted = z_init_neem - (np.max(z_init_neem) ) #adjusted to the new mesh height
                
                plt.figure()
                plt.plot(rho_init_neem,z_init_neem)
                plt.plot(rho_init_neem,z_init_neem_adjusted)
                plt.axhline(0,c='k',lw=0.75)
                plt.savefig(output_folder+'initial_densities.png',format='png',dpi=150)
                plt.close() 
                

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
                rho_prev.assign(rho_init) #froga
                
                
                #----------------------------------plotting initial densities
                plt.figure()
                rhoplot=plot(rho_prev)
                clb = plt.colorbar(rhoplot, orientation="vertical",
                                    label='Density (kg/m3)')
                plt.title(Hdldh_prefix + '   (NEEM initial density profile)')
                plt.savefig(output_folder+'initial_hole_and_densities'+Hdldh_prefix+'.png',format='png',dpi=150)
                plt.close()
                # plt.show()
                
                plt.figure()
                rhoplot=plot(rho_prev)
                clb = plt.colorbar(rhoplot, orientation="vertical",
                                    label='Density (kg/m3)')
                # plot(mesh)
                plt.title(Hdldh_prefix + '   (NEEM initial density profile)')
                plt.xlim(5-L/2,15-L/2)
                plt.ylim(20-H,30-H)
                plt.savefig(output_folder+'initial_hole_and_densities_ZOOM'+Hdldh_prefix+'.png',format='png',dpi=150)
                plt.close()
                # plt.show()
                            
                #------------------------------------------------------------------------------------
                
                u_init = Expression(('vx',"uztop - (uztop-uzbot)*pow((H-x[1])/H0,0.35)"),vx=Constant(0.0), H=H, H0=180, uztop=-acc_rate, uzbot=-acc_rate*(rho_surf/rho_ice),  degree=2)
                v_init = interpolate(u_init, V)
                
                v.assign(v_init)
                
                
                plt.figure()
                vplot=plot(v_init*yr2s)
                clb = plt.colorbar(vplot, orientation="vertical",
                                   label='Velocity (m/yr)')
                plt.title('Initial conditions')
                plt.close() 
                

            else:
                
                rho_prev.set_allow_extrapolation(True)       
                v_sol.set_allow_extrapolation(True)
                rho_prev = project(rho_prev,U)
                v_init = interpolate(v_sol,V)
                v.assign(v_init)
                rho.assign(rho_prev)
                

            #-----------------------------------TOP
            #######bc_rho_s=DirichletBC(U,rho_surf,boundary_subdomains,2) #Density at the surface
            #bc_v_s=DirichletBC(V,-acc_rate,surface_boundary) #Velocity at the surface 
            #-----------------------------------BOTTOM
            #bc_rho_b=DirichletBC(U,rho_ice_softened,bottom_boundary) #Density at the bottom
            # bc_v_b=DirichletBC(V,(0.0,-acc_rate*(rho_surf/rho_ice)),boundary_subdomains,2) #Velocity at the bottom
            bc_v_b=DirichletBC(V,(0.0,0.0),boundary_subdomains,1) #Velocity at the bottom
            #-----------------------------------LEFT
            #bc_rho_b=DirichletBC(U,rho_ice_softened,bottom_boundary) #Density at the bottom
            bc_v_l=DirichletBC(V.sub(0),0.0,boundary_subdomains,3) #Velocity at the left boundary
            #-----------------------------------RIGHT
            #bc_rho_b=DirichletBC(U,rho_ice_softened,bottom_boundary) #Density at the bottom
            bc_v_r=DirichletBC(V.sub(0),0.0,boundary_subdomains,4) #Velocity at the right boundary
        
            bcs_rho=[]#bc_rho_s] 
            bcs_v=[bc_v_b,bc_v_l,bc_v_r]
                
            
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
                
                #................................KEEP IT BELOW RHO_ICE...........................#
                
                rhovec = rho_prev.vector()[:]
                rhovec[rhovec > rho_ice_softened] = rho_ice_softened
                rho_prev.vector()[:] = rhovec
            
            rho_prev=project(rho_prev,U)
            
            
            ##############################################################################################
            
            print(tstep,'--------------------------------------------t=',tstep*dt/yr2s,' years')
            
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
            
            #---------------------------------SOLVE MASS BALANCE EQUATION------------------------------------------
            
            alpha_diff=Constant(1e-9) #factor for the diffusive term. To be adjusted. 
            
            a_rho = Constant(1/dt)*rho*wr*dx + 0.5*rho*div(v_sol)*wr*dx + 0.5*dot(v_sol,grad(rho))*wr*dx + alpha_diff*dot(grad(rho),grad(wr))*dx
            L_rho =  Constant(1/dt)*rho_prev * wr *dx - 0.5*rho_prev*div(v_sol)*wr*dx - 0.5*dot(v_sol,grad(rho_prev))*wr*dx - alpha_diff*dot(grad(rho_prev),grad(wr))*dx
            
            F_rho = a_rho - L_rho
        
            tol, relax, maxiter = 1e-2, 0.35, 50
            solparams = {"newton_solver":  {"relaxation_parameter":relax, "relative_tolerance": tol, "maximum_iterations":maxiter} }
            solve(F_rho==0, rho, bcs_rho, solver_parameters=solparams)
            
            rho_prev.assign(rho)  #<-------------UPDATE RHO PROFILE 
            
        
            #-----------------------------------------------------------------------------------------------------#
            #--------------------------------------------EVOLVE MESH----------------------------------------------#
            #-----------------------------------------------------------------------------------------------------#
            
            #Get coordinates and velocities of the points at the obstacle's surface
            mesh.init(2,1)
            dofs = []
            cell_to_facets = mesh.topology()(2,1)
            for cell in cells(mesh):
                facets = cell_to_facets(cell.index())
                for facet in facets:
                    if boundary_subdomains[facet] == 2: #We have given the number 5 to the subdomain of the obstacle
                        dofs_ = V.dofmap().entity_closure_dofs(mesh, 1, [facet])
                        for dof in dofs_:
                            dofs.append(dof)
            
            unique_dofs = np.array(list(set(dofs)), dtype=np.int32)
            boundary_coords = V.tabulate_dof_coordinates()[unique_dofs] #surfaceko puntuen koordenatuak
            zmin=np.min(boundary_coords[:,1])
            print('-----zmin=',zmin)
        
            disp=Function(V)
            disp.assign(v_sol*dt)
            ALE.move(mesh, disp)
            
            ###################################################################################################
            ############################################ PLOT #################################################
            ###################################################################################################
            
            plotYlimMIN= H - 10 -H
            plotYlimMAX = H -H
        
            if tstep % 5 == 0:
                
                
                
                plt.figure()
                rhoplot = plot(rho_prev,cmap='PuBu',vmin=200,vmax=650)
                # rhoplot = plot(rho_prev)
                # plot(mesh)
                clb = plt.colorbar(rhoplot, orientation="vertical",
                                   label='Density (kg/m3)')
                plt.title('Density ('+str("%.2f"%(tstep*dt/yr2s))+'yr)')
        
                # create file name and append it to a list
                filename_rho = f'{output_folder}density{igifs}.png'
                filenames_rho.append(filename_rho)
                
                plt.ylim(plotYlimMIN,plotYlimMAX)
                # plt.xlim(20,30)
        
                # save frame
                plt.savefig(filename_rho, dpi=300, format='png')
                print(filename_rho)
                plt.close()  # build gif
                
                
                #..................
        
                plt.figure()
                vplot = plot(v_sol*yr2s)
                clb = plt.colorbar(vplot, orientation="vertical", label='V (m/yr)')
                plt.title(' V ('+str("%.2f"%(tstep*dt/yr2s))+'yr)')
                # create file name and append it to a list
                filename_vel = f'{output_folder}velocity{igifs}.png'
                filenames_vel.append(filename_vel)
                plt.ylim(plotYlimMIN,plotYlimMAX)
                # plt.xlim(20,30) 
                # save frame
                plt.savefig(filename_vel, dpi=300, format='png')
                plt.close()  # build gif
        
                
                #..................
        
                plt.figure()
                plot(mesh, linewidth=0.25)
                plt.title('mesh ('+str("%.2f"%(tstep*dt/yr2s))+'yr)')
        
                # create file name and append it to a list
                filename_mesh = f'{output_folder}mesh{igifs}.png'
                filenames_mesh.append(filename_mesh)
                plt.ylim(plotYlimMIN,plotYlimMAX)
                # plt.xlim(20,30)
                # save frame
                plt.savefig(filename_mesh, dpi=300, format='png')
                plt.close()  # build gif
                
                
                plt.figure()
                plot(mesh, linewidth=0.25, color="gray")
                plt.title(str("%.2f"%(tstep*dt/yr2s))+'yr')
        
                # create file name and append it to a list
                filename_mesh_f = f'{output_folder}mesh_final_thicker_{igifs}.png'
                filenames_mesh_final.append(filename_mesh_f)
                plt.xlim(-5,5)
                plt.ylim(-10,0)
                # save frame
                plt.savefig(filename_mesh_f, dpi=300, format='png')
                plt.close()  # build gif
        
                igifs += 1
                
            # #------------------------------------------                    
        
            tstep += 1
        
        
        np.save(output_folder+'widths.npy',widths)
        np.save(output_folder+'heights.npy',heights)
        np.save(output_folder+'volumes.npy',volumes)
        np.save(output_folder+'times.npy',times)
        
        
        
        plt.figure()
        plt.scatter(times,volumes,s=2)
        plt.ylabel('Volume (m2)')
        plt.xlabel('time (yr)')
        plt.savefig(output_folder+'times_volumes.png',format='png',dpi=150)
        plt.close()
        
        
        plt.figure()
        plt.plot(times,heights,label='Height')
        plt.plot(times,widths,label='Widths')
        plt.ylabel('Dimension (m)')
        plt.xlabel('time (yr)')
        plt.legend()
        plt.savefig(output_folder+'times_hole_dimensions.png',format='png',dpi=150)
        plt.close()
        
        
        
        print('...........IRTEN DA WHILETIIIIK.........................')
        normal_buffer=1 #makes the gif faster or slower by repeating each fotogram n times
        last_buffer=20
        
        with imageio.get_writer(output_folder+Hdldh_prefix+'_k'+str(K)+'_densityEVOLUTION.gif', mode='I') as writer:
            for filename in filenames_rho:
                
                if filename==filenames_rho[-1]:
                    buffer_reps=last_buffer
                else:
                    buffer_reps=normal_buffer
                
                for i in range(buffer_reps):
                    image = imageio.imread(filename)
                    writer.append_data(image)
        
        # # Remove files
        # for filename in set(filenames_rho):
        #     os.remove(filename)
        
        with imageio.get_writer(output_folder+Hdldh_prefix+'_k'+str(K)+'_velocityEVOLUTION.gif', mode='I') as writer:
            for filename in filenames_vel:
                
                if filename==filenames_rho[-1]:
                    buffer_reps=last_buffer
                else:
                    buffer_reps=normal_buffer
                
                for i in range(buffer_reps):
                    image = imageio.imread(filename)
                    writer.append_data(image)
        
        # # Remove files
        # for filename in set(filenames_vel):
        #     os.remove(filename)
        
        with imageio.get_writer(output_folder+Hdldh_prefix+'_k'+str(K)+'_meshEVOLUTION.gif', mode='I') as writer:
            for filename in filenames_mesh:
                
                if filename==filenames_rho[-1]:
                    buffer_reps=last_buffer
                else:
                    buffer_reps=normal_buffer
                
                for i in range(buffer_reps):
                    image = imageio.imread(filename)
                    writer.append_data(image)
        
        # # Remove files
        # for filename in set(filenames_mesh):
        #     os.remove(filename)
        
        
        with imageio.get_writer(output_folder+Hdldh_prefix+'_k'+str(K)+'_finalmeshinitEVOLUTION.gif', mode='I') as writer:
            for filename in filenames_mesh_final:
                
                if filename==filenames_rho[-1]:
                    buffer_reps=last_buffer
                else:
                    buffer_reps=normal_buffer
                
                for i in range(buffer_reps):
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    
        # # Remove files
        # for filename in set(filenames_mesh):
        #     os.remove(filename)
        
        
        
        #--------------------------------------consider the case solved
        
        already_solved.append(all_xdmf[jj])
        
        #save array after each iteration
        with open(solved_arr_filename, 'w') as f:
            f.write("\n".join(already_solved))
            
        print('\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SUCCESS<<<<<<<<<<<<<<<<<<<<<<\n\n\n ')
        print('------------------------------------------------------added to the SOLVED list---')
        print(*already_solved,sep='\n')
        print('\n                                           ',
              len(all_xdmf) -(len(already_solved) + len(problems)), ' left')
        

    except Exception as error_statement:
        
        problems.append(all_xdmf[jj])
        
        with open(ignored_arr_filename, 'w') as f:
            f.write("\n".join(problems))
        
        print('\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ERROOOOR<<<<<<<<<<<<<<<<<<<<<<\n\n\n ')
        
        print(error_statement)
        
        print('\n\n',traceback.format_exc() )
        print('\n\n\n------------------------------------------------------added to the ignored list---')
        print(*problems,sep='\n')
        print('\n                                           ',
              len(all_xdmf) -(len(already_solved) + len(problems)), ' left')
        
    
    
        
        

