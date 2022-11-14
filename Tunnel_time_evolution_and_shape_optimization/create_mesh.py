
import numpy as np
import matplotlib.pyplot as plt

inflow_marker = 1
outflow_marker = 2
bottom_marker = 3
top_marker = 4
obstacle_marker = 5


L = 20  # Length of channel
H = 30  # Width of channel

#parameters of the hole
Lx=5 #width
h=5  #max height from floor to ceiling
Hfloor=8 #depth at which the floor is located from the surface

dh= Hfloor - h #distance to the height
r = Lx/2  # Radius of object
Lh= h - r

#center
c_x, c_y = 0, 0 -(dh + r)  # Position of the center of the upper round ceiling
x0,y0= c_x - Lx/2, 0 -(dh + h)

print('center point = x',c_x,'    y=',c_y)


def create_stokes_mesh(res):
    """
    Creates a mesh containing a circular obstacle.
    Inlet marker (left): 1
    Oulet marker (right): 2
    Wall marker (top/bottom): 3
    Obstacle marker: 4
    """
    try:
        import gmsh
        import meshio

    except ImportError:
        print("meshio and/or gmsh not installed. Requires the non-python libraries:\n",
              "- libglu1\n - libxcursor-dev\n - libxinerama1\n And Python libraries:\n"
              " - h5py",
              " (pip3 install --no-cache-dir --no-binary=h5py h5py)\n",
              "- gmsh \n - meshio")
        exit(1)

    gmsh.initialize()

    # Create geometry
    c = gmsh.model.occ.addPoint(c_x, c_y, 0)
    
    p1 = gmsh.model.occ.addPoint(x0, y0, 0)
    p2 = gmsh.model.occ.addPoint(x0 , y0+ Lh, 0)
    ptop = gmsh.model.occ.addPoint(x0 + Lx/2 , c_y+r, 0)
    p3 = gmsh.model.occ.addPoint(x0 + Lx, y0 + Lh, 0)
    p4 = gmsh.model.occ.addPoint(x0+Lx,y0, 0)
    
    arc_1 = gmsh.model.occ.addLine(p1,p2)
    arc_21 = gmsh.model.occ.addEllipseArc(p2, c, ptop,ptop)
    arc_22 = gmsh.model.occ.addEllipseArc(ptop, c, p3,p3)
    arc_3 = gmsh.model.occ.addLine(p3,p4)
    arc_4 = gmsh.model.occ.addLine(p4,p1)
    obstacle_loop = gmsh.model.occ.addCurveLoop([arc_1, arc_21, arc_22, arc_3, arc_4])
    rectangle = gmsh.model.occ.addRectangle(-L/2,-H, 0, L, H)
    gmsh.model.occ.synchronize()
    obstacle_loop = gmsh.model.occ.addCurveLoop([arc_1, arc_21, arc_22, arc_3, arc_4])

    exterior_boundary = gmsh.model.getBoundary([(2, rectangle)])
    obstacle = gmsh.model.occ.addPlaneSurface([obstacle_loop])

    fluid = gmsh.model.occ.cut([(2, rectangle)], [(2, obstacle)])
    gmsh.model.occ.synchronize()

    # Create physical markers
    lines = gmsh.model.occ.getEntities(dim=1)
    walls = []
    obstacles = []
    for line in lines:
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.allclose(com, [0-L/2, H / 2- H, 0]):
            gmsh.model.addPhysicalGroup(line[0], [line[1]], inflow_marker)
            gmsh.model.setPhysicalName(line[0], inflow_marker, "Fluid inlet")
        elif np.allclose(com, [L-L/2, H / 2- H, 0]):
            gmsh.model.addPhysicalGroup(line[0], [line[1]], outflow_marker)
            gmsh.model.setPhysicalName(line[0], outflow_marker, "Fluid outlet")
        elif np.allclose(com, [L / 2-L/2, 0- H, 0]):
            gmsh.model.addPhysicalGroup(line[0], [line[1]], bottom_marker)
            gmsh.model.setPhysicalName(line[0], bottom_marker, "Bottom")
        elif np.allclose(com, [L / 2-L/2, H- H, 0]):   
            gmsh.model.addPhysicalGroup(line[0], [line[1]], top_marker)
            gmsh.model.setPhysicalName(line[0], top_marker, "Surface")
        else:
            obstacles.append(line[1])
            
    gmsh.model.addPhysicalGroup(1, obstacles, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

    # Specify mesh resolution
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", obstacles)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", res)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 4 * res)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5 * r)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * r)
    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)

    gmsh.model.mesh.generate(2)

    gmsh.model.addPhysicalGroup(fluid[0][0][0], [fluid[0][0][1]], 12)
    gmsh.write("mesh.msh")

    # Read in and convert mesh to msh
    msh = meshio.read("mesh.msh")

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
    meshio.write("mesh.xdmf", triangle_mesh)
    meshio.write("mf.xdmf", line_mesh)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", default=0.5, type=np.float64, dest="res", #<<<<<<<<<<<<<<<<<<<<<<<< default horrek kontrolatzen du erresoluzioa, 0.25 default
                        help="Resolution near circular obstacle")
    args = parser.parse_args()
    create_stokes_mesh(args.res)
