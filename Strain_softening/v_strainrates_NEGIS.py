#Import necessary modules
import math
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray
from rasterio.warp import transform
from tqdm.notebook import tqdm

#Plotting function (edit ticks)
plt.rcParams['figure.figsize'] = (8, 8)

from matplotlib.ticker import FuncFormatter
def km_ticks():
    formatter = FuncFormatter(lambda x,pos: f'{x/1000:.0f}')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xlabel('Easting (km)')
    plt.ylabel('Northing (km)')
    return


# # What data we have 
# 
# * Horizontal ice velocities (vx and vy) from [https://nsidc.org/data/measures] - It is a multi year average.
# * The 1958-2019 annual average temperature from the RACMO regional climate model. 
# * A black and white 100m MODIS mosaic background map from MEaSUREs (aug2016).
# 
# Browse [additional QGreenland datasets here](https://sid.erda.dk/cgi-sid/ls.py?share_id=BSOsjlIvQi;current_dir=QGreenland_v1.0.1;flags=f). 


#Load data
vx_file = 'https://sid.erda.dk/share_redirect/HUyCiE8JLN/MEaSUREs/M%20Multi-year%20IV%20mosaic%20v1/greenland_vel_mosaic250_vx_v1.tif'
#vx_file = 'https://sid.erda.dk/share_redirect/HUyCiE8JLN/ITS_LIVE/GRE_G0120_0000_vx.tif'
vy_file = vx_file.replace('_vx_','_vy_')
modismap_file = 'https://sid.erda.dk/share_redirect/BSOsjlIvQi/QGreenland_v1.0.1/mog100_201608_hp1_v01.tif'
temperature_file = 'https://sid.erda.dk/share_redirect/BSOsjlIvQi/QGreenland_v1.0.1/Regional%20climate%20models/RACMO%20model%20output/Annual%20mean%20temperature%20at%202m%201958-2019%20%281km%29/racmo_t2m.tif'


# ### Define a convenience function for reading parts of a map
# I give you a convenience function that simplifies loading small subsets of large geo-tiffs (it is a thin wrapper for some xarray functionality). 
# 
# You call it like this: 
# ```python
#    data = geoimread(filename, roi_x = lon, roi_y = lat, roi_crs = 'LL', buffer=1000)
# ``` 
# Here ROI is Region Of Interest. It can be a polygon or just a single point. ROI_CRS is code specifying the coordinte reference system. "LL" is a shorthand for a WGS84 lat lon pair. This is necessary because the data file may not be using a lat-long grid but a projected coordinate system like _polar stereographic_ with a northing and easting in metres. Buffer is how large a region around the ROI should be loaded (units:projected coords - i.e. metres).  
# 

#this function is copied from https://github.com/grinsted/pyimgraft/blob/master/geoimread.py
def geoimread(fname, roi_x=None, roi_y=None, roi_crs=None, buffer=0, band=0):
    """Reads a sub-region of a geotiff/geojp2/ that overlaps a region of interest (roi)
    This is a simple wrapper of xarray functionality.
    Parameters:
    fname (str) : file.
    roi_x, roi_y (List[float]) : region of interest.
    roi_crs (dict) : ROI coordinate reference system, in rasterio dict format (if different from scene coordinates)
    buffer (float) : adds a buffer around the region of interest.
    band (int) : which bands to read.
    Returns
        xr.core.dataarray.DataArray : The cropped scene.
    """
    da = rioxarray.open_rasterio(fname, parse_coordinates =True, mask_and_scale =True)
    if roi_x is not None:
        if not hasattr(roi_x, "__len__"):
            roi_x = [roi_x]
            roi_y = [roi_y]
        if roi_crs is not None:
            if str(roi_crs) == "LL":
                roi_crs = {"init": "EPSG:4326"}
            # remove nans
            roi_x = np.array(roi_x)
            roi_y = np.array(roi_y)
            ix = ~np.isnan(roi_x + roi_y)
            roi_x, roi_y = transform(src_crs=roi_crs, dst_crs=da.rio.crs, xs=roi_x[ix], ys=roi_y[ix])
        rows = (da.y > np.min(roi_y) - buffer) & (da.y < np.max(roi_y) + buffer)
        cols = (da.x > np.min(roi_x) - buffer) & (da.x < np.max(roi_x) + buffer)
        # update transform property
        da = da[band, rows, cols].squeeze()
        da.attrs["transform"] = (da.x.values[1] - da.x.values[0], 0.0, da.x.values[0], 0.0, da.y.values[1] - da.y.values[0], da.y.values[0])
        return da
    else:
        return da[band, :, :].squeeze()

#Function that identifies the index of the value that is closest to the one of interest
def find_nearest(array, value):
    #array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

lat,lon =75.7, -36
buffer =18000 #how large a region do you want to load

#Load background    
BG = geoimread(modismap_file, roi_x = lon, roi_y= lat, roi_crs = 'LL', buffer=buffer);     #Comment rest of the line to get whole area

#Plot velocities
vx = geoimread(vx_file, roi_x = lon, roi_y= lat, roi_crs = 'LL', buffer=buffer);
vy = geoimread(vy_file, roi_x = lon, roi_y= lat, roi_crs = 'LL', buffer=buffer);

v= np.sqrt(vx**2+vy**2)

# vx.data[(v<20)|(v>1e8)] = np.nan     #Comment these lines to avoid having nans
# v.data[(v<20)|(v>1e8)] = np.nan

v.plot.imshow(alpha=1);
plt.title('vel (m/yr)');
km_ticks()


#Save the original velocity data and their x and y values (as northing as easting) as numpy arrays
vx_np=np.array(vx)
vy_np=np.array(vy)

v_xs_np=np.array(vx.x)
v_ys_np=np.array(vx.y)

#Load the x and y coordinates of the points where we have density data
rho_xs=np.load('../3. NEGIS density, accumulation, and strain/NEGISdensity_easting.npy')
rho_ys=np.load('../3. NEGIS density, accumulation, and strain/NEGISdensity_northing.npy')

#Identify the minimum and maximum values of the coordinates and save the velocity data around that area
xmin=np.min(rho_xs)
xmax=np.max(rho_xs)

ymin=np.min(rho_ys)
ymax=np.max(rho_ys)

idx_xmin=find_nearest(v_xs_np,xmin)
idx_xmax=find_nearest(v_xs_np,xmax)
idx_ymin=find_nearest(v_ys_np,ymin)
idx_ymax=find_nearest(v_ys_np,ymax)

print(idx_xmin,idx_xmax,idx_ymin,idx_ymax)

x_NEGIS=v_xs_np[idx_xmin-5:idx_xmax+5] #We have taken some more grid points so that we can calculate the gradients without problems
y_NEGIS=v_ys_np[idx_ymax-5:idx_ymin+5]
vx_NEGIS=vx_np[idx_ymax-5:idx_ymin+5,idx_xmin-5:idx_xmax+5]
vy_NEGIS=vy_np[idx_ymax-5:idx_ymin+5,idx_xmin-5:idx_xmax+5]

#Plot vx and vy and the module
plt.figure()
extent = [np.min(x_NEGIS) , np.max(x_NEGIS), np.min(y_NEGIS) , np.max(y_NEGIS)]
plt.imshow(vx_NEGIS,extent=extent)
plt.scatter(rho_xs,rho_ys)
plt.colorbar()
plt.clim(0,30) 
plt.title('vx (m/yr)')
plt.show()

plt.figure()
extent = [np.min(x_NEGIS) , np.max(x_NEGIS), np.min(y_NEGIS) , np.max(y_NEGIS)]
plt.imshow(vy_NEGIS,extent=extent)
plt.scatter(rho_xs,rho_ys)
plt.colorbar()
plt.clim(0,60) 
plt.title('vy (m/yr)')
plt.show()

plt.figure()
extent = [np.min(x_NEGIS) , np.max(x_NEGIS), np.min(y_NEGIS) , np.max(y_NEGIS)]
plt.imshow(np.sqrt(vx_NEGIS**2+vy_NEGIS**2),extent=extent)
plt.scatter(rho_xs,rho_ys)
plt.colorbar()
plt.clim(0,60) 
plt.title('v (m/yr)')
plt.show()

#The angle of the line where the geophones are located with respect to the horizontal 
alpha=-0.5890210812981336 #rad

#dx and dy
dx=abs(np.roll(x_NEGIS,-1)-x_NEGIS)[0]
dy=abs(np.roll(y_NEGIS,-1)-y_NEGIS)[0]

#Identify the points where we have the density measurements 
idx_x=np.zeros(len(rho_xs))
idx_y=np.zeros(len(rho_ys))

for i in range(len(rho_xs)):
    
    idx_x[i]=find_nearest(x_NEGIS,rho_xs[i])
    idx_y[i]=find_nearest(y_NEGIS,rho_ys[i])
    
idx_x=idx_x.astype(int)
idx_y=idx_y.astype(int)

#Identify the points in front and behind (in the direction of the flow) of those where we have the density measurements
lx=300*abs(np.sin(alpha))
ly=300*abs(np.cos(alpha))

idx_x1=np.zeros(len(rho_xs))
idx_y1=np.zeros(len(rho_ys))

idx_x2=np.zeros(len(rho_xs))
idx_y2=np.zeros(len(rho_ys))

for i in range(len(rho_xs)):
    
    idx_x1[i]=find_nearest(x_NEGIS,rho_xs[i]-lx)
    idx_y1[i]=find_nearest(y_NEGIS,rho_ys[i]-ly)
    
    idx_x2[i]=find_nearest(x_NEGIS,rho_xs[i]+lx)
    idx_y2[i]=find_nearest(y_NEGIS,rho_ys[i]+ly)
    
    
idx_x1=idx_x1.astype(int)
idx_y1=idx_y1.astype(int)

idx_x2=idx_x2.astype(int)
idx_y2=idx_y2.astype(int)

#Rotate velocity fields
vx_rot=np.zeros((3,len(idx_x)))
vy_rot=np.zeros((3,len(idx_x)))

vx_rot[0,:]=np.cos(alpha)*vx_NEGIS[idx_y1,idx_x1]+np.sin(alpha)*vy_NEGIS[idx_y1,idx_x1]
vy_rot[0,:]=-np.sin(alpha)*vx_NEGIS[idx_y1,idx_x1]+np.cos(alpha)*vy_NEGIS[idx_y1,idx_x1]

vx_rot[1,:]=np.cos(alpha)*vx_NEGIS[idx_y,idx_x]+np.sin(alpha)*vy_NEGIS[idx_y,idx_x]
vy_rot[1,:]=-np.sin(alpha)*vx_NEGIS[idx_y,idx_x]+np.cos(alpha)*vy_NEGIS[idx_y,idx_x]

vx_rot[2,:]=np.cos(alpha)*vx_NEGIS[idx_y2,idx_x2]+np.sin(alpha)*vy_NEGIS[idx_y2,idx_x2]
vy_rot[2,:]=-np.sin(alpha)*vx_NEGIS[idx_y2,idx_x2]+np.cos(alpha)*vy_NEGIS[idx_y2,idx_x2]

#Calculate spatial derivatives of the velocity field
dvxdx_rot=(np.roll(vx_rot,-1,axis=1)-np.roll(vx_rot,1,axis=1))/(2*dx)
dvxdy_rot=(np.roll(vx_rot,1,axis=0)-np.roll(vx_rot,-1,axis=0))/(2*ly)
dvydx_rot=(np.roll(vy_rot,-1,axis=1)-np.roll(vy_rot,1,axis=1))/(2*dx)
dvydy_rot=(np.roll(vy_rot,1,axis=0)-np.roll(vy_rot,-1,axis=0))/(2*ly)

#Compute the strain rates from the derivatives 
epsyy=dvydy_rot[1,1:-2]
epsxy=0.5*(dvydx_rot[1,1:-2]+dvxdy_rot[1,1:-2])

#Save strain rates
np.save('NEGISepsyy.npy',epsyy)
np.save('NEGISepsxy.npy',epsxy)

#Plot strain rate
plt.figure(figsize=(12,5))
plt.plot(dvxdx_rot[1,1:-2],label='epsxx')
plt.plot(dvydy_rot[1,1:-2],label='epsyy')
plt.plot(0.5*(dvydx_rot[1,1:-2]+dvxdy_rot[1,1:-2]),label='epsxy')
plt.ylabel('Strain rates (1/yr)')
plt.legend()
plt.show()

#Plot map
plt.figure(figsize=(12,8))
extent = [np.min(x_NEGIS) , np.max(x_NEGIS), np.min(y_NEGIS) , np.max(y_NEGIS)]
plt.imshow(np.sqrt(vx_NEGIS**2+vy_NEGIS**2),extent=extent)
plt.scatter(rho_xs,rho_ys,s=5,color='tab:orange',label='NEGIS data points')
plt.scatter(rho_xs+lx,rho_ys+ly,s=5,color='tab:grey',label='Points used to calculate gradient')
plt.scatter(rho_xs-lx,rho_ys-ly,s=5,color='tab:grey')
cbar=plt.colorbar()
plt.clim(0,60) 
cbar.set_label('Velocity (m/yr)')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.legend()
plt.show()
