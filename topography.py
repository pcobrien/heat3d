import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt 
from PIL import Image
from tqdm import tqdm 
import sys 
from scipy.interpolate import griddata
import os 
import funcs 

site_coords = {
    "Haworth Crater": [ [-5.17, -87.45], 25.7],
    "Shoemaker Crater": [ [45.91, -88.14], 25.9],
    "Faustini Crater": [ [84.31, -87.18], 39.0/2.0*1.5],
    "Shackleton Crater": [ [129.78, -89.67], 15.0],
    "Cabeus Crater": [ [-42.13, -85.33], 50.3],
    "Amundsen Crater": [ [83.07, -84.44], 51.7],
    "VIPER": [ [31.6218, -85.42088], 5.0],
    "Chandrayaan 3": [32.32, -69.3741],
    "IM-1": [ [1.0, -80.2], 5.0],
}
    

def calc_vec2pole(planet, lam, phi):
    """
    Calculate vector pointing from center of mesh to pole 

    planet: Planet object 
    lam: center longitude of mesh [rad]
    phi: center latitude of mesh [rad]
    """

    lam0 = 0.0
    if phi > 0.0:
        phi1 = np.pi/2.0
    else:
        phi1 = -np.pi/2.0

    R = planet.R
    k = (2.0 * R) / (
        1.0 + np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(lam - lam0)
    )

    x = k * np.cos(phi) * np.sin(lam - lam0)
    y = k * (np.cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(lam - lam0))

    vec2pole = [-x, -y]
    vec2pole /= np.linalg.norm(vec2pole)

    return vec2pole


def checkNormals(mesh):
    # Function to ensure that normal vectors are pointed the correct way
    normals = mesh.cell_normals
    mean_normal_z = np.mean(normals[:,2])

    if mean_normal_z < 0.0:
        mesh.flip_normals()

    return mesh


def findMeshCenter(mesh):
    # Return point on the mesh at which to reference the solar disc position (facet closest to the center of the mesh)
    center = mesh.points[mesh.find_closest_point(mesh.center)] 

    return center


def plot_mesh(mesh, mesh_center, vec2pole):
    # Basic mesh visualization
    pole = pv.Arrow([mesh_center[0], mesh_center[1], mesh_center[2] + 10000.0], [vec2pole[0], vec2pole[1], 0.0], scale=10000.0)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh)
    plotter.add_mesh(pole, color="black")
    plotter.view_xy()
    plotter.show()


def extract_subregion(coords, resolution, planet):
    # Load DEM
    if coords[0][1] < 0.0:
        pole = "S"
    else:
        pole = "N"

    dem_min_lat = {
        "5":        ["87", ".tif"],
        "20":       ["80", ".IMG"],
        "30":       ["75", ".IMG"],
        "60":       ["75", ".IMG"],
        "120" :     ["75", ".IMG"],
        "240" :     ["75", ".IMG"],
        "400" :     ["45", ".IMG"],
    }

    if planet.name == "Moon":
        dem_prefix = "LDEM"

    min_lat = dem_min_lat[str(int(resolution))][0]
    dem_extension = dem_min_lat[str(int(resolution))][1]

    fname_dem = "Topo/" + dem_prefix + "_" + min_lat + pole + "_" + str(int(resolution)) + "M" + dem_extension

    if dem_extension == ".IMG":
        topo = funcs.readIMG(fname_dem, "dtm")
    elif dem_extension == ".tif":
        topo = funcs.readTIFF(fname_dem)

    dem_size = topo.shape[0]

    center_row, center_col = funcs.lonlat_to_rowcol(planet, coords[0], pole, resolution, dem_size)

    radius = round(coords[1]*1e3/resolution)
    min_row = round(center_row - radius)
    max_row = round(center_row + radius)
    min_col = round(center_col - radius)
    max_col = round(center_col + radius)

    vec2pole = [round(dem_size/2) - center_col, round(dem_size/2) - center_row]
    vec2pole /= np.linalg.norm(vec2pole)
    
    # Lon/lat of region center
    center_lon, center_lat = funcs.rowcol_to_lonlat(planet, center_row, center_col, pole, resolution, dem_size)

    # Extract sub-region from DEM
    region = topo[min_row:max_row, min_col:max_col]

    return region, center_lon, center_lat, vec2pole


def create_mesh(topo, resolution):

    n = topo.shape[0]
    x = np.arange(n)*resolution
    y = np.arange(n)*resolution

    points = np.zeros((round(n**2), 3))
    ctr = 0
    
    for row in tqdm(range(n), leave=False):
        for col in range(n):
            points[ctr,:] = [x[col], y[row], topo[row,col]]
            ctr += 1

    cloud = pv.PolyData(points)
    mesh = cloud.delaunay_2d(progress_bar=True)

    return mesh


def createMeshFromPointCloud(fname, fname_mesh, planet, center_lon_rad, center_lat_rad, resolution, uniform=False):
    # Create mesh from [x,y,z] point cloud data 

    # Load point cloud data
    arr = np.fromfile(fname, dtype=np.double)
    arr = arr.reshape((round(arr.shape[0]/4), 4))
    points = arr[:, 0:3]*1000.0     # Make sure coordinates in m!

    if uniform == True:
        # Resample onto uniform grid (if desired)
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]

        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)

        xi = np.arange(min_x, max_x, resolution)
        yi = np.arange(min_y, max_y, resolution)
        X, Y = np.meshgrid(xi, yi)

        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='nearest')

        x = X.flatten()
        y = Y.flatten()
        z = zi.flatten()

        if np.max(z) < planet.R - 10000.0:
            z += planet.R

        points = np.zeros((len(x), 3))
        points[:,0] = x 
        points[:,1] = y
        points[:,2] = z

    # Ensure z-coordinates are in planetary radii, not referenced relative to unit sphere
    if np.max(points[:,2]) < planet.R - 10000.0:
        points[:,2] += planet.R

    cloud = pv.PolyData(points)

    mesh = cloud.delaunay_2d()
    mesh = checkNormals(mesh)

    mesh_center = findMeshCenter(mesh)
    vec2pole = calc_vec2pole(planet, center_lon_rad, center_lat_rad)

    mesh.save(fname_mesh, recompute_normals=False)
    
    return mesh, center_lon_rad, center_lat_rad, mesh_center, vec2pole


def createMeshFromTIF(fname, fname_mesh, planet, center_lon_rad, center_lat_rad, resolution):
    # Create mesh from TIF file
    topo = funcs.readTIFF(fname)

    # Ensure z-coordinates are in planetary radii, not referenced relative to unit sphere
    if np.max(topo) < planet.R - 10000.0:
        topo += planet.R

    mesh = create_mesh(topo, resolution)

    mesh = checkNormals(mesh)

    mesh_center = findMeshCenter(mesh)
    vec2pole = calc_vec2pole(planet, center_lon_rad, center_lat_rad)

    mesh.save(fname_mesh, recompute_normals=False)

    return mesh, center_lon_rad, center_lat_rad, mesh_center, vec2pole


def createMeshfromDEM(site=None, resolution=None, center_lon_deg=None, center_lat_deg=None, radius_km=None, planet=None):

    if site:
        # Site from list
        coords = site_coords[site]

    elif center_lon_rad:
        # User-specified site
        coords = [ [center_lon_deg, center_lat_deg], 25.7],

    region, center_lon_rad, center_lat_rad, vec2pole = extract_subregion(coords, resolution, planet)

    mesh = create_mesh(region, resolution)

    mesh = checkNormals(mesh)

    mesh_center = findMeshCenter(mesh)

    fname_mesh = "Mesh/" + "mesh_" + '_'.join(site.split(' ')) + "_{}M".format(int(resolution)) + ".ply"

    mesh.save(fname_mesh, recompute_normals=False)

    return mesh, center_lon_rad, center_lat_rad, mesh_center, vec2pole


def load_mesh(fname_mesh = None, center_lon_deg = None, center_lat_deg = None, site = None, planet = None):
    # Load existing mesh file

    if site:
        coords = site_coords[site]
        center_lon_rad = np.deg2rad(coords[0][0])
        center_lat_rad = np.deg2rad(coords[0][1])
    
    elif center_lon_deg:
        center_lon_rad = np.deg2rad(center_lon_deg)
        center_lat_rad = np.deg2rad(center_lat_deg)

    mesh = pv.read(fname_mesh)

    mesh_center = findMeshCenter(mesh)
    vec2pole = calc_vec2pole(planet, center_lon_rad, center_lat_rad)

    mesh = checkNormals(mesh)

    return mesh, center_lon_rad, center_lat_rad, mesh_center, vec2pole