import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm
import sys 
from scipy import interpolate 

Image.MAX_IMAGE_PIXELS = None
TWO_PI = 2.0*np.pi 

# ---------------------------------------------------------------------------- #
#                                 Loading files                                #
# ---------------------------------------------------------------------------- #

def readIMG(input_filename, tp="dtm"):
    """
    Function for loading LOLA .IMG files from PDS
    
    Inputs:
        input_filename: Function assumes that input file exists
        tp: type of file being loaded. Determines offsets and scaling factors to be applied to input array
    
    Outputs:
        arr: Numpy array
    """
    if tp == "dtm":
        scale = 0.5
        offset = 1737400.0
        dtype= "int16"      # big-endian unsigned integer (16bit)

    elif tp == "psr":
        scale = 0.000025
        offset = 0.5
        dtype= "int16"      # big-endian unsigned integer (16bit)

    elif tp == "alb":
        scale = 1.0
        offset = 0.0
        dtype = "float32"   # signed float (32bit)

    dtype = np.dtype(dtype)  

    fid = open(input_filename, "rb")
    data = np.fromfile(fid, dtype)
    grid_size = int(np.sqrt(data.shape[0]))
    shape = (grid_size, grid_size)  # matrix size

    arr = np.flipud(data.reshape(shape)) * scale + offset

    return arr


def readTIFF(input_filename):
    """
    Function for loading LOLA .tiff files
    
    Inputs:
        input_filename: Function assumes that input file exists
    
    Outputs:
        arr: Numpy array
    """
    
    im = Image.open(input_filename)
    
    return np.flipud(np.array(im))


def readIMG_cylindrical(input_filename, tp="dtm"):
    """
    Function for loading LOLA .IMG files from PDS
    
    Inputs:
        input_filename: Function assumes that input file exists
        tp: type of file being loaded. Determines offsets and scaling factors to be applied to input array
    
    Outputs:
        arr: Numpy array
    """
    scale = 1.0
    offset = 0.0
    dtype = "float32"   # signed float (32bit)

    dtype = np.dtype(dtype)  

    fid = open(input_filename, "rb")
    data = np.fromfile(fid, dtype)
    shape = (1800, 3600)  # matrix size

    arr = np.flipud(data.reshape(shape)) * scale + offset
    arr = np.roll(arr, -round(arr.shape[1]/2), axis=1)

    return arr


# ---------------------------------------------------------------------------- #
#                                Map projections                               #
# ---------------------------------------------------------------------------- #

def rowcol_to_lonlat(planet, row, col, pole, resolution, grid_size):
    """
    Parameters
    ----------
    row : [pxl]
    col : [pxl]
    pole : S or N
    resolution : resolution of DEM [m/pxl]
    grid_size : full width of square DEM [pxl]

    Returns
    -------
    lon, lat of given point [deg]
    """

    R = planet.R 

    lam0 = 0.0 
    if pole == "N":
        phi1 = np.pi/2.0
    elif pole == "S":
        phi1 = -np.pi/2.0

    x_s = (col - grid_size / 2) * resolution
    y_s = (row - grid_size / 2) * resolution

    rho = np.sqrt(x_s ** 2 + y_s ** 2)
    c = 2.0 * np.arctan2(rho, 2.0 * R)

    if rho > 0.0:
        lat0 = np.arcsin(np.cos(c) * np.sin(phi1) + (y_s * np.sin(c) * np.cos(phi1)) / rho)
        lon0 = lam0 + np.arctan2(x_s * np.sin(c), (rho * np.cos(phi1) * np.cos(c) - y_s * np.sin(phi1) * np.sin(c)))
    else:
        lat0 = phi1
        lon0 = lam0

    return lon0, lat0


def lonlat_to_xy(coords, pole):
    """
    Parameters
    ----------
    coords : lon/lat pair [deg]
    pole : S or N
    resolution : resolution of DEM [m/pxl]
    grid_size : full width of square DEM [pxl]

    Returns
    -------
        row, col of given point [pxl]
    """

    coords = np.deg2rad(coords)
    lam = coords[0]
    phi = coords[1]

    R = 1737.4e3

    lam0 = 0.0
    if pole == "N":
        phi1 = np.pi/2.0
    elif pole == "S":
        phi1 = -np.pi/2.0

    k = (2.0 * R) / (
        1.0 + np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(lam - lam0)
    )

    x = k * np.cos(phi) * np.sin(lam - lam0)
    y = k * (np.cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(lam - lam0))

    return x,y


def lonlat_to_rowcol(planet, coords, pole, resolution, grid_size):
    """
    Parameters
    ----------
    coords : lon/lat pair [deg]
    pole : S or N
    resolution : resolution of DEM [m/pxl]
    grid_size : full width of square DEM [pxl]

    Returns
    -------
        row, col of given point [pxl]
    """

    R = planet.R 

    coords = np.deg2rad(coords)
    lam = coords[0]
    phi = coords[1]

    lam0 = 0.0
    if pole == "N":
        phi1 = np.pi/2.0
    elif pole == "S":
        phi1 = -np.pi/2.0

    k = (2.0 * R) / (
        1.0 + np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(lam - lam0)
    )

    x = k * np.cos(phi) * np.sin(lam - lam0)
    y = k * (np.cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(lam - lam0))

    x = x / resolution + grid_size / 2
    y = y / resolution + grid_size / 2

    col = round(x)
    row = round(y)

    return row, col


def lonlat_to_rowcol_polar(coords, pole, resolution, grid_size):
    """
    Parameters
    ----------
    coords : lon/lat pair [deg]
    pole : S or N
    resolution : resolution of DEM [m/pxl]
    grid_size : full width of square DEM [pxl]

    Returns
    -------
        row, col of given point [pxl]
    """

    lam = coords[0]
    phi = coords[1]

    R = 1737.4e3

    lam0 = 0.0
    if pole == "N":
        phi1 = np.pi/2.0
    elif pole == "S":
        phi1 = -np.pi/2.0

    k = (2.0 * R) / (
        1.0 + np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(lam - lam0)
    )

    x = k * np.cos(phi) * np.sin(lam - lam0)
    y = k * (np.
             cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(lam - lam0))

    x = x / resolution + grid_size / 2
    y = y / resolution + grid_size / 2

    col = round(x)
    row = round(y)

    return row, col


def lonlat_to_rowcol_cylindrical(coords, grid_size_x, grid_size_y):
    """
    Parameters
    ----------
    coords : lon/lat pair [deg]
    pole : S or N
    resolution : resolution of DEM [m/pxl]
    grid_size : full width of square DEM [pxl]

    Returns
    -------
        row, col of given point [pxl]
    """

    lam = coords[0]
    phi = coords[1]

    if lam > np.pi:
        lam = lam - 2.0*np.pi

    x = grid_size_x/2.0 + grid_size_x * lam / (2.0 * np.pi)
    y = grid_size_y/2.0 + grid_size_y/ np.pi * phi

    col = round(x)
    row = round(y)

    return row, col


# ---------------------------------------------------------------------------- #
#                              Plotting/animation                              #
# ---------------------------------------------------------------------------- #

def animate_array(mesh, N_CELLS, resolution, arr, vec2pole, fname, type):

    cell_centers = mesh.cell_centers().points
    x = [cell_centers[n][0] for n in range(N_CELLS)]
    y = [cell_centers[n][1] for n in range(N_CELLS)]

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    x_interp = np.linspace(x_min, x_max, round((x_max - x_min)/resolution))
    y_interp = np.linspace(y_min, y_max, round((y_max - y_min)/resolution))

    grid_x, grid_y = np.meshgrid(x_interp, y_interp, indexing="xy")
    mean_x = np.mean(grid_x)
    mean_y = np.mean(grid_y)
    points = np.array(list(zip(x,y)))


    #x_points = mesh.points
    #tri = mesh.faces.reshape((-1,4))[:, 1:]

    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)

    arr0 = interpolate.griddata(points, arr[:,0], (grid_x, grid_y), method='linear')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    if type == "illum":
        field = ax.pcolormesh(grid_x - mean_x, grid_y - mean_y, arr0, cmap="gist_gray", alpha=1.0,  rasterized=True, shading="auto", linewidth=0.0, vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="5%", pad=0.05, pack_start=True)
        fig.add_axes(cax)
        cb = fig.colorbar(field, cax=cax, orientation="horizontal")
        cb.set_label(r"Flux [$W/m{^2}$]", fontsize=20)

    elif type == "temp":
        field = ax.pcolormesh(grid_x - mean_x, grid_y - mean_y, arr0, cmap="plasma", alpha=1.0,  rasterized=True, shading="auto", linewidth=0.0, vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="5%", pad=0.05, pack_start=True)
        fig.add_axes(cax)
        cb = fig.colorbar(field, cax=cax, orientation="horizontal")
        cb.set_label(r"Temperature [K]", fontsize=20)

    ax.quiver(0.0, 0.0, vec2pole[0], vec2pole[1], color="r")    # Arrow points to pole
    ax.set_aspect('equal')
    ax.axis("off")
    

    def animate(t):
        arr_t = interpolate.griddata(points, arr[:,t], (grid_x, grid_y), method='linear')

        field.set_array(arr_t)

    frame_arr = range(0, arr.shape[1], 10)
    #frame_arr = range(0, 100, 1)

    duration = 30.0
    interval = duration * 1000.0 / len(frame_arr)
    fps = len(frame_arr) / duration 

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=tqdm(frame_arr, leave=False), interval=interval)
    
    anim.save(fname, fps=fps, dpi=100)

    return


def hillshade(array, resolution, azimuth=315.0, angle_altitude=10.0):
    azimuth = 360.0 - azimuth 

    x,y = np.gradient(array, resolution)
    slope = np.pi/2.0 - np.arctan(np.sqrt(x*x + y*y))

    aspect = np.arctan2(-x, y)
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)

    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos((azimuth_rad - np.pi/2.0) - aspect)

    return 255 * (shaded + 1) /2
