# ---------------------------------------------------------------------------- #
#                               Import statements                              #
# ---------------------------------------------------------------------------- #
import numpy as np 
import matplotlib.pyplot as plt
import spiceypy as spice 
import sys
from scipy.sparse import load_npz
import pyvista as pv 

from planets import Planet
import config 
import topography
import viewfactor
import illumination
import heat1d 

plt.style.use("Plots/signature.mplstyle")

# ---------------------------------------------------------------------------- #
#                                Site parameters                               #
# ---------------------------------------------------------------------------- #

planet = Planet("Moon")
site = "Shackleton Crater"
resolution = 240.0

# ---------------------------------------------------------------------------- #
#                                     Mesh                                     #
# ---------------------------------------------------------------------------- #

fname_mesh = "Mesh/" + "mesh_" + '_'.join(site.split(' ')) + "_{}M".format(int(resolution)) + ".ply"
fname_mesh = "Mesh/spherical_crater.ply"

# Load mesh
#mesh, center_lon_rad, center_lat_rad, mesh_center, vec2pole = topography.load_mesh(fname_mesh=fname_mesh, site=site, planet=planet)
mesh, center_lon_rad, center_lat_rad, mesh_center, vec2pole = topography.load_mesh(fname_mesh=fname_mesh, center_lat_deg=0.0, center_lon_deg=0.0, planet=planet)

# Create mesh
#mesh, center_lon_rad, center_lat_rad, mesh_center, vec2pole = topography.createMeshfromDEM(site=site, resolution=resolution, planet=planet)

#mesh.plot_normals(mag=10.0)#, use_every=10)   # Check that mesh normls are pointing upwards
#topography.plot_mesh(mesh, mesh_center)      # Visualize mesh
#sys.exit()

# Number of mesh facets
N_CELLS = mesh.n_cells
print("Number of cells in mesh: ", N_CELLS)

# Get the bounds of the mesh
bounds = mesh.bounds

# Calculate the mesh dimensions
mesh_dimensions = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
config.mesh_width = mesh_dimensions[0]


# ---------------------------------------------------------------------------- #
#                                  Viewfactor                                  #
# ---------------------------------------------------------------------------- #

print("View factor...")
fname_vf = "ViewFactor/" + "vf_" + '_'.join(site.split(' ')) + "_{}M".format(resolution)

vf = viewfactor.calc_viewfactor(fname_mesh, fname_vf)
vf = load_npz(fname_vf + ".npz")

#print("View factor matrix sparsity: ", viewfactor.sparsity(vf))

# ---------------------------------------------------------------------------- #
#                                 Illumination                                 #
# ---------------------------------------------------------------------------- #

print("Illumination...")

nyears = 1.0

#utc_end = '2024 MAR 12 00:00:00'
#et_end = spice.str2et(utc_end)
#et_start = et_end - nyears*planet.year

utc_start = "2024 MAR 08 00:00:00"
et_start = spice.utc2et(utc_start)
et_end = et_start + nyears*planet.year

dt = (1.0/360.0)  * planet.day

nsteps  = round((et_end - et_start) / dt)

obslon = center_lon_rad
obslat = center_lat_rad 
obsalt = (planet.R + mesh_center[2])/1000.0

normal = spice.latrec(1.0, obslon, obslat)
z = [0.0, 0.0, 1.0]
xform = spice.twovec(normal, 3, z, 1)

outref = "MOON_ME"
abcorr = "CN+S"

refloc = "OBSERVER"

obspos = spice.latrec(planet.R/1000.0, obslon, obslat)
obsctr = "MOON"
obsref = "MOON_ME"


az_arr = []
el_arr = []

for n in range(nsteps):
    et = et_start + n*dt
    state, lt = spice.spkcpo("SUN", et, outref, refloc, abcorr, obspos, obsctr, obsref)

    topvec = spice.mxv(xform, state[0:3])

    target_r, target_lon, target_lat = spice.reclat(topvec)

    el =   target_lat * spice.dpr()
    az = - target_lon * spice.dpr()

    if az < 0.0:
        az +=  360.0

    az_arr.append(az)
    el_arr.append(el)

plt.figure()
plt.plot(az_arr)

plt.figure()
plt.plot(el_arr)
plt.show()
sys.exit()

# ---------------------------------------------------------------------------- #
#                          Incident Solar / Earthshine                         #
# ---------------------------------------------------------------------------- #

fname_Q_solar = "Illumination/" + "Q_solar_incident.npy"
fname_solar_incidence = "Illumination/" + "solar_incidence_angle.npy"

fname_Q_direct = "Illumination/" + "Q_solar_direct.npy"
fname_Q_scattered = "Illumination/" + "Q_solar_scattered.npy"


Q_solar, solar_incidence_angle = illumination.calc_illumination_par(mesh, center_lon_rad, center_lat_rad, vec2pole, N_CELLS, et_start, dt, nsteps, planet, "SUN", fname_mesh)
np.save(fname_Q_solar, Q_solar)
np.save(fname_solar_incidence, solar_incidence_angle)

Q_solar = np.load(fname_Q_solar)
solar_incidence_angle = np.load(fname_solar_incidence)

Q_direct, Q_scattered = illumination.calcDirectPlusScatteredIllumination(Q_solar, solar_incidence_angle, N_CELLS, nsteps, center_lon_rad, center_lat_rad, planet, mesh, vf)
np.save(fname_Q_direct, Q_direct)
np.save(fname_Q_scattered, Q_scattered)


Q_direct = np.load(fname_Q_direct)
Q_scattered = np.load(fname_Q_scattered)


mesh.cell_data["array"] = np.mean(Q_direct, axis=1)
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="array")
plotter.view_xy()
plotter.show()

mesh.cell_data["array"] = np.mean(Q_scattered, axis=1)
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="array")
plotter.view_xy()
plotter.show()
sys.exit()

# ---------------------------------------------------------------------------- #
#                                  Temperature                                 #
# ---------------------------------------------------------------------------- #

Q_direct_plus_reflected = Q_direct + Q_scattered 

Q_avg = np.mean(Q_direct_plus_reflected, axis=1)
temperature_initial, z, rho = heat1d.initialize_profiles(planet, N_CELLS, Q_avg)

"""
mesh.cell_data["array"] = temperature_initial[0,:]
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="array")
plotter.view_xy()
plotter.show()
"""

temperature_equilibrated = heat1d.equilibrate_profiles_coupled(temperature_initial, z, rho, vf, Q_direct_plus_reflected, N_CELLS, nsteps, dt, planet, mesh)


