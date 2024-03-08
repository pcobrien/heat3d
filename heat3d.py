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

plt.style.use("signature.mplstyle")

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
#mesh, center_lon_rad, center_lat_rad, mesh_center, vec2pole = topography.load_mesh(fname_mesh=fname_mesh, site=site, planet=planet)
mesh, center_lon_rad, center_lat_rad, mesh_center, vec2pole = topography.createMeshfromDEM(site=site, resolution=resolution, planet=planet)
#mesh.plot_normals(mag=1000.0, use_every=100)
topography.plot_mesh(mesh, mesh_center, vec2pole)
sys.exit()

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

#vf = viewfactor.calc_viewfactor(fname_mesh, fname_vf)
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

dt = (1.0/720.0)  * planet.day

nsteps  = round((et_end - et_start) / dt)

# ---------------------------------------------------------------------------- #
#                          Incident Solar / Earthshine                         #
# ---------------------------------------------------------------------------- #

fname_Q_solar = "Illumination/" + "Q_solar_incident.npy"
fname_solar_incidence = "Illumination/" + "solar_incidence_angle.npy"

fname_Q_direct = "Illumination/" + "Q_solar_direct.npy"
fname_Q_scattered = "Illumination/" + "Q_solar_scattered.npy"

#Q_solar, solar_incidence_angle = illumination.calc_illumination_par(mesh, center_lon_rad, center_lat_rad, vec2pole, N_CELLS, et_start, dt, nsteps, planet, "SUN", fname_mesh)
#np.save(fname_Q_solar, Q_solar)
#np.save(fname_solar_incidence, solar_incidence_angle)

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