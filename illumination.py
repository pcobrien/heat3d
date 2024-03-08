import numpy as np
import spiceypy as spice
import pyvista as pv 
from tqdm import tqdm 
import sys 
import funcs
import trimesh
from math import sin, cos
import config 
from joblib import Parallel, delayed, cpu_count
from time import time 
import matplotlib.pyplot as plt 
from numba import njit

spice.furnsh("kernels/kernels.tm")

#import pyembree
#import trimesh 
#m = trimesh.creation.icosphere()
#print(m.ray)
#sys.exit()




def calculate_A0(planet, lon, lat):
    # Use LOLA reflectance-derived albedo for the Moon, otherwise use Bond albedo specified in planets.py
    if planet.name == "Moon":
            if lat >= np.deg2rad(50.0):
                fname_albedo = "Albedo/ldam_50n_1000m_float.img"
                albedo_map = funcs.readIMG(fname_albedo, tp="alb")
                row, col = funcs.lonlat_to_rowcol_polar([lon, lat], "N", 1000.0, albedo_map.shape[0])

            elif lat <= np.deg2rad(-50.0):
                fname_albedo = "Albedo/ldam_50s_1000m_float.img"
                albedo_map = funcs.readIMG(fname_albedo, tp="alb")
                row, col = funcs.lonlat_to_rowcol_polar([lon, lat], "S", 1000.0, albedo_map.shape[0])

            else:
                fname_albedo = "Albedo/ldam_10_float.img"
                albedo_map = funcs.readIMG_cylindrical(fname_albedo, tp="alb")
                row, col = funcs.lonlat_to_rowcol_cylindrical([lon, lat], albedo_map.shape[1], albedo_map.shape[0])

            A0 = 0.39 * albedo_map[row, col-1]     # Feng+ (2020)

    else:
        A0 = planet.albedo

    return A0


@njit(fastmath=True)
def calcAlbedo(A0, incidence):
    """Calculate the variable albedo [from Feng+ (2020)]
    
    A0 : float or np.array(float)
        Normal albedo

    cos_theta: float or np.array(float)
        cosine of the solar incidence angle [rad]

    Returns
    -------
    float or np.array(float)
        Bolometric Bond albedo
    """

    albedo = A0 + (1.0 - (np.cos(incidence))**0.2752)
    albedo = np.clip(albedo, 0.0, 1.0)

    return albedo


def photoRender(mesh, planet, center_lon, center_lat, incidence, normals, directions, visible, incident, vec2pole, function="L-S"):
    A0 = calculate_A0(planet, center_lon, center_lat)
    albedo = calcAlbedo(A0, incidence)
    
    camera_normal = [0.0, 0.0, 1.0]
    emission = np.arccos(np.clip(np.dot(normals, camera_normal), -1.0, 1.0))
    phase = np.arccos(np.clip(np.dot(directions, camera_normal), -1.0, 1.0))

    if function == "CLEMENTINE":
        L = np.exp(-np.rad2deg(phase)/60.0)
        F = (1.0 - L)*np.cos(incidence) + L*np.cos(incidence)/(np.cos(incidence) + np.cos(emission))
        I = albedo * F

    elif function == "L-S":
        F = 2.0*np.cos(incidence) / ( np.cos(incidence) + np.cos(emission))
        I = albedo * F

    elif function == "LAMBERT":
        F = np.cos(incidence)
        I = albedo * F

    elif function == "MMPF-MH":
        # Mean Moon Photometric Function – Mature Highlands
        log_IF = -2.649 - 0.013 * phase - 0.274 * np.cos(emission) + 0.965 * np.cos(incidence)
        I = albedo * 10**log_IF


    I_photo = I * np.pi * visible
    
    #focal_dist = 1000.0
    #offset = 100.0
    #camera_height = 25.0
    #camera_position = (center[0] - vec2pole[0]*offset, center[1] - vec2pole[1]*offset, center[2] + camera_height)
    #closest_facet = mesh.points[mesh.find_closest_point(camera_position)]
    #camera_position = (closest_facet[0], closest_facet[1], closest_facet[2] + camera_height)
    #focal_point = (center[0] + vec2pole[0]*focal_dist, center[1] + vec2pole[1]*focal_dist, center[2])
    #view_up = (0.0, 0.0, 1.0)

    mesh.cell_data["I"] = I_photo
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="I", cmap="gist_gray")
    plotter.remove_scalar_bar()
    #plotter.camera_position = [camera_position, focal_point, view_up]
    plotter.view_xy()
    plotter.show()
    #plotter.screenshot("viper_landing_illum2.png")


    return 


def targetPoly(azimuth, inclination, dist, radius, num_facets, P):
    '''

    Parameters
    ----------
    azimuth : float
        Solar azimuth angle [rad]
    elevation : float
        Solar elevation angle [rad]
    dist : float
        Solar distance [m]
    P : float array (1x3)
        Point of interest on mesh

    Returns
    -------
    poly : pyvista PolyData
        Polygon representing solar disk

    '''
    
    x = dist * sin(inclination) * cos(azimuth) + P[0]
    y = dist * sin(inclination) * sin(azimuth) + P[1]
    z = dist * cos(inclination) + P[2]

    v = [x,y,z] - P
    vnorm = v/np.linalg.norm(v)

    poly = pv.Polygon(center=[x,y,z], radius=radius, normal=vnorm, n_sides=num_facets)
    
    return poly.extract_cells(0)


def calcTargetPosition(et, target, obspos, planet):
    """
    Return azimuth and elevation of target at specified time as seen from a point on the observer body

    VARIABLE  I/O  DESCRIPTION
    --------  ---  --------------------------------------------------
    et         I   Observation epoch.
    target     I   Name of target ephemeris object.
    obspos     I   Observer position relative to center of motion   [km]
    planet     I   Planetary parameters object [planets.py]

    [r, el, az] O   State of target with respect to observer,
                    r has same units as obspos, el/az have units of radians.

    """

    state, _ = spice.azlcpo(config.method, target, et, config.abcorr, config.azccw, config.elplsz, obspos, planet.obsctr, planet.obsref)

    # Azimuth and elevation angles
    r = state[0] * 1000.0  # Target position [m]
    az = state[1]
    el = state[2]

    return [r, el, az]


def getObserverTargetParameters(mesh, center_lon, center_lat, vec2pole, planet, target):
    """
    Calculate observer parameters. Position of the illuminating body is calculated relative to the center of the mesh

    VARIABLE                        I/O     DESCRIPTION
    --------                        ---     --------------------------------------------------
    mesh                            O       Mesh object
    center_lon                      F       Longitude at mesh center [rad]
    center_lat                      F       Latitude at mesh center [rad]
    vec2pole                        [F]     Vector pointing from center of mesh to pole
    planet                          O       Planetary parameters object
    target                          S       Target illuminating body ["SUN" or "EARTH"]

    P                               [F]     3D coordinates of observer point (the center of the mesh)
    az_offset                       F       Offset between 0 degrees azimuth in SPICE reference frame and pyvista reference frame
    target_radius_m                 F       Radius of target illuminating body [m]
    target_distance_m               F       Average distance to illuminating body [m]
    S_target                        F       Flux density (irradiance) of target illuminating body
    """

    P = mesh.points[mesh.find_closest_point(mesh.center)] # a point on the mesh at which to reference the solar disc position (facet closest to the center of the mesh)
    
    # Make sure that mesh heights are expressed in planetary radius, not measured relative to reference radius.
    obspos = spice.latrec(P[2], center_lon, center_lat) / 1000.0       # Rectangular [x,y,z] coordintes of observer position relative to center of motion [km]. Must be in km for spice.azlcpo 
    
    az_offset = np.arctan2(vec2pole[1], vec2pole[0])    # Offset between solar azimuth (0 pointing north, increasing eastward) and pyvista azimuth (0 pointing to the right of the mesh and increasing counter-clockwise)
    if center_lat < 0.0:
        az_offset += np.pi      # In the southern hemisphere, azimuth is still measured relative to north

    if target == "SUN":
        target_radius_m = planet.solar_radius 
        target_distance_m = planet.rsm
        S_target = planet.S
        
    elif target == "EARTH":
        target_radius_m = planet.earth_radius 
        target_distance_m = planet.rem 
        S_target = planet.S_earth


    return P, obspos, az_offset, target_radius_m, target_distance_m, S_target


def getTargetArrays(planet, obspos, az_offset, target, target_radius_m, target_distance_m, et_start, dt, nsteps):
    """
    Calculate position, size, and distance of target at each timestep

    VARIABLE                        I/O  DESCRIPTION
    --------                        ---  --------------------------------------------------
    planet                          O   Planetary parameters
    obspos                          F   Observer position relative to center of motion   [m]
    az_offset                       I   Offset between topocentric azimuth angle and pyvista azimuth angle
    target                          S   Name of target ephemeris object.
    target_radius_m                 F   Radius of target object [m]
    et_start                        F   Start of observation epoch [s]
    dt                              F   Timestep [s]
    nsteps                          I   Number of timesteps
    

    target_polygon_distance_arr     F   Distance between observer and target polygon. Scaled to arbitary distance to avoid over/underflow errors in raytracing model
    target_polygon_radius_arr       F   Apparent radius of target polygon. Scaled to arbitary distance to avoid over/underflow errors in raytracing model
    azimuth_arr                     F   Target azimuth angle. Measured in pyvista coordinate system (0 pointing in the positive x-direction, increasing counterclockwise)    [rad]
    incidence_arr                   F   Target incidence angle    [rad]
    squared_distance_arr            F   Squared ratio of target distance to "nominal" distance at which flux density is referenced (used in incident flux calculation)
    """

    target_polygon_distance_arr = np.zeros(nsteps)
    target_polygon_radius_arr = np.zeros(nsteps)
    azimuth_arr = np.zeros(nsteps)
    incidence_arr = np.zeros(nsteps)
    squared_distance_arr = np.zeros(nsteps)

    for time_ind in range(nsteps):
        et = et_start + time_ind*dt                                         # Time at which to compute target position, in "ephemeris time" or [s] past January 1, 2000.

        # Target position in observer reference frame. Azimuth measured from zero pointing north, increasing eastward.
        [r, el, az] = calcTargetPosition(et, target, obspos, planet)        # (distance [m], elevation angle [rad], azimuth angle [rad]).     

        DIST_SCFAC = (config.disk_distance_ratio*config.mesh_width) / r     # dist. scale factor to avoid rounding and underflow errors
        TARGET_DISTANCE = r * DIST_SCFAC                                    # scaled solar distance
        TARGET_RADIUS = target_radius_m * DIST_SCFAC                        # scaled target radius
        
        azimuth = az_offset - az                                            # target azimuth angle in pyvista frame [rad]
        inclination = np.pi/2.0 - el                                        # target inclination angle [rad]

        target_polygon_distance_arr[time_ind] = TARGET_DISTANCE
        target_polygon_radius_arr[time_ind] = TARGET_RADIUS
        azimuth_arr[time_ind] = azimuth
        incidence_arr[time_ind] = inclination
        squared_distance_arr[time_ind] = (r / target_distance_m)**2 

    return target_polygon_distance_arr, target_polygon_radius_arr, azimuth_arr, incidence_arr, squared_distance_arr


def loadRayTracer(fname_mesh):
    # Load trimesh RayMeshIntersector and pre-compute facet normals and origin positions
    tmesh = trimesh.load_mesh(fname_mesh, use_embree=True)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(tmesh)
    
    normals = tmesh.face_normals        # Normal vector of each facet
    origins = tmesh.triangles_center + normals * 1e-3   # Offset ray origins slightly to avoid intersecting starting facet

    return intersector, normals, origins


@njit(fastmath=False)
def illuminationFromVisibility(target_disc_visibility, cosine_target, squared_target_distance, S_target):
    
    # Flux is proportional to the disk fraction and direction cosine
    cosdir = 0.5 * (cosine_target + np.abs(cosine_target))
    incidence_angle = np.arccos(cosdir)

    theta = config.TWO_PI * target_disc_visibility / config.TARGET_FACETS
    frac = (theta - np.sin(theta)) / config.TWO_PI
    illum = frac * cosdir * (squared_target_distance * S_target) # Incident solar flux [W.m-2]

    return illum, incidence_angle


def calc_illumination_par(mesh, center_lon, center_lat, vec2pole, N_CELLS, et_start, dt, nsteps, planet, target, fname_mesh):
    """
    Return illumination map for mesh over given time range

    VARIABLE        I/O  DESCRIPTION
    --------        ---  --------------------------------------------------
    mesh            O   Pyvista mesh
    center_lon      F   Longitude of center of mesh [rad]
    center_lat      F   Latitude of center of mesh [rad]
    vec2pole        F   Tuple containing vector pointing from center of mesh to pole
    N_CELLS         I   Number of cells in the mesh
    et_start        F   Starting datetime for illumination model in ET format
    dt              F   Timestep [s]
    nsteps          I   Number of timesteps in illumination model
    planet          O   Planetary parameters
    target          S   Name of target ephemeris object

    illumination    F   Array of incident flux at every facet at each timestep
    incidence_angle F   Array of target incidence angle at every facet at each timestep
    """

    # ---------------------------------------------------------------------------- #
    #                          Observer-target parameters                          #
    # ---------------------------------------------------------------------------- #
    P, obspos, az_offset, target_radius_m, target_distance_m, S_target = getObserverTargetParameters(mesh, center_lon, center_lat, vec2pole, planet, target)

    # ---------------------------------------------------------------------------- #
    #                                Raycasting mesh                               #
    # ---------------------------------------------------------------------------- #
    intersector, normals, origins = loadRayTracer(fname_mesh)

    # ---------------------------------------------------------------------------- #
    #                        Precompute target disc position                       #
    # ---------------------------------------------------------------------------- #
    target_polygon_distance_arr, target_polygon_radius_arr, azimuth_arr, incidence_arr, squared_distance_arr = getTargetArrays(planet, obspos, az_offset, target, target_radius_m, target_distance_m, et_start, dt, nsteps)

    # ---------------------------------------------------------------------------- #
    #                Core illumination function for single timestep                #
    # ---------------------------------------------------------------------------- #
    def run_illumination_timestep(time_ind):
        # ---------------------------------------------------------------------------- #
        #              Create target mesh at position of disc at timestep              #
        # ---------------------------------------------------------------------------- #
        solar_disc = targetPoly(azimuth_arr[time_ind], incidence_arr[time_ind], target_polygon_distance_arr[time_ind], target_polygon_radius_arr[time_ind], config.TARGET_FACETS, P)

        # ---------------------------------------------------------------------------- #
        #                            Calculate illumination                            #
        # ---------------------------------------------------------------------------- #
        
        target_disc_visibility = np.zeros(N_CELLS, dtype=float)     # Visibility mask (number of target disc facets visible)
        cosine_arr = np.zeros((N_CELLS, config.TARGET_FACETS))      # Cosine of target incidence angle

        # Calculate visibility for each observer mesh facet that is facing the target disc
        for target_ind in range(config.TARGET_FACETS):

            endpoint = np.array(solar_disc.points[target_ind])   # Origin of rays at each facet, endpoint at current target disc facet

            directions = endpoint - origins # vector from mesh cells to target
            directions /= np.sqrt(np.einsum('...i,...i', directions, directions))[:,None]    # Normalize direction vectors

            cosine = np.einsum('ij,ij->i', normals, directions).T    # Cosine of angle between cell normal and target disc point

            cosine_arr[:, target_ind] = cosine
            ind_toward = np.where(cosine>0)[0]      # Only do raytracing for facets that are oriented towards the target disc facet (cos(i) > 0)

            if len(ind_toward) > 0:
                hits = intersector.intersects_any(origins[ind_toward], directions[ind_toward])      # Raytracing
                
                # Identify facets where hits == False, i.e., where the ray does not intersect anything between the mesh facet and the target disc
                target_visible = ind_toward[hits == False]

                target_disc_visibility[target_visible] += 1     # Increment disc visibility by 1 for those cells where disc facet is visible
            

        target_disc_visibility[target_disc_visibility == 1] = 0       # Trimesh produces some erroneous results when rays hit seam between facets. Require multiple visible solar disc facets for a terrain facet to be considered illuminated.
        cosine_target = np.mean(cosine_arr, axis=1)         # Average cosine of target incidence angle over all target disc facets. Used to calculate variable albedo of surface.

        # ---------------------------------------------------------------------------- #
        #                Convert target disc visiblity to incident flux                #
        # ---------------------------------------------------------------------------- #
        illumination_timestep, incidence_angle_timestep = illuminationFromVisibility(target_disc_visibility, cosine_target, squared_distance_arr[time_ind], S_target)
        
    
        # ---------------------------------------------------------------------------- #
        #                              Optional: Plotting                              #
        # ---------------------------------------------------------------------------- #
        
        """
        #I_photo = photoRender(mesh, planet, center_lon, center_lat, incidence_angle_timestep, normals, directions, target_disc_visibility, illumination_timestep, vec2pole, function="L-S")

        mesh.cell_data["I"] = illumination_timestep
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars="I", cmap="gist_gray")
        #plotter.add_mesh(solar_disc)
        plotter.view_xy()
        plotter.remove_scalar_bar()
        plotter.show()
        sys.exit()
        """        
        
        return illumination_timestep, incidence_angle_timestep

    # ---------------------------------------------------------------------------- #
    #                     Run timestep illumination in parallel                    #
    # ---------------------------------------------------------------------------- #
    #run_illumination_timestep(1)

    illumination_timestep, incidence_timestep = zip(*Parallel(n_jobs=cpu_count())(delayed(run_illumination_timestep)(time_ind) for time_ind in tqdm(range(nsteps))))

    # ---------------------------------------------------------------------------- #
    #                                    Output                                    #
    # ---------------------------------------------------------------------------- #
    illumination = np.vstack(illumination_timestep).T
    incidence_angle = np.vstack(incidence_timestep).T

    return illumination, incidence_angle


def calc_flux_t(vf, A0, Q_incident_t, incidence_angle_t):

    albedo_t = calcAlbedo(A0, incidence_angle_t)
    
    Q_direct_t = (1.0 - albedo_t) * Q_incident_t
    
    Q_scattered_t = (1.0 - albedo_t) * albedo_t * vf.dot(Q_incident_t)

    return Q_direct_t, Q_scattered_t
    

def calcDirectPlusScatteredIllumination(Q_incident, incidence_angle, N_CELLS, nsteps, center_lon, center_lat, planet, mesh, vf):
    """
    Calculate direct + scattered illumination from sources: Sun, Earth, starlight

    VARIABLE                I/O     DESCRIPTION
    --------                ---     --------------------------------------------------
    Q_solar_incident        F       Incident solar flux [W.m-2]
    Q_earth_incident        F       Incident Earthshine flux [W.m-2]
    solar_incidence_angle   F       Solar incidence angle [rad]
    earth_incidence_angle   F       Earth incidence angle [rad]
    N_CELLS                 I       Number of facets on mesh
    nsteps                  I       Number of timesteps
    center_lon              F       Longitude of mesh center [rad]
    center_lat              F       Latitude of mesh center  [rad]
    planet                  O       Planetary parameters object
    mesh                    O       Mesh object (used for plotting)
    vf                      F       View factor matrix (used for calculating scattered radiation)

    Q_direct_plus_reflected F       Array of total illumination flux at every facet at each timestep (directly absorbed and singly scattered) [W.m-2]
    """

    A0 = calculate_A0(planet, center_lon, center_lat)

    Q_direct_out, Q_scattered_out = zip(*Parallel(n_jobs=cpu_count())(delayed(calc_flux_t)(vf, A0, Q_incident[:, time_ind], incidence_angle[:, time_ind]) for time_ind in tqdm(range(nsteps))))

    Q_direct = np.zeros((N_CELLS, nsteps))
    Q_scattered = np.zeros((N_CELLS, nsteps))
    for time_ind in range(nsteps):
        Q_direct[:, time_ind] = Q_direct_out[time_ind]
        Q_scattered[:, time_ind] = Q_scattered_out[time_ind]
        
    return Q_direct, Q_scattered



def calc_thermal_t(vf, surface_temperature_t, emissivity):
    return emissivity**2 * config.sigma_sb * vf.dot(surface_temperature_t**4)
    

def calcThermalRadiation(N_CELLS, nsteps, planet, vf, surface_temperature):

    emissivity = planet.emissivity 

    Q_thermal_out = zip(*Parallel(n_jobs=cpu_count())(delayed(calc_thermal_t)(vf, surface_temperature[:, time_ind], emissivity) for time_ind in tqdm(range(nsteps))))

    Q_thermal = np.zeros((N_CELLS, nsteps))
    for time_ind in range(nsteps):
        Q_thermal[:, time_ind] = Q_thermal_out[time_ind]

    return Q_thermal