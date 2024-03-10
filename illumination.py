import numpy as np
import spiceypy as spice
import pyvista as pv 
from tqdm import tqdm 
import sys 
import trimesh
from math import sin, cos
import config 
from joblib import Parallel, delayed
from numba import njit
import pyembree 

import funcs

spice.furnsh("kernels/kernels.tm")

#import pyembree
#import trimesh 
#m = trimesh.creation.icosphere()
#print(m.ray)       # If "<trimesh.ray.ray_pyembree.RayMeshIntersector at 0x7f8a8fc6e668>", embree is successfully installed
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

    mesh.cell_data["I"] = I_photo
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="I", cmap="gist_gray")
    plotter.remove_scalar_bar()
    plotter.view_xy()
    plotter.show()
    #plotter.screenshot("illum.png")

    return 


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
    

    polygon_distance_arr            F   Distance between observer and target polygon. Scaled to arbitary distance to avoid over/underflow errors in raytracing model
    polygon_radius_arr              F   Apparent radius of target polygon. Scaled to arbitary distance to avoid over/underflow errors in raytracing model
    azimuth_arr                     F   Target azimuth angle. Measured in pyvista coordinate system (0 pointing in the positive x-direction, increasing counterclockwise)    [rad]
    incidence_arr                   F   Target incidence angle    [rad]
    squared_distance_arr            F   Squared ratio of target distance to "nominal" distance at which flux density is referenced (used in incident flux calculation)
    """

    polygon_distance_arr = np.zeros(nsteps)
    polygon_radius_arr = np.zeros(nsteps)
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

        polygon_distance_arr[time_ind] = TARGET_DISTANCE
        polygon_radius_arr[time_ind] = TARGET_RADIUS
        azimuth_arr[time_ind] = azimuth
        incidence_arr[time_ind] = inclination
        squared_distance_arr[time_ind] = (r / target_distance_m)**2 

    return polygon_distance_arr, polygon_radius_arr, azimuth_arr, incidence_arr, squared_distance_arr


def calcIlluminatingDiscPolyPoints(P, target_polygon_distance_arr, target_polygon_radius_arr, target_azimuth_arr, target_inclination_arr, nsteps):

    pos_target = np.zeros((nsteps, 3))
    pos_target[:, 0] = target_polygon_distance_arr * np.sin(target_inclination_arr) * np.cos(target_azimuth_arr) + P[0]
    pos_target[:, 1] = target_polygon_distance_arr * np.sin(target_inclination_arr) * np.sin(target_azimuth_arr) + P[1]
    pos_target[:, 2] = target_polygon_distance_arr * np.cos(target_inclination_arr) + P[2]

    v_target = pos_target - P 
    v_target /= np.sqrt(np.einsum('...i,...i', v_target, v_target))[:,None]    # Normalize vectors from P to target disc center

    illuminatingDiscPoints = np.zeros((config.TARGET_FACETS, 3, nsteps))
    
    for time_ind in range(nsteps):
        disc = pv.Polygon(center=[pos_target[time_ind,0], pos_target[time_ind,1], pos_target[time_ind,2]], radius=target_polygon_radius_arr[time_ind], normal=v_target[time_ind], n_sides=config.TARGET_FACETS).extract_cells(0)
        illuminatingDiscPoints[:, :, time_ind] = disc.points

    return illuminatingDiscPoints, pos_target


def loadRayTracer(fname_mesh):
    # Load trimesh RayMeshIntersector and pre-compute facet normals and origin positions
    tmesh = trimesh.load_mesh(fname_mesh, use_embree=True)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(tmesh)
    
    normals = tmesh.face_normals        # Normal vector of each facet
    origins = tmesh.triangles_center + normals * 1e-3   # Offset ray origins slightly to avoid intersecting starting facet

    return intersector, normals, origins


def run_illumination_timestep(N_CELLS, TARGET_FACETS, normals, origins, intersector, illuminating_disc, illuminating_disc_center):

        # ---------------------------------------------------------------------------- #
        #                            Calculate illumination at timestep                #
        # ---------------------------------------------------------------------------- #
        vis_frac_factor = 2.0*np.pi / TARGET_FACETS 
        inv_two_pi = 1.0 / (2.0*np.pi)

        illuminating_disc_visibility = np.zeros(N_CELLS)     # Visibility mask (number of target disc facets visible)

        # Calculate visibility for each observer mesh facet that is facing the target disc
        for target_ind in range(TARGET_FACETS):
            endpoint = illuminating_disc[target_ind]   # Origin of rays at each facet, endpoint at current target disc facet
            
            directions = endpoint - origins # vector from mesh cells to target
            directions /= np.sqrt(np.einsum('...i,...i', directions, directions))[:,None]    # Normalize direction vectors

            cosine = np.einsum('ij,ij->i', normals, directions).T    # Cosine of angle between cell normal and target disc point

            ind_toward = np.where(cosine>0)[0]      # Only do raytracing for facets that are oriented towards the target disc facet (cos(i) > 0)

            if len(ind_toward) > 0:
                hits = intersector.intersects_any(origins[ind_toward], directions[ind_toward])      # Raytracing

                # Identify facets where hits == False, i.e., where the ray does not intersect the mesh anywhere between the mesh facet and the target disc
                target_visible = ind_toward[~hits]

                illuminating_disc_visibility[target_visible] += 1     # Increment disc visibility by 1 for those cells where disc facet is visible


        illuminating_disc_visibility[illuminating_disc_visibility == 1] = 0       # Trimesh produces some erroneous results when rays hit seam between facets. Require multiple visible solar disc facets for a terrain facet to be considered illuminated.

        # Flux is proportional to the disk fraction and direction cosine, so store these quantities
        # Calculate cosine incidence angle relative to the center of the illuminating disc
        directions = illuminating_disc_center - origins # vector from mesh cells to target
        directions /= np.sqrt(np.einsum('...i,...i', directions, directions))[:,None]    # Normalize direction vectors
        cosine_target = np.einsum('ij,ij->i', normals, directions).T    # Cosine of angle between cell normal and target disc point

        cosine_incidence_angle_timestep = 0.5 * (cosine_target + np.abs(cosine_target))      # Cosine incidence angle
        theta = vis_frac_factor * illuminating_disc_visibility      # Fraction of target disc visible from each facet
        fraction_visible_timestep = (theta - np.sin(theta)) * inv_two_pi

        # ---------------------------------------------------------------------------- #
        #                              Optional: Plotting                              #
        # ---------------------------------------------------------------------------- #
        
        """
        #I_photo = photoRender(mesh, planet, center_lon, center_lat, incidence_angle_timestep, normals, directions, target_disc_visibility, illumination_timestep, vec2pole, function="L-S")

        mesh.cell_data["I"] = illumination_timestep
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars="I", cmap="gist_gray")
        #plotter.add_mesh(illuminating_disc)
        plotter.view_xy()
        plotter.remove_scalar_bar()
        plotter.show()
        sys.exit()
        """
        
        # Return fraction of disc visible and cosine of incidence angle
        return fraction_visible_timestep, cosine_incidence_angle_timestep


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
    target_polygon_distance_arr, target_polygon_radius_arr, target_azimuth_arr, target_inclination_arr, squared_target_distance_arr = getTargetArrays(planet, obspos, az_offset, target, target_radius_m, target_distance_m, et_start, dt, nsteps)

    illuminatingDiscPoints, illuminatingDiscCenter = calcIlluminatingDiscPolyPoints(P, target_polygon_distance_arr, target_polygon_radius_arr, target_azimuth_arr, target_inclination_arr, nsteps)

    # ---------------------------------------------------------------------------- #
    #                     Run timestep illumination in parallel                    #
    # ---------------------------------------------------------------------------- #
    #run_illumination_timestep(N_CELLS, config.TARGET_FACETS, normals, origins, intersector, illuminatingDiscPoints[:,:,0], illuminatingDiscCenter[0, :])
    #sys.exit()

    fraction_visible_timestep, cosine_incidence_timestep = zip(*Parallel(n_jobs=-1)(delayed(run_illumination_timestep)(N_CELLS, config.TARGET_FACETS, normals, origins, intersector, illuminatingDiscPoints[:,:,time_ind], illuminatingDiscCenter[time_ind, :]) for time_ind in tqdm(range(nsteps))))

    # ---------------------------------------------------------------------------- #
    #                                    Output                                    #
    # ---------------------------------------------------------------------------- #

    illumination = np.zeros((N_CELLS, nsteps))
    cosine_incidence_angle = np.zeros((N_CELLS, nsteps))

    for time_ind in range(nsteps):
        cosi = cosine_incidence_timestep[time_ind]
        cosine_incidence_angle[:, time_ind] = cosi

        # Convert fraction of disc visible into incident flux in [W.m-2]
        illumination[:, time_ind] = fraction_visible_timestep[time_ind] * cosi * squared_target_distance_arr[time_ind] * S_target

    incidence_angle = np.arccos(cosine_incidence_angle)

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

    Q_direct_out, Q_scattered_out = zip(*Parallel(n_jobs=-1)(delayed(calc_flux_t)(vf, A0, Q_incident[:, time_ind], incidence_angle[:, time_ind]) for time_ind in tqdm(range(nsteps))))

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

    Q_thermal_out = zip(*Parallel(n_jobs=-1)(delayed(calc_thermal_t)(vf, surface_temperature[:, time_ind], emissivity) for time_ind in tqdm(range(nsteps))))

    Q_thermal = np.zeros((N_CELLS, nsteps))
    for time_ind in range(nsteps):
        Q_thermal[:, time_ind] = Q_thermal_out[time_ind]

    return Q_thermal