"""
1-d thermal modeling functions
"""
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import sigma_sb
from funcs import *
from planets import Planet
from tqdm import tqdm 
from numba import njit, vectorize, float64
import config 
from time import time 
import pyvista as pv 

plt.style.use("/Users/paob9380/PyTools/signature.mplstyle")


@njit(fastmath=True)
def skinDepth(P, kappa):
    """Calculate Thermal skin depth.
    
    Parameters
    ----------
    P : float
        period (e.g., diurnal, seasonal)
    kappa : float
        thermal diffusivity = k/(rho*cp) [m2.s-1]

    Returns
    -------
    float
        Thermal skin depth [m]
    """
    return np.sqrt(kappa * P / np.pi)

@njit(fastmath=True)
def spatialGrid(zs, m, n, b):
    """Calculate the spatial grid. 
    
    The spatial grid is non-uniform, with layer thickness increasing downward.

    Parameters:
    m : int
        Number of layers in upper skin depth [default: 10, set in Config]
    n : int
        Layer increase with depth: dz[i] = dz[i-1]*(1+1/n) [default: 5, set in Config]
    b : int
        Number of skin depths to bottom layer [default: 20, set in Config]

    Returns
    -------
    np.array(float)
        Spatial node coordinates in meters.
    """
    dz = np.zeros(1) + zs / m  # thickness of uppermost model layer
    z = np.zeros(1)  # initialize depth array at zero
    zmax = zs * b  # depth of deepest model layer

    i = 0
    while z[i] < zmax:
        i += 1
        h = dz[i - 1] * (1 + 1 / n)  # geometrically increasing thickness
        dz = np.append(dz, h)  # thickness of layer i
        z = np.append(z, z[i - 1] + dz[i])  # depth of layer i

    return z

@njit(fastmath=True)
def initTemperature(nlayers, N_CELLS, Tb0_arr, Ts0_arr, z, H):
    temperature = np.zeros((nlayers, N_CELLS))

    for n_facet in range(N_CELLS):
        temperature[:, n_facet] = Tb0_arr[n_facet]- (Tb0_arr[n_facet] - Ts0_arr[n_facet])*np.exp(-z/H)

    return temperature


@njit(fastmath=True)
def brenth(f, xa, xb, max_iter, args, xtol):

    xpre, xcur = xa, xb
    xblk, fblk, spre, scur = 0.0, 0.0, 0.0, 0.0

    # lower bound
    fpre = f(xpre, *args)

    # upper bound
    fcur = f(xcur, *args)

    # start iterations
    for i in range(max_iter):
        if (fpre*fcur < 0):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if (abs(fblk) < abs(fcur)):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            fpre = fcur
            fcur = fblk
            fblk = fpre

        # check bracket
        sbis = (xblk - xcur) / 2

        if abs(spre) > xtol and abs(fcur) < abs(fpre):
            if xpre == xblk:
                # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)

            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk - fpre) / (fblk * dpre - fpre * dblk)

            # check short step
            if (2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - xtol)):
                # good short step
                spre = scur
                scur = stry

            else:
                # bisect
                spre = sbis
                scur = sbis

        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if (abs(scur) > xtol):
            xcur += scur
        else:
            xcur += xtol if (sbis > 0) else -xtol

        fcur = f(xcur, *args)     # function evaluation

        if abs(fcur) < xtol:
            return xcur
        
    return xcur

#@njit(fastmath=True)
@vectorize([float64(float64)], nopython=True, target="cpu") 
def calcHeatCapacity(T):
    """Calculate the temperature-dependent heat capacity [from Hayne+ (2017)]
    
    T : float or np.array(float)
        Temperature [K]

    Returns
    -------
    float or np.array(float)
        Heat capacity [J.kg-1.K-1]
    """

    return -3.6125 + 2.7431*T + 2.3616e-3*T**2 -1.2340e-5*T**3 + 8.9093e-9*T**4


#@njit(fastmath=True)
@vectorize([float64(float64, float64)], nopython=True, target="cpu") 
def calcThermalConductivity(T, rho):
    """Calculate the temperature-dependent thermal conductivity [from Martinez & Siegler+ (2021)]
    
    T : float or np.array(float)
        Temperature [K]

    rho: float or np.array(float)
        Density [kg.m-3]

    Returns
    -------
    float or np.array(float)
        Thermal conductivity [W.m-1.K-1]
    """

    #k_am = -0.203297 + (-11.472 * T**-4) + (22.5793 * T**-3) + (-14.3084 * T**-2) + (3.41742 * T**-1) + (0.01101 * T) + (-2.8049e-5 * T**2) + (3.35837e-8 * T**3) + (-1.40021e-11 * T**4)
    #return (5.0821e-6 * rho[:, None] - 5.1e-3) * k_am + (2.0022e-13 * rho[:, None] - 1.953e-10) * T**3
    
    k_am = -0.203297 + (-11.472 * T**-4) + (22.5793 * T**-3) + (-14.3084 * T**-2) + (3.41742 * T**-1) + (0.01101 * T) + (-2.8049e-5 * T**2) + (3.35837e-8 * T**3) + (-1.40021e-11 * T**4)
    return (5.0821e-6 * rho - 5.1e-3) * k_am + (2.0022e-13 * rho - 1.953e-10) * T**3


@njit(fastmath=True)
def f_surf_boundary(T0, T1, T2, emissivity, sigma, Qs, k0, dz0):
    return emissivity * sigma * T0**4 - Qs - k0 * (-3.0*T0 + 4.0*T1 - T2)*(2.0*dz0)**-1


#@njit(fastmath=True)  
@vectorize([float64(float64, float64, float64, float64, float64, float64, float64, float64 )], fastmath=True, nopython=True, target="cpu")
def calcSurfTemp(T1, T2, emissivity, sigma, k0, dz0, Qs, DTSURF):
    """
    Surface temperature calculation using the Brent root-finding method with hyperbolic extrapolation
    Testing showed this method to be fastest of the scipy 1D root finding algorithms

    Parameters
    ----------
    T1 : float
        Temperature at first layer below surface (index 1)
    T2 : f;pat
        Temperature at second layer below surface (index 2)
    emissivity : float 
        lanetary emissivity
    sigma : float
        Stefan-Boltzmann constant
    k0  : float
        thermal conductivity at surface
    dz0 : float
        Spacing between surface and first layer
    Qs : float
        Heating rate [W.m-2] (e.g., insolation and infared heating)
    DTSURF : float
        Desired accuracy of surface temperature calculation

    Ts : float
        Surface temperature at next timestep
    """
    #Ts =_zeros._brenth(f_surf_boundary, 25.0, 425.0, DTSURF, 1.e-15, 1000, (T1, T2, emissivity, sigma, Qs, k0, dz0), False, True)

    return brenth(f_surf_boundary, 25.0, 425.0, 100, (T1, T2, emissivity, sigma, Qs, k0, dz0), DTSURF)


@vectorize([float64(float64, float64, float64, float64)], fastmath=True, nopython=True, target="cpu") 
def calcBotTemp(Qb, T2, k2, dz1):
    """Calculate bottom layer.

    Bottom layer temperature is calculated from the interior heat
    flux and the temperature of the layer above.

    Parameters
    ----------
    p : Profile object
    Qb : float
        Interior heat flux
    """
    return T2 + (Qb * k2**-1) * dz1


#@njit(fastmath=True)
@vectorize([float64(float64, float64, float64, float64, float64, float64, float64, float64)], fastmath=True, nopython=True, target="cpu") 
def updateTemperatureProfiles(dt, rho_term, cp_term, alpha, beta, temperature_back, temperature_middle, temperature_front):
    # This is an efficient vectorized form of the temperature
    # formula, which is much faster than a for-loop over the layers

    return dt * (rho_term * cp_term)**-1 * (alpha * temperature_back - (alpha + beta) * temperature_middle + beta * temperature_front)


@njit(fastmath=True)
def update_T(temperature, dt, Qs, Qb, g1, g2, k, cp, rho, emissivity, sigma, dz, DTSURF):
    """Core thermal computation.

    Calls `surfTemp` and `botTemp`.

    Parameters
    ----------
    temperature : [float]
        Temperature profiles
    dt : float
        Time step [s]
    Qs : float
        surface heating rate [W.m-2]
    Qb : float
        bottom heating rate (interior heat flow) [W.m-2]
    g1 : [float]

    g2 : [float]

    k : [float]

    cp : [float]

    rho : [float]

    emissivity : float
    
    sigma : float

    dz : [float]

    N_CELLS : int 

    DTSURF : float

    temperature : [float]
        Updated temperature profiles
    """

    # Coefficients for temperature-derivative terms
    alpha = g1[:, None] * k[0:-2, :]
    beta = g2[:, None] * k[1:-1, :]

    # Temperature of first layer is determined by energy balance at the surface
    #temperature[0, :] = [calcSurfTemp(temperature[1, n], temperature[2, n], emissivity, sigma, k[0, n], dz[0], Qs[n], DTSURF) for n in range(N_CELLS)]
    temperature[0, :] = calcSurfTemp(temperature[1, :], temperature[2, :], emissivity, sigma, k[0,: ], dz[0], Qs, DTSURF)

    # Temperature of the last layer is determined by the interior heat flux
    temperature[-1, :] = calcBotTemp(Qb, temperature[-2, :], k[-2, :], dz[-1])

    # Update rest of profile layers
    temperature[1: -1] += updateTemperatureProfiles(dt, rho[1:-1, None], cp[1:-1, :], alpha, beta, temperature[0:-2, :], temperature[1:-1, :], temperature[2:, :])

    return temperature


def initialize_profiles(planet, N_CELLS, Q_avg_arr, load_profiles=None):

    # Initial temperatures
    Ts0_arr = (Q_avg_arr / (4.0*planet.emissivity*config.sigma_sb))**0.25
    Tb0_arr = Ts0_arr * (2.0**0.25)

    Ts0_arr[Tb0_arr <= planet.Tbmin] = planet.Tsmin 
    Tb0_arr[Tb0_arr <= planet.Tbmin] = planet.Tbmin 
  
    # Initialize model profile
    ks = planet.ks
    rhos = planet.rhos
    rhod = planet.rhod
    H = planet.H
    cp0 = np.min(calcHeatCapacity(Ts0_arr))
    kappa = ks / (rhos * cp0)

    z = spatialGrid(skinDepth(planet.day, kappa), config.m, config.n, config.b)

    nlayers = np.size(z)  # number of model layers

    rho = rhod - (rhod - rhos) * np.exp(-z / H)

    if load_profiles:
        temperature = np.load(load_profiles)
    
    else:
        temperature = initTemperature(nlayers, N_CELLS, Tb0_arr, Ts0_arr, z, planet.H)


    return temperature, z, rho



def equilibrate_profiles_coupled(temperature, z, rho, vf, Q_direct_plus_reflected, N_CELLS, nsteps, dt, planet, mesh):

    # ---------------------------------------------------------------------------- #
    #                                  Parameters                                  #
    # ---------------------------------------------------------------------------- #
    Qb = planet.Qb 
    DTBOT = config.DTBOT
    DTSURF = config.DTSURF
    emissivity = planet.emissivity
    sigma = config.sigma_sb 
    thermal_term = emissivity**2 * sigma 

    dz = np.diff(z)
    d3z = dz[1:] * dz[0:-1] * (dz[1:] + dz[0:-1])
    g1 = 2 * dz[1:] / d3z[0:] 
    g2 = 2 * dz[0:-1] / d3z[0:]

    mesh.cell_data["array"] = temperature[0, :]
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="array")
    plotter.view_xy()
    plotter.show()

    # Initialize 
    for n_year in tqdm(range(config.N_YEARS_EQUIL)):

        T_b_arr = np.copy(temperature[-1, :][:])        # Temperature of bottom layer at previous equilibration year

        for n_timestep in tqdm(range(nsteps), leave=False):

            # Update temperature-dependent heat capacity and thermal conductivity 
            cp = calcHeatCapacity(temperature)
            k = calcThermalConductivity(temperature, rho[:, None])
    
            # Total flux at each facet at current timestep
            Qs = Q_direct_plus_reflected[:, n_timestep] + thermal_term * vf.dot(temperature[0, :]**4)
            
            # Update temperature
            temperature = update_T(temperature, dt, Qs, Qb, g1, g2, k, cp, rho, emissivity, sigma, dz, DTSURF)

            mesh.cell_data["array"] = temperature[0, :]
            plotter = pv.Plotter()
            plotter.add_mesh(mesh, scalars="array")
            plotter.view_xy()
            plotter.show()
            sys.exit()
        

        #np.save("Profiles/temperature_coupled_YEAR{}".format(str(n_year+1).zfill(2)), temperature)

        T_b_arr_new = temperature[-1, :]        # Temperature of bottom layer at current equilibration year

        if np.max(np.abs(T_b_arr_new - T_b_arr)) <= DTBOT:
            print("Equilibrated in {} years.".format(n_year+1))
            break
        else:
            print("Maximum temperature change in bottom layer: {} K".format(np.round(np.max(np.abs(T_b_arr_new - T_b_arr)), 5)))


    return temperature


def run_temperature_coupled(temperature, z, rho, vf, Q_direct_plus_reflected, N_CELLS, nsteps, dt, planet, mesh):
    # ---------------------------------------------------------------------------- #
    #                                  Parameters                                  #
    # ---------------------------------------------------------------------------- #
    Qb = planet.Qb 
    DTSURF = config.DTSURF
    emissivity = planet.emissivity
    sigma = config.sigma_sb 
    thermal_term = emissivity**2 * sigma 

    dz = np.diff(z)
    d3z = dz[1:] * dz[0:-1] * (dz[1:] + dz[0:-1])
    g1 = 2 * dz[1:] / d3z[0:] 
    g2 = 2 * dz[0:-1] / d3z[0:]

    # ---------------------------------------------------------------------------- #
    #                                    Output                                    #
    # ---------------------------------------------------------------------------- #
    T_min = np.copy(temperature[:])
    T_avg = np.copy(temperature[:])
    T_max = np.copy(temperature[:])

    # ---------------------------------------------------------------------------- #
    #                               Run thermal model                              #
    # ---------------------------------------------------------------------------- #
    for n_timestep in tqdm(range(nsteps), leave=False):

        # Update temperature-dependent heat capacity and thermal conductivity 
        cp = calcHeatCapacity(temperature)
        k = calcThermalConductivity(temperature, rho[:, None])

        # Total flux at each facet at current timestep
        Qs = Q_direct_plus_reflected[:, n_timestep] + thermal_term * vf.dot(temperature[0, :]**4)
        
        # Update temperature
        temperature = update_T(temperature, dt, Qs, Qb, g1, g2, k, cp, rho, emissivity, sigma, dz, DTSURF)

        T_min = np.min([T_min, temperature], axis=0)
        T_max = np.max([T_max, temperature], axis=0)
        T_avg += np.subtract(temperature, T_avg) / (n_timestep + 1)


    return T_min, T_avg, T_max


    
