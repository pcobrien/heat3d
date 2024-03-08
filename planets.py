############################################################
# All units M.K.S. unless otherwise stated
class Planet:
    """
    A Planet object contains basic planetary data.

    """

    def __init__(self, planet_name):
        
        if planet_name == "Moon":
            self.name = "Moon"  # Name of the planet
            self.R = 1737.4e3  # Planetary radius [m]
            self.g = 1.62  # Surface gravitational acceleration [m.s-2]
            self.S = 1361.0  # Annual mean solar constant [W.m-2]
            self.S_earth = 0.15     # Broadband emission of Earthshine [W.m-2] Glenar+ 2019
            self.albedo = 0.12  # Bond albedo
            self.emissivity = 0.95  # IR emissivity
            self.Qb = 0.018  # Heat flow [W.m-2] Langseth et al. 1976
            self.Gamma = 55.0  # Thermal inertia [J.m-2.K-1.s-1/2]
            self.rhos = 1100.0  # Density at surface [kg.m-3]
            self.rhod = 1800.0  # Density at depth [kg.m-3]
            self.H = 0.056  # e-folding scale of conductivity and density [m]
            self.cp0 = 600.0  # heat capacity at average surface temp. [J.kg.K-1]
            self.rsm = 149.60e9  # Semi-major axis [m]
            self.rem = 3.85e8    # Mean Earth-Moon distance [m]
            self.rAU = self.rsm / (1.49597871e+11)  # Semi-major axis [AU]
            self.year = 354.367 * 24.0 * 3600.0  # Sidereal length of year [s]
            self.eccentricity = 0.0167  # Eccentricity
            self.day = 29.53059 * 24.0 * 3600.0  # Mean length of SYNODIC day [s]
            self.obliquity = 0.026878  # Obliquity to orbit [radian]
            self.Tsmin = 25.0  # Minimum surface temperature [K]
            self.Tbmin = 50.0
            self.ks = 8.0e-4    # Feng+ 2020
            self.kd = 3.8e-3    # Feng+ 2020
            self.B =  4.2e-11
            self.solar_radius = 6.957e8 # solar radius [m]
            self.earth_radius = 6.371e6 # Earth radius [m]
            self.cpCoeff = [8.9093e-09, -1.234e-05, 0.0023616, 2.7431, -3.6125]     
            self.Q_starlight = float(3.5e-06)  # Average PSR flux from IPM and starlight [W.m-2]
            self.Ac = -0.203297     # Siegler and Martinez 2021
            self.Bc = -11.472
            self.Cc = 22.5793
            self.Dc = -14.3084
            self.Ec = 3.41742
            self.Fc = 0.01101
            self.Gc = -2.8049e-5
            self.Hc = 3.35837e-8
            self.Ic = -1.40021e-11
            self.A1 = 5.0821e-6
            self.A2 = 5.1e-3
            self.B1 = 2.0022e-13
            self.B2 = 1.953e-10
            self.obsctr = "MOON"
            self.obsref = "MOON_ME"


            
        elif planet_name == "Ceres":

            self.name = "Ceres"  # Name of the planet
            # self.R = ???  # Planetary radius [m]
            self.g = 0.27  # Surface gravitational acceleration [m.s-2]
            self.albedo = 0.09  # Bond albedo    
            self.emissivity = 0.95  # IR emissivity     
            self.Qb = 4.0e-3  # Heat flow [W.m-2]    
            self.Gamma = 15.0  # Thermal inertia [J.m-2.K-1.s-1/2] 
            self.rho = 1388.0
            self.rhos = 1388.0  # Density at surface [kg.m-3]       
            self.rhod = 1700.0
            self.ks = 4.0e-4
            self.kd = 5.7e-3
            self.B = 4.2e-11
            self.H = 0.06 # e-folding scale of conductivity and density [m]      
            self.cp = 837.0  # heat capacity at average surface temp. [J.kg.K-1]     
            self.rAU = 2.768  # Semi-major axis [AU]
            self.S = 1361.0/(self.rAU**2)  # Annual mean solar constant [W.m-2]
            self.rsm = self.rAU * (1.49597871e+11)  # Semi-major axis  [m]
            self.year = 4.59984 * 354.27 * 24.0 * 3600.0  # Sidereal length of year
            self.eccentricity = 0.0758  # Eccentricity  
            self.day = 32667.0  # Mean length of SYNODIC day [s]
            self.obliquity = 0.06981  # Obliquity to orbit [radian]       
            self.Lequinox = None  # Longitude of equinox [radian]
            self.Lp = 5.2709  # Longitude of perihelion [radian]   
            self.Tsavg = 235.0  # Mean surface temperature [K]
            self.Tsmax = 168.0  # Maximum surface temperature [K]
            self.Tsmin = 40.0  # Minimum surface temperature [K]        
            self.solar_radius = 6.957e8 # solar radius [m]