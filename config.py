#import pyembree
#import trimesh 
#m = trimesh.creation.icosphere()
#print(m.ray)
#sys.exit()

import numpy as np
from astropy.constants import sigma_sb

# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #
TWO_PI = 2 * np.pi # 2 * pi 
sigma_sb = sigma_sb.value 


# ---------------------------------------------------------------------------- #
#                                     Mesh                                     #
# ---------------------------------------------------------------------------- #
fname_mesh = None       # Specify a filename to directly load mesh (rather than compute a new one each time)


# ---------------------------------------------------------------------------- #
#                                     SPICE                                    #
# ---------------------------------------------------------------------------- #
method = "ELLIPSOID"
azccw = False
elplsz = True
abcorr = "CN+S"     # or "LT+S"


# ---------------------------------------------------------------------------- #
#                                  Raytracing                                  #
# ---------------------------------------------------------------------------- #
TARGET_FACETS = 16
disk_distance_ratio = 100.0     # Place illuminating disc at disk_distance_ratio * mesh_width to avoid over/underflow errors


# ---------------------------------------------------------------------------- #
#                                  Viewfactor                                  #
# ---------------------------------------------------------------------------- #
tol_vf = 0.000000059604645     # Smallest number in float16 dtype. Ignore view factors smaller than this tolerance value


# ---------------------------------------------------------------------------- #
#                                 Thermal model                                #
# ---------------------------------------------------------------------------- #
DTBOT = 0.1     # The model will equilibrate until the change in temperature of the bottom layer is less than DTBOT over one annual cycle
DTSURF = 0.1    # Accuracy of surface temperature calculation
m = 10          # Number of layers in upper skin depth [default: 10]
n = 5           # Layer increase with depth: dz[i] = dz[i-1]*(1+1/n) [default: 5]
b = 20          # Number of skin depths to bottom layer [default: 20]
N_YEARS_EQUIL = 10


# ---------------------------------------------------------------------------- #
#                                 Illumination                                 #
# ---------------------------------------------------------------------------- #

