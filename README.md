# heat3d
3D thermal model for planetary surfaces

Language: Python

Author: Patrick O'Brien

Date: March 2024

## Overview

The model computes illumination and temperature on a mesh of triangular facets representing a planetary surface.

### Mesh ingestion

User can specify a target location (lon/lat), resolution, and radius and the model will extract that region from a DEM and triangulate a mesh. Alternatively, 
if a mesh file already exists, it can be directly loaded as long as the longitude and latitude of the mesh center are specified.

### View factor
To account for scattered light and thermal reradiation, the view factor matrix must be calculated. $F_{ij}$ gives the fraction of radiation leaving facet j that reaches facet i.
Note that the view factor is typically defined with the indices reversed, but this definition allows us to compute scattered radiation as $Q_{scattered} = F \cdot (\alpha Q_{incident})$
where $\alpha$ is albedo, $Q_{incident}$ is a vector giving the incident flux at each of the N facets, and F is the NxN view vector matrix.

The view factor calculation uses ray tracing to determine facet-to-facet visibility and takes advantage of reciprocity, i.e., $A_j F_{ij} = A_i F_{ij}$ to minimize the number of computations.

### Illumination
The model is capable of handling direct illumination (from multiple sources e.g., solar + Earthshine), scattering, and thermal radiation. User specifies a timestep and range over which to compute illumination.
Illuminating bodies are modeled as polygons and SPICE is used to determine the position of the polygon relative to the mesh as a function of time. At a given timestep, illumination is determined by raytracing from every facet on the mesh to every facet on the illuminating body polygon (fractional illumination is possible)


TO DO:
limb darkening

Earth phases


## Setup
After cloning the repository, the following files should be downloaded and placed inthe corresponding directories contained within *heat3d*.

### Albedo
Contains LOLA Albedo maps, accessible at https://imbrium.mit.edu/BROWSE/LOLA_GDR/ALBEDO/.

ldam_10_float.img

ldam_50n_1000m_float.img

ldam_50s_1000m_float.img

### Illumination
Illumination files will be stored here.

### kernels
SPICE kernels will be stored here. 

There should be a metakernel file, *kernels.tm* which contains the following:
    
\begindata

KERNELS_TO_LOAD = ( 'kernels/de440.bsp',

					'kernels/latest_leapseconds.tls',

					'kernels/moon_080317.tf',

					'kernels/moon_assoc_me.tf',

					'kernels/moon_assoc_pa.tf',

					'kernels/moon_de440_220930.tf',

					'kernels/moon_pa_de440_200625.bpc',

					'kernels/moon_pa_de440_200625.cmt',

					'kernels/naif0012.tls',

					'kernels/pck00011.tpc',

					)

\begintext
    
### Mesh
Mesh files will be stored here (.ply). User-provided mesh files can also be stored here and accessed directly from heat3d.py.


### Plots 
Optional visualization files can be saved here. 


### ViewFactor
View factor matrices will be stored here. Since these matrices are typically highly sparse for typical convex planetary surfaces,
view factors are stored as a sparse CSR matrix.


### Required modules
heat3d requires a number of modules, which can be found in the *requirements.txt* file. Note that embree/pyembree is not strictly required,
but will drastically speed up the raytracing components of the model.


## Quick Start

The main model file is heat3d.py. Running the file executes the following basic example:

1.  Create a mesh of Shackleton Crater

2.  Compute the view factor matrix

3.  Determine direct and secondary illumination over a period of one Earth year.
