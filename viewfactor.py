import numpy as np
from tqdm import tqdm
import sys 
from joblib import Parallel, delayed, cpu_count
from scipy.sparse import csr_matrix, save_npz, vstack, load_npz
import trimesh
import os 
from numba import njit, vectorize, float64, jit 
from time import time 
import config 
import math 


np.seterr(divide='ignore', invalid='ignore')

@vectorize([float64(float64, float64, float64, float64, float64)], nopython=True, target="cpu") 
def vf_M2(A_s, A_t, cos_theta_st, cos_theta_ts, d_st):
    return 4.0*np.sqrt(A_s * A_t) / ((np.pi ** 2) * A_s) * np.arctan2(np.sqrt(np.pi * A_s) * cos_theta_st, (2.0 * d_st)) * np.arctan2(np.sqrt(np.pi * A_t) * cos_theta_ts, (2.0 * d_st))

@njit(fastmath=True)
def vector_between(p_t, cell_center):
    return np.subtract(p_t, cell_center)

@vectorize([float64(float64, float64, float64)])
def calc_distance(q_st0, q_st1, q_st2):
    return np.sqrt(q_st0*q_st0 + q_st1*q_st1 + q_st2*q_st2)

def sparsity(mat):
    sparsity = 1.0 - mat.getnnz() / np.prod(mat.shape)
    return sparsity



def calc_viewfactor(fname_mesh, fname_vf):
    '''
    Calculate view factor matrix 

    Parameters
    ----------
    fname_vf : string
        mesh filename. Trimesh directly loads the mesh file
    N_CELLS : int
        Number of cells on mesh
    P : float array (1x3)
        Point of interest on mesh

    Returns
    -------
    poly : pyvista PolyData
        Polygon representing solar disk

    '''

    tmesh = trimesh.load_mesh(fname_mesh, use_embree=True)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(tmesh)

    cell_normals = tmesh.face_normals        # Normal vector of each facet
    cell_centers = tmesh.triangles_center + cell_normals * 1e-3   # Offset ray origins slightly to avoid intersecting starting facet
    cell_areas = tmesh.area_faces

    N_CELLS = len(cell_normals)     # Number of cells on mesh

    rows = []
    cols = []
    data = [] 

    original_ind_raw = np.array(range(N_CELLS))

    for target_ind in tqdm(range(N_CELLS)):

        p_t = cell_centers[target_ind]      # Target facet coordinates
        n_t = cell_normals[target_ind]      # Target facet normal
        A_t = cell_areas[target_ind]        # Target facet area

        p_s = cell_centers[target_ind+1 :]      # Source facet coordinates
        n_s = cell_normals[target_ind+1 : ]      # Source facet normal
        A_s = cell_areas[target_ind+1 : ]        # Source facet area
        source_ind = original_ind_raw[target_ind+1 : ]

        # Vectors between cell centers (from source to target)
        q_st = np.subtract(p_t, p_s)
        d_st = calc_distance(q_st[:,0], q_st[:,1], q_st[:,2])

        ########## ---------- ##########
        # NORMALS CHECK
        q_st /= np.sqrt(np.einsum('...i,...i', q_st, q_st))[:,None]     # Normalize direction vectors

        cos_theta_st = np.einsum('ij,ij->i', q_st, n_s)
        cos_theta_ts = np.einsum('j,ij->i', n_t, -q_st)


        toward_ind = np.where((cos_theta_st > 0) & (-cos_theta_ts < 0))[0]       # Indices of facets oriented toward the target facet

        # Downselect facets that are facing the target
        p_s = p_s[toward_ind]
        A_s = A_s[toward_ind]

        q_st = q_st[toward_ind]
        d_st = d_st[toward_ind]
        
        source_ind = source_ind[toward_ind]
        cos_theta_st = cos_theta_st[toward_ind]
        cos_theta_ts = cos_theta_ts[toward_ind]


        ########## ---------- ##########
        # OBSTRUCTION CHECK
        ind_intersect = intersector.intersects_first(p_s, q_st)

        del p_s, q_st, toward_ind

        clear_ind = np.where(ind_intersect == target_ind)[0]        # Index of facets with clear line of sight to target
        
        # Downselect facets with unobstructed view to target
        A_s = A_s[clear_ind]
        d_st = d_st[clear_ind]
        
        source_ind = source_ind[clear_ind]
        cos_theta_st = cos_theta_st[clear_ind]
        cos_theta_ts = cos_theta_ts[clear_ind]

        ########## ---------- ##########
        # Calculate view factors 
        vf_target = vf_M2(A_s, A_t, cos_theta_st, cos_theta_ts, d_st)

        ########## ---------- ##########
        # TOLERANCE CHECK
        min_ind = np.where(vf_target > config.tol_vf)[0]        # Index of facets with clear line of sight to target
        source_ind = source_ind[min_ind]
        vf_target = vf_target[min_ind]
        A_s = A_s[min_ind]

        rows_target = [target_ind] * len(source_ind)

        # Store row, col and value for non-zero viewfactor values (will be stored in sparse CSR matrix)
        rows.extend(rows_target)
        cols.extend(source_ind)
        data.extend(vf_target)

        # Reciprocity
        rows.extend(source_ind)
        cols.extend(rows_target)
        data.extend(vf_target * A_t / A_s)
        
    #rows = np.concatenate(rows)
    #cols = np.concatenate(cols)
    #data = np.concatenate(data)

    vf = csr_matrix((data, (rows, cols)), shape=(N_CELLS, N_CELLS))

    save_npz(fname_vf, vf)

    return vf 







    