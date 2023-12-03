import os
import time
import numpy as np
import trimesh
import external.libmcubes as libmcubes

GENERATION_DIR = os.path.join('/home/cs236finalproject/diffFactoCS236/data/external')

class Generator():
    def __init__(self, 
                 vis_n_outputs=-1, 
                 generate_mesh=True, 
                 generate_pointcloud=True,
                 threshold=0.5):
        self.threshold = threshold

        # Creates various different directories for meshes and point clouds
        self.mesh_dir = os.path.join(GENERATION_DIR, 'meshes')
        self.pointcloud_dir = os.path.join(GENERATION_DIR, 'pointcloud')
        self.in_dir = os.path.join(GENERATION_DIR, 'input')
        self.generation_vis_dir = os.path.join(GENERATION_DIR, 'vis')

        if vis_n_outputs >= 0 and not os.path.exists(self.generation_vis_dir):
            os.makedirs(self.generation_vis_dir)

        if generate_mesh and not os.path.exists(self.mesh_dir):
            os.makedirs(self.mesh_dir)

        if generate_pointcloud and not os.path.exists(self.pointcloud_dir):
            os.makedirs(self.pointcloud_dir)

        if not os.path.exists(self.in_dir):
            os.makedirs(self.in_dir)


    def extract_mesh(self, occ_hat, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5

        # Undo padding
        vertices -= 1

        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        # if self.with_normals and not vertices.shape[0] == 0:
        #     t0 = time.time()
        #     normals = self.estimate_normals(vertices, z, c)
        #     stats_dict['time (normals)'] = time.time() - t0

        normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # Refine mesh
        # if self.refinement_step > 0:
        #     t0 = time.time()
        #     self.refine_mesh(mesh, occ_hat, z, c)
        #     stats_dict['time (refine)'] = time.time() - t0

        return mesh


