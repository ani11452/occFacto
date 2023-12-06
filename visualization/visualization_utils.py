import pyvista as pv
import collada
import numpy as np
import os
import torch
import mcubes

FILE_ROOT = "./all_test"

class Visualizer():
    '''
    A class that takes in a .dae mesh and then visualizes a front, side, or isometric view
    '''
    def __init__(self, token: str, tag: str = "best",
                 background_color: str = "#2a2b2e",
                 mesh_color: str = "#ffffff",
                 view: str = "isometric"):
        pv.start_xvfb()

        self.token = token
        self.tag = tag
        self.token_path = os.path.join(FILE_ROOT + f'_{tag}', token)
        dae_to_obj(self.token_path + f'_{tag}.dae', self.token_path + f'_{tag}.obj')

        # View
        self.background_color = background_color
        self.mesh_color = mesh_color
        self.view = view

        self.mesh = pv.read(self.token_path + f'_{tag}.obj')
        self.plotter = pv.Plotter(off_screen=True)
        self.init_plotter()

    def init_plotter(self):
        '''
        Initializes the plotter with mesh, background, and camera
        '''
        self.plotter.add_mesh(self.mesh, color=self.mesh_color)
        self.plotter.background_color = self.background_color

        if self.view == "front":
            self.plotter.camera_position = np.array([
                (3, 0, 0),                                  # Camera location (directly in front)
                (0, 0, 0),                                  # Focal point (center of the object)
                (0, 1, 0)                                   # Up direction (y-axis up)
            ])

        elif self.view == "side":
            self.plotter.camera_position = np.array([
                (0, 0, 3),                                  # Camera location (directly to the side)
                (0, 0, 0),                                  # Focal point (center of the object)
                (0, 1, 0)                                   # Up direction (y-axis up)
            ])

        # Default is isometric view
        else:
            self.plotter.camera_position = np.array([
                (2, 0.5, 2), # Camera location (From isometric view)
                (0, 0, 0),                                  # Focal point (center of the object)
                (0, 1, 0)                                   # Up direction (y-axis up)
            ])

    def save_view(self):
        '''
        Saves the view of the mesh to a png
        '''
        self.plotter.show(screenshot=f"{self.token}_{self.view}_{self.tag}.png")


def dae_to_obj(dae_file_path, obj_file_path):
    '''
    Converts a .dae file to an .obj file for visualization purposes

    Args:
        dae_file_path: file path of .dae file
        obj_file_path: file path of .obj file
    '''
    # Load the DAE file
    mesh = collada.Collada(dae_file_path)

    # Open an OBJ file to write to
    with open(obj_file_path, "w") as obj_file:
        # Write vertices, normals, and texture coordinates
        for geom in mesh.scene.objects('geometry'):
            for prim in geom.primitives():
                if type(prim) is collada.triangleset.BoundTriangleSet:
                    # Write vertices
                    for vertex in prim.vertex:
                        obj_file.write(f"v {' '.join(map(str, vertex))}\n")
                    # Write normals (if available)
                    if prim.normal is not None:
                        for normal in prim.normal:
                            obj_file.write(f"vn {' '.join(map(str, normal))}\n")
                    # Write faces
                    for tri in prim.vertex_index:
                        obj_file.write(f"f {' '.join([str(i+1) for i in tri])}\n")





class Mesher():
    def __init__(self, model, save_dir, tag: str="best"):
        self.device = 'cuda'
        self.model = model.eval()

        # Params for generating 3D grid
        self.padding = 0.1
        self.box_size = 1 + self.padding
        self.resolution = 25

        # Make occupancy grid
        p = self.box_size * self.make_3d_grid((-0.5,)*3, (0.5,)*3, (self.resolution,)*3)
        self.p = p.unsqueeze(0)

        # Save the train folder
        self.save_dir = save_dir
        self.tag = tag


    def eval_points(self, latents):
        '''
        Evaluates occupancy values for the points

        p (tensor): points (100, 100, 100)
        z (tensor): latent code (256, 1)
        '''
        self.model.eval()
        with torch.no_grad():
            p = self.p.to("cuda")
            occ_h = self.model(latents, p)
            return occ_h.detach().cpu()


    def generate_mesh(self, latents, token):
        occ_hat = self.eval_points(latents)

        # Processes the points to be readable by marching cubes
        occ_hat = occ_hat.view(self.resolution, self.resolution, self.resolution).numpy()

        occ_hat = mcubes.smooth(occ_hat)

        # Apply marching cubes algorithm
        vertices, triangles = mcubes.marching_cubes(occ_hat, 0)

        # Normalization
        n_x, n_y, n_z = occ_hat.shape
        vertices -= 1.5
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = self.box_size * (vertices - 0.5)
        
        print(os.path.join(self.save_dir, token) + f'_{self.tag}.dae')
        self.save_mesh(vertices, triangles, os.path.join(self.save_dir, token) + f'_{self.tag}.dae')

        
    def make_3d_grid(self, bb_min, bb_max, shape):
        ''' Makes a 3D grid.

        Args:
            bb_min (tuple): bounding box minimum
            bb_max (tuple): bounding box maximum
            shape (tuple): output shape
        '''
        size = shape[0] * shape[1] * shape[2]

        pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
        pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
        pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

        pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
        pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
        pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
        p = torch.stack([pxs, pys, pzs], dim=1)

        return p


    def save_mesh(self, vertices, triangles, path):
        mcubes.export_mesh(vertices, triangles, path)