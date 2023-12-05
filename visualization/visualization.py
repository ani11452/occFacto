import pyvista as pv
import collada
import numpy as np
import math
import os

FILE_ROOT = "/home/cs236finalproject/diffFactoCS236/src/python/occFacto/models/occFactoDiffFreezeTrainingLegitWithSurf"

class Visualizer():
    '''
    A class that takes in a .dae mesh and then visualizes a front, side, or isometric view
    '''
    def __init__(self, token: str, 
                 background_color: str = "#2a2b2e",
                 mesh_color: str = "#ffffff",
                 view: str = "isometric"):
        pv.start_xvfb()

        self.token = token
        self.token_path = os.path.join(FILE_ROOT, token)
        dae_to_obj(self.token_path + '.dae', self.token_path + '.obj')

        # View
        self.background_color = background_color
        self.mesh_color = mesh_color
        self.view = view

        self.mesh = pv.read(self.token_path + '.obj')
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
        self.plotter.show(screenshot=f"{self.token}_{self.view}.png")


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

viz = Visualizer("def03f645b3fbd665bb93149cc0adf0_epoch_99", view="isometric")
viz.save_view()