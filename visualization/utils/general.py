import numpy as np
import open3d as o3d
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion

view_dict = {'top':(0., 500.), "left":(500, 0), 'front':(1000, 0), 'top_front':(700, 250)}

def text_3d(text, pos, direction=None, degree=0.0, font='/System/Library/Fonts/Times.ttc', font_size=50):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)


    text = str(text)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    indices[:, 2] *= -1
    indices[:, 1] *= -1
    indices = indices / 100.0
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices - indices.mean(0))

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd
def capture_screen_shot(vis, path='./img.png'):
    vis.capture_screen_image(path)
    print(f'Saved at {path}')
    return False

def normalize(pc):
    if len(pc.shape) == 3:
        B = pc.shape[0]
        shift = pc.min(1).reshape(B, 1, 3)
        scale = pc.max(1).reshape(B, 1, 3) - pc.min(1).reshape(B, 1,  3)
    else:
        shift = pc.min(0).reshape(1, 3)
        scale = pc.max(0).reshape(1, 3) - pc.min(0).reshape(1, 3)
    return (pc - shift) / scale

def save_view(_vis):
    view_contrl = _vis.get_view_control()
    return view_contrl.convert_to_pinhole_camera_parameters()

def gather_numpy(self, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)