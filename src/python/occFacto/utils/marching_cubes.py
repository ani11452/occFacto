# FROM SPAGHETTI

import skimage.measure
import time
from ..models.occ_types import *
from utils.train_utils import Logger
import constants


def mcubes_skimage(pytorch_3d_occ_tensor: T, voxel_grid_origin: List[float], voxel_size: float) -> T_Mesh:
    """
    Applies the Marching Cubes algorithm to a 3D occupancy grid tensor to generate a mesh.

    This function converts a PyTorch tensor representing a 3D occupancy grid into a NumPy array, 
    then uses the skimage's marching cubes implementation to compute the vertices, faces, 
    normals, and values of the mesh.

    Parameters:
    - pytorch_3d_occ_tensor (T): A 3D tensor representing the occupancy grid.
    - voxel_grid_origin (List[float]): A list of three floats representing the origin of the voxel grid.
    - voxel_size (float): The size of each voxel.

    Returns:
    - T_Mesh: A tuple containing the mesh vertices and faces as PyTorch tensors.

    Raises:
    - BaseException: If the marching cubes algorithm fails for any reason.
    """
    numpy_3d_occ_tensor = pytorch_3d_occ_tensor.numpy()
    try:
        marching_cubes = skimage.measure.marching_cubes if 'marching_cubes' in dir(skimage.measure) else skimage.measure.marching_cubes_lewiner
        verts, faces, normals, values = marching_cubes(numpy_3d_occ_tensor, level=0.0, spacing=[voxel_size] * 3)
    except BaseException:
        print("mc failed")
        return None
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]
    return torch.from_numpy(mesh_points.copy()).float(), torch.from_numpy(faces.copy()).long()


class MarchingCubesMeshing:
    """
    A class to handle mesh generation from occupancy values using the Marching Cubes algorithm.

    This class provides methods for filling samples with occupancy values, recursively adjusting
    resolution, and generating meshes from occupancy grids.
    """

    def fill_samples(self, decoder, samples, device: Optional[D] = None) -> T:
        """
        Fills in the occupancy values for a set of samples using a given decoder.

        The method divides the samples into batches and processes each batch through the decoder, 
        updating the occupancy values in the samples tensor.

        Parameters:
        - decoder: A function or model that takes sample coordinates and returns occupancy values.
        - samples: A tensor of sample coordinates and occupancy values.
        - device (Optional[D]): The device to which the samples should be moved for processing.

        Returns:
        - T: The updated samples tensor with occupancy values filled in.
        """
        num_samples = samples.shape[1]
        num_iters = num_samples // self.max_batch + int(num_samples % self.max_batch != 0)
        sample_coords = samples[:3]
        if self.verbose:
            logger = Logger()
            logger.start(num_iters, tag='meshing')
        for i in range(num_iters):
            sample_subset = sample_coords[:, i * self.max_batch: min((i + 1) * self.max_batch, num_samples)]
            if device is not None:
                sample_subset = sample_subset.to(device)
            sample_subset = sample_subset.T
            samples[3, i * self.max_batch: min((i + 1) * self.max_batch, num_samples)] = (
                decoder(sample_subset * self.scale).squeeze().detach()
            )
            if self.verbose:
                logger.reset_iter()
        if self.verbose:
            logger.stop()
        return samples

    def fill_recursive(self, decoder, samples: T, stride: int, base_res: int, depth: int) -> T:
        """
        Recursively fills in the occupancy values for a set of samples at varying resolutions.

        This method applies a multi-resolution approach to filling occupancy values, 
        progressively refining the resolution and filling in more detailed occupancy information.

        Parameters:
        - decoder: A function or model that returns occupancy values.
        - samples (T): A tensor of sample coordinates and initial occupancy values.
        - stride (int): The stride to use for downscaling resolutions.
        - base_res (int): The base resolution to start from.
        - depth (int): The depth of recursion.

        Returns:
        - T: The samples tensor with occupancy values filled in at the appropriate resolution.
        """
        if base_res <= self.min_res:
            samples_ = self.fill_samples(decoder, samples)
            return samples_
        kernel_size = 7 + 4 * depth
        padding = tuple([kernel_size // 2] * 6)
        samples_ = samples.view(1, 4, base_res, base_res, base_res)
        samples_ = nnf.avg_pool3d(samples_, stride, stride)
        samples_ = samples_.view(4, -1)
        res = base_res // stride
        samples_lower = self.fill_recursive(decoder, samples_, stride, res, depth - 1)
        mask = samples_lower[-1, :].lt(.3)
        mask = mask.view(1, 1, res, res, res).float()
        mask = nnf.pad(mask, padding, mode='replicate')
        mask = nnf.max_pool3d(mask, kernel_size, 1)
        mask = nnf.interpolate(mask, scale_factor=stride)
        mask = mask.flatten().bool()
        samples[:, mask] = self.fill_samples(decoder, samples[:, mask])
        return samples

    def tune_resolution(self, res: int):
        """
        Adjusts the resolution to a value suitable for the meshing process.

        This method calculates a resolution that is a power of two, ensuring compatibility with
        the multi-resolution filling process used in mesh generation.

        Parameters:
        - res (int): The initial resolution.

        Returns:
        - int: The adjusted resolution, suitable for the meshing process.
        """
        counter = 1
        while res > self.min_res:
            res = res // 2
            counter *= 2
        return res * counter

    @staticmethod
    def get_res_samples(res):
        """
        Generates a grid of samples at a specified resolution.

        This method creates a tensor of sample coordinates for a cubic grid at the given resolution,
        along with an additional dimension initialized to ones.

        Parameters:
        - res (int): The resolution of the grid.

        Returns:
        - torch.Tensor: A tensor of shape (4, res^3), containing the grid coordinates and an additional dimension.
        """
        voxel_origin = torch.tensor([-1., -1., -1.])
        voxel_size = 2.0 / (res - 1)
        overall_index = torch.arange(0, res ** 3, 1, dtype=torch.int64)
        samples = torch.ones(4, res ** 3).detach()
        samples.requires_grad = False
        # transform first 3 columns
        # to be the x, y, z index
        div_1 = torch.div(overall_index, res, rounding_mode='floor')
        samples[2, :] = (overall_index % res).float()
        samples[1, :] = (div_1 % res).float()
        samples[0, :] = (torch.div(div_1, res, rounding_mode='floor') % res).float()
        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:3] = samples[:3] * voxel_size + voxel_origin[:, None]
        # samples[0, :] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        # samples[1, :] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        # samples[2, :] = (samples[:, 2] * voxel_size) + voxel_origin[0]
        return samples

    def register_resolution(self, res: int):
        """
        Prepares and caches samples at a given resolution for meshing.

        This method tunes the resolution, generates samples if they are not already cached, and 
        prepares them for the meshing process.

        Parameters:
        - res (int): The desired resolution.

        Returns:
        - (torch.Tensor, int): A tuple containing the samples at the adjusted resolution and the resolution value.
        """
        res = self.tune_resolution(res)
        if res not in self.sample_cache:
            samples = self.get_res_samples(res)
            samples = samples.to(self.device)
            self.sample_cache[res] = samples
        else:
            samples = self.sample_cache[res]
            samples[3, :] = 1
        return samples, res

    def get_grid(self, decoder, res):
        """
        Generates an occupancy grid using the provided decoder at a specified resolution.

        This function sets up a grid of samples, applies recursive filling to obtain occupancy values,
        and reshapes the resulting data to form a 3D occupancy grid.

        Parameters:
        - decoder: A function or model that returns occupancy values.
        - res (int): The resolution for the occupancy grid.

        Returns:
        - torch.Tensor: A 3D tensor representing the occupancy grid at the specified resolution.
        """
        stride = 2
        samples, res = self.register_resolution(res)
        depth = int(np.ceil(np.log2(res) - np.log2(self.min_res)))
        samples = self.fill_recursive(decoder, samples, stride, res, depth)
        occ_values = samples[3]
        occ_values = occ_values.reshape(res, res, res)
        return occ_values

    def occ_meshing(self, decoder, res: int = 256, get_time: bool = False, verbose=False):
        """
        Generates a mesh from occupancy values using the Marching Cubes algorithm.

        This method computes an occupancy grid using the provided decoder, then applies the
        Marching Cubes algorithm to generate a mesh from this grid. Timing information can be
        optionally printed or returned.

        Parameters:
        - decoder: A function or model that returns occupancy values.
        - res (int, optional): The resolution for mesh generation. Defaults to 256.
        - get_time (bool, optional): Whether to return the time taken for the entire meshing process.
        - verbose (bool, optional): Whether to print timing information.

        Returns:
        - Mesh data as generated by the marching cubes algorithm, and optionally the time taken.
        """
        start = time.time()
        voxel_origin = [-1., -1., -1.]
        voxel_size = 2.0 / (res - 1)
        occ_values = self.get_grid(decoder, res)
        if verbose:
            end = time.time()
            print("sampling took: %f" % (end - start))
            if get_time:
                return end - start

        mesh_a = mcubes_skimage(occ_values.data.cpu(), voxel_origin, voxel_size)
        # mesh_a = mcubes_torch(occ_values, voxel_origin, voxel_size)

        if verbose:
            end_b = time.time()
            print("mcube took: %f" % (end_b - end))
            print("meshing took: %f" % (end_b - start))
        return mesh_a

    def __init__(self, device: D, max_batch: int = 64 ** 3, min_res: int = 64, scale: float = 1, verbose: bool = False):
        """
        Initializes the MarchingCubesMeshing class with the specified parameters.

        Parameters:
        - device (D): The device on which to perform computations.
        - max_batch (int, optional): The maximum batch size for processing samples. Defaults to 64^3.
        - min_res (int, optional): The minimum resolution for mesh generation. Defaults to 64.
        - scale (float, optional): Scaling factor for the decoder. Defaults to 1.
        - verbose (bool, optional): Whether to enable verbose logging.
        """
        self.device = device
        self.max_batch = 32 ** 3 if constants.IS_WINDOWS else max_batch
        self.min_res = min_res
        self.scale = scale
        self.verbose = verbose
        self.sample_cache = {}


def create_mesh_old(decoder, res=256, max_batch=64 ** 3, scale=1, device=CPU, verbose=False, get_time: bool = False):
    """
    Legacy function to create a mesh using the Marching Cubes algorithm.

    This function sets up the meshing environment, prepares samples, fills them with occupancy 
    values using the provided decoder, and then applies the Marching Cubes algorithm to generate the mesh.

    Parameters:
    - decoder: A function or model that returns occupancy values.
    - res (int, optional): The resolution of the mesh. Defaults to 256.
    - max_batch (int, optional): The maximum batch size for processing samples. Defaults to 64 ** 3.
    - scale (float, optional): Scaling factor for the decoder. Defaults to 1.
    - device: The device to perform computations on. Defaults to CPU.
    - verbose (bool, optional): Whether to print timing information. Defaults to False.
    - get_time (bool, optional): Whether to return the time taken for sampling. Defaults to False.

    Returns:
    - Mesh data as generated by the marching cubes algorithm.
    """
    meshing = MarchingCubesMeshing(device, max_batch=max_batch, scale=scale, verbose=verbose)
    start = time.time()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (res - 1)

    overall_index = torch.arange(0, res ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(res ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % res
    samples[:, 1] = (overall_index.long() // res) % res
    samples[:, 0] = ((overall_index.long() // res) // res) % res

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    samples = meshing.fill_samples(decoder, samples, device=device)
    sdf_values = samples[:, 3]
    # return sdf_values, samples[:, :3]
    sdf_values = sdf_values.reshape(res, res, res)

    end = time.time()
    print("sampling took: %f" % (end - start))
    if get_time:
        return end - start
    return mcubes_skimage(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
    )