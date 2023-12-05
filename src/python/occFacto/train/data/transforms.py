import numpy as np

# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out

class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

        return data_out
    

class SubsamplePointcloudHalf(object):
    ''' Point cloud subsampling transformation class with first half on the surface and second half outside 

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N
        self.tol = {
            'mean': 0.01,
            'std': 0.005,
            'min': 0.0005,
            'max': 0.025
        }

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        # Randomly sample indices for generating points that are inside / outside
        indices_in = np.random.randint(points.shape[0], size=self.N // 2)
        indices_out = np.random.randint(points.shape[0], size=self.N // 2)

        # Add a noise to the points via normals
        tol = np.maximum(self.tol['min'], np.minimum(np.random.normal(self.tol['mean'], self.tol['std'], self.N // 2), self.tol['max']))
        points_out = points[indices_out, :] + tol[:, np.newaxis] * normals[indices_out, :]

        # Get the points in the surface cover
        points_in = points[indices_in, :]

        # Concatenate the norms and points where first half is inside shape and second half is outside shape
        pts = np.concatenate((points_in, points_out), axis=0)
        occs = np.concatenate((np.ones(self.N // 2), np.zeros(self.N // 2)))

        idx_shuffle = np.random.permutation(self.N)
        points = points[idx_shuffle, :]
        occs = occs[idx_shuffle]

        data_out.update({
                None: points,
                'occ':  occs,
            })

        return data_out


class SubsamplePointsHalf(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']

        # Mask for Pos
        mask_p = occ == 1
        mask_n = occ == 0

        # Index pos and neg
        pos = occ[mask_p]
        neg = occ[mask_n]

        # Points Pos and Neg
        points_p = points[mask_p, :]
        points_n = points[mask_n, :]
        
        data_out = data.copy()

        idx_p = np.random.randint(pos.shape[0], size=self.N // 2)
        idx_n = np.random.randint(neg.shape[0], size=self.N // 2)

        points_p = points_p[idx_p, :]
        points_n = points_n[idx_n, :]
        points = np.concatenate((points_p, points_n), axis=0)

        occs_p = pos[idx_p]
        occs_n = neg[idx_n]
        occs = np.concatenate((occs_p, occs_n))

        idx_shuffle = np.random.permutation(self.N)
        points = points[idx_shuffle, :]
        occs = occs[idx_shuffle]

        data_out.update({
                None: points,
                'occ':  occs,
            })

        return data_out

        '''

        data_out.update({
            None: points[idx, :],

            'occ':  occ[idx],
        })


        if isinstance(self.N, int):

            idx = np.random.randnint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out
       ''' 

class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out