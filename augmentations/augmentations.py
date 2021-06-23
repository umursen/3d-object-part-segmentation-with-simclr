import numpy as np
from scipy.linalg import expm, norm

class Rescale(object):
    """
      Scales the object. The scale value is sampled from a normal distribution

      Parameters
      ----------
      points (ndarray): 3D object

      Returns
      -------
      out (ndarray) : Augmentated 3D object
      """

    def __init__(self, mean=0.7, variance=0.05):
        self.mean = mean
        self.variance = variance

    def __call__(self, input):
        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        scale = np.random.normal(self.mean, self.variance, 1)
        points_temp = points_temp * scale

        return {'point': points_temp, 'seg': seg}


class Flip(object):
    """
    Flip the object over x or y axis.

    Parameters
    ----------
    points (ndarray): 3D object

    Returns
    -------
    out (ndarray) : Augmentated 3D object
    """

    def __init__(self):
        pass

    def __call__(self, input):
        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        index = np.random.choice(2, 1)
        points_temp[:, index] = -points_temp[:, index]

        return {'point': points_temp, 'seg': seg}


class GaussianWhiteNoise(object):
    """
    Flip the object over x or y axis.

    Parameters
    ----------
    points (ndarray): 3D object

    Returns
    -------
    out (ndarray) : Augmentated 3D object
    """

    def __init__(self, noise_mu=0, noise_sigma=0.015):
        self.GAUSS_NOISE_MU = noise_mu
        self.GAUSS_NOISE_SIGMA = noise_sigma

    def __call__(self, input):
        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        noise_point = np.random.normal(self.GAUSS_NOISE_MU, self.GAUSS_NOISE_SIGMA, points_temp.shape)

        points_temp = points_temp + noise_point

        return {'point': points_temp, 'seg': seg}

class CutOut(object):
    """
    Flip the object over x or y axis.

    Parameters
    ----------
    points (ndarray): 3D object

    Returns
    -------
    out (ndarray) : Augmentated 3D object
    """

    def __init__(self, cut_ratio=0.1):
        self.cut_ratio = cut_ratio

    def __call__(self, input):
        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        seg_temp = seg.copy()

        for axis in range(3):
            max_val, min_val = np.max(points_temp.T[axis]), np.min(points_temp.T[axis])
            cut_len = np.abs(max_val - min_val) * self.cut_ratio
            start_pos = np.random.uniform(min_val, max_val - cut_len, 1)
            positions = (start_pos, start_pos + cut_len)
            indices = np.where((points_temp.T[axis] > positions[0]) & (points_temp.T[axis] < positions[1]))
            points_temp = np.delete(points_temp, indices, axis=0)
            seg_temp = np.delete(seg_temp, indices, axis=0)

        return {'point': points_temp, 'seg': seg_temp}

class Rotation(object):
    """
    Flip the object over x or y axis.

    Parameters
    ----------
    points (ndarray): 3D object

    Returns
    -------
    out (ndarray) : Augmentated 3D object
    """

    def __init__(self):
        pass

    def __call__(self, input):
        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        seg_temp = seg.copy()

        axis, theta = [4, 4, 1], np.random.uniform(0, np.pi * 2, 1)
        rotation_matrix = expm(np.cross(np.eye(3), axis/norm(axis)*theta))
        points_temp = np.dot(rotation_matrix, points_temp.T).T

        return {'point': points_temp, 'seg': seg_temp}
