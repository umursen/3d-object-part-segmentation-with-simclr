import numpy as np
import scipy
from scipy.linalg import expm, norm


def rotx(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def check_aspect(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2]) / np.max(crop_range[:2])
    xz_aspect = np.min(crop_range[[0, 2]]) / np.max(crop_range[[0, 2]])
    yz_aspect = np.min(crop_range[1:]) / np.max(crop_range[1:])
    return (xy_aspect >= aspect_min) or (xz_aspect >= aspect_min) or (yz_aspect >= aspect_min)

class RandomAugmentation(object):
    def __init__(self, p):
        self.p = p

    def apply_by_chance(self):
        prob = np.random.random_sample()
        return prob < self.p


class Rescale(RandomAugmentation):
    """
      Scales the object. The scale value is sampled from a normal distribution

      Parameters
      ----------
      points (ndarray): 3D object

      Returns
      -------
      out (ndarray) : Augmentated 3D object
      """

    def __init__(self, p, mean=0.7, variance=0.05):
        super().__init__(p)
        self.mean = mean
        self.variance = variance

    def __call__(self, input):
        if not self.apply_by_chance():
            return input

        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        # scale = np.random.normal(self.mean, self.variance, 1).astype(np.float32)
        # points_temp = points_temp * scale

        ##FROM DeepContrast
        points_temp[:, 0:3] = points_temp[:, 0:3] * np.random.uniform(0.8, 1.2)

        return {'point': points_temp, 'seg': seg}


class Flip(RandomAugmentation):
    """
    Flip the object over x or y axis.

    Parameters
    ----------
    points (ndarray): 3D object

    Returns
    -------
    out (ndarray) : Augmentated 3D object
    """

    def __init__(self, p):
        super().__init__(p)

    def __call__(self, input):
        if not self.apply_by_chance():
            return input

        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        # index = np.random.choice(3, 1)
        # points_temp[:, index] = -points_temp[:, index]

        ##FROM DEEP CONTRAST
        #TODO:FLIP ALWAYS SOMEHOW
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            points_temp[:, 0] = -1 * points_temp[:, 0]
        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            points_temp[:, 1] = -1 * points_temp[:, 1]

        return {'point': points_temp, 'seg': seg}


class GaussianWhiteNoise(RandomAugmentation):
    """
    Flip the object over x or y axis.

    Parameters
    ----------
    points (ndarray): 3D object

    Returns
    -------
    out (ndarray) : Augmentated 3D object
    """

    def __init__(self, p, noise_mu=0, noise_sigma= 0.0075):
        super().__init__(p)
        self.GAUSS_NOISE_MU = noise_mu
        self.GAUSS_NOISE_SIGMA = noise_sigma

    def __call__(self, input):
        if not self.apply_by_chance():
            return input

        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        noise_point = np.random.normal(self.GAUSS_NOISE_MU, self.GAUSS_NOISE_SIGMA, points_temp.shape)\
            .astype(np.float32)

        points_temp = points_temp + noise_point

        return {'point': points_temp, 'seg': seg}


# class CutOut(RandomAugmentation):
#     """
#     Flip the object over x or y axis.
#
#     Parameters
#     ----------
#     points (ndarray): 3D object
#
#     Returns
#     -------
#     out (ndarray) : Augmentated 3D object
#     """
#
#     def __init__(self, p, cut_ratio=0.1):
#         super().__init__(p)
#         self.cut_ratio = cut_ratio
#
#     def __call__(self, input):
#         if not self.apply_by_chance():
#             return input
#
#         point_cloud, seg = input['point'], input['seg']
#
#         points_temp = point_cloud.copy()
#         seg_temp = seg.copy()
#
#         for axis in range(3):
#             max_val, min_val = np.max(points_temp.T[axis]), np.min(points_temp.T[axis])
#             cut_len = np.abs(max_val - min_val) * self.cut_ratio
#             start_pos = np.random.uniform(min_val, max_val - cut_len, 1)
#             positions = (start_pos, start_pos + cut_len)
#             print(cut_len)
#             indices = np.where((points_temp.T[axis] > positions[0]) & (points_temp.T[axis] < positions[1]))
#             points_temp = np.delete(points_temp, indices, axis=0)
#             seg_temp = np.delete(seg_temp, indices, axis=0)
#
#         return {'point': points_temp, 'seg': seg_temp}


class Rotation(RandomAugmentation):
    """
    Flip the object over x or y axis.

    Parameters
    ----------
    points (ndarray): 3D object

    Returns
    -------
    out (ndarray) : Augmentated 3D object
    """
    def __init__(self, p):
        super().__init__(p)

    def __call__(self, input):
        if not self.apply_by_chance():
            return input

        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        seg_temp = seg.copy()

        ##FROM DepthContrast
        rot_angle = (np.random.random() * np.pi * 2) - np.pi  # -5 ~ +5 degree
        if np.random.random() <= 0.33:
            rot_mat = rotx(rot_angle)
        elif np.random.random() <= 0.66:
            rot_mat = roty(rot_angle)
        else:
            rot_mat = rotz(rot_angle)

        points_temp[:, 0:3] = np.dot(points_temp[:, 0:3], np.transpose(rot_mat))

        return {'point': points_temp, 'seg': seg_temp}



class RandomCuboid(RandomAugmentation):
    """
    Flip the object over x or y axis.

    Parameters
    ----------
    points (ndarray): 3D object

    Returns
    -------
    out (ndarray) : Augmentated 3D object
    """

    def __init__(self, p, random_crop=True, crop=0.5, randcrop=1, aspect=0.75, dist_sample=True, npoints=1):
        super().__init__(p)
        self.random_crop = random_crop
        self.crop = crop
        self.randcrop = randcrop
        self.aspect = aspect
        self.dist_sample=dist_sample
        self.npoints=npoints

    def __call__(self, input):
        if not self.apply_by_chance():
            return input

        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        seg_temp = seg.copy()

        range_xyz = np.max(points_temp[:, 0:3], axis=0) - np.min(points_temp[:, 0:3], axis=0)
        if self.random_crop:
            crop_range = float(self.crop) + np.random.rand(3) * (
                        float(self.randcrop) - float(self.crop))
            if self.aspect:
                loop_count = 0
                while not check_aspect(crop_range, float(self.aspect)):
                    loop_count += 1
                    crop_range = float(self.crop) + np.random.rand(3) * (
                                float(self.randcrop) - float(self.crop))
                    if loop_count > 100:
                        break
        else:
            crop_range = float(self.crop)

        skip_step = False
        loop_count = 0

        ### Optional for depth selection croption
        if self.dist_sample:
            numb, numv = np.histogram(points_temp[:, 2])
            max_idx = np.argmax(numb)
            minidx = max(0, max_idx - 2)
            maxidx = min(len(numv) - 1, max_idx + 2)
            range_v = [numv[minidx], numv[maxidx]]
        while True:
            loop_count += 1

            sample_center = points_temp[np.random.choice(len(points_temp)), 0:3]
            if self.dist_sample:
                if (loop_count <= 100):
                    if (sample_center[-1] <= range_v[1]) and (sample_center[-1] >= range_v[0]):
                        continue

            new_range = range_xyz * crop_range / 2.0

            max_xyz = sample_center + new_range
            min_xyz = sample_center - new_range

            upper_idx = np.sum((points_temp[:, 0:3] <= max_xyz).astype(np.int32), 1) == 3
            lower_idx = np.sum((points_temp[:, 0:3] >= min_xyz).astype(np.int32), 1) == 3

            new_pointidx = (upper_idx) & (lower_idx)

            if (loop_count > 100) or (np.sum(new_pointidx) > float(self.npoints)):
                break

        # print ("final", np.sum(new_pointidx))
        points_temp = points_temp[new_pointidx, :]
        seg_temp = seg_temp[new_pointidx]

        return {'point': points_temp, 'seg': seg_temp}


class RandomDrop(RandomAugmentation):
    """
    Flip the object over x or y axis.

    Parameters
    ----------
    points (ndarray): 3D object

    Returns
    -------
    out (ndarray) : Augmentated 3D object
    """

    def __init__(self, p, crop=0.2, dist_sample=False):
        super().__init__(p)
        self.crop = crop
        self.dist_sample = dist_sample

    def __call__(self, input):
        if not self.apply_by_chance():
            return input

        point_cloud, seg = input['point'], input['seg']

        points_temp = point_cloud.copy()
        seg_temp = seg.copy()

        range_xyz = np.max(points_temp[:, 0:3], axis=0) - np.min(points_temp[:, 0:3], axis=0)

        crop_range = float(self.crop)
        new_range = range_xyz * crop_range / 2.0

        if self.dist_sample:
            numb, numv = np.histogram(points_temp[:, 2])
            max_idx = np.argmax(numb)
            minidx = max(0, max_idx - 2)
            maxidx = min(len(numv) - 1, max_idx + 2)
            range_v = [numv[minidx], numv[maxidx]]
        loop_count = 0
        # write_ply_color(point_cloud[:,:3], point_cloud[:,3:], "before.ply")
        while True:
            sample_center = points_temp[np.random.choice(len(points_temp)), 0:3]
            loop_count += 1
            if self.dist_sample:
                if (loop_count <= 100):
                    if (sample_center[-1] > range_v[1]) or (sample_center[-1] < range_v[0]):
                        continue
            break
        max_xyz = sample_center + new_range
        min_xyz = sample_center - new_range

        upper_idx = np.sum((points_temp[:, 0:3] < max_xyz).astype(np.int32), 1) == 3
        lower_idx = np.sum((points_temp[:, 0:3] > min_xyz).astype(np.int32), 1) == 3

        new_pointidx = ~((upper_idx) & (lower_idx))
        points_temp = points_temp[new_pointidx, :]
        seg_temp = seg_temp[new_pointidx]
        # write_ply_color(point_cloud[:,:3], point_cloud[:,3:], "after.ply")

        return {'point': points_temp, 'seg': seg_temp}