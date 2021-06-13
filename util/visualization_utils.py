import k3d
import numpy as np
import trimesh

'''
# TODO: Didn't work, might try later
def visualize_pointcloud(point_cloud, point_size=1, colors=None, flip_axes=False, name='point_cloud'):
    plot = k3d.plot(name=name, grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
    plt_points = k3d.points(positions=point_cloud.astype(np.float32), point_size=point_size, colors=colors if colors is not None else [], color=0xd0d0d0)
    plot += plt_points
    plt_points.shader = '3d'
    return plot.display()
'''
def visualize_pointcloud(point_cloud):
    pc = trimesh.points.PointCloud(point_cloud)
    trimesh.Scene(pc).show()