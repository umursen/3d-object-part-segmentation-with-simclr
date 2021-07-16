from textwrap import fill
import k3d
import numpy as np
import trimesh
import os
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def visualize_pointcloud(point_cloud, point_size=0.05, colors=None, flip_axes=False, name='point_cloud'):
    plot = k3d.plot(name=name, grid_visible=False)
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
    plt_points = k3d.points(positions=point_cloud.astype(np.float32), point_size=point_size, colors=colors if colors is not None else [], color=0xd0d0d0)
    plot += plt_points
    plt_points.shader = '3d'
    return plot.display()


def visualize_mesh(mesh, flip_axes=False):
    vertices = mesh.vertices
    faces = mesh.faces
    plot = k3d.plot(name='points', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        vertices[:, 2] = vertices[:, 2] * -1
        vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
    plt_mesh = k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=0xd0d0d0)
    plot += plt_mesh
    plt_mesh.shader = '3d'
    plot.display()

def plot_tsne_points(x,y,plot_save_suffix=''):
    if not os.path.isdir('plots'):
        os.makedirs('plots')
        
    plt.figure(figsize=(16,10))
    tsne_plot = sns.scatterplot(
        x=x, y=y,
        hue=y,
        palette=sns.color_palette("hls", num_classes),
        legend="full",
        alpha=0.3
    )
    tsne_plot.figure.savefig('plots/tsne_{}i_{}p_{}.png'.format(n_iter,perplexity,plot_save_suffix))

def visualize_tsne_with_pictures(x,y,imgs,n_images=500,S=4000,s=75,image_name='tsne_with_images',background=0):
    x = x-min(x)
    y = y-min(y)
    
    x = x/max(x)
    y = y/max(y)

    imgs = np.array(imgs)

    if n_images!=-1:
        choice = np.random.choice(len(x), n_images, replace=True).astype(int)
        x = x[choice]
        y = y[choice]
        imgs = imgs[choice]

    G = np.full((S,S,3), fill_value=background, dtype=np.uint8)
    for i,(x0, y0, img_path) in enumerate(zip(x, y,imgs)):
        if i%100 == 0:
            print('{}/{}'.format(i, n_images));
        a = np.ceil(x0*(S-s)+1)
        b = np.ceil(y0*(S-s)+1)
        a = int(a - (a-1)%s + 1)
        b = int(b - (b-1)%s + 1)

        if G[a,b,1] == background:
            I = Image.open(img_path)
            I = I.resize((s,s))
            I = np.asarray(I)
            G[a:a+s, b:b+s, :] = I;
        
    final_image = Image.fromarray(G, 'RGB')
    final_image.save('{}_n{}_S{}_s{}.png'.format(image_name,n_images,S,s))