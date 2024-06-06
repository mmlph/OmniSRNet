import numpy as np
import time
from scipy.ndimage import map_coordinates
import json
import open3d
from PIL import Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import functools
from multiprocessing import Pool

from util.omni_post_proc import np_coor2xy_fisheye


def xyz_2_coorxy(xs, ys, zs, H, W):
    us = np.arctan2(xs, -ys)
    vs = -np.arctan(zs / np.sqrt(xs**2 + ys**2))
    coorx = (us / (2 * np.pi) + 0.5) * W
    coory = (vs / np.pi + 0.5) * H
    return coorx, coory


# 鱼眼从空间xyz坐标转换成像素坐标
def xyz_2_coorxy_fisheye(xs, ys, zs, W):
    ys = np.array(ys)
    xs = np.array(xs)
    # zs = np.ones_like(xs)*(-1.6)
    theta = np.arctan2(ys, xs)
    # r = xs / np.cos(theta)
    # phi = np.arctan2(zs, r)
    phi = np.arctan2(np.sqrt(xs**2 + ys**2), zs)
    coorx = W / 2 * np.sin(phi) * np.cos(theta)
    coory = W / 2 * np.sin(phi) * np.sin(theta)
    coorx = W / 2 + coorx
    coory = W / 2 + coory

    return coorx, coory

def pt_in_poly(poly, pt):
    return poly.contains(Point(pt))


def warp_walls(xy, floor_z, ceil_z, H, W, ppm, alpha):
    all_rgba = []
    all_xyz = []
    for i in range(len(xy)):
        next_i = (i + 1) % len(xy)
        xy_a = xy[i]
        xy_b = xy[next_i]
        xy_w = np.sqrt(((xy_a - xy_b)**2).sum())
        # t_h = int(round((ceil_z - floor_z) * ppm))
        t_h = int(round((floor_z) * ppm))
        t_w = int(round(xy_w * ppm))
        xs = np.linspace(xy_a[0], xy_b[0], t_w)[None].repeat(t_h, 0)
        ys = np.linspace(xy_a[1], xy_b[1], t_w)[None].repeat(t_h, 0)
        zs = np.linspace(floor_z, ceil_z, t_h)[:, None].repeat(t_w, 1)
        coorx, coory = xyz_2_coorxy_fisheye(xs, ys, zs, W)

        plane_texture = np.stack([
            map_coordinates(fisheye_texture[..., 0], [coory, coorx], order=1, mode='wrap'),
            map_coordinates(fisheye_texture[..., 1], [coory, coorx], order=1, mode='wrap'),
            map_coordinates(fisheye_texture[..., 2], [coory, coorx], order=1, mode='wrap'),
            np.zeros([t_h, t_w]) + alpha,
        ], -1)
        plane_xyz = np.stack([xs, ys, zs], axis=-1)

        all_rgba.extend(plane_texture.reshape(-1, 4))
        all_xyz.extend(plane_xyz.reshape(-1, 3))

    return all_rgba, all_xyz


def warp_floor_ceiling(xy, z_floor, z_ceiling, H, W, ppm, alpha, n_thread):
    min_x = xy[:, 0].min()
    max_x = xy[:, 0].max()
    min_y = xy[:, 1].min()
    max_y = xy[:, 1].max()
    t_h = int(round((max_y - min_y) * ppm))
    t_w = int(round((max_x - min_x) * ppm))
    xs = np.linspace(min_x, max_x, t_w)[None].repeat(t_h, 0)
    ys = np.linspace(min_y, max_y, t_h)[:, None].repeat(t_w, 1)
    zs_floor = np.zeros_like(xs) + z_floor
    zs_ceil = np.zeros_like(xs) + z_ceiling
    coorx_floor, coory_floor = xyz_2_coorxy_fisheye(xs, ys, zs_floor, W)
    coorx_ceil, coory_ceil = xyz_2_coorxy_fisheye(xs, ys, zs_ceil, W)

    floor_texture = np.stack([
        map_coordinates(fisheye_texture[..., 0], [coory_floor, coorx_floor], order=1, mode='wrap'),
        map_coordinates(fisheye_texture[..., 1], [coory_floor, coorx_floor], order=1, mode='wrap'),
        map_coordinates(fisheye_texture[..., 2], [coory_floor, coorx_floor], order=1, mode='wrap'),
        np.zeros([t_h, t_w]) + alpha,
    ], -1).reshape(-1, 4)
    floor_xyz = np.stack([xs, ys, zs_floor], axis=-1).reshape(-1, 3)

    ceil_texture = np.stack([
        map_coordinates(fisheye_texture[..., 0], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        map_coordinates(fisheye_texture[..., 1], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        map_coordinates(fisheye_texture[..., 2], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        np.zeros([t_h, t_w]) + alpha,
    ], -1).reshape(-1, 4)
    ceil_xyz = np.stack([xs, ys, zs_ceil], axis=-1).reshape(-1, 3)

    xy_poly = Polygon(xy)
    with Pool(n_thread) as p:
        sel = p.map(functools.partial(pt_in_poly, xy_poly), floor_xyz[:, :2])

    return floor_texture[sel], floor_xyz[sel], ceil_texture[sel], ceil_xyz[sel]


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', default='F:\Code\LayoutEstimation\Structured3D_17_scene_03482_239_full_rgb_rawlight_fish.png',
                        help='Fisheye Image')
    parser.add_argument('--layout', default='F:\Code\LayoutEstimation\Structured3D_17_scene_03482_239_full_rgb_rawlight_fishGT.txt',
                        help='Txt file containing layout corners (cor_id)')
    parser.add_argument('--camera_height', default=1.6, type=float,
                        help='Camera height in meter (not the viewer camera)')
    parser.add_argument('--ppm', default=120, type=int,
                        help='Points per meter')
    parser.add_argument('--point_size', default=0.0025, type=int,
                        help='Point size')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Opacity of the texture')
    parser.add_argument('--threads', default=10, type=int,
                        help='Number of threads to use')
    parser.add_argument('--ignore_floor', action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_ceiling', action='store_true',
                        help='Skip rendering ceiling')
    args = parser.parse_args()
    time_start = time.time()
    # Reading source (texture img, cor_id txt)
    fisheye_texture = np.array(Image.open(args.img)) / 255.0

    with open(args.layout) as f:
        dt = json.load(f)
        cor_id = np.array(dt['uv'], np.float32)

    # with open(args.layout) as f:
    #     cor_id = np.array([line.split() for line in f], np.float32)

    # Convert cor_id to 3d xyz
    N = len(cor_id) // 2
    H, W = fisheye_texture.shape[:2]
    # floor_z = -args.camera_height
    floor_xy = np_coor2xy_fisheye(cor_id, W)
    floor_z = floor_xy[:, 2].mean()

    # Warp each wall
    all_rgba, all_xyz = warp_walls(floor_xy, floor_z, 0, H, W, args.ppm, args.alpha)

    # Warp floor and ceiling
    if not args.ignore_floor or args.ignore_ceiling:
        fi, fp, ci, cp = warp_floor_ceiling(floor_xy, floor_z, 0, H, W,
                                            ppm=args.ppm,
                                            alpha=args.alpha,
                                            n_thread=args.threads)

        if not args.ignore_floor:
            all_rgba.extend(fi)
            all_xyz.extend(fp)

        if args.ignore_ceiling:
            all_rgba.extend(ci)
            all_xyz.extend(cp)

    # Launch point cloud viewer
    print('# of points:', len(all_rgba))
    all_xyz = np.array(all_xyz)
    all_rgb = np.array(all_rgba)[:, :3]
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(all_xyz)
    pcd.colors = open3d.Vector3dVector(all_rgb)
    open3d.draw_geometries([pcd])
    open3d.write_point_cloud("F:\Code\LayoutEstimation/temp.ply",pcd)
    time_end = time.time()
    print('Train time:', time_end - time_start)