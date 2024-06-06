import functools
import numpy as np
from scipy.ndimage import map_coordinates
import array


@functools.lru_cache()
def fisheye_uv2xyzTest(p0, p1, w, scale):
    u = p0 - w / 2
    v = p1 - w / 2
    r = np.sqrt((u ** 2 + v ** 2))
    phi = np.arcsin(r / (w / 2))
    theta = np.arctan2(v, u)

    z = r / np.tan(phi)
    y = r * np.sin(theta)
    x = r * np.cos(theta)
    phi1 = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    theta1 = np.arctan2(y, x)
    u1 = w / 2 * np.sin(phi1 * scale) * np.cos(theta1)
    v1 = w / 2 * np.sin(phi1 * scale) * np.sin(theta1)
    u1 = w / 2 + u1
    v1 = w / 2 + v1
    return u1, v1


def fisheye_uv2xyz(p0, p1, w):
    u = p0 - w / 2
    v = p1 - w / 2
    r = np.sqrt((u ** 2 + v ** 2))
    phi = np.arcsin(r / (w / 2))
    theta = np.arctan2(v, u)

    z = r / np.tan(phi)
    y = r * np.sin(theta)
    x = r * np.cos(theta)
    phi1 = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    theta1 = np.arctan2(y, x)
    u1 = w / 2 * np.sin(phi1) * np.cos(theta1)
    v1 = w / 2 * np.sin(phi1) * np.sin(theta1)
    u1 = w / 2 + u1
    v1 = w / 2 + v1
    return phi1, theta1, x, y, z


def pano_connect_points(p1, p2, w=1024, h=1024):
    if p1[0] == p2[0]:
        print("U值一样！！！！！")
    u1, v1 = fisheye_uv2xyzTest(p1[0], p1[1], w)
    u2, v2 = fisheye_uv2xyzTest(p2[0], p2[1], w)

    phi1, theta1, x1, y1, z1 = fisheye_uv2xyz(p1[0], p1[1], w)
    phi2, theta2, x2, y2, z2 = fisheye_uv2xyz(p2[0], p2[1], w)
    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1
    if theta2 > (-np.pi) and theta2 < (-np.pi / 2) and theta1 > np.pi / 2 and theta1 < np.pi:
        theta2 = theta2 + 2 * np.pi
    delta_theta = 1 / ((w / 2) * (theta2 - theta1))
    thetas = np.arange(theta1, theta2, delta_theta)
    ps = (np.tan(thetas) * x1 - y1) / (vy - np.tan(thetas) * vx)
    rs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    phis = np.arcsin(rs / (np.sqrt(rs ** 2 + (z1 + ps * vz) ** 2)))

    coorus = w / 2 * np.sin(phis * (3 / 4)) * np.cos(thetas)
    coorvs = w / 2 * np.sin(phis * (3 / 4)) * np.sin(thetas)
    coorus = w / 2 + coorus
    coorvs = w / 2 + coorvs
    return np.stack([coorus, coorvs], axis=-1), u1, v1, u2, v2


def fisheye_connect_points(p1, p2, scale):
    w = 1024
    if p1[0] == p2[0]:
        print("U值一样！！！！！")
    u1, v1 = fisheye_uv2xyzTest(p1[0], p1[1], w, scale)
    u2, v2 = fisheye_uv2xyzTest(p2[0], p2[1], w, scale)

    phi1, theta1, x1, y1, z1 = fisheye_uv2xyz(p1[0], p1[1], w)
    phi2, theta2, x2, y2, z2 = fisheye_uv2xyz(p2[0], p2[1], w)
    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1
    if theta2 > (-np.pi) and theta2 < (-np.pi / 2) and theta1 > np.pi / 2 and theta1 < np.pi:
        theta2 = theta2 + 2 * np.pi
    delta_theta = 1 / ((w / 2) * (theta2 - theta1))
    thetas = np.arange(theta1, theta2, delta_theta)
    ps = (np.tan(thetas) * x1 - y1) / (vy - np.tan(thetas) * vx)
    rs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    phis = np.arcsin(rs / (np.sqrt(rs ** 2 + (z1 + ps * vz) ** 2)))

    coorus = w / 2 * np.sin(phis * scale) * np.cos(thetas)
    coorvs = w / 2 * np.sin(phis * scale) * np.sin(thetas)
    coorus = w / 2 + coorus
    coorvs = w / 2 + coorvs
    # return np.stack([coorus, coorvs], axis=-1), u1, v1, u2, v2 # for train
    return np.stack([coorus, coorvs], axis=-1)  # for evaluation


def pano_stretch(img, corners, kx, ky, order=1):
    '''
    img:     [H, W, C]
    corners: [N, 2] in image coordinate (x, y) format
    kx:      Stretching along front-back direction
    ky:      Stretching along left-right direction
    order:   Interpolation order. 0 for nearest-neighbor. 1 for bilinear.
    '''

    # Process image
    sin_u, cos_u, tan_v = uv_tri(img.shape[1], img.shape[0])
    u0 = np.arctan2(sin_u * kx / ky, cos_u)
    v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

    refx = (u0 / np.pi + 0.5) * img.shape[1] - 0.5
    refy = (v0 / np.pi + 0.5) * img.shape[0] - 0.5

    # [TODO]: using opencv remap could probably speedup the process a little
    stretched_img = np.stack([
        map_coordinates(img[..., i], [refy, refx], order=order, mode='wrap')
        for i in range(img.shape[-1])
    ], axis=-1)

    # Process corners
    corners_u0 = coorx2u(corners[:, 0], img.shape[1])
    corners_v0 = coory2v(corners[:, 1], img.shape[0])
    corners_u = np.arctan2(np.sin(corners_u0) * ky / kx, np.cos(corners_u0))
    corners_v = np.arctan(np.tan(corners_v0) * np.sin(corners_u) / np.sin(corners_u0) / ky)
    cornersX = u2coorx(corners_u, img.shape[1])
    cornersY = v2coory(corners_v, img.shape[0])
    stretched_corners = np.stack([cornersX, cornersY], axis=-1)

    return stretched_img, stretched_corners


def visualize_pano_stretch(stretched_img, stretched_cor, title):
    '''
    Helper function for visualizing the effect of pano_stretch
    '''
    thikness = 2
    color = (0, 255, 0)
    for i in range(4):
        xys = pano_connect_points(stretched_cor[i * 2], stretched_cor[(i * 2 + 2) % 8], z=-50)
        xys = xys.astype(int)
        blue_split = np.where((xys[1:, 0] - xys[:-1, 0]) < 0)[0]
        if len(blue_split) == 0:
            cv2.polylines(stretched_img, [xys], False, color, 2)
        else:
            t = blue_split[0] + 1
            cv2.polylines(stretched_img, [xys[:t]], False, color, thikness)
            cv2.polylines(stretched_img, [xys[t:]], False, color, thikness)

    for i in range(4):
        xys = pano_connect_points(stretched_cor[i * 2 + 1], stretched_cor[(i * 2 + 3) % 8], z=50)
        xys = xys.astype(int)
        blue_split = np.where((xys[1:, 0] - xys[:-1, 0]) < 0)[0]
        if len(blue_split) == 0:
            cv2.polylines(stretched_img, [xys], False, color, 2)
        else:
            t = blue_split[0] + 1
            cv2.polylines(stretched_img, [xys[:t]], False, color, thikness)
            cv2.polylines(stretched_img, [xys[t:]], False, color, thikness)

    cv2.putText(stretched_img, title, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2, cv2.LINE_AA)

    return stretched_img.astype(np.uint8)


if __name__ == '__main__':
    import argparse
    import time
    from PIL import Image
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', default='data/valid/img/pano_abpohapclcyuuz.png')
    parser.add_argument('--i_gt', default='data/valid/label_cor/pano_abpohapclcyuuz.txt')
    parser.add_argument('--o', default='sample_stretched_pano.png')
    parser.add_argument('--kx', default=2, type=float,
                        help='Stretching along front-back direction')
    parser.add_argument('--ky', default=1, type=float,
                        help='Stretching along left-right direction')
    args = parser.parse_args()

    img = np.array(Image.open(args.i), np.float64)
    with open(args.i_gt) as f:
        cor = np.array([line.strip().split() for line in f], np.int32)
    stretched_img, stretched_cor = pano_stretch(img, cor, args.kx, args.ky)

    title = 'kx=%3.2f, ky=%3.2f' % (args.kx, args.ky)
    visual_stretched_img = visualize_pano_stretch(stretched_img, stretched_cor, title)
    Image.fromarray(visual_stretched_img).save(args.o)
