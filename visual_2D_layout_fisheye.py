import argparse
import os
from PIL import Image
import numpy as np
import cv2
from skimage import draw
import json
'''
generate cor and edge image
'''




# 在鱼眼图中检测直线段
def lsdWrap(img, LSD=None, **kwargs):
    if LSD is None:
        LSD = cv2.createLineSegmentDetector(**kwargs)
    if len(img.shape) == 3:
        img =cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 检测img中的线段
    lines, width, prec, nfa = LSD.detect(img)
    # 将检测的线段画在图像上
    edgMap = LSD.drawSegments(np.zeros_like(img), lines)[..., 2]
    return edgMap


def fisheye_uv2xyzTest(p0,p1, scale, w):
    u = p0 - w /2
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


def fisheye_uv2xyz(p0,p1,w):
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


def fisheye_connect_points(p1, p2, scale, w=1024, h=1024):
    # if p1[0] == p2[0]:
    #     print("U值一样！！！！！")
    u1, v1 = fisheye_uv2xyzTest(p1[0], p1[1], scale, w)
    u2, v2 = fisheye_uv2xyzTest(p2[0], p2[1], scale, w)

    phi1, theta1, x1, y1, z1 = fisheye_uv2xyz(p1[0],p1[1], w)
    phi2, theta2, x2, y2, z2 = fisheye_uv2xyz(p2[0],p2[1], w)
    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1
    if theta2 >= (-np.pi) and theta2 <= (-np.pi / 4) and theta1 >= np.pi / 2 and theta1 <= np.pi:
        theta2 = theta2 + 2 * np.pi

    if theta1 >= (-np.pi) and theta1 <= (-np.pi / 2) and theta2 >= np.pi / 2 and theta2 <= np.pi:
            theta1 = theta1 + 2 * np.pi

    if theta1 == theta2:
        if v2 != v1:
            coorvs = np.arange(v1, v2, (v2-v1)/500)
            coorus = np.ones_like(coorvs) * u1
        elif u2 != u1:
            coorus = np.arange(u1, u2, (u2-u1)/500)
            coorvs = np.ones_like(coorus) * v1
        return np.stack([coorus, coorvs], axis=-1), u1, v1, u2, v2
    else:
        delta_theta = 1 / ((w / 2) * (theta2 - theta1) )
        thetas = np.arange(theta1, theta2, delta_theta/4)
        if(len(thetas) < 150):
            delta_theta = (theta2-theta1)/1000
            thetas = np.arange(theta1, theta2, delta_theta)
        ps = (np.tan(thetas) * x1 - y1) / (vy - np.tan(thetas) * vx)
        rs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
        phis = np.arcsin(rs / (np.sqrt(rs ** 2 + (z1 + ps * vz) ** 2)))

        coorus = w / 2 * np.sin(phis * scale) * np.cos(thetas)
        coorvs = w / 2 * np.sin(phis * scale) * np.sin(thetas)
        coorus = w / 2 + coorus
        coorvs = w / 2 + coorvs
        return np.stack([coorus, coorvs], axis=-1), u1, v1, u2, v2


# 得到corner图
def get_cor():
    H, W = 1024, 1024
    path = '/home/pc/hdd/XLK/dataset_fisheye/Structured3D_Pro_full_500/valid/label_cor_SF/'
    cor_dir = '/home/pc/hdd/XLK/dataset_fisheye/Structured3D_Pro_full_500/valid/cor/'
    names = [name for name in os.listdir(os.path.join(path))]
    for name in names:
        if name[-3:] == 'txt':
            fname = path + '/' + name
            cor = np.loadtxt(fname, dtype=np.int)
            fisheyeI = cv2.imread('assert/background.png')
            n_cor = len(cor)
            # print(name)
            for i in range(n_cor):
                xys, u1, v1, u2, v2 = fisheye_connect_points(cor[i], cor[(i + 1) % n_cor], scale=1)
                # 将corner显示在图上
                rr, cc = draw.circle(v1, u1, 6)
                draw.set_color(fisheyeI, [rr, cc], [0, 0, 255])
                rr, cc = draw.circle(v2, u2, 6)
                draw.set_color(fisheyeI, [rr, cc], [0, 0, 255])
                cv2.namedWindow('img', 0)
                cv2.resizeWindow('img', 1024, 1024)
                cor_path = cor_dir + '/' + name[:-8] + '.png'
                # print(edge_path)
                cv2.imwrite(cor_path, fisheyeI[..., 2], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


# 得到edge图--单幅图测试
def get_edge():
    H, W = 1024, 1024
    # txt路径 & 可视化边图的路径
    path = 'K:\Fisheye\Pano2Fisheye\HorizonNet\Sun360\\test\label_cor_SF\pano_aabrsshujmskyq_fishGTSF.txt'
    edge_dir = 'K:\Fisheye\Pano2Fisheye\Structured3D_Pro_empty_noncuboid6_F\\test\label_cor_SF'
    names =[ name for name in os.listdir(os.path.join(path))]
    for name in names:
        if name[-3:] == 'txt':
            fname = path + '/' + name
            cor = np.loadtxt(fname, dtype=np.int)
            # Prepare 2d floor-wall boundary && Visualize the boundary in fisheye
            bon_floor_x, bon_floor_y = [], []
            cor_floor_x, cor_floor_y = [], []
            fisheyeI = cv2.imread('K:\Fisheye\Pano2Fisheye\HorizonNet\Sun360\\test\img\pano_aabrsshujmskyq_fish.png')
            n_cor = len(cor)
            print(name)
            for i in range(n_cor):
                xys, u1, v1, u2, v2 = fisheye_connect_points(cor[i], cor[(i + 1) % n_cor], scale=3/4)
                # for i in range(1):
                #     xys,u1, v1, u2, v2 = omni_stretch.pano_connect_points(cor[i+1],
                #                                            cor[i + 2])
                # 将corner显示在图上
                # rr, cc = draw.circle(v1, u1, 6)
                # draw.set_color(fisheyeI, [rr, cc], [0, 0, 255])
                # rr, cc = draw.circle(v2, u2, 6)
                # draw.set_color(fisheyeI, [rr, cc], [0, 0, 255])
                bon_floor_x.extend(xys[:, 0])
                bon_floor_y.extend(xys[:, 1])
                cor_floor_x.append(u1)
                cor_floor_y.append(v1)

            # bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
            bon_gt = np.zeros((2, len(bon_floor_x)))
            bon_gt[0] = bon_floor_x
            bon_gt[1] = bon_floor_y

            bon_gt[0] = bon_gt[0] - W / 2
            bon_gt[1] = bon_gt[1] - W / 2
            rs = np.sqrt((bon_gt[0] ** 2 + bon_gt[1] ** 2))
            phi = np.arcsin(rs / (W / 2))
            theta = np.arctan2(bon_gt[1], bon_gt[0])
            bon_gt[0] = phi
            bon_gt[1] = theta

            cor_gt = np.zeros((W, W))
            for i in range(len(cor_floor_x)):
                u = int(np.round(cor_floor_x[i]))
                v = int(np.round(cor_floor_y[i]))
                cor_gt[v][u] = 1

            for i in range(len(bon_floor_x)):
                rr, cc = draw.circle(bon_floor_y[i], bon_floor_x[i], 2)
                draw.set_color(fisheyeI, [rr, cc], [0, 255, 0])
            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1024, 1024)
            # cv2.imshow('img', fisheyeI)
            # bon_path = os.path.join(self.bon_dir, self.img_fnames[idx])
            edge_path = edge_dir + '/' + name[:-6] + '.png'
            # print(edge_path)
            cv2.imwrite(edge_path, fisheyeI, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # cv2.waitKey()

#得到真实Vs.预测的边  GT对应的TXT所在目录；画图之后存储的目录；原图的目录
def getEdge_GT():
    H, W = 1024, 1024
    # txt路径 & 可视化边图的路径
    path = 'K:\Fisheye\Pano2Fisheye\Structured3D_Pro_empty_noncuboid6_F\\test\label_cor'
    edge_dir = 'K:\Fisheye\Pano2Fisheye\Structured3D_Pro_empty_noncuboid6_F\\test\label_cor_SF'
    names = [name for name in os.listdir(os.path.join(path))]
    for name in names:
        if name[-3:] == 'txt': # For Horizonnet
        # if name[-3:] == 'txt':
            fname = path + '/' + name
            # imgpath = 'K:\Fisheye\Pano2Fisheye\Structured3D_Pro_empty_500\\test\img/'+name[:-6]+ '.png' #For Horizonnet
            imgpath = 'K:\Fisheye\Pano2Fisheye\Structured3D_Pro_empty_noncuboid6_F\\test\img/' + name[:-6] + '.png'
            cor = np.loadtxt(fname, dtype=np.int)
            # Prepare 2d floor-wall boundary && Visualize the boundary in fisheye
            bon_floor_x, bon_floor_y = [], []
            cor_floor_x, cor_floor_y = [], []
            fisheyeI = cv2.imread(imgpath)
            n_cor = len(cor)
            # print(name)
            for i in range(n_cor):
                xys, u1, v1, u2, v2 = fisheye_connect_points(cor[i], cor[(i + 1) % n_cor], scale=3 / 4)
                bon_floor_x.extend(xys[:, 0])
                bon_floor_y.extend(xys[:, 1])
                cor_floor_x.append(u1)
                cor_floor_y.append(v1)

            # bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
            bon_gt = np.zeros((2, len(bon_floor_x)))
            bon_gt[0] = bon_floor_x
            bon_gt[1] = bon_floor_y

            bon_gt[0] = bon_gt[0] - W / 2
            bon_gt[1] = bon_gt[1] - W / 2
            rs = np.sqrt((bon_gt[0] ** 2 + bon_gt[1] ** 2))
            phi = np.arcsin(rs / (W / 2))
            theta = np.arctan2(bon_gt[1], bon_gt[0])
            bon_gt[0] = phi
            bon_gt[1] = theta

            cor_gt = np.zeros((W, W))
            for i in range(len(cor_floor_x)):
                u = int(np.round(cor_floor_x[i]))
                v = int(np.round(cor_floor_y[i]))
                cor_gt[v][u] = 1

            for i in range(len(bon_floor_x)):
                rr, cc = draw.circle(bon_floor_y[i], bon_floor_x[i], 3)
                draw.set_color(fisheyeI, [rr, cc], [0, 255, 0])
            edge_path = edge_dir + '/' + name[:-6] + '.png'# For Finetune
            cv2.imwrite(edge_path, fisheyeI, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # cv2.waitKey()

def getEdge_GT_DT():
    H, W = 1024, 1024
    # 预测的json路径 & 可视化预测边图路径
    path = 'E:\Layout_Reconstruction\\2019-OmniNet\OmniNet-20191204\dump\ckpt_1e-4\Fisheye\Structured3D_Pro_simple_noncuboid6_F\OmniNet-stdconv\\resnet50_rnn\lin-4epoch\epoch_200\\test'
    edge_dir = 'K:\Fisheye\Pano2Fisheye\Structured3D_Pro_empty_noncuboid6_F\\test\label_cor_SF'
    names = [name for name in os.listdir(os.path.join(path))]
    for name in names:
        if name[-4:] == 'json':
            fname = path + '/' + name
            # imgpath = 'K:\Fisheye\Pano2Fisheye\Structured3D_Pro_empty_500\\test\label_cor_SF/'+name[:-5] + '.png' #For Horizonnet
            imgpath = 'K:\Fisheye\Pano2Fisheye\Structured3D_Pro_empty_noncuboid6_F\\test\label_cor_SF/' + name[:-5] + '.png'
            dt_cor = json.load(open(fname))
            dt_cor = np.array(dt_cor['uv'], np.float32)
            # Prepare 2d floor-wall boundary && Visualize the boundary in fisheye
            bon_floor_x, bon_floor_y = [], []
            cor_floor_x, cor_floor_y = [], []
            fisheyeI = cv2.imread(imgpath)
            n_cor = len(dt_cor)
            # print(name)
            for i in range(n_cor):
                xys, u1, v1, u2, v2 = fisheye_connect_points(dt_cor[i], dt_cor[(i + 1) % n_cor], scale=1)
                bon_floor_x.extend(xys[:, 0])
                bon_floor_y.extend(xys[:, 1])
                cor_floor_x.append(u1)
                cor_floor_y.append(v1)

            # bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
            bon_gt = np.zeros((2, len(bon_floor_x)))
            bon_gt[0] = bon_floor_x
            bon_gt[1] = bon_floor_y

            bon_gt[0] = bon_gt[0] - W / 2
            bon_gt[1] = bon_gt[1] - W / 2
            rs = np.sqrt((bon_gt[0] ** 2 + bon_gt[1] ** 2))
            phi = np.arcsin(rs / (W / 2))
            theta = np.arctan2(bon_gt[1], bon_gt[0])
            bon_gt[0] = phi
            bon_gt[1] = theta

            cor_gt = np.zeros((W, W))
            for i in range(len(cor_floor_x)):
                u = int(np.round(cor_floor_x[i]))
                v = int(np.round(cor_floor_y[i]))
                cor_gt[v][u] = 1

            for i in range(len(bon_floor_x)):
                rr, cc = draw.circle(bon_floor_y[i], bon_floor_x[i], 3)
                draw.set_color(fisheyeI, [rr, cc], [0,0, 255])
            edge_path = edge_dir + '/' + name[:-6] + 'SENon6F_200.png'
            cv2.imwrite(edge_path, fisheyeI, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    # dirty1 = ys.max()
    # dirty2 = int(y_small_first)*2-1
    # dirty = xs + ys / ys.max() * (int(y_small_first)*2-1)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    # assert np.all(np.diff(xs) > 0)
    return xs, ys



if __name__ == '__main__':
    # img = np.array(Image.open('assert/demo.png'))
    # get_cor()
    #可视化真实与预测的边图，先运getEdge_GT，在真实边图基础上画预测的-getEdge_GT_DT
    # getEdge_GT()
    getEdge_GT_DT()
    # get_edge()
    # cor_img = get_cor(img)
    # n = np.array(cor_img)
    # print(n.shape)
    # cor_img.show()


