import os
import numpy as np
from PIL import Image,ImageDraw
from pylab import imshow
import matplotlib.pyplot as pyplot
from pylab import *
from pylab import plot
from pylab import array
from pylab import title
import cv2
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
from skimage import draw,data
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

from util import omni_stretch


class OmniCorBonDataset(data.Dataset):
    '''
    See README.md for how to prepare the dataset.
    '''

    def __init__(self, root_dir,
                 flip=False, rotate=False, gamma=False, stretch=False,
                 p_base=0.96, max_stretch=2.0,
                 normcor=False, return_cor=False, return_path=False):
        self.img_dir = os.path.join(root_dir, 'img')
        self.cor_dir = os.path.join(root_dir, 'label_cor')
        self.bon_dir = os.path.join(root_dir, 'label_bon')
        self.img_fnames = sorted([
            fname for fname in os.listdir(self.img_dir)
            if fname.endswith('.jpg') or fname.endswith('.png')
        ])
        if 'fish' in self.img_fnames[0]: #Fisheye Dataset
            if 'Structured3D' in self.img_fnames[0]:
                #self.txt_fnames = ['%sfishGTCor.txt' % fname.split('full', 1)[0] for fname in self.img_fnames]
                self.txt_fnames = ['%sGT.txt' % fname[:-4] for fname in self.img_fnames]
            else: # For Horizonnet dataset
                self.txt_fnames = ['%sGT.txt' % fname[:-4] for fname in self.img_fnames]
        else: #Panorama Dataset
            if 'Structured3D' in self.img_fnames[0]:
                self.txt_fnames = ['%slayout.txt' % fname.split('full', 1)[0] for fname in self.img_fnames]
            else:  # For Horizonnet dataset
                self.txt_fnames = ['%s.txt' % fname[:-4] for fname in self.img_fnames]

        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.stretch = stretch
        self.p_base = p_base
        self.max_stretch = max_stretch
        self.normcor = normcor
        self.return_cor = return_cor
        self.return_path = return_path

        self._check_dataset()

    def _check_dataset(self):
        for fname in self.txt_fnames:
            assert os.path.isfile(os.path.join(self.cor_dir, fname)),\
                '%s not found' % os.path.join(self.cor_dir, fname)

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])
        # print('\n',img_path)
        # temp = plt.imread(img_path)
        # if temp.shape[0]!= 1024 or temp.shape[1]!= 1024 or temp.shape[2] != 3:
        #     print("temp.shape inconsistency")
        # else:
        #     print(temp.shape)
        img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.
        H, W = img.shape[:2]
        # imshow(img)
        # show()
        # Read ground truth corners
        with open(os.path.join(self.cor_dir,
                               self.txt_fnames[idx])) as f:
            cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)

        # Prepare 2d floor-wall boundary && Visualize the boundary in fisheye
        bon_x, bon_y = [], []
        cor_x, cor_y = [], []

        fisheyeI = np.zeros([1024, 1024, 3], np.uint8)
        n_cor = len(cor)
        if n_cor == 4:
            for i in range(n_cor):
                xys, u1, v1, u2, v2 = omni_stretch.fisheye_connect_points(cor[i],
                                                       cor[(i + 1) % n_cor], scale = 3 / 4)
                rr, cc = draw.circle(v1, u1, 6)
                cor_x.extend(cc)
                cor_y.extend(rr)
                # draw.set_color(fisheyeI, [rr, cc], [0, 0, 255])
                rr, cc = draw.circle(v2, u2, 6)
                cor_x.extend(cc)
                cor_y.extend(rr)
                # draw.set_color(fisheyeI, [rr, cc], [0, 0, 255])

                bon_x.extend(xys[:,0])
                bon_y.extend(xys[:, 1])


            # Corner-1024*1024(0-1值,以像素点为圆心，半径为6的圆作为一个点)
            cor_gt = np.zeros((W, W))
            cor_y = np.array(cor_y)
            cor_x = np.array(cor_x)
            cor_gt[(cor_y, cor_x)] = 1
            # cor_gt[(cor_y, cor_x)] = 255
            # im = Image.fromarray(cor_gt)
            # im.convert('RGB').save("out",format = 'jpeg')

            # for i in range(len(cor_x)):
            #     rr, cc = draw.circle(cor_y[i], cor_x[i], 2)
            #     draw.set_color(fisheyeI, [rr, cc], [0, 255, 0])
            # cv2.namedWindow('img',0)
            # cv2.resizeWindow('img', 1024, 1024)
            # cv2.imshow('img',fisheyeI)
            # cv2.waitKey()

            # Bon-1024*1024(0-1值)
            # bon_x, bon_y = sort_xy_filter_unique(bon_x, bon_y, y_small_first=False)
            # bon_gt = np.zeros((W, W))
            # bon_y = np.array(bon_y).astype(np.int)
            # bon_x = np.array(bon_x).astype(np.int)
            # bon_gt[(bon_y, bon_x)] = 1
            # bon_gt[(bon_y, bon_x)] = 255
            # im = Image.fromarray(bon_gt)
            # im.convert('RGB').save("out",format = 'jpeg')
            fisheyeI = cv2.imread(img_path)
            for i in range(len(bon_x)):
                rr, cc = draw.circle(bon_y[i], bon_x[i], 2)
                draw.set_color(fisheyeI, [rr, cc], [0, 255, 0])
            # imshow(fisheyeI)
            # show()
            bon_path = os.path.join(self.bon_dir, self.img_fnames[idx])
            cv2.imwrite(bon_path, fisheyeI, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # image = Image.fromarray(fisheyeI)
            # image.save(bon_path)

            # bon_gt = np.zeros((2, len(bon_x)))
            # bon_gt[0] = bon_x
            # bon_gt[1] = bon_y
            #
            # bon_gt[0] = bon_gt[0] - W / 2
            # bon_gt[1] = bon_gt[1] - W / 2
            # rs = np.sqrt((bon_gt[0] ** 2 + bon_gt[1] ** 2))
            # phi = np.arcsin(rs / (W / 2))
            # theta = np.arctan2(bon_gt[1], bon_gt[0])
            # bon_gt[0] = phi
            # bon_gt[1] = theta

            # Convert all data to tensor
            x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
            # bon_gt = torch.FloatTensor(bon_gt.copy())
            cor_gt = torch.FloatTensor(cor_gt.copy())

            # Check whether additional output are requested
            out_lst = [x, cor_gt]
            if self.return_cor:
                out_lst.append(cor)
            if self.return_path:
                out_lst.append(img_path)
        return out_lst

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

def visualize_a_data(x, cor_pd, bon_pd):
    x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    cor_pd = cor_pd.numpy() * 100
    bon_pd = bon_pd.numpy() * 10000
    # y_bon = y_bon.numpy()
    # y_bon = ((y_bon / np.pi + 0.5) * x.shape[0]).round().astype(int)

    cor_pd_img = np.zeros((1024, 1024), np.uint8)
    cor_pd_img[:] = cor_pd * 255

    pd_bon_img = np.zeros((1024, 1024), np.uint8)
    pd_bon_img[:] = bon_pd * 255

    # imshow(gt_cor)
    # show()

    img_cor_bon = (x.copy() * 1).astype(np.uint8)
    # y1 = np.round(y_bon[0]).astype(int)
    # y2 = np.round(y_bon[1]).astype(int)
    # y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
    # y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
    temp1 = np.nonzero(cor_pd_img)
    img_cor_bon[temp1[0], temp1[1], 1] = 255
    # temp2 = np.nonzero(pd_bon)
    # img_cor_bon[temp2[0], temp2[1], 0] = 255

    # imshow(img_cor_bon)
    #     # show()
    return img_cor_bon
