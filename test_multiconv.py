import os
import argparse
import glob
import pylab as pl
import json
import torch
from tqdm import tqdm
import time
from scipy import *
from PIL import Image
from skimage.measure import label, regionprops
from pylab import *
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import heapq
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import convolve
from scipy.signal import find_peaks
from shapely.geometry import Polygon
from torchvision.transforms import ToPILImage
from util import omni_model_util, omni_model, omni_post_proc
from util.omni_model import OmniNet
from util.omni_dataset import visualize_a_data



def augment_undo(x_imgs_augmented, aug_type):
    x_imgs_augmented = x_imgs_augmented.cpu().numpy()
    sz = x_imgs_augmented.shape[0] // len(aug_type)
    x_imgs = []
    for i, aug in enumerate(aug_type):
        x_img = x_imgs_augmented[i*sz : (i+1)*sz]
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()

    return np.array(x_imgs)

def augment(x_img, flip, rotate):
    x_img = x_img.numpy()
    aug_type = ['']
    x_imgs_augmented = [x_img]
    if flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for shift_p in rotate:
        shift = int(round(shift_p * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    temp3 = np.concatenate(x_imgs_augmented, 0)
    return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type

def get_cor_peaks(cor_pd, H, W, d1=21, d2=3):
    # direction保存四个最大的方向,cor_id保存预测的四个角点
    cor_peaks = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    # 按照像素遍历，生成遍历数组
    angle = np.arange(0, 2 * np.pi, 1 / (2 * np.pi * 512)).reshape(20213, -1)
    r = np.arange(0, 512)
    pos_x = (np.round(511 + r * np.cos(angle))).astype(np.int)
    pos_y = (np.round(511 + r * np.sin(angle))).astype(np.int)
    pos_x1 = pos_x.reshape(-1, pos_x.shape[0] * pos_x.shape[1])
    pos_y1 = pos_y.reshape(-1, pos_x.shape[0] * pos_x.shape[1])
    # 取出在cor_img中对应的元素并计算每一个角度上的和
    cor_value = cor_pd[pos_x1, pos_y1]
    cor_value = cor_value.reshape(20213, 512)
    cor_value_sum = cor_value.sum(axis=1)
    # 寻找四个最大的方向
    # angle_loc = find_N_peaks(tmpe[0], prominence=None,
    #                          distance=100, N=4)[0]
    # tmpe = cor_value_sum.reshape(-1, 20213)
    # angle_loc = find_N_peaks(tmpe[0], prominence=None,
    #                      distance=100, N=4)[0]
    angle_loc = np.array([0, 0, 0, 0])
    angle_loc[0] = np.argmax(cor_value_sum[: 5053])
    angle_loc[1] = np.argmax(cor_value_sum[5053: 10106]) + 5053
    angle_loc[2] = np.argmax(cor_value_sum[10106: 15159]) + 10106
    angle_loc[3] = np.argmax(cor_value_sum[15159: 20213]) + 15159
    idx = 0
    tmp = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    for i in angle_loc:
        pos = np.argmax(cor_value[i])
        posx = pos_x[i, pos]
        posy = pos_y[i, pos]
        tmp[idx] = [posy, posx]
        idx = idx + 1

    cor_peaks[0] = tmp[1]
    cor_peaks[1] = tmp[0]
    cor_peaks[2] = tmp[3]
    cor_peaks[3] = tmp[2]
    return cor_peaks
def get_cor_peaks_N(cor_pd, W):
    cor_pd = cor_pd.numpy() * 100
    cor_pd_img = np.zeros((1024, 1024), np.uint8)
    cor_pd_img[:] = cor_pd * 255
    cmap_prob = cor_pd_img.copy()

    th = 0
    cor_pd_img[cor_pd_img == th] = 0
    cor_pd_img[cor_pd_img != th] = 1
    label_cmap = label(cor_pd_img)
    regions = regionprops(label_cmap, cmap_prob)

    cor_uv = []
    cor_uv_6 = []
    for props in regions:
        y0, x0 = props.weighted_centroid
        area = props.area
        cor_uv.append([x0, y0, area])
    cor_uv = np.array(cor_uv)
    # cor_uv_temp = np.array(cor_uv_temp)
    cor_uv_area = cor_uv[np.lexsort(cor_uv.T)]
    cor_uv = cor_uv_area[:, [0, 1]]
    print('\n')
    print(cor_uv_area)
    print('\n')
    if cor_uv.shape[0] >= 6:
        row = [cor_uv.shape[0]-6,cor_uv.shape[0]-5,cor_uv.shape[0]-4,cor_uv.shape[0]-3,cor_uv.shape[0]-2,cor_uv.shape[0]-1]
        cor_uv_6 = cor_uv_area[row]
        cor_uv_6 = cor_uv_6[:,[0,1]] #取出数组的前两列
        print(cor_uv_6)
    cor_uv_6 = np.array(cor_uv_6)
    return cor_uv_6
def inference_cuboid(net, x, i_path, device, flip=False, rotate=[], visualize=True,
              force_cuboid=True, min_v=None, r=0.05):
    H, W = tuple(x.shape[2:])

    # Network feedforward (without testing augmentation)
    x, aug_type = augment(x, flip, rotate)
    cor_pd, bon_pd = net(x.to(device))
    # temp1 = torch.sigmoid(y_cor_pd)
    # temp2 = augment_undo(torch.sigmoid(y_cor_pd).cpu(), aug_type)
    cor_pd = augment_undo(torch.sigmoid(cor_pd).cpu(), aug_type).mean(0)
    bon_pd = augment_undo(bon_pd.cpu(), aug_type).mean(0)

    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0], torch.FloatTensor(cor_pd[0]), torch.FloatTensor(bon_pd[0]))
    else:
        vis_out = None

    # Corner peaks
    cor_pd = torch.FloatTensor(cor_pd[0]).numpy() * 100
    cor_peaks = get_cor_peaks(cor_pd, H, W, d1=21, d2=3)
    return cor_peaks, vis_out

def inference_general(net, x, i_path, device, flip=False, rotate=[], visualize=True,
              force_cuboid=True, min_v=None, r=0.05):
    H, W = tuple(x.shape[2:])

    # Network feedforward (without testing augmentation)
    x, aug_type = augment(x, flip, rotate)
    cor_pd, bon_pd = net(x.to(device))
    # temp1 = torch.sigmoid(y_cor_pd)
    # temp2 = augment_undo(torch.sigmoid(y_cor_pd).cpu(), aug_type)
    cor_pd = augment_undo(torch.sigmoid(cor_pd).cpu(), aug_type).mean(0)
    bon_pd = augment_undo(bon_pd.cpu(), aug_type).mean(0)

    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0], torch.FloatTensor(cor_pd[0]), torch.FloatTensor(bon_pd[0]))
    else:
        vis_out = None

    # Corner peaks
    cor_peaks = get_cor_peaks_N(torch.FloatTensor(cor_pd[0]), W)
    return cor_peaks, vis_out
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=False, default='/home/f321/mengming/code/OmniNet-20191204/ckpt_1e_4/Fisheye/Structured3D_Pro_simple_496/OmniNet_DCN_V2/layer4/resnet50_rnn/lin-1BS3KS/resnet50_rnn/epoch_200.pth',
                        help='path to load saved checkpoint from ckpt package.')
    parser.add_argument('--img_glob', required=False, default='/home/f321/mengming/dataset/Structured3D_Pro_simple_496/test/img/*',
                        help='path to load test images. NOTE: Remeber to quote your glob path. '
                             'All the given images are assumed to be aligned'
                             'or you should use preporcess.py to do so.')
    parser.add_argument('--output_dir', required=False,
                        default='/home/f321/mengming/code/OmniNet-20191204/ckpt_1e_4/Fisheye/Structured3D_Pro_simple_496/OmniNet_DCN_V2/layer4/resnet50_rnn/lin-1BS3KS/epoch_200',
                        help='path to dump the responding results')
    # parser.add_argument('--output_dir', required=False, default='E:\\Layout_Reconstruction\\2019-OmniNet\\OmniNet-20191204\\finetune\\dump\\ckpt_1e-4\\Fisheye\\SE_500_200\Sun360\win-2epoch\\epoch_200\\test',
    #                     help='path to dump the responding results')
    parser.add_argument('--visualize', action='store_true')

    # Augmentation related
    parser.add_argument('--flip', action='store_true',
                        help='whether to perfome left-right flip. '
                             '# of input x2.')
    parser.add_argument('--rotate', nargs='*', default=[], type=float,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')

    # Post-processing realted
    parser.add_argument('--r', default=0.05, type=float)
    parser.add_argument('--min_v', default=None, type=float)
    # store_true: cuboid; store_false: genaral cuboid;
    parser.add_argument('--relax_cuboid', action='store_true')

    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    # Set use which CUDA
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = parser.parse_args()

    # Prepare image to processed
    paths = sorted(glob.glob(args.img_glob))
    if len(paths) == 0:
        print('no images found')
    for path in paths:
        assert os.path.isfile(path), '%s not found' % path

    # Check target directory
    if not os.path.isdir(args.output_dir):
        print('Output directory %s not existed. Create one.' % args.output_dir)
        os.makedirs(args.output_dir)
    args = parser.parse_args()
    device = torch.device('cpu' if args.no_cuda else 'cuda:3')

    # Loaded trained model
    net = omni_model_util.load_trained_model(OmniNet, args.pth).to(device)
    print("Loaded Molde:",net)
    net.eval()

    start = time.time()
    temp = 0
    # Testing
    with torch.no_grad():
        cuboid_number = 0;
        noncuboid_number = 0;
        for i_path in tqdm(paths, desc='Testing'):
            # i_path = 'K:\Fisheye\Pano2Fisheye\Structured3D_Pro_empty_noncuboid6_F\test\img\Structured3D_16_scene_03266_487_empty_rgb_rawlight_fish.png'
            k = os.path.split(i_path)[-1][:-4]
            # Load image
            print('Test Image:',i_path)
            img_pil = Image.open(i_path)
            if img_pil.size != (1024, 1024):
                img_pil = img_pil.resize((1024, 1024), Image.BICUBIC)
            # temp1 = np.array(img_pil)
            # temp2 = temp1[..., :3]
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])
            '''
             Inferenceing corners
                If trained on cuboid dataset---not args.relax_cuboid  (HorizonnetNet dataset)
                If trained on non-cuboid dataset--- args.relax_cuboid  (MatterportLayout dataset)
                If trained on non-cuboid dataset--- args.relax_cuboid  (Structured3D_Pro dataset)
            '''
            if args.relax_cuboid: # For general cuboid inference.
                print('-----------------For general cuboid inference---------------')
                cor_peaks, vis_out = inference_general(net, x, i_path, device,
                                               args.flip, args.rotate,
                                               not args.visualize,
                                               args.relax_cuboid,
                                               args.min_v, args.r)
            else: # For cuboid inference. eg. HorizonNet dataset
                print('-----------------For cuboid inference---------------')
                cor_peaks, vis_out = inference_cuboid(net, x, i_path, device,
                                                    args.flip, args.rotate,
                                                    not args.visualize,
                                                    not args.relax_cuboid,
                                                    args.min_v, args.r)

            if cor_peaks.shape[0] == 4:
                cuboid_number = cuboid_number + 1
                # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&',temp)
                # # Output result
                with open(os.path.join(args.output_dir, k + '.json'), 'w') as f:
                    json.dump({
                        # 'z0': float(z0),
                        # 'z1': float(z1),
                        'uv': [[float(u), float(v)] for u, v in cor_peaks],
                    }, f)

                if vis_out is not None:
                    vis_path = os.path.join(args.output_dir, k + '.png')
                    vh, vw = vis_out.shape[:2]
                    Image.fromarray(vis_out).save(vis_path)
            else:
                noncuboid_number = noncuboid_number + 1

    print('Cuboid Number:', cuboid_number)
    print('NonCuboid Number:', noncuboid_number)
    end = time.time()
    print('Test time:', end-start)