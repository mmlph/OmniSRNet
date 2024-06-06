import argparse
import os
import glob
import numpy as np
import json
# import matplotlib.pyplot as plt
from pylab import imshow, show
import cv2
from skimage import draw
from PIL import Image
from tqdm import tqdm
from shapely.geometry import Polygon
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull

from util import omni_model_util, omni_model, omni_post_proc, omni_stretch


def prepare_gtdt_pairs(gt_glob, dt_glob):
    gt_paths = sorted(glob.glob(gt_glob))
    temp = glob.glob(dt_glob)
    if 'Structured3D' in gt_paths[0]:
        dt_paths = dict([(os.path.split(v)[-1].split('simple')[0], v)
                         for v in glob.glob(dt_glob) if v.endswith('json')])
    else:
        dt_paths = dict([(os.path.split(v)[-1].split('.')[0], v)
                         for v in glob.glob(dt_glob) if v.endswith('json')])
    gtdt_pairs = []

    for gt_path in gt_paths:
        if 'Structured3D' in gt_path:
            k = os.path.split(gt_path)[-1].split('simple')[0]
        else:
            k = os.path.split(gt_path)[-1].split('G')[0]
        if k in dt_paths:
            gtdt_pairs.append((gt_path, dt_paths[k]))

    return gtdt_pairs


def gen_reg_from_xy(xy, w):
    xy = xy[np.argsort(xy[:, 0])]
    return np.interp(np.arange(w), xy[:, 0], xy[:, 1], period=w)


def tri2halfspace(pa, pb, p):
    ''' Helper function for evaluating 3DIoU '''
    v1 = pa - p
    v2 = pb - p
    vn = np.cross(v1, v2)
    if -vn @ p > 0:
        vn = -vn
    return [*vn, -vn @ p]


def xyzlst2halfspaces(xyz_floor, xyz_ceil):
    '''
    Helper function for evaluating 3DIoU
    return halfspace enclose (0, 0, 0)
    '''
    N = xyz_floor.shape[0]
    halfspaces = []
    for i in range(N):
        last_i = (i - 1 + N) % N
        next_i = (i + 1) % N

        p_floor_a = xyz_floor[last_i]
        p_floor_b = xyz_floor[next_i]
        p_floor = xyz_floor[i]
        p_ceil_a = xyz_ceil[last_i]
        p_ceil_b = xyz_ceil[next_i]
        p_ceil = xyz_ceil[i]
        halfspaces.append(tri2halfspace(p_floor_a, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_floor_a, p_ceil, p_floor))
        halfspaces.append(tri2halfspace(p_ceil, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_ceil_a, p_ceil_b, p_ceil))
        halfspaces.append(tri2halfspace(p_ceil_a, p_floor, p_ceil))
        halfspaces.append(tri2halfspace(p_floor, p_ceil_b, p_ceil))
    return np.array(halfspaces)


def xylst2halfspaces(xyz_floor, xyz_ceil):
    '''
    Helper function for evaluating 3DIoU
    return halfspace enclose (0, 0, 0)
    '''
    N = xyz_floor.shape[0]
    halfspaces = []
    for i in range(N):
        last_i = (i - 1 + N) % N
        next_i = (i + 1) % N

        p_floor_a = xyz_floor[last_i]
        p_floor_b = xyz_floor[next_i]
        p_floor = xyz_floor[i]
        p_ceil_a = xyz_ceil[last_i]
        p_ceil_b = xyz_ceil[next_i]
        p_ceil = xyz_ceil[i]
        halfspaces.append(tri2halfspace(p_floor_a, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_floor_a, p_ceil, p_floor))
        halfspaces.append(tri2halfspace(p_ceil, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_ceil_a, p_ceil_b, p_ceil))
        halfspaces.append(tri2halfspace(p_ceil_a, p_floor, p_ceil))
        halfspaces.append(tri2halfspace(p_floor, p_ceil_b, p_ceil))
    return np.array(halfspaces)


def fisheye_PE(gt_path, dt_path, dt_cor_id, gt_cor_id, H=1024, W=1024):
    # print('GT_path:', gt_path)
    # print('DT_path:', dt_path)
    # img_dir = '%simg' % gt_path.split('label', 1)[0]
    # img_name = '%s.png' % gt_path.split('F\\', 1)[-1][:-8]
    # gt_img_path = os.path.join(img_dir, img_name)
    # fisheye_img = np.array(Image.open(gt_img_path), np.uint8)[..., :3]

    # dt, gt对应的surface
    surface_dt = np.zeros((H, W), dtype=np.int32)
    surface_gt = np.zeros((H, W), dtype=np.int32)
    bon_x_dt, bon_y_dt = [], []
    bon_x_gt, bon_y_gt = [], []
    cor_x, cor_y = [], []
    for i in range(dt_cor_id.shape[0]):
        coorxy_dt = omni_stretch.fisheye_connect_points(dt_cor_id[i], dt_cor_id[(i + 1) % 4], scale=1)
        # bon_x_dt.extend(coorxy_dt[:, 0])
        # bon_y_dt.extend(coorxy_dt[:, 1])
        coorxy_dt = np.round(coorxy_dt).astype(np.uint8)
        surface_dt[coorxy_dt[:, 1], coorxy_dt[:, 0]] = 1

        coorxy_gt = omni_stretch.fisheye_connect_points(gt_cor_id[i], gt_cor_id[(i + 1) % 4], scale=3 / 4)
        bon_x_gt.extend(coorxy_gt[:, 0])
        bon_y_gt.extend(coorxy_gt[:, 1])
        coorxy_gt = np.round(coorxy_gt).astype(np.uint8)
        surface_gt[coorxy_gt[:, 1], coorxy_gt[:, 0]] = 1

    # # visualize bon_dt and save it
    # for i in range(len(bon_x_dt)):
    #     rr, cc = draw.circle(bon_y_dt[i], bon_x_dt[i], 5)
    #     draw.set_color(fisheye_img, [rr, cc], [0, 255, 0])
    # bon_dir = args.dt_glob.split('*')[0]
    # img__bon_name = '%s_bon.png' % gt_path.split('F\\', 1)[-1][:-8]
    # bon_path = os.path.join(bon_dir, img__bon_name)
    # fisheye_img = Image.fromarray(fisheye_img)
    # fisheye_img.save(bon_path)
    # diff floor & wall
    for i in range(1024):
        # deal with dt_floor
        if np.max(surface_dt[i]) == 1:
            start = np.argmax(surface_dt[i])
            end = np.argmax(surface_dt[i][::-1])
            for j in range(start, 1023 - end):
                surface_dt[i][j] = 1
        # deal with gt_floor
        if np.max(surface_gt[i]) == 1:
            start = np.argmax(surface_gt[i])
            end = np.argmax(surface_gt[i][::-1])
            for j in range(start, 1023 - end):
                surface_gt[i][j] = 1
    PE_loss_error = 100 * (surface_dt != surface_gt).sum() / (args.h * args.w)
    PE_loss_acc = 100 * (surface_dt == surface_gt).sum() / (args.h * args.w)
    return PE_loss_error, PE_loss_acc


def fisheye_IoU_Polygon(dt_cor_id, gt_cor_id, ch=-1.6):
    # assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    # assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0

    # Eval 2d IoU based on top-down view
    N = len(dt_cor_id)
    ch = -3.2
    dt_floor_xy = omni_post_proc.np_coor2xy_fisheye(dt_cor_id, ch, 1024, 1024, floorW=1, floorH=1)
    gt_floor_xy = omni_post_proc.np_coor2xy_fisheye(gt_cor_id, ch, 1024, 1024, floorW=1, floorH=1)
    dt_poly = Polygon(dt_floor_xy)
    gt_poly = Polygon(gt_floor_xy)
    if not gt_poly.is_valid:
        print('Skip ground truth invalid')
        return

    # 2D IoU
    try:
        area_dt = dt_poly.area
        area_gt = gt_poly.area
        area_inter = dt_poly.intersection(gt_poly).area
        IoU2D_loss_Polygon = 100 * area_inter / (area_gt + area_dt - area_inter)
    except:
        IoU2D_loss_Polygon = 0

    # 3D IoU
    try:
        cch_dt = omni_post_proc.get_z_floor_fisheye(dt_cor_id, 1024, 1024)
        cch_gt = omni_post_proc.get_z_floor_fisheye(gt_cor_id, 1024, 1024)
        # h_dt = abs(cch_dt.mean() - ch)
        # h_gt = abs(cch_gt.mean() - ch)
        h = 3.6
        area3d_inter = area_inter * h
        area3d_dt = area_dt * h
        area3d_gt = area_gt * h
        IoU3D_loss_Polygon = 100 * area3d_inter / (area3d_dt + area3d_gt - area3d_inter)
    except:
        IoU3D_loss_Polygon = 0

    return IoU2D_loss_Polygon, IoU3D_loss_Polygon


def metric_IoU_HalfSpace(dt_floor_coor, dt_ceil_coor, gt_floor_coor, gt_ceil_coor, ch=-1.6,
                         coorW=1024, coorH=512, floorW=1024, floorH=512):
    ''' Evaluate 3D IoU using halfspace intersection '''
    dt_floor_coor = np.array(dt_floor_coor)
    dt_ceil_coor = np.array(dt_ceil_coor)
    gt_floor_coor = np.array(gt_floor_coor)
    gt_ceil_coor = np.array(gt_ceil_coor)
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0
    N = len(dt_floor_coor)

    # # IoU2D Calculate
    # dt_floor_xy = np.hstack([
    #     omni_post_proc.np_coor2xy(dt_floor_coor, ch, coorW, coorH, floorW=1, floorH=1),
    #     np.zeros((N, 1)),
    # ])
    # gt_floor_xy = np.hstack([
    #     omni_post_proc.np_coor2xy(gt_floor_coor, ch, coorW, coorH, floorW=1, floorH=1),
    #     np.zeros((N, 1)),
    # ])
    # dt_floor_c = np.sqrt((dt_floor_xy[:, :2] ** 2).sum(1))
    # gt_floor_c = np.sqrt((gt_floor_xy[:, :2] ** 2).sum(1))
    # #dt_floor_v2 = omni_post_proc.np_coory2v(dt_ceil_coor[:, 1], coorH)
    # #gt_floor_v2 = omni_post_proc.np_coory2v(gt_ceil_coor[:, 1], coorH)
    # dt_floor_ceil_z = 0
    # gt_floor_ceil_z = 0
    #
    # dt_ceil_xy = dt_floor_xy.copy()
    # dt_ceil_xy[:, 2] = dt_floor_ceil_z
    # gt_ceil_xy = gt_floor_xy.copy()
    # gt_ceil_xy[:, 2] = gt_floor_ceil_z
    #
    # dt_floor_halfspaces = xylst2halfspaces(dt_floor_xy, dt_ceil_xy)
    # gt_floor_halfspaces = xylst2halfspaces(gt_floor_xy, gt_ceil_xy)
    #
    # in_floor_halfspaces = HalfspaceIntersection(np.concatenate([dt_floor_halfspaces, gt_floor_halfspaces]), np.zeros(3))
    # in_floor_dt_halfspaces = HalfspaceIntersection(dt_floor_halfspaces, np.zeros(3))
    # in_floor_gt_halfspaces = HalfspaceIntersection(gt_floor_halfspaces, np.zeros(3))
    #
    # in_floor_volume = ConvexHull(in_floor_halfspaces.intersections).volume
    # dt_floor_volume = ConvexHull(in_floor_dt_halfspaces.intersections).volume
    # gt_floor_volume = ConvexHull(in_floor_gt_halfspaces.intersections).volume
    # un_floor_volume = dt_floor_volume + gt_floor_volume - in_floor_volume
    # IoU2D_loss_HalfSpace = 100 * in_floor_volume / un_floor_volume

    # IoU3D Calculate
    dt_floor_xyz = np.hstack([
        omni_post_proc.np_coor2xy(dt_floor_coor, ch, coorW, coorH, floorW=1, floorH=1),
        np.zeros((N, 1)) + ch,
    ])
    gt_floor_xyz = np.hstack([
        omni_post_proc.np_coor2xy(gt_floor_coor, ch, coorW, coorH, floorW=1, floorH=1),
        np.zeros((N, 1)) + ch,
    ])
    dt_c = np.sqrt((dt_floor_xyz[:, :2] ** 2).sum(1))
    gt_c = np.sqrt((gt_floor_xyz[:, :2] ** 2).sum(1))
    dt_v2 = omni_post_proc.np_coory2v(dt_ceil_coor[:, 1], coorH)
    gt_v2 = omni_post_proc.np_coory2v(gt_ceil_coor[:, 1], coorH)
    dt_ceil_z = dt_c * np.tan(dt_v2)
    gt_ceil_z = gt_c * np.tan(gt_v2)

    dt_ceil_xyz = dt_floor_xyz.copy()
    dt_ceil_xyz[:, 2] = dt_ceil_z
    gt_ceil_xyz = gt_floor_xyz.copy()
    gt_ceil_xyz[:, 2] = gt_ceil_z

    dt_halfspaces = xyzlst2halfspaces(dt_floor_xyz, dt_ceil_xyz)
    gt_halfspaces = xyzlst2halfspaces(gt_floor_xyz, gt_ceil_xyz)
    in_halfspaces = HalfspaceIntersection(np.concatenate([dt_halfspaces, gt_halfspaces]), np.zeros(3))
    in_dt_halfspaces = HalfspaceIntersection(dt_halfspaces, np.zeros(3))
    in_gt_halfspaces = HalfspaceIntersection(gt_halfspaces, np.zeros(3))

    in_volume = ConvexHull(in_halfspaces.intersections).volume
    dt_volume = ConvexHull(in_dt_halfspaces.intersections).volume
    gt_volume = ConvexHull(in_gt_halfspaces.intersections).volume
    un_volume = dt_volume + gt_volume - in_volume

    IoU3D_loss_HalfSpace = 100 * in_volume / un_volume
    return IoU3D_loss_HalfSpace


def metric_IoU_Polygon(dt_floor_coor, dt_ceil_coor, gt_floor_coor, gt_ceil_coor, ch=-1.6):
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0

    # Eval 2d IoU based on top-down view
    N = len(dt_floor_coor)
    ch = -1.6
    dt_floor_xy = omni_post_proc.np_coor2xy(dt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    gt_floor_xy = omni_post_proc.np_coor2xy(gt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    dt_poly = Polygon(dt_floor_xy)
    gt_poly = Polygon(gt_floor_xy)
    if not gt_poly.is_valid:
        print('Skip ground truth invalid (%s)' % gt_path)
        return
    area_dt = dt_poly.area
    area_gt = gt_poly.area
    area_inter = dt_poly.intersection(gt_poly).area
    IoU2D_loss_Polygon = 100 * area_inter / (area_gt + area_dt - area_inter)

    # Eval 3d IoU and height error(in meter)
    cch_dt = omni_post_proc.get_z1(dt_floor_coor[:, 1], dt_ceil_coor[:, 1], ch, 512)
    cch_gt = omni_post_proc.get_z1(gt_floor_coor[:, 1], gt_ceil_coor[:, 1], ch, 512)

    h_dt = abs(cch_dt.mean() - ch)
    h_gt = abs(cch_gt.mean() - ch)
    IoUH = min(h_dt, h_gt) / max(h_dt, h_gt)
    # 体积=底面积 * 高
    IoU3D_loss_Polygon = IoU2D_loss_Polygon * IoUH
    return IoU2D_loss_Polygon, IoU3D_loss_Polygon


def evaluate_cuboid(gtdt_pairs, losses):
    i = 0
    for gt_path, dt_path in tqdm(gtdt_pairs, desc='Testing'):
        # Parse ground truth
        with open(gt_path) as f:
            # print('Path-GT:' ,f)
            gt_cor_id = np.array([l.split() for l in f], np.float32)

        # Parse predicted results
        with open(dt_path) as f:
            dt = json.load(f)
        dt_cor_id = np.array(dt['uv'], np.float32)
        # dt_cor_id[:, 0] *= args.w
        # dt_cor_id[:, 1] *= args.h

        # 1.Evaluate corner error--CE
        if len(gt_cor_id) == len(dt_cor_id):
            MSE_CE = np.sqrt(((gt_cor_id - dt_cor_id) ** 2).sum(1)).mean()
            CE_loss = 100 * MSE_CE / np.sqrt(args.w ** 2 + args.h ** 2)
        else:
            CE_loss = 0
            i = i + 1
            # print('************Layout Inconsistence Error Numbers********* %d'% (i))

        # 2.Evaluate pixel surface error--PE (3 labels: ceiling, wall, floor)
        PE_loss_err, PE_loss_acc = fisheye_PE(gt_path, dt_path, dt_cor_id, gt_cor_id, H=1024, W=1024)

        # 3.Evaluate volumetric intersection over union--IoU3D
        # IoU3D_loss_HalfSpace = metric_IoU_HalfSpace(dt_cor_id[1::2], dt_cor_id[0::2], gt_cor_id[1::2], gt_cor_id[0::2])
        IoU2D_loss_Polygon, IoU3D_loss_Polygon = fisheye_IoU_Polygon(dt_cor_id, gt_cor_id)

        # Add all metrics to items result losses_cuboid
        n_corners = len(gt_cor_id)
        if n_corners % 2 == 1:
            n_corners = 'odd'
        elif n_corners < 10:
            n_corners = str(n_corners)
        else:
            n_corners = '10+'
        losses[n_corners]['CE'].append(CE_loss)
        losses[n_corners]['PE_err'].append(PE_loss_err)
        losses[n_corners]['PE_acc'].append(PE_loss_acc)
        losses[n_corners]['2DIoU_HalfSpace'].append(0)
        losses[n_corners]['3DIoU_HalfSpace'].append(0)
        losses[n_corners]['2DIoU_Polygon'].append(IoU2D_loss_Polygon)
        losses[n_corners]['3DIoU_Polygon'].append(IoU3D_loss_Polygon)

        losses['overall']['CE'].append(CE_loss)
        losses['overall']['PE_err'].append(PE_loss_err)
        losses['overall']['PE_acc'].append(PE_loss_acc)
        losses['overall']['2DIoU_HalfSpace'].append(0)
        losses['overall']['3DIoU_HalfSpace'].append(0)
        losses['overall']['2DIoU_Polygon'].append(IoU2D_loss_Polygon)
        losses['overall']['3DIoU_Polygon'].append(IoU3D_loss_Polygon)


def evaluate_general(gtdt_pairs, losses):
    i = 0
    j = 0
    for gt_path, dt_path in tqdm(gtdt_pairs, desc='Testing'):
        # Parse ground truth
        with open(gt_path) as f:
            # print('Path-GT:' ,f)
            gt_cor_id = np.array([l.split() for l in f], np.float32)

        # Parse predicted results
        with open(dt_path) as f:
            dt = json.load(f)
        dt_cor_id = np.array(dt['uv'], np.float32)
        # dt_cor_id[:, 0] *= args.w
        # dt_cor_id[:, 1] *= args.h

        # sort gt&dt consistent
        gt_cor_id = np.asarray(sorted(gt_cor_id, key=lambda x: (x[0], x[1])))
        dt_cor_id = np.asarray(sorted(dt_cor_id, key=lambda x: (x[0], x[1])))
        print(gt_path)
        # 1.Evaluate corner error--CE
        if len(gt_cor_id) == len(dt_cor_id):
            MSE_CE = np.sqrt(((gt_cor_id - dt_cor_id) ** 2).sum(1)).mean()
            CE_loss = 100 * MSE_CE / np.sqrt(args.w ** 2 + args.h ** 2)
            print(gt_cor_id)
            print(dt_cor_id)
            print(MSE_CE)
            j = j + 1
            print('************Layout Numbers********* %d' % (j))
        else:
            CE_loss = 0
            i = i + 1
            print('************Layout Inconsistence Error Numbers********* %d' % (i))

        # Add all metrics to items result losses_cuboid
        n_corners = len(gt_cor_id)
        if n_corners % 2 == 1:
            n_corners = 'odd'
        elif n_corners < 10:
            n_corners = str(n_corners)
        else:
            n_corners = '10+'
        losses[n_corners]['CE'].append(CE_loss)
        losses['overall']['CE'].append(CE_loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dt_glob',
                        default='/home/f321/mengming/code/OmniNet-20191204/ckpt_1e_4/Fisheye/Structured3D_Pro_simple_496/OmniNet_DCN_V2/layer4/resnet50_rnn/lin-1BS3KS/epoch_75/*json',
                        help='NOTE: Remeber to quote your glob path.'
                             'Files assumed to be json from inference.py')
    parser.add_argument('--gt_glob',
                        default='/home/f321/mengming/dataset/Structured3D_Pro_simple_496/test/label_cor_SF/*txt',
                        help='NOTE: Remeber to quote your glob path.''Files assumed to be txt')
    parser.add_argument('--w', default=1024, type=int,
                        help='GT images width')
    parser.add_argument('--h', default=1024, type=int,
                        help='GT images height')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parser.parse_args()

    # Prepare (gt, dt) pairs
    gtdt_pairs = prepare_gtdt_pairs(args.gt_glob, args.dt_glob)

    # Evaluating Cuboid dataset
    losses = dict([
        (n_corner,
         {'CE': [], 'PE_err': [], 'PE_acc': [], '2DIoU_HalfSpace': [], '3DIoU_HalfSpace': [], '2DIoU_Polygon': [],
          '3DIoU_Polygon': []})
        for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
    ])
    evaluate_cuboid(gtdt_pairs, losses)
    # Print evaluation results: CE,PE,2DIoU, 3DIoU
    print(' Testing Cuboid Result '.center(100, '='))
    for k, result in losses.items():
        ce = np.array(result['CE'])
        pe_err = np.array(result['PE_err'])
        pe_acc = np.array(result['PE_acc'])
        iou2d_HalfSpace = np.array(result['2DIoU_HalfSpace'])
        iou3d_HalfSpace = np.array(result['3DIoU_HalfSpace'])
        iou2d_general = np.array(result['2DIoU_Polygon'])
        iou3d_general = np.array(result['3DIoU_Polygon'])
        print('GT #Corners: %s  (%d instances)' % (k, len(iou2d_general)))
        print('    2DIoU_Polygon for Cuboid: %.2f' % (
            iou2d_general.mean(),
        ))
        print('    3DIoU_Polygon for Cuboid: %.2f' % (
            iou3d_general.mean(),
        ))
        print('                          CE: %.2f' % (
            ce.mean(),
        ))
        print('                          PE_err: %.2f' % (
            pe_err.mean(),
        ))
        print('                          PE_acc: %.2f' % (
            pe_acc.mean(),
        ))
        print('    2DIoU_HalfSpace for Cuboid: %.2f' % (
            iou2d_HalfSpace.mean(),
        ))
        print('    3DIoU_HalfSpace for Cuboid: %.2f' % (
            iou3d_HalfSpace.mean(),
        ))
    print('=' * 100)

    # evaluate_general(gtdt_pairs, losses)
    # Print evaluation results: CE,PE,2DIoU, 3DIoU
    # print(' Testing general Result '.center(100, '='))
    # for k, result in losses.items():
    #     ce = np.array(result['CE'])
    #     print('                          CE: %.2f' % (
    #         ce.mean(),
    #     ))
    # print('=' * 100)

    # Print evaluation results: CE,PE,2DIoU, 3DIoU
    # print(' Testing General Cuboid Result '.center(100, '='))
    # for k, result in losses.items():
    #     ce = np.array(result['CE'])
    #     pe_err = np.array(result['PE_err'])
    #     pe_acc = np.array(result['PE_acc'])
    #     iou2d_HalfSpace = np.array(result['2DIoU_HalfSpace'])
    #     iou3d_HalfSpace = np.array(result['3DIoU_HalfSpace'])
    #     iou2d_general = np.array(result['2DIoU_Polygon'])
    #     iou3d_general = np.array(result['3DIoU_Polygon'])
    #     if len(iou2d_general) == 0:
    #         continue
    #     print('GT #Corners: %s  (%d instances)' % (k, len(iou2d_general)))
    #     print('    2DIoU_Polygon for Cuboid: %.2f' % (
    #         iou2d_general.mean(),
    #     ))
    #     print('    3DIoU_Polygon for Cuboid: %.2f' % (
    #         iou3d_general.mean(),
    #     ))
    #     print('                          CE: %.2f' % (
    #         ce.mean(),
    #     ))
    #     print('                          PE_err: %.2f' % (
    #         pe_err.mean(),
    #     ))
    #     print('                          PE_acc: %.2f' % (
    #         pe_acc.mean(),
    #     ))
    # print('    2DIoU_HalfSpace for Cuboid: %.2f' % (
    #     iou2d_HalfSpace.mean(),
    # ))
    # print('    3DIoU_HalfSpace for Cuboid: %.2f' % (
    #     iou3d_HalfSpace.mean(),
    # ))
    # print('=' * 100)