import os
import argparse
import numpy as np
import torch
import torch.utils.data as data
import shutil

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset related arguments
    parser.add_argument('--data_dir', default='K:\\Fisheye\\Pano2Fisheye\\Structured3D_Pro_full',
                        help='root directory to save fisheye dataset for all type. '
                             'should contains img, label_cor, label_cor subdirectories')
    parser.add_argument('--save_dir', default='K:\\Fisheye\\Pano2Fisheye\\Structured3D_Pro_full_cuboid',
                        help='root directory to save fisheye dataset for cuboid. '
                             'should contains img, label_cor, label_cor subdirectories')
    args = parser.parse_args()
    # 分别处理train， valid和test的筛选，保存到Structured3D_Pro_full_cuboid中
        # according label_cor中每个txt中个数，若是4，则将img.label_cor,label_cor_SF中对应的文件保存
    data_train_dir = os.path.join(args.data_dir, 'train')
    data_img_dir = os.path.join(data_train_dir, 'img')
    data_cor_dir = os.path.join(data_train_dir, 'label_cor')
    data_cor_SF_dir = os.path.join(data_train_dir, 'label_cor_SF')
    save_train_dir = os.path.join(args.save_dir, 'train')
    save_img_dir = os.path.join(save_train_dir, 'img')
    save_cor_dir = os.path.join(save_train_dir, 'label_cor')
    save_cor_SF_dir = os.path.join(save_train_dir, 'label_cor_SF')

    for fname in os.listdir(data_cor_dir) :
        if fname.endswith('.txt'):
            cor = 0
            for line in open(data_cor_dir + '\\' + fname, 'r'):
                cor += 1
            if cor == 4:
                shutil.copy(data_img_dir + '\\' + '%s.png' % fname.split('GT', 1)[0], save_img_dir)
                shutil.copy(data_cor_dir + '\\' + fname, save_cor_dir)
                shutil.copy(data_cor_dir + '\\' + '%s.png' % fname.split('.', 1)[0], save_cor_dir)
                shutil.copy(data_cor_SF_dir + '\\' + '%sSF.txt' % fname.split('.', 1)[0], save_cor_SF_dir)
                shutil.copy(data_cor_SF_dir + '\\' + '%sSF.png' % fname.split('.', 1)[0], save_cor_SF_dir)

