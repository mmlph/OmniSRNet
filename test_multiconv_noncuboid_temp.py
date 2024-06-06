import numpy as np
from PIL import Image
from skimage.measure import label, regionprops




def cmap2corners(cmap):
    # when corner appear at the image borders, the heatmap can extend to both sides of the images(left or right)
    # by concatenating 3 times the corner map, we are sure that we detect the maximum once, avoiding repeated corners
    cmap = cmap / 255.
    cmap_ = np.hstack((cmap, cmap, cmap))
    cmap_prob = cmap_.copy()

    th = 0.1
    cmap_[cmap_ < th] = 0
    cmap_[cmap_ > th] = 1
    label_cmap = label(cmap_)
    regions = regionprops(label_cmap, cmap_prob)

    cor_uv = []
    for props in regions:
        y0, x0 = props.weighted_centroid
        if x0 > (pano_W-1) and x0 < (pano_W*2 + 1):
            cor_uv.append([x0-pano_W, y0])

    cor_uv = np.array(cor_uv)

    # order from left to right
    sorted_cor_uv = sorted(cor_uv, key=lambda x : x[0])
    sorted_cor_uv = np.array([arr.tolist() for arr in sorted_cor_uv])

    return sorted_cor_uv


if __name__ == '__main__':
    pano_H = 1024
    pano_W = 1024

    # load coener map
    cmap_path = 'E:/tmp/2.png'
    cmap = Image.open(cmap_path)
    cmap = np.array(cmap)

    # get 2D corners from corner map
    cor_uv = cmap2corners(cmap)
    print(cor_uv)