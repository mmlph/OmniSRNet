# OmniSRNet
This is the implementation of our "Structure Recovery from Single Omnidirectional Image with Distortion-aware Learning" 

# Method overview
![Overview](https://github.com/mmlph/OmniSRNet/assets/13580379/4359d949-4ef7-48a8-a132-c680fc8d0424)

# Prerequisites
Ubuntu20 
NVIDIA GPU + CUDA CuDNN
pytorch 1.8.1 with python 3.7.6
<details>
  <summary> Dependencies (click to expand) </summary>

   - numpy
   - scipy
   - sklearn
   - Pillow
   - tqdm
   - tensorboardX
   - opencv-python>=3.1 (for pre-processing)
   - pylsd-nova
   - open3d>=0.7 (for layout 3D viewer)
   - shapely
</details>
# Datasets(IOSR)
（1）Panorama
    - PanoContext/Stanford2D3D Dataset
    - [Download preprocessed pano/s2d3d](https://drive.google.com/open?id=1e-MuWRx3T4LJ8Bu4Dc0tKcSHF9Lk_66C) for training/validation/testing
    - Structured3D Dataset
    - See [the tutorial](https://github.com/sunset1995/HorizonNet/blob/master/README_ST3D.md#dataset-preparation) to prepare training/validation/testing.
（2）Fisheye
    PanoContext-F，Stanford2D3D-F and Structured3D-F are converted from the optimized panorama images by pano2fish.py
