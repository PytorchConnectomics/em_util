import os,sys
import numpy as np
import h5py
from imageio import imread, imwrite
from imu.io import get_bb, segBiggest

from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_opening,binary_fill_holes,binary_dilation
import cv2
import glob



def maskProp(get_im, get_mask, save_mask, pad_size, anchor_z, num_z):
    pad_offset = np.array([-pad_size, pad_size, -pad_size, pad_size])
    
    im0 = get_im(0)
    im_size = np.array(im0.shape)
    # forward pass
    mask0 = np.zeros(im_size, np.uint8)
    anchors = sorted(list(anchors))
    for fb in 'fb':
        if fb == 'f':
            # forward propagation
            p_anchors = anchors
            p_z_delta = -1
            def get_zran(z):
                if z == anchors[-1]:
                    return range(z+1, num_z)
                else:
                    # forward to the middle frame
                    z2 = anchors[np.where(anchors==z)[0]+1]
                    return range(z+1, (z+z2+1)//2 )
        else:
            # backward propagation
            p_anchors = anchors[1:]
            p_z_delta = 1
            def get_zran(z):
                if z == anchors[0]:
                    return range(z-1, -1, -1)
                else:
                    # backward to the middle frame
                    z2 = anchors[np.where(anchors==z)[0]-1]
                    return range(z-1, (z+z2+1)//2, -1)

        for zi in p_anchors:
            prev_z = zi + p_z_delta
            for zz in get_zran(zi):
                if not os.path.exists(save_mask(zz)):
                    if zz != prev_z - p_z_delta:
                        # re-compute the stats
                        prev_mask = get_mask(zz + p_z_delta)
                        bb = get_bb(prev_mask) + pad_size 
                        bb[::2] = np.maximum(0, bb[::2])
                        bb[1::2] = np.minimum(im_size-1, bb[1::2])
                    curr_im = get_im(zz)
                    curr_im_bb = curr_im[bb[0]:bb[1]+1, bb[2]:bb[3]+1]

                   # direct copy
                    curr_mask_bb = prev_mask[bb[0]:bb[1]+1, bb[2]:bb[3]+1]
                    mask0[:] = 0
                    mask0[bb[0]:bb[1]+1, bb[2]:bb[3]+1] = curr_mask_bb

                    # remove dark region
                    c0 = np.percentile(curr_im[bb[0]:bb[1]+1, bb[2]:bb[3]+1], [30,60])
                    print(sn, bb, c0)
                    mask0[curr_im < c0[0]] = 0
                    mask0 = getSegMax(binary_opening(binary_fill_holes(mask0), iterations=1)) > 0
                    # add bright region
                    # binary_openning: make sure the seg has smooth shape
                    seg_good = binary_opening(binary_fill_holes(curr_im > c0[1]), iterations=2)
                    seg_good = binary_dilation(mask0, iterations=3) * seg_good
                    mask0[seg_good > 0] = 1

                    save_mask(zz, mask0)

                    prev_mask[:] = mask0
                    bb = get_bb(prev_mask) + pad_size
                    prev_z = zz

if __name__ == "__main__":
    Dmask = Dl + 'JoAr2_Ov_aligned_new/masks/%04d_mask.png'
    Dim = Dl + 'JoAr2_Ov_aligned_new/whole_im/%04d.png'
    
    # the algorithm works better with downsampled/low-res image and mask
    def get_mask(z):
        return imread(Dmask % z)
    def get_im(z):
        curr_im = imread(Dim % z)
    def save_mask(z, mask = None):
        if mask is None:
            return Dmp%z
        imwrite(Dmask % z, (mask>0).astype(np.uint8) * 255)

    # anchor: [0,100,..,500]
    anchor_z = range(0, 501, 100)
    pad_size = 40
    num_z = len(glob.glob(Dim[:Dim.rfind('/')] + '/*.png')) 
   
    maskProp(get_im, get_mask, save_mask, pad_size, anchor_z, num_z)
