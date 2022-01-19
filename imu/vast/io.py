import numpy as np
from scipy.ndimage import zoom
from imu.io import mkdir
from imageio import imsave

def createMipImages(getInputImage, getOutputName, zran, level_start=0, level_num=3, resize_order=1):
    # getInputImage(z): image at slice z 
    # getOutputName(m, z): filename at mip m and slice z 
    for m in range(level_start, level_start+level_num):
        output_name = getOutputName(m, 0)
        output_folder = output_name[:output_name.rfind('/')]
        mkdir(output_folder)

    for z in zran:
        im = getInputImage(z)
        for m in range(level_start, level_start+level_num):
            imsave(getOutputName(m, z), im)
            im = zoom(im, 0.5, order=resize_order)
