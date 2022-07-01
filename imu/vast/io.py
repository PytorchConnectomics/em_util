import numpy as np
from scipy.ndimage import zoom
from imu.io import mkdir, segToRgb
from imageio import imsave

def createMipImages(getInputImage, getOutputName, zran, level_ran=range(3), resize_order=1, do_seg=False):
    # need helper function to get image slice from 3D volume or image namges
    # getInputImage(z): image at slice z 
    # getOutputName(m, z): filename at mip m and slice z 
    output_name = getOutputName(0, 0)
    root_folder = output_name[:output_name.rfind('/')]
    root_folder = root_folder[:root_folder.rfind('/')]
    mkdir(root_folder)
    for m in level_ran:
        output_name = getOutputName(m, 0)
        output_folder = output_name[:output_name.rfind('/')]
        mkdir(output_folder)

    for z in zran:
        im = getInputImage(z)
        # downsample until first mip level
        for i in range(level_ran[0]-1):
            im = zoom(im, 0.5, order=resize_order)

        for m in level_ran:
            if do_seg:
                imsave(getOutputName(m, z), segToRgb(im))
            else:
                imsave(getOutputName(m, z), im)
            if m != level_ran[-1]:
                im = zoom(im, 0.5, order=resize_order)
