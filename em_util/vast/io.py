import numpy as np
from scipy.ndimage import zoom
from ..io import mkdir, seg_to_rgb
from imageio import imwrite


def create_mip_images(
    get_input_image, get_output_name, zran, level_ran=range(3), resize_order=1, do_seg=False
):
    # need helper function to get image slice from 3D volume or image namges
    # get_input_image(z): image at slice z
    # get_output_name(m, z): filename at mip m and slice z
    output_name = get_output_name(0, 0)
    root_folder = output_name[: output_name.rfind("/")]
    root_folder = root_folder[: root_folder.rfind("/")]
    mkdir(root_folder)
    for m in level_ran:
        output_name = get_output_name(m, 0)
        output_folder = output_name[: output_name.rfind("/")]
        mkdir(output_folder)

    for z in zran:
        im = get_input_image(z)
        # downsample until first mip level
        for i in range(level_ran[0] - 1):
            im = zoom(im, 0.5, order=resize_order)

        for m in level_ran:
            if do_seg:
                imwrite(get_output_name(m, z), seg_to_rgb(im))
            else:
                imwrite(get_output_name(m, z), im)
            if m != level_ran[-1]:
                im = zoom(im, 0.5, order=resize_order)
