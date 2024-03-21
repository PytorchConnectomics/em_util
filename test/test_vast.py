import os,sys
from em_util.vast import *
from em_util.io import rgbToSeg
from imageio import imread, imwrite
import glob

def test_exportToVolume(folder_name):
    meta_name = glob.glob(folder_name + '/*.txt')
    if len(meta_name) < 1:
        raise ValueError('No meta file found at: %s'%folder_name)
    names = sorted(glob.glob(folder_name + '/*.png'))
    if len(names) < 1:
        raise ValueError('No exported images found at: %s'%folder_name)
    print('Found %d images.'%len(names))

    seg_relabel = vastMetaRelabel(meta_name[0])
    im = imread(names[0])
    ims = np.zeros([len(names), im.shape[0], im.shape[1]], np.uint32)
    for z in range(len(names)):
        ims[z] = seg_relabel[rgbToSeg(imread(names[z]))]
    return ims


if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == '0':
        # python test_vast.py 0 output_folder/
        output_folder = sys.argv[2]
        out = test_exportToVolume(output_folder)
