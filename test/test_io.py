import os,sys
from em_util.io import *


def test_bfly():
    Di = '/n/boslfs/LABS/lichtman_lab/aligned_datasets/Moritz_L4_2019/'
    out_name = Di + 'em/im.json'

    volume_size = [3306, 1024*9, 1024*6]
    num_tile = [9, 6]
    tile_size = 1024
    def filename_template(z):
        return Di + 'em/mip0/%04d/'%(z)+'%d_%d.png' 
    writeBfly(volume_size, num_tile, filename_template, tile_size, out_name=out_name, image_id = range(118,3424))

if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == '0':
        test_bfly()
