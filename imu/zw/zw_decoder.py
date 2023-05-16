import os
import numpy as np
from imageio import imwrite

from ..io import seg_relabel, mkdir, segToVast


class zwDecoder(object):
    def __init__(self, aff_file, output_folder="./", job_id=0, job_num=1):
        self.aff_file = aff_file
        self.output_folder = output_folder
        self.job_id = job_id
        self.job_num = job_num

    def affToSeg(
        self,
        vol_key="vol0",
        num_slice=-1,
        filename_suf="",
        T_low=0.15,
        T_high=0.9,
        T_rel=False,
        T_thres=150,
        T_dust=150,
        T_dust_merge=0.2,
        T_mst_merge=0.7,
    ):
        import h5py
        from zwatershed import zwatershed

        # sa2 zw2
        aff = h5py.File(self.aff_file, "r")[vol_key]
        aff_size = aff.shape
        output = self.output_folder + "zw2d-%s/" % (vol_key)
        mkdir(output)
        num_slice = aff_size[1] if num_slice < 0 else num_slice
        for zi in range(self.job_id, num_slice, self.job_num):
            output_file = output + "%04d%s.png" % (zi, filename_suf)
            if not os.path.exists(output_file):
                print(zi)
                aff_s = (
                    np.array(aff[:, zi : zi + 1]).astype(np.float32) / 255.0
                )
                if aff_size[0] == 2:
                    # 2D: need add a channel
                    aff_s = np.concatenate(
                        [
                            np.zeros(
                                [1, 1, aff_size[2], aff_size[3]], np.float32
                            ),
                            aff_s,
                        ],
                        axis=0,
                    )
                out = zwatershed(
                    aff_s,
                    T_threshes=[T_thres],
                    T_dust=T_dust,
                    T_aff=[T_low, T_high, T_dust_merge],
                    T_aff_relative=T_rel,
                    T_merge=T_mst_merge,
                )[0][0][0]
                imwrite(output_file, segToVast(seg_relabel(out, do_type=True)))
