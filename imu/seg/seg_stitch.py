import numpy as np
import h5py
from imageio import imsave
import imu
from imu.seg import seg_iou2d, merge_id
from imu.io import read_h5, write_h5, mkdir, rgb_to_seg


class seg_chunk_stitcher(object):
    def __init__(self, zran, yran, xran, getVol, job_id=0, job_num=1):
        self.zran = zran
        self.yran = yran
        self.xran = xran
        self.znum = len(zran) - 1
        self.ynum = len(yran) - 1
        self.xnum = len(xran) - 1
        self.getVol = getVol
        self.job_id = job_id
        self.job_num = job_num

    def set_job(self, job_id, job_num):
        self.job_id = job_id
        self.job_num = job_num

    def chunk_count(self, fn_count):
        # get seg id count for each chunk
        cid = 0
        for zi in range(self.znum):
            for yi in range(self.ynum):
                for xi in range(self.xnum):
                    if cid % self.job_num == self.job_id:
                        sn = fn_count % (zi, yi, xi)
                        if not os.path.exists(sn):
                            seg = getVol(
                                self.zran[zi],
                                self.zran[zi + 1],
                                self.yran[yi],
                                self.yran[yi + 1],
                                self.xran[xi],
                                self.xran[xi + 1],
                            )
                            ui, uc = np.unique(seg, return_counts=True)
                            del seg
                            uc = uc[ui > 0]
                            ui = ui[ui > 0]
                            write_h5(sn, np.vstack([ui, uc]).T)
                    cid += 1

    def chunk_count_cum(self, fn_count, fn_count_cum):
        # get seg count offset
        count = np.zeros([self.znum, self.ynum, self.xnum], int)
        for zi in range(self.znum):
            for yi in range(self.ynum):
                for xi in range(self.xnum):
                    sn = fn_count % (zi, yi, xi)
                    count[zi, yi, xi] = read_h5(sn).shape[0]
        count_cum = np.cumsum(count)
        count_offset = np.hstack([[0], count_cum[:-1]]).reshape(count.shape)
        write_h5(
            fn_count_cum,
            [count, count_offset, count_cum[-1:]],
            ["count", "offset", "max_id"],
        )

    def chunk_merge_xy(self, fn_mid_xy, fn_count=None, offset=None):
        cid = 0
        for zi in range(self.znum):
            for yi in range(self.ynum):
                for xi in range(self.xnum):
                    if cid % self.job_num == self.job_id:
                        # y-axis merge
                        mid_all = np.zeros([0, 2], np.uint32)
                        if yi != self.ynum - 1:
                            seg = getVol(
                                self.zran[zi],
                                self.zran[zi + 1],
                                self.yran[yi + 1] - 1,
                                self.yran[yi + 1] + 1,
                                self.xran[xi],
                                self.xran[xi + 1],
                            )
                            if fn_count is not None:
                                stat = read_h5(fn_count % (zi, yi, xi))
                                rl = np.zeros(stat[:, 0].max() + 1, np.uint32)
                                rl[stat[:, 0]] = offset[
                                    zi, yi, xi
                                ] + np.arange(1, stat.shape[0] + 1)
                                seg[:, 0] = rl[seg[:, 0]]
                                stat_next = read_h5(
                                    fn_count % (zi, yi + 1, xi)
                                )
                                rl_next = np.zeros(
                                    stat_next[:, 0].max() + 1, np.uint32
                                )
                                rl_next[stat_next[:, 0]] = offset[
                                    zi, yi + 1, xi
                                ] + np.arange(1, stat_next.shape[0] + 1)
                                seg[:, 1] = rl_next[seg[:, 1]]
                            seg_diff = (seg > 0).min(axis=1)
                            mid = np.unique(
                                np.vstack(
                                    [seg[:, 0][seg_diff], seg[:, 1][seg_diff]]
                                ).T,
                                axis=0,
                            ).astype(seg.dtype)
                            mid_all = np.vstack([mid_all, mid])
                        if xi != self.xnum - 1:
                            seg = getVol(
                                self.zran[zi],
                                self.zran[zi + 1],
                                self.yran[yi],
                                self.yran[yi + 1],
                                self.xran[xi + 1] - 1,
                                self.xran[xi + 1] + 1,
                            )
                            if fn_count is not None:
                                seg[:, :, 0] = rl[seg[:, :, 0]]
                                stat_next = read_h5(
                                    fn_count % (zi, yi, xi + 1)
                                )
                                rl_next = np.zeros(
                                    stat_next[:, 0].max() + 1, np.uint32
                                )
                                rl_next[stat_next[:, 0]] = offset[
                                    zi, yi + 1, xi
                                ] + np.arange(1, stat_next.shape[0] + 1)
                                seg[:, :, 1] = rl_next[seg[:, :, 1]]
                            seg_diff = (seg > 0).min(axis=2)
                            mid = np.unique(
                                np.vstack(
                                    [
                                        seg[:, :, 0][seg_diff],
                                        seg[:, :, 1][seg_diff],
                                    ]
                                ).T,
                                axis=0,
                            ).astype(seg.dtype)
                            mid_all = np.vstack([mid_all, mid])
                        if yi != self.ynum - 1 or xi != self.xnum - 1:
                            sn = fn_mid_xy % (zi, yi, xi)
                            if not os.path.exists(sn):
                                write_h5(sn, mid_all)
                    cid += 1

    def chunk_merge_xy_all(self, fn_mid_xy, fn_mid_xy_all, max_id=-1):
        if max_id > 0:
            mid = np.ones([1, 2], np.uint32) * max_id
        else:
            mid = np.zeros([0, 2], np.uint32)
        for zi in range(self.znum):
            for yi in range(self.ynum):
                for xi in range(self.xnum):
                    if yi != self.ynum - 1 or xi != self.xnum - 1:
                        mid = np.vstack(
                            [mid, read_h5(fn_mid_xy % (zi, yi, xi))]
                        )

        rl = merge_id(mid[:, 0].astype(np.uint32), mid[:, 1].astype(np.uint32))
        write_h5(fn_mid_xy_all, rl)

    def section_merge_z(
        self,
        fn_mid_xy_all,
        fn_mid_z,
        iou_thres=0.2,
        fn_count=None,
        offset=None,
    ):
        rl_xy = read_h5(fn_mid_xy_all)
        for zi in range(self.znum - 1)[self.job_id :: self.job_num]:
            sn = fn_mid_z % zi
            if not os.path.exists(sn):
                seg_z0 = np.zeros(
                    [
                        self.yran[-1] - self.yran[0],
                        self.xran[-1] - self.xran[0],
                    ],
                    np.uint32,
                )
                seg_z1 = np.zeros(
                    [
                        self.yran[-1] - self.yran[0],
                        self.xran[-1] - self.xran[0],
                    ],
                    np.uint32,
                )
                for yi in range(self.ynum):
                    for xi in range(self.xnum):
                        seg = getVol(
                            self.zran[zi + 1] - 1,
                            self.zran[zi + 1] + 1,
                            self.yran[yi],
                            self.yran[yi + 1],
                            self.xran[xi],
                            self.xran[xi + 1],
                        )

                        if fn_count is not None:
                            stat = read_h5(fn_count % (zi, yi, xi))
                            rl = np.zeros(stat[:, 0].max() + 1, np.uint32)
                            rl[stat[:, 0]] = offset[zi, yi, xi] + np.arange(
                                1, stat.shape[0] + 1
                            )
                            seg[0] = rl[seg[0]]

                            stat_next = read_h5(fn_count % (zi, yi + 1, xi))
                            rl_next = np.zeros(
                                stat_next[:, 0].max() + 1, np.uint32
                            )
                            rl_next[stat_next[:, 0]] = offset[
                                zi, yi + 1, xi
                            ] + np.arange(1, stat_next.shape[0] + 1)
                            seg[1] = rl_next[seg[1]]
                        seg_z0[
                            self.yran[yi]
                            - self.yran[0] : self.yran[yi + 1]
                            - self.yran[0],
                            self.xran[xi]
                            - self.xran[0] : self.xran[xi + 1]
                            - self.xran[0],
                        ] = rl_xy[seg[0]]
                        seg_z1[
                            self.yran[yi]
                            - self.yran[0] : self.yran[yi + 1]
                            - self.yran[0],
                            self.xran[xi]
                            - self.xran[0] : self.xran[xi + 1]
                            - self.xran[0],
                        ] = rl_xy[seg[1]]
                mid = np.zeros([0, 2], np.uint32)
                iou_f = seg_iou2d(seg_z0, seg_z1)
                rr = iou_f[:, -1] / (iou_f[:, -3:-1].max(axis=1).astype(float))
                mid = np.vstack([mid, iou_f[rr >= iou_thres, :2]])

                iou_b = seg_iou2d(seg_z1, seg_z0)
                rr = iou_b[:, -1] / (iou_b[:, -3:-1].max(axis=1).astype(float))
                mid = np.vstack([mid, iou_b[rr >= iou_thres, :2]])

                mid = np.unique(
                    np.vstack([mid.min(axis=1), mid.max(axis=1)]).T, axis=0
                )
                write_h5(sn, mid)

    def section_merge_z_all(self, fn_mid_z, fn_mid_z_all, max_id=-1):
        if max_id > 0:
            mid = np.ones([1, 2], np.uint32) * max_id
        else:
            mid = np.zeros([0, 2], np.uint32)
        for zi in range(self.znum - 1):
            mid = np.vstack([mid, read_h5(fn_mid_z % (zi))])

        rl = merge_id(mid[:, 0].astype(np.uint32), mid[:, 1].astype(np.uint32))
        write_h5(fn_mid_z_all, rl)

    def section_output(
        self, fn_mid_xy_all, fn_mid_z_all, fn_out, fn_count=None, offset=None
    ):
        rl_xy = read_h5(fn_mid_xy_all)
        rl_z = read_h5(fn_mid_z_all)
        for zi in range(self.znum)[self.job_id :: self.job_num]:
            seg_z = np.zeros(
                [
                    self.zran[zi + 1] - self.zran[zi],
                    self.yran[-1] - self.yran[0],
                    self.xran[-1] - self.xran[0],
                ],
                np.uint32,
            )
            for yi in range(self.ynum):
                for xi in range(self.xnum):
                    seg = getVol(
                        self.zran[zi],
                        self.zran[zi + 1],
                        self.yran[yi],
                        self.yran[yi + 1],
                        self.xran[xi],
                        self.xran[xi + 1],
                    )

                    if fn_count is not None:
                        stat = read_h5(fn_count % (zi, yi, xi))
                        rl = np.zeros(stat[:, 0].max() + 1, np.uint32)
                        rl[stat[:, 0]] = offset[zi, yi, xi] + np.arange(
                            1, stat.shape[0] + 1
                        )
                        seg = rl[seg]

                    seg_z[
                        :,
                        self.yran[yi]
                        - self.yran[0] : self.yran[yi + 1]
                        - self.yran[0],
                        self.xran[xi]
                        - self.xran[0] : self.xran[xi + 1]
                        - self.xran[0],
                    ] = rl_z[rl_xy[seg]]
            for z in range(self.zran[zi], self.zran[zi + 1]):
                imwrite(fn_out % z, seg2rgb(seg_z[z - self.zran[zi]]))
