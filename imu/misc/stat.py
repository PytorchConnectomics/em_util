from scipy.ndimage.morphology import binary_erosion
import numpy as np


def get_query_count(ui, uc, qid, mm=0):
    # memory efficient
    # mm: ignore value
    if len(qid) == 0:
        return []
    ui_r = [ui[ui > mm].min(), max(ui.max(), qid.max())]
    rl = mm * np.ones(1 + int(ui_r[1] - ui_r[0]), uc.dtype)
    rl[ui[ui > mm] - ui_r[0]] = uc[ui > mm]

    cc = mm * np.ones(qid.shape, uc.dtype)
    gid = np.logical_and(qid >= ui_r[0], qid <= ui_r[1])
    cc[gid] = rl[qid[gid] - ui_r[0]]
    return cc


def get_sphericity(seg):
    # compute the sphericity for all segments at the same time
    # https://en.wikipedia.org/wiki/Sphericity
    seg_erode = binary_erosion(seg > 0, iterations=1)
    sid, vol = np.unique(seg, return_counts=True)
    sid2, vol2 = np.unique(seg_erode * seg, return_counts=True)
    vol_erode = get_query_count(sid2, vol2, sid)
    vol_diff = vol - vol_erode
    vol_diff[sid == 0] = 0
    sphe = -np.ones(vol.shape)
    sphe[vol_diff > 0] = (
        np.pi ** (1.0 / 3) * ((6 * vol[vol_diff > 0]) ** (2.0 / 3))
    ) / vol_diff[vol_diff > 0]
    return sid, sphe, vol
