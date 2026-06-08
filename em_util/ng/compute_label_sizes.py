"""Compute per-label voxel counts for a dense segmentation HDF5 by streaming
Z-slabs (bounded memory), and save as ``np.save`` (label 0 = background).

The result feeds the ``--sizes`` whitelist of ``mesh_seg_volume.py`` (mesh only
segments above a global voxel-size threshold) and any size-based analysis.

Env: any with numpy + h5py (e.g. the ``emu`` conda env).

Usage:
    python compute_label_sizes.py --volume <seg>.h5 --out sizes.npy [--dataset main]
"""
import argparse

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume", required=True, help="path to segmentation .h5 (X,Y,Z)")
    p.add_argument("--dataset", default="main")
    p.add_argument("--out", required=True, help="output .npy of the bincount")
    p.add_argument("--z-slab", type=int, default=128)
    args = p.parse_args()

    with h5py.File(args.volume, "r") as h:
        d = h[args.dataset]
        nz = d.shape[2]
        maxlabel = int(d[:, :, :1].max())  # cheap lower bound; grow below
        sizes = None
        for z0 in range(0, nz, args.z_slab):
            z1 = min(z0 + args.z_slab, nz)
            sub = d[:, :, z0:z1].ravel()
            bc = np.bincount(sub)
            if sizes is None:
                sizes = bc.astype(np.int64)
            elif bc.size > sizes.size:
                bc[: sizes.size] += sizes
                sizes = bc.astype(np.int64)
            else:
                sizes[: bc.size] += bc
            print(f"  z[{z0}:{z1}] / {nz}")
    sizes[0] = 0
    np.save(args.out, sizes)
    nlab = int((sizes > 0).sum())
    print(f"labels (nonzero): {nlab} | max id {sizes.size - 1} | saved -> {args.out}")
    for thr in (1000, 10000, 100000):
        print(f"  > {thr:>7} voxels: {int((sizes > thr).sum())}")


if __name__ == "__main__":
    main()
