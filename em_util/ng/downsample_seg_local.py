"""Downsample a local dense segmentation HDF5 to a coarser mip (mode pooling)
and save next to it, for later analysis. Streams Z-slabs (Z is not downsampled
at factor (2,2,1), so slab-wise == whole-volume and the result matches
igneous/CloudVolume mode-downsampled mips exactly).

Env: needs ``tinybrain`` + h5py (added tinybrain to ``emu``:
``.../envs/emu/bin/pip install tinybrain``).

Usage:
    # mip2 = 4x XY downsample (two iterative 2x2 mode rounds)
    python downsample_seg_local.py --volume <seg>.h5 --mip 2 \
        --resolution-out 36 36 20
"""
import argparse

import h5py
import numpy as np
import tinybrain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume", required=True, help="mip0 segmentation .h5 (X,Y,Z)")
    p.add_argument("--dataset", default="main")
    p.add_argument("--mip", type=int, default=2, help="target mip (XY factor = 2**mip)")
    p.add_argument("--out", default=None, help="output .h5 (default: <volume>_mip<MIP>.h5)")
    p.add_argument("--resolution-out", type=float, nargs=3, default=None,
                   help="resolution attr to stamp on the output")
    p.add_argument("--z-slab", type=int, default=128)
    args = p.parse_args()

    out = args.out or args.volume.replace(".h5", f"_mip{args.mip}.h5")
    factor_xy = 2 ** args.mip

    with h5py.File(args.volume, "r") as h:
        d = h[args.dataset]
        X, Y, Z = d.shape
        mx, my = (X + factor_xy - 1) // factor_xy, (Y + factor_xy - 1) // factor_xy
        print(f"mip0 {d.shape} -> mip{args.mip} ({mx},{my},{Z})")
        with h5py.File(out, "w") as fo:
            od = fo.create_dataset(
                "main", shape=(mx, my, Z), dtype=d.dtype, compression="gzip",
                compression_opts=4, chunks=(min(256, mx), min(256, my), 64),
            )
            for z0 in range(0, Z, args.z_slab):
                z1 = min(z0 + args.z_slab, Z)
                slab = d[:, :, z0:z1]
                mips = tinybrain.downsample_segmentation(
                    slab, factor=(2, 2, 1), num_mips=args.mip)
                od[:, :, z0:z1] = mips[args.mip - 1]
                print(f"  z[{z0}:{z1}] -> {mips[args.mip - 1].shape}")
            if args.resolution_out:
                od.attrs["resolution"] = list(args.resolution_out)
            od.attrs["mip"] = args.mip
            od.attrs["parent"] = args.volume.split("/")[-1]
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
