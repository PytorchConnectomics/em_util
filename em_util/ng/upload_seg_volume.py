"""Upload a dense segmentation volume (HDF5) to a precomputed neuroglancer
segmentation layer on Google Cloud Storage.

The HDF5 holds a single dataset (default ``main``) of shape (X, Y, Z) in the
same axis order / resolution as the existing ``nisb_seed101_seg`` layer, so the
array is written to CloudVolume verbatim (no transpose). Data is streamed in
Z-slabs to bound memory.

Usage:
    python upload_seg_volume.py \
        --volume <decoded>.h5 \
        --cloudpath gs://princeton-eric-medial-entorhinal-cortex-scratch/nisb_seed101_cc3d_t066 \
        --key /projects/weilab/weidf/lib/pytorch_connectomics/lib/keys/hi-mc-collab-f8437ce66bd3.json \
        --resolution 9 9 20
"""
import argparse
import os

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", required=True, help="path to segmentation .h5")
    parser.add_argument("--dataset", default="main", help="h5 dataset name")
    parser.add_argument("--cloudpath", required=True, help="gs:// destination layer path")
    parser.add_argument("--key", required=True, help="GCS service-account json key")
    parser.add_argument("--resolution", type=float, nargs=3, default=[9, 9, 20])
    parser.add_argument("--chunk-size", type=int, nargs=3, default=[128, 128, 64])
    parser.add_argument("--z-slab", type=int, default=128, help="Z voxels per upload slab")
    args = parser.parse_args()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.key

    from cloudvolume import CloudVolume

    with h5py.File(args.volume, "r") as h:
        dset = h[args.dataset]
        shape = tuple(int(s) for s in dset.shape)
        dtype = np.dtype(dset.dtype)
        print(f"volume: {args.volume}[{args.dataset}] shape={shape} dtype={dtype}")
        assert len(shape) == 3, "expected a 3D (X, Y, Z) segmentation"

        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type="segmentation",
            data_type=str(dtype),
            encoding="raw",
            resolution=args.resolution,
            voxel_offset=[0, 0, 0],
            volume_size=list(shape),
            chunk_size=args.chunk_size,
        )
        cv = CloudVolume(args.cloudpath, info=info, compress=True, progress=False)
        cv.commit_info()

        nz = shape[2]
        for z0 in range(0, nz, args.z_slab):
            z1 = min(z0 + args.z_slab, nz)
            slab = dset[:, :, z0:z1]
            cv[:, :, z0:z1] = slab[..., np.newaxis]
            print(f"  uploaded z[{z0}:{z1}] / {nz}")

    print(f"done -> {args.cloudpath}")


if __name__ == "__main__":
    main()
