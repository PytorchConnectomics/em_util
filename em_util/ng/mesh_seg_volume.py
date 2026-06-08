"""Generate 3D neuroglancer meshes for a precomputed segmentation layer on GCS,
restricted to segments above a voxel-size threshold.

Steps (igneous LocalTaskQueue):
  1. downsample the base layer to add mip levels (mode pooling for segmentation),
  2. mesh the selected mip for the whitelisted object ids only,
  3. write per-object mesh manifests.

The object whitelist is computed from a global label-size pass (``--sizes`` is a
``np.save`` of bincount over the full volume, label 0 = background) so a segment
is meshed completely across chunks iff its TOTAL voxel count exceeds the
threshold (per-chunk dust thresholds would clip thin neurites).

Usage:
    python mesh_seg_volume.py \
        --cloudpath gs://.../nisb_seed101_cc3d_t066 \
        --key .../hi-mc-collab-f8437ce66bd3.json \
        --sizes /tmp/cc3d_t066_sizes.npy --min-voxels 10000 \
        --mesh-mip 2 --error 40 --parallel 8
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cloudpath", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--sizes", required=True, help="np.save of per-label voxel bincount")
    p.add_argument("--min-voxels", type=int, default=10000)
    p.add_argument("--mesh-mip", type=int, default=2)
    p.add_argument("--num-mips", type=int, default=2, help="downsample levels to add")
    p.add_argument("--error", type=float, default=40.0, help="max simplification error (nm)")
    p.add_argument("--parallel", type=int, default=8)
    args = p.parse_args()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.key

    import numpy as np
    import igneous.task_creation as tc
    from taskqueue import LocalTaskQueue

    sizes = np.load(args.sizes)
    sizes[0] = 0
    big_ids = np.where(sizes > args.min_voxels)[0].tolist()
    print(f"meshing {len(big_ids)} segments with > {args.min_voxels} voxels "
          f"(of {(sizes > 0).sum()} total labels)")

    # explicit mesh_dir passed to every step so fragments, manifests and the
    # info 'mesh' key agree (igneous's default name embeds the float error,
    # e.g. 'mesh_mip_2_err_40.0', which is easy to mismatch).
    mesh_dir = f"mesh_mip_{args.mesh_mip}_err_{int(args.error)}"

    tq = LocalTaskQueue(parallel=args.parallel)

    # 1. downsample (segmentation -> mode pooling); skips levels that already exist
    print(f"downsampling: adding {args.num_mips} mip levels ...")
    tq.insert(tc.create_downsampling_tasks(
        args.cloudpath, mip=0, num_mips=args.num_mips, preserve_chunk_size=True,
    ))
    tq.execute()

    # 2. mesh selected objects at the chosen mip
    print(f"meshing at mip {args.mesh_mip}, err {args.error} -> {mesh_dir} ...")
    tq.insert(tc.create_meshing_tasks(
        args.cloudpath, mip=args.mesh_mip, shape=(448, 448, 448),
        simplification=True, max_simplification_error=args.error,
        object_ids=big_ids, mesh_dir=mesh_dir, compress="gzip",
    ))
    tq.execute()

    # 3. per-object manifests
    print("writing mesh manifests ...")
    tq.insert(tc.create_mesh_manifest_tasks(args.cloudpath, mesh_dir=mesh_dir, magnitude=3))
    tq.execute()

    # ensure the info advertises the mesh dir for neuroglancer
    from cloudvolume import CloudVolume
    cv = CloudVolume(args.cloudpath)
    if cv.info.get("mesh") != mesh_dir:
        cv.info["mesh"] = mesh_dir
        cv.commit_info()
    print(f"done -> {args.cloudpath} (mesh: {cv.info.get('mesh')})")


if __name__ == "__main__":
    main()
