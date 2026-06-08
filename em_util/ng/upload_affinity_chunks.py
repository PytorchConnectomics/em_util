"""Stitch per-chunk affinity predictions (the chunked-raw inference output) into
one precomputed neuroglancer IMAGE layer on GCS, placing each chunk at its true
global location.

Input: a ``*.h5.chunks/`` directory of ``chunk_z{Z}_y{Y}_x{X}.h5`` files, each
holding ``main`` as float16 ``(3, dz, dy, dx)`` in CZYX order = the CORE region
(halo already cropped). Per-file attrs ``chunk_start_zyx`` / ``chunk_stop_zyx``
give the exact global voxel placement; chunks tile the volume with no overlap and
no gaps, so stitching is pure placement (read attrs -> write block).

float16 is not a precomputed dtype, so choose an output dtype:
  * uint8   : affinity [0,1] -> [0,255] (round). ~1/255 precision; compact.
  * float32 : full precision; matches the NISB ``*_raw_x1`` affinity layers; large.

CHUNK SIZE: the storage ``--chunk-size`` should DIVIDE the tiling step (1008 for
the zebrafinch run) so no storage-chunk straddles two data-chunks -- otherwise
unaligned writes need read-modify-write and parallel shards race on seam chunks.
Default 112 (1008/112 = 9). Tile seams sit at multiples of 1008 (= 9*112); only
volume-boundary partials are unaligned, and those touch a single data-chunk.

RESUMABLE: a marker file per chunk under ``<chunks>/.ng_uploaded/`` is written on
success and skipped on re-run. Parallelize across nodes with ``--shard-id`` /
``--num-shards`` (run ``--create-info`` ONCE first to avoid info-write races).

Env: cloudvolume + h5py + numpy (e.g. ``emu``). Memory ~6 GB (float16 load) +
~3 GB (uint8) / ~12 GB (float32) per chunk; one chunk at a time per process.

Usage:
    # 1) create the layer info once
    python stitch_affinity_to_ng.py --chunks-dir <...>.h5.chunks \
        --cloudpath gs://.../zebrafinch/affinity --key <key>.json \
        --dtype uint8 --resolution 10 10 10 --create-info
    # 2) upload (optionally sharded across N jobs)
    python stitch_affinity_to_ng.py --chunks-dir <...>.h5.chunks \
        --cloudpath gs://.../zebrafinch/affinity --key <key>.json \
        --dtype uint8 --shard-id 0 --num-shards 13
"""
import argparse
import ast
import glob
import os
import re

import h5py
import numpy as np

NAME_RE = re.compile(r"chunk_z(\d+)_y(\d+)_x(\d+)\.h5$")


def list_chunks(chunks_dir):
    """Return sorted (path, key, start_zyx, shape_zyx) for every chunk file."""
    out = []
    for path in glob.glob(os.path.join(chunks_dir, "chunk_z*_y*_x*.h5")):
        m = NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        with h5py.File(path, "r") as h:
            d = h["main"]
            start = ast.literal_eval(d.attrs["chunk_start_zyx"])  # global ZYX
            shape = tuple(int(s) for s in d.shape[1:])            # (dz,dy,dx)
        out.append((path, f"z{m[1]}_y{m[2]}_x{m[3]}", tuple(start), shape))
    out.sort(key=lambda t: t[1])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks-dir", required=True)
    p.add_argument("--cloudpath", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--dtype", choices=["uint8", "float32"], default="uint8")
    p.add_argument("--encoding", choices=["raw", "jpeg"], default="raw")
    p.add_argument("--resolution", type=float, nargs=3, default=[10, 10, 10])
    p.add_argument("--chunk-size", type=int, nargs=3, default=[112, 112, 112],
                   help="storage chunk (should divide the 1008 tiling step)")
    p.add_argument("--create-info", action="store_true",
                   help="create/commit the layer info, then exit (run once)")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="process at most N chunks (debug)")
    args = p.parse_args()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.key
    from cloudvolume import CloudVolume

    chunks = list_chunks(args.chunks_dir)
    # global volume size (XYZ) = max over chunk stops
    stop_zyx = np.zeros(3, dtype=np.int64)
    for _, _, start, shape in chunks:
        stop_zyx = np.maximum(stop_zyx, np.array(start) + np.array(shape))
    volume_xyz = [int(stop_zyx[2]), int(stop_zyx[1]), int(stop_zyx[0])]
    print(f"{len(chunks)} chunks -> volume ZYX {tuple(int(s) for s in stop_zyx)} "
          f"= XYZ {volume_xyz}; dtype={args.dtype} encoding={args.encoding}")

    if args.create_info:
        info = CloudVolume.create_new_info(
            num_channels=3, layer_type="image", data_type=args.dtype,
            encoding=args.encoding, resolution=args.resolution,
            voxel_offset=[0, 0, 0], volume_size=volume_xyz, chunk_size=args.chunk_size,
        )
        cv = CloudVolume(args.cloudpath, info=info)
        cv.commit_info()
        print(f"created info -> {args.cloudpath}")
        return

    cv = CloudVolume(args.cloudpath, mip=0, compress=True, progress=False,
                     fill_missing=True, non_aligned_writes=True)
    marker_dir = os.path.join(args.chunks_dir, ".ng_uploaded")
    os.makedirs(marker_dir, exist_ok=True)

    shard = [c for i, c in enumerate(chunks) if i % args.num_shards == args.shard_id]
    if args.limit:
        shard = shard[: args.limit]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(shard)} chunks")

    for n, (path, key, start, shape) in enumerate(shard):
        marker = os.path.join(marker_dir, key)
        if os.path.exists(marker):
            print(f"  [{n+1}/{len(shard)}] {key} skip (done)")
            continue
        z0, y0, x0 = start
        dz, dy, dx = shape
        with h5py.File(path, "r") as h:
            arr = h["main"][:]                       # (3, dz, dy, dx) float16
        data = np.transpose(arr, (3, 2, 1, 0))       # -> (dx, dy, dz, 3)
        if args.dtype == "uint8":
            data = (np.clip(data.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            data = data.astype(np.float32)
        cv[x0:x0 + dx, y0:y0 + dy, z0:z0 + dz, :] = data
        open(marker, "w").close()
        print(f"  [{n+1}/{len(shard)}] {key} -> XYZ[{x0}:{x0+dx},{y0}:{y0+dy},{z0}:{z0+dz}]")

    print(f"shard {args.shard_id} done")


if __name__ == "__main__":
    main()
