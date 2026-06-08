"""Add downsampled mip levels to an existing precomputed layer on GCS (igneous).

Works for both image layers (averaging; use for the affinity map) and
segmentation layers (mode pooling, auto-detected from ``type`` in the info).
Run this as the follow-up "mips" step after a large mip0 upload.

Env: igneous (throwaway venv ``/tmp/igneous_env``).

Usage:
    /tmp/igneous_env/bin/python downsample_ng.py \
        --cloudpath gs://.../zebrafinch/affinity \
        --key .../hi-mc-collab-f8437ce66bd3.json \
        --num-mips 5 --parallel 8
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cloudpath", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--num-mips", type=int, default=4)
    p.add_argument("--mip", type=int, default=0, help="source mip to downsample from")
    p.add_argument("--parallel", type=int, default=8)
    args = p.parse_args()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.key
    import igneous.task_creation as tc
    from taskqueue import LocalTaskQueue

    tq = LocalTaskQueue(parallel=args.parallel)
    print(f"downsampling {args.cloudpath}: +{args.num_mips} mips from mip {args.mip} ...")
    tq.insert(tc.create_downsampling_tasks(
        args.cloudpath, mip=args.mip, num_mips=args.num_mips, preserve_chunk_size=True))
    tq.execute()
    print("done")


if __name__ == "__main__":
    main()
