"""Upload a ground-truth skeleton (networkx graph) to a precomputed
neuroglancer skeleton layer on Google Cloud Storage.

The skeleton pickle is a single ``networkx.Graph`` whose nodes carry
``nm_position`` (xyz nanometers), ``index_position`` (voxel index) and an
``id`` grouping nodes into per-segment skeletons. We emit one precomputed
skeleton file per ``id`` (vertices in nm, identity transform) under a
standalone segmentation layer.

Usage:
    python upload_gt_skeleton.py \
        --skeleton /projects/weilab/dataset/nisb/base/test/seed101/skeleton.pkl \
        --cloudpath gs://princeton-eric-medial-entorhinal-cortex-scratch/nisb_seed101_skeleton \
        --key /projects/weilab/weidf/lib/pytorch_connectomics/lib/keys/hi-mc-collab-f8437ce66bd3.json \
        --resolution 9 9 20 --size 3000 3000 1350
"""
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np


def build_skeletons(graph):
    """Group graph nodes by their ``id`` into per-segment skeletons.

    Returns a list of (segid, vertices_nm, edges) tuples where vertices are
    (N, 3) float32 nanometers and edges are (M, 2) uint32 local indices.
    """
    from cloudvolume import Skeleton

    nodes_by_id = defaultdict(list)
    for node, attr in graph.nodes(data=True):
        nodes_by_id[int(attr["id"])].append(node)

    skeletons = []
    for segid, nodes in nodes_by_id.items():
        local = {node: i for i, node in enumerate(nodes)}
        vertices = np.array(
            [graph.nodes[n]["nm_position"] for n in nodes], dtype=np.float32
        )
        edges = [
            (local[u], local[v])
            for u, v in graph.subgraph(nodes).edges()
        ]
        edges = (
            np.array(edges, dtype=np.uint32)
            if edges
            else np.zeros((0, 2), dtype=np.uint32)
        )
        skel = Skeleton(vertices=vertices, edges=edges, segid=segid)
        # minimal skeleton: vertices + edges only (no radius/vertex_types)
        skel.extra_attributes = []
        skeletons.append(skel)
    return skeletons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", required=True, help="path to skeleton.pkl")
    parser.add_argument(
        "--cloudpath", required=True, help="gs:// destination layer path"
    )
    parser.add_argument(
        "--key", required=True, help="GCS service-account json key"
    )
    parser.add_argument(
        "--resolution", type=float, nargs=3, default=[9, 9, 20]
    )
    parser.add_argument(
        "--size", type=int, nargs=3, default=[3000, 3000, 1350]
    )
    args = parser.parse_args()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.key

    from cloudvolume import CloudVolume

    with open(args.skeleton, "rb") as f:
        graph = pickle.load(f)
    print(
        f"loaded graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    skeletons = build_skeletons(graph)
    print(f"built {len(skeletons)} per-segment skeletons")

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint32",
        encoding="raw",
        resolution=args.resolution,
        voxel_offset=[0, 0, 0],
        volume_size=args.size,
        chunk_size=[128, 128, 64],
        skeletons="skeletons",
    )
    cv = CloudVolume(args.cloudpath, info=info)
    cv.commit_info()

    # minimal skeleton metadata: identity transform, no vertex attributes
    cv.skeleton.meta.info["transform"] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    cv.skeleton.meta.info["vertex_attributes"] = []
    cv.skeleton.meta.commit_info()

    cv.skeleton.upload(skeletons)
    print(f"uploaded {len(skeletons)} skeletons to {args.cloudpath}/skeletons")


if __name__ == "__main__":
    main()
