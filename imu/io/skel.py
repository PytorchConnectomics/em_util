import networkx as nx
import numpy as np


def skel_to_length(vertices, edges, res=[1, 1, 1]):
    """
    Returns cable length of connected skeleton vertices in the same
    metric that this volume uses (typically nanometers).
    """
    if vertices.shape[0] == 0:
        return 0
    v1 = vertices[edges[:, 0]]
    v2 = vertices[edges[:, 1]]

    delta = (v2 - v1) * res
    delta *= delta
    dist = np.sum(delta, axis=1)
    dist = np.sqrt(dist)
    return np.sum(dist)


def skel_to_networkx(nodes, edges, seg_list, res):
    # for ERL evaluation
    gt_graph = nx.Graph()
    node_segment_lut = [{}] * len(seg_list)
    count = 0
    for k, node in enumerate(nodes):
        edge = edges[k] + count
        for node_row in np.array(node):
            gt_graph.add_node(
                cc,
                skeleton_id=k,
                z=node_row[0] * res[0],
                y=node_row[1] * res[1],
                x=node_row[2] * res[2],
            )
            for i, seg_id in enumerate(seg_list):
                node_segment_lut[i][cc] = seg_id[
                    node_row[0], node_row[1], node_row[2]
                ]
            count += 1
        for edge_row in np.array(edge):
            gt_graph.add_edge(edge_row[0], edge_row[1])
    return gt_graph, node_segment_lut
