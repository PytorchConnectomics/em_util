import networkx as nx
import numpy as np

def skelToLength(vertices, edges, res = [1,1,1]):
    """
    Returns cable length of connected skeleton vertices in the same
    metric that this volume uses (typically nanometers).
    """
    if vertices.shape[0] == 0:
        return 0
    v1 = vertices[edges[:,0]]
    v2 = vertices[edges[:,1]]

    delta = (v2 - v1) * res
    delta *= delta
    dist = np.sum(delta, axis=1)
    dist = np.sqrt(dist)
    return np.sum(dist)

def skelToNetworkX(nodes, edges, seg_list, res):
    # for ERL evaluation
    gt_graph = nx.Graph()
    node_segment_lut = [{}]*len(seg_list)
    cc = 0
    for k in range(len(nodes)):
        node = nodes[k]
        edge = edges[k] + cc
        for l in range(node.shape[0]):
            gt_graph.add_node(cc, skeleton_id = k, z=node[l,0]*res[0], y=node[l,1]*res[1], x=node[l,2]*res[2])
            for i in range(len(seg_list)):
                node_segment_lut[i][cc] = seg_list[i][node[l,0], node[l,1], node[l,2]]
            cc += 1
        for l in range(edge.shape[0]):
            gt_graph.add_edge(edge[l,0], edge[l,1])
    return gt_graph, node_segment_lut
