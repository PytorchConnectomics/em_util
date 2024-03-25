import networkx as nx
import numpy as np


def skel_to_length(vertices, edges, res=None):
    """
    Compute the cable length of connected skeleton vertices.

    Args:
        vertices (numpy.ndarray): The coordinates of the skeleton vertices.
        edges (numpy.ndarray): The indices of the connected vertices.
        res (list, optional): The resolution of the volume in each dimension. Defaults to None.

    Returns:
        float: The cable length of the connected skeleton vertices.

    Notes:        
        - The cable length is computed using the Euclidean distance between connected vertices.
        - The resolution of the volume can be provided using the `res` parameter.
    """    
    #Returns cable length of connected skeleton vertices in the same metric that this volume uses (typically nanometers).
    
    if res is None:
        res = [1, 1, 1]
    if vertices.shape[0] == 0:
        return 0
    v1 = vertices[edges[:, 0]]
    v2 = vertices[edges[:, 1]]

    delta = (v2 - v1) * res
    delta *= delta
    dist = np.sum(delta, axis=1)
    dist = np.sqrt(dist)
    return np.sum(dist)



def skel_to_networkx(
    skeletons, skeleton_resolution=None, return_all_nodes=False, data_type=np.uint16
):
    """
    The function `skel_to_networkx` converts a skeleton object into a networkx graph, with an option
    to return all nodes.

    :param skeletons: The "skeletons" parameter is a list of skeleton objects. Each skeleton object
    represents a graph structure with nodes and edges. The function converts these skeleton objects into
    a networkx graph object
    :param skeleton_resolution: The `skeleton_resolution` parameter is an optional parameter that
    specifies the resolution of the skeleton. It is used to scale the node coordinates in the skeleton.
    If provided, the node coordinates will be multiplied by the skeleton resolution
    :param return_all_nodes: The `return_all_nodes` parameter is a boolean flag that determines whether
    or not to return all the nodes in the graph. If `return_all_nodes` is set to `True`, the function
    will return both the graph object and an array of all the nodes in the graph. If `return_all,
    defaults to False (optional)
    :return: The function `skeleton_to_networkx` returns a networkx graph object representing the
    skeleton. Additionally, if the `return_all_nodes` parameter is set to `True`, the function also
    returns an array of all the nodes in the skeleton.
    """

    # node in gt_graph: physical unit
    gt_graph = nx.Graph()
    count = 0
    all_nodes = [None] * len(skeletons)
    for skeleton_id, skeleton in enumerate(skeletons):
        if len(skeleton.edges) == 0:
            continue
        if skeleton_resolution is not None:
            node_arr = node_arr * skeleton_resolution
        node_arr = skeleton.vertices.astype(data_type)
        # augment the node index
        edge_arr = skeleton.edges + count
        for node in node_arr:
            # unit: physical
            gt_graph.add_node(
                count, skeleton_id=skeleton_id, z=node[0], y=node[1], x=node[2]
            )
            count += 1
        for edge in edge_arr:
            gt_graph.add_edge(edge[0], edge[1])
        if return_all_nodes:
            all_nodes[skeleton_id] = node_arr

    if return_all_nodes:
        all_nodes = np.vstack(all_nodes)
        return gt_graph, all_nodes
    return gt_graph


def node_edge_to_networkx(
    nodes, edges, skeleton_resolution=None, return_all_nodes=False, data_type=np.uint16
):
    """
    The function `node_edge_to_networkx` converts a set of nodes and edges into a networkx graph object,
    optionally returning all nodes as well.

    :param nodes: A list of arrays, where each array represents the coordinates of nodes in a skeleton.
    Each array corresponds to a different skeleton
    :param edges: The `edges` parameter is a list of lists, where each inner list represents an edge in
    the graph. Each inner list contains two elements, which are the indices of the nodes that the edge
    connects
    :param skeleton_resolution: The `skeleton_resolution` parameter is an optional argument that
    specifies the resolution of the skeleton. It is used to scale the node coordinates in the `node_arr`
    array. If `skeleton_resolution` is provided, the node coordinates are multiplied by the resolution
    value. This is useful when working with
    :param return_all_nodes: The `return_all_nodes` parameter is a boolean flag that determines whether
    or not to return all the nodes in the graph. If set to `True`, the function will return both the
    graph and an array containing all the nodes. If set to `False` (default), the function will only
    return, defaults to False (optional)
    :return: a networkx graph object. If the parameter `return_all_nodes` is set to `True`, it also
    returns an array of all the nodes in the graph.
    """

    gt_graph = nx.Graph()
    count = 0
    all_nodes = [None] * len(nodes)
    for skeleton_id, node_arr in enumerate(nodes):
        if len(edges[skeleton_id]) == 0:
            continue
        node_arr = node_arr.astype(data_type)
        if skeleton_resolution is not None:
            node_arr = node_arr * skeleton_resolution
        # augment the node index
        edge_arr = edges[skeleton_id].astype(data_type) + count
        for node in node_arr:
            # unit: physical
            gt_graph.add_node(
                count, skeleton_id=skeleton_id, z=node[0], y=node[1], x=node[2]
            )
            count += 1
        for edge in edge_arr:
            gt_graph.add_edge(edge[0], edge[1])
        if return_all_nodes:
            all_nodes[skeleton_id] = node_arr
    if return_all_nodes:
        all_nodes = np.vstack(all_nodes)
        return gt_graph, all_nodes
    return gt_graph

def vol_to_skel(
    labels,
    scale=4,
    const=500,
    obj_ids=None,
    dust_size=100,
    res=(32, 32, 30),
    num_thread=1,
):
    try:
        import kimimaro
    except:
        raise "need to install kimimaro: pip install kimimaro"
    """
    This function takes in a label image and returns the skeletonized version of the
    objects in the image using the Kimimaro library.

    :param labels: The input labels represent a 3D volume where each voxel is assigned a unique integer
    label. These labels typically represent different objects or regions in the volume
    :param scale: The scale parameter determines the scale at which the skeletonization is performed. It
    is used to control the level of detail in the resulting skeleton. Higher values of scale will result
    in a coarser skeleton, while lower values will result in a more detailed skeleton, defaults to 4
    (optional)
    :param const: The `const` parameter is a physical unit that determines the resolution of the
    skeletonization process. It represents the distance between two points in the skeletonized output. A
    higher value of `const` will result in a coarser skeleton, while a lower value will result in a
    finer skeleton, defaults to 500 (optional)
    :param obj_ids: The obj_ids parameter is a list of object IDs that specifies which labels in the
    input image should be processed. If obj_ids is set to None, it will default to all unique labels
    greater than 0 in the input image
    :param dust_size: The dust_size parameter specifies the minimum size (in terms of number of voxels)
    for connected components to be considered as valid objects. Connected components with fewer voxels
    than the dust_size will be skipped and not processed, defaults to 100 (optional)
    :param res: The "res" parameter specifies the resolution of the input volume data. It is a tuple of
    three values representing the voxel size in each dimension. For example, (32, 32, 30) means that the
    voxel size is 32 units in the x and y dimensions, and 30
    :param num_thread: The `num_thread` parameter specifies the number of threads to use for parallel
    processing. A value of 1 means single-threaded processing, while a value greater than 1 indicates
    multi-threaded processing. A value of 0 or less indicates that all available CPU cores should be
    used for parallel processing, defaults to 1 (optional)
    :return: The function `skeletonize` returns the result of the `kimimaro.skeletonize` function, which
    is the skeletonized version of the input labels.
    """
    if obj_ids is None:
        obj_ids = np.unique(labels)
        obj_ids = list(obj_ids[obj_ids > 0])
    return kimimaro.skeletonize(
        labels,
        teasar_params={
            "scale": scale,
            "const": const,  # physical units
            "pdrf_exponent": 4,
            "pdrf_scale": 100000,
            "soma_detection_threshold": 1100,  # physical units
            "soma_acceptance_threshold": 3500,  # physical units
            "soma_invalidation_scale": 1.0,
            "soma_invalidation_const": 300,  # physical units
            "max_paths": 50,  # default  None
        },
        object_ids=obj_ids,  # process only the specified labels
        # object_ids=[ ... ], # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        dust_threshold=dust_size,  # skip connected components with fewer than this many voxels
        #       anisotropy=(30,30,30), # default True
        anisotropy=res,  # default True
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        progress=True,  # default False, show progress bar
        parallel=num_thread,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
    )[0]
