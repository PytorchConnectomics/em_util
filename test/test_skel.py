import pickle
import sys,os
from funlib import evaluate
import h5py
import numpy as np
import networkx as nx
from em_util.io import readH5, skelToLength, skelToNetworkX

def test_len(skel_pickle_path, res= [30,6,6]):
    nodes, edges = pickle.load(open(skel_pickle_path, 'rb'), encoding="latin1")
   
    skel_len = [skelToLength(nodes[x], edges[x], res) for x in len(nodes)]
    print('Skelton length:', skel_len)

def test_erl(skel_pickle_path, seg_path, res= [30,6,6]):
    nodes, edges = pickle.load(open(skel_pickle_path, 'rb'), encoding="latin1")
    seg = readH5(seg_path)
   
    gt_graph, node_segment_lut = skelToNetworkX(nodes, edges, [seg], res)
    
    scores = evaluate.expected_run_length(
                    skeletons=gt_graph,
                    skeleton_id_attribute='skeleton_id',
                    edge_length_attribute='length',
                    node_segment_lut=node_segment_lut[0],
                    skeleton_position_attributes=['z', 'y', 'x'])
    print('ERL:', scores)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('need an argument to select the test')
    opt = sys.argv[1]
    if opt=='0': # length
        test_len(sys.argv[2])
    elif opt=='1': # erl
        if len(sys.argv) < 4:
            print('need two arguments: skel_pickle_path, seg_path')
        test_erl(sys.argv[2], sys.argv[3])
