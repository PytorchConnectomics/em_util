from libc.stdint cimport uint8_t, uint64_t, uint32_t
from libcpp cimport bool
import numpy as np
cimport numpy as np

def merge_id(id1, id2, score=None, count=None, id_thres=0, aff_thres=1, count_thres=50, dust_thres = 50):
    # avoid wrong memory access
    if len(id1) != 0:
        if not id1.flags['C_CONTIGUOUS']:
            id1 = np.ascontiguousarray(id1)
        if not id2.flags['C_CONTIGUOUS']:
            id2 = np.ascontiguousarray(id2)
        mid = max(id1.max(), id2.max()) + 1
        if count is not None:
            mid = max(mid, len(count))
        mapping = np.arange(mid).astype(id1.dtype)
    else: # if id is empty, count is not empty
        mapping = np.arange(len(count)).astype(np.uint32)

    if len(id1) == 0:
        __remove_small(mapping, count, dust_thres)
    elif (score is None) and (count is None):
        # merge by id
        __merge_id(id1, id2, mapping, id_thres)
    elif (score is not None) and (count is None):
        # merge by id and aff
        __merge_id_byAff(id1, id2, mapping, score, id_thres, aff_thres)
    elif (score is None) and (count is not None):
        # merge by id and aff
        __merge_id_byCount(id1, id2, mapping, count, id_thres, count_thres, dust_thres)
    elif (score is not None) and (count is not None):
        # merge by id, aff and count
        __merge_id_byAffCount(id1, id2, mapping, score, count, id_thres, aff_thres, count_thres, dust_thres)
    return mapping 

def __remove_small(np.ndarray[np.uint32_t, ndim=1] mapping, 
                 np.ndarray[np.uint32_t, ndim=1] count, 
                 dust_thres):
    cdef uint32_t* mapping_data;
    cdef uint32_t* count_data;

    mapping_data = &mapping[0];
    count_data = &count[0];
    cpp_remove_small(mapping_data, count_data, len(mapping), dust_thres)

def __merge_id_byAff(np.ndarray[np.uint32_t, ndim=1] id1,
                 np.ndarray[np.uint32_t, ndim=1] id2,
                 np.ndarray[np.uint32_t, ndim=1] mapping, 
                 np.ndarray[np.uint8_t, ndim=1] score, 
                 id_thres, aff_thres):
    cdef uint32_t*  id1_data;
    cdef uint32_t*  id2_data;
    cdef uint8_t*   score_data;
    cdef uint32_t*  mapping_data;

    id1_data = &id1[0];
    id2_data = &id2[0];
    score_data = &score[0];
    mapping_data = &mapping[0];

    cpp_merge_id_byAff(id1_data, id2_data, mapping_data, score_data, len(id1), len(mapping), \
                id_thres, aff_thres)

def __merge_id_byAffCount(np.ndarray[np.uint32_t, ndim=1] id1,
                 np.ndarray[np.uint32_t, ndim=1] id2,
                 np.ndarray[np.uint32_t, ndim=1] mapping, 
                 np.ndarray[np.uint8_t, ndim=1] score, 
                 np.ndarray[np.uint32_t, ndim=1] count, 
                 id_thres, aff_thres, count_thres, dust_thres):
    cdef uint32_t*  id1_data;
    cdef uint32_t*  id2_data;
    cdef uint8_t*   score_data;
    cdef uint32_t*  mapping_data;
    cdef uint32_t*  count_data;

    id1_data = &id1[0];
    id2_data = &id2[0];
    score_data = &score[0];
    mapping_data = &mapping[0];
    count_data = &count[0];

    cpp_merge_id_byAffCount(id1_data, id2_data, mapping_data, score_data, count_data, len(id1), len(mapping), \
                id_thres, aff_thres, count_thres, dust_thres)


def __merge_id_byCount(np.ndarray[np.uint32_t, ndim=1] id1,
                 np.ndarray[np.uint32_t, ndim=1] id2,
                 np.ndarray[np.uint32_t, ndim=1] mapping, 
                 np.ndarray[np.uint32_t, ndim=1] count, 
                 id_thres, count_thres, dust_thres):
    cdef uint32_t*  id1_data;
    cdef uint32_t*  id2_data;
    cdef uint32_t*  mapping_data;
    cdef uint32_t*  count_data;

    id1_data = &id1[0];
    id2_data = &id2[0];
    mapping_data = &mapping[0];
    count_data = &count[0];

    cpp_merge_id_byCount(id1_data, id2_data, mapping_data, count_data, len(id1), len(mapping), \
                id_thres, count_thres, dust_thres)


def __merge_id(np.ndarray[np.uint32_t, ndim=1] id1,
                 np.ndarray[np.uint32_t, ndim=1] id2,
                 np.ndarray[np.uint32_t, ndim=1] mapping, 
                 id_thres):
    '''Find the global mapping of IDs from the region graph without count constraints
    
    The region graph should be ordered by decreasing affinity and truncated
    at the affinity threshold.
    :param id1: a 1D array of the lefthand side of the two adjacent regions
    :param id2: a 1D array of the righthand side of the two adjacent regions
    :returns: a 1D array of the global IDs per local ID
    '''
    cdef uint32_t* id1_data;
    cdef uint32_t* id2_data;
    cdef uint32_t* mapping_data;
    id1_data = &id1[0];
    id2_data = &id2[0];
    mapping_data = &mapping[0];

    cpp_merge_id(id1_data, id2_data, mapping_data, len(id1), len(mapping), id_thres);

cdef extern from "cpp/region_graph.h":
    void cpp_merge_id(
        uint32_t*          id1,
        uint32_t*          id2,
        uint32_t*          mapping,
        uint32_t           num_edge,
        uint32_t           num_id,
        uint32_t           id_thres);

    void cpp_merge_id_byAff(
         uint32_t* id1,
         uint32_t* id2,
         uint32_t* mapping,
         uint8_t* score,
         uint32_t  num_edge,
         uint32_t  num_id,
         uint32_t  id_thres,
         uint8_t  aff_thres);

    void cpp_merge_id_byCount(
         uint32_t* id1,
         uint32_t* id2,
         uint32_t* mapping,
         uint32_t* count,
         uint32_t  num_edge,
         uint32_t  num_id,
         uint32_t  id_thres,
         uint32_t  count_thres,
         uint32_t  dust_thres);

    void cpp_merge_id_byAffCount(
         uint32_t* id1,
         uint32_t* id2,
         uint32_t* mapping,
         uint8_t* score,
         uint32_t* count,
         uint32_t  num_edge,
         uint32_t  num_id,
         uint32_t  id_thres,
         uint8_t  aff_thres,
         uint32_t  count_thres,
         uint32_t  dust_thres);

    void cpp_remove_small(
        uint32_t*   mapping,
        uint32_t*   count,
        uint32_t    num_id,
        uint32_t    dust_thres);
