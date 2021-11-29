#ifndef C_REGION_H
#define C_REGION_H

#include "basic.h"

void cpp_merge_id(
    SegID*          id1,
    SegID*          id2,
    SegID*          mapping,
    SegID           num_edge,
    SegID           num_id,
    SegID           id_thres);

void cpp_merge_id_byAff(
     SegID* id1,
     SegID* id2,
     SegID* mapping,
     AffValue* score,
     SegID  num_edge,
     SegID  num_id,
     SegID  id_thres,
     AffValue  aff_thres);

void cpp_merge_id_byCount(
     SegID* id1,
     SegID* id2,
     SegID* mapping,
     SegID* count,
     SegID  num_edge,
     SegID  num_id,
     SegID  id_thres,
     SegID  count_thres,
     SegID  dust_thres);


void cpp_merge_id_byAffCount(
     SegID* id1,
     SegID* id2,
     SegID* mapping,
     AffValue* score,
     SegID* count,
     SegID  num_edge,
     SegID  num_id,
     SegID  id_thres,
     AffValue  aff_thres,
     SegID  count_thres,
     SegID  dust_thres);

void cpp_remove_small(
    SegID*   mapping,
    SegID*   count,
    SegID    num_id,
    SegID    dust_thres);
#endif
