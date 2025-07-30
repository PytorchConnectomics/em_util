import numpy as np


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


# Columns: Nr  flags  red1 green1 blue1 pattern1  red2 green2 blue2 pattern2  anchorx anchory anchorz  parentnr childnr prevnr nextnr   collapsednr   bboxx1 bboxy1 bboxz1 bboxx2 bboxy2 bboxz2   "name"
def read_vast_seg(fn):
    a = open(fn).readlines()
    # remove comments
    st_id = 0
    while a[st_id][0] in ["%", '\n']:
        st_id += 1
    
    # remove segment name
    out = np.zeros((len(a) - st_id, 24), dtype=int)
    name = [None] * (len(a) - st_id)
    for i in range(st_id, len(a)):
        out[i - st_id] = np.array(
            [int(x) for x in a[i][: a[i].find('"')].split(" ") if len(x) > 0]
        )
        name[i - st_id] = a[i][a[i].find('"') + 1 : a[i].rfind('"')]
    return out, name


def write_vast_anchor(fn, bb):
    # plain structure
    # x0,y0,z0,x1,y1,z1
    oo = open(fn, "w")
    vast_str0 = (
        "0   0   0 0 0 0   0 0 0 0   -1 -1 -1  0 0 0 1   0   -1 -1 -1 -1 -1 -1"
        '   "Background"\n'
    )
    oo.write(vast_str0)

    vast_str = (
        "%d   0   255 0 0 0   255 0 0 0   %d %d %d  0 0 %d %d %d   %d %d %d %d"
        ' %d %d "seg%d"\n'
    )
    for i in range(bb.shape[0]):
        nn = i + 2 if i != bb.shape[0] - 1 else 0
        oo.write(
            vast_str
            % (
                i + 1,
                (bb[i, 0] + bb[i, 3]) // 2,
                (bb[i, 1] + bb[i, 4]) // 2,
                (bb[i, 2] + bb[i, 5]) // 2,
                i,
                nn,
                i + 1,
                bb[i, 0],
                bb[i, 1],
                bb[i, 2],
                bb[i, 3],
                bb[i, 4],
                bb[i, 5],
                i + 1,
            )
        )
    oo.close()


def write_vast_anchor_tree_by_id(
    fn, sids, bbs, nn=["good", "bad"], pref="seg", id_rl=None
):
    # x0,y0,z0,x1,y1,z1
    # sid: min>=1
    # mid: extra merge
    sid_m = sids[0].max()
    for i in range(1, len(sids)):
        sid_m = max(sid_m, sids[i].max())
    sid_m += 1

    oo = open(fn, "w")
    vast_str0 = (
        "0 0 0 0 0 0 0 0 0 0 -1 -1 -1 0 0 0 %d 0 -1 -1 -1 -1 -1"
        ' -1  "Background"\n' % (sid_m)
    )
    oo.write(vast_str0)

    ccs = np.array(get_spaced_colors(bbs.shape[0]))
    ccs = ccs[np.random.permutation(ccs.shape[0])]
    vast_str = (
        "%d 1 %d %d %d 0 255 0 0 0 %d %d %d %d 0 %d %d %d %d %d %d %d %d %d"
        ' "%s%d"\n'
    )
    cid = [None] * len(sids)
    cc_id = 0
    out = [""] * bbs.shape[0]
    for ii, sid in enumerate(sids):
        numS = len(sid) if isinstance(sid, list) else sid.size
        if numS == 1:
            sid = [int(sid)]
        for i in range(numS):
            bb = bbs[sid[i]]
            prevn = sid[i - 1] if i != 0 else 0
            nextn = sid[i + 1] if i != numS - 1 else 0
            parent = sid_m + ii
            if id_rl is not None:
                i0 = i - 1
                while id_rl[prevn] != prevn:
                    prevn = sid[i0 - 1] if i0 != 0 else 0
                    i0 -= 1
                i0 = i + 1
                while id_rl[nextn] != nextn:
                    nextn = sid[i0 + 1] if i0 != numS - 1 else 0
                    i0 += 1
            try:
                out[sid[i] - 1] = vast_str % (
                    sid[i],
                    ccs[cc_id, 0],
                    ccs[cc_id, 1],
                    ccs[cc_id, 2],
                    (bb[0] + bb[3]) // 2,
                    (bb[1] + bb[4]) // 2,
                    (bb[2] + bb[5]) // 2,
                    parent,
                    prevn,
                    nextn,
                    parent,
                    bb[0],
                    bb[1],
                    bb[2],
                    bb[3],
                    bb[4],
                    bb[5],
                    pref,
                    sid[i],
                )
            except:
                import pdb

                pdb.set_trace()
            if i == 0:
                cid[ii] = sid[0]
            cc_id += 1

    # modify if seg has children
    if id_rl is not None:
        ui, uc = np.unique(id_rl, return_counts=True)
        for ii in ui[uc > 1]:
            jjs = np.where(id_rl == ii)[0]
            jjs = jjs[jjs != ii]
            # if parent
            tmp = out[ii - 1].split(" ")
            tmp[14] = str(jjs[0])
            out[ii - 1] = " ".join(tmp)
            # if child
            for jid, jj in enumerate(jjs):
                tmp = out[jj - 1].split(" ")
                tmp[13] = str(ii)
                tmp[15] = "0" if jid == 0 else str(jjs[jid - 1])
                tmp[16] = "0" if jid == len(jjs) - 1 else str(jjs[jid + 1])
                out[jj - 1] = " ".join(tmp)

    for i in range(bbs.shape[0]):
        oo.write(out[i])

    ccs = get_spaced_colors(len(nn))
    for nid, n in enumerate(nn):
        prevn = sid_m + nid - 1 if nid != 0 else 0
        nextn = sid_m + nid + 1 if nid != len(nn) - 1 else 0
        vast_strF = (
            "%d   1   %d %d %d %d   %d %d %d %d   -1 -1 -1  0 %d %d %d   %d  "
            ' -1 -1 -1 -1 -1 -1   "%s"\n'
            % (
                sid_m + nid,
                ccs[nid][0],
                ccs[nid][1],
                ccs[nid][2],
                sid_m + nid,
                ccs[nid][0],
                ccs[nid][1],
                ccs[nid][2],
                sid_m + nid,
                cid[nid],
                prevn,
                nextn,
                sid_m + nid,
                n,
            )
        )
        oo.write(vast_strF)
    oo.close()


def vast_meta_relabel(
    fn, kw_bad=["bad", "del"], kw_nm=["merge"], do_print=False
):
    # if there is meta data
    print("load meta")
    dd, nn = read_vast_seg(fn)
    rl = np.arange(1 + dd.shape[0], dtype=np.uint16)
    pid = np.unique(dd[:, 13])
    if do_print:
        print(
            ",".join(
                [
                    nn[x]
                    for x in np.where(np.in1d(dd[:, 0], pid))[0]
                    if "Imported Segment" not in nn[x]
                ]
            )
        )

    pid_b = []
    if len(kw_bad) > 0:
        # delete seg id
        pid_b = [
            i
            for i, x in enumerate(nn)
            if max([y in x.lower() for y in kw_bad])
        ]
        bid = np.where(np.in1d(dd[:, 13], pid_b))[0]
        bid = np.hstack([pid_b, bid])
        if len(bid) > 0:
            rl[bid] = 0
        print("found %d bad" % (len(bid)))

    # not to merge
    kw_nm += ["background"]
    pid_nm = [
        i for i, x in enumerate(nn) if max([y in x.lower() for y in kw_nm])
    ]
    # pid: all are children of background
    pid_nm = np.hstack([pid_nm, pid_b])
    print("apply meta")
    # hierarchical
    for p in pid[np.in1d(pid, pid_nm, invert=True)]:
        rl[dd[dd[:, 13] == p, 0]] = p
    # consolidate root labels
    for u in np.unique(rl[np.where(rl[rl] != rl)[0]]):
        u0 = u
        while u0 != rl[u0]:
            rl[rl == u0] = rl[u0]
            u0 = rl[u0]
        #print(u, rl)
    return rl
