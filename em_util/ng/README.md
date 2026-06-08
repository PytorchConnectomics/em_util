# neuroglancer scripts

Reusable tools for pushing connectomics outputs to Google Cloud Storage as
precomputed neuroglancer layers, and for meshing / downsampling them. This is
the canonical home for generic NG-upload utilities (project repos call these as
thin glue; don't fork copies into them).

## Dependencies

Precomputed image/segmentation IO uses **cloud-volume** (customized fork for
hierarchical folder structure) and downsampling uses **tinybrain**:
```
git clone https://github.com/donglaiw/cloud-volume.git && cd cloud-volume && pip install -e .
pip install tinybrain
```
Meshing uses **igneous** (keep it in its own env so it doesn't pin cloudvolume
for everything else):
```
python -m venv /tmp/igneous_env && /tmp/igneous_env/bin/pip install igneous-pipeline h5py
# or the fork: git clone https://github.com/donglaiw/igneous.git && pip install -e .
```

GCS auth: point `GOOGLE_APPLICATION_CREDENTIALS` (or the `--key` arg the scripts
take) at a service-account JSON key.

## Scripts

| script | needs | what |
|---|---|---|
| `upload_gt_skeleton.py` | cloudvolume | networkx `skeleton.pkl` -> precomputed skeleton layer (one Skeleton per node `id`) |
| `upload_seg_volume.py` | cloudvolume, h5py | dense seg h5 `(X,Y,Z)` -> precomputed segmentation layer (Z-slab stream, no transpose) |
| `upload_affinity_chunks.py` | cloudvolume, h5py | chunked affinity `*.h5.chunks/` (CZYX core tiles) -> one precomputed image layer; resumable, shardable |
| `compute_label_sizes.py` | numpy, h5py | stream-bincount a seg h5 -> `sizes.npy` (drives the mesh whitelist) |
| `mesh_seg_volume.py` | igneous | downsample + mesh (`object_ids` > N voxels, global whitelist) + manifests |
| `downsample_ng.py` | igneous | add a mip pyramid to any existing cloud layer (image avg / seg mode) |
| `downsample_seg_local.py` | tinybrain, h5py | local seg h5 -> coarser-mip h5 (mode pooling; matches the cloud mip) |

## PNI bucket conventions

Bucket `gs://princeton-eric-medial-entorhinal-cortex-scratch/` (shared NISB
dataset; seed101 = test), key `lib/keys/hi-mc-collab-f8437ce66bd3.json`. A
`buckets.get` 403 (metadata perm) is expected — object read/write still works.
NISB seed101 layers: res `[9,9,20]` nm, vol `[3000,3000,1350]` XYZ, chunk
`[128,128,64]`, raw. Affinity layers are float32 3-channel.

Axis order: the connectomics decode seg h5 (`main`, `(X,Y,Z)`) is already in
CloudVolume XYZ order (upload verbatim). The chunked-raw affinity is CZYX
`(3,Z,Y,X)` core tiles and is transposed by `upload_affinity_chunks.py`.

## Worked example — seg upload + mesh (seed101 cc3d @0.66)

```bash
EMU=/projects/weilab/weidf/lib/miniconda3/envs/emu/bin/python   # cloudvolume+h5py+tinybrain
IGN=/tmp/igneous_env/bin/python
KEY=lib/keys/hi-mc-collab-f8437ce66bd3.json
NG=lib/em_util/em_util/ng
SEG=outputs/.../seed101/decoded_x1_ch0-1-2_affinity_cc_numba-0-0.66.h5
CP=gs://princeton-eric-medial-entorhinal-cortex-scratch/nisb_seed101_cc3d_t066

$EMU $NG/upload_seg_volume.py  --volume $SEG --cloudpath $CP --key $KEY --resolution 9 9 20
$EMU $NG/compute_label_sizes.py --volume $SEG --out /tmp/sizes.npy
$IGN $NG/mesh_seg_volume.py     --cloudpath $CP --key $KEY --sizes /tmp/sizes.npy \
     --min-voxels 10000 --mesh-mip 2 --error 40 --parallel 8
$EMU $NG/downsample_seg_local.py --volume $SEG --mip 2 --resolution-out 36 36 20  # local mip2
```

## Worked example — chunked affinity -> one image layer (zebrafinch)

`*.h5.chunks/` of `chunk_z{Z}_y{Y}_x{X}.h5`, each float16 CZYX core tile with
`chunk_start_zyx`/`chunk_stop_zyx` attrs; tiles cover the volume with no overlap.
float16 is not a precomputed dtype — pick uint8 (compact) or float32 (matches
the NISB affinity layers; ~4x larger). Storage `--chunk-size` should divide the
tiling step (1008) so sharded writes don't race on seam chunks.

```bash
$EMU $NG/upload_affinity_chunks.py --chunks-dir <...>.h5.chunks --cloudpath $CP --key $KEY \
     --dtype uint8 --resolution 10 10 10 --create-info        # run ONCE
$EMU $NG/upload_affinity_chunks.py --chunks-dir <...>.h5.chunks --cloudpath $CP --key $KEY \
     --dtype uint8 --shard-id $i --num-shards 13              # sharded, resumable
$IGN $NG/downsample_ng.py --cloudpath $CP --key $KEY --num-mips 5   # low-zoom mips
```
