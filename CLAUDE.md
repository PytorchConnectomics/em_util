# em_util — Electron Microscopy Utility Library

Utility library for EM (electron microscopy) connectomics: volume I/O, segmentation ops, evaluation metrics, neuroglancer visualization, and SLURM job management.

**Install:** `pip install em_util` or `pip install -e .` from source
**License:** MIT | **Author:** Donglai Wei
**Repo:** https://github.com/PytorchConnectomics/em_util

## Core Dependencies

scipy, numpy, networkx, h5py, imageio, scikit-image, tqdm, connected-components-3d (cc3d), pyyaml

Optional: cloudvolume, zarr, aicsimageio (nd2), kimimaro (skeletons), neuroglancer, igneous (meshes)

## Conventions

- **Volume axis order:** ZYX (depth, height, width) — standard in EM connectomics
- **Neuroglancer axis order:** XYZ (reversed from volume convention)
- **Segmentation volumes:** integer arrays where each value is a segment ID; 0 typically means background
- **Bounding boxes:** stored as `[y0, y1, x0, x1]` (2D) or `[z0, z1, y0, y1, x0, x1]` (3D)

## Quick Import Patterns

```python
# I/O (most common)
from em_util.io import read_vol, write_h5, read_h5, read_image, write_image
from em_util.io import compute_bbox_all, seg_to_rgb, rgb_to_seg
from em_util.io import vol_to_skel, skel_to_length

# Segmentation operations
from em_util.seg import seg_to_count, seg_relabel, seg_remove_small
from em_util.seg import seg_to_cc, seg3d_to_cc, seg_biggest_cc
from em_util.seg import seg_to_iou

# Evaluation metrics
from em_util.eval import adapted_rand

# Neuroglancer visualization
from em_util.ng import NgDataset, ng_layer

# SLURM cluster
from em_util.cluster import get_parser, write_slurm, write_slurm_all

# Union-Find data structure
from em_util.io import UnionFind
```

---

## API Reference

### em_util.io — File I/O

#### Universal Reader
```python
read_vol(filename, dataset=None)
```
Reads any supported format based on file extension. Returns numpy array.
- `.npy` — numpy binary
- `.pkl` — pickle
- `.tif/.tiff` — TIFF stack (returns ZYX)
- `.png/.jpg` — 2D image
- `.h5` — HDF5 (reads first dataset if `dataset=None`)
- `.zarr` — Zarr archive
- `.nd2` — Nikon microscopy
- `gs://` or `precomputed://` — CloudVolume precomputed

```python
read_vol_bbox(filename, bbox, dataset=None, ratio=1, resize_order=0, image_type="image")
```
Read a subregion of a volume. `bbox = [z0, z1, y0, y1, x0, x1]`.

#### HDF5
```python
read_h5(filename, dataset=None)        # Returns numpy array; reads first key if dataset=None
write_h5(filename, data, dataset="main")  # Writes array to HDF5
```

#### Images
```python
read_image(filename, image_type="image", ratio=1, resize_order=0, data_type=None, crop=None)
write_image(filename, image, image_type="image")
read_image_folder(filename, index=None, image_type="image", ratio=1, resize_order=0, crop=None)
write_image_folder(filename, data, index=None, image_type="image")
```
`image_type`: `"image"` for grayscale, `"seg"` for segmentation (nearest-neighbor resize), `"rgb"` for color.

#### Other Formats
```python
read_pkl(filename)                     # Read pickle
write_pkl(filename, content)           # Write pickle
read_txt(filename)                     # Read text file lines
write_txt(filename, content)           # Write string to text file
write_json(filename, content)          # Write dict to JSON
read_yml(filename)                     # Read YAML file
write_gif(filename, data, duration=0.5)  # Write numpy array as GIF
```

#### Utilities
```python
mkdir(foldername, opt="")              # Create directory; opt="rm" to remove existing first
get_file_vol_shape(filename, dataset_name=None)  # Get volume dimensions without loading
get_seg_dtype(mid)                     # Returns smallest uint dtype that fits max ID `mid`
```

### em_util.io — Image Processing

```python
resize_image(image, ratio=1, resize_order=0)     # Resize; order=0 for segmentation
seg_to_rgb(seg)                                    # Segmentation IDs -> RGB color image
rgb_to_seg(seg)                                    # RGB color image -> segmentation IDs
im_adjust(I, threshold=0.3, auto_scale=False)     # Histogram-based intensity adjustment
im_trim_by_intensity(I, threshold=0.5, return_ind=False)  # Crop dark borders
im2col(A, BSZ, stepsize=1)                         # Image to overlapping column blocks
```

### em_util.io — Bounding Boxes

```python
compute_bbox(seg, do_count=False)                  # BBox of nonzero region; optionally count
compute_bbox_all(seg, do_count=True, uid=None)     # BBox per segment ID -> [N, 4 or 6] array
compute_bbox_all_2d(seg, do_count=True, uid=None)  # 2D version: [y0, y1, x0, x1]
compute_bbox_all_3d(seg, do_count=True, uid=None)  # 3D version: [z0, z1, y0, y1, x0, x1]
merge_bbox(bbox_a, bbox_b)                         # Merge two bounding boxes
```
Returns: `(uid, uid_count, bbox_matrix)` when `do_count=True`.

### em_util.io — Skeleton Operations

Requires `kimimaro` for extraction.

```python
vol_to_skel(labels, scale=4, const=500, obj_ids=None, dust_size=100, res=[30,8,8], num_thread=1)
```
Extract skeletons from labeled volume. Returns dict of kimimaro Skeleton objects.

```python
skel_to_length(vertices, edges, res=[1,1,1])       # Cable length from skeleton
skel_to_networkx(skeletons, skeleton_resolution=None, return_all_nodes=True, data_type="float32")
```
Convert kimimaro skeletons to NetworkX graph for analysis.

### em_util.io — Array Utilities

```python
arr_dim_convertor(arr, factor)         # Convert flat index <-> ZYX coords using shape factor
arr_to_str(arr)                        # Array -> comma-separated string
print_arr(arr, num=3)                  # Pretty print with formatting
get_query_count(ui, uc, qid, mm=0)    # Look up counts for query IDs from unique/count arrays
```

### em_util.io — Chunked Processing (for large volumes)

```python
vol_func_chunk(input_file, vol_func, output_file=None, output_chunk=None,
               chunk_num=1, no_tqdm=False, dtype=None)
```
Apply a function to a volume in chunks. Reads from HDF5, processes chunk-by-chunk, optionally writes output.

```python
vol_downsample_chunk(input_file, ratio=[1,2,2], output_file=None,
                     output_chunk=None, chunk_num=1)
```
Downsample a large volume in chunks.

```python
compute_bbox_all_chunk(seg_file, do_count=True, uid=None, chunk_num=1)
```
Compute bounding boxes from a large segmentation file in chunks.

### em_util.io — Tiled Volume I/O

```python
read_tile_volume(filenames, z0p, z1p, y0p, y1p, x0p, x1p, tile_sz, tile_st=None,
                 tile_dtype=np.uint8, tile_resize_mode=0, tile_type="image")
```
Read a subregion from a tiled volume dataset. `tile_sz` is the tile dimensions.

### em_util.io — Union-Find

```python
uf = UnionFind()
uf.add(x)                    # Add element
uf.union(x, y)               # Merge two sets
uf.find(x)                   # Find root representative
uf.connected(x, y)           # Check if same set
uf.components()              # List all components
uf.component_mapping()       # Dict: root -> set of members
uf.component_relabel_arr()   # Numpy array for relabeling: arr[old_id] = new_id
```

---

### em_util.seg — Segmentation Operations

#### Statistics
```python
seg_to_count(seg, do_sort=True, rm_zero=True)  # Returns (unique_ids, counts), sorted by count desc
seg3d_to_zcount(seg)                            # Per-slice segment presence counts
```

#### Manipulation
```python
seg_relabel(seg, uid=None, nid=None, do_sort=False, do_type=True)
    # Relabel segment IDs. If uid/nid given, maps uid->nid. If do_sort, relabels by size.
seg_remove_id(seg, bid, invert=False)           # Zero out specific IDs (or keep only them if invert)
seg_remove_small(seg, threshold, invert=False)  # Remove segments with fewer than `threshold` voxels
seg_biggest(seg)                                # Keep only the largest segment
seg_widen_border(seg, tsz_h=1)                  # Widen borders between segments
```

#### Morphological Operations
```python
seg_binary_fill_holes(seg)                      # Fill holes per-label
seg_binary_opening(seg, iteration=1)            # Morphological opening per-label
seg_binary_closing(seg, iteration=1)            # Morphological closing per-label
```

#### Connected Components
```python
seg_to_cc(seg, num_conn=None)                   # Label connected components (2D: 4-conn, 3D: 6-conn)
seg3d_to_cc(seg3d, num_conn=6)                  # 3D connected components using cc3d
seg_remove_small_cc(seg, num_conn=None, threshold=100, invert=False)
seg_biggest_cc(seg, num_conn=None)              # Keep largest connected component per label
```

#### IoU (Intersection over Union)
```python
seg_to_iou(seg0, seg1, uid0=None, bb0=None, uid1=None, uc1=None, th_iou=0)
    # Compute IoU matrix between two segmentations. Returns (iou_matrix, uid0, uid1).
segs_to_iou(get_seg, index, th_iou=0.3)
    # Track segments across slices by IoU. get_seg(i) returns segmentation for slice i.
```

---

### em_util.eval — Evaluation Metrics

```python
from em_util.eval import adapted_rand
from em_util.eval.seg import voi, split_vi, confusion_matrix, get_binary_jaccard

adapted_rand(seg, gt, all_stats=False)
    # SNEMI3D Adapted Rand error. Returns float (or (error, precision, recall) if all_stats=True).

voi(reconstruction, groundtruth, ignore_reconstruction=(0,), ignore_groundtruth=(0,))
    # Variation of Information. Returns [split_vi, merge_vi].

split_vi(x, y, ignore_x=(0,), ignore_y=(0,))
    # Split and merge VI components. Returns (H(x|y), H(y|x)).

confusion_matrix(pred, gt, thres=0.5)
    # Binary confusion matrix. Returns (tp, fp, fn).

get_binary_jaccard(pred, gt, thres=0.5)
    # Binary IoU/Jaccard score.
```

---

### em_util.ng — Neuroglancer Visualization

```python
ng_layer(data, res=[30, 8, 8], oo=[0, 0, 0], tt="image")
    # Create a neuroglancer layer from numpy array.
    # tt: "image" for grayscale, "segmentation" for labels.
    # res: voxel resolution in nm [z, y, x].
```

```python
ds = NgDataset(volume_size, resolution=[8,8,30], mip_ratio=[[1,1,1],[2,2,1]],
               chunk_size=[64,64,64], offset=[0,0,0], cloudpath="")

ds.create_info(cloudpath, data_type="uint8", num_channel=1)
ds.create_tile(getVolume, cloudpath, data_type="uint8", mip_levels=[0])
    # getVolume(z_range): function returning volume slice
ds.create_mesh(cloudpath, mip_level=0, volume_size=None, num_thread=1)
```

---

### em_util.cluster — SLURM Job Submission

```python
from em_util.cluster import get_parser, write_slurm, write_slurm_all

parser = get_parser()
args = parser.parse_args()
# Provides: args.task, args.cluster, args.cmd, args.env, args.job_id, args.job_num,
#           args.chunk_num, args.neuron, args.ratio, args.partition, args.memory, args.run_time

write_slurm(cmd, file_name="slurm.sh", job_id=1, job_num=1, partition="", num_cpu=2,
            num_gpu=0, memory="50GB", time="0-12:00")
    # Write a single SLURM job script.

write_slurm_all(cmd, file_name="slurm.sh", job_num=1, partition="", num_cpu=2,
                num_gpu=0, memory="50GB", time="0-12:00")
    # Write SLURM array job script.
```

---

### em_util.vast — VAST Annotation Format

```python
from em_util.vast import create_mip_images, read_vast_seg, write_vast_anchor_tree_by_id

create_mip_images(get_input_image, get_output_name, zran, level_ran, resize_order=0, do_seg=False)
read_vast_seg(fn)                         # Read VAST segmentation metadata
write_vast_anchor_tree_by_id(fn, sids, bbs, nn=None, pref="s", id_rl=None)
    # Write hierarchical VAST anchor file with bounding boxes.
```

---

## Common Usage Patterns

### Read and process a segmentation volume
```python
from em_util.io import read_vol, write_h5, compute_bbox_all
from em_util.seg import seg_remove_small, seg_to_cc, seg_to_count

seg = read_vol("segmentation.h5")
seg = seg_to_cc(seg)                    # Relabel connected components
seg = seg_remove_small(seg, 100)        # Remove segments < 100 voxels
uid, uc, bb = compute_bbox_all(seg)     # Get IDs, counts, bounding boxes
write_h5("cleaned.h5", seg)
```

### Evaluate segmentation quality
```python
from em_util.io import read_vol
from em_util.eval import adapted_rand
from em_util.eval.seg import voi

seg = read_vol("prediction.h5")
gt = read_vol("groundtruth.h5")
are = adapted_rand(seg, gt)             # Lower is better
vi_split, vi_merge = voi(seg, gt)       # Lower is better
```

### Process large volumes in chunks
```python
from em_util.io import vol_downsample_chunk, compute_bbox_all_chunk

vol_downsample_chunk("large_vol.h5", ratio=[1,2,2], output_file="downsampled.h5", chunk_num=10)
uid, uc, bb = compute_bbox_all_chunk("large_seg.h5", chunk_num=10)
```

### Extract and measure skeletons
```python
from em_util.io import read_vol, vol_to_skel, skel_to_length

seg = read_vol("neurons.h5")
skels = vol_to_skel(seg, res=[30, 8, 8])
for sid, skel in skels.items():
    length = skel_to_length(skel.vertices, skel.edges, res=[30, 8, 8])
    print(f"Segment {sid}: {length:.1f} nm")
```

### Visualize in neuroglancer
```python
from em_util.ng import ng_layer
import neuroglancer

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    s.layers["image"] = ng_layer(image_vol, res=[30, 8, 8], tt="image")
    s.layers["seg"] = ng_layer(seg_vol, res=[30, 8, 8], tt="segmentation")
```
