# isl/extracted/ — Generated Pose Data

**Do NOT commit any `.npz` files here.**

## Purpose
This folder is written to by the P1 pipeline. Each accepted sign video
produces one `.npz` file here. Generated files are excluded from Git by
the root `.gitignore`.

## Folder layout (created automatically by the pipeline)
```
isl/extracted/
└── <gloss>/
    └── <sample_id>.npz
```

## What each .npz file contains
| Array key   | dtype   | Shape        | Description |
|-------------|---------|--------------|-------------|
| `pose_seq`  | float32 | `(T, 21, 3)` | Per-frame hand landmarks (wrist-origin normalised). T = accepted frame count. |
| `vec`       | float32 | `(252,)`     | Global feature vector: mean, std, min, max across T frames (21×3×4 = 252). |
| `flat_vec`  | float32 | `(1890,)`    | Sequence resampled to 30 frames, then flattened (30×21×3 = 1890). |
| `gloss`     | str     | scalar       | The sign label (folder name under `raw_videos/`). |

## How to inspect a file
```bash
python -c "
import numpy as np, pathlib, sys
p = sys.argv[1]
d = np.load(p, allow_pickle=True)
for k, v in d.items():
    print(k, getattr(v, 'shape', v), getattr(v, 'dtype', ''))
" isl/extracted/hello/hello_001.npz
```

## Regenerate at any time
Run the extraction pipeline again from the repo root; existing `.npz`
files are overwritten only if the source video is newer than the output.
