# isl/raw_videos/ — Input Area for Sign Videos

**Do NOT commit any video files here.**

## Purpose
This folder is where teammates drop their sign-language video clips.
Raw videos stay on each person's local machine only — they are excluded
from Git by the root `.gitignore`.

## How to add your videos
1. Pull the latest code from GitHub.
2. Under this folder, create a subfolder named after the sign (the *gloss*).
   Use lowercase letters, no spaces: e.g. `hello`, `thankyou`, `namaste`.
3. Place one or more `.mp4` (or `.mov`, `.avi`, `.webm`) video clips inside
   that subfolder. Each file's name (without the extension) becomes its
   `sample_id` in the generated metadata.
4. Run the extraction pipeline from the repo root:

```bash
python -m isl.pipeline
```

## Example layout after adding videos
```
isl/raw_videos/
├── hello/
│   ├── hello_001.mp4
│   └── hello_002.mp4
├── namaste/
│   └── namaste_001.mp4
└── thankyou/
    └── thankyou_001.mp4
```

## Important rules
- One subfolder per sign label. The folder name is the class label.
- Videos placed directly in `raw_videos/` (without a label subfolder)
  are **rejected by the pipeline** with a clear error message.
- Never rename or delete original videos. The pipeline is read-only;
  it never modifies source files.
- Supported extensions: `.mp4`, `.mov`, `.avi`, `.webm`.
