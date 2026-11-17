isl/raw_videos/ - Input Area for Sign Videos

Do NOT commit any video files here.

Purpose
Teammates drop raw sign-language video clips here locally; files are excluded from Git by root `.gitignore`.

How to add your videos
1. Pull the latest code from GitHub.
2. Create a lowercase, no-spaces subfolder named after the sign gloss (e.g. `hello`, `thankyou`, `namaste`).
3. Place `.mp4`, `.mov`, `.avi`, or `.webm` clips inside that subfolder (filename without extension becomes `sample_id`).
4. Run the extraction pipeline from the repo root:

```bash
python -m isl.pipeline
```

Example layout after adding videos
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

Important rules
- One subfolder per sign label; the folder name is the class label.
- Videos placed directly in `raw_videos/` without a label subfolder are rejected by the pipeline.
- Never rename or delete original videos; the pipeline is read-only and never modifies source files.
- Supported extensions: `.mp4`, `.mov`, `.avi`, `.webm`.
