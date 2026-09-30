# Vision Dataset Workbench

Vision Dataset Workbench is a local, CPU-friendly command-line foundation for organizing image datasets. It stores a small manifest beside the dataset, imports image metadata without uploading files, and reports basic dataset statistics.

## Requirements

- Python 3.11 or newer
- No third-party runtime dependencies

## Quick start

```bash
python3 -m vision_workbench init ./workspace
python3 -m vision_workbench add ./workspace ./examples/cat.jpg --label cat
python3 -m vision_workbench summary ./workspace
```

The manifest is written to `.vision-workbench/manifest.json`. Importing the same file twice is idempotent because records are keyed by the file content hash.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
