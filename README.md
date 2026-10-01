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

## Train/validation/test splits

Save a reproducible, stratified split scheme from the samples currently in the manifest:

```bash
python3 -m vision_workbench split create ./workspace \
  --name main --seed 42 --train 0.5 --validation 0.25 --test 0.25
python3 -m vision_workbench split show ./workspace --name main
```

- The three ratios must be finite numbers in `[0, 1]` that sum to 1; a zero ratio leaves that set empty.
- Samples are stratified by label. Unlabeled samples (no label, or an empty-string label) form their own stratum, separate from any real class literally named `unlabeled`.
- For every set, both the total count and the per-class count differ from the expected count (ratio x count) by strictly less than one.
- Assignment depends only on the sample digests, labels, seed, and ratios — not on import order, manifest arrangement, workspace path, or restart count. Samples are listed stably by digest.
- Schemes are stored as JSON under `.vision-workbench/splits/`. They record the sample identity, label, and source at creation time; later imports never change them, and the scheme remains viewable after source files are moved, deleted, or modified. Writes are atomic, so an interrupted write never exposes a partial scheme.
- Creating with an existing name returns the stored scheme only when the current sample digests and labels, seed, and ratios all match; otherwise it reports a name conflict and leaves the original scheme untouched.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
