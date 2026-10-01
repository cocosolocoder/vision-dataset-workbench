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

## Split plans

A split plan divides every sample currently in the manifest into a
train/validation/test set. Plans are saved under
`.vision-workbench/splits/<name>.json` and never change afterwards.

```bash
python3 -m vision_workbench split create ./workspace baseline \
    --seed 42 --train 0.6 --validation 0.2 --test 0.2
python3 -m vision_workbench split show ./workspace baseline
```

- The three ratios must be finite numbers in `[0, 1]` that sum to exactly
  one; any of them may be zero. Decimals (`0.1`) and fractions (`1/3`) are
  both accepted and handled with exact rational arithmetic.
- Stratification guarantees that, for every class and for the dataset as a
  whole, each set's count differs from its proportional expectation by
  strictly less than one — even for classes with one or two samples.
- Assignments depend only on sample content digests, labels and the seed,
  so they are stable across import order, manifest ordering, workspace
  location and restarts. Samples are listed by digest in each set.
- Unlabeled samples (no label or an empty label) form their own stratum,
  reported under the empty-string key `""`; a real class literally named
  `unlabeled` is kept separate.
- Creating under an existing name reuses the plan only when the seed,
  ratios, sample identities and labels all match; otherwise it reports a
  name conflict and leaves the original untouched.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Offline export

A saved plan can be exported into a self-contained ZIP data package that is
trivial to carry around or feed to an ImageFolder loader:

```bash
python3 -m vision_workbench export ./workspace baseline ./baseline.zip
python3 -m vision_workbench export ./workspace baseline ./baseline.zip --skip-unlabeled
```

- The ZIP contains `train/`, `validation/` and `test/` directories, each
  organized as `<set>/<category>/<sha256><ext>`; every set uses the same
  set of category directories (and numeric class indices), including
  categories that have no sample in that set.  File names are the full
  content digest with the source file's extension, and every file stays
  inside its own set directory.
- A UTF-8 `manifest.json` records the plan name, seed, ratios, the mapping
  from original labels to category directories, and each sample's digest,
  set and package-relative path.  No machine-specific absolute path is
  written into the package.
- Chinese labels, labels containing path separators and labels that differ
  only in case each keep their own identity; category directories are
  sanitized and disambiguated deterministically so they never collide.
- Empty labels mean *unlabeled* (the literal class name `unlabeled` is a
  normal category).  By default a single unlabeled sample rejects the whole
  export and names its digest; `--skip-unlabeled` skips them instead and
  reports the skipped count per set both in the command output and in the
  manifest.  Skipping never reassigns the remaining samples.  An empty plan
  still exports all three directories and an empty sample manifest.
- Sources are read from the paths stored in the saved plan, re-hashed while
  streaming, and a missing, non-regular or unreadable file — or any byte
  sequence that does not match the saved digest, including a change while
  reading — fails the entire export, naming the offending digest.  Later
  imports, batch relabels and undos never change an older plan's export, and
  export never modifies the workspace.
- The same plan, options and source contents always produce byte-identical
  ZIPs regardless of time, workspace location, target path or source
  modification time.  Media is streamed in fixed-size chunks, so memory use
  does not grow with the total media size.
- An existing target is refused and left untouched; concurrent exports to
  the same target leave at most one file in place, and a failed or
  interrupted write never leaves a partial target (a re-run once the target
  is absent works normally).

On success the command prints JSON naming the plan, the total number of
samples exported, the per-set category distribution and the per-set skip
counts.

## Batch label updates

Labels of already-imported samples can be changed in batches. A batch is a
UTF-8 JSON file:

```json
{
  "batch": "renumber-2026-01",
  "changes": [
    {"sha256": "<full SHA-256 digest>", "old": "cat", "new": "kitten"},
    {"sha256": "<full SHA-256 digest>", "old": "dog", "new": null}
  ]
}
```

- `batch` must be a non-empty string.
- `old` is the expected current label; `new` is the target label. Both may
  be a string or `null`; `null` and the empty string both mean *unlabeled*,
  any other string is taken literally (including Chinese text and a class
  named `unlabeled`).
- The whole batch is verified before anything is written: an unknown
  digest, a sample listed twice, a non-string/non-null label, a malformed
  file, or any sample whose current label differs from its expected old
  label rejects the entire batch with the offending record identified.
- Records whose target equals the current label are no-ops; if every
  record is a no-op, the batch reports `no-changes`, occupies no batch
  number and writes no history.

```bash
python3 -m vision_workbench label ./workspace <digest>      # current label
python3 -m vision_workbench batch ./workspace changes.json   # submit a batch
python3 -m vision_workbench history ./workspace              # successful batches
python3 -m vision_workbench undo ./workspace <batch-number>  # undo a batch
```

Submitting the same number with the same content again returns the
original result without re-modifying anything (list order and the
`null`/`""` spelling do not change the content), even if labels were
changed in the meantime. The same number with different content is a
conflict.

`history` lists successful batches in submission order: the samples
involved, their before/after labels, and the number of samples actually
changed. Failed batches never enter history.

`undo` restores the old labels of a batch, but only for samples the batch
actually changed and only when none of them was modified afterwards (even
if the label was changed back). Other samples' later changes do not block
the undo. Undo leaves a queryable record; repeating it reports
`already-undone`.

Batch writes and undos are crash-safe: a write-ahead journal installs the
manifest and the history together, so reopening after an interruption
shows either the old state or the complete new state with its history.
Concurrent submissions are serialized by a workspace lock, so no
successful batch is lost and two batches modifying the same sample from
the same old label cannot both win. Saved split plans are never touched
by label changes; plans created afterwards use the current labels, and
the same-name reuse/conflict rules still apply.
