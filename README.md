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

The manifest is written to `.vision-workbench/manifest.json`. Importing the same file twice is idempotent because records are keyed by the file content hash. Imports, batch writes, undos and plan creation are serialized by a single workspace lock and committed through the same write-ahead journal, so several local processes can operate on one workspace at once: every successful operation survives, concurrent imports of different content all land, and duplicate imports report `already present` once while keeping the first registration's source and label.

A single-file `add` registers the file's full SHA-256 digest, byte size and
resolved absolute source path, and all three are read from the same stable
regular file: the source is checked before and after the read, and the read
itself is pinned to the confirmed file. If the source is rewritten, appended
to, truncated, deleted and recreated, or replaced by another file (even one
with identical content, size and modification time) while it is being
confirmed or read — or if the path vanishes, becomes a directory or a
symlink, or cannot be opened or read in full — the import fails with the
source path and the reason, and no sample is registered. Content that
happens to match an existing digest never masks such a failure. A source
path that is a symlink when the command starts is still followed and its
target imported.

## Importing a directory

A whole directory of images can be registered in one batch:

```bash
python3 -m vision_workbench import-dir ./workspace ./examples --label cat
python3 -m vision_workbench import-dir ./workspace ./examples --recursive
```

(The shorter `import` alias is equivalent.)

- Only regular files with an extension of `.jpg`, `.jpeg`, `.png`, `.bmp`,
  `.gif` or `.webp` (matched case-insensitively) are candidates. Symlinks
  and other entry types never qualify; recursion stays on the top level
  unless `--recursive` is given and never descends through symlinked
  directories. Files that appear after the scan are left for the next
  import.
- Candidates are ordered by their path relative to the source directory,
  with `/` separators, compared by Unicode code point.
- Samples are identified by their full SHA-256 digest. When several
  candidates share content that the workspace does not yet know, only the
  sorted-first candidate is added; the rest report `duplicate`. Digests
  already in the workspace always report `duplicate` and keep their
  original source, label and revision history — the batch label never
  overwrites an existing record.
- `--label` applies only to genuinely new samples. Omitting it leaves
  them unlabeled; the empty string also means unlabeled; any other string
  is kept literally.
- The result is JSON: the candidate count, the number added, the number
  of duplicates, and every candidate's relative path, full digest and
  `added`/`duplicate` status in scan order. Empty directories and
  all-duplicate imports succeed with zero added.
- The batch is all-or-nothing: a missing or non-directory source, a
  directory that cannot be scanned, a candidate that cannot be read in
  full, or a candidate whose identity, size or modification time changes
  while it is read fails the whole import with the path and reason and
  leaves no new samples. The batch commits through the same journal as
  single-file imports, so an interruption shows either the old state or
  the complete batch, and concurrent label batches are never lost.

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
  name conflict and leaves the original untouched. Sample identities and
  labels are snapshotted from one complete manifest state, so a batch that
  changes several labels is seen wholly before or wholly after the plan,
  never half-applied. Concurrent creation of one name is atomic: exactly
  one request creates it, identical requests reuse it and differing
  requests conflict; plans with different names all save, and a plan is
  only ever read whole or reported as not found.

## Exporting a split plan

A saved split plan can be exported as a portable, offline ZIP package:

```bash
python3 -m vision_workbench export ./workspace baseline ./baseline.zip
```

- The package contains `train/`, `validation/` and `test/` directories,
  each with the same numbered ImageFolder class directories (`class_01`,
  `class_02`, ...), plus a UTF-8 `manifest.json`. Samples are stored as
  `<set>/<class>/<full-sha256><source-extension>`.
- The export uses only the saved plan: its samples, labels, set
  assignments and recorded source paths. Later imports, batch label
  changes or undos never alter an exported package, and exporting never
  writes to the workspace.
- All sets share one class directory numbering, even when a class has no
  samples in a set. Chinese labels, labels containing path separators and
  labels differing only in case each keep a distinct identity in the
  manifest's label-to-directory mapping; no machine-specific absolute
  paths are written.
- An empty plan exports the three set directories and an empty manifest.
- Samples with no label are unlabeled. By default the whole export is
  rejected and the offending digest is reported; `--skip-unlabeled` skips
  them instead, listing the per-set skip counts in both the result and the
  manifest. Skipping never redistributes other samples. A real class
  literally named `unlabeled` is exported normally.
- Source files are streamed and verified against the saved content digest
  while copying: a missing, non-regular, unreadable or changed source
  fails the whole export with the sample digest and reason.
- `--source-dir` reads sample files from a directory tree instead of the
  recorded source paths, which is useful after moving the dataset:
  ```bash
  python3 -m vision_workbench export ./workspace baseline ./baseline.zip \
      --source-dir ./moved-dataset
  ```
  Every regular file under the directory and its subdirectories is
  considered (no extension filter, since files may have been renamed);
  symlinks and other non-regular entries are skipped and symlinked
  directories are never descended. Each exported sample must be matched
  to a file whose full SHA-256 matches the plan — a same-named file with
  different content does not qualify. When several files share content,
  the one whose `/`-separated relative path sorts first by Unicode code
  point is used. The selected copy is pinned to the actual read: the path
  is lstat()ed without following symlinks, opened without following a
  final symlink, and the opened descriptor is fstat()ed and proved to be
  the exact regular file (same device and inode) captured during the
  lookup — the check covers the object that is really read, not just one
  look at the path before it opens. If the selected path is replaced
  after the lookup but before or while it is opened by another regular
  file — even one with identical content, size and modification time —
  or by a symlink, even one pointing at the original file, the whole
  export fails with the sample digest, the selected source path and the
  reason; no other same-content copy is substituted and the plan's
  recorded source path is not used as a fallback. Files unrelated to the
  plan are read (an unreadable file anywhere fails the export) but never
  included. The plan still decides
  sample identities, labels, set assignments and package file extensions;
  the lookup is used for this export only and is never written back. A
  missing or non-directory source, an unreadable directory or file, or a
  sample with no matching content fails the whole export with the path
  and reason; the package is byte-identical to a normal export of the
  same plan, options and content. For an empty plan (or when every sample
  is skipped) the directory only has to exist and be a directory.
- The target must not already exist; it is written atomically, so a failure
  or interruption leaves no incomplete file. Concurrent exports to the
  same path are serialized and at most one succeeds.
- Package bytes are deterministic for a given plan, options and source
  content: exporting at a different time, from a copied workspace or to a
  different path yields byte-identical ZIPs, and source modification times
  do not matter.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

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

Imports, batch writes and undos are crash-safe: a write-ahead journal
installs the manifest and the history together, so after an interruption
the workspace shows either the old state or the complete new state with
its history — labels and their history never straddle the two. Recovery
runs under the workspace lock at the start of every operation, so a
process that already has the workspace open also completes a peer's
prepared transaction without reopening; a sample imported successfully
afterwards is never rolled back by that recovery. Concurrent operations
are serialized by the lock, so no successful import or batch is lost and
two batches modifying the same sample from the same old label cannot both
win. Summary, label and history queries each observe one complete state
and never report corruption from an in-flight write. Saved split plans are
never touched by imports, label changes or undos; plans created afterwards
use the current labels, and the same-name reuse/conflict rules still
apply.
