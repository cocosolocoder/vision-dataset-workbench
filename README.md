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
  leaves no new samples. With `--recursive`, a subdirectory that was
  confirmed as a real directory and is then replaced by a symbolic link —
  wherever the link points — likewise fails the whole import with the
  subdirectory's path, rather than following the link or skipping the
  directory; the link target's images are neither imported nor reported
  as duplicates, and directories that were symlinks from the start are
  still simply ignored. The batch commits through the same journal as
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
- A single sample at or above 2 GiB (2,147,483,648 bytes of uncompressed
  content) is written as a ZIP64 entry automatically; large and ordinary
  samples may coexist in one plan and one package, and the large entry is
  still streamed in fixed chunks, so exporting it never needs memory for
  the whole file. This also covers multi-gigabyte content that compresses
  well: the decision follows the uncompressed size, so a tiny resulting
  package is not rejected. Plans containing only smaller samples keep
  exactly the classic archive format, byte for byte — ZIP64 fields appear
  only in the entries that require them. The manifest, the result counts
  and the set/class layout are identical either way, and packages remain
  extractable with standard ZIP64-aware tools.
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
  recorded source path is not used as a fallback.

  The directory walk is itself pinned to directory descriptors rather
  than path strings: each directory is entered with `openat`
  (`O_NOFOLLOW` and `O_DIRECTORY`) relative to its parent's descriptor
  and verified with `fstatat`, and the confirmed root stays open for the
  whole export. A subdirectory that is a real directory when the walk
  first reaches it but is replaced by a symlink before it is entered,
  while its contents are read, after the lookup or while the package is
  copied fails the whole export with the changed directory's path and
  the symlink reason — whether the link points outside the specified
  directory, inside it, or at the directory moved to its new location,
  and at any nesting level. The link's target is never read, matched or
  packaged, even if it holds a file byte-identical to the planned
  sample; no other normal copy and no still-readable recorded source are
  substituted. Before each matched file is copied its ancestor
  directories are re-verified the same way, and after all samples are
  streamed every directory from the walk is verified once more before
  the ZIP is published, so the target ZIP is never created and no
  incomplete export file remains. Only entries that are already
  symlinks the first time the walk meets them are skipped, as before.

  Files unrelated to the
  plan are read (an unreadable file anywhere fails the export) but never
  included. The plan still decides
  sample identities, labels, set assignments and package file extensions;
  the lookup is used for this export only and is never written back. A
  missing or non-directory source, an unreadable directory or file, or a
  sample with no matching content fails the whole export with the path
  and reason; the package is byte-identical to a normal export of the
  same plan, options and content. For an empty plan (or when every sample
  is skipped) the directory only has to exist and be a directory.
- The target name is owned only once the whole package has been generated
  and fsynced and the source verification has passed, so a reader never
  sees a half-written file.  The name must be free both when the export
  starts and at the moment the finished package is published: another
  program that creates a file at the target path while the images are
  copied or compressed — even an empty file or one byte-identical to the
  package, and even in the last instant before publication — makes the
  export fail with the user-given target path as a name-already-taken
  conflict, a non-zero exit status and no result on standard output.
  That occupant is never overwritten, unlinked or moved, its content and
  file type are preserved, and this run's temporary package is removed.
  A symbolic link at the name counts as occupied too, including one that
  points at a location that does not exist; whether the link is there at
  the start or appears during the export, the link itself and its target
  are left in place and the ZIP is never written through it.  There is no
  overwrite option and no automatic alternative file name.  A failure or
  interruption otherwise leaves no incomplete file, and concurrent
  exports to the same path are serialized so at most one succeeds and the
  rest fail clearly with the same conflict.
- Package bytes are deterministic for a given plan, options and source
  content: exporting at a different time, from a copied workspace or to a
  different path yields byte-identical ZIPs, and source modification times
  do not matter.

## Finding samples by label

Samples can be listed by their current category label, so a class leads
straight to the full sample summaries needed for the label queries and
batch updates below — no need to page through the dataset listing.

```bash
python3 -m vision_workbench find-label ./workspace cat     # a normal class
python3 -m vision_workbench find-label ./workspace 猫      # Chinese text, kept as-is
python3 -m vision_workbench find-label ./workspace ''      # unlabeled samples
python3 -m vision_workbench find-label ./workspace unlabeled  # the literal class
```

The result is readable UTF-8 JSON: the queried label, the match count and
the sample list, each record giving the full SHA-256 digest, the source
path saved at first registration, the byte size and the current label.
Records are sorted ascending by full digest and `count` always equals the
list length. A digest is listed once no matter how many paths once
imported identical content, with the source from its first registration.

- Matching is exact on the **current** registered label: case, leading or
  trailing whitespace and path separators are matched literally and are
  never trimmed, rewritten or interpreted as paths. Querying `cat` does
  not return `Cat` or `cat ` (cat followed by a space).
- The empty string queries **unlabeled** samples: records whose label is
  `null` or `""` both match, and every returned record shows its label as
  `null`. Querying the text `unlabeled` instead returns only samples
  whose real class is literally named `unlabeled`; it never includes
  unlabeled samples and does not use the summary statistics' display that
  merges the two.
- An empty workspace or a label with no matches is a normal result with
  `count` 0 and an empty list, not an error.
- The query reads only registered records. A successful batch update or
  undo is reflected by the next query; labels snapshotted in saved split
  plans neither match nor are changed, and a source image that has since
  moved, been deleted or become unreadable is still returned — the source
  file is never reopened and its recorded path is never updated.

## Corrupted registrations

Every command that uses the current manifest — `summary`, `label`,
`find-label`, `add`, `import-dir`, `batch`, `undo` and `split create` —
validates the **whole** `manifest.json` before using any sample, so a
damaged registration list can never surface as a runtime exception on one
record or make a label change touch only one of two duplicate records:

- The manifest must be a JSON object whose `schema_version` is the integer
  `1` and whose `items` is a list. Booleans (`true`) and decimals (`1.0`)
  cannot stand in for the version.
- Each sample record must be an object with a full 64-character lowercase
  hexadecimal SHA-256 digest, a non-empty string source, a non-negative
  integer size, and a string or `null` label. Booleans, numeric strings and
  decimals are never accepted as sizes. One digest may be registered only
  once — even two otherwise identical records are an error, never merged.
- Valid records at the front of the list never mask a later bad record;
  the check covers the complete list, including records a query would not
  match.

On any such problem the command exits with a non-zero status, prints
nothing on standard output and no traceback, and reports on standard error
that the **dataset manifest** is corrupted — naming the offending part, or
the record's 1-based position in the list together with the bad field (for
a duplicate digest, the digest and both positions). The operation is
refused outright: the existing manifest, the batch history and saved split
plans are never rewritten, pruned, default-filled or replaced by an empty
list. Manifests produced by older versions remain valid without migration —
records may lack the optional label revision and carry extra fields, both
of which are kept — and the empty list and zero-byte samples stay legal,
with `null` and `""` still meaning *unlabeled*.

A source image that has since moved or been deleted is not corruption:
queries keep using the registered information. Features that depend only on
saved data also keep working regardless of the manifest: `history` reads
the batch history, and `split show` and `export` use the saved plan alone.

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
python3 -m vision_workbench find-label ./workspace <label>  # samples of one label
python3 -m vision_workbench batch ./workspace changes.json   # submit a batch
python3 -m vision_workbench history ./workspace              # successful batches
python3 -m vision_workbench undo ./workspace <batch-number>  # undo a batch
```

See [Finding samples by label](#finding-samples-by-label) for the
`find-label` query, including the empty-string query for unlabeled
samples.

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

Each history record of the target batch must carry the `rev` the sample
had when that batch ended, and it must be a genuine non-negative integer
for **every** record — whether the batch actually changed that sample or
not (an unchanged record's integer zero is valid, but the field is still
required). Booleans (`true` never stands in for revision `1`), decimals
(`1.0`), numeric strings (`"1"`), `null` and a missing field are all
history corruption, never coerced or treated as zero. Undo verifies the
whole target batch before restoring anything and before the
already-undone short-circuit, so a damaged record refuses the entire
operation: the command exits with a non-zero status, prints nothing on
standard output and no traceback, and reports on standard error that the
**batch history** is corrupted, naming the batch number, the offending
sample's full SHA-256 and the problem with its `rev`. No label is
restored (a sample that met every restore condition keeps its current
label), no missing value is backfilled, no field type is converted, and
no new undo marker is written. This refusal is distinct from the normal
"sample was modified after the batch" rejection, from a successful undo,
from a no-change batch and from an unknown batch number — and it is
surfaced even for a batch that was already undone. Old registration
records without the optional manifest `rev` remain valid; only
batch-history records require it.

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
