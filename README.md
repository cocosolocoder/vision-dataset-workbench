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
  import. The recursive scan walks an explicit stack rather than Python
  call frames, so a chain of directories deeper than the interpreter's
  recursion limit is scanned in full — images at the top, on side branches
  and at the end of a long chain are all imported — as long as the paths,
  directory permissions and system resources allow; the scan never stops
  at some depth and reports success for only the shallower part.
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

### Corrupted saved plans

Every command that reads a saved plan — `split show`, `export` (even with
`--skip-unlabeled`) and `split create` under an existing name — re-checks
the saved plan before anything is shown, packaged or reused.

Every member of every set (train, validation and test — including
unlabeled members and members listed after other, valid ones) must carry
a `sha256` identity that is **exactly 64 lowercase hexadecimal
characters**, the same spelling a registered sample uses. An empty
string, a missing field, a value of any other type (a number, boolean,
array or null), a truncated or padded digest, an uppercase spelling, a
non-hexadecimal character, or whitespace before, after or inside the
digest is corruption. Such a value is never repaired by truncating,
padding, stripping spaces or changing case — even when the lowercase
digest happens to name a sample that is currently registered.

In addition, the plan is re-checked against the proportions it was
created with, using only the members, labels and ratios saved in the
plan file. Statistics that merely agree with the member list are not
enough: a member moved into another set together with adjusted counts
still violates the rule and is rejected as a corrupted plan.

For the dataset **as a whole** (every member in the plan) and, separately,
for **each category** (that category's members summed across the three
sets; the unlabeled category and a real class literally named
`unlabeled` are counted independently), each set's count must satisfy

```
|actual − range-total × set-ratio| < 1
```

with exact rational arithmetic. An integral expectation has to match
exactly; a non-integral one accepts either floor or ceil (the rounding
direction is not fixed); a difference of exactly one is always rejected,
and a decimal ratio and the equal fraction always give the same verdict.
A balanced total never hides one mis-allocated class.

On a violation the command exits non-zero, prints nothing on standard
output and no traceback, and reports on standard error that the split
plan is corrupted. A bad identity names the plan, the offending set, the
member's 1-based position inside that set, the bad value and what is
wrong with it (wrong length, uppercase/non-hexadecimal characters,
whitespace, empty or not a string); a proportion violation names the
plan, the offending set, whether the overall count or which category is
wrong, and the actual and expected counts. Either verdict is reported
before any reuse/conflict verdict, so such a plan is never shown as a
normal result, exported, reported as already existing, or treated as an
ordinary name conflict. Nothing is rewritten: the plan, the current
sample records and the label history are left untouched, an export
leaves no package or temporary package, and the members and ratios are
never silently redistributed, normalized or "repaired". The empty plan
stays legal, a zero-ratio set must be empty, and classes with only one
or two samples follow the same count rule. Both judgements read the
saved plan alone — its images need not still be readable and its
members need not still appear in the current registration list — so
later imports, label changes, or moved/deleted source images never
invalidate an otherwise legal plan.

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
  fails the whole export with the sample digest and reason. Without
  `--source-dir` the bytes read must come from the one ordinary regular
  file confirmed immediately before that read: a source path that is a
  symbolic link when the export starts is followed once and its target
  is the confirmed file, but between the confirmation and the actual
  open the path may not be remapped — another regular file renamed onto
  it is rejected as a replaced source even when its content, size and
  modification time match exactly (a matching final digest never masks
  the replacement), and a named pipe swapped into that gap, even one
  with no writer, fails immediately and explicitly rather than making
  the export wait for data or streaming pipe content into the package.
  Such a failure names the sample's full SHA-256, the recorded source
  path and the concrete reason, leaves the target unpublished and this
  run's temporary package removed, and never rewrites the recorded
  source path or substitutes another same-content file.
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
  point is used. That pinning applies to *every* file read while the
  tree is looked up, not just to the copies that end up in the package:
  each entry is lstat()ed without following symlinks, opened without
  following a final symlink and without blocking, and the opened
  descriptor is fstat()ed and proved to be the exact regular file
  (same device and inode) that was confirmed a moment earlier, so the
  bytes hashed during the lookup come from the object the walk typed.
  If a confirmed entry is replaced in that gap — by another regular
  file, even one with identical content, size and modification time,
  by a symbolic link, or by a named pipe, even one with no writer or
  one whose writer would supply the planned bytes — the whole export
  fails immediately, naming that path and what it became; it never
  waits for a writer and never hashes or matches the new occupant's
  bytes. The selected copy is then pinned again to the actual read:
  the path is lstat()ed without following symlinks, opened without
  following a final symlink, and the opened descriptor is fstat()ed
  and proved to be the exact regular file (same device and inode)
  captured during the lookup — the check covers the object that is
  really read, not just one look at the path before it opens. If the
  selected path is replaced after the lookup but before or while it
  is opened by another regular file — even one with identical
  content, size and modification time — or by a symlink, even one
  pointing at the original file, the whole export fails with the
  sample digest, the selected source path and the reason; no other
  same-content copy is substituted and the plan's recorded source
  path is not used as a fallback.

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
  The lookup walk and the copy-phase verification both walk an explicit
  stack of directory frames rather than Python call frames, so a moved
  tree whose directory chain is deeper than the interpreter's recursion
  limit is still searched and verified in full — files at the root, at
  the end of the long chain and on side branches are all matched — as
  long as the paths, permissions and system resources allow; the export
  never stops at some depth, exports only the shallower files, or fails a
  deep directory check after the images were copied.

  Files unrelated to the
  plan are read (an unreadable file anywhere fails the export) but never
  included, and a confirmed entry swapped while the lookup reads it —
  even one unrelated to the plan, or replaced only after every planned
  sample was already found — fails the export just the same. The plan still decides
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

Before either verdict, the submitted number must identify **exactly one
entry** in the full saved history list. When the same batch number
occurs on a second history entry — whether the two sit next to each
other or have other, legitimate batches between them — the whole
submission is refused as corrupted batch history, with the same rule an
undo uses. The two entries' contents never resolve the ambiguity:
byte-for-byte identical entries are a repeat too, and entries touching
different samples, recording different label changes or carrying
different undo states are refused just the same. Submission never picks
the first occurrence, the last, the still-active one or one not yet
undone, and the two entries are never merged, renumbered or
de-duplicated. The refusal comes before the same-content replay and the
different-content conflict (and before the no-change short-circuit), so
matching one entry's content exactly never selects it, and an entry
already marked undone surfaces the ambiguity instead of replaying. The
error names the batch number as submitted and the 1-based positions of
the first and second occurrence in the full history list; with three or
more occurrences the first two positions are named. The command exits
non-zero with nothing on standard output and no traceback, changes no
sample label or label revision, appends no history, and deletes,
renumbers or repairs neither entry. The full list is scanned, so a
restorable first occurrence never masks a repeat later. Numbers are
matched by their saved raw string exactly — surrounding whitespace and
case are never trimmed or folded, so `"b1"`, `"B1"` and `" b1"` are
different numbers. Only the submitted number is judged: a duplicate of
some *other* number neither blocks a new batch number nor reuse of an
intact, uniquely numbered batch, and this check still runs after the
unknown-sample rejection, which keeps its precedence.

Before either verdict, the saved batch with that number is checked for
the same one-row-per-sample integrity an undo relies on: the same full
SHA-256 digest may appear on at most one of its records. Because
same-content comparison is order-insensitive and works on record sets, a
duplicated row would otherwise collapse into the original record and a
repeat of the exact first submission would report `already-applied`
again, counting a single label change twice. A second occurrence — an
exact copy of a record that changed a label, two records carrying
different before/after labels for one digest, or a copy of a record that
never changed a label — is therefore corrupted batch history, never
records to merge or de-duplicate: the digest alone decides, even when
other samples sit between the two occurrences. The whole record list is
scanned, so a unique record at the front never masks a repeat further
back, and the damage is reported even when the repeat content would
otherwise be a number conflict or the batch has already been undone —
never as an ordinary conflict or as a normal replay. The error names the
submitted batch number, the duplicated sample's full SHA-256 digest and
the two 1-based record positions (the first occurrence and the one that
repeats it).

The saved batch is also checked for the same label-field integrity an
undo relies on: every one of its records must explicitly carry both the
before label `old` and the after label `new`. A record missing either
key — distinguished as missing `old`, missing `new`, or missing both —
makes the batch history corrupted, and the whole repeat submission is
refused before it can return the original statistics or report a number
conflict. The command exits non-zero with nothing on standard output and
no traceback, and the error names the submitted batch number, the
offending sample's full SHA-256 digest and the record's 1-based position
inside that batch. Key presence alone decides: an explicitly saved
`null` or empty string is a recorded unlabeled label and still takes
part in same-content comparison, and a real class literally named
`unlabeled` stays its own category; full labels keep their case,
whitespace and Chinese text exactly. Every record is examined,
including unchanged rows and records behind otherwise valid ones, so a
complete record at the front never masks a gap later, and the damage is
reported even when the repeat content would otherwise conflict or the
batch has already been undone. The missing value is never reconstructed
from this submission or from the sample's current label. On either
refusal nothing changes — sample labels, their revisions, the history
records, undo markers and saved split plans stay as they were, no field
is filled in and no new result is appended. Only the batch named by the
submission is inspected: a repeated sample or missing label fields in a
*different* batch neither block a new batch number nor reuse of another
batch's intact history.

The saved batch is likewise checked for the same `changed`-flag
integrity an undo relies on: every record's flag must agree with the
before/after labels saved on that same record. A replay reconstructs
the original statistics from those flags, so a record that saved `cat`
→ `dog` but is flagged unchanged would be replayed as `already-applied`
with the sample miscounted as untouched, and a record flagged changed
on identical before/after labels would inflate the changed count. Both
are corrupted batch history, and the whole repeat submission is refused
before it can return the original statistics or report a number
conflict — never reported as a successful submission, and never
auto-corrected and then accepted. Whether the two labels differ is
judged exactly as the undo judges it: `null` and the empty string spell
the same unlabeled state, so swapping them is not a change, while a
real class literally named `unlabeled` is an ordinary label; every
other label compares as the exact original string, with case, leading
or trailing whitespace, Chinese text and path separators all
significant. The decision uses only the labels stored in the history
record — never the sample's current label and never the labels in this
submission — so a record that recorded `cat` → `dog` but is flagged
unchanged is corruption even if the sample has since been changed back
to `cat`. The command exits non-zero with nothing on standard output
and no traceback, and the error names the submitted batch number, the
sample's full SHA-256 digest, the record's 1-based position inside that
batch and the exact contradiction (a false flag with differing labels,
or a true flag with identical labels). Every record is examined,
including records that recorded no real change and records behind
otherwise valid ones, so a consistent record at the front never masks a
contradiction later, and the damage is reported even when the repeat
content would otherwise conflict or the batch has already been undone.
On refusal nothing changes — sample labels, their revisions, the
history records and undo markers stay as they were, no flag is
rewritten and no new result is appended. Only the batch named by the
submission is inspected: the same contradiction in a *different* batch
neither blocks a new batch number nor reuse of another batch's intact
history.

`history` lists successful batches in submission order: the samples
involved, their before/after labels, and the number of samples actually
changed. Failed batches never enter history.

`undo` restores the old labels of a batch, but only for samples the batch
actually changed and only when none of them was modified afterwards (even
if the label was changed back). Other samples' later changes do not block
the undo. Undo leaves a queryable record; repeating it reports
`already-undone`.

### Corrupted batch history

The number an `undo` names must identify **exactly one batch** in the saved
history list. (Re-submitting a batch number is held to the same rule: see
the number-uniqueness refusal above.) When the same batch number occurs on
a second entry — whether the two entries sit next to each other or have
other, legitimate batches between them — the whole undo is refused as
corrupted batch history. The two entries' contents never resolve the
ambiguity: byte-for-byte identical entries are a repeat too, and entries
touching different samples, recording different label changes or carrying
different undo states are refused just the same. Undo never picks the
first occurrence, the last, the still-active one or one not yet undone,
and the two entries are never merged, renumbered or de-duplicated. The error names the batch number as given and the
1-based positions of the first and second occurrence in the full history
list; with three or more occurrences the first two positions are named. The
command exits non-zero with nothing on standard output and no traceback,
changes no sample label or label revision, rewrites neither the
registration list nor the batch history, sets no undo marker or undo time,
and deletes, renumbers or repairs neither conflicting entry. The full list
is scanned, so a restorable first occurrence never masks a repeat that
appears later, and a first occurrence already marked undone surfaces this
ambiguity instead of returning `already-undone`. Numbers are matched by
their saved raw string exactly — surrounding whitespace and case are never
trimmed or folded, so `"b1"` and `" b1"` are different numbers. Only the
requested number is judged: a duplicate of some *other* number never blocks
a unique, otherwise valid target from undoing normally, and a number absent
from the list still returns the ordinary unknown-batch error.

Every record of the uniquely resolved target batch must name its sample in
the exact identity spelling registration uses: a `sha256` of **exactly 64
lowercase hexadecimal characters**. An empty string, any other length, an
uppercase letter, a non-hexadecimal character, or whitespace before, after
or inside the value makes the batch history corrupted. Such a value is
never repaired — not by trimming the whitespace, folding it to lowercase,
truncating or padding it to 64 characters — and the original identity is
never inferred from the current sample list: even when the rewritten value
would name a registered sample, the record as saved is refused. The whole
record list is examined, including records that did not change a label and
records standing behind otherwise valid ones, and the check finishes before
any sample is restored or an `already-undone` answer is returned, so a
restorable record at the front can never hide a malformed one further back.
The command exits with a non-zero status, prints nothing on standard output
and no traceback, and reports on standard error that the **batch history**
is corrupted — naming the requested batch number, the record's 1-based
position inside that batch, the raw `sha256` value and exactly what is
wrong with its format (empty, wrong length, uppercase/non-hexadecimal
characters, or whitespace). Sample labels and their revisions, the batch
records, the undo marker and saved split plans all stay exactly as they
were. This judges format only, on the one target batch: the same malformed
value in a *different* batch does not stop an intact batch from undoing
normally, a number absent from the history still returns the ordinary
unknown-batch error, and a record whose `sha256` is well formed but names a
sample that has since left the registration list is still handled by the
ordinary missing-sample rule below.

Every sample record stored for a successful batch carries a `rev`: the
sample's label revision at the moment the batch finished, which an undo
pins against. That `rev` is mandatory on **every** history record —
whether the record actually changed a label or not — and must be a genuine
non-negative integer. A boolean (`true`), a decimal (`1.0`), a numeric
string (`"1"`), `null` and a missing field are all invalid; a missing
revision is never treated as zero, and the integer `0` on an unchanged
record stays legal. This requirement is specific to batch history: the
optional `rev` on manifest registrations keeps its old compatibility
behaviour, including records that omit it.

When an undo targets a batch containing such a record, the command exits
with a non-zero status, prints nothing on standard output and no
traceback, and reports on standard error that the **batch history** is
corrupted — naming the batch number, the sample's full SHA-256 digest and
the exact `rev` problem. The whole undo is refused even if an earlier
record in the same batch was restorable: no sample changes label, and
neither the registration list nor the batch history is rewritten,
default-filled or given a new undo marker. This refusal is distinct from
the normal "sample modified after the batch" rejection and from an
unknown batch number — a corrupt record can never be reported as undone,
as having no changes, or as an unknown batch. Corruption is checked before
the repeated-undo short-circuit too, so a batch already marked undone
still surfaces its bad record instead of returning `already-undone`.

A batch history is likewise corrupted when the **same sample content
digest appears on more than one record of the target batch**. Undo is one
restore per sample, pinned by that digest, so a duplicated row — even an
exact copy of a record that genuinely changed a label, or a copy of a
record that did not change one — would otherwise restore one sample
several times, add its label revision repeatedly and overstate the
restored count. Such a batch is never merged or de-duplicated: the whole
undo is refused. The decision is made from the full SHA-256 digest alone,
regardless of the labels, whether the before/after labels differ or
whether the record actually changed anything, so two records with
different label content for one digest are still a duplicate. Every
record in the batch is examined, so a perfectly restorable record at the
front can never hide a repeat that appears later; the error names the
batch number, the duplicated sample's full digest and the two 1-based
record positions (the first occurrence and the one that repeats it). As
with a bad revision, the command exits non-zero with nothing on standard
output and no traceback, changes no sample label or revision, rewrites
neither the registration list nor the batch history, sets no new undo
marker, and is reported even when the batch is already marked undone
rather than as `already-undone`. The later-modification conflict, an
unknown batch number and a successful undo are all distinct from this
refusal.

A batch history is likewise corrupted when a record's **`changed` flag
contradicts the before/after labels saved on that same record**. Undo treats
a flagged-changed row as a sample to restore from `new` back to `old` and to
count, and a flagged-unchanged row as one it never restores, so the flag must
match the record's own stored labels:

- the before and after labels are judged equal when both are unlabeled
  (`null` and the empty string spell the same state, so swapping them is not
  a change), and otherwise compared as the exact original strings — case,
  leading or trailing whitespace, Chinese text and path separators keep
  their meaning, and a real class literally named `unlabeled` is a label and
  never equals the unlabeled state;
- differing before/after labels therefore require `changed: true`, and
  identical labels require `changed: false`.

The decision uses only the labels stored in the history record; it never
substitutes the sample's current label, so a row that recorded `cat` → `dog`
but is flagged unchanged is corruption even if the sample has since been
changed back to `cat`. Such a hidden change would otherwise leave the sample
at `dog` while the batch is recorded as undone, and a false change on
identical labels would count and "restore" a sample the batch never touched.
The whole undo is refused: the error names the batch number, the sample's
full SHA-256 digest, the record's 1-based position inside the target batch
and the exact contradiction (false flag with differing labels, or true flag
with identical labels), with nothing on standard output and no traceback.
Every record in the batch is examined, so a restorable record at the front
is never partially restored first, and nothing is auto-corrected, dropped or
rewritten — no sample label or revision changes, neither the registration
list nor the batch history is rewritten, and no undo marker or undo time is
added. As with the other integrity damage, it is reported even when the
batch is already marked undone rather than as `already-undone`, is distinct
from the later-modification conflict and an unknown batch number, and only
the target batch is examined — the same contradiction in another batch does
not stop a valid batch from undoing normally and restoring exactly the
samples it actually changed.

The same sample participating in two *different* batches over time is
normal and stays legal; only a repeat inside the one target batch is an
error. Damage in a different batch does not matter: only the target batch's
records are examined, so a valid batch still undoes normally.

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
