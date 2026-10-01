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

## Batch label changes

Labels of already-imported samples can be changed in batches from a
UTF-8 JSON file:

```bash
python3 -m vision_workbench label apply ./workspace ./relabel.json
python3 -m vision_workbench label show ./workspace <sha256>
python3 -m vision_workbench label history ./workspace
python3 -m vision_workbench label undo ./workspace <batch-id>
```

The batch file is a JSON object with a non-empty `batch_id` and a
`changes` list; each entry names one sample by its **full SHA-256
digest** and gives the label it expects to find and the wanted label:

```json
{
  "batch_id": "2026-10-corrections",
  "changes": [
    {"sha256": "2515a4…c6ab", "old_label": "cat", "new_label": "猫"},
    {"sha256": "87eb4f…d324", "old_label": null, "new_label": "unlabeled"},
    {"sha256": "ab12…90ef", "old_label": "cat", "new_label": null}
  ]
}
```

- `null` and `""` both mean *unlabeled*; every other string is taken
  literally, including Chinese labels and a real class named
  `"unlabeled"`. A label can be set, changed or cleared.
- The whole batch is checked before anything is written. An unknown
  digest, a duplicated digest, a label that is not a string or null, a
  malformed document, or a current label different from `old_label`
  rejects the entire batch; the command exits non-zero, reports a
  machine-readable `code` and the offending record(s) on stderr, and
  changes no sample. The two unlabeled spellings compare equal during
  verification.
- Entries whose target equals the current label count as unchanged. A
  batch where nothing changes is reported as `unchanged`; it writes no
  history and does not consume the batch id.
- On success the command reports the number of changed samples; the
  classification statistics in `summary` update immediately while
  content digest, source and size of every sample stay untouched.

### Batch ids, history and undo

- Resubmitting a successful batch with the **same id and content**
  returns the original result (`duplicate`, with the original change
  detail) without modifying labels again — even if the labels were
  changed by someone else afterwards. Reordering the change list or
  swapping the `null`/`""` spelling of a label does not count as a
  different content; the same id with genuinely different changes is an
  explicit `batch_id_conflict`.
- `label history` lists successful batches in submission order with the
  samples involved, the before/after labels and how many samples really
  changed; failed and entirely-unchanged batches never enter history.
- `label undo <id>` restores every label a successful batch changed. It
  succeeds only if each sample that batch actually changed has not been
  modified since; later edits to *other* samples are fine, but a later
  edit to one of the batch's samples blocks undo even if that sample was
  edited back to the same value. Undo leaves its own queryable record;
  repeating it reports `already_undone` without writing again, and an
  unknown id is a clear error.

### Concurrency, durability and splits

- Two local processes submitting at the same time are serialized with a
  lock file; neither successful batch is lost. If different batches with
  the same expected old label actually modify the same sample, exactly
  one commits and the other receives a label-mismatch conflict.
- The manifest is written through a temp file, fsync and atomic
  replace, so after a failed write or an interrupted operation a
  reopened workspace shows either the old or the fully updated labels
  together with the matching history — never a mix. Old (schema-1)
  workspaces keep working without re-import.
- Label changes (and undos) never alter a saved split: its members,
  label snapshot and distribution stay as they were. New splits created
  afterwards use the current labels; an old plan name still follows the
  usual reuse-or-conflict rule.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
