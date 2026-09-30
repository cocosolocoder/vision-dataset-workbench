from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vision_workbench.store import DatasetStore


class DatasetStoreTest(unittest.TestCase):
    def test_add_is_idempotent_and_summary_counts_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            image.write_bytes(b"sample-image")
            store = DatasetStore(root / "dataset")

            first = store.add(image, "cat")
            second = store.add(image, "cat")

            self.assertTrue(first.added)
            self.assertFalse(second.added)
            self.assertEqual(first.digest, second.digest)
            self.assertEqual(
                store.summary(),
                {"items": 1, "bytes": 12, "labels": {"cat": 1}},
            )

    def test_empty_summary_initializes_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(Path(directory) / "dataset")
            self.assertEqual(store.summary(), {"items": 0, "bytes": 0, "labels": {}})


if __name__ == "__main__":
    unittest.main()
