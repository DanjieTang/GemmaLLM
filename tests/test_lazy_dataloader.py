import tempfile
import unittest
from pathlib import Path

import numpy as np

from lazy_dataloader import LazyLoadDataset


class LazyLoadDatasetTest(unittest.TestCase):
    def test_loads_relative_image_paths_and_empty_text_only_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            tokens_path = temporary_path / "tokens.npy"
            image_paths_path = temporary_path / "image_paths.npy"
            np.save(tokens_path, np.array([[1, 2], [3, 4]], dtype=np.int32))
            np.save(image_paths_path, np.array(["images/one.png", ""]))

            dataset = LazyLoadDataset(
                str(tokens_path),
                str(image_paths_path),
            )

            _, first_image_path = dataset[0]
            _, second_image_path = dataset[1]
            self.assertEqual(
                first_image_path,
                str(image_paths_path.resolve().parent / "images/one.png"),
            )
            self.assertEqual(second_image_path, "")


if __name__ == "__main__":
    unittest.main()
