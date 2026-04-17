import unittest

import torch

from latent_meanflow.data.subset import FixedSubsetDataset


class _ToySourceDataset:
    def __init__(self, length=6):
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        index = int(idx)
        mask_index = torch.full((2, 2), fill_value=index, dtype=torch.long)
        mask_onehot = torch.nn.functional.one_hot(mask_index, num_classes=max(self.length, 1)).permute(2, 0, 1).float()
        return {
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
            "metadata": {"source_index": index},
        }


class FixedSubsetDatasetTest(unittest.TestCase):
    def _dataset_config(self, length=6):
        return {
            "target": f"{__name__}._ToySourceDataset",
            "params": {"length": int(length)},
        }

    def test_first_n_one_has_expected_length_and_item_structure(self):
        dataset = FixedSubsetDataset(
            dataset_config=self._dataset_config(length=6),
            first_n=1,
        )
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(set(sample.keys()), {"mask_index", "mask_onehot", "metadata"})
        self.assertEqual(int(sample["metadata"]["source_index"]), 0)
        self.assertTrue(torch.equal(sample["mask_index"], torch.zeros((2, 2), dtype=torch.long)))

    def test_first_n_four_has_expected_length_and_preserves_items(self):
        dataset = FixedSubsetDataset(
            dataset_config=self._dataset_config(length=6),
            first_n=4,
        )
        self.assertEqual(len(dataset), 4)
        last_sample = dataset[3]
        self.assertEqual(int(last_sample["metadata"]["source_index"]), 3)
        self.assertEqual(tuple(last_sample["mask_onehot"].shape[-2:]), (2, 2))

    def test_indices_subset_preserves_original_item_structure(self):
        dataset = FixedSubsetDataset(
            dataset_config=self._dataset_config(length=6),
            indices=[4, 1],
        )
        self.assertEqual(len(dataset), 2)
        first_sample = dataset[0]
        second_sample = dataset[1]
        self.assertEqual(int(first_sample["metadata"]["source_index"]), 4)
        self.assertEqual(int(second_sample["metadata"]["source_index"]), 1)
        self.assertEqual(set(first_sample.keys()), set(second_sample.keys()))

    def test_first_n_and_indices_together_should_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly one subset selector"):
            FixedSubsetDataset(
                dataset_config=self._dataset_config(length=6),
                first_n=1,
                indices=[0],
            )
