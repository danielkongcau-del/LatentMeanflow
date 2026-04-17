from collections.abc import Sequence

from ldm.util import instantiate_from_config
from torch.utils.data import Dataset


class FixedSubsetDataset(Dataset):
    def __init__(self, dataset_config, first_n=None, indices=None):
        super().__init__()
        if (first_n is None) == (indices is None):
            raise ValueError(
                "FixedSubsetDataset requires exactly one subset selector: "
                "pass either first_n or indices."
            )

        self.dataset_config = dataset_config
        self.dataset = instantiate_from_config(dataset_config)
        dataset_length = len(self.dataset)

        if first_n is not None:
            first_n = int(first_n)
            if first_n <= 0:
                raise ValueError(f"first_n must be positive, got {first_n}")
            if first_n > dataset_length:
                raise ValueError(
                    f"first_n={first_n} exceeds dataset length {dataset_length}"
                )
            self.indices = list(range(first_n))
            self.selection_mode = "first_n"
        else:
            if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
                raise ValueError(
                    "indices must be a sequence of integers when first_n is not provided."
                )
            self.indices = [int(index) for index in indices]
            if not self.indices:
                raise ValueError("indices must not be empty")
            invalid = [index for index in self.indices if index < 0 or index >= dataset_length]
            if invalid:
                raise IndexError(
                    f"FixedSubsetDataset indices out of range for dataset length {dataset_length}: {invalid}"
                )
            self.selection_mode = "indices"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        subset_index = int(idx)
        source_index = self.indices[subset_index]
        return self.dataset[source_index]

    def __getattr__(self, name):
        if name in {"dataset", "dataset_config", "indices", "selection_mode"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)
