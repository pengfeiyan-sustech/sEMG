import torch
import torch.utils.data


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    """Define a class to create an infinite dataloader"""

    def __init__(self, dataset, weights, batch_size, num_workers, sampler, drop_last=True):
        super().__init__()

        # If no weights are given, set them to one
        if weights == None:
            weights = torch.ones(len(dataset))

        # Create a batch sampler from the given sampler
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        # Create an infinite iterator from the given dataset and batch sampler
        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        while True:
            # Yield the next batch from the infinite iterator
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
