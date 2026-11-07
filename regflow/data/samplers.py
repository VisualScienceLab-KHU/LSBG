import random

from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, *, rank=0, world_size=1, seed=42):
        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size % 2:
            raise ValueError("Balanced batches require a positive, even batch_size per process.")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size), with world_size > 0.")
        self.real_indices = list(dataset.real_indices)
        self.fake_indices = list(dataset.fake_indices)
        if not self.real_indices or not self.fake_indices:
            raise ValueError("Balanced training requires both real and fake samples.")
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.rank = rank
        self.world_size = world_size
        self.seed = int(seed)
        self.epoch = 0
        if len(self) == 0:
            raise ValueError(
                "Not enough samples for one balanced batch on every process. "
                "Reduce batch_size/world_size or increase data_scale."
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        real, fake = self.real_indices.copy(), self.fake_indices.copy()
        rng.shuffle(real)
        rng.shuffle(fake)
        for index in range(len(self) * self.world_size):
            start, stop = index * self.half, (index + 1) * self.half
            batch = real[start:stop] + fake[start:stop]
            rng.shuffle(batch)
            if index % self.world_size == self.rank:
                yield batch

    def __len__(self):
        total = min(len(self.real_indices) // self.half, len(self.fake_indices) // self.half)
        return total // self.world_size


class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, *, rank=0, world_size=1):
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size), with world_size > 0.")
        self.length = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self):
        return max(0, (self.length - self.rank + self.world_size - 1) // self.world_size)
