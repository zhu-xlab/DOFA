import math
from typing import Iterator, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import RandomGeoSampler, Units
from torchgeo.samplers.utils import get_random_bounding_box


class DistributedRandomGeoSampler(RandomGeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: int,
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # Pad the last batch
        self.num_samples = math.ceil(length / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

        #####  TODO: Check if this is actually working####
        super().__init__(dataset, size, length, roi, units)

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.areas, self.total_size, generator=g, replacement=True
        ).tolist()
        assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        for idx in indices:
            # Choose a random tile, weighted by area
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
