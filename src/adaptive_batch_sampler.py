#MIT License
#
#Copyright (c) 2025 Ifigeneia Apostolopoulou
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



from typing import Union, Iterable
from collections.abc import Iterator
import itertools

from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler


class AdaptiveBatchSizeDataLoader(DataLoader):
    """
    Customized Dataloader that supports online batch sizes for sampling.
    It  makes use of AdaptiveBatchSampler and its built on torch Dataloader.
    
    For its arguments see:
    
    https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py
    """

    def __init__(self,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 drop_last=False,
                 *args,
                 **kwargs):

        return super().__init__(
            ############### ATTENTION ###############
            ##  batch_sampler is mutually exclussive with batch_size,
            ##  shuffle, sampler, drop_last. see args explanation:
            ## https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py
            ######################################
            batch_size=1,
            shuffle=False,
            sampler=None,
            drop_last=False,
            ############### ATTENTION ###############
            ## we use manually DDP instead of accelerate and we wrap
            ## sampler with  DistributedSampler to support multi-gpu
            ## training. see details below.
            ######################################
            batch_sampler=AdaptiveBatchSampler(
                DistributedSampler(sampler, shuffle=True), batch_size,
                drop_last),
            *args,
            **kwargs)

    def set_epoch(self, epoch: int):
        """
            Set the epoch for this sampler.

            When :attr:`shuffle=True`, this ensures all replicas
            use a different random ordering for each epoch. 
            Otherwise, the next iteration of this sampler will yield 
            the same ordering.
        """

        if hasattr(self.batch_sampler.sampler, "set_epoch"):
            self.batch_sampler.sampler.set_epoch(epoch)

    def set_per_device_batch_size(self, new_size):
        """
        It updates on the fly the local (per device) batch size,
        """
        self.batch_sampler.set_per_device_batch_size(new_size)

    def __len__(self) -> int:
        return len(self.batch_sampler)


class AdaptiveBatchSampler(Sampler[list[int]]):
    r"""
    It revises torch's BatchSampler to yield a mini-batch of indices with size that can change
    dynamically and syncrhonously.
    
    Args:
    
    batch_size: initial batch size.
    
    The full list of args an be found here:
        
    https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py
    """
    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,  #TODO: remove it or optional
        drop_last: bool,
    ) -> None:
        
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool) or
                batch_size <= 0):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.drop_last = drop_last
        self.current_per_device_batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        
        sampler_iter = iter(self.sampler)

        batch = [
            *itertools.islice(sampler_iter, self.current_per_device_batch_size)
        ]
        # note that self.current_per_device_batch_size might vary across 
        # different loop iterations
        while batch:
            if self.drop_last and len(
                    batch) != self.current_per_device_batch_size:
                break
            yield batch
            batch = [
                *itertools.islice(sampler_iter,
                                  self.current_per_device_batch_size)
            ]


    # Note. the actual length  of the sampler can't be determined a priori.
    # We here define a preemptive length. We assume x2 current (with current batch size) length of dataloader.
    # By this interpretation, length of the dataloader corresponds to the maximum length (optimization steps) per epoch,
    # regardless of the actual batch sizes.
    def __len__(self) -> int:
        return (len(self.sampler) // self.current_per_device_batch_size) * 2

    def set_per_device_batch_size(self, new_size):
        """
        It updates per device batch sizes of the sampler,
        """
        self.current_per_device_batch_size = new_size
