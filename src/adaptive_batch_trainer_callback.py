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

from typing import Callable

from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerState, TrainerCallback, TrainerControl

from .adaptive_batch_sampler import AdaptiveBatchSizeDataLoader


class AdaptiveFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the adaptive flow of the training loop with online batch sizes.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        # check for epoch's termination and update logs.

        # all training datapoints have been parsed. you should restart the epoch.
        if state.train_coverage == 1.0:
            control.should_epoch_stop = True

        #override Trainer's epoch (current number of epochs so far) for the logs.
        state.epoch = state.num_epochs

        return control

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        # restart epoch's logs.

        # extend trainer's state with number of effective epochs during training.
        # this is done only once at the beginning of the training (of the first epoch).
        if not hasattr(state, "num_epochs"):
            state.num_epochs = 0

        # reset train coverage (percentage of train examples seen so far) for current epoch.
        if hasattr(state, "num_seen_train_examples"):
            state.num_seen_train_examples.zero_()


class DynamicBatchSizeCallback(TrainerCallback):
    """
     A [`TrainerCallback`] that handles that updates the current per device batch size of the trainer.
     
     Args:
        train_dataloader: dataloader where the batch size changes will be applied to.
        batch_size_scheduler: callable that determines next batch size given current training step 
        and batch size.
     """

    def __init__(self, batch_size_scheduler: Callable[[int, int], int],
                 train_dataloader: AdaptiveBatchSizeDataLoader):
        self.train_dataloader = train_dataloader
        self.batch_size_scheduler = batch_size_scheduler

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):

        # apply batch size changes to the dataloader.
        new_train_batch_size = self.batch_size_scheduler(
            state.global_step,
            self.train_dataloader.batch_sampler.current_per_device_batch_size)
        self.train_dataloader.batch_sampler.set_per_device_batch_size(
            new_train_batch_size)
