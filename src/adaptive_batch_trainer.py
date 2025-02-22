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

from typing import Dict, Callable
import torch
import torch.distributed as dist

from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

if is_datasets_available():
    import datasets

from .adaptive_batch_sampler import AdaptiveBatchSizeDataLoader
from .adaptive_batch_trainer_callback import AdaptiveFlowCallback, DynamicBatchSizeCallback

debugging = True  #set it to True for demo purposes


class AdaptiveBatchSizeTrainer(Trainer):
    """
    It extends hf  Trainer to support training with on-the-fly batch sizes.
    
    https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer.py#L290
    
    Args:
        batch_size_scheduler: callable that takes two integer arguments (optimization step
        and current batch size) and returns the local (per device) batch size to be used at the
        next step.
        
    The rest of arguments are identical to hf's Trainer and a list can be found:
    
    https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/training_args.py
    """

    def __init__(self,
                 batch_size_scheduler: Callable[[int, int], int],
                 callbacks=None,
                 *args,
                 **kwargs):

        self.batch_size_scheduler = batch_size_scheduler
        #add a callback to handle trainer's flow and state. needed to determine the end of an epoch.
        default_callbacks = [AdaptiveFlowCallback]
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        super().__init__(callbacks=callbacks, *args, **kwargs)

    def log(self, logs: Dict[str, float], *args):
        """
        It extends logs to also report train_coverage, i.e,  percent of train examples seen so far in current epoch.
        """
        if self.current_train_dataloader is not None:
            logs["train coverage"] = self.state.train_coverage

        super().log(logs, *args)

    def demo(self, model, inputs, return_outputs=False):
        """
        testing code.
        
        Check that:
            i)    the number of train examples that are actually passed to the model might change across steps and according to the current batch size of the scheduler.
                  Note, however, that the last batch of the epoch might have size other than the  current batch size of the scheduler if drop_past=False. This is because
                    there aren't sufficient data in the dataloader at the end of the epoch.
            ii)  when a new epoch begins, the inputs are coming with a different order (the sampler reshuffles the order of the training examples).
        """

        indices = inputs['indices']
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        print(
            f"compute loss Current GPU device: {device_name} (Device {current_device}) local batch size {len(indices)},  indices {indices}"
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        It overrides Trainer's compute_loss to update the adaptive batch trainer state.
        """

        if debugging:
            self.demo(model, inputs)
            inputs.pop('indices')

        labels = inputs['labels']

        # compute current lglobal batch size across all devices
        local_num_seen_train_examples = torch.tensor(0,
                                                     device=self.args.device,
                                                     dtype=torch.long)
        local_num_seen_train_examples += len(labels)
        # gather across devices
        if torch.distributed.is_initialized():
            dist.all_reduce(local_num_seen_train_examples, op=dist.ReduceOp.SUM)

        # update the state
        if dist.get_rank() == 0:
            self.state.num_seen_train_examples += local_num_seen_train_examples
            self.state.total_num_seen_train_examples += local_num_seen_train_examples

        # Broadcast the updated global count to all devices
        dist.broadcast(self.state.num_seen_train_examples, src=0)
        # Update train coverage for current epoch to all devices
        self.state.train_coverage = round(
            self.state.num_seen_train_examples.item() /
            self.state.num_train_examples, 5)
        # Each epoch might have different number of optimization steps due to varying batch size.
        # we therefore define an epoch, not based on the number of optimization steps, but based on
        # the number of the training examples we have passed to the model so far.
        self.state.num_epochs = round(
            self.state.total_num_seen_train_examples.item() /
            self.state.num_train_examples, 5)

        # call parent compute_loss
        return super().compute_loss(model, inputs, return_outputs)

    def _inner_training_loop(self, *args, **kwargs):

        # this is a convention for simplicity and to be compatible with current code
        # in inner_training_loop for detrmining num_train_samples and num_train_epochs
        # in case args.max_steps > 0. in other words, only:
        # num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs in [1]
        # for the case self.args.max_steps==0 and has_length(train_dataloader) holds for the adaptive
        # dataloader.
        # [1] https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer.py#L2037

        if self.args.max_steps > 0:
            raise ValueError(
                "For trainer with adaptive batch size, the number of training epochs"
                "should determine the length of training.")

        #It's needed to access the train_dataloader outside inner_training_loop (in the customized training_step)
        self.current_train_dataloader = None

        # Proceed with the usual training loop
        return super()._inner_training_loop(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        
        ############### ATTENTION ####################
        ## We initialize extended trainer state.
        ## This needs to be done here and not at the beginning of our 
        ## _inner_training_loop. This is because parent  _inner_training_loop
        ## resets self.state hence any change prior to its call wouldn't be visible.
        ############################################  
        if not hasattr(self.state, "train_coverage"):
            self.state.train_coverage = 0
           # We extend trainer's state with number of training examples 
            self.state.num_train_examples = self.num_examples(
                self.current_train_dataloader)

            # We extend trainer's state with number of training examples passed to 
            # model for the current epoch and in total. These are needed for the logs.
            ############### ATTENTION ####################
            ## we define a tensor here to enforce syncrhonization across devices.
            ###################### ####################
            self.state.num_seen_train_examples = torch.tensor(
                0, device=self.args.device, dtype=torch.long)
            self.state.total_num_seen_train_examples = torch.tensor(
                0, device=self.args.device, dtype=torch.long)
                
        # Proceed with the usual training step
        return super().training_step(*args, **kwargs)

    def get_train_dataloader(self):
        """ 
        Overrides Trainer's returned train dataloader [1] to support adaptive batch sampler.
        i) It properly sets DataLoader to make use of the AdaptiveBatchSampler and,
        ii) provides access to the train dataloader outside the inner_training_loop. 
        
        [1] https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
        
        """

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset,
                                                  datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset,
                                                        description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training")

        ############### ATTENTION ###############
        ## set num_workers=0 otherwise prefetching will be applied
        ## and the change of batch size might not be put into effect
        ## syncrhonously
        
        ## see args explanation :
        ## https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py
        ####################################

        self.args.dataloader_num_workers = 0
        self.args.dataloader_prefetch_factor = None

        self.current_train_dataloader = AdaptiveBatchSizeDataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._get_train_sampler(),
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            worker_init_fn=seed_worker,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
        ############### ATTENTION ###############
        ## we do not wrap Dataloader with self.accelerator.prepare
        ## as in https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer.py
        ## this is because it creates a DataloaderShard object.
        ## see prepare_data_loader call here: https://github.com/huggingface/accelerate/blob/main/src/accelerate/accelerator.py#L2192
        ## and prepare_data_loader definition here: https://github.com/huggingface/accelerate/blob/main/src/accelerate/data_loader.py#L1187
        ## in __iter__ of DataloaderShard (see https://github.com/huggingface/accelerate/blob/main/src/accelerate/data_loader.py#L1187)
        ## we notice that it *always* prefetches a batch which might cause the batch size change not be synchronously applied.
        ######################################
        
        # activate batch size scheduler once train_dataloader becomes available.
        # we need access to dataloader's batch sampler to set its current batch size.
        self.add_callback(
            DynamicBatchSizeCallback(self.batch_size_scheduler,
                                     self.current_train_dataloader))

        return self.current_train_dataloader
