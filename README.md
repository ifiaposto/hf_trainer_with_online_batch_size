# Online Batch Size Adaptation in Hugging Face Trainer

## 🎉 New Feature
This repository provides a minimal extension of Hugging Face's Trainer to support dynamically changing batch sizes at each training step.

## :dart: Motivation 

Training with batch sizes that can adapt on-the-fly has been shown to be beneficial in multiple ways:

1. **Improving training efficiency.**: Empirical studies suggest that gradually increasing the batch size—combined with learning rate decay—can achieve the convergence benefits of smaller batch sizes while leveraging the performance advantages of large, multi-GPU batch sizes [1]. This training scheme has been revisited in cutting-edge large language models like DeepSeek-V2 [2].
2. **Supporting advanced learning algorithms for adaptively mixing multiple training data sources.** In such cases, the number of examples drawn from each data stream may be unknown *a priori* and should be dynamically balanced based on training metrics to optimize their impact on the loss function.
   * In **multi-task learning**, the data sources correspond to different tasks [3].
   * In **incremental learning**, one must balance knowledge retention with the ability to generalize to new data by effectively mixing novel and past training examples [4,5].


## :hammer_and_wrench: Installation

The repo is built on Python 3.12.9. You can use the following command to install the requirements:
```
pip install -r requirements.txt
```
Below is the exact cuda environment:

```
nvidia-cublas-cu12       12.4.5.8
nvidia-cuda-cupti-cu12   12.4.127
nvidia-cuda-nvrtc-cu12   12.4.127
nvidia-cuda-runtime-cu12 12.4.127
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.2.1.3
nvidia-curand-cu12       10.3.5.147
nvidia-cusolver-cu12     11.6.1.9
nvidia-cusparse-cu12     12.3.1.170
nvidia-cusparselt-cu12   0.6.2
nvidia-nccl-cu12         2.21.5
nvidia-nvjitlink-cu12    12.4.127
nvidia-nvtx-cu12         12.4.127

```

## :rocket: Quickstart

### Using the trainer

You need only to define a callable implementing the batch size scheduler and pass it as an argument to the trainer:

```python
#   < ------- Write your own batch size scheduler ------- >
def custom_batch_size_scheduler(step: int,
                                batch_size: int,
                                interval=5,
                                increment=1):
    """
        step: current optimization step to be provided by the trainer.
        batch_size: current optimization step  to be provided by the trainer.
    """

    if step % interval == 0 and step > 0:
        return batch_size + increment
    return batch_size


# Extended trainer using the adaptive batch sampler.
trainer = AdaptiveBatchSizeTrainer(
    batch_size_scheduler=partial(custom_batch_size_scheduler,
                                 interval=5,
                                 increment=1),
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)
```
`AdaptiveBatchSizeTrainer`  inherits from the Trainer and TrainingArguments. You can fine the full list of training arguments for running your script [here]([https://www.google.com](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments)).

> [!IMPORTANT]
> The repo currently controls the optimization length exclusively via the argument `num_train_epochs`. This corresponds to the number of times the training dataset will be parsed by the model. Note that the number of optimization steps might not be known a priori if your batch size scheduler works online. To avoid overriding `Trainer`'s `_inner_training_loop` and maintain consistency, `max_steps` should be set to -1.
>

### Understanding the logs
`AdaptiveBatchSizeTrainer` extends and revises `Trainer`'s training logs to account for the varying length of each epoch:

* `train_coverage`: It refers to the percent of train examples seen so far in the current epoch. Note that this might not increase linearly depending on the batch size scheduler you use.
* `epoch`: It is now updated not given the optimization steps performed so far but based on the percent of train examples seen in total up to the current optimization step. This is equivalent to summing `train_coverage` across epochs.


##  :monocle_face: Demo

Below are some sample run commands for our test code using different configurations of single/multi-gpu training and `dataloader_drop_last` (which ignores the last batch when there aren't sufficient training examples left). We inspect the output for the last one.

 <details><summary>single-gpu, ignore the last batch.   </summary>
 
```
python -m torch.distributed.run --nproc-per-node=1 --master_port=29619 -m test_replay_dataloader \
    --output_dir ./results \
    --logging_dir ./logs \
    --logging_steps 1 \
    --save_strategy no \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --dataloader_drop_last
```  
</details>

 <details><summary>multi-gpu, ignore the last batch.   </summary>
 
```
python -m torch.distributed.run --nproc-per-node=2 --master_port=29619 -m test_replay_dataloader \
    --output_dir ./results \
    --logging_dir ./logs \
    --logging_steps 1 \
    --save_strategy no \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --dataloader_drop_last
```  
</details>

</details>

<details><summary>single-gpu, use the last batch.   </summary>
 
```
python -m torch.distributed.run --nproc-per-node=1 --master_port=29619 -m test_replay_dataloader \
    --output_dir ./results \
    --logging_dir ./logs \
    --logging_steps 1 \
    --save_strategy no \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --dataloader_drop_last False
    
```  
</details>

<details><summary>multi-gpu, use the last batch.   </summary>
 
```
python -m torch.distributed.run --nproc-per-node=2 --master_port=29619 -m test_replay_dataloader \
    --output_dir ./results \
    --logging_dir ./logs \
    --logging_steps 1 \
    --save_strategy no \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --dataloader_drop_last False
    
```  
</details>

* we notice every 5 steps (epoch 1.4, 3.6 etc) the local batch size increases by one and according to the scheduler.
* since dataloader_drop_last= False when train_coverage=1.0 the true batch size might be different than the scheduler's batch size.
  
   <details><summary>output for multi-gpu, with the last batch included.   </summary>
  
  ```
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 1,  indices tensor([1], device='cuda:1')                                                                                  
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 1,  indices tensor([4], device='cuda:0')                                                                                  
  [rank0]:[W224 16:51:33.619610762 reducer.cpp:1400] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in $
  n extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. 
  Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())                                                     
  [rank1]:[W224 16:51:33.629522783 reducer.cpp:1400] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in $
  n extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. 
  Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())                                                     
  {'loss': 0.5852, 'grad_norm': 11.941976547241211, 'learning_rate': 4.9500000000000004e-05, 'train coverage': 0.2, 'epoch': 0.2}                                                                             
    1%|█▋                                                                                                                                                                     | 1/100 [00:00<00:52,  1.90it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 1,  indices tensor([7], device='cuda:0')                                                                                  
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 1,  indices tensor([5], device='cuda:1')                                                                                  
  {'loss': 0.7744, 'grad_norm': 7.989934921264648, 'learning_rate': 4.9e-05, 'train coverage': 0.4, 'epoch': 0.4}                                                                                             
    2%|███▎                                                                                                                                                                   | 2/100 [00:00<00:28,  3.38it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 1,  indices tensor([3], device='cuda:0')                                                                                  
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 1,  indices tensor([9], device='cuda:1')                                                                                  
  {'loss': 0.6889, 'grad_norm': 3.215988874435425, 'learning_rate': 4.85e-05, 'train coverage': 0.6, 'epoch': 0.6}                                                                                            
    3%|█████                                                                                                                                                                  | 3/100 [00:00<00:23,  4.18it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 1,  indices tensor([0], device='cuda:0')                                                                                  
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 1,  indices tensor([8], device='cuda:1')                                                                                  
  {'loss': 0.7219, 'grad_norm': 5.315413475036621, 'learning_rate': 4.8e-05, 'train coverage': 0.8, 'epoch': 0.8}                                                                                             
    4%|██████▋                                                                                                                                                                | 4/100 [00:01<00:20,  4.73it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 1,  indices tensor([2], device='cuda:1')                                                                                  
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 1,  indices tensor([6], device='cuda:0')                                                                                  
  {'loss': 0.7349, 'grad_norm': 9.186591148376465, 'learning_rate': 4.75e-05, 'train coverage': 1.0, 'epoch': 1.0}                                                                                            
    5%|████████▎                                                                                                                                                              | 5/100 [00:01<00:17,  5.46it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 2,  indices tensor([6, 2], device='cuda:1')                                                                               
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 2,  indices tensor([5, 1], device='cuda:0')                                                                               
  {'loss': 0.6667, 'grad_norm': 7.5692901611328125, 'learning_rate': 4.7e-05, 'train coverage': 0.4, 'epoch': 1.4}                                                                                            
    6%|██████████                                                                                                                                                             | 6/100 [00:01<00:14,  6.41it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 2,  indices tensor([8, 3], device='cuda:1')                                                                               
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 2,  indices tensor([0, 9], device='cuda:0')                                                                               
  {'loss': 0.6908, 'grad_norm': 1.293290615081787, 'learning_rate': 4.6500000000000005e-05, 'train coverage': 0.8, 'epoch': 1.8}                                                                              
    7%|███████████▋                                                                                                                                                           | 7/100 [00:01<00:14,  6.41it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 1,  indices tensor([7], device='cuda:0')
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 1,  indices tensor([4], device='cuda:1')
  {'loss': 0.6941, 'grad_norm': 0.7848922610282898, 'learning_rate': 4.600000000000001e-05, 'train coverage': 1.0, 'epoch': 2.0}
    8%|█████████████▎                                                                                                                                                         | 8/100 [00:01<00:11,  8.29it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 2,  indices tensor([8, 1], device='cuda:0')
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 2,  indices tensor([7, 5], device='cuda:1')
  {'loss': 0.6856, 'grad_norm': 9.592278480529785, 'learning_rate': 4.55e-05, 'train coverage': 0.4, 'epoch': 2.4}
    9%|███████████████                                                                                                                                                        | 9/100 [00:01<00:11,  7.86it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 2,  indices tensor([6, 0], device='cuda:0')
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 2,  indices tensor([9, 4], device='cuda:1')
  {'loss': 0.6908, 'grad_norm': 0.9208756685256958, 'learning_rate': 4.5e-05, 'train coverage': 0.8, 'epoch': 2.8}
   10%|████████████████▌                                                                                                                                                     | 10/100 [00:01<00:11,  7.94it/s$
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 1,  indices tensor([2], device='cuda:0')
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 1,  indices tensor([3], device='cuda:1')
  {'loss': 0.7796, 'grad_norm': 20.710803985595703, 'learning_rate': 4.4500000000000004e-05, 'train coverage': 1.0, 'epoch': 3.0}
   11%|██████████████████▎                                                                                                                                                   | 11/100 [00:01<00:10,  8.41it/s]
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 3,  indices tensor([6, 3, 8], device='cuda:0')
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 3,  indices tensor([0, 7, 5], device='cuda:1')
  {'loss': 0.5507, 'grad_norm': 9.137839317321777, 'learning_rate': 4.4000000000000006e-05, 'train coverage': 0.6, 'epoch': 3.6}
   12%|███████████████████▉                                                                                                                                                  | 12/100 [00:01<00:10,  8.41it/s]
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 2,  indices tensor([9, 4], device='cuda:1')
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 2,  indices tensor([1, 2], device='cuda:0')
  {'loss': 0.8429, 'grad_norm': 9.582053184509277, 'learning_rate': 4.35e-05, 'train coverage': 1.0, 'epoch': 4.0}
   13%|█████████████████████▌                                                                                                                                                | 13/100 [00:01<00:09,  9.06it/s]
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 0) local batch size 3,  indices tensor([0, 9, 7], device='cuda:0')
  compute loss Current GPU device: Tesla V100-SXM2-32GB (Device 1) local batch size 3,  indices tensor([4, 6, 3], device='cuda:1')
  {'loss': 0.7207, 'grad_norm': 5.453138828277588, 'learning_rate': 4.3e-05, 'train coverage': 0.6, 'epoch': 4.6}
  
  ```
  </details>

##  :nerd_face: Solution Outline

### Challenges

### Main Components

### Future Optimizations

## :books: References

[1] Devarakonda, A., Naumov, M. and Garland, M., 2017. Adabatch: Adaptive batch sizes for training deep neural networks. arXiv preprint arXiv:1712.02029.

[2] Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D. and Yang, D., 2024. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434.

[3] Li, Z., Deng, Y., Zhong, P., Razaviyayn, M. and Mirrokni, V., 2025. PiKE: Adaptive Data Mixing for Multi-Task Learning Under Low Gradient Conflicts. arXiv preprint arXiv:2502.06244.

[4] Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P.K., Torr, P.H. and Ranzato, M.A., 2019. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486.

[5] Wu, T., Luo, L., Li, Y.F., Pan, S., Vu, T.T. and Haffari, G., 2024. Continual learning for large language models: A survey. arXiv preprint arXiv:2402.01364.

## :bouquet: Buy Me Flowers (Citation)

If you use this repo in your research, please cite using the following BibTeX entry:

```
@software{adaptive_batch_size_hf_trainer,
title={Online Batch Size Adaptation in Hugging Face Trainer},
author={Apostolopoulou, Ifigeneia},
howpublished = {\url{https://github.com/ifiaposto/hf_trainer_with_online_batch_size}},
year={2025},
}

```



