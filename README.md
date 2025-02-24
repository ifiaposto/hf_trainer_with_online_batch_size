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

##  :nerd_face: Solution Outline

## :books: References

[1] Devarakonda, A., Naumov, M. and Garland, M., 2017. Adabatch: Adaptive batch sizes for training deep neural networks. arXiv preprint arXiv:1712.02029.

[2] Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D. and Yang, D., 2024. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434.

[3] Li, Z., Deng, Y., Zhong, P., Razaviyayn, M. and Mirrokni, V., 2025. PiKE: Adaptive Data Mixing for Multi-Task Learning Under Low Gradient Conflicts. arXiv preprint arXiv:2502.06244.

[4] Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P.K., Torr, P.H. and Ranzato, M.A., 2019. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486.

[5] Wu, T., Luo, L., Li, Y.F., Pan, S., Vu, T.T. and Haffari, G., 2024. Continual learning for large language models: A survey. arXiv preprint arXiv:2402.01364.

## :bouquet: Buy Me Flowers (Cite This Work)

If you use this repo in your research, please cite using the following BibTeX entry:

```
@software{adaptive_batch_size_hf_trainer,
title={Online Batch Size Adaptation in Hugging Face Trainer},
author={Apostolopoulou, Ifigeneia},
howpublished = {\url{https://github.com/ifiaposto/hf_trainer_with_online_batch_size}},
year={2025},
}

```



