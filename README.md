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



