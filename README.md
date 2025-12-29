# Shifted Window (Swin) Transformer Implementation

This is a clean implementation of the [Swin Transformer](https://arxiv.org/pdf/2103.14030). It includes the model architecture and training logic, as well as configuration options for the Swin variants found in the original paper. The original architecture implementation can be found [here](https://github.com/microsoft/Swin-Transformer). 

The training loop expects either a single CUDA-compatible GPU or a CPU for training. The [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is used for testing purposes. 

All code was written without AI assistance. This [video](https://www.youtube.com/watch?v=Ws2RAh_VDyU) was helpful for understanding the relative position bias.

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run training:
```bash
python ./train.py
```

## Contents
* [Hydra](https://hydra.cc) configuration management
* [PyTorch](https://pytorch.org) architecture implementation and training script