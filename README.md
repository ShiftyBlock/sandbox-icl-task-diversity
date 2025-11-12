# Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression

Code for [Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression
](https://arxiv.org/abs/2306.15063)

**PyTorch Implementation**

This is a PyTorch implementation of the original JAX/Flax codebase. The implementation supports training on both CPU and GPU.

## Setup

Create a Python virtual environment using:
```sh
conda create -n icl -y python=3.10
conda activate icl
```

Install dependencies using:
```sh
pip install -r requirements.txt
pip install -e .
```

## Training

To train a model, modify `icl/configs/example.py` and then run:
```sh
python run.py --config=icl/configs/example.py
```

## Hardware Requirements

This PyTorch version can run on:
- **CPU**: For smaller experiments and testing
- **GPU**: Recommended for full-scale experiments (CUDA-compatible GPUs)
- **Multi-GPU**: Support for distributed training can be added if needed

## Differences from Original Implementation

The original codebase was designed for Google's TPU Research Cloud using JAX/Flax. This PyTorch version:
- Uses PyTorch instead of JAX/Flax
- Supports CPU and GPU (CUDA) training
- Maintains the same model architecture and training procedures
- Uses PyTorch's native optimizers and learning rate schedulers
- Provides equivalent functionality with PyTorch idioms

## Citation

If you use this code, please cite the original paper:
```bibtex
@article{pretraining2023,
  title={Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression},
  author={Your Authors Here},
  journal={arXiv preprint arXiv:2306.15063},
  year={2023}
}
```
