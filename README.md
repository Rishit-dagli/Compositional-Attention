# Compositional Attention

![PyPI](https://img.shields.io/pypi/v/compositional-attention)
[![Upload Python Package](https://github.com/Rishit-dagli/compositional-attention/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Rishit-dagli/compositional-attention/actions/workflows/python-publish.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![GitHub stars](https://img.shields.io/github/stars/Rishit-dagli/compositional-attention?style=social)](https://github.com/Rishit-dagli/compositional-attention/stargazers)
[![GitHub followers](https://img.shields.io/github/followers/Rishit-dagli?label=Follow&style=social)](https://github.com/Rishit-dagli)
[![Twitter Follow](https://img.shields.io/twitter/follow/rishit_dagli?style=social)](https://twitter.com/intent/follow?screen_name=rishit_dagli)

This repository is an implementation of [Compositional Attention: Disentangling Search and Retrieval](https://arxiv.org/abs/2110.09419) by MILA. Revisiting standard Multi-head attention through the lens of multiple parallel and independent search and retrieval mechanisms, this leads to static pairings between searches and retrievals, often leading to redundancy of parameters. They reframe the "heads" of multi-head attention as "searches", and once the multi-headed/searched values are aggregated, there is an extra retrieval step (using attention) off the searched results. The experiments establish this as an easy drop-in replacement for Multi-head attention.

![](media/architecture.PNG)

## Installation

Run the following to install:

```sh
pip install compositional-attention
```

## Developing `compositional-attention`

To install `compositional-attention`, along with tools you need to develop and test, run the following in your virtualenv:

```sh
git clone https://github.com/Rishit-dagli/compositional-attention.git
# or clone your own fork

cd compositional-attention
pip install -e .[dev]
```

## Usage

```py
import tensorflow as tf
from compositional_attention import CompositionalAttention

attn = CompositionalAttention(
    dim = 1024,            # input dimension
    dim_head = 64,         # dimension per attention 'head' - head is now either search or retrieval
    num_searches = 8,      # number of searches
    num_retrievals = 2,    # number of retrievals
    dropout = 0.1,          # dropout of attention of search and retrieval
)

tokens = tf.random.uniform([1, 512, 1024])  # tokens
mask = tf.ones([1, 512], dtype=tf.dtypes.bool)  # mask

out = attn(tokens, mask = mask) # (1, 512, 1024)
```

## Want to Contribute 🙋‍♂️?

Awesome! If you want to contribute to this project, you're always welcome! See [Contributing Guidelines](CONTRIBUTING.md). You can also take a look at [open issues](https://github.com/Rishit-dagli/Compositional-Attention/issues) for getting more information about current or upcoming tasks.

## Want to discuss? 💬

Have any questions, doubts or want to present your opinions, views? You're always welcome. You can [start discussions](https://github.com/Rishit-dagli/Compositional-Attention/discussions).

## Citations

```bibtex
@misc{https://doi.org/10.48550/arxiv.2110.09419,
  doi = {10.48550/ARXIV.2110.09419},
  
  url = {https://arxiv.org/abs/2110.09419},
  
  author = {Mittal, Sarthak and Raparthy, Sharath Chandra and Rish, Irina and Bengio, Yoshua and Lajoie, Guillaume},
  
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Compositional Attention: Disentangling Search and Retrieval},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

Official [PyTorch implmentation](https://github.com/sarthmit/compositional-attention) and Phil Wang's [PyTorch implmenetation](https://github.com/lucidrains/compositional-attention-pytorch).
