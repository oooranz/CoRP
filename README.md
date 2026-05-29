# CoRP

Initial code release for the paper: Consolidating Rewarded Perturbations for LLM Post-Training.

This code is built on top of the great work
[Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights](https://github.com/sunrainyg/RandOpt).

CoRP performs isotropic perturbation sampling, constructive population collapse,
and adaptive local recentering around pretrained LLM weights.

## Installation

```bash
pip install -r requirements.txt
```

The runtime follows the original RandOpt/vLLM setup. A GPU environment with
PyTorch, vLLM, Ray, Transformers, and Datasets is expected.

## Data

CoRP uses the same data layout conventions as
[RandOpt](https://github.com/sunrainyg/RandOpt/tree/main/data). You can download the all-in-one
data package and place the downloaded files under `data/`.

## Quick Start

```bash
bash scripts/run.sh gsm8k
```

Example with overrides:

```bash
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct CUDA_DEVICES=0,1,2,3 bash scripts/run.sh gsm8k
```


<!-- ## Citation

If you found the resources in this repository useful, please cite our work:

```bibtex

```
 -->
