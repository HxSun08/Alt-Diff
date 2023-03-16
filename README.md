# Alternating Differentiation for Optimization Layers

The official code repo for the paper "Alternating Differentiation for Optimization Layers" (ICLR 2023). 


![framework](https://user-images.githubusercontent.com/106318028/224615804-75e7a4e0-3216-4d75-9a0b-8c2f273ecd9c.png)


Alt-Diff is an algorithm that decouples optimization layers in a fast and recursive way. 


## Experimental Results

We have first uploaded some numerical experiments to substantiate the efficiency of Alt-Diff. 


## Usage

The `numerical_experiment` directory contains randomly generated parameters that can be adjusted to test different dimensions. These experiments are designed to compare the runtime and results of different algorithms.

Please note that these experiments are only simple examples, and more detailed code examples will be added to the repository in the future.

## Updates

2023/3/16 Update the numerical experiments with the optimization parameter $b(\theta)$ in some experiments.

## Citations
If you find this paper useful in your research, please consider citing:
```
@inproceedings{
sun2023alternating,
title={Alternating Differentiation for Optimization Layers},
author={Haixiang Sun and Ye Shi and Jingya Wang and Hoang Duong Tuan and H. Vincent Poor and Dacheng Tao},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=KKBMz-EL4tD}
}
```
