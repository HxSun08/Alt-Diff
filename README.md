# Alternating Differentiation for Optimization Layers

The official code repo for the paper "Alternating Differentiation for Optimization Layers" (ICLR 2023). 


![framework](https://user-images.githubusercontent.com/106318028/224615804-75e7a4e0-3216-4d75-9a0b-8c2f273ecd9c.png)


Alt-Diff is an algorithm that decouples optimization layers in a fast and recursive way. Assuming $\theta$ and $x$ represent the input and output of the optimization layer, the Jacobian $\frac{\partial x_{k+1}}{\partial \theta}$ is computed iteratively using the following formula:

$$
  \begin{aligned}[left= {\empheqlbrace}]
     \dfrac{\partial x_{k+1}}{\partial \theta}&=
    -\left(\nabla_x^2 \mathcal{L}(x_{k+1})\right)^{-1}\nabla_{x,\theta} \mathcal{L}(x_{k+1}),\\
     \dfrac{\partial s_{k+1}}{\partial \theta}&=-\dfrac{1}{\rho}{\bf sgn}(s_{k+1})\cdot {\bf 1}^T\odot
    \left(\dfrac{\partial \nu_k}{\partial \theta}+\rho\dfrac{\partial {(Gx_{k+1}-h)}}{\partial \theta}\right),\\
     \dfrac{\partial \lambda_{k+1}}{\partial \theta}&=\dfrac{\partial \lambda_k}{\partial \theta}+\rho\dfrac{\partial (A x_{k+1}-b)}{\partial \theta},\\
     \dfrac{\partial \nu_{k+1}}{\partial \theta}&=\dfrac{\partial \nu_k}{\partial \theta}+\rho\dfrac{\partial (Gx_{k+1}+s_{k+1}-h)}{\partial \theta}.
   \end{aligned}
$$

## Experimental Results

We have uploaded numerical experiments and image classification test on MNIST and CIFAR-10 to substantiate the efficiency of Alt-Diff. 


## Usage

The `numerical_experiment` directory contains randomly generated parameters that can be adjusted to test different dimensions. These experiments are designed to compare the runtime and results of different algorithms.

In the `classification` directory, you can find examples demonstrating the integration of Alt-diff into neural network training processes.

For training the optimization layer in image classification tasks, execute the following command:
```sh
cd classification
python train.py --dataset cifar-10 [MODEL_NAME]
```

We strongly recommend running the code in a CPU environment when comparing our model with existing methods, as `qpth` benefits from GPU acceleration, introducing an unfair advantage otherwise.

## Updates

2023/3/16 Update the numerical experiments with the optimization parameter $b(\theta)$ in some experiments.

2024/1/8 Update the classification test to show the combination of optimization layer within neural networks.

Please note that we will continuously add more detailed code examples and implement GPU acceleration in Alt-Diff to the repository in the future.


## Citations
If you find this paper useful in your research, please consider citing:
```
@inproceedings{
sun2023alternating,
title={Alternating Differentiation for Optimization Layers},
author={Haixiang Sun and Ye Shi and Jingya Wang and Hoang Duong Tuan and H. Vincent Poor and Dacheng Tao},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=KKBMz-EL4tD}
}
```
