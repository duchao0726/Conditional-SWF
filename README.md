# Conditional Sliced-Wasserstein Flows

Code for the paper [Nonparametric Generative Modeling with Conditional Sliced-Wasserstein Flows](https://arxiv.org/pdf/2305.02164.pdf) (ICML 2023).

## Environment settings and libraries we used in our experiments

This project has been tested under the following environment settings:
- OS: Ubuntu 20.04.5
- GPU: NVIDIA A100 80GB ($\times$ 8)
- Python: 3.8.10
- jax: 0.3.24
- jaxlib: 0.3.24+cuda11.cudnn805

## Datasets
We use the standard MNIST, FashionMNIST, and CIFAR10 datasets from `torchvision`. For the CelebA 64\*64 dataset we follow the preprocessing of [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) to first crop the original images to 140\*140 and then resize them to 64\*64:
```python
cd data
python get_celebA.py
```
## Commands
The configuration files for all experiments can be found in the `config` folder. For example, the following command produces the results of the class-conditional generation on MNIST:
```python
python main.py -c configs/mnist_class_cond.yaml
```

## Online Generative Modeling (Experimental)
Our method relies on the CDF functions of the projected target data distributions, which we estimate using empirical distributions represented as sorted arrays. The advantage of sorted arrays, besides being nonparametric, lies in their efficient updating capability, achieved in $\mathcal{O}(\log N)$ time as new data is observed. This unique attribute makes our method potentially adaptable for online generative modeling. Here we present two preliminary demonstrations.

**Online _Unconditional_ Generative Modeling**

In the unconditional setting, we assume sequential observation of data class by class, starting from the first class of MNIST (digit '0') to the tenth class (digit '9'), followed by Fashion-MNIST classes from the first to the tenth. The CDF functions of the projected target data distributions are continuously updated as new data arrives, leading to dynamic changes in the _batched samples_ (i.e., run Algorithm 1 with latest CDF functions), as shown below.

<div align="center">
  <img src="./assets/online_uncond.gif"/>
</div>

**Online _Conditional_ Generative Modeling**

In the class-conditional setting, we begin with fully observed MNIST and proceed with sequential observation of Fashion-MNIST data class by class, starting from the first class (T-shirt/top) to the tenth class (Ankle boot). The resulting dynamic changes in class-conditional _batched samples_ are illustrated below.

<div align="center">
  <img src="./assets/online_class_cond.gif"/>
</div>

## Citation
If you find this project helpful in your research, please consider citing our paper:
  ```
@inproceedings{du2023nonparametric,
  title={Nonparametric Generative Modeling with Conditional Sliced-Wasserstein Flows},
  author={Du, Chao and Li, Tianbo and Pang, Tianyu and Yan, Shuicheng and Lin, Min},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
  ```