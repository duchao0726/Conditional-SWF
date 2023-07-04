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