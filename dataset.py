import numpy as np
import torch
import torchvision


def mnist():
  ds = torchvision.datasets.MNIST(root="./data", train=True, download=True)
  dst = torchvision.datasets.MNIST(root="./data", train=False, download=True)
  mx = ds.data.float()
  mxt = dst.data.float()
  my = torch.nn.functional.one_hot(ds.targets, num_classes=10).float().numpy()
  myt = torch.nn.functional.one_hot(dst.targets, num_classes=10).float().numpy()
  mx = mx / 256.0
  mxt = mxt / 256.0
  mx = mx.flatten(1).numpy() * 2.0 - 1.0
  mxt = mxt.flatten(1).numpy() * 2.0 - 1.0
  return mx, mxt, my, myt


def fashionmnist():
  ds = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
  dst = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)
  mx = ds.data.float()
  mxt = dst.data.float()
  my = torch.nn.functional.one_hot(ds.targets, num_classes=10).float().numpy()
  myt = torch.nn.functional.one_hot(dst.targets, num_classes=10).float().numpy()
  mx = mx / 256.0
  mxt = mxt / 256.0
  mx = mx.flatten(1).numpy() * 2.0 - 1.0
  mxt = mxt.flatten(1).numpy() * 2.0 - 1.0
  return mx, mxt, my, myt


def cifar10(flip=True):
  ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
  dst = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
  mx = np.moveaxis(ds.data.astype(np.float32), 3, 1)
  mxt = np.moveaxis(dst.data.astype(np.float32), 3, 1)
  eye = np.eye(10)
  my = eye[ds.targets]
  myt = eye[dst.targets]
  if flip:
    mx_flip = mx[:, :, :, ::-1]
    mx = np.concatenate([mx, mx_flip], axis=0)
    my = np.concatenate([my, my], axis=0)
  mx = mx / 256.0
  mxt = mxt / 256.0
  mx = mx.reshape(mx.shape[0], -1) * 2.0 - 1.0
  mxt = mxt.reshape(mxt.shape[0], -1) * 2.0 - 1.0
  return mx, mxt, my, myt


def get_dataset(name):
  """Return numpy array of dataset"""
  if name == "mnist":
    mx, mxt, my, myt = mnist()
    data_shape = (1, 28, 28)
  elif name == "fashion":
    mx, mxt, my, myt = fashionmnist()
    data_shape = (1, 28, 28)
  elif name == "cifar10":
    mx, mxt, my, myt = cifar10()
    data_shape = (3, 32, 32)
  elif name == "celeba":
    mx = np.load("./data/celeba_train.npy") * 255.0 / 256.0 * 2.0 - 1.0
    mxt = np.load("./data/celeba_eval.npy") * 255.0 / 256.0 * 2.0 - 1.0
    data_shape = (3, 64, 64)
    my = np.zeros((mx.shape[0], 1))
    myt = np.zeros((mxt.shape[0], 1))
  else:
    raise NotImplementedError(f"Dataset {name} unknown.")

  perm = np.random.permutation(mx.shape[0])
  mx = mx[perm]
  my = my[perm]
  permt = np.random.permutation(mxt.shape[0])
  mxt = mxt[permt]
  myt = myt[permt]

  assert mx.shape[0] == my.shape[0]
  assert mxt.shape[0] == myt.shape[0]
  assert mx.shape[1] == mxt.shape[1]
  assert my.shape[1] == myt.shape[1]
  mx.flags.writeable = False
  my.flags.writeable = False
  mxt.flags.writeable = False
  myt.flags.writeable = False
  return mx.T, mxt.T, my.T, myt.T, data_shape
