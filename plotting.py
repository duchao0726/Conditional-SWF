import os

import imageio
import numpy as np
import torch
import torchvision


def save_image(args, i, data, prefix="", nrow=None):
  data = (np.array(data) + 1.0) / 2.0
  if args.dataset in ["mnist", "fashion"]:
    data_shape = (1, 28, 28)
  if args.dataset == "cifar10":
    data_shape = (3, 32, 32)
  if args.dataset == "celeba":
    data_shape = (3, 64, 64)
  nrow = int(np.sqrt(data.shape[0])) if nrow is None else nrow
  torchvision.utils.save_image(torch.from_numpy(data.reshape(-1, *data_shape)), os.path.join(args.dirname, "images", prefix + f"_samples_{i:04d}.png"), nrow=nrow)


def make_video(args, prefix="", fps=24, max_frame=100000):
  fileList = [os.path.join(args.dirname, "images", f"{prefix}_{i:04d}.png") for i in range(max_frame + 1)]
  writer = imageio.get_writer(os.path.join(args.dirname, "videos", f"{prefix}.mp4"), fps=fps)
  for im in fileList:
    if os.path.exists(im):
      writer.append_data(imageio.v2.imread(im))
  writer.close()
