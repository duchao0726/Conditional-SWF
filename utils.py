import errno
import logging
import os
import random
import time

import coloredlogs
import jax
import jax.numpy as jnp
import numpy as np

param_dict = dict(
  seed=0,
  hdim=10000,
  hdim_per_conv=10,
  layer_steps=200,
  step_size=1.0,
  n_batched_particles=250000,
  n_offline_particles=4000,
  forward="sorting",
  inverse="sorting",
  n_bins_particles=200,
  n_bins_data=200,
  downsample="lanczos5",
  dequantize=True,
  pyramidal=True,
  basedir="output",
  expname="experiment",
  dataset="mnist",
  n_viz=400,
  viz_every=100,
  restore_path=None,
  cond=False,
  cond_type="bottom",
  amplifier=1.0,
  init_std=0.1,
  baseline=False,
)


def str2bool(v):
  """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  else:
    raise ValueError("Boolean value expected.")


def add_dict_to_argparser(parser, default_dict):
  for k, v in default_dict.items():
    v_type = type(v)
    if v is None:
      v_type = str
    elif isinstance(v, bool):
      v_type = str2bool
    parser.add_argument(f"--{k}", default=v, type=v_type)


def setup_parser(parser):
  add_dict_to_argparser(parser, default_dict=param_dict)


def setup_seed(seed):
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  random.seed(seed)


def setup_logging(args):
  dirname_base = args.basedir if hasattr(args, "basedir") else "basedir"
  logger = logging.getLogger("COLOREDLOGS")
  FORMAT = "[%(asctime)s] %(message)s"
  DATEFMT = "%H:%M:%S"
  LEVEL_STYLES = dict(
    debug=dict(color="blue"),
    info=dict(color="green"),
    verbose=dict(),
    warning=dict(color="yellow"),
    error=dict(color="red"),
    critical=dict(color="magenta"),
  )
  coloredlogs.install(logger=logger, level="info", fmt=FORMAT, datefmt=DATEFMT, level_styles=LEVEL_STYLES)

  # Determine suffix
  suffix = ""
  suffix += args.dataset if hasattr(args, "dataset") else ""
  suffix += "-" if suffix else ""
  suffix += args.cond_type if hasattr(args, "cond") and hasattr(args, "cond_type") and args.cond else "uncond"
  suffix += "-" if suffix else ""
  suffix += args.forward if hasattr(args, "forward") else ""
  suffix += str(args.n_bins_particles) if hasattr(args, "forward") and hasattr(args, "n_bins_particles") and args.forward == "rqspline" else ""
  suffix += "-" if suffix else ""
  suffix += args.inverse if hasattr(args, "inverse") else ""
  suffix += str(args.n_bins_data) if hasattr(args, "inverse") and hasattr(args, "n_bins_data") and args.inverse == "rqspline" else ""
  suffix += "-" if suffix else ""
  suffix += args.downsample if hasattr(args, "downsample") else ""
  suffix += "-" if suffix else ""
  suffix += "{{" + (str(args.expname if args.expname else "debug") if hasattr(args, "expname") else "") + "}}"
  suffix += "-hd" + str(args.hdim) if hasattr(args, "hdim") else ""
  suffix += "-hdc" + str(args.hdim_per_conv) if hasattr(args, "hdim_per_conv") else ""
  suffix += "-lst" + str(args.layer_steps) if hasattr(args, "layer_steps") else ""
  suffix += "-lr" + str(args.step_size) if hasattr(args, "step_size") else ""
  suffix += "-std" + str(args.init_std) if hasattr(args, "init_std") else ""
  suffix += "-np" + str(args.n_batched_particles) if hasattr(args, "n_batched_particles") else ""
  suffix += "-xi" + str(args.amplifier) if hasattr(args, "amplifier") and hasattr(args, "cond") and args.cond else ""
  suffix += "-seed" + str(args.seed) if hasattr(args, "seed") else ""

  # Determine prefix
  prefix = time.strftime("%Y-%m-%d--%H-%M")

  prefix_counter = 0
  dirname = dirname_base + "/%s.%s" % (prefix, suffix)
  while True:
    try:
      os.makedirs(dirname)
      os.makedirs(os.path.join(dirname, "code"))
      os.makedirs(os.path.join(dirname, "images"))
      os.makedirs(os.path.join(dirname, "videos"))
      os.makedirs(os.path.join(dirname, "particles"))
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise e
      prefix_counter += 1
      dirname = dirname_base + "/%s+%d.%s" % (prefix, prefix_counter, suffix)
      if prefix_counter >= 10:
        exit()
    else:
      break

  formatter = logging.Formatter(FORMAT, DATEFMT)
  logger_fname = os.path.join(dirname, "logfile.txt")
  fh = logging.FileHandler(logger_fname)
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  # logger.propagate = False

  args.dirname = dirname

  return logger, dirname


def find_neighbors(x, data):
  data_sqnorm = jnp.sum(jnp.square(data), axis=0)
  sqdist = jnp.sum(jnp.square(x)) + data_sqnorm - 2 * jnp.matmul(x, data)
  _, idx = jax.lax.top_k(-sqdist, 10)
  return jnp.vstack([x, data[:, idx].T])
