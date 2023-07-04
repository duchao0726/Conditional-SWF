import glob
import os
import shutil

import configargparse
import jax
import jax.numpy as jnp
import numpy as np

import dataset
import models
import plotting
import utils

parser = configargparse.ArgumentParser()
parser.add("-c", "--config", required=True, is_config_file=True, help="config file path")
utils.setup_parser(parser)
args = parser.parse_args()

# print configs and copy code for reproducibility
logger, dirname = utils.setup_logging(args)
files_to_copy = glob.glob(os.path.dirname(os.path.realpath(__file__)) + "/*.py")
for script_src in files_to_copy:
  script_dst = os.path.abspath(os.path.join(dirname, "code", os.path.basename(script_src) + ".bak"))
  shutil.copyfile(script_src, script_dst)

for k, v in sorted(vars(args).items()):
  logger.info("  %30s: %s" % (k, v))

# experimental setups
n_devices = jax.device_count()
devices = jax.devices()
logger.info(f"{n_devices} devices found.")

utils.setup_seed(args.seed)
data_train, data_test, label_train, label_test, data_shape = dataset.get_dataset(args.dataset)
dim, n_train_data = data_train.shape
_, n_test_data = data_test.shape
n_labels = label_train.shape[0]
cdim = dim
hdim = args.hdim
hdim_per_conv = args.hdim_per_conv
layer_steps = args.layer_steps
step_size = args.step_size
n_batched_particles = args.n_batched_particles
n_offline_particles = args.n_offline_particles
n_bins_particles = args.n_bins_particles
n_bins_data = args.n_bins_data
init_std = args.init_std
max_layer = 100000
logger.info(f"dim={dim}, #data={n_train_data}, #test data={n_test_data}")
logger.info(f"hdim={hdim}, hdim_per_conv={hdim_per_conv}, #batched particles={n_batched_particles}, #offline particles={n_offline_particles}, #layer_steps={layer_steps}, stepsize={step_size}")
logger.info(f"forward: {args.forward}, inverse: {args.inverse}")
if args.forward == "rqspline":
  logger.info(f"#bins for particles={n_bins_particles}")
if args.inverse == "rqspline":
  logger.info(f"#bins for data={n_bins_data}")

assert dim == np.prod(data_shape)
nrow = int(np.sqrt(args.n_viz))
assert nrow * nrow == args.n_viz  # use a square number for easy visualization
assert n_offline_particles // n_devices >= args.n_viz  # make sure there are enough particles on device 0 for visualization
assert args.forward in ["rqspline", "sorting"]
assert args.inverse in ["rqspline", "sorting"]
assert n_batched_particles % n_devices == n_offline_particles % n_devices == 0
assert args.downsample.lower() in ["nearest", "lanczos3", "lanczos5"]

# for class-conditional generation, data/particles are in the XxY space
if args.cond and args.cond_type == "class":
  amplifier = args.amplifier
  data_train = np.concatenate([data_train, label_train * amplifier], axis=0)
  cdim = dim + n_labels

# make sure the dataset can be evenly split across devices
if n_train_data % n_devices != 0:
  data_train = np.concatenate([data_train, data_train[:, :n_devices - n_train_data % n_devices]], axis=1)
  n_train_data = data_train.shape[1]

# initialize/restore particles
if args.restore_path:
  if os.path.isfile(os.path.join(args.restore_path, "particles_batched.npy")) and os.path.isfile(os.path.join(args.restore_path, "particles_offline.npy")):
    particles_batched = np.load(os.path.join(args.restore_path, "particles_batched.npy"))
    particles_offline = np.load(os.path.join(args.restore_path, "particles_offline.npy"))
  else:
    raise ValueError(f"Cannot restore from {args.restore_path}")
else:
  particles_batched = np.random.randn(cdim, n_batched_particles) * init_std
  particles_offline = np.random.randn(cdim, n_offline_particles) * init_std

# generate mask and initialize particles for conditional tasks
if args.cond:
  if args.cond_type.lower() == "bottom":
    mask = np.ones(data_shape, dtype=np.float32)
    mask[:, :data_shape[1] // 2, :] = 0.0
  elif args.cond_type.lower() == "right":
    mask = np.ones(data_shape, dtype=np.float32)
    mask[:, :, :data_shape[1] // 2] = 0.0
  elif args.cond_type.lower() == "class":
    mask = np.ones(cdim, dtype=np.float32)
    mask[dim:] = 0.0
  else:
    raise NotImplementedError(f"Condition type {args.cond_type} unknown.")
  mask = np.reshape(mask, (-1, 1))

  if args.cond_type.lower() == "class":
    # for class-conditional generation, we use uniform distribution of class labels
    batched_idx = np.tile(np.repeat(np.arange(n_labels), nrow), n_batched_particles // (n_labels * nrow) + 1)
    offline_idx = np.tile(np.repeat(np.arange(n_labels), nrow), n_offline_particles // (n_labels * nrow) + 1)
    onehot = np.eye(n_labels) * amplifier
    particles_batched[dim:, :] = onehot[:, batched_idx[:n_batched_particles]]
    particles_offline[dim:, :] = onehot[:, offline_idx[:n_offline_particles]]
  else:
    # for image inpainting, we create partially-observed images from the dataset
    n_copies = n_batched_particles // n_train_data
    data_train_samples = data_train[:, :n_batched_particles - n_copies * n_train_data]
    data_train_samples = np.concatenate([data_train] * n_copies + [data_train_samples], axis=1)
    if args.dequantize:  # TODO check if necessary
      data_train_samples = data_train_samples + np.random.rand(*data_train_samples.shape) / 128.0
    particles_batched = particles_batched * mask + data_train_samples * (1.0 - mask)
    assert n_offline_particles % nrow == 0 and n_offline_particles // nrow <= n_test_data  # for easy visualization
    data_test_samples = np.repeat(data_test[:, :n_offline_particles // nrow], nrow, axis=1)
    particles_offline = particles_offline * mask + data_test_samples * (1.0 - mask)
else:
  mask = None

# plot initial particles
samples_0 = np.concatenate([np.reshape(particles_batched[:dim, :args.n_viz].T, (nrow, nrow, -1)), np.reshape(particles_offline[:dim, :args.n_viz].T, (nrow, nrow, -1))], axis=1)
plotting.save_image(args, 0, samples_0, prefix="batched_offline", nrow=nrow * 2)
plotting.save_image(args, 0, data_train[:dim, :args.n_viz].T, prefix="data", nrow=nrow)

# copy data to devices
particles_batched_sh = jax.device_put_sharded(np.split(particles_batched, n_devices, axis=1), devices)
particles_offline_sh = jax.device_put_sharded(np.split(particles_offline, n_devices, axis=1), devices)
data_train_sh = jax.device_put_sharded(np.split(data_train, n_devices, axis=1), devices)
particles_batched_to_save = None

# the "model" defines locally-connected projections and pyramidal schedules
if args.dataset in ["mnist", "fashion"]:
  model = models.mnist_model
elif args.dataset in ["cifar10"]:
  model = models.cifar10_model
elif args.dataset in ["celeba"]:
  model = models.celeba_model
else:
  raise NotImplementedError(f"Model for {args.dataset} unknown.")

if args.baseline:
  model = models.swf_model

transform_layers, transform_steps = model(
  data_shape=data_shape, mask=mask, hdim=hdim, hdim_per_conv=hdim_per_conv, step_size=step_size, layer_steps=layer_steps, forward=args.forward, inverse=args.inverse, n_bins_particles=n_bins_particles, n_bins_data=n_bins_data, downsample_method=args.downsample, dequantize=args.dequantize
)

# generate batched & offline samples
key = jax.random.PRNGKey(args.seed)
steps_mark = list(np.cumsum(transform_steps))
assert len(steps_mark) == len(transform_layers)
for i in range(1, max_layer + 1):
  if args.pyramidal:
    if i > steps_mark[0]:
      steps_mark = steps_mark[1:]
      transform_layers = transform_layers[1:]
      if not transform_layers:
        break
      logger.info(f"Now use {transform_layers[0]}")

    key, wkey = jax.random.split(key)
    key, dkey = jax.random.split(key)
    dkeys = jax.random.split(dkey, n_devices)
    particles_batched_sh, particles_offline_sh, ws_dist_batched_sh, ws_dist_offline_sh, particles_batched_to_save = transform_layers[0](wkey, dkeys, data_train_sh, particles_batched_sh, particles_offline_sh)
  else:
    nf = len(transform_layers)
    key, wkey = jax.random.split(key)
    key, dkey = jax.random.split(key)
    dkeys = jax.random.split(dkey, n_devices)
    particles_batched_sh, particles_offline_sh, ws_dist_batched_sh, ws_dist_offline_sh, particles_batched_to_save = transform_layers[np.random.randint(nf)](wkey, dkeys, data_train_sh, particles_batched_sh, particles_offline_sh)

  logger.info(f"Iter {i:3d}: ws_dist_batched={jnp.mean(ws_dist_batched_sh):.5f}, ws_dist_offline={jnp.mean(ws_dist_offline_sh):.5f}")

  if i % args.viz_every == 0:
    samples_i = jnp.concatenate([jnp.reshape(particles_batched_to_save[0, :dim, :args.n_viz].T, (nrow, nrow, -1)), jnp.reshape(particles_offline_sh[0, :dim, :args.n_viz].T, (nrow, nrow, -1))], axis=1)
    plotting.save_image(args, i, samples_i, prefix="batched_offline", nrow=nrow * 2)

# save final particles and their nearest neighbors
particles_batched = np.moveaxis(np.array(particles_batched_to_save), 0, 1).reshape(cdim, -1)
particles_offline = np.moveaxis(np.array(particles_offline_sh), 0, 1).reshape(cdim, -1)
with open(os.path.join(dirname, "particles", "particles_batched.npy"), "wb") as f:
  np.save(f, particles_batched)
  logger.info(f"{f.name} saved.")
with open(os.path.join(dirname, "particles", "particles_offline.npy"), "wb") as f:
  np.save(f, particles_offline)
  logger.info(f"{f.name} saved.")
plotting.make_video(args, "batched_offline_samples", max_frame=max_layer)

del particles_batched_sh, particles_offline_sh, data_train_sh
data_train = jnp.array(data_train)

# save nearest neighbors of generated particles
all_find_neighbors = jax.vmap(utils.find_neighbors, in_axes=(1, None))
particles_batched_with_neighbors = jnp.reshape(all_find_neighbors(particles_batched[:dim, :args.n_viz], data_train[:dim]), (-1, dim))
particles_offline_with_neighbors = jnp.reshape(all_find_neighbors(particles_offline[:dim, :args.n_viz], data_train[:dim]), (-1, dim))
plotting.save_image(args, 0, particles_batched_with_neighbors, prefix="nn_batched", nrow=11)
plotting.save_image(args, 0, particles_offline_with_neighbors, prefix="nn_offline", nrow=11)
