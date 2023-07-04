import jax
import jax.numpy as jnp
import numpy as np


def uniform(key, dim, hdim, **kwargs):
  w = jax.random.normal(key, shape=(hdim, dim))
  w_norm = jnp.linalg.norm(w, axis=1, keepdims=True)
  w = w / w_norm
  return w


def conv(key, input_shape, hdim, n_filters, kernel_sizes, strides=1, paddings="SAME", dilations=1, normalize=True, **kwargs):
  kernel_sizes = kernel_sizes if isinstance(kernel_sizes, (list, tuple)) else (kernel_sizes,)
  n_filters = n_filters if isinstance(n_filters, (list, tuple)) else (n_filters,) * len(kernel_sizes)
  strides = strides if isinstance(strides, (list, tuple)) else (strides,) * len(kernel_sizes)
  paddings = paddings if isinstance(paddings, (list, tuple)) else (paddings,) * len(kernel_sizes)
  dilations = dilations if isinstance(dilations, (list, tuple)) else (dilations,) * len(kernel_sizes)
  assert len(n_filters) == len(kernel_sizes) == len(strides) == len(paddings) == len(dilations)

  n_convs = len(n_filters)
  n_filters = (input_shape[0],) + n_filters

  kernels = []
  for i in range(n_convs):
    key, subkey = jax.random.split(key)
    kernels.append(jax.random.normal(subkey, shape=(n_filters[i + 1], n_filters[i], kernel_sizes[i], kernel_sizes[i])) * 0.1)
    # kernels.append(jax.random.laplace(subkey, shape=(n_filters[i + 1], n_filters[i], kernel_sizes[i], kernel_sizes[i])) * 0.1)

  # we obtain the equivalent projections through vjp through the forward mapping
  def f(x):
    for i in range(n_convs):
      stride = (strides[i],) * 2
      padding = paddings[i] if isinstance(paddings[i], str) else (paddings[i],) * 2
      dilation = (dilations[i],) * 2
      x = jax.lax.conv_general_dilated(x, kernels[i], window_strides=stride, padding=padding, rhs_dilation=dilation)
    return x

  x_dummy = jnp.zeros((1, *input_shape))
  f_value, f_vjp = jax.vjp(f, x_dummy)
  outdim = np.prod(f_value.shape)

  hdim = outdim if hdim is None else hdim
  assert outdim >= hdim

  if outdim > hdim:
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, outdim)
    I = jax.nn.one_hot(perm[:hdim], outdim)
  else:
    I = jnp.eye(outdim)

  def wi(v):
    return f_vjp(v.reshape(f_value.shape))[0]

  w = jax.vmap(wi, in_axes=0)(I)
  w = jnp.reshape(w, (w.shape[0], np.prod(input_shape)))

  if normalize:
    w_norm = jnp.linalg.norm(w, axis=1, keepdims=True)
    w = w / w_norm

  return w


# a wrapper function to obtain the upsampled projection from lower resolutions
def downsample_slicer(key, slice_fn, input_shape, down_size, **kwargs):
  down_shape = (input_shape[0], *down_size)
  kwargs["dim"] = np.prod(down_shape)
  kwargs["input_shape"] = down_shape
  sub_w = slice_fn(key, **kwargs)
  sub_w = jnp.reshape(sub_w, (sub_w.shape[0], *down_shape))
  method = kwargs["method"] if "method" in kwargs else "lanczos3"
  w = jax.image.resize(sub_w, (sub_w.shape[0], *input_shape), method=method)
  w = jnp.reshape(w, (w.shape[0], np.prod(input_shape)))
  w_norm = jnp.linalg.norm(w, axis=1, keepdims=True)
  w = w / w_norm
  return w
