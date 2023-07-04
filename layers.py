import functools

import jax
import jax.numpy as jnp
import jax.scipy


def sorting_forward(xs, x):
  nx = xs.shape[0]
  idx = jnp.searchsorted(xs, x)
  im1 = jnp.clip(idx - 1, 0, nx - 1)
  i = jnp.clip(idx, 0, nx - 1)
  # if falls in the middle
  delta_x = xs[i] - xs[im1]
  offset_x = x - xs[im1]
  rel_offset = jnp.clip(jnp.nan_to_num(offset_x / delta_x), 0.0, 1.0)
  cdf = (im1 + rel_offset + 0.5) / nx
  return cdf


def sorting_inverse(ys, cdf):
  ny = ys.shape[0]
  jdy = jnp.int32(jnp.floor(cdf * ny))
  jdy = jnp.clip(jdy, 0, ny - 1)
  jp1 = jnp.clip(jdy + 1, 0, ny - 1)
  a = ys[jdy] + (cdf - jdy / ny) * (ys[jp1] - ys[jdy])
  return a


def rq_spline_compute_shared(bin_widths, bin_heights, knot_slopes, x_or_y, range_min=0.0, is_x=True):
  """Captures shared computations across the rational quadratic spline forward/inverse."""
  assert bin_widths.ndim == bin_heights.ndim == knot_slopes.ndim == 1
  kx = jnp.concatenate([jnp.full((1,), range_min), jnp.cumsum(bin_widths, axis=-1) + range_min], axis=-1)
  ky = jnp.concatenate([jnp.full((1,), range_min), jnp.cumsum(bin_heights, axis=-1) + range_min], axis=-1)
  kd = jnp.concatenate([jnp.full((1,), 1.0), knot_slopes, jnp.full((1,), 1.0)], axis=-1)
  kx_or_ky = kx if is_x else ky
  kx_or_ky_min = kx_or_ky[0]
  kx_or_ky_max = kx_or_ky[-1]
  out_of_bounds = (x_or_y <= kx_or_ky_min) | (x_or_y >= kx_or_ky_max)
  x_or_y = jnp.where(out_of_bounds, kx_or_ky_min, x_or_y)
  idx = jnp.clip(jnp.searchsorted(kx_or_ky, x_or_y) - 1, 0, kx_or_ky.shape[0] - 2)
  x_k = kx[idx]
  x_kp1 = kx[idx + 1]
  y_k = ky[idx]
  y_kp1 = ky[idx + 1]
  d_k = kd[idx]
  d_kp1 = kd[idx + 1]
  h_k = y_kp1 - y_k
  w_k = x_kp1 - x_k
  s_k = h_k / w_k
  return out_of_bounds, x_k, y_k, d_k, d_kp1, h_k, w_k, s_k


def rq_spline_forward(bin_widths, bin_heights, knot_slopes, x, range_min=0.0):
  """Compute the rational quadratic spline forward transformation"""
  out_of_bounds, x_k, y_k, d_k, d_kp1, h_k, w_k, s_k = rq_spline_compute_shared(bin_widths, bin_heights, knot_slopes, x, range_min=range_min, is_x=True)
  relx = (x - x_k) / w_k
  spline_val = y_k + ((h_k * (s_k * relx**2 + d_k * relx * (1 - relx))) / (s_k + (d_kp1 + d_k - 2 * s_k) * relx * (1 - relx)))
  y_val = jnp.where(out_of_bounds, x, spline_val)
  return y_val


def rq_spline_inverse(bin_widths, bin_heights, knot_slopes, y, range_min=0.0):
  """Compute the rational quadratic spline inverse transformation"""
  out_of_bounds, x_k, y_k, d_k, d_kp1, h_k, w_k, s_k = rq_spline_compute_shared(bin_widths, bin_heights, knot_slopes, y, range_min=range_min, is_x=False)
  rely = jnp.where(out_of_bounds, 0.0, y - y_k)
  term2 = rely * (d_kp1 + d_k - 2 * s_k)
  # These terms are the a, b, c terms of the quadratic formula.
  a = h_k * (s_k - d_k) + term2
  b = h_k * d_k - term2
  c = -s_k * rely
  # The expression used here has better numerical behavior for small 4*a*c.
  relx = jnp.where((rely == 0.0), 0.0, (2 * c) / (-b - jnp.sqrt(b**2 - 4 * a * c)))
  return jnp.where(out_of_bounds, y, relx * w_k + x_k)


def layer(wkey, dkey, data_train, x_batched, x_offline, slicer_dict, dim, hdim, mask=None, step_size=1.0, forward="rqspline", inverse="rqspline", n_bins_particles=200, n_bins_data=200, dequantize=True, multi_devices=True, clip=None, fix_slopes=False):
  assert isinstance(slicer_dict, dict)
  ws = []
  for slicer, num in slicer_dict.items():
    wkey, subkey = jax.random.split(wkey)
    skeys = jax.random.split(subkey, num)
    wi = jax.vmap(slicer)(skeys)
    ws.append(jnp.reshape(wi, (-1, dim)))
    print(f"Slicer {slicer}: hdim = {ws[-1].shape[0]}")

  w = jnp.vstack(ws)
  print(f"Image slicer shape = {w.shape}")

  if w.shape[0] > hdim:
    wkey, subkey = jax.random.split(wkey)
    w = jax.random.choice(subkey, w, (hdim,), replace=False)
    print(f"[After subsampling] Slicer shape = {w.shape}")
  assert w.shape[0] == hdim

  # generate projections for labels
  if data_train.shape[0] > dim:
    n_labels = data_train.shape[0] - dim
    wkey, subkey = jax.random.split(wkey)
    w_labels = jax.random.laplace(subkey, shape=(hdim, n_labels))
    w_labels_norm = jnp.linalg.norm(w_labels, axis=1, keepdims=True)
    w_labels = w_labels / w_labels_norm
    print(f"Label slicer shape = {w_labels.shape}")
    w = jnp.concatenate([w * jnp.sqrt(dim / data_train.shape[0]), w_labels * jnp.sqrt(n_labels / data_train.shape[0])], axis=1)

  print(f"Final slicer shape = {w.shape}")

  # compute projection
  x_proj = jnp.matmul(w, x_batched)
  if dequantize:
    data_train = data_train + jax.random.uniform(dkey, data_train.shape) / 128.0
  data_proj = jnp.matmul(w, data_train)

  # prepare forward and inverse functions
  if forward == "rqspline":
    x_proj = jnp.sort(x_proj)
    x_min = x_proj[:, :1]
    x_max = x_proj[:, -1:]
    if multi_devices:
      x_min = jax.lax.pmin(x_min, axis_name="device")
      x_max = jax.lax.pmax(x_max, axis_name="device")
    x_proj = (x_proj - x_min) / (x_max - x_min)
    bin_edges_idx_x = jnp.int32(jnp.linspace(0.0, 1.0, num=n_bins_particles + 1)[1:-1] * x_proj.shape[-1])
    bin_edges_x = x_proj[:, bin_edges_idx_x]
    if multi_devices:
      bin_edges_x = jax.lax.pmean(bin_edges_x, axis_name="device")
    bin_edges_x = jnp.concatenate([jnp.full(bin_edges_x.shape[:-1] + (1,), 0.0), bin_edges_x, jnp.full(bin_edges_x.shape[:-1] + (1,), 1.0)], axis=-1)
    hist_x, _ = jax.vmap(functools.partial(jnp.histogram, range=(0.0, 1.0), density=False))(x_proj, bin_edges_x)
    hist_x = hist_x / x_proj.shape[1]
    if multi_devices:
      hist_x = jax.lax.pmean(hist_x, axis_name="device")
    bin_widths_x = bin_edges_x[:, 1:] - bin_edges_x[:, :-1]
    knot_slopes_x = (hist_x[:, :-1] + hist_x[:, 1:]) / (bin_widths_x[:, :-1] + bin_widths_x[:, 1:])
    if fix_slopes:
      knot_slopes_x = jnp.ones_like(knot_slopes_x)

  if forward == "sorting":
    if multi_devices:
      x_proj = jax.lax.all_gather(x_proj, axis_name="device", axis=1)
    x_proj = jnp.reshape(x_proj, (hdim, -1))
    x_proj = jnp.sort(x_proj)

  if inverse == "rqspline":
    data_proj = jnp.sort(data_proj)
    data_min = data_proj[:, :1]
    data_max = data_proj[:, -1:]
    if multi_devices:
      data_min = jax.lax.pmin(data_min, axis_name="device")
      data_max = jax.lax.pmax(data_max, axis_name="device")
    data_proj = (data_proj - data_min) / (data_max - data_min)
    bin_edges_idx_data = jnp.int32(jnp.linspace(0.0, 1.0, num=n_bins_data + 1)[1:-1] * data_proj.shape[-1])
    bin_edges_data = data_proj[:, bin_edges_idx_data]
    if multi_devices:
      bin_edges_data = jax.lax.pmean(bin_edges_data, axis_name="device")
    bin_edges_data = jnp.concatenate([jnp.full(bin_edges_data.shape[:-1] + (1,), 0.0), bin_edges_data, jnp.full(bin_edges_data.shape[:-1] + (1,), 1.0)], axis=-1)
    hist_data, _ = jax.vmap(functools.partial(jnp.histogram, range=(0.0, 1.0), density=False))(data_proj, bin_edges_data)
    hist_data = hist_data / data_proj.shape[1]
    if multi_devices:
      hist_data = jax.lax.pmean(hist_data, axis_name="device")
    bin_widths_data = bin_edges_data[:, 1:] - bin_edges_data[:, :-1]
    knot_slopes_data = (hist_data[:, :-1] + hist_data[:, 1:]) / (bin_widths_data[:, :-1] + bin_widths_data[:, 1:])
    if fix_slopes:
      knot_slopes_data = jnp.ones_like(knot_slopes_data)

  if inverse == "sorting":
    if multi_devices:
      data_proj = jax.lax.all_gather(data_proj, axis_name="device", axis=1)
    data_proj = jnp.reshape(data_proj, (hdim, -1))
    data_proj = jnp.sort(data_proj)

  # prepare unidimensional optimal transport functions
  if forward == "sorting" and inverse == "sorting":

    def unidim_transport(xs, ys, x):
      return sorting_inverse(ys, sorting_forward(xs, x))

  elif forward == "sorting" and inverse == "rqspline":

    def unidim_transport(xs, bin_widths_data, bin_heights_data, knot_slopes_data, x):
      return rq_spline_inverse(bin_widths_data, bin_heights_data, knot_slopes_data, sorting_forward(xs, x))

  elif forward == "rqspline" and inverse == "sorting":

    def unidim_transport(bin_widths_x, bin_heights_x, knot_slopes_x, ys, x):
      return sorting_inverse(ys, rq_spline_forward(bin_widths_x, bin_heights_x, knot_slopes_x, x))

  elif forward == "rqspline" and inverse == "rqspline":

    def unidim_transport(bin_widths_x, bin_heights_x, knot_slopes_x, bin_widths_data, bin_heights_data, knot_slopes_data, x):
      return rq_spline_inverse(bin_widths_data, bin_heights_data, knot_slopes_data, rq_spline_forward(bin_widths_x, bin_heights_x, knot_slopes_x, x))

  else:
    raise NotImplementedError(f"forward method {forward} or inverse method {inverse} unknown.")
  print(f"Forward method: {forward}, Inverse method: {inverse}")

  def transport(x):
    y = jnp.matmul(w, x)

    if forward == "sorting" and inverse == "sorting":
      a = jax.vmap(unidim_transport)(x_proj, data_proj, y)
    elif forward == "sorting" and inverse == "rqspline":
      a_normalized = jax.vmap(unidim_transport)(x_proj, bin_widths_data, hist_data, knot_slopes_data, y)
      a = a_normalized * (data_max - data_min) + data_min
    elif forward == "rqspline" and inverse == "sorting":
      y_normalized = (y - x_min) / (x_max - x_min)
      a = jax.vmap(unidim_transport)(bin_widths_x, hist_x, knot_slopes_x, data_proj, y_normalized)
    elif forward == "rqspline" and inverse == "rqspline":
      y_normalized = (y - x_min) / (x_max - x_min)
      a_normalized = jax.vmap(unidim_transport)(bin_widths_x, hist_x, knot_slopes_x, bin_widths_data, hist_data, knot_slopes_data, y_normalized)
      a = a_normalized * (data_max - data_min) + data_min

    movement = a - y
    delta_x = jnp.matmul(w.T, movement) * (step_size * dim / hdim)

    if mask is not None:
      z = x + delta_x * mask
    else:
      z = x + delta_x

    if clip is not None:
      print(f"Enabled data clipping = {clip}")
      z = jnp.clip(z, -clip, clip)
    ws_dist = jnp.mean(jnp.abs(movement))
    return z, ws_dist

  x_batched, ws_dist_batched = transport(x_batched)
  x_offline, ws_dist_offline = transport(x_offline)

  # a workaround to prevent copying all particles to device 0 for storage
  n_save_device = 50000 // jax.device_count()
  x_batched_to_save = x_batched[:, :n_save_device]
  return x_batched, x_offline, ws_dist_batched, ws_dist_offline, x_batched_to_save
