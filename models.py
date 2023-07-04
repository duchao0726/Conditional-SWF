import functools

import jax
import numpy as np

import layers
import slicers

nfs = 20


def downsample_kxk_dense_layer(layer, data_shape, k, hdim, step_size=1.0, method="lanczos3"):
  down_k_size = (k, k)
  dim_ratio = np.prod(down_k_size) / np.prod(data_shape[1:])
  down_k_slicer = functools.partial(
    slicers.downsample_slicer,
    slice_fn=slicers.uniform,
    input_shape=data_shape,
    down_size=down_k_size,
    hdim=hdim,
    method=method,
  )
  step_size = np.minimum(step_size, hdim / np.prod(down_k_size) / data_shape[0])
  step_size = step_size * dim_ratio
  down_k_layer = jax.pmap(functools.partial(layer, slicer_dict={down_k_slicer: 1}, step_size=step_size), axis_name="device", in_axes=(None, 0, 0, 0, 0))

  return down_k_layer


def downsample_kxk_conv_layer(layer, data_shape, k, hdim, hdim_per_conv, n_filters, kernel_sizes, strides=1, paddings="SAME", dilations=1, step_size=1.0, method="lanczos3"):
  down_k_size = (k, k)
  dim_ratio = np.prod(down_k_size) / np.prod(data_shape[1:])
  down_k_slicer = functools.partial(
    slicers.downsample_slicer,
    slice_fn=slicers.conv,
    input_shape=data_shape,
    down_size=down_k_size,
    hdim=hdim_per_conv,
    n_filters=n_filters,
    kernel_sizes=kernel_sizes,
    strides=strides,
    paddings=paddings,
    dilations=dilations,
    method=method,
  )
  step_size = np.minimum(step_size, hdim / np.prod(down_k_size) / data_shape[0])
  step_size = step_size * dim_ratio
  down_k_layer = jax.pmap(functools.partial(layer, slicer_dict={down_k_slicer: hdim // hdim_per_conv}, step_size=step_size), axis_name="device", in_axes=(None, 0, 0, 0, 0))

  return down_k_layer


def low_rez_dense_model(layer, data_shape, hdim, step_size, downsample_method="lanczos3", rezs=(1, 2, 3, 4, 5, 6), steps=(10, 100, 200, 300, 300, 300)):
  assert len(rezs) == len(steps)
  transform_layers, transform_steps = [], []
  for rez, step in zip(rezs, steps):
    dense_layer_ixi = downsample_kxk_dense_layer(layer=layer, data_shape=data_shape, k=rez, hdim=hdim, step_size=step_size, method=downsample_method)
    transform_layers.append(dense_layer_ixi)
    transform_steps.append(step)
  assert len(transform_layers) == len(transform_steps)
  return transform_layers, transform_steps


def downsample_kxk_model(layer, data_shape, k, hdim, hdim_per_conv, step_size, nfs, kss, sts, pds, dls, steps, min_convs=1, downsample_method="lanczos3", init_dense=False):
  assert len(nfs) == len(kss) == len(sts) == len(pds) == len(dls)
  assert len(steps) == len(nfs) + (1 if init_dense else 0)
  assert min_convs >= 1
  nl = len(nfs)
  transform_layers, transform_steps = [], []

  if init_dense:
    dense_layer_kxk = downsample_kxk_dense_layer(layer=layer, data_shape=data_shape, k=k, hdim=hdim, step_size=step_size, method=downsample_method)
    transform_layers.append(dense_layer_kxk)
    transform_steps.append(steps[0])
    steps = steps[1:]

  for i in range(nl):
    nf = nfs[i] if isinstance(nfs[i], (list, tuple)) else (nfs[i],) * (nl - i + min_convs - 1)
    ks = kss[i] if isinstance(kss[i], (list, tuple)) else (kss[i],) * (nl - i + min_convs - 1)
    st = sts[i] if isinstance(sts[i], (list, tuple)) else (sts[i],) * (nl - i + min_convs - 1)
    pd = pds[i] if isinstance(pds[i], (list, tuple)) else (pds[i],) * (nl - i + min_convs - 1)
    dl = dls[i] if isinstance(dls[i], (list, tuple)) else (dls[i],) * (nl - i + min_convs - 1)
    conv_i_layer_kxk = downsample_kxk_conv_layer(layer=layer, data_shape=data_shape, k=k, hdim=hdim, hdim_per_conv=hdim_per_conv, n_filters=nf, kernel_sizes=ks, strides=st, paddings=pd, dilations=dl, step_size=step_size, method=downsample_method)
    transform_layers.append(conv_i_layer_kxk)
    transform_steps.append(steps[i])

  assert len(transform_layers) == len(transform_steps)
  return transform_layers, transform_steps


def kxk_model(layer, data_shape, hdim, hdim_per_conv, step_size, nfs, kss, sts, pds, dls, steps, min_convs=1, init_dense=False):
  assert len(nfs) == len(kss) == len(sts) == len(pds) == len(dls)
  assert len(steps) == len(nfs) + (1 if init_dense else 0)
  assert min_convs >= 1
  nl = len(nfs)
  transform_layers, transform_steps = [], []

  dim = np.prod(data_shape)
  step_size = np.minimum(step_size, hdim / dim)

  if init_dense:
    dense_slicer = functools.partial(
      slicers.uniform,
      dim=dim,
      hdim=hdim,
    )
    dense_layer = jax.pmap(functools.partial(layer, slicer_dict={dense_slicer: 1}, step_size=step_size), axis_name="device", in_axes=(None, 0, 0, 0, 0))
    transform_layers.append(dense_layer)
    transform_steps.append(steps[0])
    steps = steps[1:]

  for i in range(nl):
    nf = nfs[i] if isinstance(nfs[i], (list, tuple)) else (nfs[i],) * (nl - i + min_convs - 1)
    ks = kss[i] if isinstance(kss[i], (list, tuple)) else (kss[i],) * (nl - i + min_convs - 1)
    st = sts[i] if isinstance(sts[i], (list, tuple)) else (sts[i],) * (nl - i + min_convs - 1)
    pd = pds[i] if isinstance(pds[i], (list, tuple)) else (pds[i],) * (nl - i + min_convs - 1)
    dl = dls[i] if isinstance(dls[i], (list, tuple)) else (dls[i],) * (nl - i + min_convs - 1)
    conv_i_slicer = functools.partial(
      slicers.conv,
      input_shape=data_shape,
      hdim=hdim_per_conv,
      n_filters=nf,
      kernel_sizes=ks,
      strides=st,
      paddings=pd,
      dilations=dl,
    )
    conv_i_layer = jax.pmap(functools.partial(layer, slicer_dict={conv_i_slicer: hdim // hdim_per_conv}, step_size=step_size), axis_name="device", in_axes=(None, 0, 0, 0, 0))
    transform_layers.append(conv_i_layer)
    transform_steps.append(steps[i])

  assert len(transform_layers) == len(transform_steps)
  return transform_layers, transform_steps


def swf_model(data_shape, mask, hdim, step_size, layer_steps=200, forward="rqspline", inverse="rqspline", n_bins_particles=200, n_bins_data=200, dequantize=True, **kwargs):
  dim = np.prod(data_shape)
  step_size = np.minimum(step_size, hdim / dim)

  layer = functools.partial(layers.layer, dim=dim, hdim=hdim, mask=mask, forward=forward, inverse=inverse, n_bins_particles=n_bins_particles, n_bins_data=n_bins_data, dequantize=dequantize)

  dense_slicer = functools.partial(
    slicers.uniform,
    dim=dim,
    hdim=hdim,
  )
  dense_layer = jax.pmap(functools.partial(layer, slicer_dict={dense_slicer: 1}, step_size=step_size), axis_name="device", in_axes=(None, 0, 0, 0, 0))

  transform_layers, transform_steps = [dense_layer], [layer_steps]

  return transform_layers, transform_steps


def mnist_model(data_shape, mask, hdim, hdim_per_conv, step_size, layer_steps=200, forward="rqspline", inverse="rqspline", n_bins_particles=200, n_bins_data=200, downsample_method="lanczos3", dequantize=True):
  dim = np.prod(data_shape)
  layer = functools.partial(layers.layer, dim=dim, hdim=hdim, mask=mask, forward=forward, inverse=inverse, n_bins_particles=n_bins_particles, n_bins_data=n_bins_data, dequantize=dequantize)

  transform_layers, transform_steps = [], []

  lowres_layers, lowres_steps = low_rez_dense_model(layer, data_shape, hdim, step_size, downsample_method, rezs=list(range(1, 7)), steps=[20] + [layer_steps] * 5)
  transform_layers.extend(lowres_layers)
  transform_steps.extend(lowres_steps)

  res_7x7_dl2_layers, res_7x7_dl2_steps = downsample_kxk_model(layer, data_shape, 7, hdim, hdim_per_conv, step_size, nfs=[nfs] * 2, kss=[3] * 2, sts=[1] * 2, pds=["SAME"] * 2, dls=[2] * 2, steps=[layer_steps] * 2, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_7x7_dl2_layers)
  transform_steps.extend(res_7x7_dl2_steps)

  res_7x7_layers, res_7x7_steps = downsample_kxk_model(layer, data_shape, 7, hdim, hdim_per_conv, step_size, nfs=[nfs] * 3, kss=[3] * 3, sts=[1] * 3, pds=["SAME"] * 3, dls=[1] * 3, steps=[layer_steps] * 4, min_convs=1, downsample_method=downsample_method, init_dense=True)
  transform_layers.extend(res_7x7_layers)
  transform_steps.extend(res_7x7_steps)

  res_11x11_dl2_layers, res_11x11_dl2_steps = downsample_kxk_model(layer, data_shape, 11, hdim, hdim_per_conv, step_size, nfs=[nfs] * 3, kss=[3] * 3, sts=[1] * 3, pds=["SAME"] * 3, dls=[2] * 3, steps=[layer_steps] * 3, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_11x11_dl2_layers)
  transform_steps.extend(res_11x11_dl2_steps)

  res_11x11_layers, res_11x11_steps = downsample_kxk_model(layer, data_shape, 11, hdim, hdim_per_conv, step_size, nfs=[nfs] * 5, kss=[3] * 5, sts=[1] * 5, pds=["SAME"] * 5, dls=[1] * 5, steps=[layer_steps] * 5, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_11x11_layers)
  transform_steps.extend(res_11x11_steps)

  res_14x14_dl2_layers, res_14x14_dl2_steps = downsample_kxk_model(layer, data_shape, 14, hdim, hdim_per_conv, step_size, nfs=[nfs] * 3, kss=[3] * 3, sts=[1] * 3, pds=["SAME"] * 3, dls=[2] * 3, steps=[layer_steps] * 3, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_14x14_dl2_layers)
  transform_steps.extend(res_14x14_dl2_steps)

  res_14x14_layers, res_14x14_steps = downsample_kxk_model(layer, data_shape, 14, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 7, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_14x14_layers)
  transform_steps.extend(res_14x14_steps)

  res_21x21_dl2_layers, res_21x21_dl2_steps = downsample_kxk_model(layer, data_shape, 21, hdim, hdim_per_conv, step_size, nfs=[nfs] * 4, kss=[3] * 4, sts=[1] * 4, pds=["SAME"] * 4, dls=[2] * 4, steps=[layer_steps] * 4, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_21x21_dl2_layers)
  transform_steps.extend(res_21x21_dl2_steps)

  res_21x21_layers, res_21x21_steps = downsample_kxk_model(layer, data_shape, 21, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 7, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_21x21_layers)
  transform_steps.extend(res_21x21_steps)

  res_28x28_layers, res_28x28_steps = kxk_model(layer, data_shape, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 7, min_convs=1, init_dense=False)
  transform_layers.extend(res_28x28_layers)
  transform_steps.extend(res_28x28_steps)

  return transform_layers, transform_steps


def cifar10_model(data_shape, mask, hdim, hdim_per_conv, step_size, layer_steps=300, forward="rqspline", inverse="rqspline", n_bins_particles=200, n_bins_data=200, downsample_method="lanczos3", dequantize=True):
  dim = np.prod(data_shape)
  layer = functools.partial(layers.layer, dim=dim, hdim=hdim, mask=mask, forward=forward, inverse=inverse, n_bins_particles=n_bins_particles, n_bins_data=n_bins_data, dequantize=dequantize)

  transform_layers, transform_steps = [], []

  lowres_layers, lowres_steps = low_rez_dense_model(layer, data_shape, hdim, step_size, downsample_method, rezs=list(range(1, 8)), steps=[20] + [layer_steps] * 6)
  transform_layers.extend(lowres_layers)
  transform_steps.extend(lowres_steps)

  res_8x8_dl2_layers, res_8x8_dl2_steps = downsample_kxk_model(layer, data_shape, 8, hdim, hdim_per_conv, step_size, nfs=[nfs] * 2, kss=[3] * 2, sts=[1] * 2, pds=["SAME"] * 2, dls=[2] * 2, steps=[layer_steps] * 2, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_8x8_dl2_layers)
  transform_steps.extend(res_8x8_dl2_steps)

  res_8x8_layers, res_8x8_steps = downsample_kxk_model(layer, data_shape, 8, hdim, hdim_per_conv, step_size, nfs=[nfs] * 3, kss=[3] * 3, sts=[1] * 3, pds=["SAME"] * 3, dls=[1] * 3, steps=[layer_steps] * 4, min_convs=1, downsample_method=downsample_method, init_dense=True)
  transform_layers.extend(res_8x8_layers)
  transform_steps.extend(res_8x8_steps)

  res_12x12_dl2_layers, res_12x12_dl2_steps = downsample_kxk_model(layer, data_shape, 12, hdim, hdim_per_conv, step_size, nfs=[nfs] * 3, kss=[3] * 3, sts=[1] * 3, pds=["SAME"] * 3, dls=[2] * 3, steps=[layer_steps] * 3, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_12x12_dl2_layers)
  transform_steps.extend(res_12x12_dl2_steps)

  res_12x12_layers, res_12x12_steps = downsample_kxk_model(layer, data_shape, 12, hdim, hdim_per_conv, step_size, nfs=[nfs] * 5, kss=[3] * 5, sts=[1] * 5, pds=["SAME"] * 5, dls=[1] * 5, steps=[layer_steps] * 5, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_12x12_layers)
  transform_steps.extend(res_12x12_steps)

  res_16x16_dl2_layers, res_16x16_dl2_steps = downsample_kxk_model(layer, data_shape, 16, hdim, hdim_per_conv, step_size, nfs=[nfs] * 3, kss=[3] * 3, sts=[1] * 3, pds=["SAME"] * 3, dls=[2] * 3, steps=[layer_steps] * 3, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_16x16_dl2_layers)
  transform_steps.extend(res_16x16_dl2_steps)

  res_16x16_layers, res_16x16_steps = downsample_kxk_model(layer, data_shape, 16, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 7, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_16x16_layers)
  transform_steps.extend(res_16x16_steps)

  res_24x24_layers, res_24x24_steps = downsample_kxk_model(layer, data_shape, 24, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 7, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_24x24_layers)
  transform_steps.extend(res_24x24_steps)

  res_32x32_layers, res_32x32_steps = kxk_model(layer, data_shape, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 5 + [1000] * 2, min_convs=1, init_dense=False)
  transform_layers.extend(res_32x32_layers)
  transform_steps.extend(res_32x32_steps)

  return transform_layers, transform_steps


def celeba_model(data_shape, mask, hdim, hdim_per_conv, step_size, layer_steps=300, forward="rqspline", inverse="rqspline", n_bins_particles=200, n_bins_data=200, downsample_method="lanczos3", dequantize=True):
  dim = np.prod(data_shape)
  layer = functools.partial(layers.layer, dim=dim, hdim=hdim, mask=mask, forward=forward, inverse=inverse, n_bins_particles=n_bins_particles, n_bins_data=n_bins_data, dequantize=dequantize, clip=1.0)

  transform_layers, transform_steps = [], []

  lowres_layers, lowres_steps = low_rez_dense_model(layer, data_shape, hdim, step_size, downsample_method, rezs=list(range(1, 8)), steps=[50] + [layer_steps] * 6)
  transform_layers.extend(lowres_layers)
  transform_steps.extend(lowres_steps)

  res_8x8_dl2_layers, res_8x8_dl2_steps = downsample_kxk_model(layer, data_shape, 8, hdim, hdim_per_conv, step_size, nfs=[nfs] * 2, kss=[3] * 2, sts=[1] * 2, pds=["SAME"] * 2, dls=[2] * 2, steps=[layer_steps] * 2, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_8x8_dl2_layers)
  transform_steps.extend(res_8x8_dl2_steps)

  res_8x8_layers, res_8x8_steps = downsample_kxk_model(layer, data_shape, 8, hdim, hdim_per_conv, step_size, nfs=[nfs] * 3, kss=[3] * 3, sts=[1] * 3, pds=["SAME"] * 3, dls=[1] * 3, steps=[layer_steps] * 4, min_convs=1, downsample_method=downsample_method, init_dense=True)
  transform_layers.extend(res_8x8_layers)
  transform_steps.extend(res_8x8_steps)

  res_12x12_dl2_layers, res_12x12_dl2_steps = downsample_kxk_model(layer, data_shape, 12, hdim, hdim_per_conv, step_size, nfs=[nfs] * 3, kss=[3] * 3, sts=[1] * 3, pds=["SAME"] * 3, dls=[2] * 3, steps=[layer_steps] * 3, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_12x12_dl2_layers)
  transform_steps.extend(res_12x12_dl2_steps)

  res_12x12_layers, res_12x12_steps = downsample_kxk_model(layer, data_shape, 12, hdim, hdim_per_conv, step_size, nfs=[nfs] * 5, kss=[3] * 5, sts=[1] * 5, pds=["SAME"] * 5, dls=[1] * 5, steps=[layer_steps] * 5, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_12x12_layers)
  transform_steps.extend(res_12x12_steps)

  res_16x16_dl2_layers, res_16x16_dl2_steps = downsample_kxk_model(layer, data_shape, 16, hdim, hdim_per_conv, step_size, nfs=[nfs] * 3, kss=[3] * 3, sts=[1] * 3, pds=["SAME"] * 3, dls=[2] * 3, steps=[layer_steps] * 3, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_16x16_dl2_layers)
  transform_steps.extend(res_16x16_dl2_steps)

  res_16x16_layers, res_16x16_steps = downsample_kxk_model(layer, data_shape, 16, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 7, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_16x16_layers)
  transform_steps.extend(res_16x16_steps)

  res_24x24_layers, res_24x24_steps = downsample_kxk_model(layer, data_shape, 24, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 7, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_24x24_layers)
  transform_steps.extend(res_24x24_steps)

  res_32x32_layers, res_32x32_steps = downsample_kxk_model(layer, data_shape, 32, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 7, min_convs=1, downsample_method=downsample_method, init_dense=False)
  transform_layers.extend(res_32x32_layers)
  transform_steps.extend(res_32x32_steps)

  res_64x64_layers, res_64x64_steps = kxk_model(layer, data_shape, hdim, hdim_per_conv, step_size, nfs=[nfs] * 7, kss=[3] * 7, sts=[1] * 7, pds=["SAME"] * 7, dls=[1] * 7, steps=[layer_steps] * 5 + [1000] * 2, min_convs=1, init_dense=False)
  transform_layers.extend(res_64x64_layers)
  transform_steps.extend(res_64x64_steps)

  return transform_layers, transform_steps
