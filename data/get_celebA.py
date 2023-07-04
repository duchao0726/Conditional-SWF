import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  # h = tf.round(h * ratio, tf.int32)
  # w = tf.round(w * ratio, tf.int32)
  h = int(h * ratio)
  w = int(w * ratio)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_celeba_dataset(uniform_dequantization=False):
  dataset_builder = tfds.builder("celeb_a")
  train_split_name = "train"
  eval_split_name = "validation"

  def resize_op(img):
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = central_crop(img, 140)
    img = resize_small(img, 64)
    return img

  def preprocess_fn(d):
    """Basic preprocessing function scales data to [0, 1) and randomly flips."""
    img = resize_op(d["image"])
    if uniform_dequantization:
      img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0) / 256.0

    # return dict(image=img, label=d.get('label', None))
    return dict(image=img, label=d.get("label", 0))

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    # ds = ds.repeat(count=num_epochs)
    # ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds = ds.batch(batch_size, drop_remainder=True)
    # return ds.prefetch(prefetch_size)
    return ds

  train_ds = create_dataset(dataset_builder, train_split_name)
  eval_ds = create_dataset(dataset_builder, eval_split_name)
  return train_ds, eval_ds, dataset_builder


train_ds, eval_ds, dataset_builder = get_celeba_dataset()

train_ds_numpy = train_ds.as_numpy_iterator()
train_list = []
for ex in train_ds_numpy:
  train_list.append(np.moveaxis(ex["image"], -1, 0))
xtrain = np.stack(train_list)

test_ds_numpy = eval_ds.as_numpy_iterator()
test_list = []
for ex in test_ds_numpy:
  test_list.append(np.moveaxis(ex["image"], -1, 0))
xtest = np.stack(test_list)

with open("celeba_train.npy", "wb") as f:
  np.save(f, xtrain.reshape(xtrain.shape[0], -1))
with open("celeba_eval.npy", "wb") as f:
  np.save(f, xtest.reshape(xtest.shape[0], -1))
