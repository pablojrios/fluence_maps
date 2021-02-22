import math
import os
import sys
import tensorflow as tf
import pydicom
import numpy as np
from PIL import Image
# import lodgepole.image_tools as lit
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

DIMENSIONS_FILENAME = 'dimensions.txt'
#===================================================  Dataset Utils  ===================================================


def _words_to_feature(words):
  """Returns a TF-Feature of UTF-8 encoded strings.

  :param words: a list of strings ex.: ["When", "shall", "we", "three", "meet", "again", "?"]
  :return: a TF-Feature.
  """
  return Feature(bytes_list=BytesList(value=[word.encode("utf-8")
                                             for word in words]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, (tuple, list)):
        values = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, (tuple, list)):
        values = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_to_tfexample(image_filename, image_data, image_format, height, width, gamma_value):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/filename': _bytes_feature(image_filename),
      'image/encoded': _bytes_feature(image_data),
      'image/format': _bytes_feature(image_format),
      'image/gamma_index': _float_feature(gamma_value),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
  }))


#=======================================  Conversion Utils  ===================================================
def build_dicom_filename(base, year, filename):
    return os.path.join(base, year, filename + '.dcm')


def _get_filenames_and_gamma_values(dicom_and_gamma_csv, dataset_dir, oversampling=False, sample=1.0, seed=12345):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

      dicom_and_gamma_csv:

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    df_dcm_in = pd.read_csv(os.path.join(dataset_dir, dicom_and_gamma_csv), sep=';', delimiter=None, header=0)
    # agrego una columna constante al dataframe para hacer más simple el armado del full file path de los archivos dicom
    df_dcm_in['base_dir'] = dataset_dir

    df_dcm_out = pd.DataFrame(columns=['dicom_full_filepath', 'gamma_index'])
    df_dcm_out['dicom_full_filepath'] = [build_dicom_filename(row[0], str(row[1]), row[2]) for row in df_dcm_in[['base_dir', 'año', 'fluencia calculada']].values]
    df_dcm_out['gamma_index'] = df_dcm_in.apply(lambda row: 100.0 - row['uno menos gamma index'], axis=1)
    # Pass-by-object-reference: As we know, in Python, “Object references are passed by value”, so
    # displayHistogramOfGammaValues(df_dcm_out)
    
    if 0.0 < sample < 1.0:
        df_dcm_out = df_dcm_out.sample(n=int(len(df_dcm_in)*sample), random_state=seed)
    
    return df_dcm_out


# list = [0]
# append(list)
# print(list) returns [0, 1]
def append(list):
    list.append(1)


# list = [0]
# reassign(list)
# print(list) returns [0]
def reassign(list):
    list = [0, 1]


def displayHistogramOfGammaValues(df_dcm_out):
    # df_dcm_out is reassigned so it doesn't affect caller
    df_dcm_out = df_dcm_out[df_dcm_out['gamma_index'] >= 95.0]
    bins_list = [95, 96, 97, 98, 98.25, 98.5, 98.75, 99, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.85, 99.9,
                 99.95, 100]
    sns.distplot(df_dcm_out['gamma_index'], bins=bins_list, kde=False);
    plt.xlim(95, 100)
    plt.title("Gamma Index")
    plt.savefig("gamma_index.png")
    plt.show()


def _get_dataset_filename(dataset_dir, split_name, shard_id, _NUM_SHARDS):
  output_filename = '%s-%05d-of-%05d.tfrecords' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def crop_square(im):
    width = im.size[0]
    height = im.size[1]
    left = (width - height) / 2
    top = 0
    right = width - left
    bottom = height
    return im.crop((left, top, right, bottom))


class Image:
    def __init__(self, filename):
        self.__filename = filename  # private attribute
        self.image_decoded = None
        self.image_raw = None

    @property
    def filename(self):
        return self.__filename

    @property
    def height(self):
        return self.image_decoded.shape[0]

    @property
    def width(self):
        return self.image_decoded.shape[1]

    def assert_shape(self):
        assert len(self.image_decoded.shape) == 3
        assert self.image_decoded.shape[2] == 3


class RGBImage(Image):
    def __init__(self, filename):
        super().__init__(filename)

        # Al convertir a grayscale, conviene guardar el 2D array decoded (no hace falta encoded/compressed a jpg),
        # pero nunca el 3D array decoded (canal grayscale x 3) ya que ocupa mucho más espacio en disco el TFRecordDataset.
        # Por ej.: mientras que el TFRecordDataset con los las imagenes jpg color para el dataset de cats-and-dogs ocupa
        # 552 MB, ¡el TFRecordDataset con los 3D array decoded/uncompressed ocupa 11 GB!. El 2D grayscale array ocupa entonces
        # 11 GB / 3 == 3.6 GB. Los archivos .jpg en disco que se copian al TFRecordDataset ocupan 596 MB.
        # Read the file as bytes: <class 'bytes'>
        self.image_raw = tf.io.gfile.GFile(filename, 'rb').read()
        # convert the compressed string to a 3D uint8 tensor: <class 'tensorflow.python.framework.ops.EagerTensor'> (213, 320, 3) <dtype: 'uint8'>
        self.image_decoded = tf.image.decode_jpeg(self.__image_raw, channels=3)


class GrayscaleImage(RGBImage):
    def __init__(self, filename):
        super().__init__(filename)

        # <class 'numpy.ndarray'> (213, 320, 3) uint8
        self.image_decoded = self.image_decoded.numpy()
        self.image_decoded = lit.rgb2gray_approx(self.image_decoded)
        # convert to 2D array of uint8 dtype
        self.image_decoded = self.image_decoded.astype(np.uint8)
        # convert to a 3D grayscale image (the same grayscale channel 3 times)
        self.image_decoded = np.repeat(self.image_decoded[:, :, np.newaxis], 3, axis=2)
        # convert to a 1D array of bytes
        self.image_raw = tf.io.encode_jpeg(self.image_decoded, format='rgb', quality=95, optimize_size=False,
                                      chroma_downsampling=False)


class DicomImage(Image):
    def __init__(self, filename):
        super().__init__(filename)

        ds = pydicom.dcmread(filename)
        shape = ds.pixel_array.shape  # ds.pixel_array.dtype is dtype('uint16')
        assert len(shape) == 2

        # Convert to float to avoid overflow or underflow losses.
        self.image_decoded = ds.pixel_array.astype('float64')

        # convert to Image so it's easier to crop to an square
        # im = Image.fromarray(image_2d)
        # im_cropped = crop_square(im)
        # image_2d = np.array(im)

        # Rescaling grey scale between 0-255 (0-32767 para JPEG 2000)
        self.image_decoded = (np.maximum(self.image_decoded, 0) / self.image_decoded.max()) * 255.0

        # Convert to uint8 (convert to uint16 para JPEG 2000)
        self.image_decoded = np.uint8(self.image_decoded)

        # Convert to array of bytes required from TF Features
        # TODO: probar sin encodear a jpeg
        # image_2d_scaled = image_2d_scaled.tobytes()

        # convert to a 3D grayscale image (the same grayscale channel 3 times)
        self.image_decoded = np.repeat(self.image_decoded[:, :, np.newaxis], 3, axis=2)
        # convert to a 1D array of bytes
        self.image_raw = tf.io.encode_jpeg(self.image_decoded, format='rgb', quality=95, optimize_size=False,
                                             chroma_downsampling=False)


class ImageType(Enum):
    RBG = 0
    Grayscale = 1
    Dicom = 2


def _convert_dataset(split_name, filenames, gamma_values, dataset_dir, _NUM_SHARDS, image_type = ImageType.RBG):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    fileids: A list of consecutive numbers starting from 1 to map to file names.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation', 'test']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  dims = set()

  for shard_id in range(_NUM_SHARDS):
    output_filename = _get_dataset_filename(
      dataset_dir, split_name, shard_id, _NUM_SHARDS)

    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_ndx = shard_id * num_per_shard
      end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
      for i in range(start_ndx, end_ndx):

        if image_type == ImageType.RBG.value:
            image = RGBImage(filenames[i])
        elif image_type == ImageType.Grayscale.value:
            image = GrayscaleImage(filenames[i])
        elif image_type == ImageType.Dicom.value:
            image = DicomImage(filenames[i])

        image.assert_shape()

        height, width = image.height, image.width
        dims.add((width, height))

        gamma_value = gamma_values[i]

        example = image_to_tfexample(filenames[i].encode('utf-8'), image.image_raw, b'jpg', height, width, gamma_value)
        tfrecord_writer.write(example.SerializeToString())

        sys.stdout.write('\r>> Image %d/%d with %d bytes in shard %d converted' % (
            i+1, len(filenames), len(image.image_raw.numpy()), shard_id))
        sys.stdout.flush()

  sys.stdout.write('\n')
  sys.stdout.flush()

  with open(os.path.join(dataset_dir, DIMENSIONS_FILENAME), 'w') as f:
    for d in dims:
      f.write("{0}x{1}\n".format(*d))


def _dataset_split_exists(dataset_dir, _NUM_SHARDS, split_name):
    for shard_id in range(_NUM_SHARDS):
        tfrecord_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, _NUM_SHARDS)
        if not tf.io.gfile.exists(tfrecord_filename):
            return False
    return True


def _dataset_exists(dataset_dir, _NUM_SHARDS):
  for split_name in ['train', 'validation','test']:
    for shard_id in range(_NUM_SHARDS):
      tfrecord_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id, _NUM_SHARDS)
      if not tf.io.gfile.exists(tfrecord_filename):
        return False
  return True

