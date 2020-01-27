import os
import glob
from tqdm import tqdm
import tensorflow as tf

from utils import cartesian_coords_to_dihedral_angles
from datasets.feeder import _context_feature_description, _sequence_feature_description


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()   # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _compose_new_example(id_, prim, evol, tert, angl, mask):
  """ Construst a new protein dict, which includes the calculated angular data. """

  context_dict = {'id': _bytes_feature(id_)}
  feature_dict = {
      'primary': tf.train.FeatureList(feature=[_int64_feature([aa]) for aa in prim]),
      'evolutionary': tf.train.FeatureList(feature=[_float_feature(evo) for evo in evol]),
      'tertiary': tf.train.FeatureList(feature=[_float_feature(coord) for coord in tert]),
      'angular': tf.train.FeatureList(feature=[_float_feature(ang) for ang in angl]),
      'mask': tf.train.FeatureList(feature=[_float_feature([mas]) for mas in mask])
  }

  return tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=feature_dict),
                                  context=tf.train.Features(feature=context_dict))


def _parse_func(example_proto):
  return tf.io.parse_single_sequence_example(
      example_proto,
      context_features=_context_feature_description,
      sequence_features=_sequence_feature_description)


def process_file(datadir, outfname='processed.tfrecord'):
  """ Calculate and add angular information into new TFRecord.

      NOTE: The original TFRecords in ProteinNet contains only tertiary structures as targets,
            but in many research projects we see that dihedral angles are more essential in
            protein structure prediction. So this function will add the missing angles and wrap
            them all in a new tfrecord.

      NOTE 2: The primary sequence in the original ProteinNet is an array of integers in
              range of [0, 19], each of which represents one of the 20 amino acids. Here we
              reset the integer to [1, 20], leaving '0' for later batch padding.
  """

  processed_file = os.path.join(datadir, outfname)
  path = os.path.join(datadir, '[0-9]*')
  raw_files = sorted(glob.glob(path), key=lambda x: int(os.path.split(x)[-1]))

  data_raw = tf.data.TFRecordDataset(raw_files)
  writer = tf.io.TFRecordWriter(processed_file)

  parsed_sequence = data_raw.map(_parse_func)

  # add dihedral angles and write to new tfrecord
  for i, (context, sequence) in tqdm(enumerate(parsed_sequence), desc='Progress'):
    id_ = context['id'].numpy()
    prim = sequence['primary'].numpy()
    prim = prim + 1
    evol = sequence['evolutionary'].numpy()
    tert = sequence['tertiary'].numpy()
    mask = sequence['mask'].numpy()
    angl = cartesian_coords_to_dihedral_angles(tert)

    new_seq = _compose_new_example(id_, prim, evol, tert, angl, mask)
    writer.write(new_seq.SerializeToString())


if __name__ == '__main__':
  datadir = 'data/casp7/training/30'
  process_file(datadir, outfname='processed.tfrecord')
