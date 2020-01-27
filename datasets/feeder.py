import os
import tensorflow as tf


NUM_EVO_ENTRIES = 21
NUM_DIMENSIONS = 3
NUM_AMINO_ACIDS = 20

_context_feature_description = {'id': tf.io.FixedLenFeature((1,), tf.string)}
_sequence_feature_description = {
    'primary': tf.io.FixedLenSequenceFeature((1,), tf.int64, allow_missing=True),
    'evolutionary': tf.io.FixedLenSequenceFeature((NUM_EVO_ENTRIES,), tf.float32, allow_missing=True),
    'secondary': tf.io.FixedLenSequenceFeature((1,), tf.int64, allow_missing=True),
    'tertiary': tf.io.FixedLenSequenceFeature((NUM_DIMENSIONS,), tf.float32, allow_missing=True),
    'angular': tf.io.FixedLenSequenceFeature((NUM_DIMENSIONS,), tf.float32, allow_missing=True),
    'mask': tf.io.FixedLenSequenceFeature((1,), tf.float32, allow_missing=True)
}


class Feeder():
  """
    Loads minibatch of data from preprocessed TFRecords.
  """

  def __init__(self, datadir, hparams):

    self._max_sequence_length = hparams.max_sequence_length

    train_file = os.path.join(datadir, hparams.train_dir, 'processed.tfrecord')
    test_file = os.path.join(datadir, hparams.test_dir, 'processed.tfrecord')

    self.train = self._construct_dataloader(train_file, hparams.batch_size, hparams.shuffle_size)
    self.test = self._construct_dataloader(test_file, hparams.batch_size_test, hparams.shuffle_size)
    # self.val = ...


  def _construct_dataloader(self, fname, batch_size, shuffle_size):
    ### helper functions ###
    def parse_fn(sample):
      context, features = tf.io.parse_single_sequence_example(
          sample,
          context_features=_context_feature_description,
          sequence_features=_sequence_feature_description
      )

      id_ = context['id'][0]
      primary = tf.cast(features['primary'][:, 0], tf.int32)
      evolutionary = features['evolutionary']
      # secondary = tf.cast(features['secondary'][:, 0], tf.int32)
      tertiary = features['tertiary']
      angular = features['angular']
      ter_mask = features['mask'][:, 0]

      # NOTE:
      # 1. The above ter_mask from ProteinNet is for missing atomic coordinates of the tertiary
      #   structure, and it will be used to mask out the final loss,
      #   see https://github.com/aqlaboratory/proteinnet/blob/master/docs/proteinnet_records.md
      #   for details.
      # 2. Here we need another mask for the padded primary sequence to be used in rnn layers.
      #    Converting it to tf.bool due to a weird issue with tf.RNN layer which only takes boolean
      #    mask as input in Tensorflow's Windows (or maybe GPU) version.
      pri_length = tf.size(primary)
      pri_mask = tf.cast(tf.ones(pri_length), dtype=tf.bool)


      # Generate tertiary masking matrix. If mask is missing then assume all residues are present
      # mask = tf.cond(tf.not_equal(tf.size(mask), 0), lambda: mask, lambda: tf.ones([pri_length - num_edge_residues]))
      # ter_mask = masking_matrix(mask, name='ter_mask')

      return id_, primary, evolutionary, tertiary, angular, pri_mask, ter_mask, pri_length

    def filter_fn(*parsed):
      """ Predicate for filtering out protein longer than max length """
      return parsed[-1] <= self._max_sequence_length
    ### helper functions end ###

    padded_shapes = ([], [-1], [-1, NUM_EVO_ENTRIES], [-1, NUM_DIMENSIONS], [-1, NUM_DIMENSIONS], [-1], [-1], [])
    loader = tf.data.TFRecordDataset(fname)
    loader = loader.map(parse_fn).filter(filter_fn).shuffle(buffer_size=shuffle_size)
    loader = loader.padded_batch(batch_size, padded_shapes=padded_shapes)

    return loader
