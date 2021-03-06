import tensorflow as tf

from models.module import Recurrent, Embedding, RNadeMoG, RNadeMoVM, RNadeMoIVM


class Pronet(tf.keras.Model):
  """

  """
  def __init__(self, hparams):
    super().__init__()

    self._hp = hp = hparams
    # use non-trainable, one-hot encoding
    # - decided not to use 'mask_zero' here, will pass 'mask' to rnn layer explicitly
    if hp.one_hot_embedding:
      self._embedding = Embedding(vocab_size=hp.NUM_AMINO_ACIDS + 1,
                                  embedding_dim=None,
                                  initializer='identity',
                                  trainable=False)
      in_features = hp.NUM_AMINO_ACIDS + 1
    # use trainable embedding
    else:
      self._embedding = Embedding(vocab_size=hp.NUM_AMINO_ACIDS + 1,
                                  embedding_dim=hp.embedding_dim,
                                  initializer='normal')
      in_features = hp.embedding_dim

    # Initializing encoder
    if hp.use_evolutionary:
      in_features += hp.NUM_EVO_ENTRIES
    self._recurrent = Recurrent(units=hp.rnn_units,
                                in_features=in_features,
                                rnn_class='gru',
                                bidirectional=hp.bidirectional)

    # Initializing decoder
    if hp.distribution == 'gaussian':
      nade = RNadeMoG
    elif hp.distribution == 'von_mises':
      nade = RNadeMoVM
    elif hp.distribution == 'independent_von_mises':
      nade = RNadeMoIVM
    else:
      raise ValueError("hparam.distribution has 3 options: 'gaussian' | 'von_mises' | 'independent_von_mises'.")

    if hp.bidirectional:
      cond_dim = 2 * hp.rnn_units
    else:
      cond_dim = hp.rnn_units
    self._rnade = nade(hidden_dim=hp.autoregressive_unit,
                       condition_dim=cond_dim,
                       seq_length=hp.max_sequence_length,
                       dihedral_dim=hp.dihedral_dim,
                       num_mixtures=hp.num_mixtures,
                       use_tfp=hp.use_tfp,
                       burn_in=hp.burn_in,
                       avg_count=hp.average_count)

  def reset_sample_function(self, fn):
    self._rnade.reset_sample_fn(fn)

  def call(self, inputs, is_sampling=False):
    """
    Args:
      - inputs: A dict containing
        1) prim_seq: primary sequence
        2) prim_mask: mask for primary sequence
        3) evol: evolutionary sequence, shape=[B, L, NUM_EVO_ENTRIES=21]
        4) angle (None if is_sampling=True): dihedral angles, shape=[B, L, dihedral_dim]
        NOTE: there are additional entries in the dict but unused here

    Returns:
      - y_hats: logits
    """
    emb = self._embedding(inputs['primary'])
    if self._hp.use_evolutionary:
      emb = tf.concat([emb, inputs['evolutionary']], axis=-1)
    latent_z = self._recurrent(emb, inputs['primary_mask'])
    y_hats = self._rnade(inputs['angle'], latent_z, is_sampling)

    return y_hats
