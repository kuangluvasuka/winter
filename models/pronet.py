import tensorflow as tf

from models.module import Recurrent, Embedding, RNade


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

    # Initializing RNN layer
    if hp.use_evolutionary:
      in_features += hp.NUM_EVOL
    self._recurrent = Recurrent(units=hp.rnn_units,
                                in_features=in_features,
                                rnn_class='gru',
                                bidirectional=hp.bidirectional)

    # Initializing RNADE layer
    self._rnade = RNade(hidden_dim=hp.autoregressive_unit,
                        condition_dim=hp.rnn_units,
                        seq_length=hp.max_sequence_length,
                        dihedral_dim=hp.dihedral_dim,
                        num_mixtures=hp.num_mixtures)


  def call(self, inputs, sampling=False):
    """
    Args:
      - prim_seq: primary sequence
      - pri_mask: mask for primary sequence
      - angle: dihedral angles, shape=[B, L, dihedral_dim]
    """

    prim_seq, pri_mask, angles = inputs[:3]
    emb = self._embedding(prim_seq)
    if self._hp.use_evolutionary:
      evol = inputs[3]
      emb = tf.concat([emb, evol], axis=-1)
    latent_z = self._recurrent(emb, pri_mask)
    y_hats = self._rnade(angles, latent_z)

    return y_hats

  def sample(self, ):
    pass

