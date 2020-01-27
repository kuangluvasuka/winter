import tensorflow as tf
from tensorflow.keras.layers import Layer


# TODO: modify all modules using lazy building?
# https://github.com/tensorflow/community/blob/master/rfcs/20190117-tf-module.md


class Embedding(Layer):
  def __init__(self, vocab_size, embedding_dim, initializer,
               std=0.1, trainable=True, name='embedding'):
    super().__init__(name=name)

    if initializer == 'identity':
      m = tf.eye(vocab_size, dtype=tf.float32)
    elif initializer == 'normal':
      m = tf.random.normal([vocab_size, embedding_dim], stddev=std, dtype=tf.float32)
    else:
      raise ValueError("Invalid 'init' argument: %s" % initializer)

    self._embedding_matrix = tf.Variable(m, trainable=trainable, name=name)

  def call(self, x):
    return tf.nn.embedding_lookup(self._embedding_matrix, x)


class Linear(Layer):
  def __init__(self, in_features, out_features, name='linear'):
    super().__init__(name=name)
    self.w = tf.Variable(tf.random.normal([in_features, out_features]), name=name + '_W')
    self.b = tf.Variable(tf.zeros(out_features), name=name + '_b')

  def call(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)


class Recurrent(Layer):
  def __init__(self, units, in_features, rnn_class='gru', bidirectional=False, name='recurrent'):
    """
    Args:
      - units: int, dimension of output vectors
      - in_features: int, dimension of input features
      - rnn_class: str, recurrent unit to use. {'vanilla', 'lstm', 'gru'}

    """
    super().__init__(name=name)

    self.layer = None
    r = rnn_class.lower()
    if r == 'vanilla':
      rnn_cls = tf.keras.layers.SimpleRNN
    elif r == 'gru':
      rnn_cls = tf.keras.layers.GRU
    elif r == 'lstm':
      rnn_cls = tf.keras.layers.LSTM
    else:
      raise ValueError("Invalid 'rnn_class' argument: %s" % rnn_class)

    if bidirectional:
      forward_cell = rnn_cls(units, return_sequences=True)
      backward_cell = rnn_cls(units, return_sequences=True, go_backwards=True)
      self.layer = tf.keras.layers.Bidirectional(forward_cell, backward_cell, input_shape=(None, in_features))

    else:
      self.layer = rnn_cls(units, return_sequences=True, input_shape=(None, in_features))

  def call(self, x, mask, evol=None):
    output = self.layer(x, mask=mask)

    return output



class RNade(Layer):
  """[1] Neural Autoregressive Distribution Estimator.
     https://arxiv.org/abs/1605.02226

     [2] RNADE: The real-valued neural autoregressive density-estimator.
     https://arxiv.org/abs/1306.0186
  """
  def __init__(self, hidden_dim, condition_dim=32, seq_length=500, dihedral_dim=3,
               num_mixtures=5, act=tf.nn.relu, name='real_nade'):
    """
    Args:
      - hidden_dim: Number of hidden units in NADE
      - condition_dim: Number of units from conditional vectors per time step
      - seq_length: Maximum length of input sequence (see hparams.json: max_sequence_length)
      - dihedral_dim: Number of dihedral angles per time step of input
      - num_mixtures: Number of mixing distributions
      - act: Activation unit for encoded hidden state
    """
    super().__init__(name=name)

    # TODO: concat or merge?
    self.concat = True
    if self.concat:
      input_dim = dihedral_dim + condition_dim
      output_dim = dihedral_dim
      # NADE encoder
      self.W_enc = tf.Variable(tf.random.normal([seq_length, input_dim, hidden_dim]), name='w_enc')
      self.b_enc = tf.Variable(tf.zeros([1, hidden_dim]), name='b_enc')

      # mixture decoder
      self.V_mu = tf.Variable(tf.random.normal([seq_length, hidden_dim, output_dim * num_mixtures]), name='v_mu')
      self.b_mu = tf.Variable(tf.zeros([seq_length, 1, output_dim * num_mixtures]), name='b_mu')
      self.V_sigma = tf.Variable(tf.random.normal([seq_length, hidden_dim, output_dim * num_mixtures]), name='v_sigma')
      self.b_sigma = tf.Variable(tf.zeros([seq_length, 1, output_dim * num_mixtures]), name='b_sigma')
      self.V_pi = tf.Variable(tf.random.normal([seq_length, hidden_dim, num_mixtures]), name='v_pi')
      self.b_pi = tf.Variable(tf.zeros([seq_length, 1, num_mixtures]), name='b_pi')

    self.act = act

  #def _init_net(self, ):
  #  """Self implementation of lazy building mechanism, or rewrite Layer's build() method."""
  #  pass

  def call(self, inputs, z, sampling=False):
    if sampling:
      assert (inputs is None), "No input dihedrals for sampling"
      return self._sample(z)
    else:
      return self._cal_prob(inputs, z)

  def _cal_prob(self, inputs, z):
    """
    Args:
      - inputs: Dihedral angles, shape=[B, L, num_dihedrals]
      - z: Conditional vectors, shape=[B, L, condition_dim]

    Returns:
      - logit_mean: The logit means of mixtures at each time steps, shape=[B, L, out_dim * num_mix]
      - logit_std: The logit stddevs of mixtures, shape=[B, L, out_dim * num_mix].
      - logit_pi: The logit probability of mixtures, shape=[B, L, num_mix]

      NOTE:
        - logit_mean is the mean of gaussians: mean = logit_mean. Prefixing with 'logit' just to
              keep the naming constant.
        - logit_std is not the final standard deviation of gaussians: stddev = tf.exp(logit_std)
        - logit_pi is the unnormalized log probability of mixtures, where pi = tf.softmax(logit_pi)
    """

    batch_size, time_steps, _ = tf.shape(inputs)
    if self.concat:
      x = tf.transpose(tf.concat([inputs, z], axis=2), perm=[1, 0, 2])
    # TODO: else
    else:
      x = tf.transpose(inputs, perm=[1, 0, 2])

    # init a_0
    a = tf.tile(self.b_enc, [batch_size, 1])
    logits = []

    for i in tf.range(time_steps):
      h_mu, h_sigma, h_pi = self._get_mixture_coeff(i, a)
      logits.append(tf.concat([h_mu, h_sigma, h_pi], axis=1))
      a = a + tf.matmul(x[i], self.W_enc[i])

    # back to batch-major
    logits = tf.transpose(tf.stack(logits), perm=[1, 0, 2])

    return logits

  def _sample(self, z):
    """
    Args:
      - z: [B, L, condition_dim]

    Returns:
      - xs: samples
    """

    batch_size, time_steps, _ = tf.shape(z)

    xs = []
    a = tf.tile(self.b_enc, [batch_size, 1])

    for i in tf.range(time_steps):
      h_mu, h_sigma, h_pi = self._get_mixture_coeff(i, a)
      s_i = ...

      xs.append(s_i)

    xs = tf.transpose(tf.stack(xs), perm=[1, 0, 2])
    return xs

  def _get_mixture_coeff(self, i, a_i):
    """ Compute
    Args:
      - i: integer, time step
      - a_i: hidden state prior to the activation layer, shape=[B, hidden]

    Returns:
      - h_mu: mean of gaussians, shape=[B, out_dim * num_mix]
      - h_sigma: stddev of gaussians, shape=[B, out_dim * num_mix]
      - h_pi: mixing fractions, shape=[B, num_mix]
    """

    h = self.act(a_i)
    h_mu = tf.matmul(h, self.V_mu[i]) + self.b_mu[i]
    h_sigma = tf.matmul(h, self.V_sigma[i]) + self.b_sigma[i]
    h_pi = tf.matmul(h, self.V_pi[i]) + self.b_pi[i]

    return h_mu, h_sigma, h_pi





