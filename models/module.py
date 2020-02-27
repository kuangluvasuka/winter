import tensorflow as tf
from tensorflow.keras.layers import Layer
from models.mixture import gaussian_mixture_sample_fn, vonmises_mixture_sample_fn


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


class Addressable_Linear(Layer):
  """
    Linear layer with shape=[a,b,c] where the first dimension can be indexed in computation.
  """
  # TODO:
  #def _init_net(self, ):
  #  """Self implementation of lazy building mechanism, or rewrite Layer's build() method."""
  #  pass

  def __init__(self, seq_length, in_features, out_features, act=lambda x: x, name='linear'):
    super().__init__(name=name)
    self.w = tf.Variable(tf.random.normal([seq_length, in_features, out_features]), name='w_' + name)
    self.b = tf.Variable(tf.zeros([seq_length, 1, out_features]), name='b_' + name)
    self._act = act

  def __getitem__(self, idx):
    return (self.w[idx], self.b[idx])

  def call(self, x, idx):
    y = tf.matmul(x, self.w[idx]) + self.b[idx]
    return self._act(y)


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
    return self.layer(x, mask=mask)


class RNadeBase(Layer):
  """[1] Neural Autoregressive Distribution Estimator.
     https://arxiv.org/abs/1605.02226

     [2] RNADE: The real-valued neural autoregressive density-estimator.
     https://arxiv.org/abs/1306.0186
  """

  def __init__(self, name='real_nade'):
    super().__init__(name=name)

    self.W_enc = None
    self.b_enc = None
    self._act = None
    self._sample_fn = None
    self.rescaling = None
    #self.concat = None

  def call(self, x, z, is_sampling=False):
    if is_sampling:
      return self._sample(z)
    return self._cal_prob(x, z)

  def _cal_prob(self, x, z):
    """
    Args:
      - x: Dihedral angles, shape=[B, L, num_dihedrals]
      - z: Conditional vectors, shape=[B, L, condition_dim]

    Returns:
      - logits: Parameterized coefficients of the mixture model, shape=[B, L, model dependent dim]
    """
    batch_size, time_steps, _ = tf.shape(x)

    # conditional nade
    # TODO: is there any other ways to combine x and z?
    if self.concat:
      x = tf.transpose(tf.concat([x, z], axis=-1), perm=[1, 0, 2])
    else:
      x = tf.transpose(x, perm=[1, 0, 2])

    # init a_0
    a = tf.tile(self.b_enc, [batch_size, 1])
    logits = []

    for i in tf.range(time_steps):
      mixture_coeff = self._get_mixture_coeff(i, a)
      logits.append(mixture_coeff)
      a = self.rescaling[i+1] * (a / self.rescaling[i] + tf.matmul(x[i], self.W_enc[i]))

    # back to batch-major
    logits = tf.transpose(tf.stack(logits), perm=[1, 0, 2])
    return logits

  def _sample(self, z):
    """
    Args:
      - z: [B, L, condition_dim]

    Returns:
      - samples: [B, L, num_dihedrals]
    """
    batch_size, time_steps, _ = tf.shape(z)
    z = tf.transpose(z, perm=[1, 0, 2])
    samples = []
    a = tf.tile(self.b_enc, [batch_size, 1])

    for i in tf.range(time_steps):
      mixture_coeff = self._get_mixture_coeff(i, a)
      s = self._sample_fn(mixture_coeff)                # [B, 3]

      # conditional nade
      # TODO: is there any other ways to combine s and z?
      if self.concat:
        s_z = tf.concat([s, z[i]], axis=-1)
      else:
        pass

      a = self.rescaling[i+1] * (a / self.rescaling[i] + tf.matmul(s_z, self.W_enc[i]))
      samples.append(s)

    samples = tf.transpose(tf.stack(samples), perm=[1, 0, 2])
    return samples

  def _get_mixture_coeff(self, i, a_i):
    raise NotImplementedError("Need implementation of _get_mixture_coeff().")


class RNadeMoG(RNadeBase):
  """Real Nade for mixture of multivariate Gaussian model."""

  def __init__(self, hidden_dim, condition_dim=32, seq_length=500, output_dim=3,
               num_mixtures=5, act=tf.nn.relu, use_tfp=False, name='RNadeMoG'):
    """
    Args:
      - hidden_dim: Number of hidden units
      - condition_dim: Number of units from conditional vectors per time step
      - seq_length: Maximum length of input sequence (see hparams.json: max_sequence_length)
      - output_dim: Dimension of output per time step, same as dihedral_dim=3
      - num_mixtures: Number of mixing distributions
      - act: Activation unit for encoded hidden state
    """
    super().__init__(name)

    self.concat = True
    if self.concat:
      input_dim = output_dim + condition_dim
      # NADE encoder
      self.W_enc = tf.Variable(tf.random.normal([seq_length, input_dim, hidden_dim]), name='w_enc')
      self.b_enc = tf.Variable(tf.zeros([1, hidden_dim]), name='b_enc')

      # mixture decoder
      self.linear_mean = Addressable_Linear(seq_length, hidden_dim, output_dim * num_mixtures, name='linear_mean')
      self.linear_sigma = Addressable_Linear(seq_length, hidden_dim, output_dim * num_mixtures, name='linear_sigma')
      self.linear_pi = Addressable_Linear(seq_length, hidden_dim, num_mixtures, name='linear_pi')

    self.rescaling = tf.Variable(
        [1 / float(i) if i > 0 else 1.0 for i in range(seq_length + 1)], trainable=False, name='rescaling_factor')
    self._sample_fn = gaussian_mixture_sample_fn(out_dim=output_dim, num_mix=num_mixtures, use_tfp=use_tfp)
    self._act = act

  def _get_mixture_coeff(self, i, a_i):
    """ Compute mixture coefficients for mixture model.
    Args:
      - i: integer, time step
      - a_i: hidden state prior to the activation layer, shape=[B, hidden]

    Returns:
      parameterized coefficients for mixture of gaussian containing the following:
        - h_mean: The means of mixtures at each time steps, shape=[B, out_dim * num_mix]
        - h_std: The stddevs of mixtures, shape=[B, out_dim * num_mix].
        - h_pi: The probability of mixtures, shape=[B, num_mix]

      NOTE:
        - h_mean is the mean of gaussians: mean = h_mean
        - h_std is not the final standard deviation of gaussians: stddev = tf.exp(h_std)
              or stddev = tf.math.softplus(h_std)
        - h_pi is the unnormalized log probability of mixtures, where pi = tf.softmax(h_pi)
    """
    h = self._act(a_i)
    h_mean = self.linear_mean(h, i)
    h_sigma = self.linear_sigma(h, i)
    h_pi = self.linear_pi(h, i)
    return tf.concat([h_mean, h_sigma, h_pi], axis=1)


class RNadeMoVM(RNadeBase):
  """Real Nade for mixture of multivariate Von Mises model."""

  def __init__(self, hidden_dim, condition_dim=32, seq_length=500, output_dim=3,
               num_mixtures=5, act=tf.nn.relu, use_tfp=False, name='RNadeMoVM'):

    super().__init__(name)

    self.concat = True
    if self.concat:
      input_dim = output_dim + condition_dim
      # NADE encoder
      self.W_enc = tf.Variable(tf.random.normal([seq_length, input_dim, hidden_dim]), name='w_enc')
      self.b_enc = tf.Variable(tf.zeros([1, hidden_dim]), name='b_enc')

      # mixture decoder
      self.linear_mean = Addressable_Linear(seq_length, hidden_dim, output_dim * num_mixtures, name='linear_mean')
      self.linear_kappa = Addressable_Linear(seq_length, hidden_dim, output_dim * num_mixtures, name='linear_kappa')
      self.linear_lambda = Addressable_Linear(seq_length, hidden_dim, output_dim * num_mixtures, name='linear_lambda')
      self.linear_pi = Addressable_Linear(seq_length, hidden_dim, num_mixtures, name='linear_pi')

    self.rescaling = tf.Variable(
        [1 / float(i) if i > 0 else 1.0 for i in range(seq_length + 1)], trainable=False, name='rescaling_factor')
    self._sample_fn = vonmises_mixture_sample_fn(out_dim=output_dim, num_mix=num_mixtures, use_tfp=use_tfp)
    self._act = act

  def _get_mixture_coeff(self, i, a_i):
    h = self._act(a_i)
    h_mean = self.linear_mean(h, i)
    h_kappa = self.linear_kappa(h, i)
    h_lambda = self.linear_lambda(h, i)
    h_pi = self.linear_pi(h, i)

    return tf.concat([h_mean, h_kappa, h_lambda, h_pi], axis=1)
