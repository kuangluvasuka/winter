from math import pi as PI
import tensorflow as tf
from tensorflow_probability import distributions as tfd


def gaussian_mixture_loss_fn(out_dim, num_mix, use_tfp=False, reduce=True, log_scale_min_gauss=-7.0):
  """
  Args:
    - out_dim: 3
    - num_mix: Number of gaussians
    - use_cdf:
    - log_scale_min_gauss:
    - reduce:
  """
  LOGTWOPI = tf.constant(tf.math.log(2.0 * PI), dtype=tf.float32, name='log_two_pi')

  def mixture_loss(y_true, logit, mask):
    """
    Args:
      - y_true: [B, L, out_dim]
      - logit: [B, L, 2 * out_dim * num_mix + num_mix]
      - mask: [B, L]

    Returns:
      - loss:
    """
    batch_size, time_step, _ = tf.shape(y_true)
    mean, logit_std, logit_pi = tf.split(logit, num_or_size_splits=[out_dim * num_mix, out_dim * num_mix, num_mix],
                                         axis=-1, name='mix_gaussian_coeff_split')
    # mean, std = [B, L, out_dim * num_mix]; pi = [B, L, num_mix]

    mask = tf.reshape(mask, [-1])                                               # [B*L]
    mean = tf.reshape(mean, [-1, num_mix, out_dim])                             # [B*L, num_mix, out_dim]
    logit_std = tf.reshape(
        tf.maximum(logit_std, log_scale_min_gauss), [-1, num_mix, out_dim])     # [B*L, num_mix, out_dim]
    logit_pi = tf.reshape(logit_pi, [-1, num_mix])                              # [B*L, num_mix]

    if use_tfp:
      y_true = tf.reshape(y_true, [-1, out_dim])
      means = tf.unstack(mean, axis=1)
      logit_stds = tf.unstack(logit_std, axis=1)

      mixture = tfd.Mixture(
          cat=tfd.Categorical(logits=logit_pi),
          components=[tfd.MultivariateNormalDiag(
              loc=loc,
              scale_diag=tf.math.softplus(scale)) for loc, scale in zip(means, logit_stds)])

      loss = -mixture.log_prob(y_true)
    else:
      y_true = tf.reshape(y_true, [-1, 1, out_dim])                             # [B*L, 1, out_dim]

      # get log-probabilities of multivariate gaussian with identity covariance
      #  shape=[B*L, num_mix]
      # NOTE:
      # 1. The link below shows log-likelihood function of multivariate gaussian:
      #    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
      #    Here, we distributed the term k*ln(2PI) to each of the k dimensions inside the
      #    tf.reduce_sum, thus each dimension should only be added with LOGTWOPI in stead of K*LOGTWOPI
      #
      # 2. Use softplus() to rescale logit_std so as to keep it the same as the above tfp branch,
      #    this change results in smaller loss, but can prevent overflow in the sampling function.
      std = tf.math.softplus(logit_std)
      log_scale = tf.math.log(std)

      log_probs = -0.5 * tf.reduce_sum(
          #LOGTWOPI + 2 * logit_std + tf.math.square(y_true - mean) * tf.math.exp(-2 * logit_std),
          LOGTWOPI + 2 * log_scale + tf.math.square(y_true - mean) * tf.math.exp(-2 * log_scale),
          axis=-1)
      # get weighted of log-probability by summing mixed fraction, pi, in log-space
      mixed_log_probs = log_probs + tf.nn.log_softmax(logit_pi, axis=-1)        # [B*L, num_mix]
      tf.summary.histogram('mixed_log_probs', mixed_log_probs)
      # get negative log-probability
      loss = -tf.reduce_logsumexp(mixed_log_probs, axis=-1)                     # [B*L]

    # mask out loss of missing entries in atom coordinates
    loss = tf.multiply(loss, mask, name='masking')
    if reduce:
      return tf.reduce_sum(loss)
    else:
      return tf.reshape(loss, [batch_size, time_step])

  with tf.name_scope('gaussian_mixture_loss'):
    return mixture_loss


def gaussian_mixture_sample_fn(out_dim, num_mix, use_tfp=False, log_scale_min_gauss=-7.0):
  """
  https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb
  """
  def mixture_sampling(logit):
    """
    Args:
      - logit: [B, 2 * out_dim * num_mix + num_mix]

    Returns:
      - sample: [B, out_dim]
    """
    mean, logit_std, logit_pi = tf.split(logit, num_or_size_splits=[out_dim * num_mix, out_dim * num_mix, num_mix],
                                         axis=-1, name='mix_gaussian_coeff_split_sampling')
    mean = tf.reshape(mean, [-1, num_mix, out_dim])
    logit_std = tf.reshape(tf.maximum(logit_std, log_scale_min_gauss), [-1, num_mix, out_dim])
    logit_pi = tf.reshape(logit_pi, [-1, num_mix])
    #tf.print(mean)
    #tf.print(tf.shape(logit_std))
    #tf.print(logit_pi)

    #use_tfp = True
    if use_tfp:
      means = tf.unstack(mean, axis=1)
      logit_stds = tf.unstack(logit_std, axis=1)

      mixture = tfd.Mixture(
          cat=tfd.Categorical(logits=logit_pi),
          components=[tfd.MultivariateNormalDiag(
              loc=loc,
              scale_diag=tf.math.softplus(scale)) for loc, scale in zip(means, logit_stds)])

      #tf.print(mixture.mean())
      sample = mixture.sample()
    else:
      # sample mixture distribution from softmax-ed pi
      # see https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
      batch_size, _ = tf.shape(logit_pi)
      u = tf.random.uniform(tf.shape(logit_pi), minval=1e-5, maxval=1. - 1e-5)
      argmax = tf.argmax(logit_pi - tf.math.log(-tf.math.log(u)), axis=-1)
      onehot = tf.expand_dims(tf.one_hot(argmax, depth=num_mix, dtype=tf.float32), axis=-1)   # [B, num_mix, 1]
      # sample from selected gaussian
      # NOTE: we use softplus() to resacale the logit_std since exp() causes explosion of std values
      u = tf.random.uniform([batch_size, out_dim], minval=1e-5, maxval=1. - 1e-5)
      mean = tf.reduce_sum(tf.multiply(mean, onehot), axis=1)                                 # [B, out_dim]

      # TODO: cap std like np.maximum(logit_std, 1)
      logit_std = tf.reduce_sum(tf.multiply(logit_std, onehot), axis=1)

      sample = mean + tf.math.softplus(logit_std) * u

    # clip sample to [-pi, pi]?
    return sample

  with tf.name_scope('gaussian_mixture_sample'):
    return mixture_sampling


def vonmises_mixture_loss_fn(out_dim, num_mix, use_tfp=False, reduce=True, log_scale_min_gauss=-7.0):
  """ multivariate von Mises mixture distribution. """
  LOGTWOPI = tf.constant(tf.math.log(2.0 * PI), dtype=tf.float32, name='log_two_pi')

  def mixture_loss(y_true, logit, mask):
    """
    Args:
      - y_true: [B, L, out_dim]
      - logit: [B, L, 3 * out_dim * num_mix + num_mix]
      - mask: [B, L]

    Return:
      - loss
    """
    batch_size, time_step, _ = tf.shape(y_true)
    mean, logit_kappa, logit_lambda, logit_pi = tf.split(
        logit, num_or_size_splits=[out_dim * num_mix, out_dim * num_mix, out_dim * num_mix, num_mix],
        axis=-1, name='mix_vm_coeff_split')

    y_true = tf.reshape(y_true, [-1, 1, out_dim])                               # [B*L, 1, out_dim]
    mask = tf.reshape(mask, [-1])                                               # [B*L]
    mean = tf.reshape(mean, [-1, num_mix, out_dim])                             # [B*L, num_mix, out_dim]
    logit_kappa = tf.reshape(logit_kappa, [-1, num_mix, out_dim])               # [B*L, num_mix, out_dim]
    logit_lambda = tf.reshape(logit_lambda, [-1, num_mix, out_dim])
    logit_pi = tf.reshape(logit_pi, [-1, num_mix])                              # [B*L, num_mix]

    # rescale parameters
    kappa = tf.math.softplus(logit_kappa)
    # lambda_ = tf.math.softplus(logit_lambda)
    lambda_ = logit_lambda

    sin_diff = tf.sin(y_true - mean)                                            # [B*L, num_mix, out_dim]
    cos_diff = tf.cos(y_true - mean)

    # NOTE: indexing the last dimension using 0:1 and the like only to maintain the resultant
    #       matrix with shape=[B*l, num_mix, 1]
    theta_0 = lambda_[:, :, 0:1] * sin_diff[:, :, 1:2] + lambda_[:, :, 1:2] * sin_diff[:, :, 2:]
    theta_1 = lambda_[:, :, 0:1] * sin_diff[:, :, 0:1] + lambda_[:, :, 2:] * sin_diff[:, :, 2:]
    theta_2 = lambda_[:, :, 1:2] * sin_diff[:, :, 0:1] + lambda_[:, :, 2:] * sin_diff[:, :, 1:2]
    theta = tf.concat([theta_0, theta_1, theta_2], axis=-1)                     # [B*L, num_mix, out_dim]

    k_neg = tf.sqrt(kappa * kappa + theta * theta)
    log_probs = tf.reduce_sum(
        #-LOGTWOPI - tf.math.log(tf.math.bessel_i0(k_neg)) + cos_diff * kappa + sin_diff * theta,
        -LOGTWOPI - (tf.math.log(tf.math.bessel_i0e(k_neg)) + k_neg) + cos_diff * kappa + sin_diff * theta,
        axis=-1)

    mixed_log_probs = log_probs + tf.nn.log_softmax(logit_pi, axis=-1)          # [B*L, num_mix]
    loss = -tf.reduce_logsumexp(mixed_log_probs, axis=-1)                       # [B*L]
    loss = tf.multiply(loss, mask, name='masking')

    if reduce:
      return tf.reduce_sum(loss)
    else:
      return tf.reshape(loss, [batch_size, time_step])

  with tf.name_scope('von_mises_mixture_loss'):
    return mixture_loss


def masked_angular_mean_absolute_error(y_true, y_pred, mask, reduce=False):
  """
  Args:
    - y_true: Dihedral angle triplet (omega, phi, psi), [B, L, out_dim=3]
    - y_pred: [B, L, out_dim]
    - mask: [B, L]
    - reduce: bool, whether to sum the MAE over the angle triplet

  Returns:
    - MAE: mean absolute error of angles, [out_dim] (or scalar if reduce=True)
  """

  # NOTE: currently this function produces valid MAE only if y_pred is in range of [-pi, pi],
  #     otherwise the output value is meaningless. This usually happens on mixture gaussian models
  #     at the early training steps. If using von Mises mixture, the issue may be resolved.

  dist = tf.math.abs(y_pred - y_true)
  shifted = tf.math.abs(2 * PI - (dist))
  ae = tf.math.minimum(dist, shifted)                           # [B, L, out_dim]
  ae_masked = tf.multiply(ae, tf.expand_dims(mask, axis=-1))    # [B, L, out_dim]
  mae_per_amino_acid = tf.math.divide_no_nan(
      tf.reduce_sum(ae_masked, axis=1), tf.reduce_sum(mask, axis=-1, keepdims=True))  # [B, out_dim]
  mae = tf.reduce_mean(mae_per_amino_acid, axis=0)
  if reduce:
    return tf.reduce_sum(mae)
  return mae
