import numpy as np
import tensorflow as tf

from models.mixture import gaussian_mixture_loss_fn


def train(model, feeder, hparams, saver=None):

  loss_fn = gaussian_mixture_loss_fn(out_dim=hparams.dihedral_dim,
                                     num_mix=hparams.num_mixtures,
                                     use_tfp=hparams.use_tfp)

  optimizer = tf.optimizers.Adam(learning_rate=hparams.learning_rate)

  global_step = 0

  for epoch in range(hparams.epochs):

    avg_loss = []

    for (id_, primary, evolutionary, tertiary, angle, pri_mask, ter_mask, slen) in feeder.test:
      global_step += 1
      if hparams.use_evolutionary:
        inputs = [primary, pri_mask, angle, evolutionary]
      else:
        inputs = [primary, pri_mask, angle]

      with tf.GradientTape() as tape:
        y_hat = model(inputs)
        loss = loss_fn(angle, y_hat, ter_mask)
        avg_loss.append(loss)

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      if global_step % 100 == 0:
        print("Loss at global step %d: %f" % (global_step, loss))

    print("========Loss at epoch %d: %f" % (epoch, np.mean(avg_loss)))


    eval_loss = []
    for (id_, primary, evolutionary, tertiary, angle, pri_mask, ter_mask, slen) in feeder.test:
      if hparams.use_evolutionary:
        inputs = [primary, pri_mask, angle, evolutionary]
      else:
        inputs = [primary, pri_mask, angle]

      y_hat = model(inputs)
      loss = loss_fn(angle, y_hat, ter_mask)
      eval_loss.append(loss)

    print("~~~~~~~~Eval loss at epoch %d: %f" % (epoch, np.mean(eval_loss)))


