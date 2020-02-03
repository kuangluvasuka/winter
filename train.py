import time
import numpy as np
import tensorflow as tf

from models.mixture import gaussian_mixture_loss_fn


def construct_inputs(primary, primary_mask, evolutionary, angle):
  return {'primary': primary,
          'primary_mask': primary_mask,
          'evolutionary': evolutionary,
          'angle': angle}


def train(args, model, feeder, hparams):
  loss_fn = gaussian_mixture_loss_fn(out_dim=hparams.dihedral_dim,
                                     num_mix=hparams.num_mixtures,
                                     use_tfp=hparams.use_tfp)
  optimizer = tf.optimizers.Adam(learning_rate=hparams.learning_rate)

  #TODO: move ckpt to function
  ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), step=tf.Variable(1), model=model, optimizer=optimizer)
  manager = tf.train.CheckpointManager(ckpt, args.ckpt_dir, max_to_keep=3)
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  for epoch in range(int(ckpt.epoch), hparams.epochs + 1):
    ckpt.epoch.assign_add(1)
    avg_loss = []
    start = time.time()
    for (id_, primary, evolutionary, tertiary, angle, prim_mask, ter_mask, slen) in feeder.train:
      ckpt.step.assign_add(1)
      inputs = construct_inputs(primary, prim_mask, evolutionary, angle)
      with tf.GradientTape() as tape:
        y_hat = model(inputs)
        loss = loss_fn(angle, y_hat, ter_mask)
        avg_loss.append(loss)

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

    eval_loss = []
    for (id_, primary, evolutionary, tertiary, angle, prim_mask, ter_mask, slen) in feeder.test:
      inputs = construct_inputs(primary, prim_mask, evolutionary, angle)
      y_hat = model(inputs)
      loss = loss_fn(angle, y_hat, ter_mask)
      eval_loss.append(loss)

    print("Epoch: {} | train loss: {:.3f} | time: {:.2f}s | eval loss: {:.3f}".format(
        epoch, np.mean(avg_loss), time.time() - start, np.mean(eval_loss)))

    if epoch % args.ckpt_inteval == 0:
      save_path = manager.save()
      print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

