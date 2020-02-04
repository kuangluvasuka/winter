import time
import numpy as np
import tensorflow as tf

from models.mixture import gaussian_mixture_loss_fn


def init_checkpoint(args, model, optimizer, **kwargs):
  checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1), step=tf.Variable(1),
                                   model=model, optimizer=optimizer)
  manager = tf.train.CheckpointManager(checkpoint, args.ckpt_dir, max_to_keep=args.max_to_keep)
  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")
  return checkpoint, manager


#@tf.function
def train_step(model, inputs, loss_fn, optimizer):
  with tf.GradientTape() as tape:
    y_hat = model(inputs)
    loss = loss_fn(inputs['angle'], y_hat, inputs['tertiary_mask'])
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss


#@tf.function
def eval_step(model, inputs, loss_fn):
  y_hat = model(inputs)
  return loss_fn(inputs['angle'], y_hat, inputs['tertiary_mask'])


def train(args, model, feeder, hparams):
  loss_fn = gaussian_mixture_loss_fn(out_dim=hparams.dihedral_dim,
                                     num_mix=hparams.num_mixtures,
                                     use_tfp=hparams.use_tfp)
  optimizer = tf.optimizers.Adam(learning_rate=hparams.learning_rate)
  ckpt, manager = init_checkpoint(args, model, optimizer)

  for epoch in range(int(ckpt.epoch), hparams.epochs + 1):
    ckpt.epoch.assign_add(1)
    avg_loss = []
    start = time.time()
    for data_dict in feeder.train:
      ckpt.step.assign_add(1)
      loss = train_step(model, data_dict, loss_fn, optimizer)
    avg_loss.append(loss)

    eval_loss = []
    for data_dict in feeder.test:
      loss = eval_step(model, data_dict, loss_fn)
      eval_loss.append(loss)

    print("Epoch: {} | train loss: {:.3f} | time: {:.2f}s | eval loss: {:.3f}".format(
        epoch, np.mean(avg_loss), time.time() - start, np.mean(eval_loss)))

    if epoch % args.ckpt_inteval == 0:
      save_path = manager.save()
      print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

