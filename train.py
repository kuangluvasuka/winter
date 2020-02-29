import os
import time
import numpy as np
import tensorflow as tf

from utils import time_string
from models.mixture import gaussian_mixture_loss_fn, vonmises_mixture_loss_fn, masked_angular_mean_absolute_error, \
    independent_vonmises_mixture_loss_fn


def create_checkpoint(args, model, optimizer, **kwargs):
  checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1), step=tf.Variable(1),
                                   model=model, optimizer=optimizer)
  manager = tf.train.CheckpointManager(checkpoint, args.ckpt_dir, max_to_keep=args.max_to_keep)
  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")
  return checkpoint, manager


def create_summary(summary_dir, summary_off=False):
  if summary_off:
    return {'train': tf.summary.create_noop_writer(), 'eval': tf.summary.create_noop_writer()}

  logdir = os.path.join(summary_dir, time_string())
  train_log = os.path.join(logdir, 'train')
  eval_log = os.path.join(logdir, 'eval')
  os.makedirs(train_log, exist_ok=True)
  os.makedirs(eval_log, exist_ok=True)
  return {'train': tf.summary.create_file_writer(train_log), 'eval': tf.summary.create_file_writer(eval_log)}


#@tf.function
def train_step(model, inputs, loss_fn, optimizer):
  with tf.GradientTape() as tape:
    logits = model(inputs)
    loss = loss_fn(inputs['angle'], logits, inputs['tertiary_mask'])
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss


#@tf.function
def eval_step(model, inputs, loss_fn):
  logits = model(inputs)
  return loss_fn(inputs['angle'], logits, inputs['tertiary_mask'])


def train(args, model, feeder, hparams):
  if hparams.distribution == 'gaussian':
    loss_fn = gaussian_mixture_loss_fn(out_dim=hparams.dihedral_dim,
                                       num_mix=hparams.num_mixtures, use_tfp=hparams.use_tfp)
  elif hparams.distribution == 'von_mises':
    loss_fn = vonmises_mixture_loss_fn(out_dim=hparams.dihedral_dim,
                                       num_mix=hparams.num_mixtures, use_tfp=hparams.use_tfp)
  elif hparams.distribution == 'independent_von_mises':
    loss_fn = independent_vonmises_mixture_loss_fn(out_dim=hparams.dihedral_dim,
                                       num_mix=hparams.num_mixtures, use_tfp=hparams.use_tfp)
  else:
    raise ValueError("hparam.distribution has only two options: 'gaussian' or 'von_mises'.")
  optimizer = tf.optimizers.Adam(learning_rate=hparams.learning_rate)
  ckpt, manager = create_checkpoint(args, model, optimizer)
  summary_writer = create_summary(args.summary_dir, args.summary_off)

  for epoch in range(int(ckpt.epoch), hparams.epochs + 1):
    ckpt.epoch.assign_add(1)
    losses = []
    start = time.time()
    tf.summary.experimental.set_step(epoch)
    with summary_writer['train'].as_default():
      for (i, data_dict) in enumerate(feeder.train):
        ckpt.step.assign_add(1)
        # TODO: add summary_step?
        with tf.summary.record_if(i == 0):
          loss = train_step(model, data_dict, loss_fn, optimizer)
        losses.append(loss)
      train_loss_average = np.mean(losses) / hparams.batch_size
      tf.summary.scalar('loss', train_loss_average)

    losses = []
    with summary_writer['eval'].as_default():
      for (i, data_dict) in enumerate(feeder.test):
        with tf.summary.record_if(i == 0):
          loss = eval_step(model, data_dict, loss_fn)
        losses.append(loss)
      eval_loss_average = np.mean(losses) / hparams.batch_size_test
      tf.summary.scalar('loss', eval_loss_average)

    if epoch % args.prediction_inteval == 0:
      metrics = []
      for (i, data_dict) in enumerate(feeder.test):
        y_pred = model(data_dict, is_sampling=True)
        mae = masked_angular_mean_absolute_error(data_dict['angle'], y_pred, data_dict['tertiary_mask'])
        metrics.append(mae)
      eval_metric_average = np.mean(metrics, axis=0)

      print("Epoch: {} | train loss: {:.3f} | time: {:.2f}s | eval loss: {:.3f} | angle MAE: ({:.3f}, {:.3f}, {:.3f})".format(
          epoch, train_loss_average, time.time() - start, eval_loss_average, *eval_metric_average))
    else:
      print("Epoch: {} | train loss: {:.3f} | time: {:.2f}s | eval loss: {:.3f}".format(
          epoch, train_loss_average, time.time() - start, eval_loss_average))

    if epoch % args.ckpt_inteval == 0:
      save_path = manager.save()
      print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
