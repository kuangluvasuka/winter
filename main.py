import argparse
import tensorflow as tf

from datasets.feeder import Feeder
from models.pronet import Pronet
from train import train
from utils import Params


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', default='.', help="")
  parser.add_argument('--data_dir', default='./data/casp7', help="")
  parser.add_argument('--restore_dir', default=None, help="")
  parser.add_argument('--param_path', default='./hparams.json', help="Hyperparameter file")

  args = parser.parse_args()

  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  tf.random.set_seed(117)

  hparams = Params(args.param_path)
  model = Pronet(hparams)
  feeder = Feeder(args.data_dir, hparams)

  #if args.restore:
  #  pass
  #else:
  #  pass

  train(model, feeder, hparams)


if __name__ == '__main__':
  main()
