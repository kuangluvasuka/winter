import argparse
import tensorflow as tf

from datasets.feeder import Feeder
from models.pronet import Pronet
from train import train
from utils import Params


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', default='.', help="")
  parser.add_argument('--data_dir', default='.', help="")
  parser.add_argument('--restore_dir', default=None, help="")
  parser.add_argument('--param_path', default='./hparams.json', help="Hyperparameter file")

  args = parser.parse_args()

  tf.random.set_seed(117)

  hparams = Params(args.param_path)
  model = Pronet(hparams)
  feeder = Feeder(args.datadir, hparams)

  if args.restore:
    pass
  else:
    pass

  train(model, feeder, hparams)


if __name__ == '__main__':
  main()
