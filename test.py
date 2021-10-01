import argparse

parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-cpu', type=bool, required=False, default=False,
              help='Use CPU; mutually exclusive with -gpu and -slurm')
parser.add_argument('-gpu', type=bool, required=False, default=True,
              help='Use GPU; precedes -cpu')
parser.add_argument('-slurm', type=bool, required=False, default=False,
              help='Use SLURM; precedes -gpu and -cpu')
parser.add_argument('-n-gpus', type=int, required=False, default=2,
              help='Number of GPUs to use for training. Ignored for -cpu')
parser.add_argument('-log', type=bool, required=False, default=True,
              help='Is logging required?')
parser.add_argument('-ckpt-freq', type=int, required=False, default=50,
              help='How frequently to save checkpoints. Pass 0 to not save any')
parser.add_argument('-ckpt-dir', type=str, required=False, default='./models',
              help='Directory to store checkpoints in. Used only if -ckpt-freq is not zero')
parser.add_argument('-e', '--epochs', type=int, required=False, default=50,
              help='The number of epochs to train')
parser.add_argument('-b', '--batch-size', type=int, required=False, default=64,
              help='Training batch size')
parser.add_argument('-s', '--seed', type=int, required=False, default=1640,
              help='Seed to use before training')
parser.add_argument('-val', '--validate', type=bool, required=False, default=False,
            help='Validate the model at the end of each epoch during training')
parser.add_argument('-b-val', '--batch-size-val', type=int, required=False,
              default=0, help='Validation batch size; 0 to vaidate at once')
parser.add_argument('-pr', '--protocol', type=str, required=False,
              default='tcp', help='Used for distributed training')
parser.add_argument('-host', '--hostname', type=str, required=False,
              default='localhost', help='Used for distributed training')
parser.add_argument('-p', '--port', type=str, required=False,
              default='1640', help='Used for distributed training')

if __name__ == '__main__':
  args = parser.parse_args()
  print(args)
