import os
from enum import Enum
from datetime import datetime

import click
import torch
from torchvision import transforms as T

from src.model import Net
from src.dataset import Dataset
from src.loss import Loss
from src.trainer import CustomTrainer
from ddpwrapper import DDPWrapper, Platform, add_click_options


train_resume_options = [
  click.option('-cpu', type=bool, required=False, default=False, show_default=True,
                help='Use CPU; mutually exclusive with -gpu and -slurm'),
  click.option('-gpu', type=bool, required=False, default=True, show_default=True,
                help='Use GPU; precedes -cpu'),
  click.option('-slurm', type=bool, required=False, default=False, show_default=True,
                help='Use SLURM; precedes -gpu and -cpu'),
  click.option('-n-gpus', type=int, required=False, default=2, show_default=True,
                help='Number of GPUs to use for training. Ignored for -cpu'),
  click.option('-log', type=bool, required=False, default=True, show_default=True,
                help='Is logging required?'),
  click.option('-ckpt-freq', type=int, required=False, default=0, show_default=True,
                help='How frequently to save checkpoints. Pass 0 to not save any'),
  click.option('-runs', type=str, required=False, default='runs', show_default=True,
                help='Runs folder'),
  click.option('-ckpt-dir', type=str, required=False, default='./models', show_default=True,
                help='Directory to store checkpoints in. Used only if -ckpt-freq is not zero'),
  click.option('-e', '--epochs', type=int, required=False, default=50, show_default=True,
                help='The number of epochs to train'),
  click.option('-b', '--batch-size', type=int, required=False, default=64, show_default=True,
                help='Training batch size'),
  click.option('-s', '--seed', type=int, required=False, default=1640, show_default=True,
                help='Seed to use before training'),
  click.option('-val', '--validate', type=click.IntRange(0, 100), required=False,
                default=0, show_default=True,
                help='Percentage of the dataset to set aside for validation at each epoch; 0 implies no validation needed'),
  click.option('-pr', '--protocol', type=str, required=False, show_default=True,
                default='tcp', help='Used for distributed training'),
  click.option('-host', '--hostname', type=str, required=False, show_default=True,
                default='localhost', help='Used for distributed training'),
  click.option('-p', '--port', type=str, required=False, show_default=True,
                default='1640', help='Used for distributed training')
]

model_option = [
 click.option('-ckpt', type=int, required=True, help='Model number')
]


class CommandType(Enum):
  Train = 0
  Resume = 1
  Evaluate = 2
  Infer = 3


def wrapper(flag: CommandType, **kwargs):
  if kwargs['ckpt_freq'] != 0:
    if not os.path.isdir(kwargs['ckpt_dir']):
      os.mkdir(kwargs['ckpt_dir'], mode=0o775)

  model = Net()

  r = kwargs["runs"]
  log_dir = f'{r}/{datetime.now().strftime("%I:%M:%S%p_%a_%d_%b_%Y")}'

  transformations = T.Compose([
    T.ToTensor(),
    T.Resize((28,28)),
    T.Normalize((0.1307,), (0.3081,))
  ])

  train_set = Dataset(root='data', download=True,
                      train=flag!=CommandType.Evaluate,
                      transform=transformations)
  val_set = None

  if flag!=CommandType.Evaluate and kwargs['validate'] > 0:
    total_size = len(train_set)
    val_size = total_size - (kwargs['validate'] * total_size // 100)
    train_size = total_size - val_size
    train_set, val_set = torch.utils.data.random_split(train_set,
                                                       [train_size, val_size])


  optimiser = torch.optim.Adadelta(model.parameters(), lr=.034)
  options = {
    'model': model,
    'loss_fn': Loss(),
    'optimiser': optimiser,
    # 'optimiser_step': StepLR(optimiser, step_size=1, gamma=0.7),
    'dataset': train_set,
    'trainer': CustomTrainer(),
    'validate': bool(kwargs['validate']),
    'validation_dataset': val_set,
    'nprocs': kwargs['n_gpus']
  }

  platform = Platform.GPU
  if kwargs['cpu']: platform = Platform.CPU
  elif kwargs['slurm']: platform = Platform.SLURM

  print('Running on', platform)

  job = DDPWrapper(platform=platform, **options)
  job.port = kwargs['port']
  job.seed = kwargs['seed']

  if flag == CommandType.Train:
    print('Starting train')
    job.start(
      epochs=kwargs['epochs'],
      ckpt_every=kwargs['ckpt_freq'],
      ckpt_dir=kwargs['ckpt_dir'],
      batch_size=kwargs['batch_size'],
      logdir=log_dir
    )
  elif flag == CommandType.Resume:
    job.resume(
      platform=platform,
      epochs=kwargs['epochs'],
      ckpt_every=kwargs['ckpt_freq'],
      ckpt=kwargs['ckpt'],
      ckpt_dir=kwargs['ckpt_dir'],
      batch_size=kwargs['batch_size'],
      logdir=log_dir
    )
  elif flag == CommandType.Evaluate:
    job.evaluate(
      ckpt_dir=kwargs['ckpt_dir'],
      ckpt=kwargs['ckpt']
    ).print()


@click.command(help='Start training a model afresh')
@add_click_options(train_resume_options)
def train(**kwargs):
  print('Wrapper called with train')
  wrapper(CommandType.Train, **kwargs)

@click.command(help='Resume training from a given model')
@add_click_options(train_resume_options)
@add_click_options(model_option)
def resume(**kwargs):
  wrapper(CommandType.Resume, **kwargs)

@click.command(help='Evaluate an already trained model')
@add_click_options(train_resume_options)
@add_click_options(model_option)
def evaluate(**kwargs):
  wrapper(CommandType.Evaluate, **kwargs)

@click.group()
def main():
  pass

main.add_command(train)
main.add_command(resume)
main.add_command(evaluate)


if __name__ == '__main__':
  main()
