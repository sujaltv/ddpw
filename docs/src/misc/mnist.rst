:orphan:

.. _sec:mnist-example:

Example
^^^^^^^

.. code-block:: python
    :caption: src/model.py
    :linenos:

    import torch
    from torch.nn import functional as F


    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            self.conv1 = torch.nn.Conv2d(1, 32, (3,3))
            self.conv2 = torch.nn.Conv2d(32, 64, (3,3))

            self.bn = torch.nn.BatchNorm2d(64)

            self.drop1 = torch.nn.Dropout2d(0.25)
            self.drop2 = torch.nn.Dropout2d(0.5)

            self.fc1 = torch.nn.Linear(9216, 128)
            self.fc2 = torch.nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))

            x = self.bn(x)

            x = F.dropout(F.max_pool2d(x, 2), 0.25)

            x = x.flatten(1)
            x = F.relu(self.fc1(x))

            x = F.dropout(x, 0.5)
            x = self.fc2(x)

            return F.log_softmax(x, 1)


.. code-block:: python
    :caption: main.py
    :linenos:
    :emphasize-lines: 1,10,18

    from ddpw import Platform, wrapper

    from torch.cuda import set_device
    from torch.optim import SGD
    from torchvision.datasets.mnist import MNIST
    from torchvision import transforms as T

    from src.model import Model

    platform = Platform(device='slurm', n_nodes=2, n_gpus=4, n_cpus=8, ram=16)

    EPOCS = 50
    BATCH_SIZE = 64

    t = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    dataset = MNIST(root='./datasets/MNSIT', train=True, download=True, transform=t)

    @wrapper(platform)
    def train(*args, **kwargs):
        global_rank, local_rank = kwargs['global_rank'], kwargs['local_rank']
        print(f'Global rank {global_rank}; local rank {local_rank}')

        set_device(local_rank)

        # model
        model = DF.to(Model(), local_rank, device=platform.device)

        # dataset
        sampler=DF.get_dataset_sampler(dataset, global_rank, platform)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)

        # optimiser
        optimiser = SGD(model.parameters(), lr=1e-3)

        training_loss = torch.Tensor([0.0]).to(DF.device(model))
        print(f'Model on device {DF.device(model)}; dataset size: {len(dataloader) * batch_size}')

        # for every epoch
        for e in range(EPOCS):
            print(f'Epoch {e} of {epochs}')

            for _, (imgs, labels) in enumerate(dataloader, position=local_rank):
                optimiser.zero_grad()

                preds = model(imgs.to(DF.device(model)))
                loss = F.nll_loss(preds, labels.to(DF.device(model)))
                training_loss += loss
                loss.backward()

                optimiser.step()

            training_loss /= len(dataloader)

            # synchronise metrics
            if platform.requires_ipc:
                dist.all_reduce(training_loss, dist.ReduceOp.SUM)
                training_loss /= dist.get_world_size()

            if global_rank == 0:
                # code for storing logs and saving state
                print(training_loss.item())


    if __name__ == '__main__':
        train(args)
