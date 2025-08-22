:orphan:

.. _sec:mnist-example:

Example
^^^^^^^

.. code-block:: python
    :caption: src/model.py
    :linenos:

    from torch.nn import BatchNorm2d, Conv2d, Dropout2d, Linear, Module
    from torch.nn import functional as F


    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            self.conv1 = Conv2d(1, 32, (3,3))
            self.conv2 = Conv2d(32, 64, (3,3))

            self.bn = BatchNorm2d(64)

            self.drop1 = Dropout2d(0.25)
            self.drop2 = Dropout2d(0.5)

            self.fc1 = Linear(9216, 128)
            self.fc2 = Linear(128, 10)

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
    :caption: src/train.py
    :linenos:
    :emphasize-lines: 1,19,22,25

    from ddpw import functional as DF
    from torch import distributed as dist
    from torch import empty
    from torch.nn import functional as F
    from torch.optim import SGD
    from torch.utils.data import DataLoader


    def train(*args, **kwargs):
        model, batch_size, epochs, dataset = args
        global_rank, local_rank, platform = (
            kwargs["global_rank"],
            kwargs["local_rank"],
            kwargs["platform"],
        )
        print(f"Global rank {global_rank}; local rank {local_rank}")

        # set the current device
        DF.set_device(local_rank, platform)

        # move the (distributed) model to correct device (MPS/GPU)
        model = DF.to(model, local_rank)

        # sample the dataset
        sampler = DF.get_dataset_sampler(dataset, global_rank, platform)
        data = DataLoader(dataset, batch_size, sampler=sampler, pin_memory=True)

        # the optimiser
        optim = SGD(model.parameters(), lr=1e-2)

        # losses
        training_loss = empty((1,)).to(global_rank)

        # training over epochs...
        for e in range(epochs):
            # ...and over batches
            for imgs, labels in data:
                optim.zero_grad()
                preds = model(imgs.to(global_rank))
                loss = F.nll_loss(preds, labels.to(global_rank))
                loss.backward()
                optim.step()
            training_loss /= len(data)

            if platform.requires_ipc:
                dist.all_reduce(training_loss, dist.ReduceOp.SUM)
                training_loss /= platform.world_size

            # logging: console/tensorboard/wandb/etc.
            if global_rank == 0:
                print(training_loss.item())


.. code-block:: python
    :caption: src/evaluate.py
    :linenos:
    :emphasize-lines: 1,16,20,23

    from ddpw import functional as DF
    from torch import distributed as dist
    from torch import empty, load, no_grad
    from torch.backends import cudnn
    from torch.utils.data import DataLoader

    cudnn.deterministic = True


    @no_grad()
    def evaluate(*args, **kwargs):
        model, batch_size, dataset, ckptfile = args
        global_rank, platform = kwargs["global_rank"], kwargs["platform"]

        # set the current device
        DF.set_device(global_rank, platform)

        # move the (distributed) model to correct device (MPS/GPU)
        model.load_state_dict(load(ckptfile))
        model = DF.to(model, global_rank).eval()

        # sample the dataset
        sampler = DF.get_dataset_sampler(dataset, global_rank, platform)
        if platform.world_size > 1:
            sampler.set_epoch(0)
        data = DataLoader(dataset, batch_size, sampler=sampler, pin_memory=True)

        # evaluation metrics
        accuracy = empty((1,)).to(global_rank)

        # evaluation in batches
        for imgs, labels in data:
            preds = model(imgs.to(global_rank))
            accuracy += (
                (preds.argmax(-1) == labels.to(global_rank)).sum()
            ) / batch_size
        accuracy /= len(data) / 100

        if platform.requires_ipc:
            dist.all_reduce(accuracy, dist.ReduceOp.SUM)
            accuracy /= platform.world_size

        if global_rank == 0:
            print(accuracy.item())


.. code-block:: python
    :caption: main.py
    :linenos:
    :emphasize-lines: 1,13,14

    from ddpw import Wrapper, Platform
    from torchvision.datasets.mnist import MNIST

    from src import MNISTModel, train, evaluate


    if __name__ == "__main__":
        epochs = ...
        batch_size = ...
        dataset = MNIST(root="./input/datasets/MNIST/", train=True, transform=...)
        model = MNISTModel()

        platform = Platform(...)
        Wrapper(platform).start(train, model, batch_size, epochs, dataset)
