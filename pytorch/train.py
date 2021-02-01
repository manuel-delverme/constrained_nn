import sys
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import pytorch.extragradient
from pytorch.network import ConstrNetwork


class MNIST(datasets.MNIST):
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        x_T, h, rhs, (x, multi) = model(data, indices)
        loss = F.nll_loss(x_T, target)
        # grad lambda = h
        lagr = loss + rhs
        lagr.backward()
        plot(lagr, model)
        # plot(multi, model)
        for i, m in enumerate(multi):
            torch.index_add()
            model.multipliers[i].grad[indices] *= -1
        # (-rhs).backward()

        if batch_idx % 2 == 0:
            optimizer.extrapolation()
        else:
            optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def plot(loss, model):
    import torchviz
    import os
    torchviz.make_dot(loss, params=dict(model.named_parameters())).render("/tmp/plot.gv")
    os.system("evince /tmp/plot.gv.pdf")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    use_cuda = torch.cuda.is_available()
    DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
    use_cuda = not DEBUG

    torch.manual_seed(1337)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = ConstrNetwork(len(train_loader.dataset)).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = pytorch.extragradient.ExtraAdam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(0, 14):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()


if __name__ == '__main__':
    main()
