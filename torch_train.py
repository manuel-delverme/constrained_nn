import sys
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms

from pytorch import config
from pytorch.network import ConstrNetwork


class MNIST(datasets.MNIST):
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


def train(model, device, train_loader, optimizer, epoch, step):
    model.train()
    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target, indices = data.to(device), target.to(device), indices.to(device)
        data = data  # .double()
        optimizer.zero_grad()
        # model.multipliers.requires_grad = False
        x_T, rhs = model(data, indices)
        loss = F.nll_loss(x_T, target)
        # grad lambda = h
        constr_loss = rhs.pow(2).mean(1).mean(0)
        lagr = loss + constr_loss
        lagr.backward()
        # clip_grad_value_(model.parameters(), 1.1)

        # for name, p in model.named_parameters():
        #     if not p.grad.is_sparse:
        #         p.grad.clamp(max=1.1)

        # for i, m in enumerate(multi):
        #     # torch.index_add()
        #     model.multipliers[i].weight.grad *= -1

        # if batch_idx % 2 == 0:
        #     optimizer.extrapolation()
        # else:
        #     optimizer.step()
        optimizer.step()
        # if batch_idx % 10 == 0:
        config.tb.add_scalar("train/loss", float(loss.item()), batch_idx + step)
        config.tb.add_scalar("train/constr_loss", float(constr_loss), batch_idx + step)
        config.tb.add_scalar("train/lagr", float(lagr.item()), batch_idx + step)
        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
              f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} rhs: {constr_loss.item():.6f}')
        if constr_loss > 10:
            sys.exit()
    return batch_idx + step


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
        for (data, target, idx) in test_loader:
            data, target = data.to(device), target.to(device)
            data = data  # .double()
            output = model.full_rollout(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    torch.manual_seed(config.random_seed)

    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.batch_size * 4}
    if config.use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = ConstrNetwork(
        torch.utils.data.DataLoader(dataset1, batch_size=test_kwargs['batch_size'])).to(config.device)
    # optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.initial_lr_theta)
    # https://discuss.pytorch.org/t/sparse-embedding-failing-with-adam-torch-cuda-sparse-floattensor-has-no-attribute-addcmul/5589/9
    # optimizer = pytorch.extragradient.ExtraAdam(model.parameters(), lr=0.001)

    config.tb.watch(model, criterion=None, log="all", log_freq=10)
    step = 0
    for epoch in range(config.num_epochs):
        step = train(model, config.device, train_loader, optimizer, epoch, step)
        test(model, config.device, test_loader)


if __name__ == '__main__':
    main()
