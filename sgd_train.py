import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim
import torch.utils.data

import config
import network
import utils


def train(model, device, train_loader, optimizer, epoch, step, adversarial, aux_optimizer=None):
    model.train()

    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target, indices = data.to(device), target.to(device), indices.to(device)
        config.tb.add_scalar("train/epoch", epoch, batch_idx + step)
        config.tb.add_histogram("train/indices", indices.cpu(), batch_idx + step)

        optimizer.zero_grad()

        y_hat = model(data)
        loss = F.nll_loss(y_hat, target)
        config.tb.add_scalar("train/loss", float(loss.item()), batch_idx + step)

        loss.backward()
        optimizer.step()

        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return batch_idx + step


def test(model, device, test_loader, step):
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
    config.tb.add_scalar("test/loss", test_loss, step)
    config.tb.add_scalar("test/accuracy", correct / len(test_loader.dataset), step)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    assert not config.adversarial_sampling
    torch.manual_seed(config.random_seed)
    train_loader, test_loader = utils.load_datasets()

    if config.dataset == "mnist":
        model = network.MNISTNetwork(train_loader).to(config.device)
    elif config.dataset == "cifar10":
        model = network.CIFAR10Network(train_loader).to(config.device)

    step = 0
    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.warmup_lr)

    for epoch in range(config.warmup_epochs):
        step = train(model, config.device, train_loader, optimizer, epoch, step, adversarial=False)
        test(model, config.device, test_loader, step)


if __name__ == '__main__':
    main()
