import os

import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms

import config
import extragradient
import network


class MNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.DEBUG:
            self.data, self.targets = self.data[:config.batch_size * 2], self.targets[:config.batch_size * 2]

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


def train(model, device, train_loader, optimizer, epoch, step, adversarial, aux_optimizer=None):
    model.train()
    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target, indices = data.to(device), target.to(device), indices.to(device)
        optimizer.zero_grad()
        x_T, rhs = model(data, indices)
        loss = F.nll_loss(x_T, target)

        config.tb.add_scalar("train/loss", float(loss.item()), batch_idx + step)
        config.tb.add_scalar("train/mean_defect", float(rhs.mean()), batch_idx + step)
        config.tb.add_scalar("train/adversarial", float(adversarial), batch_idx + step)
        config.tb.add_scalar("train/epoch", epoch, batch_idx + step)

        if adversarial:
            # Extrapolation

            constr_loss = torch.einsum('bh,bh->', model.multipliers(indices), rhs)
            config.tb.add_scalar("train/constr_loss", float(constr_loss), batch_idx + step)

            lagr = loss + constr_loss
            config.tb.add_scalar("train/lagrangian0", lagr, batch_idx + step)
            aug_lagr = lagr  # + config.lambda_ * rhs.pow(2).mean(1).mean(0)

            aug_lagr.backward()  # Player 1
            optimizer.extrapolation()

            # Player 2
            model.multipliers[0].weight.grad = torch.sparse_coo_tensor(indices.unsqueeze(0), -rhs, model.multipliers[0].weight.shape)
            aux_optimizer.extrapolation()

            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            # Step
            # Eval
            x_T, rhs = model(data, indices)
            loss = F.nll_loss(x_T, target)

            # Loss
            constr_loss = torch.einsum('bh,bh->', model.multipliers(indices).squeeze(), rhs)
            lagr = loss + constr_loss
            config.tb.add_scalar("train/lagrangian1", lagr, batch_idx + step)

            aug_lagr = lagr + config.lambda_ * rhs.pow(2).mean(1).mean(0)

            # Grads
            aug_lagr.backward()  # Player 1
            optimizer.step()

            # Player 2
            model.multipliers[0].weight.grad = torch.sparse_coo_tensor(indices.unsqueeze(0), -rhs, model.multipliers[0].weight.grad.shape)
            aux_optimizer.step()

            with torch.no_grad():
                # Eval
                x_T, rhs = model(data, indices)
                loss = F.nll_loss(x_T, target)

                # Loss
                constr_loss = torch.einsum('bh,bh->', model.multipliers(indices).squeeze(), rhs)
                lagr = loss + constr_loss
                config.tb.add_scalar("train/lagrangian-1", lagr, batch_idx + step)
        else:
            constr_loss = config.lambda_ * rhs.pow(2).mean(1).mean(0)
            lagr = loss + constr_loss
            lagr.backward()
            optimizer.step()

        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
              f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} rhs: {constr_loss.item():.6f}')
        # if constr_loss > 10:
        #     sys.exit()
    return batch_idx + step


def grad_step(aux_optimizer, batch_idx, data, indices, model, optimizer, step, target):
    optimizer.zero_grad()
    aux_optimizer.zero_grad()
    # Step
    # Eval
    x_T, rhs = model(data, indices)
    loss = F.nll_loss(x_T, target)
    # Loss
    constr_loss = torch.einsum('bh,bh->', model.multipliers(indices).squeeze(), rhs)
    lagr = loss + constr_loss
    config.tb.add_scalar("train/lagrangian1", lagr, batch_idx + step)
    aug_lagr = lagr + config.lambda_ * rhs.pow(2).mean(1).mean(0)
    # Grads
    aug_lagr.backward()  # Player 1
    return rhs


def plot(loss, model):
    import torchviz
    import os
    torchviz.make_dot(loss, params=dict(model.named_parameters())).render("/tmp/plot.gv")
    os.system("evince /tmp/plot.gv.pdf")


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
    if os.stat("../data"):
        dataset1 = MNIST("../data", train=True, transform=transform)
        dataset2 = MNIST("../data", train=False, transform=transform)
    else:
        dataset1 = MNIST(config.dataset_path, train=True, transform=transform)
        dataset2 = MNIST(config.dataset_path, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = network.ConstrNetwork(
        torch.utils.data.DataLoader(dataset1, batch_size=test_kwargs['batch_size'])).to(config.device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr_theta)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.initial_lr_theta)
    # https://discuss.pytorch.org/t/sparse-embedding-failing-with-adam-torch-cuda-sparse-floattensor-has-no-attribute-addcmul/5589/9

    # config.tb.watch(model, criterion=None, log="all", log_freq=10)
    step = 0
    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.warmup_lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    print("WARMUP")
    for epoch in range(config.warmup_epochs):
        step = train(model, config.device, train_loader, optimizer, epoch, step, adversarial=False)
        test(model, config.device, test_loader, step)
        # scheduler.step(epoch)

    print("Adversarial")
    theta = [v for k, v in model.named_parameters() if not k.startswith("x1") and not k.startswith("multipliers")]
    x = [v for k, v in model.named_parameters() if k.startswith("x1")]
    multi = [v for k, v in model.named_parameters() if k.startswith("multipliers")]
    optimizer = extragradient.ExtraAdagrad(
        [
            {'params': theta, 'lr': config.initial_lr_theta},
            {'params': x, 'lr': config.initial_lr_x},
        ])
    aux_optimizer = extragradient.ExtraSGD([{'params': multi, 'lr': config.initial_lr_y}])

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(config.num_epochs):
        step = train(model, config.device, train_loader, optimizer, epoch, step, adversarial=True, aux_optimizer=aux_optimizer)
        test(model, config.device, test_loader, step)
        # scheduler.step(epoch)


if __name__ == '__main__':
    main()
