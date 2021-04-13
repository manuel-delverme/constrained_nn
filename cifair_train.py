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


def train(model, device, train_loader, optimizer, epoch, step, aux_optimizer=None):
    model.train()
    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target, indices = data.to(device), target.to(device), indices.to(device)

        opt_step(aux_optimizer, batch_idx, data, epoch, indices, model, optimizer, step, target, extrapolate=True)
        opt_step(aux_optimizer, batch_idx, data, epoch, indices, model, optimizer, step, target, extrapolate=False)
        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t')

    return batch_idx + step


def opt_step(aux_optimizer, batch_idx, data, epoch, indices, model, optimizer, step, target, extrapolate=False):
    optimizer.zero_grad()
    y_hat, sample_weights, rhs = model(data, indices)
    loss = F.nll_loss(y_hat, target, reduce=False) * sample_weights.squeeze()
    loss = loss.mean()
    # Extrapolation
    constr_loss = torch.einsum('bh,bh->', model.multipliers(indices), rhs)
    lagrangian = loss * constr_loss

    lagrangian.backward()  # Player 1

    if extrapolate:
        optimizer.extrapolation()
    else:
        optimizer.step()

    # Player 2
    model.multipliers[0].weight.grad = torch.sparse_coo_tensor(indices.unsqueeze(0), -rhs, model.multipliers[0].weight.shape)

    if extrapolate:
        aux_optimizer.extrapolation()
    else:
        aux_optimizer.step()

    if extrapolate:
        config.tb.add_scalar("train/loss", float(loss.item()), batch_idx + step)
        config.tb.add_scalar("train/mean_defect", float(rhs.mean()), batch_idx + step)
        config.tb.add_scalar("train/epoch", epoch, batch_idx + step)
        config.tb.add_scalar("train/lambda_h", float(constr_loss), batch_idx + step)
        config.tb.add_scalar("train/lagrangian0", lagrangian, batch_idx + step)
    return y_hat


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
        cuda_kwargs = {'num_workers': 0, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if "SLURM_JOB_ID" in os.environ.keys():
        dataset1 = CIFAR10(config.dataset_path.format("cifar10", "cifar10"), train=True, transform=transform)
        dataset2 = CIFAR10(config.dataset_path.format("cifar10", "cifar10"), train=False, transform=transform)
    else:
        dataset1 = CIFAR10("../data", train=True, transform=transform)
        dataset2 = CIFAR10("../data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = network.ConstrNetwork(
        torch.utils.data.DataLoader(dataset1, batch_size=test_kwargs['batch_size'])).to(config.device)

    step = 0

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
        step = train(model, config.device, train_loader, optimizer, epoch, step, aux_optimizer=aux_optimizer)
        test(model, config.device, test_loader, step)
        # scheduler.step(epoch)


if __name__ == '__main__':
    main()
