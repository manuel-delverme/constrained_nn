import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim
import torch.utils.data

import config
import extragradient
import network
import utils


def train(model, device, train_loader, optimizer, epoch, step, adversarial, update_weights, aux_optimizer=None):
    model.train()
    if not adversarial:
        raise NotImplemented

    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target, indices = data.to(device), target.to(device), indices.to(device)

        # zero grad
        # forward
        # loss

        # lambda * h
        # L = l + h
        # L.backward
        # optimizer.extrapolation

        # multi = -rhs
        # aux extrap

        # optimizer.zero_grad
        # aux_optimizer.zero_grad

        # forward
        # loss

        # costr losses
        # L = loss + costrloss

        # (lagr + config.lambda_ * rhs.pow(2).mean(1).mean(0)).backward()
        # optimizer.step()

        # multi.grad = -rhs
        # aux_optimizer.step

        opt_step(aux_optimizer, batch_idx, data, epoch, indices, model, optimizer, step, target, update_weights, extrapolate=True)
        opt_step(aux_optimizer, batch_idx, data, epoch, indices, model, optimizer, step, target, update_weights, extrapolate=False)
        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t')
    return batch_idx + step


def opt_step(aux_optimizer, batch_idx, data, epoch, indices, model, optimizer, step, target, update_weights, extrapolate=False):
    optimizer.zero_grad()

    constr_loss, loss, defect = forward_step(data, indices, model, target)

    loss.backward(retain_graph=True)
    unconstrained_loss_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
    optimizer.zero_grad()

    constr_loss.backward(retain_graph=True)
    constraint_satisfaction_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
    optimizer.zero_grad()

    lagrangian = loss + constr_loss

    if not extrapolate:
        lagrangian = lagrangian + config.lambda_ * defect.pow(2).mean(1).mean(0)

    lagrangian.backward()  # Player 1

    if extrapolate:
        optimizer.extrapolation()
    else:
        optimizer.step()
    if config.adversarial_sampling:
        update_weights()


    # Player 2
    model.multipliers[0].weight.grad = torch.sparse_coo_tensor(indices.unsqueeze(0), -defect, model.multipliers[0].weight.shape)
    #  constr_grad_norm = torch.norm(torch.stack(
    #      [torch.norm(p.grad.detach(), 2) for p in constr_loss.parameters() if p.grad is not None]
    #  ), 2)

    if extrapolate:
        aux_optimizer.extrapolation()
    else:
        aux_optimizer.step()

    metrics = {
        "train/loss": float(loss.item()),
        "train/l1_defect": float(defect.abs().sum()),
        "train/epoch": epoch,
        "train/constr_loss": float(constr_loss),
        "train/unconstrained_loss_grad_norm": float(unconstrained_loss_grad_norm),
        "train/constraint_satisfaction_grad_norm": float(constraint_satisfaction_grad_norm),
        "train/lagrangian": lagrangian
    }
    for k, v in metrics.items():
        if extrapolate:
            k += "_ext"
        config.tb.add_scalar(k, v, batch_idx + step)
    config.tb.add_histogram("train/indices", indices.cpu(), batch_idx + step)


def forward_step(data, indices, model, target):
    if config.experiment == "target_prop":
        y_hat, rhs = model(data, indices)
        loss = F.nll_loss(y_hat, target)
    else:
        y_hat, sample_weights, rhs = model(data, indices)
        loss = F.nll_loss(y_hat, target, reduce=False) * sample_weights.squeeze()
        loss = loss.mean()
    # Extrapolation
    constr_loss = torch.einsum('bh,bh->', model.multipliers(indices), rhs)
    return constr_loss, loss, rhs


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
    torch.manual_seed(config.random_seed)
    train_loader, test_loader = utils.load_datasets()
    if config.experiment == "target_prop":
        model = network.TargetPropNetwork(train_loader).to(config.device)
    else:
        raise NotImplemented
        model = network.ConstrNetwork(train_loader).to(config.device)

    @torch.no_grad()
    def update_weights():
        if config.adversarial_sampling:
            train_loader.sampler.weights = torch.softmax(torch.max(model.multipliers[0].weight, dim=-1).values, dim=0)

    update_weights()

    step = 0

    # print("WARMUP")
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=config.warmup_lr)
    # for epoch in range(config.warmup_epochs):
    #    step = train(model, config.device, train_loader, optimizer, epoch, step, adversarial=False)
    #    test(model, config.device, test_loader, step)

    # print("Adversarial")
    theta = [v for k, v in model.named_parameters() if not k.startswith("x1") and not k.startswith("multipliers")]
    x = [v for k, v in model.named_parameters() if k.startswith("x1")]
    multi = [v for k, v in model.named_parameters() if k.startswith("multipliers")]
    optimizer = extragradient.ExtraAdagrad(
        [
            {'params': theta, 'lr': config.initial_lr_theta},
            {'params': x, 'lr': config.initial_lr_x},
        ])
    aux_optimizer = extragradient.ExtraSGD([{'params': multi, 'lr': config.initial_lr_y}])

    for epoch in range(config.num_epochs):
        step = train(model, config.device, train_loader, optimizer, epoch, step, adversarial=True, aux_optimizer=aux_optimizer, update_weights=update_weights)
        test(model, config.device, test_loader, step)


if __name__ == '__main__':
    main()
