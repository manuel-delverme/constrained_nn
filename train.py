import sys
import inspect

import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim
import torch.utils.data

import config
import extragradient
import network
import utils


def train(model, device, train_loader, optimizer, epoch, step, adversarial, aux_optimizer=None):
    config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
    model.train()

    for batch_idx, (data, target, indices) in enumerate(train_loader):
        config.tb.run.alert("log", f"{inspect.currentframe().f_lineno} {batch_idx}")

        data, target, indices = data.to(device), target.to(device), indices.to(device)
        config.tb.add_scalar("train/epoch", epoch, batch_idx + step)
        config.tb.add_scalar("train/adversarial", float(adversarial), batch_idx + step)
        config.tb.add_histogram("train/indices", indices.cpu(), batch_idx + step)

        if adversarial:
            # Extrapolation
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            rhs, loss, defect = forward_step(data, indices, model, target)

            config.tb.add_scalar("train/loss", float(loss.item()), batch_idx + step)
            config.tb.add_scalar("train/abs_mean_defect", float(defect.abs().mean()), batch_idx + step)
            config.tb.add_scalar("train/mean_rhs", float(rhs.mean()), batch_idx + step)

            if config.distributional:
                config.tb.add_histogram("train/x1_scale", model.x1.scale.mean(axis=1).cpu().detach().numpy(), batch_idx + step)
                config.tb.add_scalar("train/x1_mean", model.x1.mean.mean(), batch_idx + step)
                config.tb.add_histogram("train/x1_dist_hist", torch.pdist(model.x1.mean).cpu().detach().numpy(), batch_idx + step)

            if config.constraint_satisfaction == "extra-gradient":
                (loss + rhs).backward()
                optimizer.extrapolation()

                # Player 2
                multiplier_grad = -defect
                if not config.eps_constraint:
                    multiplier_grad = F.softshrink(multiplier_grad, config.constr_margin)

                model.multipliers[0].weight.grad = torch.sparse_coo_tensor(indices.unsqueeze(0), multiplier_grad, model.multipliers[0].weight.shape)

                aux_optimizer.extrapolation()

                optimizer.zero_grad()
                aux_optimizer.zero_grad()

                # Step
                rhs, loss, defect = forward_step(data, indices, model, target)

            # Grads
            (loss + rhs).backward()  # Player 1
            optimizer.step()

            # Player 2
            # RYAN: this is not necessary
            multiplier_grad = -defect
            if not config.eps_constraint:
                multiplier_grad = F.softshrink(multiplier_grad, config.constr_margin)

            model.multipliers[0].weight.grad = torch.sparse_coo_tensor(indices.unsqueeze(0), multiplier_grad, model.multipliers[0].weight.shape)

            aux_optimizer.step()
        else:
            optimizer.zero_grad()

            rhs, loss, defect = forward_step(data, indices, model, target)
            config.tb.add_scalar("train/loss", float(loss.item()), batch_idx + step)

            if config.experiment != "sgd":
                config.tb.add_scalar("train/mean_defect", float(defect.mean()), batch_idx + step)
                config.tb.add_scalar("train/rhs_defect", float(rhs.mean()), batch_idx + step)

                constr_loss = config.lambda_ * defect.pow(2).mean(1).mean(0)
                lagr = loss + constr_loss
            else:
                lagr = loss

            lagr.backward()
            optimizer.step()

        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}', file=sys.stderr)

    return batch_idx + step


def forward_step(data, indices, model, target):
    config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
    if config.experiment == "target-prop":
        y_hat, defect = model(data, indices)
        loss = F.nll_loss(y_hat, target)
        rhs = torch.einsum('bh,bh->', model.multipliers(indices), defect)

    elif config.experiment == "robust-classification":
        y_hat = model(data, indices)
        sample_weights = model.x1(indices)

        p_data_ignored = 1. - sample_weights.mean()
        prob_defect = torch.relu(p_data_ignored - config.chance_constraint)
        defect = prob_defect.repeat(sample_weights.shape)

        loss = F.nll_loss(y_hat, target, reduce=False)
        robust_loss = loss * sample_weights.squeeze()
        loss = robust_loss.mean()
        rhs = torch.einsum('bh,bh->', model.multipliers(indices), defect)

    elif config.experiment == "sgd":
        y_hat = model(data, indices)
        loss = F.nll_loss(y_hat, target)
        rhs, defect = None, None
    else:
        raise NotImplemented
    return rhs, loss, defect


def test(model, device, test_loader, step):
    config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
    model.eval()
    test_loss = 0
    correct = 0
    log_prob = 0

    with torch.no_grad():
        for (data, target, idx) in test_loader:
            data, target = data.to(device), target.to(device)
            data = data  # .double()
            output = model.full_rollout(data)
            if config.distributional:
                for xi, yi in zip(data, target):
                    x1_hat = model.block1(xi.unsqueeze(0))
                    log_prob += model.x1.log_prob(x1_hat)[yi].mean()

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    if config.distributional:
        config.tb.add_scalar("test/prob_x1", torch.exp(log_prob / len(test_loader.dataset)), step)
    test_loss /= len(test_loader.dataset)
    config.tb.add_scalar("test/loss", test_loss, step)
    config.tb.add_scalar("test/accuracy", correct / len(test_loader.dataset), step)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)),
          file=sys.stderr)


def main():
    config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
    torch.manual_seed(config.random_seed)
    train_loader, test_loader = utils.load_datasets()

    model = load_model(train_loader).to(config.device)

    step = 0

    if config.experiment == "sgd" or config.constraint_satisfaction == "penalty":
        unconstrained_epochs = config.num_epochs
        constrained_epochs = None
    else:
        unconstrained_epochs = config.warmup_epochs
        constrained_epochs = config.num_epochs

    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.warmup_lr)
    for epoch in range(unconstrained_epochs):
        step = train(model, config.device, train_loader, optimizer, epoch, step, adversarial=False)
        test(model, config.device, test_loader, step)

    config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
    if constrained_epochs is not None:
        if config.constraint_satisfaction == "extra-gradient":
            optimizer_primal = extragradient.ExtraAdagrad
            optimizer_dual = extragradient.ExtraSGD
        elif config.constraint_satisfaction == "descent-ascent":
            optimizer_primal = torch.optim.Adagrad
            optimizer_dual = torch.optim.SGD
        else:
            raise NotImplemented

        theta = [v for k, v in model.named_parameters() if not k.startswith("x1") and not k.startswith("multipliers")]
        x = [v for k, v in model.named_parameters() if k.startswith("x1")]
        multi = [v for k, v in model.named_parameters() if k.startswith("multipliers")]
        primal_variables = [
            {'params': theta, 'lr': config.initial_lr_theta},
            {'params': x, 'lr': config.initial_lr_x},
        ]
        dual_variables = [{'params': multi, 'lr': config.initial_lr_y}]

        optimizer = optimizer_primal(primal_variables)
        aux_optimizer = optimizer_dual(dual_variables)
        config.tb.watch(model, log="all")

        config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
        for epoch in range(config.num_epochs):
            step = train(model, config.device, train_loader, optimizer, epoch, step, adversarial=True, aux_optimizer=aux_optimizer)
            test(model, config.device, test_loader, step)
        config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
    config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
    print("Done", file=sys.stderr)


def load_model(train_loader):
    config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
    if config.experiment == "target-prop":
        if config.dataset == "mnist":
            model = network.TargetPropNetwork(train_loader)
        elif config.dataset == "cifar10":
            model = network.CIAR10TargetProp(train_loader)
        else:
            raise NotImplemented
    elif config.experiment == "robust-classification":
        if config.dataset == "mnist":
            model = network.MNISTLiftedNetwork(len(train_loader.dataset))
        elif config.dataset == "cifar10":
            model = network.CIFAR10LiftedNetwork(len(train_loader.dataset))
        else:
            raise NotImplemented
    elif config.experiment == "sgd":
        if config.dataset == "mnist":
            model = network.MNISTNetwork()
        elif config.dataset == "cifar10":
            model = network.CIFAR10Network()
        else:
            raise NotImplemented
    else:
        raise NotImplemented
    return model


if __name__ == '__main__':
    config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
    main()
    config.tb.run.alert("log", f"{inspect.currentframe().f_lineno}")
