import sys

import torch
import torch.autograd
import torch.functional
import torch.nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import tqdm

import config
import torch_constrained
import extragradient
import network
import utils


def train(primal, train_loader, optimizer: torch_constrained.ConstrainedOptimizer, epoch, step, adversarial):
    primal.train()

    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target, indices = data.to(config.device), target.to(config.device), indices.to(config.device)
        config.tb.add_scalar("train/epoch", epoch, batch_idx + step)
        config.tb.add_scalar("train/adversarial", float(adversarial), batch_idx + step)
        config.tb.add_histogram("train/indices", indices.cpu(), batch_idx + step)

        if adversarial:
            loss, defect = optimizer.step(lambda: (*forward_step(data, indices, primal, target, len(train_loader.dataset)), None))
            config.tb.add_scalar("train/loss", float(loss.item()), batch_idx + step)

        # parameter_metrics(batch_idx, defects, loss, primal, rhs, step)
        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}', file=sys.stderr)
        sys.stderr.flush()

    return batch_idx + step


def parameter_metrics(batch_idx, defects, loss, primal, rhs, step):
    config.tb.add_scalar("train/loss", float(loss.item()), batch_idx + step)
    assert len(defects) == len(rhs) == len(primal.state_model.state_params)
    for idx, (defect, rh, xi) in enumerate(zip(defects, rhs, primal.state_model.state_params)):
        if idx < 5 or (idx % 10) == 0:
            config.tb.add_scalar(f"h{idx}/train/abs_mean_defect", float(defect.abs().mean()), batch_idx + step)
            config.tb.add_scalar(f"h{idx}/train/mean_defect", float(defect.mean()), batch_idx + step)
            config.tb.add_scalar(f"h{idx}/train/mean_rhs", float(rh.mean()), batch_idx + step)
            config.tb.add_histogram(f"h{idx}/train/defect", defect.detach().cpu().numpy(), batch_idx + step)

            if config.distributional:
                mean, scale = xi.means(xi.ys), xi.scales(xi.ys)
                if idx == 0:
                    config.tb.add_scalar(f"h{idx}/train/multipliers_abs_mean", dual[0].weight.abs().mean().cpu().detach().numpy(), batch_idx + step)
                else:
                    config.tb.add_histogram(f"h{idx}/train/distributional_multipliers", dual[1][idx - 1].abs().mean(axis=1).cpu().detach().numpy(), batch_idx + step)

                config.tb.add_histogram(f"h{idx}/train/state_scale_mean", scale.mean(axis=1).cpu().detach().numpy(), batch_idx + step)
                config.tb.add_scalar(f"h{idx}/train/state_loc_mean", mean.mean(), batch_idx + step)
                config.tb.add_histogram(f"h{idx}/train/state_mean_pdist", torch.pdist(mean).cpu().detach().numpy(), batch_idx + step)


def dual_backward(defects, indices, multipliers):
    multiplier_grad = -defects[0]
    multipliers[0].weight.grad = torch.sparse_coo_tensor(indices.unsqueeze(0), multiplier_grad, multipliers[0].weight.shape)
    for idx, (h_i, lambda_i) in enumerate(zip(defects[1:], multipliers[1:])):
        if config.distributional:
            multiplier_grad = -h_i
            lambda_i.grad = multiplier_grad.mean(0)
        else:
            raise NotImplemented


def forward_step(x, indices, model: network.TargetPropNetwork, targets, dataset_size):
    if config.experiment == "target-prop":
        states = [x, ] + model.state_model(indices)
        activations = model.transition_model.one_step(states)
        y_i, y_T = activations[:-1], activations[-1]
        defects = defect_fn(indices, model, y_i, targets, dataset_size)

        if config.distributional:
            num_classes = y_T.shape[1]
            targets = torch.arange(num_classes, device=y_T.device).repeat(y_T.shape[0])
            y_hat = y_T.flatten(0, 1)  # [bs, classes]
            loss = F.nll_loss(y_hat, targets)
        else:
            loss = F.nll_loss(y_T, targets)

    elif config.experiment == "sgd":
        y_hat = model(x)
        loss = F.nll_loss(y_hat, targets)
        defects = None
    else:
        raise NotImplemented
    return loss, defects


def defect_fn(indices, model, hat_y, targets, dataset_size):
    defects = []
    if config.distributional:
        first_distr = model.state_model.state_params[0]
        loc, scale = first_distr.means(first_distr.ys), first_distr.scales(first_distr.ys)
        h = (hat_y[0] - loc[targets]) / scale[targets]
        h = F.softshrink(h, config.distributional_margin)

        h = torch.sparse_coo_tensor(indices.unsqueeze(0), h, (dataset_size, h.shape[1]))
        defects.append(h)

        for hat_y_i, y_i in zip(hat_y[1:], model.state_model.state_params[1:]):
            loc, scale = first_distr.means(y_i.ys), first_distr.scales(y_i.ys)
            h = (hat_y_i - loc[targets]) / scale[targets]
            h = F.softshrink(h, config.distributional_margin)
            defects.append(h)

    else:
        for a_i, state in zip(hat_y, model.state_model.state_params):
            h = a_i - state(indices)
            h = F.softshrink(h, config.tabular_margin)
            defects.append(h)
    return defects


def test(model, device, test_loader, step):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for (data, target, idx) in test_loader:
            data, target = data.to(device), target.to(device)
            data = data  # .double()
            output = model.transition_model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    config.tb.add_scalar("test/loss", test_loss, step)
    config.tb.add_scalar("test/accuracy", correct / len(test_loader.dataset), step)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)),
          file=sys.stderr)


def main():
    torch.manual_seed(config.random_seed)
    train_loader, test_loader = utils.load_datasets()

    tp_net = load_models(train_loader)
    step = 0

    if config.experiment == "sgd" or config.constraint_satisfaction == "penalty":
        unconstrained_epochs = config.num_epochs
        constrained_epochs = None
    else:
        unconstrained_epochs = config.warmup_epochs
        constrained_epochs = config.num_epochs

    optimizer = torch.optim.Adagrad(tp_net.parameters(), lr=config.warmup_lr)
    for epoch in range(unconstrained_epochs):
        step = train(tp_net, config.device, train_loader, optimizer, epoch, step, adversarial=False)
        test(tp_net, config.device, test_loader, step)

    if constrained_epochs is not None:
        if config.constraint_satisfaction == "extra-gradient":
            optimizer_primal = extragradient.ExtraAdagrad
            optimizer_dual = extragradient.ExtraSGD
        elif config.constraint_satisfaction == "descent-ascent":
            optimizer_primal = torch.optim.Adagrad
            optimizer_dual = torch.optim.SGD
        else:
            raise NotImplemented

        primal_variables = [
            {'params': tp_net.transition_model.parameters(), 'lr': config.initial_lr_theta},
            {'params': tp_net.state_model.parameters(), 'lr': config.initial_lr_x},
        ]

        optimizer = torch_constrained.ConstrainedOptimizer(optimizer_primal, optimizer_dual, config.initial_lr_x, config.initial_lr_y, primal_variables)
        config.tb.watch(tp_net, log="all")
        # config.tb.watch(multipliers, log="all")

        for epoch in tqdm.trange(config.num_epochs):
            step = train(tp_net, train_loader, optimizer, epoch, step, adversarial=True)
            test(tp_net, config.device, test_loader, step)
    print("Done", file=sys.stderr)


def load_models(train_loader):
    model = network.SplitNet(config.dataset)
    dataset_size = len(train_loader.dataset)
    state_sizes = [b[0].in_features for b in model.blocks[1:]]

    if config.distributional:
        num_classes = len(train_loader.dataset.classes)
        state_net = network.GaussianStateNet(dataset_size, num_classes, state_sizes, config.num_samples)
    else:
        weights = [torch.zeros(dataset_size, state_size) for state_size in state_sizes]
        with torch.no_grad():
            for batch_idx, (data, target, indices) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                a_i = data
                for t, block in enumerate(model.blocks[:-1]):
                    a_i = block(a_i)
                    weights[t][indices] = a_i
        state_net = network.TabularStateNet(dataset_size, weights, state_sizes)
    tp_net = network.TargetPropNetwork(model, state_net)
    return tp_net.to(config.device)  # , multipliers.to(config.device)


if __name__ == '__main__':
    main()
