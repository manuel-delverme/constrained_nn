import math
import os
import sys
import time

import numpy as np
import torch
import torch.autograd
import torch.functional
import torch.nn
import torch.nn.functional as F  # noqa
import torch.optim
import torch.utils.data
import tqdm

import config
import experiment_buddy
import network
import torch_constrained
import utils


def train(logger, model, train_loader, optimizer: torch_constrained.ConstrainedOptimizer, epoch, step, adversarial):
    model.train()
    loss, batch_idx = None, None

    data_load_start = time.time()
    epoch_data_time = 0

    for batch_idx, (images, target, indices) in enumerate(train_loader):
        images = images.to(config.device, non_blocking=True)
        target = target.to(config.device, non_blocking=True)
        indices = indices.to(config.device, non_blocking=True)

        batch_data_time = time.time() - data_load_start
        epoch_data_time += batch_data_time
        logger.add_scalar("performance/step_data_time", batch_data_time, batch_idx + step)
        logger.add_scalar("performance/epoch_data_time", epoch_data_time, batch_idx + step)

        logger.add_scalar("train/epoch", epoch, batch_idx + step)
        logger.add_scalar("train/adversarial", float(adversarial), batch_idx + step)
        logger.add_histogram("train/indices", indices.cpu(), batch_idx + step)
        begin_batch_time = time.time()

        if adversarial:
            def closure():
                loss_, eq_defect = forward_step(images, indices, model, target, len(train_loader.dataset))
                return loss_, eq_defect, None

            lagrangian, loss, defect, _ = optimizer.step(closure)  # noqa

            if math.isnan(lagrangian):
                raise ValueError

            logger.add_scalar("train/loss", float(loss.item()), batch_idx + step)
            logger.add_scalar("train/lagrangian", float(lagrangian), batch_idx + step)
            if (batch_idx + step) // config.model_log_freq:
                parameter_metrics(logger, batch_idx, defect, loss, model, step, optimizer)

        logger.add_scalar("performance/batch_time", time.time() - begin_batch_time, batch_idx + step)
        data_load_start = time.time()

    return batch_idx + step


def parameter_metrics(logger, batch_idx, defects, loss, primal, step, optimizer):
    rhs = optimizer.weighted_constraint(defects, None)
    logger.add_scalar("train/loss", float(loss.item()), batch_idx + step)
    assert len(defects) == len(rhs) == len(primal.state_model.state_params)
    for idx, (defect, rh, xi, dual) in enumerate(zip(defects, rhs, primal.state_model.state_params, optimizer.equality_multipliers)):
        if defect.is_sparse:
            defect = defect.to_dense()
        if idx < 5 or (idx % 10) == 0:
            logger.add_scalar(f"h{idx}/train/abs_mean_defect", float(defect.abs().mean()), batch_idx + step)
            logger.add_scalar(f"h{idx}/train/mean_defect", float(defect.mean()), batch_idx + step)
            logger.add_scalar(f"h{idx}/train/mean_rhs", float(rh.mean()), batch_idx + step)
            logger.add_histogram(f"h{idx}/train/defect", defect.detach().cpu().numpy(), batch_idx + step)

            if config.distributional:
                mean, scale = xi.means(xi.ys), xi.scales(xi.ys)
                if idx == 0:
                    logger.add_scalar(f"h{idx}/train/multipliers_abs_mean", dual.weight.abs().mean().cpu().detach().numpy(), batch_idx + step)
                else:
                    logger.add_histogram(f"h{idx}/train/distributional_multipliers", dual[1][idx - 1].abs().mean(axis=1).cpu().detach().numpy(), batch_idx + step)

                logger.add_histogram(f"h{idx}/train/state_scale_mean", scale.mean(axis=1).cpu().detach().numpy(), batch_idx + step)
                logger.add_scalar(f"h{idx}/train/state_loc_mean", mean.mean(), batch_idx + step)
                logger.add_histogram(f"h{idx}/train/state_mean_pairwise_dist", torch.pdist(mean).cpu().detach().numpy(), batch_idx + step)


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
        states = [x, *model.state_model(indices)]
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


def defect_fn(dataset_index, model, hat_y, targets, dataset_size):
    defects = []
    if config.distributional:
        first_distribution = model.state_model.state_params[0]
        loc, scale = first_distribution.means(first_distribution.ys), first_distribution.scales(first_distribution.ys)
        h = (hat_y[0] - loc[targets]) / scale[targets]

        h_eps = torch.hardshrink(h, config.distributional_margin)

        sparse_h = torch.sparse_coo_tensor(dataset_index.unsqueeze(0), h_eps.flatten(start_dim=1), (dataset_size, int(np.prod(h.shape[1:]))))
        defects.append(sparse_h)  # dataset_size x num_features

        for hat_y_i, layer_distr in zip(hat_y[1:], model.state_model.state_params[1:]):
            loc, scale = layer_distr.means(layer_distr.ys), layer_distr.scales(layer_distr.ys)
            h = (hat_y_i - loc) / scale
            defects.append(h.flatten())  # num_classes x num_features

    else:
        for a_i, state in zip(hat_y, model.state_model.state_params):
            h = a_i - state(dataset_index)
            h_eps = F.softshrink(h, config.tabular_margin)

            sparse_h = torch.sparse_coo_tensor(dataset_index.unsqueeze(0), h_eps, (dataset_size, h.shape[1]))
            defects.append(sparse_h)
    return defects


def evaluate(logger: experiment_buddy.WandbWrapper, model: torch.nn.Module, device, test_loader, step: int):
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
    logger.add_scalar("test/loss", test_loss, step)
    logger.add_scalar("test/accuracy", correct / len(test_loader.dataset), step)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)),
          file=sys.stderr)


def main(logger):
    torch.manual_seed(config.random_seed)
    train_loader, test_loader = utils.load_datasets()

    tp_net = load_models(train_loader, config)
    step = 0

    if config.experiment == "sgd" or config.constraint_satisfaction == "penalty":
        unconstrained_epochs = config.num_epochs
        constrained_epochs = None
    else:
        unconstrained_epochs = config.warmup_epochs
        constrained_epochs = config.num_epochs

    optimizer = torch.optim.Adagrad(tp_net.parameters(), lr=config.warmup_lr)

    for epoch in range(unconstrained_epochs):
        step = train(logger, tp_net, train_loader, optimizer, epoch, step, adversarial=False)
        evaluate(logger, tp_net, config.device, test_loader, step)

    if constrained_epochs is not None:
        if config.constraint_satisfaction == "extra-gradient":
            optimizer_primal = torch_constrained.ExtraAdagrad
            optimizer_dual = torch_constrained.ExtraSGD
        elif config.constraint_satisfaction == "descent-ascent":
            optimizer_primal = torch.optim.Adagrad
            optimizer_dual = torch.optim.SGD
        else:
            raise NotImplemented

        primal_variables = [
            {'params': tp_net.transition_model.parameters(), 'lr': config.initial_lr_theta},
            {'params': tp_net.state_model.parameters(), 'lr': config.initial_lr_x},
        ]

        margin = config.distributional_margin if config.distributional else config.tabular_margin

        optimizer = torch_constrained.ConstrainedOptimizer(
            optimizer_primal,
            optimizer_dual,
            config.initial_lr_x,
            config.initial_lr_y,
            primal_variables,
        )
        logger.watch(tp_net, config.model_log, log_freq=config.model_log_freq)

        for epoch in tqdm.trange(config.num_epochs):
            step = train(logger, tp_net, train_loader, optimizer, epoch, step, adversarial=True)
            evaluate(logger, tp_net, config.device, test_loader, step)
    print("Done", file=sys.stderr)


def load_models(train_loader, config):
    model = network.SplitNet(dataset=config.dataset, distributional=config.distributional, every_layer=config.every_layer)
    dataset_size = len(train_loader.dataset)

    features_size = []

    def get_shapes():
        def hook(_model, _input, output):
            print(_input[0].shape, output.shape)
            features_size.append(output.shape[1:])

        return hook

    hooks = []
    for name, module in model.named_modules():
        # if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        if name.count(".") == 1 and name.split(".")[0] == "blocks":
            hook = module.register_forward_hook(get_shapes())
            hooks.append(hook)

    x0 = train_loader.dataset.data[0].view(1, 1, *train_loader.dataset.data[0].shape).to(torch.float)
    _y = model(x0)
    [h.remove() for h in hooks]
    features_size.pop(-1)  # outputs are not activations

    if config.distributional:
        num_classes = len(train_loader.dataset.classes)
        state_net = network.GaussianStateNet(dataset_size, num_classes, features_size, config.num_samples)
    else:
        weights = [torch.zeros(dataset_size, state_size) for state_size in features_size]
        with torch.no_grad():
            if not config.DEBUG:
                for batch_idx, (data, target, indices) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                    a_i = data
                    for t, block in enumerate(model.blocks[:-1]):
                        a_i = block(a_i)
                        weights[t][indices] = a_i
        state_net = network.TabularStateNet(dataset_size, weights, features_size)
    tp_net = network.TargetPropNetwork(model, state_net)
    return tp_net.to(config.device)  # , multipliers.to(config.device)


if __name__ == '__main__':
    experiment_buddy.register_defaults(vars(config))
    tb = experiment_buddy.deploy(
        host="mila" if config.REMOTE else "",
        sweep_yaml="sweep_hyper.yaml" if config.RUN_SWEEP else False,
        extra_slurm_headers="""
        #SBATCH --gres=gpu:rtx8000:1
        """,
        proc_num=25 if config.RUN_SWEEP else 1
    )
    print("CWD", os.getcwd())
    # utils.update_hyper_parameters()
    if config.dataset == "imagenet":
        import imagenet
        imagenet.main(tb, config)
    else:
        main(tb)
