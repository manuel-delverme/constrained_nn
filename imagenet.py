import os
import shutil
import time

import torch
import torch.autograd
import torch.functional
import torch.nn
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.nn.parallel
import torch.optim
import torch.optim
import torch.utils.data
import torch.utils.data
import torch.utils.data.distributed
import torch_constrained
import torchvision.models

import config
import train as train_module
import utils

best_acc1 = 0


def sgd(params, d_p_list, momentum_buffer_list, *, weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            # was: d_p = d_p.add(param, alpha=weight_decay)
            d_p = (weight_decay * param).add(d_p)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)


import torch.optim._functional

torch.optim._functional.sgd = sgd


def main(tb, args, task_config):
    class MomentumSXGD(torch_constrained.ExtraSGD):
        def __init__(self, params, lr):
            super().__init__(params, lr, momentum=task_config.momentum, weight_decay=task_config.weight_decay)

    global best_acc1

    train_loader, val_loader = utils.load_datasets()
    model = train_module.load_models(train_loader)

    if args.device != "cuda":
        print('using CPU, this will be slow')

    # DataParallel will divide and allocate batch_size to all available GPUs
    model.transition_model.features = torch.nn.DataParallel(model.transition_model.features)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch_constrained.ConstrainedOptimizer(
        MomentumSXGD,
        torch_constrained.ExtraSGD,
        task_config.initial_lr_theta,
        task_config.initial_lr_y,
        model.parameters(),
    )

    # optionally resume from a checkpoint
    if task_config.resume:
        resume(task_config, model, optimizer)

    total_gradients = 0

    if config.evaluate:
        validate(tb, val_loader, model, criterion, task_config, total_gradients)
        return

    for epoch in range(task_config.start_epoch, task_config.epochs):
        adjust_learning_rate(optimizer, epoch, task_config)

        # train for one epoch
        total_gradients = train(tb, train_loader, model, criterion, optimizer, epoch, task_config, total_gradients)

        # evaluate on validation set
        acc1 = validate(tb, val_loader, model, criterion, task_config, total_gradients)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def defect_fn(indices, model, hat_y, targets, dataset_size):
    defects = []
    if config.distributional:
        first_distribution = model.state_model.state_params[0]
        loc, scale = first_distribution.means(first_distribution.ys), first_distribution.scales(first_distribution.ys)
        h = (hat_y[0] - loc[targets]) / scale[targets]

        sparse_h = torch.sparse_coo_tensor(indices.unsqueeze(0), h, (dataset_size, h.shape[1]))
        defects.append(sparse_h)

        for hat_y_i, y_i in zip(hat_y[1:], model.state_model.state_params[1:]):
            loc, scale = first_distribution.means(y_i.ys), first_distribution.scales(y_i.ys)
            h = (hat_y_i - loc[targets]) / scale[targets]
            defects.append(h)

    else:
        for a_i, state in zip(hat_y, model.state_model.state_params):
            h = a_i - state(indices)
            sparse_h = torch.sparse_coo_tensor(indices.unsqueeze(0), h, (dataset_size, h.shape[1]))
            defects.append(sparse_h)
    return defects


def resume(args, model, optimizer):
    global best_acc1
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


def train(tb, train_loader, model, criterion, optimizer, epoch, args, num_gradient_steps):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    defects = AverageMeter('defect', ':6.2f')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    i = 0

    for i, (images, target, indices) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        indices = indices.cuda(non_blocking=True)

        def closure():
            loss_, eq_defect = train_module.forward_step(images, indices, model, target, len(train_loader.dataset))
            # output = model(images)
            # loss_ = criterion(output, target)
            # defect_ = output.sum(-1, keepdim=True)
            return loss_, eq_defect, None

        lagrangian = optimizer.step(closure)  # noqa
        loss, (defect,), _ = closure()
        output = model.transition_model(images)

        if loss.isnan():
            raise ValueError

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        if defect.is_sparse:
            defect = defect.to_dense()
        defects.update(defect.mean().item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(tb, i)
    return num_gradient_steps + i


def validate(tb, val_loader, model: torchvision.models.AlexNet, criterion, args, total_batches):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.device == "cuda":
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(tb, total_batches + i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, tb, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        mtr = []
        for meter in self.meters:
            tb.add_scalar(meter.name, meter.val, batch)
            mtr.append(str(meter))
        entries += mtr
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
