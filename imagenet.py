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

import train as train_module
import utils

best_acc1 = 0


def main(tb, config):
    global best_acc1

    train_loader, val_loader = utils.load_datasets()
    model = train_module.load_models(train_loader, config)

    if config.device != "cuda":
        print('using CPU, this will be slow')

    # DataParallel will divide and allocate batch_size to all available GPUs
    model.transition_model.features = torch.nn.DataParallel(model.transition_model.features)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch_constrained.ConstrainedOptimizer(
        torch_constrained.ExtraAdagrad,
        torch_constrained.ExtraSGD,
        config.initial_lr_theta,
        config.initial_lr_y,
        model.parameters(),
        dual_dtype=torch.float16,
    )

    # optionally resume from a checkpoint
    if config.resume:
        resume(config, model, optimizer)

    total_gradients = 0

    if config.evaluate:
        validate(tb, val_loader, model, criterion, config, total_gradients)
        return

    for epoch in range(config.start_epoch, config.epochs):
        adjust_learning_rate(optimizer, epoch, config)

        total_gradients = train_module.train(tb, model, train_loader, optimizer, epoch, total_gradients, adversarial=True)
        acc1 = validate(tb, val_loader, model, criterion, config, total_gradients)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


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
        for i, (images, target, indices) in enumerate(val_loader):
            if args.device == "cuda":
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                # TODO: why no indices here?

            # compute output
            output = model.transition_model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        progress.display(tb, total_batches)
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

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
    lr = args.initial_lr_theta * (0.1 ** (epoch // 30))
    for param_group in optimizer.primal_optimizer.param_groups:
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
