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
import torchvision.datasets as datasets
import torchvision.models
import torchvision.models as models
import torchvision.transforms as transforms

import config
import experiment_buddy

best_acc1 = 0


def main(tb, args, task_config):
    global best_acc1
    model = models.alexnet()

    if args.device != "cuda":
        print('using CPU, this will be slow')

    # DataParallel will divide and allocate batch_size to all available GPUs
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), task_config.lr, momentum=task_config.momentum, weight_decay=task_config.weight_decay)

    # optionally resume from a checkpoint
    if task_config.resume:
        resume(task_config, model, optimizer)

    # Data loading code
    if "SLURM_JOB_ID" in os.environ.keys():
        dataset_path = args.dataset_path.format("imagenet", "imagenet")
    else:
        dataset_path = "../data/ImageNet"
    print("dataset:", dataset_path)

    traindir = os.path.join(dataset_path, 'train')
    valdir = os.path.join(dataset_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=task_config.batch_size, shuffle=True, num_workers=task_config.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=task_config.batch_size, shuffle=False,
        num_workers=task_config.workers, pin_memory=True)

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


def train(tb, train_loader, model, criterion, optimizer, epoch, args, total_gradients):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        if loss.isnan():
            raise ValueError

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(tb, i)
    return total_gradients + i


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


if __name__ == '__main__':
    experiment_buddy.register_defaults(vars(config))
    tb = experiment_buddy.deploy("mila")
    main(tb, config, config.ImageNet)
