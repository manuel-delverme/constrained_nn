import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import tqdm

import imagenet_config as config


def main():
    global best_acc1
    model = models.resnet18()
    assert torch.cuda.is_available()

    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), config.initial_lr_theta, momentum=0.9, weight_decay=1e-4)

    # Data loading code
    train_dir = os.path.join(config.dataset_path, 'train')
    valdir = os.path.join(config.dataset_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(train_dir,
                                         transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize,
                                         ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.dataloader_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.dataloader_workers, pin_memory=True)

    step = 0
    for epoch in range(config.num_epochs):
        adjust_learning_rate(optimizer, epoch, config)

        step = train(train_loader, model, criterion, optimizer, step)
        validate(val_loader, model, criterion, step)

        config.tb.add_object("checkpoint", {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, step)


def train(train_loader, model, criterion, optimizer, step):
    model.train()

    end = time.time()

    episode_loss = 0.
    top1 = []
    top5 = []
    losses = []
    batch_time = []

    for batch_idx, (images, target) in tqdm.tqdm(enumerate(train_loader)):
        # measure data loading time

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_loss += float(loss.item())

        losses.append(float(loss.cpu()))
        top1.append(float(acc1[0]))
        top5.append(float(acc5[0]))
        batch_time.append(time.time() - end)

        end = time.time()

    new_step = step + len(train_loader)
    config.tb.add_scalar("train/epoch_time", torch.sum(batch_time), new_step)
    config.tb.add_scalar("train/loss", torch.mean(losses), new_step)
    config.tb.add_scalar("train/top1", torch.mean(top1), new_step)
    config.tb.add_scalar("train/top5", torch.mean(top5), new_step)
    return new_step


def validate(val_loader, model, criterion, step):
    model.eval()

    top1 = []
    top5 = []
    losses = []
    batch_time = []

    with torch.no_grad():
        end = time.time()
        for batch_idx, (images, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.append(float(loss.cpu()))
            top1.append(float(acc1[0]))
            top5.append(float(acc5[0]))
            batch_time.append(time.time() - end)

            end = time.time()

    config.tb.add_scalar("train/epoch_time", torch.sum(batch_time), step)
    config.tb.add_scalar("train/loss", torch.mean(losses), step)
    config.tb.add_scalar("train/top1", torch.mean(top1), step)
    config.tb.add_scalar("train/top5", torch.mean(top5), step)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.initial_lr_theta * (0.1 ** (epoch // 30))
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
    main()
