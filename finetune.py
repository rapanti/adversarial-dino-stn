# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import utils
import vision_transformer as vits


def finetune(args, dist_initiated=False):
    utils.init_distributed_mode(args) if not dist_initiated else None
    cudnn.benchmark = True

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    with (Path(args.output_dir) / "settings.eval").open("w") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # infer per gpu batch size
    batch_size_per_gpu = int(args.batch_size / args.world_size)
    # ============ preparing data ... ============
    dataset_val, args.num_classes = build_dataset(is_train=False, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset_val)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    dataset_train, _ = build_dataset(is_train=True, args=args)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](
            img_size=args.img_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes
        )
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=args.num_classes)
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=args.num_classes)
    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # set optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr * args.batch_size / 768.,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            args.lr * args.batch_size / 768.,
            weight_decay=args.weight_decay,
        )
    else:
        print(f"Optimizer {args.optimizer} not supported.")
        sys.exit(1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    writer = None
    if utils.is_main_process() == 0:
        path = Path(args.output_dir).joinpath("summary")
        writer = SummaryWriter(path)

    print("Setup completed ---> Starting Training and Evaluation")
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, optimizer, train_loader, epoch, args.avg_global_pool, writer)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, args.avg_global_pool)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            if writer:
                writer.add_scalar(tag="acc1", scalar_value=test_stats["acc1"], global_step=epoch)
                writer.add_scalar(tag="acc5", scalar_value=test_stats["acc5"], global_step=epoch)
                writer.add_scalar(tag="best-acc", scalar_value=best_acc, global_step=epoch)

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.eval").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, optimizer, loader, epoch, avg_pool, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    loss_fn = nn.CrossEntropyLoss()
    for it, (samples, targets) in enumerate(metric_logger.log_every(loader, 20, header)):
        # global iteration
        it = len(loader) * epoch + it
        # move to gpu
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        preds = model(samples, avg_pool)

        # compute cross entropy loss
        loss = loss_fn(preds, targets)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer:
            writer.add_scalar(tag="loss(eval)", scalar_value=loss.item(), global_step=it)
            writer.add_scalar(tag="lr(eval)", scalar_value=optimizer.param_groups[0]["lr"], global_step=it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(loader, model, avg_pool):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Test:'
    loss_fn = nn.CrossEntropyLoss()
    for samples, targets in metric_logger.log_every(loader, 20, header):
        # move to gpu
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(samples, avg_pool)
        loss = loss_fn(output, targets)

        if model.module.num_classes >= 5:
            acc1, acc5 = utils.accuracy(output, targets, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, targets, topk=(1,))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if model.module.num_classes >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if model.module.num_classes >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.dataset == 'CIFAR10':
        return datasets.CIFAR10(args.data_path, download=True, train=is_train, transform=transform), 10
    if args.dataset == 'CIFAR100':
        return datasets.CIFAR100(args.data_path, download=True, train=is_train, transform=transform), 100
    elif args.dataset == 'ImageNet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        return dataset, 1000
    print(f"Does not support dataset: {args.dataset}")
    sys.exit(1)


def build_transform(is_train, args):
    if args.dataset == 'CIFAR10':
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(args.img_size),
                transforms.RandomHorizontalFlip(),
                normalize,
            ])
        factor = args.img_size // 32
        return transforms.Compose([
            transforms.Resize(args.img_size + factor * 4, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.img_size),
            normalize,
        ])
    if args.dataset == 'ImageNet':
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            normalize,
        ])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--batch_size', default=768, type=int, help='mini-batch size (default: 768)')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training. (default: 100)')
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str, help='Network architecture. (default: "vit_small")')
    parser.add_argument('--img_size', default=224, type=int, help='images input size (default: 224)')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model (default: 16)')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (default: "teacher")')
    parser.add_argument('--avg_global_pool', default=False, type=bool,
                        help="Whether or not to use average global pooling.")
    # Optimizer parameters
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['adamw', 'sgd'],
                        help='Type of optimizer. (default: sgd)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    # Dataset parameters
    parser.add_argument('--dataset', default="ImageNet", choices=["ImageNet", "CIFAR10", "CIFAR100"], type=str,
                        help='Specify name of dataset (default: ImageNet)')
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str, help='Specify path to your dataset.')
    # distributed training parameters
    parser.add_argument("--dist_url", default="env://", type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='Type of Distributed backend. (default: "nccl"')
    # Misc
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    finetune(args)
