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
import argparse
import os
import sys
import datetime
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import kornia.augmentation as K
from kornia.geometry.transform import resize

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

from diff_patch_selection.patchnet import PatchNet


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    with (Path(args.output_dir) / "settings.json").open("w") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # ============ preparing data ... ============
    dataset = utils.build_dataset(True, args)
    sampler = DistributedSampler(dataset, shuffle=True)
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            img_size=args.img_size,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](img_size=args.img_size, patch_size=args.patch_size)
        embed_dim = student.embed_dim
    else:
        print(f"Unknown architecture: {args.arch}")

    patch_net = PatchNet(
        patch_size=args.pnet_patch_size,
        k=args.local_crops_number,
        use_scorer_se=args.use_scorer_se,
        selection_method=args.selection_method,
        normalization_str=args.normalization_str,
        hard_topk_probability=0.,
        random_patch_probability=0.,
    )

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher, patch_net = student.cuda(), teacher.cuda(), patch_net.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    if utils.has_batchnorms(patch_net):
        patch_net = nn.SyncBatchNorm.convert_sync_batchnorm(patch_net)

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    patch_net = nn.parallel.DistributedDataParallel(patch_net, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        patch_net_optimizer = torch.optim.AdamW(patch_net.parameters())
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        patch_net_optimizer = torch.optim.SGD(patch_net.parameters(), lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
        patch_net_optimizer = torch.optim.LARS(patch_net.parameters())
    else:
        print(f"Optimizer {args.optimizer} not supported.")
        sys.exit(1)

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    args.batch_size = args.batch_size_per_gpu * utils.get_world_size()
    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))

    patch_net_lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    print(f"Loss, optimizer and schedulers ready.")

    # ============ ColorAugments after STN ============
    color_augment = utils.ColorAugmentation(args.dataset) if args.color_augment else None

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
        patch_net=patch_net,
        patch_net_optimizer=patch_net_optimizer,
    )
    start_epoch = to_restore["epoch"]

    summary_writer = None
    if utils.is_main_process():
        summary_writer = utils.SummaryWriterCustom(log_dir=Path(args.output_dir) / "summary",
                                                   plot_size=args.plot_size)

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args, summary_writer,
                                      patch_net, patch_net_optimizer, patch_net_lr_schedule, color_augment)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'patch_net': patch_net.state_dict(),
        }
        if patch_net_optimizer:
            save_dict['patch_net_optimizer'] = patch_net_optimizer.state_dict()
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.train").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,
                    epoch, fp16_scaler, args, summary_writer,
                    patch_net, patch_net_optimizer, patch_net_lr_schedule, color_augment):
    rcrop = K.RandomResizedCrop((32, 32), args.global_crops_scale)
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        for i, param_group in enumerate(patch_net_optimizer.param_groups):
            param_group["lr"] = patch_net_lr_schedule[it]

        # move images to gpu
        if isinstance(images, list):
            images = [im.cuda(non_blocking=True) for im in images]
        else:
            images = images.cuda(non_blocking=True)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            out = patch_net(images)
            out = [patch.squeeze() for patch in out]

            patches = [resize(out[0], 32), resize(out[1], 32)] + [resize(o, 16) for o in out[2:]]

            if utils.is_main_process() and args.summary_writer_freq and it % args.summary_writer_freq == 0:
                summary_writer.write_image_grid("images", images=patches, original_images=images,
                                                epoch=epoch, global_step=it)

            if color_augment:
                patches = color_augment(patches)

            teacher_output = teacher(patches[:2])  # only the global view passes through the teacher
            student_output = student(patches)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not loss.isfinite().all():
            print("DINOLoss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        patch_net_optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
                param_norms = torch.nn.utils.clip_grad_norm_(patch_net.parameters(), args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
            patch_net_optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
                fp16_scaler.unscale_(patch_net_optimizer)
                param_norms = torch.nn.utils.clip_grad_norm_(patch_net.parameters(), args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.step(patch_net_optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(norms=param_norms)

        if utils.is_main_process():
            summary_writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=it)
            summary_writer.add_scalar(tag="lr", scalar_value=optimizer.param_groups[0]["lr"], global_step=it)
            summary_writer.add_scalar(tag="weight decay", scalar_value=optimizer.param_groups[0]["weight_decay"],
                                      global_step=it)
            summary_writer.add_scalar(tag="grad norm (patch_net)", scalar_value=param_norms, global_step=it)

        # Print gradients to STDOUT
        if utils.is_main_process() and args.grad_check_freq and it % args.grad_check_freq == 1:
            utils.print_gradients(patch_net)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'],
                        help="""Name of architecture to train""")
    parser.add_argument('--img_size', default=224, type=int,
                        help='Parameter if the Vision Transformer. (default: 224) ')
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels of input square patches - 
        default 16 (for 16x16 patches). Using smaller values leads to better performance but requires more memory.
        Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling mixed precision
        training (--use_fp16 false) to avoid instabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag, help="""Whether or not to weight
        normalize the last layer of the DINO head. Not normalizing leads to better performance but can make the training
        unstable. In our experiments, we typically set this parameter to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag, help="""Whether to use batch
        normalizations in projection head (Default: False)""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help="""Initial value for the teacher
        temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int,
                        help='mini-batch size (default: 256), this is the total batch size of all GPUs')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs during which we keep the
        output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value
        if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.), help="""Scale range of the 
        cropped image before resizing, relatively to the origin image. Used for large global view cropping. When 
        disabling multi-crop (--local_crops_number 0), we recommend using a wider range of scale""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="""Scale range of the
        cropped image before resizing, relatively to the origin image. Used for small local view cropping of 
        multi-crop.""")

    # Misc
    parser.add_argument("--dataset", default="ImageNet", type=str, choices=["ImageNet", "CIFAR10", "CIFAR100"],
                        help="Specify the name of your dataset. Choose from: ImageNet, CIFAR10, CIFAR100")
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Specify path to the training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='Type of Distributed backend. (default: "nccl"')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--resize_all_inputs", default=False, type=utils.bool_flag,
                        help="Whether or not to resize all images to one size")
    parser.add_argument("--color_augment", default=False, type=utils.bool_flag,
                        help="Whether or not to use color augmentations")

    # PatchNet parameters
    parser.add_argument("--selection_method", default="perturbed-topk", type=str,
                        choices=["random", "hard-topk", "topk", "perturbed-topk"],
                        help="Whether or not to use color augmentations")
    parser.add_argument("--pnet_patch_size", default=16, type=int, help="Size of the patches from the images.")
    parser.add_argument("--use_scorer_se", default=False, type=utils.bool_flag,
                        help="Whether or not to use SqueezeExciteLayer in ScorerNet.")
    parser.add_argument("--normalization_str", default="identity", type=str,
                        help="""String specifying the normalization of the scores.""")
    parser.add_argument("--invert_gradients", default=False, type=utils.bool_flag)
    parser.add_argument("--hard_topk_probability", default=0, type=float)
    parser.add_argument("--random_patch_probability", default=0, type=float)


    # Misc
    parser.add_argument("--plot_size", default=8, type=int, help="The number of different images to plot.")
    parser.add_argument("--grad_check_freq", default=0, type=int, help="Frequency to print gradients of PatchNet.")
    parser.add_argument("--summary_writer_freq", default=0, type=int, help="Frequency to print images to tensorboard.")

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
