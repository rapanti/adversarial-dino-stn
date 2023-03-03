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
from torchvision import models as torchvision_models
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

import penalties
from stn import AugmentationNetwork, STN

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

penalty_list = sorted(name for name in penalties.__dict__
                      if name[0].isupper() and not name.startswith("__") and callable(penalties.__dict__[name]))
penalty_dict = {
    penalty: penalties.__dict__[penalty] for penalty in penalty_list
}

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
    args.batch_size_per_gpu = int(args.batch_size / utils.get_world_size())
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
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")

    transform_net = STN(
        mode=args.stn_mode,
        invert_gradients=args.invert_stn_gradients,
        separate_localization_net=args.separate_localization_net,
        conv1_depth=args.stn_conv1_depth,
        conv2_depth=args.stn_conv2_depth,
        theta_norm=args.stn_theta_norm,
        local_crops_number=args.local_crops_number,
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        resolution=args.stn_res,
        unbounded_stn=args.use_unbounded_stn,
    )
    stn = AugmentationNetwork(
        transform_net=transform_net,
        resize_input=args.resize_input,
        resize_size=args.resize_size,
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
    student, teacher, stn = student.cuda(), teacher.cuda(), stn.cuda()

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
    if utils.has_batchnorms(stn):
        stn = nn.SyncBatchNorm.convert_sync_batchnorm(stn)

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    stn = nn.parallel.DistributedDataParallel(stn, device_ids=[args.gpu])
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

    stn_penalty = penalty_dict[args.stn_penalty](
        invert=args.invert_penalty,
        eps=args.epsilon,
        target=args.penalty_target,
        local_crops_scale=args.local_crops_scale,
        global_crops_scale=args.global_crops_scale,
        min_glb_overlap=args.min_glb_overlap,
        min_lcl_overlap=args.min_lcl_overlap,
        resolution=32,
        exponent=2,
        bins=100,
    ).cuda() if args.stn_penalty else None

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)

    if args.stn_pretrained_weights:
        if not os.path.isfile(args.stn_pretrained_weights):
            print("Given path of stn pretrained weights is not a file.")
            sys.exit(1)
        utils.load_stn_pretrained_weights(stn, args.stn_pretrained_weights)
        # for p in stn.parameters():
        #     p.requires_grad = False

    if not args.use_stn_optimizer:
        # do not use wd scheduling of STN params
        student_params = params_groups[1]['params']
        all_params = student_params + list(stn.parameters())
        params_groups[1]['params'] = all_params

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        print(f"Optimizer {args.optimizer} not supported.")
        sys.exit(1)

    stn_optimizer = torch.optim.SGD(stn.parameters(), lr=args.stn_lr, momentum=0.9) if args.use_stn_optimizer else None

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
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

    rrc_or_stn = [[False] * args.alternate + [True] * args.alternate for _ in range(args.epochs // args.alternate + 1)]

    print(f"Loss, optimizer and schedulers ready.")

    # ============ ColorAugments after STN ============
    color_augment = utils.ColorAugmentation(args.dataset) if args.stn_color_augment else None

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
        stn=stn,
        stn_optimizer=stn_optimizer,
    )
    start_epoch = to_restore["epoch"]

    summary_writer = None
    if utils.is_main_process():
        summary_writer = utils.SummaryWriterCustom(log_dir=Path(args.output_dir) / "summary",
                                                   plot_size=args.summary_plot_size)

    stn_epoch = 0
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        if epoch and epoch % args.train_stn_freq == 0:
            for _ in range(args.stn_epochs):
                train_stn(student, teacher, dino_loss, data_loader, stn, stn_optimizer, stn_penalty, color_augment,
                          stn_epoch, args, summary_writer)
                stn_epoch += 1

        data_loader.sampler.set_epoch(epoch)

        use_stn = rrc_or_stn[epoch]
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args, summary_writer,
                                      stn, stn_optimizer, color_augment, use_stn)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'stn': stn.state_dict(),
        }
        if stn_optimizer:
            save_dict['stn_optimizer'] = stn_optimizer.state_dict()
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


def train_stn(student, teacher, dino_loss, data_loader, stn, stn_optimizer, stn_penalty, color_augment, epoch, args,
              summary_writer):
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration

        # move images to gpu
        if isinstance(images, list):
            images = [im.cuda(non_blocking=True) for im in images]
        else:
            images = images.cuda(non_blocking=True)

        stn_images, thetas = stn(images)

        penalty = stn_penalty(images=stn_images, target=images, thetas=thetas)

        if utils.is_main_process() and args.summary_writer_freq and it % args.summary_writer_freq == 0:
            utils.summary_writer_write_images_thetas(summary_writer, stn_images, images, thetas, epoch, it)

        if color_augment:
            stn_images = color_augment(stn_images)

        teacher_output = teacher(stn_images[:2])  # only the 2 global views pass through the teacher
        student_output = student(stn_images)
        dino = dino_loss(student_output, teacher_output, epoch)
        loss = dino + penalty

        if not penalty.isfinite():
            print("Penalty is {}, stopping training".format(penalty.item()), force=True)
            sys.exit(2)

        if not dino.isfinite():
            print("DINOLoss is {}, stopping training".format(dino.item()), force=True)
            sys.exit(2)

        stn_optimizer.zero_grad()
        loss.backward()
        param_norms = torch.nn.utils.clip_grad_norm_(stn.parameters(), args.clip_grad)
        stn_optimizer.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(dino=dino.item())
        metric_logger.update(norms=param_norms)
        metric_logger.update(penalty=penalty.item())

        if utils.is_main_process():
            summary_writer.add_scalar(tag="stn training: loss", scalar_value=loss.item(), global_step=it)
            summary_writer.add_scalar(tag="stn training: dino", scalar_value=dino.item(), global_step=it)
            summary_writer.add_scalar(tag="stn training: grad norm (stn)", scalar_value=param_norms, global_step=it)
            summary_writer.add_scalar(tag="stn training: penalty", scalar_value=penalty.item(), global_step=it)
            summary_writer.add_scalar(tag="stn training: lr stn", scalar_value=stn_optimizer.param_groups[0]["lr"], global_step=it)
            tmp = torch.stack([torch.det(theta[:, :, :2].float()).abs() for theta in thetas[:2]])
            scale = tmp.mean()
            std = tmp.std(0).mean()
            summary_writer.add_scalar(tag="scale global - mean", scalar_value=scale.item(), global_step=it)
            summary_writer.add_scalar(tag="scale global - std", scalar_value=std.item(), global_step=it)
            tmp = torch.stack([torch.det(theta[:, :, :2].float()).abs() for theta in thetas[2:]])
            scale = tmp.mean()
            std = tmp.std(0).mean()
            summary_writer.add_scalar(tag="scale local - mean", scalar_value=scale.item(), global_step=it)
            summary_writer.add_scalar(tag="scale local - std", scalar_value=std.item(), global_step=it)

        # Print gradients to STDOUT
        if utils.is_main_process() and args.grad_check_freq and it % args.grad_check_freq == 1:
            utils.print_gradients(stn)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,
                    epoch, fp16_scaler, args, summary_writer,
                    stn, stn_optimizer, color_augment, use_stn):
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    dino_crops = DINOCrops(args.global_crops_scale, args.local_crops_scale, args.local_crops_number)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        images = images.cuda(non_blocking=True)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if use_stn:
                stn_images, thetas = stn(images)
            else:
                stn_images = dino_crops(images)
                thetas = None

            if utils.is_main_process() and args.summary_writer_freq and it % args.summary_writer_freq == 0:
                utils.summary_writer_write_images_thetas(summary_writer, stn_images, images, thetas, epoch, it)

            if color_augment:
                stn_images = color_augment(stn_images)

            teacher_output = teacher(stn_images[:2])  # only the 2 global views pass through the teacher
            student_output = student(stn_images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not loss.isfinite():
            print("DINOLoss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(2)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
                fp16_scaler.unscale_(stn_optimizer) if stn_optimizer else None
                param_norms = torch.nn.utils.clip_grad_norm_(stn.parameters(), args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            if stn_optimizer:
                fp16_scaler.step(stn_optimizer)
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
            summary_writer.add_scalar(tag="grad norm", scalar_value=param_norms, global_step=it)

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


class DINOCrops(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=InterpolationMode.BILINEAR),
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=InterpolationMode.BILINEAR),
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(16, scale=local_crops_scale, interpolation=InterpolationMode.BILINEAR),
        ])

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_pico', 'vit_micro', 'vit_nano', 'vit_tiny', 'vit_small', 'vit_base', 'xcit',
                                 'deit_tiny', 'deit_small'] + torchvision_archs + torch.hub.list(
                            "facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.""")
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
    parser.add_argument('--batch_size', default=256, type=int,
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
                        help="Resizes all images of the ImageNet dataset to one size. Here: 224x224")
    parser.add_argument("--resize_input", default=False, type=utils.bool_flag,
                        help="Set this flag to resize the images of the dataset, images will be resized to the value given "
                             "in parameter --resize_size (default: 512). Can be useful for datasets with varying resolutions.")
    parser.add_argument("--resize_size", default=512, type=int,
                        help="If resize_input is True, this will be the maximum for the longer edge of the resized image.")

    # STN parameters
    parser.add_argument('--stn_mode', default='affine', type=str, help="""Determines the STN mode. Choose from:  
        affine, rotation, rotation_scale, rotation_scale_symmetric, rotation_translation, rotation_translation_scale,
        rotation_translation_scale_symmetric, scale, scale_symmetric, translation, translation_scale, 
        translation_scale_symmetric""")
    parser.add_argument('--stn_pretrained_weights', default='', type=str, help="""Path to pretrained weights of the 
        STN network. If specified, the STN is not trained and used to pre-process images solely.""")
    parser.add_argument("--invert_stn_gradients", default=False, type=utils.bool_flag,
                        help="Set this flag to invert the gradients used to learn the STN")
    parser.add_argument("--use_stn_optimizer", default=False, type=utils.bool_flag, help="""Set this flag to use a 
        separate optimizer for the STN parameters; annealed with cosine and no warmup""")
    parser.add_argument("--stn_lr", default=5e-4, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training) of the STN optimizer. The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--separate_localization_net", default=False, type=utils.bool_flag,
                        help="Set this flag to use a separate localization network for each head.")
    parser.add_argument("--summary_writer_freq", default=0, type=int,
                        help="Defines the number of iterations the summary writer will write output.")
    parser.add_argument("--grad_check_freq", default=0, type=int,
                        help="Defines the number of iterations the current tensor grad of the global 1 localization head is printed to stdout.")
    parser.add_argument("--stn_res", default=(224, 96), type=int, nargs='+',
                        help="Set the resolution of the global and local crops of the STN (default: 224x and 96x)")
    parser.add_argument("--use_unbounded_stn", default=False, type=utils.bool_flag,
                        help="Set this flag to not use a tanh in the last STN layer (default: use bounded STN).")
    parser.add_argument("--stn_warmup_epochs", default=0, type=int,
                        help="Specifies the number of warmup epochs for the STN (default: 0).")
    parser.add_argument("--stn_conv1_depth", default=32, type=int,
                        help="Specifies the number of feature maps of conv1 for the STN localization network (default: 32).")
    parser.add_argument("--stn_conv2_depth", default=32, type=int,
                        help="Specifies the number of feature maps of conv2 for the STN localization network (default: 32).")
    parser.add_argument("--stn_theta_norm", default=False, type=utils.bool_flag,
                        help="Set this flag to normalize 'theta' in the STN before passing to affine_grid(theta, ...). Fixes the problem with cropping of the images (black regions)")
    parser.add_argument("--stn_penalty", default=None, type=str, choices=penalty_list,
                        help="Specify the name of the similarity to use.")
    parser.add_argument("--epsilon", default=1., type=float,
                        help="Scalar for the penalty loss. Rescales the gradient by multiplication.")
    parser.add_argument("--invert_penalty", default=False, type=utils.bool_flag,
                        help="Invert the penalty loss.")
    parser.add_argument("--stn_color_augment", default=False, type=utils.bool_flag,
                        help="Color augmentations from DINO.")
    parser.add_argument("--summary_plot_size", default=16, type=int,
                        help="Defines the number of samples to show in the summary writer.")
    parser.add_argument("--penalty_target", default='mean', type=str, choices=['zero', 'one', 'mean', 'rand'],
                        help="Specify the type of target of the penalty. Here, the target is the area with respect to"
                             "the original image. `zero` and `one` are the values itself. `mean` and `rand` are"
                             "inferred with respect to given crop-scales.")
    parser.add_argument("--min_glb_overlap", default=0.5, type=float,
                        help="The minimal overlap between the two global crops.")
    parser.add_argument("--min_lcl_overlap", default=0.1, type=float,
                        help="The minimal overlap between two local crops.")

    parser.add_argument("--stn_epochs", default=5, type=int,
                        help="Number of epochs that the STN is trained.")
    parser.add_argument("--alternate", default=30, type=int,
                        help="Number of epochs that the STN and RRC training for DINO is alternated.")
    parser.add_argument("--train_stn_freq", default=30, type=int,
                        help="How often the stn is trained.")

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    argz = parser.parse_args()

    Path(argz.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(argz)
