import datetime
import json
import os
from copy import deepcopy
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from eval_linear import build_dataset, LinearClassifier, train, validate_network
from main_dino import DINOLoss, train_one_epoch
import utils
import vision_transformer as vits
from vision_transformer import DINOHead

import penalties
from stn import STN

penalty_list = sorted(name for name in penalties.__dict__
                      if name[0].isupper() and not name.startswith("__") and callable(penalties.__dict__[name]))
penalty_dict = {
    penalty: penalties.__dict__[penalty] for penalty in penalty_list
}


def pretrain(args):
    mp.spawn(dino, (args, ), args.world_size)


def dino(rank, args):
    init_distributed_mode(rank, args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    with (Path(args.exp_dir) / "settings.json").open("w") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # ============ preparing data ... ============
    args.warmup_epochs = args.epochs // 10

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

    student = vits.__dict__[args.arch](
        img_size=args.img_size,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,  # stochastic depth
    )
    teacher = vits.__dict__[args.arch](img_size=args.img_size, patch_size=args.patch_size)
    embed_dim = student.embed_dim

    stn = STN(
        mode=args.stn_mode,
        invert_gradients=True,
        separate_localization_net=args.separate_localization_net,
        conv1_depth=args.stn_conv1_depth,
        conv2_depth=args.stn_conv2_depth,
        theta_norm=True,
        local_crops_number=args.local_crops_number,
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        resolution=args.stn_res,
        unbounded_stn=True,
    )

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
        min_lcl_overlap=args.min_lcl_overlap
    ).cuda() if args.stn_penalty else None

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)

    if not args.use_stn_optimizer:
        # do not use wd scheduling of STN params
        student_params = params_groups[1]['params']
        all_params = student_params + list(stn.parameters())
        params_groups[1]['params'] = all_params

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        stn_optimizer = torch.optim.AdamW(stn.parameters()) if args.use_stn_optimizer else None
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        stn_optimizer = torch.optim.SGD(stn.parameters(), lr=0, momentum=0.9) if args.use_stn_optimizer else None
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
        stn_optimizer = torch.optim.LARS(stn.parameters()) if args.use_stn_optimizer else None
    else:
        print(f"Optimizer {args.optimizer} not supported.")
        sys.exit(1)

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

    stn_lr_schedule = utils.cosine_scheduler(
        args.stn_lr * args.batch_size / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.stn_warmup_epochs,
    ) if stn_optimizer else None
    print(f"Loss, optimizer and schedulers ready.")

    # ============ ColorAugments after STN ============
    color_augment = utils.ColorAugmentation(args.dataset) if args.stn_color_augment else None

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.exp_dir, "checkpoint.pth"),
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
        summary_writer = utils.SummaryWriterCustom(log_dir=Path(args.exp_dir) / "summary",
                                                   plot_size=args.summary_plot_size)

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args, summary_writer,
                                      stn, stn_optimizer, stn_lr_schedule, stn_penalty, color_augment)

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
        utils.save_on_master(save_dict, os.path.join(args.exp_dir, 'checkpoint.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.exp_dir) / "log.train").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def evaluate(args):
    eval_args = deepcopy(args)
    eval_args.batch_size = 768
    eval_args.epochs = 100
    eval_args.lr = 0.01
    eval_args.weight_decay = 0
    eval_args.momentum = 0.9
    eval_args.val_freq = 20
    eval_args.checkpoint_key = "teacher"
    eval_args.n_last_blocks = 4
    eval_args.avgpool_patchtokens = False
    eval_args.pretrained_weights = Path(args.exp_dir) / "checkpoint.pth"

    mp.spawn(eval_linear, (eval_args, ), args.world_size)


def eval_linear(rank, args):
    init_distributed_mode(rank, args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    with (Path(args.exp_dir) / "settings.eval").open("w") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # ============ building network ... ============
    model = vits.__dict__[args.arch](
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=0
    )
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")
    model = model.cuda()
    model.eval()

    # infer per gpu batch size
    batch_size_per_gpu = int(args.batch_size / args.world_size)
    # ============ preparing data ... ============
    dataset_val, args.num_labels = build_dataset(is_train=False, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset_val)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    dataset_train, args.num_labels = build_dataset(is_train=True, args=args)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * args.batch_size / 768.,  # linear scaling rule
        momentum=args.momentum,
        weight_decay=args.weight_decay,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.exp_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    writer = None
    if utils.is_main_process() == 0:
        path = Path(args.exp_dir).joinpath("summary")
        writer = SummaryWriter(path)

    print("Setup completed ---> Starting Training and Evaluation")
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks,
                            args.avgpool_patchtokens, writer, args)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks,
                                          args.avgpool_patchtokens, args)
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
            with (Path(args.exp_dir) / "log.eval").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.exp_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def init_distributed_mode(rank, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(
        backend=args.dist_backend,
        world_size=args.world_size,
        rank=rank,
    )
    gpu = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu)
    print('distributed init (rank {})'.format(args.rank), flush=True)
    dist.barrier()
    setup_for_distributed(rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
