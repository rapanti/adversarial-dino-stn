import argparse
import time
from pathlib import Path

import eval_linear
import main_dino
import penalties
import utils

penalty_list = sorted(name for name in penalties.__dict__
                      if name[0].isupper() and not name.startswith("__") and callable(penalties.__dict__[name]))
penalty_dict = {
    penalty: penalties.__dict__[penalty] for penalty in penalty_list
}

parser = argparse.ArgumentParser('DINO pipeline')
# ======================================================================================================================
# ===============================================   PRETRAIN PARAMETER   ===============================================
# ======================================================================================================================
# Model parameters
parser.add_argument('--arch', default='vit_small', type=str,
                    choices=['vit_pico', 'vit_micro', 'vit_nano', 'vit_tiny', 'vit_small', 'vit_base'],
                    help='Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.')
parser.add_argument('--img_size', default=224, type=int, help='Parameter if the Vision Transformer. (default: 224) ')
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
parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
    during which we keep the output layer fixed. Typically doing so during
    the first epoch helps training. Try increasing this value if the loss does not decrease.""")
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
parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
    recommend using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
    local views to generate. Set this parameter to 0 to disable multi-crop training.
    When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for small local view cropping of multi-crop.""")

# Misc
parser.add_argument("--dataset", default="ImageNet", type=str, choices=["ImageNet", "CIFAR10", "CIFAR100"],
                    help="Specify the name of your dataset. Choose from: ImageNet, CIFAR10, CIFAR100")
parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                    help='Specify path to the training data.')
parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
parser.add_argument('--seed', default=0, type=int, help='Random seed.')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
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
parser.add_argument('--stn_mode', default='affine', type=str,
                    help='Determines the STN mode (choose from: affine, translation, scale, rotation, '
                         'rotation_scale, translation_scale, rotation_translation, rotation_translation_scale')
parser.add_argument('--stn_pretrained_weights', default='', type=str,
                    help="Path to pretrained weights of the STN network. If specified, the STN is not trained and used to pre-process images solely.")
parser.add_argument("--invert_stn_gradients", default=False, type=utils.bool_flag,
                    help="Set this flag to invert the gradients used to learn the STN")
parser.add_argument("--use_stn_optimizer", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate optimizer for the STN parameters; "
                         "annealed with cosine and no warmup")
parser.add_argument("--stn_lr", default=5e-4, type=float, help="""Learning rate at the end of
    linear warmup (highest LR used during training) of the STN optimizer. The learning rate is linearly scaled
    with the batch size, and specified here for a reference batch size of 256.""")
parser.add_argument("--separate_localization_net", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate localization network for each head.")
parser.add_argument("--summary_writer_freq", default=1e6, type=int,
                    help="Defines the number of iterations the summary writer will write output.")
parser.add_argument("--grad_check_freq", default=1e6, type=int,
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
parser.add_argument("--use_stn_penalty", default=False, type=utils.bool_flag,
                    help="Set this flag to add a penalty term to the loss. Similarity between input and output image of STN.")
parser.add_argument("--penalty_loss", default="ThetaLoss", type=str, choices=penalty_list,
                    help="Specify the name of the similarity to use.")
parser.add_argument("--epsilon", default=1., type=float,
                    help="Scalar for the penalty loss. Rescales the gradient by multiplication.")
parser.add_argument("--invert_penalty", default=False, type=utils.bool_flag,
                    help="Invert the penalty loss.")
parser.add_argument("--stn_color_augment", default=False, type=utils.bool_flag, help="todo")
parser.add_argument("--summary_plot_size", default=16, type=int,
                    help="Defines the number of samples to show in the summary writer.")
parser.add_argument("--penalty_target", default='mean', type=str, choices=['zero', 'one', 'mean', 'rand'],
                    help="Specify the type of target of the penalty. Here, the target is the area with respect to"
                         "the original image. `zero` and `one` are the values itself. `mean` and `rand` are"
                         "inferred with respect to given crop-scales.")
parser.add_argument("--min_glb_overlap", default=0.5, type=float, help="The minimal overlap between the two global crops.")
parser.add_argument("--min_lcl_overlap", default=0.1, type=float, help="The minimal overlap between two local crops.")
# ======================================================================================================================
# =================================================   EVAL PARAMETER   =================================================
# ======================================================================================================================
# Model parameters
parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
parser.add_argument('--pretrained_linear_weights', default='', type=str, help="Path to pretrained linear weights.")
parser.add_argument("--checkpoint_key", default="teacher", type=str,
                    help='Key to use in the checkpoint (example: "teacher")')
parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens for the `n` last 
                    blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                    help='Whether or not to concatenate the global average pooled features to the [CLS] token. '
                         'We typically set this to False for ViT-Small and to True with ViT-Base.')
# Misc
parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# ======================================================================================================================
# ===============================================   PIPELINE PARAMETER   ===============================================
# ======================================================================================================================
parser.add_argument("--pipeline_mode", default=('pretrain', 'eval'), choices=['pretrain', 'eval'], type=str, nargs='+')


if __name__ == "__main__":
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if 'pretrain' in args.pipeline_mode:
        print('STARTING PRETRAINING')
        main_dino.train_dino(args)
        time.sleep(10)
        print('FINISHED PRETRAINING')

    if 'eval' in args.pipeline_mode:
        # change linear specific parameters
        args.epochs = 300
        args.lr = 0.01
        args.momentum = 0.9
        args.weight_decay = 0
        args.batch_size = 768
        args.pretrained_weights = f"{args.output_dir}/checkpoint.pth"
        print('STARTING EVALUATION')
        eval_linear.eval_linear(args, True)
        print('FINISHED EVALUATION')
