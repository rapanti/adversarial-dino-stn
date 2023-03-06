import argparse
import time
from pathlib import Path

import eval_linear
import main_dino


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO TRAINING PIPELINE', parents=[main_dino.get_args_parser()])
    parser.add_argument("--pipeline_mode", default=('pretrain', 'eval'), choices=['pretrain', 'eval'], type=str, nargs='+')
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
        args.checkpoint_key = "teacher"
        args.n_last_blocks = 4
        args.avgpool_patchtokens = False
        args.val_freq = 1
        print('STARTING EVALUATION')
        eval_linear.eval_linear(args, True)
        print('FINISHED EVALUATION')
