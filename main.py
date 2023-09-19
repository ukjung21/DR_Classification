import os
import sys
import random
import builtins

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from utils.func import *
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model


def main():
    args = parse_config()
    cfg = load_config(args.config)

    # print configuration
    if args.print_config:
        print_config({
            'BASE CONFIG': cfg.base,
            'DATA CONFIG': cfg.data,
            'TRAIN CONFIG': cfg.train
        })
    else:
        print_msg('LOADING CONFIG FILE: {}'.format(args.config))

    # create folder
    save_path = cfg.base.save_path
    if os.path.exists(save_path):
        if cfg.base.overwrite:
            print('Save path {} exists and will be overwrited.'.format(save_path))
        else:
            warning = 'Save path {} exists.\nDo you want to overwrite it? (y/n)\n'.format(save_path)
            if not input(warning) == 'y':
                sys.exit(0)
    else:
        os.makedirs(save_path)
    copy_config(args.config, cfg.base.save_path)

    worker(cfg)


def worker(cfg):
    if cfg.base.random_seed != -1:
        seed = cfg.base.random_seed
        set_random_seed(seed, cfg.base.cudnn_deterministic)

    logger = SummaryWriter(cfg.base.log_path)

    # train
    model = generate_model(cfg)
    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    estimator = Estimator(cfg.train.criterion, cfg.data.num_classes)
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
        logger=logger
    )

    # test
    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(cfg.base.save_path, 'best_validation_weights.pt')
    cfg.train.checkpoint = checkpoint
    model = generate_model(cfg)
    evaluate(cfg, model, test_dataset, estimator)

    print('This is the performance of the final model:')
    checkpoint = os.path.join(cfg.base.save_path, 'final_weights.pt')
    cfg.train.checkpoint = checkpoint
    model = generate_model(cfg)
    evaluate(cfg, model, test_dataset, estimator)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


if __name__ == '__main__':
    main()
