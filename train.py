# Copyright 2023 - Valeo Comfort and Driving Assistance - Corentin Sautier @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code inspired by OpenPCDet.
# Credit goes to OpenMMLab: https://github.com/open-mmlab/OpenPCDet

import os
import tqdm
import torch
import argparse
import numpy as np
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime as dt
from tensorboardX import SummaryWriter

from utils.logger import make_logger
from bevlab.models import make_models
from bevlab.dataloader import make_dataloader
from utils.config import generate_config, log_config
from utils.optimizer import make_optimizer, make_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank % world_size)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size_per_gpu', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--lr', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--num_workers_per_gpu', type=int, default=None, help='number of workers for dataloader')
    parser.add_argument('--name', type=str, default='default', help='name of the experiment')
    parser.add_argument('--debug', action='store_true', default=False, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--resume_path', type=str, default=None, help='checkpoint to resume training from')
    # parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')

    args = parser.parse_args()

    config = generate_config(args.config_file)
    config.SAVE_FOLDER = Path('output', args.name, dt.today().strftime("%d%m%y-%H%M"))
    return args, config


def main(rank, world_size):
    multigpu = world_size > 1
    if multigpu:
        ddp_setup(rank, world_size)
    args, config = parse_config()

    if args.batch_size_per_gpu is not None:
        config.OPTIMIZATION.BATCH_SIZE_PER_GPU = args.batch_size_per_gpu
    if args.epochs is not None:
        config.OPTIMIZATION.NUM_EPOCHS = args.epochs
    if args.num_workers_per_gpu is not None:
        config.OPTIMIZATION.NUM_WORKERS_PER_GPU = args.num_workers_per_gpu
    if args.lr is not None:
        config.OPTIMIZATION.LR = args.lr
    config.DEBUG = args.debug

    config.LOCAL_RANK = rank

    # if args.fix_random_seed:
    #     # unfortunately as grid_sampler_2d_backward_cuda is non-deterministic, reproductibility isn't possible
    #     torch.use_deterministic_algorithms(True)
    #     torch.backends.cudnn.benchmark = False
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #     random.seed(0)
    #     np.random.seed(0)
    #     torch.manual_seed(0)

    ckpt_dir = config.SAVE_FOLDER / 'ckpt'
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=config.DEBUG)
    log_file = config.SAVE_FOLDER / 'log_train.txt'
    logger = make_logger(log_file, rank=rank)

    logger.info("==============Logging config==============")
    log_config(config, logger)

    logger.info('World size : %s' % world_size)

    if rank == 0:
        os.system('cp %s %s' % (args.config_file, config.SAVE_FOLDER))

    train_dataloader = make_dataloader(
        config=config,
        phase=config.DATASET.DATA_SPLIT['train'],
        world_size=world_size,
        rank=rank
    )

    model = make_models(config=config)
    if multigpu and not config.ENCODER.COLLATE == "collate_torchsparse":
        # sync batchnorm doesn't work with torchsparse
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = make_optimizer(model, config)

    model = model.to(rank)
    if multigpu:
        model = DDP(model, device_ids=[rank])
    # load checkpoint if it is possible
    if args.resume_path is not None:
        logger.warning(f"Continuing previous training: {args.resume_path}")
        load_dict = torch.load(args.resume_path)
        model.load_state_dict(load_dict['state_dict'])
        optimizer.load_state_dict(load_dict['optimizer'])

    scheduler = make_scheduler(
        config, total_iters=len(train_dataloader) * config.OPTIMIZATION.NUM_EPOCHS
    )

    train(
        model,
        train_dataloader,
        optimizer,
        scheduler=scheduler,
        config=config,
        rank=rank,
        multigpu=multigpu
    )

    if multigpu:
        dist.destroy_process_group()


def train(model, train_dataloader, optimizer, scheduler, config, rank, multigpu):
    debug = config.DEBUG
    if not debug:
        tb_log = SummaryWriter(log_dir=str(config.SAVE_FOLDER / 'tensorboard')) if rank == 0 else None

    train_iter = 0
    total_epochs = config.OPTIMIZATION.NUM_EPOCHS
    disp_dict = {}
    with tqdm.trange(0, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_dataloader)

        for cur_epoch in tbar:
            if multigpu:
                train_dataloader.sampler.set_epoch(cur_epoch)
            train_dataloader_iter = iter(train_dataloader)
            statistics = {"losses": []}

            if rank == 0:
                pbar = tqdm.tqdm(total=total_it_each_epoch, leave=False, desc='train', dynamic_ncols=True)

            model.train()

            for cur_it in range(len(train_dataloader)):
                cur_lr = scheduler[train_iter]
                batch = next(train_dataloader_iter)

                batch['voxels_in'] = batch['voxels_in'].to(rank, non_blocking=True)
                batch['coordinates_in'] = batch['coordinates_in'].to(rank, non_blocking=True)
                batch['voxels_out'] = batch['voxels_out'].to(rank, non_blocking=True)
                batch['coordinates_out'] = batch['coordinates_out'].to(rank, non_blocking=True)

                for param_group in optimizer.param_groups:
                    param_group["lr"] = cur_lr

                optimizer.zero_grad(set_to_none=True)

                loss, metrics = model(batch)
                statistics["losses"].append(loss.item())

                loss.backward()
                optimizer.step()

                # log to console and tensorboard
                if rank == 0:

                    pbar.update()
                    pbar.set_postfix(dict(total_it=train_iter, loss=loss.item(), **metrics))

                    if not debug:
                        tb_log.add_scalar('train/loss', loss.item(), train_iter)
                        for key, value in metrics.items():
                            tb_log.add_scalar(f'train/{key}', value, train_iter)
                        tb_log.add_scalar('meta_data/learning_rate', cur_lr, train_iter)

                del loss, metrics
                train_iter += 1

            if rank == 0:
                loss = np.mean(statistics['losses'])
                disp_dict.update({'loss': np.mean(statistics['losses'])})
                tbar.set_postfix(disp_dict)
                if not debug:
                    tb_log.add_scalar('epoch/loss', loss, cur_epoch)
                pbar.close()

                # save trained model
                if isinstance(model, DDP):
                    torch.save({
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iters": train_iter,
                        "config": config},
                        config.SAVE_FOLDER / 'ckpt' / 'model.pt')
                else:
                    torch.save({
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iters": train_iter,
                        "config": config},
                        config.SAVE_FOLDER / 'ckpt' / 'model.pt')


if __name__ == '__main__':
    multigpu = torch.cuda.device_count() > 1
    if multigpu:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size,), nprocs=world_size)
    else:
        main(0, 1)
