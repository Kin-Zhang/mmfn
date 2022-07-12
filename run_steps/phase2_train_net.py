"""
# Copyright (c) 2021-2022 RAM-LAB
# Authors: kinzhang (qzhangcb@connect.ust.hk)
# Usage: Training

For ** training**:
0. You don't have to launch carla
1. please prepare your data path and modify on config file
2. please remember to install library
3. This script is for training data, please! run preprocess first!
"""

import argparse, random, json
import os, sys, gc
import wandb, hydra
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.benchmark = True

from mmfn_utils.datasets.config import GlobalConfig
from mmfn_utils.datasets.dataloader import CARLA_Data, PRE_Data
from mmfn_utils.datasets.data_utils import collate_single_cpu
from utils import bcolors as bc
from utils import load_entry_point, init_torch

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
afc = parser.parse_args()

class Engine(object):
    """Engine that runs training and inference.
    Args
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.
        
    """

    def __init__(self, device, log_dir, cur_epoch=0, cur_iter=0):
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.device = device
        self.logdir = log_dir

    def train(self, model, dataloader_train, config, optimizer):
        loss_epoch = 0.
        num_batches = 0
        model.train()
        total_num = len(dataloader_train)
        # Train loop
        for cur_num, data in enumerate(dataloader_train):
            
            # efficiently zero gradients
            for p in model.parameters():
                p.grad = None
            
            # create batch and move to GPU
            fronts_in = data['fronts']
            lidars_in = data['lidars']
            vectormaps_in = data['vectormaps']
            maps_in = data['maps']
            radar_in = data['radar']

            fronts, lidars, maps = [], [], []
            vectormaps_lane = []
            vectormaps_lane_num = []
            radar = []
            radar_adj = []
            for i in range(config.seq_len):
                # camera
                fronts.append(fronts_in[i].to(self.device, dtype=torch.float32))
                # lidar
                lidars.append(lidars_in[i].to(self.device, dtype=torch.float32))
                # img map
                maps.append(maps_in[i].to(self.device, dtype=torch.float32))
                # vectormap
                vectormaps_lane.append(vectormaps_in[i][0].to(self.device, dtype=torch.float32))
                vectormaps_lane_num.append(vectormaps_in[i][1].to(self.device, dtype=torch.float32))
                vectormaps = [vectormaps_lane, vectormaps_lane_num, vectormaps_in[0][2]]
                # radar
                radar.append(radar_in[i].to(self.device, dtype=torch.float32))
                radar_adj.append(data['radar_adj'].to(self.device, dtype=torch.float32))

            # driving labels
            gt_velocity = data['velocity'].to(self.device, dtype=torch.float32)

            # target point
            target_point = torch.stack(data['target_point'], dim=1).to(self.device, dtype=torch.float32)
            
            # model input
            pred_wp = model(fronts, lidars, maps, vectormaps, radar, radar_adj, target_point, gt_velocity)

            gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(self.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(self.device, dtype=torch.float32)
            loss = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()

            # backward
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            loss_epoch += float(loss.item())
            optimizer.step()

            # log
            if afc.local_rank == 0:
                wandb.log({"loss": loss.item(),"iter": self.cur_iter})
                print(f"Epoch {self.cur_epoch+1} progress: {cur_num:d}/{total_num:d}", end="\r")
            
            self.cur_iter += 1
            num_batches += 1
        
        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self, model, dataloader_val, config):
        model.eval()

        with torch.no_grad():    
            num_batches = 0
            wp_epoch = 0.

            # Validation loop
            for batch_num, data in enumerate(dataloader_val, 0):
                # create batch and move to GPU
                fronts_in = data['fronts']
                lidars_in = data['lidars']
                maps_in = data['maps']
                vectormaps_in = data['vectormaps']

                maps_in = data['maps']
                radar_in = data['radar']
                fronts, maps = [], []
                lidars = []
                vectormaps_lane = []
                vectormaps_lane_num = []
                radar = []
                radar_adj = []
                for i in range(config.seq_len):
                    # camera
                    fronts.append(fronts_in[i].to(self.device, dtype=torch.float32))
                    # lidar
                    lidars.append(lidars_in[i].to(self.device, dtype=torch.float32))
                    # img map
                    maps.append(maps_in[i].to(self.device, dtype=torch.float32))
                    # vectormap
                    vectormaps_lane.append(vectormaps_in[i][0].to(self.device, dtype=torch.float32))
                    vectormaps_lane_num.append(vectormaps_in[i][1].to(self.device, dtype=torch.float32))
                    vectormaps = [vectormaps_lane, vectormaps_lane_num, vectormaps_in[0][2]]
                    # radar
                    radar.append(radar_in[i].to(self.device, dtype=torch.float32))
                    radar_adj.append(data['radar_adj'].to(self.device, dtype=torch.float32))

                # driving labels
                gt_velocity = data['velocity'].to(self.device, dtype=torch.float32)

                # target point
                target_point = torch.stack(data['target_point'], dim=1).to(self.device, dtype=torch.float32)

                pred_wp = model(fronts, lidars, maps, vectormaps, radar, radar_adj, target_point, gt_velocity)

                gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(self.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
                gt_waypoints = torch.stack(gt_waypoints, dim=1).to(self.device, dtype=torch.float32)
                wp_epoch += float(F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean())

                num_batches += 1
            
            if num_batches != 0:
                wp_loss = wp_epoch / float(num_batches)
                if afc.local_rank == 0:
                    print(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')
                    wandb.log({"val loss": wp_loss,"iter": self.cur_epoch})
                
                self.val_loss.append(wp_loss)

    def save(self, args, model, optimizer):

        save_best = False
        if self.val_loss[-1] <= self.bestval:
            self.bestval = self.val_loss[-1]
            self.bestval_epoch = self.cur_epoch
            save_best = True
        
        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
            'bestval': self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }

        # Log other data corresponding to the recent model
        with open(os.path.join(self.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))
        
        if save_best:
            if args.is_multi_gpu:
                torch.save(model.module.state_dict(), os.path.join(self.logdir, 'best_model.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(self.logdir, 'best_model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(self.logdir, 'best_optim.pth'))
            print('====== Overwrote best model on epoch %d======>'%self.cur_epoch)
        
        print('====== Saved recent model with recent log ======>')
        # Save the recent model/optimizer states
        torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))
        # Log other data corresponding to the recent model
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

from omegaconf import DictConfig
@hydra.main(config_path="config", config_name="train")
def main(args: DictConfig):
    n_gpu = torch.cuda.device_count()
    if n_gpu>1 and args.is_multi_gpu:
        torch.distributed.init_process_group(backend='nccl')
    else:
        args.is_multi_gpu = False
    assert n_gpu, "Can't find any GPU device on this machine."

    # 从local_rank 获得gpu
    afc.device = torch.device('cuda', afc.local_rank)
    print(afc.device)
    init_torch()

    # Config
    config = GlobalConfig()
    config.data_folder(args)

    args.logdir = os.path.join(args.absolute_path, args.logdir)
    config.train_data = os.path.join(args.absolute_path, args.data_folder.split('/')[0],'pro_train_f'+args.data_folder.split('/')[1])
    config.val_data = os.path.join(args.absolute_path, args.data_folder.split('/')[0],'pro_eval_f'+args.data_folder.split('/')[1])


    # Data
    train_set = PRE_Data(root=config.train_data, config=config, data_use='train')
    val_set = PRE_Data(root=config.val_data, config=config, data_use='val')
    if afc.local_rank == 0:
        print(f"{bc.OKGREEN} Data Path:  {str(config.train_data)} {bc.ENDC}, {bc.OKCYAN}total train frame: {len(train_set)} {bc.ENDC}")

    # Model ===> will change through train.yaml config file
    MMFN = load_entry_point(args.train_agent.entry_point)
    model = MMFN(config, afc.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    trainer = Engine(afc.device, args.logdir)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # 多GPU训练
    if args.is_multi_gpu and n_gpu>1 and afc.local_rank == 0:
        print(f"{bc.OKGREEN} OPEN THE MULTIP GPU MODE DDP {bc.ENDC}")
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
        dataloader_train = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, collate_fn = collate_single_cpu)
        dataloader_val = DataLoader(val_set, batch_size=args.batch_size,  sampler=val_sampler, num_workers=4, pin_memory=True, collate_fn = collate_single_cpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[afc.local_rank], output_device=afc.local_rank, find_unused_parameters=True)
    else:
        print(f"{bc.OKGREEN} SINGLE GPU MODE {bc.ENDC}")
        dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn = collate_single_cpu) 
        dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn = collate_single_cpu)

    torch.cuda.set_device(afc.local_rank)
    torch.cuda.empty_cache()


    
    if afc.local_rank == 0:
        print ('Total trainable parameters: ', params)

        # loading
        if not os.path.isdir(args.logdir):
            os.makedirs(args.logdir)
            print('Created dir:', args.logdir)

        elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
            print('Loading checkpoint from ' + args.logdir)
            with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
                log_table = json.load(f)

            # Load variables
            trainer.cur_epoch = log_table['epoch']
            if 'iter' in log_table: trainer.cur_iter = log_table['iter']
            print('cur epoch', trainer.cur_epoch, 'future epoch', args.epochs)
            trainer.bestval = log_table['bestval']
            trainer.train_loss = log_table['train_loss']
            trainer.val_loss = log_table['val_loss']

            optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'best_optim.pth')))
            model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model.pth')))

        wandb.init(project="best_data", entity="mmfn", config = args, mode=args.wandb_mode)
        wandb.watch(model)

    for epoch in range(trainer.cur_epoch, args.epochs):
        epoch_t_start = time.time()
        trainer.train(model, dataloader_train, config, optimizer)
        # log
        if afc.local_rank == 0:
            print("--------------------------------------------------------")
            print("Epoch: ", epoch+1)
            print("  --Train avg loss: %.4f" % trainer.train_loss[-1])
            print("  --Epoch time    : %.2f mins" % ((time.time() - epoch_t_start)/60))
            print("--------------------------------------------------------")
            wandb.log({"spend time": ((time.time() - epoch_t_start)/60),"epoch": epoch})
        
        if epoch % args.val_every == 0 and afc.local_rank == 0: 
            trainer.validate(model, dataloader_val, config)
            if epoch % args.save_every == 0:
                trainer.save(args, model, optimizer)
                
    if afc.local_rank == 0:
        wandb.save(os.path.join(args.logdir, 'best_model.pth'))
        wandb.save(os.path.join(args.logdir, 'recent.log'))
    

if __name__ == '__main__':
    import time
    start_time = time.time()
    main()

    if afc.local_rank == 0:
        print(f"{bc.OKGREEN}TOTAL RUNNING TIME{bc.ENDC}: --- %s hours --- Clean Memory %s" % (round((time.time() - start_time)/3600,2), gc.collect()))
        wandb.finish()