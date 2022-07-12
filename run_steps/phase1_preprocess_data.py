"""
# Copyright (c) 2021-2022 RAM-LAB
# Authors: kinzhang (qzhangcb@connect.ust.hk)
#          tmkk (mtangag@connect.ust.hk)
# Usage: Preprocess data for training

For **training**:
0. You don't have to launch carla
1. please prepare your data path and modify on config file
2. please remember to install library
3. This script is for preprocess data so that GPU can have higher util
"""

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import hydra
import gc, sys, os, time
from omegaconf import DictConfig
import pickle
from pathlib import Path

from mmfn_utils.datasets.config import GlobalConfig
from mmfn_utils.datasets.dataloader import CARLA_Data, PRE_Data
from utils import bcolors as bc

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

def preprocess_dataset_for(config, args):
    
    # Data
    train_set = CARLA_Data(root=config.train_data, config=config)
    val_set = CARLA_Data(root=config.val_data, config=config)
    
    train_dir = os.path.join(args.absolute_path, args.data_folder.split('/')[0],'pro_train_f'+args.data_folder.split('/')[1])
    val_dir = os.path.join(args.absolute_path, args.data_folder.split('/')[0],'pro_eval_f'+args.data_folder.split('/')[1])
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(train_set))):
        with open(f'{train_dir}/%d.pkl'%(i), 'wb') as fd:
            pickle.dump(train_set[i], fd)

    for i in tqdm(range(0, len(val_set))):
        with open(f'{val_dir}/%d.pkl'%(i), 'wb') as fd:
            pickle.dump(val_set[i], fd)

def check_data(config, stage = 0, data_set = None):
    if stage == 0:
        try:
            train_set = CARLA_Data(root=config.train_data, config=config)
            val_set = CARLA_Data(root=config.val_data, config=config)
            del train_set, val_set
        except Exception as e:
            print(e, "The data may have problem, please check whether the data is correct")
    else:
        if data_set==None:
            print('pls get the data set to read through')
            return
        for data in tqdm(data_set):
            try:
                x = data['velocity']
            except Exception as e:
                print(e, "The data may have problem===> after inside")

@hydra.main(config_path="config", config_name="train")
def main(args: DictConfig):
    # Config
    config = GlobalConfig()
    config.data_folder(args)
    
    # checking whether files is ok since measurements sometimes will have problem
    check_data(config)

    # preprocess_dataset_ray(config, args)
    preprocess_dataset_for(config, args)

    config.train_data = os.path.join(args.absolute_path, args.data_folder.split('/')[0],'pro_train_f'+args.data_folder.split('/')[1])
    config.val_data = os.path.join(args.absolute_path, args.data_folder.split('/')[0],'pro_eval_f'+args.data_folder.split('/')[1])
    train_set = PRE_Data(root=config.train_data, config=config, data_use='train')
    val_set = PRE_Data(root=config.val_data, config=config, data_use='val')

    # try to read data once to test whether it's great
    check_data(config, stage=1, data_set=train_set)

    
if __name__ == '__main__':
    start_time = time.time()
    try:
        main()
        print(f"{bc.OKGREEN}TOTAL DATASET RUNNING TIME{bc.ENDC}: --- %s mins --- Clean Memory %s" % (round((time.time() - start_time)/60,2), gc.collect()))
    
    except Exception as e:
        print("> {}\033[0m\n".format(e))
        print(bc.FAIL + "There is a error or close by users"+ bc.ENDC, 'clean memory', gc.collect())
        sys.exit()
    