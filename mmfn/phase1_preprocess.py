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
from datasets.config import GlobalConfig
from datasets.dataloader import CARLA_Data, PRE_Data
import hydra
import gc, sys, os

