"""
# Copyright (c) 2021-2022 RAM-LAB
# Authors: kinzhang (qzhangcb@connect.ust.hk)
# Usage: Training

For **training**:
0. You don't have to launch carla
1. please prepare your data path and modify on config file
2. please remember to install library
3. This script is for training data, please! run preprocess first!
"""

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets.config import GlobalConfig
from datasets.dataloader import CARLA_Data, PRE_Data
import hydra
import gc, sys, os

