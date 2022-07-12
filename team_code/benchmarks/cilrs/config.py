import os

class GlobalConfig:
	# Data
    seq_len = 1 # input timesteps
    pred_len = 0 # future waypoints predicted, not required for CILRS

    root_dir = '/home/kin/mmfn/data/new_mmfn'
    train_towns = ['Town01','Town03','Town04','Town05','Town06']
    val_towns = ['Town02','Town05']
    train_data, val_data = [], []
    for town in train_towns:
        train_data.append(os.path.join(root_dir, town))
    for town in val_towns:
        val_data.append(os.path.join(root_dir, town))

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    max_throttle = 0.75 # upper limit on throttle signal value in dataset

    lr = 1e-4 # learning rate

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
