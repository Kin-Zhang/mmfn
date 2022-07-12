import os

class GlobalConfig():
    """ base architecture configurations """
    # Data
    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras
    n_views = 1 # no. of camera views

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    lr = 1e-4 # learning rate

    # Conv Encoder
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors

    # GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # Controller
    turn_KP = 1.0
    turn_KI = 0.65
    turn_KD = 0.2
    turn_n = 30 # buffer size

    speed_KP = 4.0
    speed_KI = 0.4
    speed_KD = 0.8
    speed_n = 30 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.1 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller


    # GAT
    hidden = 81
    nb_heads = 2
    alpha = 0.2

    # vector map
    lane_node_num = 10
    feature_num = 5
    up = 28
    down = 28
    left = 28
    right = 28
    tmp_town_for_save_opendrive = '/tmp/opendrvie_tmp'
    def __init__(self, **kwargs):
        self.train_data, self.val_towns = [], []
        for k,v in kwargs.items():
            setattr(self, k, v)
        
    def data_folder(self, args):
        root_dir = os.path.join(args.absolute_path, args.data_folder)
        train_data, val_data = [], []
        train_towns = args.train_towns
        
        for town in train_towns:
            # train_data.append(os.path.join(root_dir, town+'_tiny'))
            train_data.append(os.path.join(root_dir, town+'_short'))

        val_towns = args.val_towns
        for town in val_towns:
            val_data.append(os.path.join(root_dir, town+'_short'))
            
        # for town in train_towns:
        #     train_data.append(os.path.join(root_dir, town+'_long'))

        self.train_data = train_data
        self.val_data = val_data

