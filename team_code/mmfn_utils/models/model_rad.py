import math
from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange

class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    # 这里应该是-均值/方差
    # 但是这个好像 emmm 直接给出了均值和方差？ 咋给的？
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=2):
        super().__init__()

        self._model = models.resnet18()
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)

    def forward(self, inputs):
        features = 0
        for lidar_data in inputs:
            lidar_feature = self._model(lidar_data)
            features += lidar_feature

        return features


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar + map
        self.pos_emb = nn.Parameter(torch.zeros(1, (self.config.n_views + 2) * seq_len * vert_anchors * horz_anchors, n_embd))
        
        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor, map_tensor,velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            map_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        
        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]
        
        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        map_tensor = map_tensor.view(bz, self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor, map_tensor], dim=1).permute(0,1,3,4,2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)

        # project velocity to n_embed
        velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        x = x.view(bz, (self.config.n_views + 2) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings

        image_tensor_out = x[:, :self.config.n_views*self.seq_len, :, :, :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.config.n_views*self.seq_len:(self.config.n_views+1)*self.seq_len, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        map_tensor_out   = x[:, (self.config.n_views+1)*self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        
        return image_tensor_out, lidar_tensor_out, map_tensor_out

class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.LayerNorm(out_channels), nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x

class Subgraph(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(Subgraph, self).__init__()
        self.num_subgraph_layers = num_layers
        self.layers = nn.Sequential()

        for i in range(num_layers):
            self.layers.add_module(f"mlp_{i}", MLP(in_channels, out_channels))
            in_channels = out_channels * 2

    def forward(self, x: torch.Tensor):
        """
        Args:
            lanes (torch.Tensor): [B, obj_num, num_vectors, d]
        """

        for layer in self.layers:
            x = layer(x)
            max_pool, _ = torch.max(x, dim=-2)
            max_pool = max_pool.unsqueeze(dim=2).repeat(1, 1, x.shape[2], 1)
            x = torch.cat([x, max_pool], dim=-1)

        x, _ = torch.max(x, dim=-2)
        return x

class MaskSelfAttention(nn.Module):
    """
    Efficient Multihead self-attention
    """

    def __init__(self, dim: int, heads=1, dropout=0.0):
        super(MaskSelfAttention, self).__init__()
        assert dim % heads == 0

        self.dim_head = dim // heads
        self.heads = heads
        self.scale = self.dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (Tensor): input tokens, [b, n, d]
            mask (Tensor, optional): [b, n], ignore token with mask=0
        """
        # [b, n, h*d] * 3
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # [b, h, n, d]
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        # [b, h, n, d] * [b, h, d, n] -> [b, h, n, n]
        dots: torch.Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            assert mask.shape[0] == dots.shape[0]
            mask = mask.unsqueeze(1)
            dots = dots.masked_fill(mask == 0, value=-1e9)

        attn = self.attend(dots)

        x = torch.matmul(attn, v)
        x = rearrange(x, "b h n d -> b n (h d)")

        return (self.to_out(x), attn)

class VectornetEncoder(nn.Module):
    def __init__(self, lane_channels, hidden_size, subgraph_layers,pos_dim,heads,fusion_dim):
        super(VectornetEncoder, self).__init__()
        self.lane_subgraph = Subgraph(
            lane_channels, hidden_size, subgraph_layers
        )
        # pos embedding
        self.pos_emb = nn.Sequential(
            nn.Linear(2, pos_dim, bias=True),
            nn.LayerNorm(pos_dim),
            nn.GELU(),
            nn.Linear(pos_dim, pos_dim, bias=True),
        )
        self.L2L = MaskSelfAttention(hidden_size * 2, heads)
        # agent fusion
        self.agent_fusion = nn.Sequential(
            nn.Linear(
                pos_dim + hidden_size * 2, fusion_dim, bias=True
            ),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, hidden_size * 2, bias=True),
        )

        self.generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 64*64*64, bias=True),
        )

        
    def _get_mask(self, lane_num: torch.Tensor, max_lane_num:int):
        lane_num= lane_num.type(torch.int)
        lane_mask = torch.zeros(self.batch, 1, max_lane_num).to(
            lane_num.device
        )
        
        for i in range(self.batch):
            lane_mask[i, 0, : lane_num[i]] = 1
        return lane_mask

    def _lane_to_vector(self, lane: torch.Tensor):
        """ Convert lane to vector form

        Args:
            lane (Tensor): [b, max_lane_num, 10, 5] 
                           (5 for [x, y, intersection, turn_dir, traffic_control])

        Returns:
            lane_vec (Tensor): [b, max_lane_num, hidden]
        """
        lane_vec = torch.cat(
            [lane[:, :, :-1, 0:2], lane[:, :, 1:, 0:2], lane[:, :, 1:, 2:]], dim=-1,
        ).type(torch.float32)
        return self.lane_subgraph(lane_vec)

    def forward(self, data: torch.Tensor):
        """
        Param: 
            data is a list with two element
              data[0] is a seq_len list, each element is a tensor. [b, max_lane_num, 10, 5]
                (10 is the max length of a lane, 5 for [x, y, intersection, turn_dir, traffic_control])
              data[1] is a seq_len list, each element is a tensor. [b]. 
                (each represent the lane_num in the tensor in data[0], 
                 For example, we use **x** to represent The i^th element in data[1] and j^th element in data[1][i]
                 it means in the i^th element in data[0], in the batch j, the number of lane is **x**)
        """

        #Here we haven't use the sequence information of vectormap. just use data[0][0] and data[0][1]
        lane = data[0][0]
        lane_num = data[1][0]
        max_lane_num = data[2]

        self.batch = lane.shape[0]

        lane_token = self._lane_to_vector(lane)

        lane_mask = self._get_mask(lane_num, max_lane_num)

        lane_token, _ = self.L2L(lane_token, lane_mask)

        pos_emb = self.pos_emb(torch.zeros((lane_token.shape[0],lane_token.shape[1], 2)).to(lane_num.device))
        agent_token_fuse = self.agent_fusion(torch.cat([lane_token, pos_emb], dim=-1))


        output = self.generator(agent_token_fuse[:, 0, :].squeeze(1))  # 0 is target agent

        output = rearrange(output, "b (n d a) -> b n d a", d=64, a=64)

        return output

class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        
        self.image_encoder   = ImageCNN(512, normalize=True)
        self.img_map_encoder = ImageCNN(512)
        self.lidar_encoder   = LidarEncoder(num_classes=512, in_channels=2)
        
        self.vectornet_encoder = VectornetEncoder(
            lane_channels = 7, 
            hidden_size = 64, 
            subgraph_layers = 3,
            pos_dim = 64,
            heads = 2,
            fusion_dim = 128
        )
        
        # 32768 = 8x8x512, 5 = number of features
        self.radar_encoder = SpGAT(nfeat = 5, nhid=config.hidden, dropout=config.attn_pdrop, nheads=config.nb_heads, alpha=config.alpha)

        self.transformer1 = GPT(n_embd=64,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer2 = GPT(n_embd=128,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer3 = GPT(n_embd=256,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer4 = RadarGPT(n_embd=512,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)

        
    def forward(self, image_list, lidar_list, vectormaps, radar_list, radar_adj_list, velocity):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            maps_list (list):  list of map input images
            velocity (tensor): input velocity from speedometer
        '''
        if self.image_encoder.normalize:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]

        bz, _, h, w = lidar_list[0].shape
        img_channel = 3
        lidar_channel = lidar_list[0].shape[1]
        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
        lidar_tensor = torch.stack(lidar_list, dim=1).view(bz * self.config.seq_len, lidar_channel, h, w)

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.relu(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)


        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.relu(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        image_features = self.image_encoder.features.layer1(image_features)

        lidar_features = self.lidar_encoder._model.layer1(lidar_features)

        map_features = self.vectornet_encoder(vectormaps)

        # fusion at (B, 64, 64, 64) =========================================
        image_embd_layer1 = self.avgpool(image_features)
        map_embd_layer1   = self.avgpool(map_features)
        lidar_embd_layer1 = self.avgpool(lidar_features)
        image_features_layer1, lidar_features_layer1, map_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, map_embd_layer1, velocity)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=8, mode='bilinear', align_corners=True)
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=8, mode='bilinear', align_corners=True)
        map_features_layer1   = F.interpolate(map_features_layer1,   scale_factor=8, mode='bilinear', align_corners=True)
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1
        map_features   = map_features + map_features_layer1


        image_features = self.image_encoder.features.layer2(image_features)
        map_features   = self.img_map_encoder.features.layer2(map_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        

        # fusion at (B, 128, 32, 32) =========================================
        image_embd_layer2 = self.avgpool(image_features)
        map_embd_layer2   = self.avgpool(map_features)
        lidar_embd_layer2 = self.avgpool(lidar_features)
        image_features_layer2, lidar_features_layer2, map_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, map_embd_layer2, velocity)
        
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=4, mode='bilinear', align_corners=True)
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=4, mode='bilinear', align_corners=True)
        map_features_layer2   = F.interpolate(map_features_layer2,   scale_factor=4, mode='bilinear', align_corners=True)
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2
        map_features   = map_features   + map_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        map_features   = self.img_map_encoder.features.layer3(map_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        # fusion at (B, 256, 16, 16) =========================================
        image_embd_layer3 = self.avgpool(image_features)
        map_embd_layer3   = self.avgpool(map_features)
        lidar_embd_layer3 = self.avgpool(lidar_features)
        image_features_layer3, lidar_features_layer3, map_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, map_embd_layer3, velocity)
        
        image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=2, mode='bilinear', align_corners=True)
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, scale_factor=2, mode='bilinear', align_corners=True)
        map_features_layer3   = F.interpolate(map_features_layer3,   scale_factor=2, mode='bilinear', align_corners=True)
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3
        map_features   = map_features + map_features_layer3


        image_features = self.image_encoder.features.layer4(image_features)
        map_features   = self.img_map_encoder.features.layer4(map_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        # fusion at (B, 512, 8, 8) =========================================
        image_embd_layer4 = self.avgpool(image_features)
        map_embd_layer4   = self.avgpool(map_features)
        lidar_embd_layer4 = self.avgpool(lidar_features)

        radar_tensor = torch.stack(radar_list, dim=1)
        radar_tensor = torch.stack(radar_list, dim=1).view(bz * self.config.seq_len, 81, 5)
        radar_features = self.radar_encoder(radar_tensor, radar_adj_list[0])

        image_features_layer4, lidar_features_layer4, map_features_layer4, radar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, map_embd_layer4, radar_features, velocity)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4
        map_features   = map_features +   map_features_layer4
        radar_features = radar_features + radar_features_layer4

        image_features = self.image_encoder.features.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)
        lidar_features = self.lidar_encoder._model.avgpool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = lidar_features.view(bz, self.config.seq_len, -1)
        map_features = self.img_map_encoder.features.avgpool(map_features)
        map_features = torch.flatten(map_features, 1)
        map_features = map_features.view(bz, self.config.seq_len, -1)
        radar_features = self.radar_encoder.avgpool(radar_features)
        radar_features = torch.flatten(radar_features, 1)
        radar_features = radar_features.view(bz, self.config.seq_len, -1)

        fused_features = torch.cat([image_features, lidar_features, map_features, radar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)

        return fused_features


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class MMFN(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len

        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.encoder = Encoder(config).to(self.device)

        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)
        self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        self.output = nn.Linear(64, 2).to(self.device)
        
    def forward(self, image_list, lidar_list, map_list, vectormaps_list, radar_list, radar_adj, target_point, velocity):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            maps_list  (list): list of input opendrive map birdview
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
        '''
        fused_features = self.encoder(image_list, lidar_list, vectormaps_list, radar_list, radar_adj, velocity)
        z = self.join(fused_features)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            # x_in = torch.cat([x, target_point], dim=1)
            x_in = x + target_point
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp

    def control_pid(self, waypoints, velocity):
        ''' 
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if(speed < 0.01):
            angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata


# for GAT Layers
# reference: https://github.com/Diego999/pyGAT

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, 2*out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, h, adj):
        """
        :param h: (batch_zize, number_nodes, in_features)
        :param adj: (batch_size, number_nodes, number_nodes)
        :return: (batch_zize, number_nodes, out_features)
        """
        # batchwise matrix multiplication
        # h = h.view(h.shape[0], 5,81)
        Wh = torch.matmul(h, self.W)  # (B, N, in_features) * (in_features, out_features) -> (B, N, out_features)
        e = self.prepare_batch(Wh)  # (B, N, N)

        # (B, N, N)
        zero_vec = -9e15 * torch.ones_like(e)

        # (B, N, N)
        attention = torch.where(adj > 0, e, zero_vec)

        # (B, N, N)
        attention = F.softmax(attention, dim=-1)

        # (B, N, N)
        attention = F.dropout(attention, p = self.dropout, training=self.training)

        # batched matrix multiplication (B, N, out_features)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def prepare_batch(self, Wh):
        """
        with batch training
        :param Wh: (batch_zize, number_nodes, out_features)
        :return:
        """
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)

        B, N, E = Wh.shape  # (B, N, N)

        # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh1 = torch.matmul(Wh, self.a)  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)

        return self.leakyrelu(Wh1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.mlp_1 = nn.Sequential(
            nn.Linear(nheads*nhid, 256),
            nn.Dropout(self.dropout),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(nheads*nhid, 128),
            nn.Dropout(self.dropout),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp_1(F.elu(x))
        x = self.mlp_2(x.transpose(1,2))
        x = x.view(x.shape[0], 8, 8, 512).transpose(1, 3)
        return F.log_softmax(x, dim=1)


class RadarGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar + map + radar
        self.pos_emb = nn.Parameter(torch.zeros(1, (self.config.n_views + 3) * seq_len * vert_anchors * horz_anchors, n_embd))
        
        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor, map_tensor, radar_tensor, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            map_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        
        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]
        
        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        map_tensor = map_tensor.view(bz, self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)
        radar_tensor = radar_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor, map_tensor, radar_tensor], dim=1).permute(0,1,3,4,2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)

        # project velocity to n_embed
        velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        x = x.view(bz, (self.config.n_views + 3) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings

        image_tensor_out   = x[:, :self.config.n_views*self.seq_len, :, :, :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor_out   = x[:, self.config.n_views*self.seq_len:(self.config.n_views+1)*self.seq_len, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        map_tensor_out     = x[:, (self.config.n_views+1)*self.seq_len:(self.config.n_views+2)*self.seq_len, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        radar_tensor_out   = x[:, (self.config.n_views+2)*self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)

        return image_tensor_out, lidar_tensor_out, map_tensor_out, radar_tensor_out