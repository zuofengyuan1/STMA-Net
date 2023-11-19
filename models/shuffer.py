# -*- coding: utf-8 -*-
# -----------------------------------------------------
# Time :  2023/7/27 14:54
# Auth :  Written by zuofengyuan
# File :  shuffer.py
# Copyright (c) Shenyang Pedlin Technolofy Co., Ltd.
# -----------------------------------------------------
"""
 Description: TODO
"""
import torch
import torch.nn as nn

class Channel_Shuffle(nn.Module):
    def __init__(self, num_groups):
        super(Channel_Shuffle, self).__init__()
        self.num_groups = num_groups
    def forward(self, x: torch.FloatTensor):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = torch.reshape(x, (batch_size, self.num_groups, chs_per_group, h, w))
         # (batch_size, num_groups, chs_per_group, h, w)
        x = x.transpose(1, 2)  # dim_1 and dim_2
        out = torch.reshape(x, (batch_size, -1, h, w))
        return out

tensor = torch.rand((1, 6, 5, 5))
print(tensor)
cs = Channel_Shuffle(2)
print(cs(tensor).shape)
