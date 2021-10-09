#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 19:05

import torch

def move2cpu(data_lists):
    data_lists = [data.to(torch.device('cpu')) for data in data_lists]
    return data_lists