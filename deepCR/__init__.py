# -*- coding: utf-8 -*-
# @Author: https://github.com/profjsb/deepCR
# @Date:   2023-10-20 17:38:53
# @Last Modified by:   lshuns
# @Last Modified time: 2024-01-17 16:44:37

"""
deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

Apply a learned DL model to a 2d numpy array to remove
cosmic rays.

Code which accompanies the paper: Zhang & Bloom (2019)

See https://github.com/profjsb/deepCR

"""
from deepCR.model import deepCR
from deepCR.model_ssli import deepECR
from deepCR.training import train_mask, train_inpaint, train_inpaint_obj

__all__ = ["deepECR", "deepCR", "train_mask", "train_inpaint", "train_inpaint_obj"]

__version__ = '0.2.0.4'