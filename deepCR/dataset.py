# -*- coding: utf-8 -*-
# @Author: lshuns
# @Date:   2023-10-21 14:28:56
# @Last Modified by:   lshuns
# @Last Modified time: 2024-01-17 16:27:58
import os
import numpy as np

import torch
from torch.utils.data import Dataset

__all__ = ['dataset', 'dataset_inpaint', 'dataset_inpaint_obj']

class dataset(Dataset):
    def __init__(self, data_dir, image_file_list, mask_file_list, ignore_file_list=None, 
                    sky_val_list=None, aug_sky=[0, 0], gpu=False):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param data_dir: path where all images, masks and ignores are saved.
        :param image_file_list: A list of file names for images (file in .npy)
        :param mask_file_list: A list of file names for masks (file in .npy)
        :param ignore_file_list: A list of file names for loss masks, e.g., bad pixel, saturation, etc.
        :param sky_val_list: (np.ndarray) [N,] sky background level
        :param aug_sky: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        """

        self.data_dir = data_dir
        self.image_file_list = image_file_list
        self.mask_file_list = mask_file_list
        self.ignore_file_list = ignore_file_list

        self.sky_val_list = sky_val_list
        self.aug_sky = aug_sky

        ## determine which device to use
        if gpu:
            self.device = "cuda"
            self.dtype = torch.cuda.FloatTensor
        else:
            self.device = "cpu"
            self.dtype = torch.FloatTensor

    def __len__(self):
        # assume each path contains one image
        return len(self.image_file_list)

    def __getitem__(self, i):

        # get image
        image_path = os.path.join(self.data_dir, self.image_file_list[i])
        image = np.load(image_path)

        shape_tmp = image.shape[-1]

        # get mask
        mask_path = os.path.join(self.data_dir, self.mask_file_list[i])
        mask = np.load(mask_path)

        # add sky
        if self.sky_val_list is not None:
            image = image + (self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])) * self.sky_val_list[i]

        # the ignore map
        if self.ignore_file_list is not None:
            ignore_path = os.path.join(self.data_dir, self.ignore_file_list[i])
            ignore = np.load(ignore_path)

            # expanding the dimension for convolution
            return torch.from_numpy(image).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(mask).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(ignore).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp)
        else:
            return torch.from_numpy(image).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(mask).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp)

class dataset_inpaint(Dataset):
    def __init__(self, data_dir, image_file_list, mask_file_list, inpaint_file_list, ignore_file_list=None, 
                    sky_val_list=None, aug_sky=[0, 0], gpu=False, scale=1):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param data_dir: path where all images, masks and ignores are saved.
        :param image_file_list: A list of file names for images (file in .npy)
        :param mask_file_list: A list of file names for masks (file in .npy)
        :param inpaint_file_list: A list of file names for inpainted images (file in .npy)
        :param ignore_file_list: A list of file names for loss masks, e.g., bad pixel, saturation, etc.
        :param sky_val_list: (np.ndarray) [N,] sky background level
        :param aug_sky: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        """

        self.data_dir = data_dir
        self.image_file_list = image_file_list
        self.mask_file_list = mask_file_list
        self.inpaint_file_list = inpaint_file_list
        self.ignore_file_list = ignore_file_list

        self.sky_val_list = sky_val_list
        self.aug_sky = aug_sky
        self.scale = scale

        ## determine which device to use
        if gpu:
            self.device = "cuda"
            self.dtype = torch.cuda.FloatTensor
        else:
            self.device = "cpu"
            self.dtype = torch.FloatTensor

    def __len__(self):
        # assume each path contains one image
        return len(self.image_file_list)

    def __getitem__(self, i):

        # get image
        image_path = os.path.join(self.data_dir, self.image_file_list[i])
        image = np.load(image_path)

        shape_tmp = image.shape[-1]

        # get mask
        mask_path = os.path.join(self.data_dir, self.mask_file_list[i])
        mask = np.load(mask_path)

        # get inpaint
        inpaint_path = os.path.join(self.data_dir, self.inpaint_file_list[i])
        inpaint = np.load(inpaint_path)

        # add sky
        if self.sky_val_list is not None:
            sky_tmp = (self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])) * self.sky_val_list[i]
            image = image + sky_tmp
            inpaint = inpaint + sky_tmp

        # the ignore map
        if self.ignore_file_list is not None:
            ignore_path = os.path.join(self.data_dir, self.ignore_file_list[i])
            ignore = np.load(ignore_path)

            # expanding the dimension for convolution
            return torch.from_numpy(image/self.scale).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(mask).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(inpaint/self.scale).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(ignore).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp)
        else:
            return torch.from_numpy(image/self.scale).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(mask).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(inpaint/self.scale).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp)        
        

class dataset_inpaint_obj(Dataset):
    def __init__(self, data_dir, image_file_list, CRmask_file_list, 
                 clean_file_list, OBJmask_file_list,
                 ignore_file_list=None, sky_val_list=None, aug_sky=[0, 0], gpu=False, scale=1):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param data_dir: path where all images, masks and ignores are saved.
        :param image_file_list: A list of file names for images (file in .npy)
        :param CRmask_file_list: A list of file names for CR masks (file in .npy)
        :param clean_file_list: A list of file names for clean images (file in .npy)
        :param OBJmask_file_list: A list of file names for clean object masks (file in .npy)
        :param ignore_file_list: A list of file names for loss masks, e.g., bad pixel, saturation, etc.
        :param sky_val_list: (np.ndarray) [N,] sky background level
        :param aug_sky: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        """

        self.data_dir = data_dir
        self.image_file_list = image_file_list
        self.CRmask_file_list = CRmask_file_list
        self.clean_file_list = clean_file_list
        self.OBJmask_file_list = OBJmask_file_list
        self.ignore_file_list = ignore_file_list

        self.sky_val_list = sky_val_list
        self.aug_sky = aug_sky
        self.scale = scale

        ## determine which device to use
        if gpu:
            self.device = "cuda"
            self.dtype = torch.cuda.FloatTensor
        else:
            self.device = "cpu"
            self.dtype = torch.FloatTensor

    def __len__(self):
        # assume each path contains one image
        return len(self.image_file_list)

    def __getitem__(self, i):

        # get image
        image = np.load(os.path.join(self.data_dir, self.image_file_list[i]))

        shape_tmp = image.shape[-1]

        # get CR mask
        CRmask = np.load(os.path.join(self.data_dir, self.CRmask_file_list[i]))

        # get clean image
        clean_image = np.load(os.path.join(self.data_dir, self.clean_file_list[i]))

        # get object mask
        OBJmask = np.load(os.path.join(self.data_dir, self.OBJmask_file_list[i]))

        # add sky
        if self.sky_val_list is not None:
            sky_tmp = (self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])) * self.sky_val_list[i]
            image = image + sky_tmp
            clean_image = clean_image + sky_tmp

        # the ignore map
        if self.ignore_file_list is not None:
            ignore_path = os.path.join(self.data_dir, self.ignore_file_list[i])
            ignore = np.load(ignore_path)

            # expanding the dimension for convolution
            return torch.from_numpy(image/self.scale).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(CRmask).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(clean_image/self.scale).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(OBJmask).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(ignore).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp)
        else:
            return torch.from_numpy(image/self.scale).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(CRmask).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(clean_image/self.scale).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp), \
                    torch.from_numpy(OBJmask).to(self.device).type(self.dtype).view(1, shape_tmp, shape_tmp)