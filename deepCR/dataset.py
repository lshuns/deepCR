import os
import numpy as np

import torch
from torch.utils.data import Dataset

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
        