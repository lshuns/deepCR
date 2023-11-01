"""main module to instantiate deepCR models and use them
"""

import torch
import torch.nn as nn
from torch import from_numpy

import numpy as np

from deepCR.unet_ssli import UNet

__all__ = ['deepECR']

class deepECR():

    def __init__(self, mask_model_path, 
                mask_nChannel_in=1, mask_nChannel_out=1, mask_nChannel_hidden=32, 
                mask_nLayers_down=1, mask_return_type='sigmoid',
                inpaint_model_path=None,
                inpaint_nChannel_in=2, inpaint_nChannel_out=1, inpaint_nChannel_hidden=32, 
                inpaint_nLayers_down=2, inpaint_return_type='ori',
                gpu=False, scale=1):

        """
            Instantiation of deepECR with specified model configurations

        Parameters
        ----------
        mask_model_path : str
            The file path to the trained model for mask (incl. '.pth')
        inpaint_model_path : (optional) str
            The file path to the trained model for inpaint (incl. '.pth')
        gpu : bool
            Use GPU or not
        scale : float, optional 
            Scaling the input image as img0/scale.

        Returns
        -------
        None
            """

        if gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f">>> Using {self.device} for training")

        # load the learned model
        state_dict = torch.load(mask_model_path, map_location=self.device)
        # initialise the network
        self.maskNet = UNet(mask_nChannel_in, mask_nChannel_out, mask_nChannel_hidden, 
                            mask_nLayers_down, mask_return_type).to(self.device)
        # set states
        self.maskNet.load_state_dict(state_dict)
        del state_dict
        # evaluation mode
        self.maskNet.eval()

        # for the inpaint
        if inpaint_model_path is not None:
            # load the learned model
            state_dict = torch.load(inpaint_model_path, map_location=self.device)
            # initialise the network
            self.inpaintNet = UNet(inpaint_nChannel_in, inpaint_nChannel_out, inpaint_nChannel_hidden, 
                                inpaint_nLayers_down, inpaint_return_type).to(self.device)
            # set states
            self.inpaintNet.load_state_dict(state_dict)
            # evaluation mode
            self.inpaintNet.eval()

        else:
            self.inpaintNet = None

        self.scale = scale

    def PredMask(self, img0, threshold=0.5, binary=True):
        """
            Identify cosmic rays in an input image, return the predicted cosmic ray mask.
        :param img0: (np.ndarray) 2D input image conforming to model requirements. For HST ACS/WFC, must be from
        _flc.fits and in units of electrons in native resolution.
        :param threshold: (float; [0, 1]) applied to probabilistic mask to generate binary mask
        :param binary: return binary CR mask if True. probabilistic mask if False
        :return: CR mask.
        """

        # to reduce unnecessary gradient computations
        with torch.no_grad():

            # to Tensor
            img0 = from_numpy(np.expand_dims(img0 / self.scale, axis=(0, 1))).to(self.device)
            # make prediction
            mask = self.maskNet(img0)
            # back to numpy array
            mask = np.squeeze(mask.cpu().numpy())

        if binary:
            mask = (mask > threshold).astype(int)
        return mask

    def PredInpaint(self, img0, mask):
        """
            inpaint img0 under mask
        :param img0: (np.ndarray) input image
        :param mask: (np.ndarray) inpainting mask
        :return: inpainted clean image
        """

        if (self.inpaintNet is None):
            raise Exception(f'The inpaint model is not initialised, cannot inpaint!')
        else:
            # to reduce unnecessary gradient computations
            with torch.no_grad():
    
                # to Tensor
                img0 = from_numpy(np.expand_dims(img0 / self.scale, axis=(0, 1))).to(self.device)
                mask = from_numpy(np.expand_dims(mask, axis=(0, 1))).to(self.device)
                cat = torch.cat((img0 * (1 - mask), mask), dim=1)

                # make prediction
                img1 = self.inpaintNet(cat)
                inpainted = img1 * mask + img0 * (1 - mask)

                ## back to numpy array
                inpainted = np.squeeze(inpainted.cpu().numpy())

        return inpainted * self.scale