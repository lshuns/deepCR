"""main module to instantiate deepCR models and use them
"""

import torch
import torch.nn as nn
from torch import from_numpy

import numpy as np

from deepCR.unet_ssli import UNet
from deepCR.unet import UNet2Sigmoid, WrappedModel
from learned_models import mask_dict, inpaint_dict, default_model_path

__all__ = ['deepCR']


class deepCR():

    def __init__(self, mask_model_path, inpaint_model_path=None, device='CPU', 

                n_channels_mask=1, n_classes_mask=1, hidden_mask=32, 
                num_downs_mask=1, return_type_mask='sigmoid',

                n_channels_inpaint=1, n_classes_inpaint=1, hidden_inpaint=32, 
                num_downs_inpaint=1, return_type_inpaint='sigmoid',

                scale=1, 
                model_from='ssli', mask=None, inpaint=None
                ):

        """
            Instantiation of deepCR with specified model configurations

        Parameters
        ----------
        mask_model_path : str
            The file path to the trained model for mask (incl. '.pth')
        inpaint_model_path : (optional) str
            The file path to the trained model for inpaint (incl. '.pth')
        device : str
            One of 'CPU' or 'GPU'
        n_channels : int, optional 
            Number of channels for the first convolution layer.
        n_classes : int, optional 
            Number of classes for the final convolution layer.
        hidden : int, optional 
            Number of hidden layers in the U-Net.
        num_downs : int, optional 
            Number of downsampling blocks.
        return_type : str, optional 
            What type of values should the U-Net forward process return.
        scale : float, optional 
            Scaling the input image as img0/scale.
        model_from : str, optional 
            Who trained the model? Unfortunately, a guy named ssli changed the UNet coding, 
            which makes the saved models have different formats between their trainings...
        mask : str, optional
            The old mask model name used by the public training.
        inpaint : str, optional
            The old inpaint model name used by the public training.

        Returns
        -------
        None
            """

        if model_from == 'ssli':
            if device == 'GPU':
                self.dtype = torch.cuda.FloatTensor
                self.dint = torch.cuda.ByteTensor

                # initialise the network
                self.maskNet = nn.DataParallel(
                                    UNet(n_channels_mask, n_classes_mask, hidden_mask, 
                                        num_downs_mask, return_type_mask))
                # load the learned model
                self.maskNet.load_state_dict(torch.load(mask_model_path))

                # for the inpaint
                if inpaint_model_path is not None:
                    self.inpaintNet = nn.DataParallel(
                                    UNet(n_channels_inpaint, n_classes_inpaint, hidden_inpaint, 
                                        num_downs_inpaint, return_type_inpaint))
                    self.inpaintNet.load_state_dict(torch.load(inpaint_model_path))

                    self.inpaintNet.eval()
                else:
                    self.inpaintNet = None

            else:
                self.dtype = torch.FloatTensor
                self.dint = torch.ByteTensor

                # initialise the network
                self.maskNet = UNet(n_channels_mask, n_classes_mask, hidden_mask, 
                    num_downs_mask, return_type_mask)

                # load the learned model
                self.maskNet.load_state_dict(torch.load(mask_model_path, map_location='cpu'))

                # for the inpaint
                if inpaint_model_path is not None:
                    self.inpaintNet = UNet(n_channels_inpaint, n_classes_inpaint, hidden_inpaint, 
                                        num_downs_inpaint, return_type_inpaint)
                    self.inpaintNet.load_state_dict(torch.load(inpaint_model_path, map_location='cpu'))

                    self.inpaintNet.eval()
                else:
                    self.inpaintNet = None

            # evaluate the parameters
            self.maskNet.eval()

            self.scale = scale

        ### the public deepCR model
        else:
            if device == 'GPU':
                self.dtype = torch.cuda.FloatTensor
                self.dint = torch.cuda.ByteTensor
                wrapper = nn.DataParallel
            else:
                self.dtype = torch.FloatTensor
                self.dint = torch.ByteTensor
                wrapper = WrappedModel
            if mask in mask_dict.keys():
                self.scale = mask_dict[mask][2]
                mask_path = default_model_path + '/mask/' + mask + '.pth'
                self.maskNet = wrapper(mask_dict[mask][0](*mask_dict[mask][1]))
            else:
                self.scale = 1
                mask_path = mask
                self.maskNet = wrapper(UNet2Sigmoid(1, 1, hidden))
            self.maskNet.type(self.dtype)
            if device != 'GPU':
                self.maskNet.load_state_dict(torch.load(mask_path, map_location='cpu'))
            else:
                self.maskNet.load_state_dict(torch.load(mask_path))
            self.maskNet.eval()

            if inpaint is not None:
                inpaint_path = default_model_path + '/inpaint/' + inpaint + '.pth'
                self.inpaintNet = wrapper(inpaint_dict[inpaint][0](*inpaint_dict[inpaint][1])).type(self.dtype)
                if device != 'GPU':
                    self.inpaintNet.load_state_dict(torch.load(inpaint_path, map_location='cpu'))
                else:
                    self.inpaintNet.load_state_dict(torch.load(inpaint_path))
                self.inpaintNet.eval()
            else:
                self.inpaintNet = None

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

            # data pre-processing
            img0 = (img0 / self.scale).astype(np.float32) 

            # to Tensor
            shape = img0.shape[-2:]
            img0 = from_numpy(img0).type(self.dtype). \
                              view(1, -1, shape[0], shape[1])

            # make prediction
            mask = self.maskNet(img0)

            # back to numpy array
            mask = mask.cpu().view(shape[0], shape[1]).numpy()

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
    
                # data pre-processing
                img0 = (img0 / self.scale).astype(np.float32)
                mask = mask.astype(np.float32)
    
                # to Tensor
                shape = img0.shape[-2:]
                img0 = from_numpy(img0).type(self.dtype). \
                                  view(1, -1, shape[0], shape[1])
                mask = from_numpy(mask).type(self.dtype). \
                       view(1, -1, shape[0], shape[1])
                cat = torch.cat((img0 * (1 - mask), mask), dim=1)

                # make prediction
                img1 = self.inpaintNet(cat)
                inpainted = img1 * mask + img0 * (1 - mask)

                ## back to numpy array
                inpainted = inpainted.cpu().view(shape[0], shape[1]).numpy()

        return inpainted * self.scale