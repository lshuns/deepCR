""" module for training new deepCR-mask models
"""
import os
import re
import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deepCR.util import maskMetric
from deepCR.dataset import dataset
from deepCR.unet_ssli import UNet

__all__ = 'train'

class train():

    def __init__(self, image, mask, 
                    ignore=None, sky=None, aug_sky=[0, 0], 
                    name='model', 
                    n_channels=1, n_classes=1, hidden=32, num_downs=1, return_type='sigmoid',
                    gpu=False, epoch_train=20, epoch_evaluate=20, batch_size=1, 
                    lr=0.005, auto_lr_decay=True, lr_decay_patience=4, lr_decay_factor=0.1, 
                    verbose=True, use_tqdm=True, use_tqdm_notebook=False, directory='./'):
        """ 
            Train deepCR-mask model.

        Parameters
        ----------
        image : np.ndarray (N*W*W) or a list of pathes to the image files
            Training data: images with CR.
        mask : np.ndarray (N*W*W) or a list of pathes to the mask files
            Training data: CR mask.
        ignore : np.ndarray (N*W*W) or a list of pathes to the ignore files, optional
            Training data: Mask for taking loss. e.g., bad pixel, saturation, etc.
        sky : np.ndarray (N,), optional
            Sky background
        aug_sky : [float, float], optional
            If sky is provided, use random sky background in the range
            [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers 
            to allow the trained model to adapt to a wider range of sky background 
            or equivalently exposure time. Remedy the fact that exposure time in the
            training set is discrete and limited.
        name : str, optional 
            Model name, model saved as name_epoch.pth
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
        gpu : bool, optional
            To use GPU or not.
        epoch_train : int, optional
            Number of epochs for training mode.
        epoch_evaluate : int, optional
            Number of epochs for evaluation mode.
        batch_size : int, optional 
            How many samples per batch to load.
        lr : float, optional
            Learning rate. 
        auto_lr_decay : bool, optional
            Reduce learning rate by "lr_decay_factor" after validation loss do not decrease for
            "lr_decay_patience" + 1 epochs.
        lr_decay_patience : float, optional
            Reduce learning rate by lr_decay_factor after validation loss do not decrease for
            "lr_decay_patience" + 1 epochs.
        lr_decay_factor : float, optional
            Factor by which the learning rate will be reduced. new_lr = lr * factor. 
        verbose : bool, optional
            Print validation loss and detection rates for every epoch.
        use_tqdm : bool, optional
            Whether to show tqdm progress bar.
        use_tqdm_notebook : bool, optional
            Whether to use jupyter notebook version of tqdm. Overwrites tqdm_default.
        directory : str, optional
            Directory to save trained model.

        Returns
        -------
        None
        """

        # check sky map
        if sky is None and aug_sky != [0, 0]:
            raise AttributeError('Var (sky) is required for sky background augmentation!')

        ## images are provided in numpy array
        if (type(image) == np.ndarray) and (type(mask) == np.ndarray):
            # image and mask should match
            assert image.shape == mask.shape
            # image should be square
            assert image.shape[1] == image.shape[2]
            # save image size info
            self.shape = image.shape[1]
        ## images are provided in path list
        elif (type(image[0]) == str) and (type(mask[0]) == str):
            # image and mask should match
            assert len(image) == len(mask)
            ima_tmp = np.load(image[0])
            mask_tmp = np.load(mask[0])
            assert ima_tmp.shape == mask_tmp.shape
            # image should be square
            assert ima_tmp.shape[0] == ima_tmp.shape[1]
            # save image size info
            self.shape = ima_tmp.shape[0]
            del ima_tmp, mask_tmp
        else:
            raise TypeError('Input image and mask must be numpy data arrays or list of file paths!')

        # create torch dataset
        data_train = dataset(image, mask, ignore, sky, part='train', aug_sky=aug_sky)
        data_val = dataset(image, mask, ignore, sky, part='val', aug_sky=aug_sky)
        del image, mask, ignore, sky

        # create torch iterator
        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0)
        self.Ntrain = len(self.TrainLoader.dataset)
        print(f'>>> Number images for training: {self.Ntrain}')
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f'>>> Number images for validation: {len(self.ValLoader.dataset)}')
        del data_train, data_val

        self.name = name

        # initialise the network
        if gpu:
            self.dtype = torch.cuda.FloatTensor
            self.dint = torch.cuda.ByteTensor
            self.network = nn.DataParallel(
                                UNet(n_channels, n_classes, hidden, 
                                    num_downs, return_type))
            self.network.type(self.dtype)
        else:
            self.dtype = torch.FloatTensor
            self.dint = torch.ByteTensor
            self.network = UNet(n_channels, n_classes, hidden, 
                num_downs, return_type)
            self.network.type(self.dtype)

        # initialise the optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # initialise the scheduler
        if auto_lr_decay:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=lr_decay_factor, patience=lr_decay_patience,
                                                  cooldown=2, verbose=True, threshold=0.005)
        else:
            self.lr_scheduler = self._void_lr_scheduler

        # initialise the loss function
        self.loss_fn = nn.BCELoss()

        self.validation_loss = []
        self.lr = lr

        self.epoch_mask = 0
        self.n_epochs_train = epoch_train
        self.n_epochs_eva = epoch_evaluate

        self.directory = directory
        self.verbose = verbose

        if use_tqdm_notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = tqdm
        self.disable_tqdm = not (use_tqdm_notebook or use_tqdm)

    def set_input(self, img0, mask, ignore=None):
        """
        :param img0: input image
        :param mask: CR mask
        :param ignore: loss mask
        :return: None
        """
        self.img0 = Variable(img0.type(self.dtype)).view(-1, 1, self.shape, self.shape)
        del img0
        self.mask = Variable(mask.type(self.dtype)).view(-1, 1, self.shape, self.shape)
        del mask

        if ignore is not None:
            self.ignore = Variable(ignore.type(self.dtype)).view(-1, 1, self.shape, self.shape)
            del ignore
        else:
            self.ignore = None

    @staticmethod
    def _void_lr_scheduler(self, metric):
        pass

    def validate_mask(self):
        """
        :return: validation loss. print TPR and FPR at threshold = 0.5.
        """

        # for evaluation
        self.network.eval()

        lmask = 0; count = 0
        metric = np.zeros(4)
        # to reduce unnecessary gradient computations
        with torch.no_grad():
            for i, dat in enumerate(self.ValLoader):
                n = dat[0].shape[0]
                count += n

                # prepare the data
                self.set_input(*dat)
                del dat
                # make prediction
                pdt_mask = self.network(self.img0)
                del self.img0
                # compute loss
                if self.ignore is not None:
                    loss = self.loss_fn(pdt_mask * (1 - self.ignore), self.mask * (1 - self.ignore))
                else:
                    loss = self.loss_fn(pdt_mask, self.mask)
                del self.ignore
                # sum up loss
                lmask += float(loss.detach()) * n
                del loss

                # calculate matrix
                metric += maskMetric(pdt_mask.reshape(-1, self.shape, self.shape).detach().cpu().numpy() > 0.5, 
                                     self.mask.reshape(-1, self.shape, self.shape).detach().cpu().numpy())
                del pdt_mask, self.mask

            lmask /= count
            TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)

        if self.verbose:
            print('validation [TPR=%.3f, FPR=%.3f] @threshold = 0.5' % (TPR, FPR))
        return (lmask)

    def train(self, resume=False):
        """ call this function to start training network
        :return: None
        """

        if resume:
            # find the checkpoint
            checkpoint_list = glob.glob(os.path.join(self.directory, '*.checkpoint'))
            if len(checkpoint_list) > 1:
                print('XXX found more than one checkpoint, using the one with larger epoch ID...')
                # get epoch number
                epoch_list = [int(re.search(r'_epoch(\d+).checkpoint', os.path.basename(checkpoint_file))[1]) for checkpoint_file in checkpoint_list]
                # get the last epoch
                max_index = epoch_list.index(max(epoch_list))
                del epoch_list
                # get the last file
                checkpoint_file = checkpoint_list[max_index]
                del checkpoint_list, max_index
            else:
                checkpoint_file = checkpoint_list[0]
                del checkpoint_list

            # load the checkpoint
            checkpoint = torch.load(checkpoint_file)
            self.epoch_mask = checkpoint['epoch'] + 1
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.validation_loss = checkpoint['validation_loss']
            del checkpoint
            if self.verbose:
                print('>>> Resume for {} epoch'.format(self.epoch_mask))
                print('>>> Checkpoint loaded from {}'.format(os.path.basename(checkpoint_file)))

            if self.epoch_mask < self.n_epochs_train:
                self._train(self.n_epochs_train - self.epoch_mask, mode='training')
                self._train(self.n_epochs_eva, mode='evaluation')
            else:
                self._train(self.n_epochs_eva + self.n_epochs_train - self.epoch_mask, mode='evaluation')

        else:
            if self.verbose:
                print('Begin first {} epochs for training mode'.format(self.n_epochs_train))
                print('Use batch activate statistics for batch normalization; keep running mean to be used after '
                      'these epochs')
                print('')
            self._train(self.n_epochs_train, mode='training')

            if self.verbose:
                print('Continue onto next {} epochs for evaluation mode'.format(self.n_epochs_eva))
                print('Batch normalization running statistics frozen and used')
                print('')
            self._train(self.n_epochs_eva, mode='evaluation')

    def _train(self, epochs, mode='training'):

        # training or evaluation mode
        if mode == 'training':
            self.network.train()
        elif mode == 'evaluation':
            self.network.eval()
        else:
            raise Exception(f"Unknown mode {mode}!")

        # loop over all epochs and train
        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):

            # running information
            if self.verbose:
                print(f'>>>>>>>>> start epoch ({mode}) {self.epoch_mask} ---------')

            # loop over all images and train for each epoch
            for t, dat in enumerate(self.TrainLoader):

                # get the image data
                self.set_input(*dat)
                del dat
                # predict the mask
                pdt_mask = self.network(self.img0)
                del self.img0
                # compute the loss
                if self.ignore is not None:
                    loss = self.loss_fn(pdt_mask * (1 - self.ignore), self.mask * (1 - self.ignore))
                else:
                    loss = self.loss_fn(pdt_mask, self.mask)
                del self.ignore, pdt_mask, self.mask
                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # running information
                if (self.verbose) and (t % 10 == 0):
                    print(f'++++++ loss: {loss.item():>7f} [{t:>5d}/{self.Ntrain:>5d}]')

            # collect the validation loss for each finished epoch
            val_loss = self.validate_mask()
            self.validation_loss.append(val_loss)

            # Decay the learning rate is needed
            self.lr_scheduler.step(self.validation_loss[-1])

            # running information
            if self.verbose:
                print('validation loss = %.4f' % (self.validation_loss[-1]))

            ## save the model if the loss is improved
            if (np.array(self.validation_loss)[-1] == np.array(self.validation_loss).min()):
                filename = self._save(mode=mode)
                if self.verbose:
                    print('Model saved to {}.pth'.format(filename))

            # save the checkpoint
            filename = self._save(mode='checkpoint')
            if self.verbose:
                print('Checkpoint saved to {}.checkpoint'.format(filename))

            # running information
            if self.verbose:
                print(f'--------- finished epoch ({mode}) {self.epoch_mask} <<<<<<<<<')

            # record the number of finished epoches
            self.epoch_mask += 1

    def _save(self, mode=None):

        # time saved in the filename
        time = datetime.datetime.now()
        time = str(time)[:10]

        if mode != 'checkpoint':
            # saving model
            filename = f'{time}_{self.name}_{mode}_epoch{self.epoch_mask}'
            torch.save(self.network.state_dict(), os.path.join(self.directory, filename + '.pth'))
        else:
            # saving checkpoint
            filename = f'{time}_{self.name}_epoch{self.epoch_mask}'
            torch.save({
                        'epoch': self.epoch_mask,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.lr_scheduler.state_dict(),
                        'validation_loss': self.validation_loss
                        },
                        os.path.join(self.directory, filename + '.checkpoint'))

            # remove previous checkpoint
            file_list = glob.glob(os.path.join(self.directory, '*.checkpoint'))
            for file in file_list:
                if file != os.path.join(self.directory, filename + '.checkpoint'):
                    os.remove(file)

        return filename