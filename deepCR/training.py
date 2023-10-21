""" module for training new deepCR-mask models
"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from tqdm import tqdm_notebook as tqdm_notebook

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deepCR.util import maskMetric
from deepCR.dataset import dataset
from deepCR.unet import WrappedModel, UNet2Sigmoid

__all__ = 'train'

class train():

    def __init__(self, image, mask, 
                    ignore=None, sky=None, aug_sky=[0, 0], 
                    name='model', 
                    hidden=32, gpu=False, epoch=50, batch_size=16, 
                    lr=0.005, auto_lr_decay=True, lr_decay_patience=4, lr_decay_factor=0.1, 
                    save_after=1e5, plot_every=10, 
                    verbose=True, use_tqdm=False, use_tqdm_notebook=False, directory='./'):
        """ 
            Train deepCR-mask model.

        Parameters
        ----------
        image : np.ndarray (N*W*W)
            Training data: image array with CR.
        mask : np.ndarray (N*W*W)
            Training data: CR mask array.
        ignore : np.ndarray (N*W*W), optional
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
        hidden : int, optional 
            Number of channels for the first convolution layer.
        gpu : bool, optional
            To use GPU or not.
        epoch : int, optional
            Number of epochs to train.
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
        save_after : float, optional
            Epoch after which trainer automatically saves model state with lowest validation loss.
        plot_every : float, optional
            For every "plot_every" epoch, plot mask prediction and ground truth for 1st image in
            validation set.
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

        # check sky and ignore map
        if sky is None and aug_sky != [0, 0]:
            raise AttributeError('Var (sky) is required for sky background augmentation!')
        if ignore is None:
            ignore = np.zeros_like(image)
        assert image.shape == mask.shape == ignore.shape
        assert image.shape[1] == image.shape[2]

        # create torch dataset
        data_train = dataset(image, mask, ignore, sky, part='train', aug_sky=aug_sky)
        data_val = dataset(image, mask, ignore, sky, part='val', aug_sky=aug_sky)
        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=1)
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=1)
        self.shape = image.shape[1]
        self.name = name

        # use gpu or cpu
        if gpu:
            self.dtype = torch.cuda.FloatTensor
            self.dint = torch.cuda.ByteTensor
            self.network = nn.DataParallel(UNet2Sigmoid(1,1,hidden))
            self.network.type(self.dtype)
        else:
            self.dtype = torch.FloatTensor
            self.dint = torch.ByteTensor
            self.network = WrappedModel(UNet2Sigmoid(1,1,hidden))
            self.network.type(self.dtype)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        if auto_lr_decay:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=lr_decay_factor, patience=lr_decay_patience,
                                                  cooldown=2, verbose=True, threshold=0.005)
        else:
            self.lr_scheduler = self._void_lr_scheduler

        self.BCELoss = nn.BCELoss()
        self.validation_loss = []
        self.epoch_mask = 0
        self.save_after = save_after
        self.n_epochs = epoch
        self.every = plot_every
        self.directory = directory
        self.verbose = verbose
        self.mode0_complete = False

        if use_tqdm_notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = tqdm
        self.disable_tqdm = not (use_tqdm_notebook or use_tqdm)

    def set_input(self, img0, mask, ignore):
        """
        :param img0: input image
        :param mask: CR mask
        :param ignore: loss mask
        :return: None
        """
        self.img0 = Variable(img0.type(self.dtype)).view(-1, 1, self.shape, self.shape)
        self.mask = Variable(mask.type(self.dtype)).view(-1, 1, self.shape, self.shape)
        self.ignore = Variable(ignore.type(self.dtype)).view(-1, 1, self.shape, self.shape)

    @staticmethod
    def _void_lr_scheduler(self, metric):
        pass

    def validate_mask(self):
        """
        :return: validation loss. print TPR and FPR at threshold = 0.5.
        """
        lmask = 0; count = 0
        metric = np.zeros(4)
        for i, dat in enumerate(self.ValLoader):
            n = dat[0].shape[0]
            count += n
            self.set_input(*dat)
            self.pdt_mask = self.network(self.img0)
            loss = self.backward_network()
            lmask += float(loss.detach()) * n
            metric += maskMetric(self.pdt_mask.reshape(-1, self.shape, self.shape).detach().cpu().numpy() > 0.5, dat[1].numpy())
        lmask /= count
        TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if self.verbose:
            print('[TPR=%.3f, FPR=%.3f] @threshold = 0.5' % (TPR, FPR))
        return (lmask)

    def train(self):
        """ call this function to start training network
        :return: None
        """
        if self.verbose:
            print('Begin first {} epochs of training'.format(int(self.n_epochs * 0.4 + 0.5)))
            print('Use batch activate statistics for batch normalization; keep running mean to be used after '
                  'these epochs')
            print('')
        self.train_initial(int(self.n_epochs * 0.4 + 0.5))

        filename = self.save()
        self.load(filename)
        self.set_to_eval()
        if self.verbose:
            print('Continue onto next {} epochs of training'.format(self.n_epochs - int(self.n_epochs * 0.4 + 0.5)))
            print('Batch normalization running statistics frozen and used')
            print('')
        self.train_continue(self.n_epochs - int(self.n_epochs * 0.4 + 0.5))

    def train_initial(self, epochs):
        self.network.train()
        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):
            for t, dat in enumerate(self.TrainLoader):
                self.optimize_network(dat)
            self.epoch_mask += 1

            if self.epoch_mask % self.every == 0:
                self.plot_example()

            if self.verbose:
                print('----------- epoch = %d -----------' % (self.epoch_mask))
            val_loss = self.validate_mask()
            self.validation_loss.append(val_loss)
            if self.verbose:
                print('loss = %.4f' % (self.validation_loss[-1]))
            if (np.array(self.validation_loss)[-1] == np.array(
                    self.validation_loss).min() and self.epoch_mask > self.save_after):
                filename = self.save()
                if self.verbose:
                    print('Saved to {}.pth'.format(filename))
            self.lr_scheduler.step(self.validation_loss[-1])
            if self.verbose:
                print('')

    def train_continue(self, epochs):
        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):
            for t, dat in enumerate(self.TrainLoader):
                self.optimize_network(dat)
            self.epoch_mask += 1

            if self.epoch_mask % self.every==0:
                self.plot_example()

            if self.verbose:
                print('----------- epoch = %d -----------' % self.epoch_mask)
            valLossMask = self.validate_mask()
            self.validation_loss.append(valLossMask)
            if self.verbose:
                print('loss = %.4f' % (self.validation_loss[-1]))
            if (np.array(self.validation_loss)[-1] == np.array(
                    self.validation_loss).min() and self.epoch_mask > self.save_after):
                filename = self.save()
                if self.verbose:
                    print('Saved to {}.pth'.format(filename))
            self.lr_scheduler.step(self.validation_loss[-1])
            if self.verbose:
                print('')

    def plot_example(self):
        plt.figure(figsize=(10, 30))
        plt.subplot(131)
        plt.imshow(np.log(self.img0[0, 0].detach().cpu().numpy()), cmap='gray')
        plt.title('epoch=%d' % self.epoch_mask)
        plt.subplot(132)
        plt.imshow(self.pdt_mask[0, 0].detach().cpu().numpy() > 0.5, cmap='gray')
        plt.title('prediction > 0.5')
        plt.subplot(133)
        plt.imshow(self.mask[0, 0].detach().cpu().numpy(), cmap='gray')
        plt.title('ground truth')
        plt.show()

    def set_to_eval(self):
        self.network.eval()

    def optimize_network(self, dat):
        self.set_input(*dat)
        self.pdt_mask = self.network(self.img0)
        self.optimizer.zero_grad()
        loss = self.backward_network()
        loss.backward()
        self.optimizer.step()

    def backward_network(self):
        loss = self.BCELoss(self.pdt_mask * (1 - self.ignore), self.mask * (1 - self.ignore))
        return loss

    def plot_loss(self):
        """ plot validation loss vs. epoch
        :return: None
        """
        plt.figure(figsize=(10,5))
        plt.plot(range(self.epoch_mask), self.validation_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Validation loss')
        plt.show()

    def save(self):
        """ save trained network parameters to date_model_name_epoch*.pth
        :return: None
        """
        time = datetime.datetime.now()
        time = str(time)[:10]
        filename = '%s_%s_epoch%d' % (time, self.name, self.epoch_mask)
        torch.save(self.network.state_dict(), self.directory + filename + '.pth')
        return filename

    def load(self, filename):
        """ Continue training from a previous model state saved to filename
        :param filename: (str) filename (without ".pth") to load model state
        :return: None
        """
        self.network.load_state_dict(torch.load(self.directory + filename + '.pth'))
        loc = filename.find('epoch') + 5
        self.epoch_mask = int(filename[loc:])


