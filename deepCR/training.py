""" module for training new deepCR-mask models
"""
import os
import re
import glob
import datetime
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deepCR.util import maskMetric
from deepCR.dataset import dataset, dataset_inpaint
from deepCR.unet_ssli import UNet

__all__ = ['train_mask', 'train_inpaint']

class train_mask():

    def __init__(self, train_data_dir, train_image_files, train_mask_files, 
                    validation_data_dir, validation_image_files, validation_mask_files,
                    train_ignore_files=None, validation_ignore_files=None, 
                    train_sky_val_list=None, validation_sky_val_list=None,
                    aug_sky=[0, 0],
                    name='model', 
                    nChannel_in=1, nChannel_out=1, nChannel_hidden=32, 
                    nLayers_down=1, return_type='sigmoid',
                    gpu=False, epoch_train=20, epoch_evaluate=20, batch_size=1, 
                    lr=5e-3, auto_lr_decay=True, lr_decay_patience=4, lr_decay_factor=0.1, stop_lr=5e-6,
                    use_tqdm=True, use_tqdm_notebook=False, directory='./',
                    max_cores=1):
        """ 
            Train deepCR-mask model.

        Parameters
        ----------
        train_data_dir : str
            Directory where all training data are saved
        train_image_files : a list of str
            A list of image file names for training
        train_mask_files : a list of str
            A list of mask file names for training
        validation_data_dir : str
            Directory where all validation data are saved
        validation_image_files : a list of str
            A list of image file names for validation
        validation_mask_files : a list of str
            A list of mask file names for validation
        train_ignore_files : a list of str, optional
            A list of file names for mask of bad pixel, saturation, etc.
        validation_ignore_files : a list of str, optional
            A list of file names for mask of bad pixel, saturation, etc.
        train_sky_val_list : np.ndarray (N,), optional
            Sky background for training 
        validation_sky_val_list : np.ndarray (N,), optional
            Sky background for validation
        aug_sky : [float, float], optional
            If sky is provided, use random sky background in the range
            [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers 
            to allow the trained model to adapt to a wider range of sky background 
            or equivalently exposure time. Remedy the fact that exposure time in the
            training set is discrete and limited.
        name : str, optional 
            Model name, model saved as name_epoch.pth
        nChannel_in : int, optional 
            Number of channels in the input (1 for mask, 2 for inpaint).
        nChannel_out : int, optional 
            Number of channels in the output (always 1)
        nChannel_hidden : int, optional 
            Number of hidden channels for each convolution.
        nLayers_down : int, optional 
            Number of downsampling layers.
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
        stop_lr : float, optional
            When learning rate decreased to stop_lr, the training will stop. 
        use_tqdm : bool, optional
            Whether to show tqdm progress bar.
        use_tqdm_notebook : bool, optional
            Whether to use jupyter notebook version of tqdm. Overwrites tqdm_default.
        directory : str, optional
            Directory to save trained model.
        max_cores : int, optional
            Maximum number of cores for parallel calculation

        Returns
        -------
        None
        """

        # check images and masks
        for data_dir, image_files, mask_files in ((train_data_dir, train_image_files, train_mask_files), 
                                                (validation_data_dir, validation_image_files, validation_mask_files)):
            ## number of images and masks should match
            assert (len(image_files) == len(mask_files))
            ## shape of image and mask should match
            ima_tmp = np.load(os.path.join(data_dir, image_files[0]))
            mask_tmp = np.load(os.path.join(data_dir, mask_files[0]))
            del image_files, mask_files
            assert (ima_tmp.shape == mask_tmp.shape)
            del mask_tmp
            ## image should be square
            assert (ima_tmp.shape[0] == ima_tmp.shape[1])
            del ima_tmp

        # if ignore map is used
        if (train_ignore_files is not None):
            self.ignore_map = True
        else:
            self.ignore_map = False

        # create torch dataset
        data_train = dataset(train_data_dir, train_image_files, train_mask_files, train_ignore_files, 
                    train_sky_val_list, aug_sky, gpu)
        data_val = dataset(validation_data_dir, validation_image_files, validation_mask_files, validation_ignore_files, 
                    validation_sky_val_list, aug_sky, gpu)
        del train_data_dir, train_image_files, train_mask_files, train_ignore_files, train_sky_val_list, aug_sky
        del validation_data_dir, validation_image_files, validation_mask_files, validation_ignore_files, validation_sky_val_list

        # create torch iterator
        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.Ntrain = len(self.TrainLoader.dataset)
        print(f'>>> Number images for training: {self.Ntrain}')
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
        print(f'>>> Number images for validation: {len(self.ValLoader.dataset)}')
        del data_train, data_val

        # the name tag for saving
        self.name = name

        # initialise the network
        ## determine which device to use
        if gpu:
            device = "cuda"
        else:
            device = "cpu"
        print(f">>> Using {device} for training")
        ## initialise
        self.network = UNet(nChannel_in, nChannel_out, nChannel_hidden, 
                                nLayers_down, return_type).to(device)
        ## save parameter info
        self.network_config = [nChannel_in, nChannel_out, nChannel_hidden, 
                                nLayers_down, return_type]

        # initialise the optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # initialise the scheduler
        if auto_lr_decay:
            self.lr_decay_factor = lr_decay_factor
            self.lr_decay_patience = lr_decay_patience
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=lr_decay_factor, patience=lr_decay_patience,
                                                  cooldown=2, verbose=True, threshold=0.005)
        else:
            self.lr_scheduler = self._void_lr_scheduler

        # initialise the loss function
        self.loss_fn = nn.BCELoss()

        # make sure the maximum number of cores used is reasonable
        torch.set_num_threads(max_cores)
        print(f'>>> Maximum allowed cores: {max_cores}')

        self.validation_loss = []
        self.init_lr = lr
        self.stop_lr = stop_lr

        self.epoch_mask = 0
        self.n_epochs_train = epoch_train
        self.n_epochs_eva = epoch_evaluate

        self.directory = directory
        print(f">>> Outputs will be saved to {self.directory}")

        if use_tqdm_notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = tqdm
        self.disable_tqdm = not (use_tqdm_notebook or use_tqdm)

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

                # number of samples per batch
                n = dat[0].shape[0]
                count += n

                # get data
                if self.ignore_map:
                    images, masks, ignores = dat[0], dat[1], dat[2]
                else:
                    images, masks = dat[0], dat[1]
                    ignores = None
                del dat

                # predict the mask
                pdt_masks = self.network(images)
                del images

                # compute the loss
                if ignores is not None:
                    loss = self.loss_fn(pdt_masks * (1 - ignores), masks * (1 - ignores))
                else:
                    loss = self.loss_fn(pdt_masks, masks)                    
                del ignores

                # sum up loss
                lmask += float(loss) * n
                del loss

                # calculate confusion matrix
                metric += maskMetric(pdt_masks.cpu().numpy() > 0.5, masks.cpu().numpy())
                del pdt_masks, masks

        lmask /= count
        TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
        del metric
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        print('Validation [TPR=%.3f, FPR=%.3f] @threshold = 0.5' % (TPR, FPR))
        print('Validation loss = %.4f' % (lmask))

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
            print('XXX Resume for {} epoch'.format(self.epoch_mask))
            print('XXX Checkpoint loaded from {}'.format(os.path.basename(checkpoint_file)))

            if self.epoch_mask < self.n_epochs_train:
                # training mode
                self._train(self.n_epochs_train - self.epoch_mask, mode='training')

                # reset the lr and scheduler for the new mode
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.init_lr)
                if auto_lr_decay:
                    self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=self.lr_decay_factor, 
                                                            patience=self.lr_decay_patience,
                                                              cooldown=2, verbose=True, threshold=0.005)
                else:
                    self.lr_scheduler = self._void_lr_scheduler

                # evaluation mode
                self._train(self.n_epochs_eva, mode='evaluation')

            else:
                # evaluation mode
                self._train(self.n_epochs_eva + self.n_epochs_train - self.epoch_mask, mode='evaluation')

        else:
            print('Begin first {} epochs for training mode'.format(self.n_epochs_train))
            print('(Use batch activate statistics for batch normalization; keep running mean to be used after '
                  'these epochs)')
            print('')
            self._train(self.n_epochs_train, mode='training')

            # reset the lr and scheduler for the new mode
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.init_lr)
            if auto_lr_decay:
                self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=self.lr_decay_factor, 
                                                        patience=self.lr_decay_patience,
                                                          cooldown=2, verbose=True, threshold=0.005)
            else:
                self.lr_scheduler = self._void_lr_scheduler

            print('Continue onto next {} epochs for evaluation mode'.format(self.n_epochs_eva))
            print('(Batch normalization running statistics frozen and used)')
            print('')
            self._train(self.n_epochs_eva, mode='evaluation')

    def _train(self, epochs, mode='training'):

        # loop over all epochs and train
        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):

            # training or evaluation mode
            if mode == 'training':
                self.network.train()
            elif mode == 'evaluation':
                self.network.eval()
            else:
                raise Exception(f"Unknown mode {mode}!")

            # running information
            print(f'>>>>>>>>> start epoch ({mode}) {self.epoch_mask} ---------')

            # loop over all images and train for each epoch
            for t, dat in enumerate(self.TrainLoader):

                # number of samples per batch
                n = dat[0].shape[0]

                # get data
                if self.ignore_map:
                    images, masks, ignores = dat[0], dat[1], dat[2]
                else:
                    images, masks = dat[0], dat[1]
                    ignores = None
                del dat

                # predict the mask
                pdt_masks = self.network(images)
                del images

                # compute the loss
                if ignores is not None:
                    loss = self.loss_fn(pdt_masks * (1 - ignores), masks * (1 - ignores))
                else:
                    loss = self.loss_fn(pdt_masks, masks)                    
                del ignores, pdt_masks, masks

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # running information
                if (t % 10 == 0):
                    loss, current = loss.item(), (t+1) * n
                    print(f'++++++ loss: {loss:>7f} [{current:>5d}/{self.Ntrain:>5d}]')

            # collect the validation loss for each finished epoch
            val_loss = self.validate_mask()
            self.validation_loss.append(val_loss)

            # Decay the learning rate if needed
            self.lr_scheduler.step(self.validation_loss[-1])

            ## save the model if the validation loss is improved
            if (np.array(self.validation_loss)[-1] == np.array(self.validation_loss).min()):
                filename = self._save(mode=mode)
                print(f'Model saved to {filename}.pth')

            # save the checkpoint
            filename = self._save(mode='checkpoint')
            print(f'Checkpoint saved to {filename}.checkpoint')

            # running information
            print(f'--------- finished epoch ({mode}) {self.epoch_mask} <<<<<<<<<')

            # early stop if the lr is too low
            current_lr = [group['lr'] for group in self.optimizer.param_groups][0]
            if current_lr <= self.stop_lr:
                print(f'xxxxxxxxx stop {mode} because of convergence xxxxxxxxx')
                break

            # record the number of finished epoches
            self.epoch_mask += 1

    def _save(self, mode=None):

        # time saved in the filename
        time = datetime.datetime.now()
        time = str(time)[:10]

        if mode != 'checkpoint':
            # saving model
            filename = f'{time}_{self.name}_{mode}_epoch{self.epoch_mask}'
            torch.save({
                        'network_config': self.network_config,
                        'model_state_dict': self.network.state_dict()
                        },
                        os.path.join(self.directory, filename + '.pth'))
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
            file_list = glob.glob(os.path.join(self.directory, f'*_{self.name}_*.checkpoint'))
            for file in file_list:
                if file != os.path.join(self.directory, filename + '.checkpoint'):
                    os.remove(file)

        return filename



class train_inpaint():

    def __init__(self, train_data_dir, train_image_files, train_mask_files, train_inpaint_files, 
                    validation_data_dir, validation_image_files, validation_mask_files, validation_ipaint_files,
                    train_ignore_files=None, validation_ignore_files=None, 
                    train_sky_val_list=None, validation_sky_val_list=None,
                    aug_sky=[0, 0],
                    name='model', 
                    nChannel_in=2, nChannel_out=1, nChannel_hidden=32, 
                    nLayers_down=2, return_type='ori',
                    gpu=False, epoch_train=40, epoch_evaluate=220, batch_size=1, 
                    lr=1e-2, auto_lr_decay=True, lr_decay_patience=4, lr_decay_factor=0.1, stop_lr=1e-5,
                    use_tqdm=True, use_tqdm_notebook=False, directory='./',
                    max_cores=1,
                    scale=1):
        """ 
            Train deepCR-inpaint model.

        Parameters
        ----------
        train_data_dir : str
            Directory where all training data are saved
        train_image_files : a list of str
            A list of image file names for training
        train_mask_files : a list of str
            A list of mask file names for training
        train_inpaint_files : a list of str
            A list of inpainted image file names for training
        validation_data_dir : str
            Directory where all validation data are saved
        validation_image_files : a list of str
            A list of image file names for validation
        validation_mask_files : a list of str
            A list of mask file names for validation
        validation_ipaint_files : a list of str
            A list of inpainted image file names for validation
        train_ignore_files : a list of str, optional
            A list of file names for mask of bad pixel, saturation, etc.
        validation_ignore_files : a list of str, optional
            A list of file names for mask of bad pixel, saturation, etc.
        train_sky_val_list : np.ndarray (N,), optional
            Sky background for training 
        validation_sky_val_list : np.ndarray (N,), optional
            Sky background for validation
        aug_sky : [float, float], optional
            If sky is provided, use random sky background in the range
            [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers 
            to allow the trained model to adapt to a wider range of sky background 
            or equivalently exposure time. Remedy the fact that exposure time in the
            training set is discrete and limited.
        name : str, optional 
            Model name, model saved as name_epoch.pth
        nChannel_in : int, optional 
            Number of channels in the input (1 for mask, 2 for inpaint).
        nChannel_out : int, optional 
            Number of channels in the output (always 1)
        nChannel_hidden : int, optional 
            Number of hidden channels for each convolution.
        nLayers_down : int, optional 
            Number of downsampling layers.
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
        stop_lr : float, optional
            When learning rate decreased to stop_lr, the training will stop. 
        use_tqdm : bool, optional
            Whether to show tqdm progress bar.
        use_tqdm_notebook : bool, optional
            Whether to use jupyter notebook version of tqdm. Overwrites tqdm_default.
        directory : str, optional
            Directory to save trained model.
        max_cores : int, optional
            Maximum number of cores for parallel calculation
        scale : float, optional 
            Scaling the input image as img0/scale.

        Returns
        -------
        None
        """

        # check images and masks
        for data_dir, image_files, mask_files, inpaint_files in ((train_data_dir, train_image_files, train_mask_files, train_inpaint_files), 
                                                (validation_data_dir, validation_image_files, validation_mask_files, validation_ipaint_files)):
            ## number of images and masks should match
            assert ((len(image_files) == len(mask_files)) and (len(image_files) == len(inpaint_files)))
            ## shape of image and mask should match
            ima_tmp = np.load(os.path.join(data_dir, image_files[0]))
            mask_tmp = np.load(os.path.join(data_dir, mask_files[0]))
            inpaint_tmp = np.load(os.path.join(data_dir, inpaint_files[0]))
            del image_files, mask_files, inpaint_files
            assert ((ima_tmp.shape == mask_tmp.shape) and (ima_tmp.shape == inpaint_tmp.shape))
            del mask_tmp, inpaint_tmp
            ## image should be square
            assert (ima_tmp.shape[0] == ima_tmp.shape[1])
            del ima_tmp

        # if ignore map is used
        if (train_ignore_files is not None):
            self.ignore_map = True
        else:
            self.ignore_map = False

        # create torch dataset
        data_train = dataset_inpaint(train_data_dir, train_image_files, train_mask_files, train_inpaint_files,
                    train_ignore_files, train_sky_val_list, aug_sky, gpu, scale)
        data_val = dataset_inpaint(validation_data_dir, validation_image_files, validation_mask_files, validation_ipaint_files,
                    validation_ignore_files, validation_sky_val_list, aug_sky, gpu, scale)
        del train_data_dir, train_image_files, train_mask_files, train_inpaint_files, train_ignore_files, train_sky_val_list, aug_sky
        del validation_data_dir, validation_image_files, validation_mask_files, validation_ipaint_files, validation_ignore_files, validation_sky_val_list

        # create torch iterator
        self.TrainLoader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.Ntrain = len(self.TrainLoader.dataset)
        print(f'>>> Number images for training: {self.Ntrain}')
        self.ValLoader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
        print(f'>>> Number images for validation: {len(self.ValLoader.dataset)}')
        del data_train, data_val

        # the name tag for saving
        self.name = name

        # initialise the network
        ## determine which device to use
        if gpu:
            device = "cuda"
        else:
            device = "cpu"
        print(f">>> Using {device} for training")
        ## initialise
        self.network = UNet(nChannel_in, nChannel_out, nChannel_hidden, 
                                nLayers_down, return_type).to(device)
        ## save parameter info
        self.network_config = [nChannel_in, nChannel_out, nChannel_hidden, 
                                nLayers_down, return_type]

        # initialise the optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # initialise the scheduler
        if auto_lr_decay:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, factor=lr_decay_factor, patience=lr_decay_patience,
                                                  cooldown=2, verbose=True, threshold=0.005)
        else:
            self.lr_scheduler = self._void_lr_scheduler

        # initialise the loss function
        self.loss_fn = nn.MSELoss()

        # make sure the maximum number of cores used is reasonable
        torch.set_num_threads(max_cores)
        print(f'>>> Maximum allowed cores: {max_cores}')

        self.validation_loss = []
        self.init_lr = lr
        self.stop_lr = stop_lr

        self.epoch_mask = 0
        self.n_epochs_train = epoch_train
        self.n_epochs_eva = epoch_evaluate

        self.directory = directory
        print(f">>> Outputs will be saved to {self.directory}")

        if use_tqdm_notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = tqdm
        self.disable_tqdm = not (use_tqdm_notebook or use_tqdm)

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
        # to reduce unnecessary gradient computations
        with torch.no_grad():
            for i, dat in enumerate(self.ValLoader):

                # number of samples per batch
                n = dat[0].shape[0]
                count += n

                # get data
                if self.ignore_map:
                    images, masks, inpaints, ignores = dat[0], dat[1], dat[2], dat[3]
                else:
                    images, masks, inpaints = dat[0], dat[1], dat[2]
                    ignores = None
                del dat

                # predict the inpaint
                pdt_images = self.network(torch.cat((images, masks), dim=1))
                del images
                pdt_noCRs = pdt_images * masks
                del pdt_images
                true_noCRs = inpaints * masks
                del inpaints, masks

                # compute the loss
                if ignores is not None:
                    loss = self.loss_fn(pdt_noCRs * (1 - ignores), true_noCRs * (1 - ignores))
                else:
                    loss = self.loss_fn(pdt_noCRs, true_noCRs)                    
                del ignores, pdt_noCRs, true_noCRs

                # sum up loss
                lmask += float(loss) * n
                del loss

        lmask /= count
        print('Validation loss = %.4f' % (lmask))

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
            print('XXX Resume for {} epoch'.format(self.epoch_mask))
            print('XXX Checkpoint loaded from {}'.format(os.path.basename(checkpoint_file)))

            if self.epoch_mask < self.n_epochs_train:
                # training mode
                self._train(self.n_epochs_train - self.epoch_mask, mode='training')

                # reset the lr and scheduler for the new mode
                self.lr_scheduler._reset()
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.init_lr)

                # evaluation mode
                self._train(self.n_epochs_eva, mode='evaluation')

            else:
                # evaluation mode
                self._train(self.n_epochs_eva + self.n_epochs_train - self.epoch_mask, mode='evaluation')

        else:
            print('Begin first {} epochs for training mode'.format(self.n_epochs_train))
            print('(Use batch activate statistics for batch normalization; keep running mean to be used after '
                  'these epochs)')
            print('')
            self._train(self.n_epochs_train, mode='training')

            # reset the lr and scheduler for the new mode
            self.lr_scheduler._reset()
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.init_lr)

            print(">>>>>>>>>> current_lr (after reset)", [group['lr'] for group in self.optimizer.param_groups][0])

            print('Continue onto next {} epochs for evaluation mode'.format(self.n_epochs_eva))
            print('(Batch normalization running statistics frozen and used)')
            print('')
            self._train(self.n_epochs_eva, mode='evaluation')

    def _train(self, epochs, mode='training'):

        # loop over all epochs and train
        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):

            # training or evaluation mode
            if mode == 'training':
                self.network.train()
            elif mode == 'evaluation':
                self.network.eval()
            else:
                raise Exception(f"Unknown mode {mode}!")

            # running information
            print(f'>>>>>>>>> start epoch ({mode}) {self.epoch_mask} ---------')

            # loop over all images and train for each epoch
            for t, dat in enumerate(self.TrainLoader):

                # number of samples per batch
                n = dat[0].shape[0]

                # get data
                if self.ignore_map:
                    images, masks, inpaints, ignores = dat[0], dat[1], dat[2], dat[3]
                else:
                    images, masks, inpaints = dat[0], dat[1], dat[2]
                    ignores = None
                del dat

                # predict the inpaint
                pdt_images = self.network(torch.cat((images, masks), dim=1))
                del images
                pdt_noCRs = pdt_images * masks
                del pdt_images
                true_noCRs = inpaints * masks
                del inpaints, masks

                # compute the loss
                if ignores is not None:
                    loss = self.loss_fn(pdt_noCRs * (1 - ignores), true_noCRs * (1 - ignores))
                else:
                    loss = self.loss_fn(pdt_noCRs, true_noCRs)                    
                del ignores, pdt_noCRs, true_noCRs

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # running information
                if (t % 10 == 0):
                    loss, current = loss.item(), (t+1) * n
                    print(f'++++++ loss: {loss:>7f} [{current:>5d}/{self.Ntrain:>5d}]')

            # collect the validation loss for each finished epoch
            val_loss = self.validate_mask()
            self.validation_loss.append(val_loss)

            # Decay the learning rate if needed
            self.lr_scheduler.step(self.validation_loss[-1])

            ## save the model if the validation loss is improved
            if (np.array(self.validation_loss)[-1] == np.array(self.validation_loss).min()):
                filename = self._save(mode=mode)
                print(f'Model saved to {filename}.pth')

            # save the checkpoint
            filename = self._save(mode='checkpoint')
            print(f'Checkpoint saved to {filename}.checkpoint')

            # running information
            print(f'--------- finished epoch ({mode}) {self.epoch_mask} <<<<<<<<<')

            # early stop if the lr is too low
            current_lr = [group['lr'] for group in self.optimizer.param_groups][0]

            print(">>>>>>>>>> current_lr", current_lr)

            if current_lr <= self.stop_lr:
                print(f'xxxxxxxxx stop {mode} because of convergence xxxxxxxxx')
                break

            # record the number of finished epoches
            self.epoch_mask += 1

    def _save(self, mode=None):

        # time saved in the filename
        time = datetime.datetime.now()
        time = str(time)[:10]

        if mode != 'checkpoint':
            # saving model
            filename = f'{time}_{self.name}_{mode}_epoch{self.epoch_mask}'
            torch.save({
                        'network_config': self.network_config,
                        'model_state_dict': self.network.state_dict()
                        },
                        os.path.join(self.directory, filename + '.pth'))
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
            file_list = glob.glob(os.path.join(self.directory, f'*_{self.name}_*.checkpoint'))
            for file in file_list:
                if file != os.path.join(self.directory, filename + '.checkpoint'):
                    os.remove(file)

        return filename