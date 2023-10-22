import numpy as np
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, image, mask, ignore=None, sky=None, aug_sky=[0, 0], part=None, f_val=0.1, seed=1):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param image: image with CR
        :param mask: CR mask
        :param ignore: loss mask, e.g., bad pixel, saturation, etc.
        :param sky: (np.ndarray) [N,] sky background level
        :param aug_sky: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param part: either 'train' or 'val'. split by 0.8, 0.2
        :param f_val: percentage of dataset reserved as validation set.
        :param seed: fix numpy random seed to seed, for reproducibility.
        """

        np.random.seed(seed)

        ## images are provided in numpy array
        if (type(image) == np.ndarray):
            self.Nima = image.shape[0]
        ## images are provided in path list
        else:
            self.Nima = len(image)

        # split to train and validation sets
        assert f_val < 1 and f_val > 0
        f_train = 1 - f_val

        # determine sets for train and validation
        if part == 'train':
            s = np.s_[:int(self.Nima * f_train)]
        elif part == 'val':
            s = np.s_[int(self.Nima * f_train):]
        else:
            s = np.s_[0:]
        # print(f'>>> Images for {part}: {s}')

        # select train or validation sets
        self.image = image[s]
        # print('---- selected image', self.image)
        del image
        self.mask = mask[s]
        del mask

        ## update number of images
        if (type(self.image) == np.ndarray):
            self.Nima = self.image.shape[0]
        ## images are provided in path list
        else:
            self.Nima = len(self.image)
        print(f'>>> Number images for {part}: {self.Nima}')

        if ignore is not None:
            self.ignore = ignore[s]
            del ignore
        else:
            self.ignore = None

        if sky is not None:
            self.sky = sky[s]
            del sky
        else:
            self.sky = None

        del s
        self.aug_sky = aug_sky

    def __len__(self):
        return self.Nima

    def __getitem__(self, i):

        # get image
        image = self.image[i]
        if (type(image) == str):
            image = np.load(image)

        # get mask
        mask = self.mask[i]
        if (type(mask) == str):
            mask = np.load(mask)

        # add sky
        if self.sky is not None:
            image = image + (self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])) * self.sky[i]

        # the ignore map
        if self.ignore is not None:
            ignore = self.ignore[i]
            if (type(ignore) == str):
                ignore = np.load(ignore)
            return image, mask, ignore
        else:
            return image, mask
        