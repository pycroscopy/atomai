"""Utility functions"""

import random
import torch
import numpy as np
import cv2
from scipy import ndimage, fftpack
from skimage import exposure
from skimage.util import random_noise
import matplotlib.pyplot as plt


def load_model(model, weights_path):
    '''Loads weights saved in a pytorch format (.pt) into a model skeleton'''
    if torch.cuda.device_count() > 0:
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model

### Utilities commonly used for experimental/test image data preprocessing ###

def torch_format(image_data):
    '''Reshapes and normalizes (optionally) image data
    to make it compatible with pytorch format'''
    image_data = np.expand_dims(image_data, axis=1)
    image_data = (image_data - np.amin(image_data))/np.ptp(image_data)
    image_data = torch.from_numpy(image_data).float()
    return image_data


def img_resize(image_data, rs):
    '''Image resizing'''
    if image_data.shape[1:3] == rs:
        return image_data.copy()
    image_data_r = np.zeros(
        (image_data.shape[0], rs[0], rs[1]))
    for i, img in enumerate(image_data):
        img = cv2.resize(img, (rs[0], rs[1]))
        image_data_r[i, :, :] = img
    return image_data_r


def img_pad(image_data, pooling):
    '''Pads the image if its size (w, h)
    is not divisible by 2**n, where n is a number
    of pooling layers in a network'''
    # Pad image rows (height)
    while image_data.shape[1] % pooling != 0:
        d0, _, d2 = image_data.shape
        image_data = np.concatenate(
            (image_data, np.zeros((d0, 1, d2))), axis=1)
    # Pad image columns (width)
    while image_data.shape[2] % pooling != 0:
        d0, d1, _ = image_data.shape
        image_data = np.concatenate(
            (image_data, np.zeros((d0, d1, 1))), axis=2)
    return image_data

### Utilities for atom finding ###

def find_com(image_data):
    '''Find atoms via center of mass methods'''
    labels, nlabels = ndimage.label(image_data)
    coordinates = np.array(
        ndimage.center_of_mass(
            image_data, labels, np.arange(nlabels) + 1))
    coordinates = coordinates.reshape(coordinates.shape[0], 2)
    return coordinates
            

def cv_thresh(imgdata,
              threshold=.5):
    """
    Wrapper for opencv binary threshold method.
    Returns thresholded image.
    """
    _, thresh = cv2.threshold(
                    imgdata, 
                    threshold, 1, 
                    cv2.THRESH_BINARY)
    return thresh

### Utilities for inferring basic characteristics of neural network ###

class Hook():
    """
    Returns the input and output of a
    layer during forward/backward pass
    see https://www.kaggle.com/sironghuang/
        understanding-pytorch-hooks/notebook
        
    Args:
        module: torch modul(single layer or sequential block)
        backward (bool): replace forward_hook with backward_hook
    """
    def __init__(self, module, backward=False):
        if backward is False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def mock_forward(model, dims=(1, 64, 64)):
    '''Passes a dummy variable throuh a network'''
    x = torch.randn(1, dims[0], dims[1], dims[2])
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    out = model(x)
    return out

### Vizualization utilities ###

def plot_losses(train_loss, test_loss):
    """Plots train and test losses"""
    print('Plotting training history')
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(train_loss, label='Train')
    ax.plot(test_loss, label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()


def plot_coord(img, coord, fsize=6):
    """Plots coordinates (colored according to atom class)"""
    y, x, c = coord.T
    plt.figure(figsize=(fsize, fsize))
    plt.imshow(img, cmap='gray')
    plt.scatter(x, y, c=c, cmap='RdYlGn', s=8)
    plt.show()

###  Some utilities that can help preparing labeled training set ### 

class MakeAtom:
    """
    Creates an image of atom modelled as
    2D Gaussian and a corresponding mask

    Args:
        sc: float 
            scale parameter, which determines Gaussian width
        intensity: float 
            parameter of 2D gaussian function
        theta: float
            parameter of 2D gaussian function
        offset: float:
            parameter of 2D gaussian function
    """
    def __init__(self, sc, cfp=2, intensity=1, theta=0, offset=0):
        if sc % 2 == 0:
            sc += 1
        self.xo, self.yo = sc/2, sc/2
        x = np.linspace(0, sc, sc)
        y = np.linspace(0, sc, sc)
        self.x, self.y = np.meshgrid(x, y)
        self.sigma_x, self.sigma_y = sc/4, sc/4
        self.theta = theta
        self.offset = offset
        self.intensity = intensity
    
    def atom2dgaussian(self):
        '''Models atom as 2d Gaussian'''
        a = (np.cos(self.theta)**2)/(2*self.sigma_x**2) +\
            (np.sin(self.theta)**2)/(2*self.sigma_y**2)
        b = -(np.sin(2*self.theta))/(4*self.sigma_x**2) +\
             (np.sin(2*self.theta))/(4*self.sigma_y**2)
        c = (np.sin(self.theta)**2)/(2*self.sigma_x**2) +\
            (np.cos(self.theta)**2)/(2*self.sigma_y**2)
        g = self.offset + self.intensity*np.exp(
            -(a*((self.x-self.xo)**2) + 2*b*(self.x-self.xo)*(self.y-self.yo) +\
            c*((self.y-self.yo)**2)))
        return g

def create_lattice_mask(lattice, xy_atoms, create_mask_func):
    """
    Given experimental image and xy atomic coordinates
    creates ground truth image. Currently works only for the case
    where all atoms are one class. Notice that it will round fractional pixels.

    Args:
        lattice: 2D numpy array
            experimental image as 2D numpy array
        xy_atoms: 2 x N numpy array
            position of atoms in the experimental data
        create_mask_func: python function
            function that creates a 2D numpy array with ground truth
            for each atomic coordinate. For example,

            # def create_atomic_mask(r_mask=7, thresh=.2):
            #     atom = MakeAtom(r_mask).atom2dgaussian()
            #     _, mask = cv2.threshold(atom, thresh, 1, cv2.THRESH_BINARY)
            #     return mask

    Returns:
        2D numpy array with ground truth data
    """
    lattice_mask = np.zeros_like(lattice)
    for i in range(xy_atoms.shape[-1]):
        x, y = xy_atoms[:, i]
        x = int(np.around(x))
        y = int(np.around(y))
        mask = create_mask_func()
        r_m = mask.shape[0] / 2
        r_m1 = int(r_m + .5)
        r_m2 = int(r_m - .5)
        lattice_mask[x-r_m1:x+r_m2, y-r_m1:y+r_m2] = mask
    return lattice_mask

def create_atomic_mask(r_mask=7, thresh=.2):
    """
    Helper function for creating atomic maks
    from 2D gaussian via simple thresholding
    """
    atom = MakeAtom(r_mask).atom2dgaussian()
    _, mask = cv2.threshold(atom, thresh, 1, cv2.THRESH_BINARY)
    mask = mask[np.min(np.where(mask == 1)[0]):
                    np.max(np.where(mask==1)[0] + 1),
                    np.min(np.where(mask==1)[1]):
                    np.max(np.where(mask==1)[1]) + 1]
    return mask

class data_transform:
    """
    Applies a sequence of pre-defined operations for data augmentation.

    Args:
        batch_size: int 
            number of images in the batch,
        width: int 
            width of images in the batch,
        height: int
            height of images in the batch,
        channels:
            number of classes (channels) in the ground truth
        dim_order: str 
            channel first (pytorch) or channel last (otherwise) ordering
        norm: int 
            normalization to (0, 1)
        **flip: bool 
            image vertical/horizonal flipping,
        **rotate90: bool
            rotating image by +- 90 deg
        **zoom: tuple
            values for zooming-in (min height, max height, step);
            assumes height==width
        **noise: dict 
            dictionary of noise values for each type of noise,
        **resize: tuple
            values for image resizing (min height, max height, step);
            assumes heght==width.
    """
    def __init__(self, batch_size, width, height,
                 channels, dim_order='pytorch',
                 norm=1, **kwargs):
        self.n, self.w, self.h = batch_size, width, height
        self.ch = channels
        self.dim_order = dim_order
        self.norm = norm
        self.flip = kwargs.get('flip')
        self.rotate90 = kwargs.get('rotate90')
        self.zoom = kwargs.get('zoom')
        self.noise = kwargs.get('noise')
        self.resize = kwargs.get('resize')

    def transform(self, images, masks):
        """
        Applies a sequence of augmentation procedures
        to images and (except for noise) ground truth
        """
        images = (images - np.amin(images))/np.ptp(images)
        if self.flip:
            images, masks = self.batch_flip(images, masks)
        if self.noise is not None:
            images, masks = self.batch_noise(images, masks)
        if self.zoom is not None:
            images, masks = self.batch_zoom(images, masks)
        if self.resize is not None:
            images, masks = self.batch_resize(images, masks)
        if self.dim_order == 'pytorch':
            images = np.expand_dims(images, axis=1)
            masks = np.transpose(masks, (0, 3, 1, 2))
        else:
            images = np.expand_dims(images, axis=3)
            images = images.astype('float32')
        if self.norm != 0:
            images = (images - np.amin(images))/np.ptp(images)
        return images, masks

    def batch_noise(self, X_batch, y_batch,):
        """
        Takes an image stack and applies
        various types of noise to each image
        """
        def make_pnoise(image, l):
            vals = len(np.unique(image))
            vals = (l/50) ** np.ceil(np.log2(vals))
            image_n_filt = np.random.poisson(image * vals) / float(vals)
            return image_n_filt
        pnoise_range = self.noise['poisson']
        spnoise_range = self.noise['salt and pepper']
        gnoise_range = self.noise['gauss']
        blevel_range = self.noise['blur']
        c_level_range = self.noise['contrast']
        X_batch_a = np.zeros((self.n, self.w, self.h))
        for i, img in enumerate(X_batch):
            pnoise = random.randint(pnoise_range[0], pnoise_range[1])
            spnoise = random.randint(spnoise_range[0], spnoise_range[1])
            gnoise = random.randint(gnoise_range[0], gnoise_range[1])
            blevel = random.randint(blevel_range[0], blevel_range[1])
            clevel = random.randint(c_level_range[0], c_level_range[1])
            img = ndimage.filters.gaussian_filter(img, blevel*1e-1)
            img = make_pnoise(img, pnoise)
            img = random_noise(img, mode='gaussian', var=gnoise*1e-4)
            img = random_noise(img, mode='pepper', amount=spnoise*1e-3)
            img = random_noise(img, mode='salt', amount=spnoise*5e-4)
            img = exposure.adjust_gamma(img, clevel*1e-1)
            X_batch_a[i, :, :] = img
        return X_batch_a, y_batch

    def batch_zoom(self, X_batch, y_batch):
        """
        Crops and then resizes to the original size
        all images in one batch
        """
        zoom_list = np.arange(self.zoom[0], self.zoom[1], self.zoom[2])
        X_batch_a = np.zeros((self.n, self.w, self.h))
        y_batch_a = np.zeros((self.n, self.w, self.h, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            rs = np.random.choice(zoom_list)
            w1 = int((self.w-rs)/2)
            w2 = int(rs + (self.w-rs)/2)
            h1 = int((self.h-rs)/2)
            h2 = int(rs + (self.h-rs)/2)
            img = img[w1:w2, h1:h2]
            gt = gt[w1:w2, h1:h2]
            img = cv2.resize(img, (self.w, self.h))
            gt = cv2.resize(gt, (self.w, self.h))
            _, gt = cv2.threshold(gt, 0.25, 1, cv2.THRESH_BINARY)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_a[i, :, :] = img
            y_batch_a[i, :, :, :] = gt
        return X_batch_a, y_batch_a

    def batch_flip(self, X_batch, y_batch):
        """
        Flips and rotates all images and in one batch
        and correponding labels (ground truth)
        """
        X_batch_a = np.zeros((self.n, self.w, self.h))
        y_batch_a = np.zeros((self.n, self.w, self.h, self.ch))
        int_r = (-1, 3) if self.rotate90 else (-1, 1)
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            flip_type = random.randint(int_r[0], int_r[1])
            if flip_type == 3:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            elif flip_type == 2:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = cv2.flip(img, flip_type)
                gt = cv2.flip(gt, flip_type)
            if len(gt.shape) != 3:
                gt = np.expand_dims(gt, axis=2)
            X_batch_a[i, :, :] = img
            y_batch_a[i, :, :, :] = gt
        return X_batch_a, y_batch_a

    def batch_resize(self, X_batch, y_batch):
        """
        Resize all images in one batch and
        corresponding labels (ground truth)
        """
        rs_arr = np.arange(self.resize[0], self.resize[1], self.resize[2])
        rs = np.random.choice(rs_arr)
        if X_batch.shape[1:3] == (rs, rs):
            return X_batch, y_batch
        X_batch_a = np.zeros((self.n, rs, rs))
        y_batch_a = np.zeros((self.n, rs, rs, self.ch))
        for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
            img = cv2.resize(img, (rs, rs), cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (rs, rs), cv2.INTER_CUBIC)
            _, gt = cv2.threshold(gt, 0.25, 1, cv2.THRESH_BINARY)
            if len(gt.shape) < 3:
                gt = np.expand_dims(gt, axis=-1)
            X_batch_a[i, :, :] = img
            y_batch_a[i, :, :, :] = gt
        return X_batch_a, y_batch_a
    
def FFTmask(imgsrc, maskratio=10):
    """
    Takes a square real space image and filter out a disk with radius equal to:
    1/maskratio * image size.
    Retruns FFT transform of the image and the filtered FFT transform
    """
    # Take the fourier transform of the image.
    F1 = fftpack.fft2((imgsrc))
    # Now shift so that low spatial frequencies are in the center.
    F2 = (fftpack.fftshift((F1)))
    # copy the array and zero out the center
    F3 = F2.copy()
    l = int(imgsrc.shape[0]/maskratio)
    m = int(imgsrc.shape[0]/2)
    y, x = np.ogrid[1: 2*l + 1, 1:2*l + 1]
    mask = (x - l)*(x - l) + (y - l)*(y - l) <= l*l
    F3[m-l:m+l, m-l:m+l] = F3[m-l:m+l, m-l:m+l] * (1 - mask)
    return F2, F3


def FFTsub(imgsrc, F3):
    """
    Takes real space image and filtred FFT.
    Reconstructs real space image and subtracts it from the original.
    Returns normalized image.
    """
    reconstruction = np.real(fftpack.ifft2(fftpack.ifftshift(F3)))
    diff = np.abs(imgsrc - reconstruction)
    # normalization
    diff = diff - np.amin(diff)
    diff = diff/np.amax(diff)
    return diff


def threshImg(diff, threshL=0.25, threshH=0.75):
    """
    Takes in difference image, low and high thresold values,
    and outputs a map of all defects.
    """
    threshIL = diff < threshL
    threshIH = diff > threshH
    threshI = threshIL + threshIH
    return threshI
