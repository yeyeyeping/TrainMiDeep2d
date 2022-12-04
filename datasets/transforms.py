import numpy as np
import random

from datasets.BTCVDataset import BTCVDataset
import GeodisTK
from scipy import ndimage

def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)
def make_gt(img, labels, sigma=10, one_mask_per_point=False):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h // 2, w // 2), sigma=sigma)
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]
        if one_mask_per_point:
            gt = np.zeros(shape=(h, w, labels.shape[0]))
            for ii in range(labels.shape[0]):
                gt[:, :, ii] = make_gaussian((h, w), center=labels[ii, :], sigma=sigma)
        else:
            gt = np.zeros(shape=(h, w), dtype=np.float64)
            for ii in range(labels.shape[0]):
                gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))

    gt = gt.astype(dtype=img.dtype)

    return gt
def extreme_points(mask, pert):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x) + pert)),  # left
                     find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x) - pert)),  # right
                     find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y) + pert)),  # top
                     find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y) - pert))  # bottom
                     ])

class ExtremePoints(object):
    """
    Returns the four extreme points (left, right, top, bottom) (with some random perturbation) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pert: number of pixels fo the maximum perturbation
    """

    def __init__(self, sigma=10, pert=5):
        self.sigma = sigma
        self.pert = pert

    def __call__(self, mask):
        if mask.ndim == 3:
            raise ValueError('ExtremePoints not implemented for multiple object per image.')
        _target = np.array(mask)
        if np.max(_target) == 0:
            points = np.zeros(_target.shape, dtype=_target.dtype)  # TODO: handle one_mask_per_point case
        else:
            _points = extreme_points(_target, self.pert)
            points = make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)

        return points
# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   2 Sep., 2021
# Implementation of MIDeepSeg for interactive medical image segmentation and annotation.
# Reference:
#     X. Luo and G. Wang et al. MIDeepSeg: Minimally interactive segmentation of unseen objects
#     from medical images using deep learning. Medical Image Analysis, 2021. DOI:https://doi.org/10.1016/j.media.2021.102102.

'''
为了尽可能的统一本工程的命名规则，这个文件直接拷贝自https://github.com/HiLab-git/MIDeepSeg
'''


def cropped_image(image, bbox, pixel=0):
    random_bbox = [bbox[0] - pixel, bbox[1] -
                   pixel, bbox[2] + pixel, bbox[3] + pixel]
    cropped = image[random_bbox[0]:random_bbox[2],
                    random_bbox[1]:random_bbox[3]]
    return cropped
def extends_points(seed):
    if (seed.sum() > 0):
        points = ndimage.distance_transform_edt(seed == 0)
        points[points > 2] = 0
        points[points > 0] = 1
    else:
        points = seed
    return points.astype(np.uint8)


def QImageToCvMat(incomingImage):
    '''
    Converts a QImage into an opencv MAT format
    from https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
    '''

    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGB32)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.constBits()
    arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
    return arr


def QImageToGrayCvMat(incomingImage):
    '''
    Converts a QImage into an opencv MAT format
    from https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
    '''

    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.constBits()
    arr = np.array(ptr).reshape(height, width)  # Copies the data
    return arr




def itensity_normalization(image):
    out = (image - image.min()) / (image.max() - image.min())
    out = out.astype(np.float32)
    return out


def cstm_normalize(im, max_value=1.0):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value * (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def interaction_geodesic_distance(img, seed, threshold=0):
    if seed.sum() > 0:
        # I = itensity_normalize_one_volume(img)
        I = np.asanyarray(img, np.float32)
        S = seed
        geo_dis = GeodisTK.geodesic2d_fast_marching(I, S)
        # geo_dis = GeodisTK.geodesic2d_raster_scan(I, S, 1.0, 2.0)
        if threshold > 0:
            geo_dis[geo_dis > threshold] = threshold
            geo_dis = geo_dis / threshold
        else:
            geo_dis = np.exp(-geo_dis)
    else:
        geo_dis = np.zeros_like(img, dtype=np.float32)
    return cstm_normalize(geo_dis)


def zoom_image(data, outputsize=(96, 96)):
    """
    reshape image to 64*64 pixels
    """
    x, y = data.shape
    zoomed_image = ndimage.zoom(data, (outputsize[0] / x, outputsize[0] / y))
    return zoomed_image

def itensity_standardization(image):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    pixels = image[image > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (image - mean)/std
    out = out.astype(np.float32)
    return out

def interaction_refined_geodesic_distance(img, seed, threshold=0):
    if seed.sum() > 0:
        # I = itensity_normalize_one_volume(img)
        I = np.asanyarray(img, np.float32)
        S = seed
        geo_dis = GeodisTK.geodesic2d_fast_marching(I, S)
        if threshold > 0:
            geo_dis[geo_dis > threshold] = threshold
            geo_dis = geo_dis / threshold
        else:
            geo_dis = np.exp(-geo_dis**2)
    else:
        geo_dis = np.zeros_like(img, dtype=np.float32)
    return geo_dis
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ep = ExtremePoints()
    train = BTCVDataset(
        datafolder="/data/home/yeep/Desktop/graduate/research/annotation/code/TrainMiDeep2d/datasets/data/train")
    _, (im, label) = next(enumerate(train))
    plt.subplot(1, 2, 1)
    plt.imshow(label, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.imshow(ep(label), cmap="gray")

    plt.show()
