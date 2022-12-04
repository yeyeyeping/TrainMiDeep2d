import os.path

import matplotlib.pyplot as plt
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData
import cv2
from nibabel.spatialimages import SpatialImage
import nibabel as nib
import logging

fmt = '[%(levelname)s]%(asctime)s-%(message)s'
logging.basicConfig(level=logging.ERROR, format=fmt)


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    return (volume - volume.min()) / (volume.max() - volume.min())
    # volume = (volume - volume.min()) / (volume.max() - volume.min())
    # pixels = volume[volume > 0]
    # mean = pixels.mean()
    # std = pixels.std()
    # out = (volume - mean) / std
    # out_random = np.random.normal(0, 1, size=volume.shape)
    # out[volume == 0] = out_random[volume == 0]
    # out = out.astype(np.float32)
    # return out


# filepath = "/media/yeep/7fd96b65-0eb2-4193-9f7c-b353a0971045/graduate/research/annotation/datasets/BTCV/labelsTr/label0001.nii.gz"
# source: SpatialImage = nib.load(filepath)


# data = np.asanyarray(source.get_fdata())
# # data = itensity_normalize_one_volume(data)
# mask = np.asarray(np.rot90(data[:, :, 73], 1) == 6, dtype=np.uint8)
# print(np.sum(mask))
# print(np.where(mask == 1))
# cv2.imshow("1", mask * 255)
# cv2.waitKey(0)
# print(np.min(data[:, :, 73]))
# print(np.max(data[:, :, 73]))


# cv2.imwrite("1.png", np.asarray(data[:, :, 73] * 255, dtype=np.uint8))
#
# plt.imshow(data[:, :, 73].transpose(1, 0), cmap="gray")
# plt.show()


# io = ImageIO()
# source = io.LoadVolumeData(filepath)
# shape = source.GetDimensions()
# data = vtk_to_numpy(source.GetPointData().GetScalars())
# data: np.ndarray = vtk_to_numpy(source.GetPointData().GetScalars()).reshape(shape[2], shape[1], shape[0])
#
# img = data[8, :, :]
# print(img[80])
# img = cv2.resize(img, dsize = (512,147*5),interpolation=cv2.INTER_NEAREST)
# plt.imshow(img, cmap="gray")
# plt.show()
def get_slice(index, fname):
    source: SpatialImage = nib.load(fname)
    data = np.asanyarray(source.get_fdata())
    logging.error(f"slice {index} min:{np.min(data)} max:{np.max(data)}")
    return np.asarray(np.rot90(data[:, :, index]))


# labelpath = "/media/yeep/7fd96b65-0eb2-4193-9f7c-b353a0971045/graduate/research/annotation/datasets/BTCV/labelsTr/label0001.nii.gz"
# imgpath = "/media/yeep/7fd96b65-0eb2-4193-9f7c-b353a0971045/graduate/research/annotation/datasets/BTCV/imagesTr/img0001.nii.gz"
# img = get_slice(73, imgpath)
# label = get_slice(73, labelpath)
# out = cv2.normalize(img, dst=None, norm_type=cv2.NORM_MINMAX)
# cv2.imwrite("1.png", (out * 255).astype(np.uint8))
# logging.error(f"area:{np.sum(label == 0)}")
# cv2.imwrite("2.png", ((label == 8) * 255).astype(np.uint8))
import cv2

img = cv2.imread("1.png")
plt.imshow(img)
plt.show()

# cv2.imshow("1", )
# cv2.waitKey(0)
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap="gray")
# plt.imsave("1.png", img, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(label, cmap="gray")
# plt.imsave("2.png", label, cmap="gray")
# plt.show()
