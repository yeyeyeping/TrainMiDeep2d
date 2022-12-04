import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pathlib
from os.path import join
from PIL import Image
import random
from datasets.transforms import interaction_geodesic_distance, itensity_normalization, zoom_image


class BTCVDataset(Dataset):

    def __init__(self, datafolder: str, transform=None):
        super().__init__()
        self.data_paths = list(pathlib.Path(join(datafolder, "img")).glob("*.png"))
        self.datafolder = datafolder
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.data_paths[index]
        mask_path = join(self.datafolder, "mask", img_path.name)
        img_tensor, mask_tensor = \
            np.asarray(Image.open(img_path).convert('L')), (np.asarray(Image.open(mask_path)) == 255).astype(np.uint8)

        if self.transform != None:
            transformed = self.transform(image=img_tensor, mask=mask_tensor)
            img_tensor = transformed['image']
            mask_tensor = transformed['mask']
        extre_points = self.extreme_points(mask_tensor)
        sim_points = self.random_select_points(mask_tensor)
        seeds = self.point2img(np.zeros_like(mask_tensor), extre_points, sim_points)

        bbox = self.create_bbox(mask_tensor)
        crop_img = img_tensor[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        crop_mask = mask_tensor[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        cropped_seed = seeds[bbox[0]:bbox[1], bbox[2]:bbox[3]]

        norm_img = itensity_normalization(crop_img)
        cropped_geos = interaction_geodesic_distance(
            norm_img, cropped_seed)
        zoomed_img, zoomed_geos = zoom_image(norm_img, (96, 96)), zoom_image(cropped_geos, (96, 96))
        input = np.asarray([[zoomed_img, zoomed_geos]])
        return torch.from_numpy(input.copy()), torch.from_numpy(crop_mask.copy()).long()

    def point2img(self, seeds, points, sim_points):
        for (y, x) in points:
            seeds[y, x] = 1
        for (x, y) in sim_points:
            seeds[x, y] = 1
        return seeds

    def create_bbox(self, mask):
        x, y = np.where(mask == 1)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        return [x_min - 5, x_max + 5, y_min - 5, y_max + 5]

    def random_select_points(self, mask):
        data = np.array(mask.squeeze(0), dtype=np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        erode_data = cv2.erode(data, kernel, iterations=1)
        dilate_data = cv2.dilate(data, kernel, iterations=1)
        dilate_data[erode_data == 1] = 0
        k = random.randint(0, 5)
        x, y = np.where(dilate_data)
        rand_idx = random.choices(x, k=k)
        rand_points = list(zip(x[rand_idx], y[rand_idx]))
        return rand_points

    def extreme_points(self, mask, pert=0):
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


def __len__(self):
    return len(self.data_paths)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import albumentations as A


    def test1():
        transform = A.Compose([
            A.RandomCrop(416, 416),
            A.Rotate(),
            A.RandomScale(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ])

        train = BTCVDataset(
            datafolder="/data/home/yeep/Desktop/graduate/research/annotation/code/TrainMiDeep2d/datasets/data/train",
            transform=transform)
        _, (im, label) = next(enumerate(train))
        notransf = BTCVDataset(
            datafolder="/data/home/yeep/Desktop/graduate/research/annotation/code/TrainMiDeep2d/datasets/data/train")
        _, (imraw, labelraw) = next(enumerate(notransf))

        plt.subplot(2, 2, 1)
        plt.imshow(im.squeeze(), cmap="gray")

        plt.subplot(2, 2, 3)
        plt.imshow(label.squeeze(), cmap="gray")

        plt.subplot(2, 2, 2)
        plt.imshow(imraw.squeeze(), cmap="gray")

        plt.subplot(2, 2, 4)
        plt.imshow(labelraw.squeeze(), cmap="gray")
        plt.show()


    def test2():
        train = BTCVDataset(
            datafolder="/data/home/yeep/Desktop/graduate/research/annotation/code/TrainMiDeep2d/datasets/data/train",
        )
        _, (im, label) = next(enumerate(train))
        out = train.random_select_points(label)
        plt.subplot(1, 2, 1)
        plt.imshow(out, cmap="gray")
        print(np.sum(out))

        plt.subplot(1, 2, 2)
        plt.imshow(label.squeeze(), cmap="gray")
        print(torch.sum(label.squeeze()))
        plt.show()


    def test3():
        train = BTCVDataset(
            datafolder="/data/home/yeep/Desktop/graduate/research/annotation/code/TrainMiDeep2d/datasets/data/train",
        )
        _, (im, label) = next(enumerate(train))
        out = train.extreme_points(label)
        print(out)
        print(label)


    def test4():
        train = BTCVDataset(
            datafolder="/data/home/yeep/Desktop/graduate/research/annotation/code/TrainMiDeep2d/datasets/data/train",
        )
        _, (im, label) = next(enumerate(train))
        # plt.subplot(2, 2, 1)
        # plt.imshow(im.squeeze(), cmap="gray")
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(label.squeeze(), cmap="gray")