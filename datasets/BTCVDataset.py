import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pathlib
from os.path import join
from PIL import Image
import random
from datasets.transforms import interaction_geodesic_distance, itensity_normalization, zoom_image, \
    itensity_standardization


class BTCVDataset(Dataset):

    def __init__(self, datafolder: str, transform=None):
        super().__init__()
        self.data_paths = list(pathlib.Path(join(datafolder, "img")).glob("*.png"))
        self.datafolder = datafolder
        self.transform = transform

    def norm_img(self, img: np.ndarray):
        std = img.std()
        mean = img.mean()
        return (img - mean) / std

    def __getitem__(self, index):
        img_path = self.data_paths[index]
        mask_path = join(self.datafolder, "mask", img_path.name)
        img_np, mask_np = \
            np.asarray(Image.open(img_path).convert('L')), \
                (np.asarray(Image.open(mask_path)) == 255).astype(np.uint8)
        # show(mask_np, "mask")
        # show(img_np, "raw_img")
        img_np = itensity_standardization(img_np)
        # show(img_np, "norm_img")
        # 应用数据增强
        if self.transform != None:
            transformed = self.transform(image=img_np, mask=mask_np)
            img_np = transformed['image']
            mask_np = transformed['mask']
        if np.sum(mask_np) < 20:
            return torch.zeros(2, 96, 96), torch.zeros(96, 96)
        # show(img_np, "transform_img")
        # show(mask_np, "transform_mask")
        # 提取extrem points
        extre_points = self.extreme_points(mask_np)
        # show_seed(np.zeros_like(mask_np), extre_points, "extre_points")
        # 生成模拟的点击
        sim_points = self.random_select_points(mask_np)
        # show_seed(np.zeros_like(mask_np), sim_points, "sim_points")
        # 将这些点编码成和图像一样大小的矩阵上
        seeds = self.point2img(np.zeros_like(mask_np), extre_points, sim_points)
        # 生成bbox
        bbox = self.create_bbox(mask_np)
        self.resolve_bbox(mask_np.shape, bbox)
        # show_bbox(bbox, mask_np)
        # show_bbox(bbox, img_np, title="img_np")
        # 裁剪图片、mask、点击图
        crop_img = img_np[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        crop_mask = mask_np[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        cropped_seed = seeds[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        # show(crop_img, "crop_img")
        # show(crop_mask, "crop_mask")
        # show(cropped_seed, "cropped_seed")
        # print(crop_img.shape)
        # print(bbox)
        norm_img = itensity_normalization(crop_img)
        # show(norm_img, "norm_img")
        cropped_geos = interaction_geodesic_distance(
            norm_img, cropped_seed)
        # show(cropped_geos, "cropped_geos")
        # 放大到96x96
        zoom_mask = zoom_image(crop_mask, (96, 96))
        zoomed_img, zoomed_geos = zoom_image(norm_img, (96, 96)), zoom_image(cropped_geos, (96, 96))
        input = np.stack([zoomed_img, zoomed_geos])
        return  torch.from_numpy(input), torch.from_numpy(zoom_mask).long()

    def point2img(self, seeds, points, sim_points):
        for (x, y) in points:
            seeds[x, y] = 1
        for (x, y) in sim_points:
            seeds[x, y] = 1
        return seeds

    def create_bbox(self, mask):
        x, y = np.where(mask == 1)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        return [x_min - 5, x_max + 5, y_min - 5, y_max + 5]

    def random_select_points(self, mask):
        data = np.array(mask, dtype=np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        erode_data = cv2.erode(data, kernel, iterations=1)
        # show(erode_data, "erode_img")
        dilate_data = cv2.dilate(data, kernel, iterations=1)
        # show(dilate_data, "dilate_img")
        dilate_data[erode_data == 1] = 0
        # show(dilate_data, "random_select_seeds")
        k = random.randint(0, 5)
        x, y = np.where(dilate_data)
        rand_idx = random.choices(range(len(x)), k=k)
        rand_points = list(zip(x[rand_idx], y[rand_idx]))

        return rand_points

    def extreme_points(self, mask, pert=0):
        def find_point(id_x, id_y, ids):
            sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
            return [id_y[sel_id], id_x[sel_id]]

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

    def resolve_bbox(self, shape, bbox):
        if bbox[0] < 0: bbox[0] = 0
        if bbox[1] > shape[0]: bbox[1] = shape[0]
        if bbox[2] < 0: bbox[2] = 0
        if bbox[3] > shape[1]: bbox[3] = shape[1]


def save(img, filename):
    plt.imsave(join("/data/home/yeep/Desktop/graduate/research/annotation/code/TrainMiDeep2d/debugimg", filename), img,
               cmap="gray")


def show(img, title, rgb=False):
    global idx
    plt.subplot(4, 4, idx)
    plt.title(title, fontsize=8, y=-0.2)
    if rgb:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.xticks([])
    plt.yticks([])
    idx += 1


def show_bbox(point, mask, title="bbox", rgb=False):
    copy = (mask.copy() * 255).astype(np.uint8)
    style = 255
    if rgb:
        copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2BGR)
        style = (255, 0, 255)
    cv2.rectangle(copy, (point[2], point[0]),
                  (point[3], point[1]), style, 2)
    show(copy, title, rgb=rgb)


def show_seed(seed, points, title, rgb=False):
    style = 255
    if rgb:
        seed = cv2.cvtColor(seed, cv2.COLOR_GRAY2BGR)
        style = (255, 0, 255)
    for [x, y] in points:
        cv2.rectangle(seed, (y - 3, x - 3),
                      (y + 3, x + 3), style, 2)
    show(seed, title, rgb=rgb)


idx = 1
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
        transform = A.Compose([
            A.RandomCrop(416, 416),
            A.Rotate(),
            A.RandomScale(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ])
        train = BTCVDataset(
            datafolder="/data/home/yeep/Desktop/graduate/research/annotation/code/TrainMiDeep2d/datasets/data/train",
            transform=transform
        )
        _, (im, label) = next(enumerate(train))
        pass


    test4()
    plt.show()
