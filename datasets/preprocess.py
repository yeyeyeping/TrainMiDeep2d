import json
import os
import random
from os.path import join, basename
import nibabel as nib
import numpy as np
import cv2
import logging

fmt = '[%(levelname)s]%(asctime)s-%(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
data_folder = "/media/yeep/7fd96b65-0eb2-4193-9f7c-b353a0971045/graduate/research/annotation/datasets/BTCV"
_min_area = 512 * 512


def load_json(folder):
    with open(os.path.join(folder, "dataset_0.json"), "r+") as fp:
        datasetjson = "".join(fp.readlines())
    return json.loads(datasetjson)


def select_image(fpath: list, keep_mask: int, output_dir, num_per_volume=6, min_area=5 * 5):
    '''
    :param fpath:list类型，要处理的所有nfti的文件路径的list
    :param keep_mask: 提取keep_mask对应的标签
    :param output_dir:输出的文件夹路径，必须存在
    :param num_per_volume:每一volume抽出几个切片
    :param min_area:mask最小区域要大于多少
    :return:
    '''
    for item in fpath:
        img_path, label_path = join(data_folder, item["image"]), join(data_folder, item["label"])
        logging.info(f"img:{img_path} mask:{label_path} readed into memory")
        img_data, label_data = nib.load(img_path).get_fdata(), nib.load(label_path).get_fdata()
        img_np, label_np = np.asarray(img_data), np.asarray(label_data, dtype=np.uint8)
        max_slice = img_np.shape[2]
        slice_num = 0
        while num_per_volume != slice_num:
            rand_slice = random.randint(0, max_slice - 1)
            filemame = "%s_%s" % (basename(img_path).split(".")[0], rand_slice)
            mask_save_path, img_save_path = \
                join(output_dir, "mask", filemame) + ".png", \
                join(output_dir, "img", filemame) + ".png"
            # 路径下有这个图片，说明随机数抽重复了，跳过本次重新抽
            if os.path.exists(mask_save_path) or os.path.exists(img_save_path):
                continue
            # btcv数据集包含了很多其他器官的mask，这里提取出keep_mask对应的标签
            label_slice_data = (label_np[:, :, rand_slice] == keep_mask)
            area = np.sum(label_slice_data)
            if area < min_area:
                continue
            # 统计一下最小的区域，可以删除
            global _min_area
            if area < _min_area:
                _min_area = area

            slice_num += 1
            # nib读取的ct会自动顺时针旋转90度，使用np把他转回来
            label_slice_data = np.rot90(label_slice_data).astype(np.uint8) * 255
            cv2.imwrite(mask_save_path, label_slice_data)
            logging.info("write mask to %s" % mask_save_path)

            img2d = cv2.normalize(img_np[:, :, rand_slice], dst=None, norm_type=cv2.NORM_MINMAX)
            img2d = np.rot90((img2d * 255)).astype(np.uint8)
            cv2.imwrite(img_save_path, img2d)
            logging.info("write img to %s" % img_save_path)
if __name__ == '__main__':
    if not os.path.exists("data/train/mask"):
        os.makedirs("data/train/mask")

    if not os.path.exists("data/train/img"):
        os.makedirs("data/train/img")

    if not os.path.exists("data/val/mask"):
        os.makedirs("data/val/mask")

    if not os.path.exists("data/val/img"):
        os.makedirs("data/val/img")


    jsondes = load_json(data_folder)
    train_list, val_list = jsondes["training"], jsondes["validation"]
    test_list = jsondes["test"]
    spleen_label = -1
    for (key, value) in jsondes["labels"].items():
        if value == "spleen":
            spleen_label = int(key)
    select_image(jsondes["training"], output_dir="data/train", keep_mask=spleen_label)
    select_image(jsondes["validation"], output_dir="data/val", keep_mask=spleen_label)
    logging.info(_min_area)
