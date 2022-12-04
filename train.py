import argparse
from os.path import join

import numpy as np
import torch
from torch.nn import Softmax
from model.loss import BinaryDiceLoss
import albumentations as A
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets.BTCVDataset import BTCVDataset
from model.network import UNet, initialize_weights
from torch.optim.lr_scheduler import LambdaLR
import logging
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter


def get_logger():
    # random rotation, random scaling, random flipping in space and intensit
    logger = logging.getLogger("mideepseg")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    # 设置屏幕打印的格式
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    timestamp = datetime.datetime.timestamp(datetime.datetime.now())
    # 设置log保存
    fh = logging.FileHandler(f"log/{timestamp}.log", encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data-folder", type=str, required=True,
                       help="dataset path")
    parse.add_argument("--batch-size", type=int, default=4,
                       help="batch_size to train")
    parse.add_argument("--devices", type=str, default="cpu",
                       help="use which devices to train")
    parse.add_argument("--weight-decay", type=float, default=1e-4,
                       help="weight decay in optimizer")
    parse.add_argument("--lr", type=float, default=1e-4,
                       help="learning rate")
    parse.add_argument("--epoch", type=int, default=100, help="train how many epoch")
    return parse.parse_args()


def get_lambda(batch_size, data_size, baselr):
    iter_per_epoch = data_size // batch_size

    def lr_lambda(cur_it):
        it = cur_it - 150 * iter_per_epoch
        if it < 0:
            return baselr
        else:
            it //= 30
            lr = baselr * 0.5 ** (it)
            logging.getLogger("mideepseg").info(f"learning rate adjust to {lr}")
            return lr

    return lr_lambda


def main():
    arg = parse_arg()
    logger = get_logger()
    writer = SummaryWriter("log/tensorboard")

    transform = A.Compose([
        A.RandomCrop(416, 416),
        A.Rotate(),
        A.RandomScale(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
    ])

    train_path, test_path = join(arg.data_folder, "train"), join(arg.data_folder, "val")
    train_dataset, test_dataset = BTCVDataset(train_path, transform=transform), BTCVDataset(test_path),
    train_loder, test_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True), DataLoader(
        test_dataset)

    network = UNet(2, 2, 16).to(arg.devices)
    network.apply(initialize_weights)
    opt = Adam(network.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    sched = LambdaLR(opt, lr_lambda=get_lambda(arg.batch_size, data_size=len(train_dataset), baselr=arg.lr))
    criterion = BinaryDiceLoss(num_classes=2)
    for epoch in range(arg.epoch):
        network.train()
        for batch_id, (img, mask) in enumerate(tqdm(train_loder)):
            img, mask = img.to(arg.devices), mask.to(arg.devices)
            output: torch.Tensor = network(img)
            output = torch.softmax(output, dim=1)
            loss = criterion(output, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if batch_id % 10 == 0:
                logger.info(f"epoch {epoch}, batch_id:{batch_id}, loss:{loss.cpu().item():.3f}")
                niter = epoch * len(train_dataset) + batch_id
                writer.add_scalar("Train/loss", loss.cpu().item(), niter)
                # out: np.ndarray = output.cpu().numpy()

        network.eval()
        with torch.no_grad():
            avg_dice = 0.0
            for _, (img, mask) in enumerate(test_loader):
                output = network(img)
                dice = 1 - criterion(output, mask)
                avg_dice += dice.cpu().item()
            avg_dice /= len(test_dataset)
            logger.info(f"epoch {epoch}, loss:{avg_dice:.3f}")
            writer.add_scalar("Test/dice", avg_dice, epoch)

    writer.close()


if __name__ == '__main__':
    main()
