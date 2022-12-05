import argparse
import os
import random
import time
from os.path import join, exists
from model.metrics import AverageMeter, eval_metrics
import numpy as np
import torch
from torch.nn import Softmax
from model.loss import DiceLoss
import albumentations as A
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets.BTCVDataset import BTCVDataset
from model.network import UNet, initialize_weights
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


def get_logger(fpath):
    # random rotation, random scaling, random flipping in space and intensit
    logger = logging.getLogger("mideepseg")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    timestamp = datetime.datetime.timestamp(datetime.datetime.now())
    # 设置log保存
    fh = logging.FileHandler(join(fpath, f"{timestamp}.log"), encoding='utf8')
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
    parse.add_argument("--log", type=str, default="log", help="log path")
    parse.add_argument("--checkpoint", type=str, default="checkpoint", help="path to save checkpoints")
    return parse.parse_args()


def get_lambda(batch_size, data_size, baselr):
    iter_per_epoch = data_size // batch_size

    def lr_lambda(cur_it):
        it = (cur_it - 150 * iter_per_epoch) // iter_per_epoch
        if it < 0:
            return baselr
        else:
            it //= 30
            lr = baselr * 0.5 ** (it)
            lr = lr < 1e-6 if lr < 1e-6 else lr
            return lr

    return lr_lambda


def get_seg_metrics(total_correct, total_label, total_inter, total_union, num_classes):
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    return {
        "Pixel_Accuracy": np.round(pixAcc, 3),
        "Mean_IoU": np.round(mIoU, 3),
        "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 3)))
    }


def main():
    arg = parse_arg()
    if not exists(arg.log):
        os.mkdir(arg.log)
    if not exists(arg.checkpoint):
        os.mkdir(arg.checkpoint)
    logger = get_logger(arg.log)

    writer = SummaryWriter(join(arg.log, "tensorboard"))

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
        test_dataset, batch_size=arg.batch_size)

    network = UNet(2, 2, 16).to(arg.devices)
    network.apply(initialize_weights)
    opt = Adam(network.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=len(train_dataset) / arg.batch_size * arg.epoch, eta_min=1e-7, )
    criterion = DiceLoss(num_classes=2)

    for epoch in range(arg.epoch):
        total_correct, total_label, total_inter, total_union = 0, 0, 0, 0
        eval_loss, train_loss, batch_time, data_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        network.train()
        tbar = tqdm(train_loder)
        tlc = time.time()
        for batch_id, (img, mask) in enumerate(tbar):
            data_time.update(time.time() - tlc)
            img, mask = img.to(arg.devices), mask.to(arg.devices)
            output: torch.Tensor = network(img)
            output = torch.softmax(output, dim=1)
            loss = criterion(output, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            train_loss.update(loss.cpu().item())
            batch_time.update(time.time() - tlc)
            tlc = time.time()
            seg_metrics = eval_metrics(output.detach().cpu(), mask.detach().cpu(), 2)

            total_correct += seg_metrics[0]
            total_label += seg_metrics[1]
            total_inter += seg_metrics[2]
            total_union += seg_metrics[3]

            pixAcc, mIoU, _ = get_seg_metrics(total_correct, total_label, total_inter, total_union, 2).values()
            tbar.set_description(
                f"TRAIN {epoch}| Loss:{loss.cpu().item():.3f} PixelAcc: {pixAcc:.2f} Mean IoU: {mIoU:.2f} |B {batch_time.average:.2f} |D {data_time.average:.2f}")

            if batch_id % 10 == 0:
                niter = epoch * len(train_dataset) + batch_id
                writer.add_scalar("Train/loss", train_loss.average, niter)
                writer.add_scalar("Train/pixAcc", pixAcc, niter)
                writer.add_scalar("Train/mIoU", mIoU, niter)

                pred = output.detach().cpu() > 0.5
                pred = torch.argmax(torch.as_tensor(pred, dtype=torch.int), dim=1)
                image_grid = make_grid(tensor=torch.from_numpy(np.expand_dims(pred, axis=1).repeat(3, axis=1)),
                                       nrow=arg.batch_size)
                writer.add_image("Train/pred", image_grid, niter)
                img = np.expand_dims(img[:, 0, :, :].cpu().numpy(), axis=1).repeat(3, axis=1)
                mask = np.expand_dims(mask.cpu().numpy(), axis=1).repeat(3, axis=1)
                img_mask = np.concatenate([img, mask])
                raw_imgs = make_grid(tensor=torch.from_numpy(img_mask), nrow=arg.batch_size)
                writer.add_image("Train/raw", raw_imgs, niter)

        logger.info(f"epoch {epoch},  Loss:{train_loss.average:.3f}  PixelAcc: {pixAcc:.2f},Mean IoU: {mIoU:.2f}")
        network.eval()

        with torch.no_grad():
            total_correct, total_label, total_inter, total_union = 0, 0, 0, 0
            tbar = tqdm(test_loader)
            for _, (img_t, mask_t) in enumerate(tbar):
                img_t, mask_t = img_t.to(arg.devices), mask_t.to(arg.devices)
                output = torch.softmax(network(img_t), dim=1)
                dice = 1 - criterion(output, mask_t)
                eval_loss.update(dice.cpu().item())

                seg_metrics = eval_metrics(output.detach().cpu(), mask_t.detach().cpu(), 2)
                total_correct += seg_metrics[0]
                total_label += seg_metrics[1]
                total_inter += seg_metrics[2]
                total_union += seg_metrics[3]
                pixAcc, mIoU, _ = get_seg_metrics(total_correct, total_label, total_inter, total_union, 2).values()
                tbar.set_description(
                    f"EVAl {epoch}| Dice:{eval_loss.average:.3f} PixelAcc: {pixAcc:.2f} Mean IoU: {mIoU:.2f}")
            logger.info(f"epoch {epoch}, dice:{eval_loss.average:.3f}")
            writer.add_scalar("Test/dice", eval_loss.average, epoch)
            writer.add_scalar("Test/pixAcc", pixAcc, niter)
            writer.add_scalar("Test/mIoU", mIoU, niter)

        if epoch % 50 == 0:
            checkpointpth = join(arg.checkpoint, f"epoch{epoch}_checkpoint_avg_dice{dice}.pth")
            torch.save(network.state_dict(), checkpointpth)
            logger.info(f"save checkpoint to {checkpointpth}")

    writer.close()


if __name__ == '__main__':
    main()
