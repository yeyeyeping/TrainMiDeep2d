from model.network import UNet
import torch
import argparse
import albumentations as A
# random rotation, random scaling, random flipping in space and intensit

from datasets import BTCVDataset


def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_folder", type=str, required=True,
                       help="dataset path")
    return parse.parse_args()


def main():
    arg = parse_arg()
    train_dataset = BTCVDataset()


if __name__ == '__main__':
    main()
