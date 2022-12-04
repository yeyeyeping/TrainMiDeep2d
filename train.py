from model.network import UNet
import torch
import argparse
import albumentations as A
# random rotation, random scaling, random flipping in space and intensit

from datasets import BTCVDataset
BTCVDataset()