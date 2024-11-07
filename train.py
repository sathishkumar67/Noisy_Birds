from __future__ import annotations
import gin
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

from dataset import Birddataset
from encoder import EncoderConfig, EncoderModel
from decoder import DecoderConfig, DecoderModel


def main():
    train_dataset = Birddataset("dataset", ["budgie", "canary", "duckling", "rubber duck", "unlabeled"], "train")
    test_dataset = Birddataset("dataset", ["budgie", "canary", "duckling", "rubber duck", "unlabeled"], "test")


if __name__ == '__main__':
    main()