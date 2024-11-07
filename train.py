from __future__ import annotations
import gin
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

from dataset import Birddataset
from encoder import *
from decoder import *


def main():
    # load the wrapper config
    gin.parse_config_file("config/wrapper_config1.gin")
    wrapper_config = WrapperConfig()
    
    # set the seed
    torch.manual_seed(wrapper_config.seed)

    # Load the dataset
    train_dataset = Birddataset("dataset", ["budgie", "canary", "duckling", "rubber duck", "unlabeled"], "train")
    test_dataset = Birddataset("dataset", ["budgie", "canary", "duckling", "rubber duck", "unlabeled"], "test")

    # create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=wrapper_config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=wrapper_config.batch_size, shuffle=False)

    # Load the encoder and decoder configs
    gin.parse_config_file("config/encoder_config1.gin")
    encoder_config = EncoderConfig()
    gin.parse_config_file("config/decoder_config1.gin")
    decoder_config = DecoderConfig()

    # load mae model
    mae_model = MAE(encoder_config, decoder_config)

    # load the wrapper model
    wrapper_model = WrapperModel(wrapper_config, mae_model)

    # train the model
    trainer = L.Trainer(max_epochs=wrapper_config.num_epochs, accelerator=wrapper_config.device, devices=wrapper_config.gpu_count)
    trainer.fit(wrapper_model, train_loader, test_loader)


if __name__ == '__main__':
    main()