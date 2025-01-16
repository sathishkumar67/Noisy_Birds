from __future__ import annotations
import gin
import torch
from torch.utils.data import DataLoader
import lightning as L

from dataset import Birddataset
from mae.encoder import *
from mae.decoder import *
from model import MAE, MAEWrapper, MAEWrapperConfig

def main():
    # Load the dataset
    train_dataset = Birddataset("dataset", ["budgie", "canary", "duckling", "rubber duck", "unlabeled"], "train")
    test_dataset = Birddataset("dataset", ["budgie", "canary", "duckling", "rubber duck", "unlabeled"], "test")

    # create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=MAEWrapperConfig.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=MAEWrapperConfig.batch_size, shuffle=False)

    # Load the encoder and decoder configs
    gin.parse_config_file("config/encoder_config1.gin")
    encoder_config = EncoderConfig()
    gin.parse_config_file("config/decoder_config1.gin")
    decoder_config = DecoderConfig()
    # load mae wrapper config
    gin.parse_config_file("config/wrapper_config1.gin")
    wrapper_config = MAEWrapperConfig()
    
    # load the encoder model
    encoder = EncoderModel(encoder_config)
    # load the decoder model
    decoder = DecoderModel(decoder_config)
    # load mae model
    mae_model = MAE(encoder=encoder, decoder=decoder)
    # load the wrapper model
    wrapper_model = MAEWrapper(mae_model, wrapper_config)

    # train the model
    trainer = L.Trainer(max_epochs=wrapper_config.num_epochs, accelerator=wrapper_config.device, devices=wrapper_config.gpu_count)
    trainer.fit(wrapper_model, train_loader, test_loader)


if __name__ == '__main__':
    main()