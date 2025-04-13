import torch
from torch import nn
from decoder import VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def _init__(self):
        super().__init__(
            #(batch_size,channel,height,width)-->(batch_size,128,height,width)
            nn.Conv2d(3,128,kernel_size=3,padding=1)
            #(batch_size,128,height,width) -->(batch_size,128,height,width)
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            #(batch_size,128,height,width) -->(batch_size,128,height/2,width/2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0)

            VAE_ResidualBlock(128,256),
            VAE_ResidualBlock(256,256),
        )
