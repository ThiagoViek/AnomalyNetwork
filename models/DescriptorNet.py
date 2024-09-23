"""
DESCRIPTORNET: Deep Stochastic Anomaly Network for Product Inspection

Script contains the implementation of the Neural Networks used to 
detect the presence of anomalies in the images taken from the product.
"""

__name__ = "DESCRIPTORNET_NODE"
__author__ = "Thiago Deeke Viek"
__version__ = "1.1.0"
__maintainer__ = "Thiago Deeke Viek"
__status__ = "Development"

############################################################
##### IMPORTS
############################################################

# Utils Imports
import pdb
import logging
from .FDFE import multiPoolPrepare, multiMaxPooling, unwarpPrepare, unwarpPool

# Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################
##### EVENTS LOGGER
############################################################

# Define Events Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", "%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

############################################################
##### DESCRIPTOR NETWORK CLASS FOR PATCH (65,65)
############################################################

class DescriptorNet65(nn.Module):
    """
    Class defining the Neural Networks used to extract descriptors from 
    the input images.
    The descriptors are extracted using patches with size (65,65). 
    """
    PATCH_DIM = 65
    NUM_DESCRIPTORS = 128

    def __init__(self, img_width : int, img_height : int, channels : int, output_descriptors : bool = False, softmax_temperature : float = 1.) -> None:
        super(DescriptorNet65, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.output_descriptors = output_descriptors
        self.softmax_temperature = softmax_temperature
        
        # Network Layers
        self.conv1 = nn.Conv2d(self.channels,128,5,1)
        self.conv2 = nn.Conv2d(128,128,5,1)
        self.conv3 = nn.Conv2d(128,256,5,1)
        self.conv4 = nn.Conv2d(256,256,4,1)
        self.conv5 = nn.Conv2d(256,self.NUM_DESCRIPTORS,1,1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.activation = nn.LeakyReLU(5e-3)
        self.decoder = nn.Linear(self.NUM_DESCRIPTORS,512)

        # FDFE Layers
        self.multipool_preparation = multiPoolPrepare(self.PATCH_DIM,self.PATCH_DIM)
        self.unwarp_preparation = unwarpPrepare()
        self.multipool = multiMaxPooling(2,2,2,2) 
        self.unwarp1 = unwarpPool(self.NUM_DESCRIPTORS, img_height / (2 * 2 * 2), img_width / (2 * 2 * 2), 2, 2)
        self.unwarp2 = unwarpPool(self.NUM_DESCRIPTORS, img_height / (2 * 2), img_width / (2 * 2), 2, 2)
        self.unwarp3 = unwarpPool(self.NUM_DESCRIPTORS, img_height / 2, img_width / 2, 2, 2)

        # Initialize Weights
        nn.init.normal_(self.conv1.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.02)

        logger.info("DeepDescriptorNet65 Created")
    
    #@torch.cuda.amp.autocast()
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.output_descriptors:
            y = self.forward_descriptors(x)
            return y
        else:            
            y = self.forward_decoder(x)
            y = F.softmax(y / self.softmax_temperature, dim=1)
            return y

    def forward_descriptors(self, x : torch.Tensor) -> torch.Tensor:
        x = self.multipool_preparation(x)
        y = self.activation(self.conv1(x))
        y = self.multipool(y)
        y = self.activation(self.conv2(y))
        y = self.multipool(y)
        y = self.activation(self.conv3(y))
        y = self.multipool(y)
        y = self.activation(self.conv4(y))
        y = self.activation(self.conv5(y))
        y = self.unwarp_preparation(y)
        y = self.unwarp1(y)
        y = self.unwarp2(y)
        y = self.unwarp3(y)
        y = y.view(-1, self.NUM_DESCRIPTORS, self.img_width, self.img_height)
        return y

    def forward_decoder(self, x : torch.Tensor) -> torch.Tensor:
        y = self.activation(self.conv1(x))
        y = self.maxpool(y)
        y = self.activation(self.conv2(y))
        y = self.maxpool(y)
        y = self.activation(self.conv3(y))
        y = self.maxpool(y)
        y = self.activation(self.conv4(y))
        y = self.activation(self.conv5(y))
        y = y.view(-1, self.NUM_DESCRIPTORS)
        y = self.decoder(y)
        return y

############################################################
##### DESCRIPTOR NETWORK CLASS FOR PATCH (33,33)
############################################################

class DescriptorNet33(nn.Module):
    """
    Class defining the Neural Networks used to extract descriptors from 
    the input images.
    The descriptors are extracted using patches with size (33,33). 
    """
    PATCH_DIM = 33
    NUM_DESCRIPTORS = 128

    def __init__(self, img_width : int, img_height : int, channels : int, output_descriptors : bool = False, softmax_temperature : float = 1.) -> None:
        super(DescriptorNet33, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.output_descriptors = output_descriptors
        self.softmax_temperature = softmax_temperature
        
        # Network Layers
        self.conv1 = nn.Conv2d(self.channels,128,3,1)
        self.conv2 = nn.Conv2d(128,256,5,1)
        self.conv3 = nn.Conv2d(256,256,2,1)
        self.conv4 = nn.Conv2d(256,self.NUM_DESCRIPTORS,4,1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.activation = nn.LeakyReLU(5e-3)
        self.decoder = nn.Linear(self.NUM_DESCRIPTORS,512)

        # FDFE Layers
        self.multipool_preparation = multiPoolPrepare(self.PATCH_DIM,self.PATCH_DIM)
        self.unwarp_preparation = unwarpPrepare()
        self.multipool = multiMaxPooling(2,2,2,2) 
        self.unwarp1 = unwarpPool(self.NUM_DESCRIPTORS, img_height / (2 * 2), img_width / (2 * 2), 2, 2)
        self.unwarp2 = unwarpPool(self.NUM_DESCRIPTORS, img_height / 2, img_width / 2, 2, 2)

        # Initialize Weights
        nn.init.normal_(self.conv1.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.02)

        logger.info("DeepDescriptorNet33 Created")
    
    #@torch.cuda.amp.autocast()
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.output_descriptors:
            y = self.forward_descriptors(x)
            return y
        else:            
            y = self.forward_decoder(x)
            y = F.softmax(y / self.softmax_temperature, dim=1)
            return y

    def forward_descriptors(self, x : torch.Tensor) -> torch.Tensor:
        x = self.multipool_preparation(x)
        y = self.activation(self.conv1(x))
        y = self.multipool(y)
        y = self.activation(self.conv2(y))
        y = self.multipool(y)
        y = self.activation(self.conv3(y))
        y = self.activation(self.conv4(y))
        y = self.unwarp_preparation(y)
        y = self.unwarp1(y)
        y = self.unwarp2(y)
        y = y.view(-1, self.NUM_DESCRIPTORS, self.img_width, self.img_height)
        return y

    def forward_decoder(self, x : torch.Tensor) -> torch.Tensor:
        y = self.activation(self.conv1(x))
        y = self.maxpool(y)
        y = self.activation(self.conv2(y))
        y = self.maxpool(y)
        y = self.activation(self.conv3(y))
        y = self.activation(self.conv4(y))
        y = y.view(-1, self.NUM_DESCRIPTORS)
        y = self.decoder(y)
        return y

############################################################
##### DESCRIPTOR NETWORK CLASS FOR PATCH (17,17)
############################################################

class DescriptorNet17(nn.Module):
    """
    Class defining the Neural Networks used to extract descriptors from 
    the input images.
    The descriptors are extracted using patches with size (17,17). 
    """
    PATCH_DIM = 17
    NUM_DESCRIPTORS = 128

    def __init__(self, img_width : int, img_height : int, channels : int, output_descriptors : bool = False, softmax_temperature : float = 1.) -> None:
        super(DescriptorNet17, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.output_descriptors = output_descriptors
        self.softmax_temperature = softmax_temperature
        
        # Network Layers
        self.conv1 = nn.Conv2d(self.channels,128,6,1)
        self.conv2 = nn.Conv2d(128,256,5,1)
        self.conv3 = nn.Conv2d(256,256,5,1)
        self.conv4 = nn.Conv2d(256,self.NUM_DESCRIPTORS,4,1)
        self.activation = nn.LeakyReLU(5e-3)
        self.decoder = nn.Linear(self.NUM_DESCRIPTORS,512)
        self.multipool_preparation = multiPoolPrepare(self.PATCH_DIM,self.PATCH_DIM)

        # Initialize Weights
        nn.init.normal_(self.conv1.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.02)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.02)

        logger.info("DeepDescriptorNet17 Created")
    
    #@torch.cuda.amp.autocast()
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.output_descriptors:
            y = self.forward_descriptors(x)
            return y
        else:            
            y = self.forward_decoder(x)
            y = F.softmax(y / self.softmax_temperature, dim=1)
            return y

    def forward_descriptors(self, x : torch.Tensor) -> torch.Tensor:
        x = self.multipool_preparation(x)
        y = self.activation(self.conv1(x))
        y = self.activation(self.conv2(y))
        y = self.activation(self.conv3(y))
        y = self.activation(self.conv4(y))
        y = y.view(-1, self.NUM_DESCRIPTORS, self.img_width, self.img_height)
        return y

    def forward_decoder(self, x : torch.Tensor) -> torch.Tensor:
        y = self.activation(self.conv1(x))
        y = self.activation(self.conv2(y))
        y = self.activation(self.conv3(y))
        y = self.activation(self.conv4(y))
        y = y.view(-1, self.NUM_DESCRIPTORS)
        y = self.decoder(y)
        return y

############################################################
##### DEVELOPMENT TESTING
############################################################

def test() -> None:
    x_decoder = torch.rand(1,3,17,17)
    x_FDFE = torch.rand(1,3,256,256)

    net_decoder = DescriptorNet17(256,256,True)
    y_descriptor = net_decoder(x_FDFE)

    logger.info(y_descriptor.size())

if __name__ == "__main__":
    test()
