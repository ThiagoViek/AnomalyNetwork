"""
MASTER NETWORK: Pretrained ResNet18 for Knowledge Distillation

Classes defining the Source of Knowledge for Distillation on
the Teacher training pipeline.
"""

__name__ = "MASTER_NETWORK_NODE"
__author__ = "Thiago Deeke Viek"
__version__ = "1.1.2"
__maintainer__ = "Thiago Deeke Viek"
__status__ = "Development"

############################################################
##### IMPORTS
############################################################

# Utils Imports
import pdb
import logging

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
##### MASTER NETWORK CLASS
############################################################

class IdentityLayer(nn.Module):
    def __init__(self, softmax_temperature : float) -> None:
        super(IdentityLayer,self).__init__()
        self.softmax_temperature = softmax_temperature
    
    #@torch.cuda.amp.autocast()
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        y = F.softmax(x / self.softmax_temperature,dim=1)
        return y

class MasterNetwork(nn.Module):
    def __init__(self, channels : int, softmax_temperature : float = 1.) -> None:
        super(MasterNetwork,self).__init__()
        # Utils
        self.channels = channels
        
        # Load Pretrained ResNet18 and exclude last FC Layer
        self.network = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
        self.network.fc = IdentityLayer(softmax_temperature)
        logger.info("Master Network Created")
    
    #@torch.cuda.amp.autocast()
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.channels == 1:
            x = torch.stack([x,x,x],dim=1).squeeze(dim=2)
            y = self.network(x)
            return y
        elif self.channels == 3:
            y = self.network(x)
            return y

############################################################
##### UNIT TEST MASTER NETWORK
############################################################

def unit_test_master_network():
    net = MasterNetwork()
    print(net)
    pdb.set_trace()

if __name__ == "__main__":
    unit_test_master_network()
