"""
TESTING DATASET

Class defining the loading and preprocessing steps related 
to the data for training the Student Network.
"""

__name__ = "TESTING DATASET NODE"
__author__ = "Thiago Deeke Viek"
__version__ = "1.0.0"
__maintainer__ = "Thiago Deeke Viek"
__status__ = "Development"

############################################################
##### IMPORTS
############################################################

# Utils Imports
from fileinput import filename
import pdb
import random
import glob
import logging
from PIL import Image

# Torch Imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
##### TEACHER TRAINING DATASET
############################################################

class TestingDataset(Dataset):
    
    RGB_MEAN_IMAGENET = [0.485, 0.456, 0.406]
    RGB_STD_IMAGENET = [0.229, 0.224, 0.225]
    GRY_MEAN_IMAGENET = [0.445]
    GRY_STD_IMAGENET = [0.269]
    
    def __init__(self, dataset_path : str, img_width : int, img_height : int, channels : int,
                patch_size : int, file_format : str = ".JPEG", defect : bool = False
    ) -> None:
        super(TestingDataset,self).__init__()
        
        # Dataset Settings
        self._dataset_path = dataset_path
        self._dataset_examples_paths = glob.glob(self._dataset_path + "*" + file_format)
        self._num_examples = len(self._dataset_examples_paths)
        logger.info(f"Dataset created with {self._num_examples} examples!")
        
        self._img_size = (img_height,img_width)
        self._channels = channels
        self._patch_size = patch_size

        self._defect_dataset = defect
        
    def __len__(self) -> int:
        return self._num_examples
    
    def __getitem__(self, idx : int) -> dict:
        patch,filename = self._generate_img(idx)
        return patch,filename
    
    def _generate_img(self, idx : int) -> list:
        # Read Anchor Image
        img_anchor_original_path = self._dataset_examples_paths[idx]
        img_anchor_original = Image.open(img_anchor_original_path).convert("RGB")
        filename = img_anchor_original_path.split('\\')[-1].split('.')[0]
        
        # Augmentations
        transformations_good = transforms.Compose([
            transforms.Resize(self._img_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.GRY_MEAN_IMAGENET,std=self.GRY_STD_IMAGENET)
        ])
        transformations_defect = transforms.Compose([
            transforms.Resize(self._img_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.GRY_MEAN_IMAGENET,std=self.GRY_STD_IMAGENET)
        ])
        if self._defect_dataset:
            patch = transformations_defect(img_anchor_original)
        else:
            patch = transformations_good(img_anchor_original)
        return patch, filename
