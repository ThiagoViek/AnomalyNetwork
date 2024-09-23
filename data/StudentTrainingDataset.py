"""
STUDENT TRAINING DATASET

Class defining the loading and preprocessing steps related 
to the data for training the Student Network.
"""

__name__ = "STUDENT DATASET NODE"
__author__ = "Thiago Deeke Viek"
__version__ = "1.0.0"
__maintainer__ = "Thiago Deeke Viek"
__status__ = "Development"

############################################################
##### IMPORTS
############################################################

# Utils Imports
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

class StudentTrainingDataset(Dataset):
    
    RGB_MEAN_IMAGENET = [0.485, 0.456, 0.406]
    RGB_STD_IMAGENET = [0.229, 0.224, 0.225]
    GRY_MEAN_IMAGENET = [0.445]
    GRY_STD_IMAGENET = [0.269]
    
    def __init__(self, dataset_path : str, img_width : int, img_height : int, channels : int,
                patch_size : int, file_format : str = ".JPEG", fine_tune_flag : bool = False
    ) -> None:
        super(StudentTrainingDataset,self).__init__()
        
        # Dataset Settings
        self._dataset_path = dataset_path
        self._dataset_examples_paths = glob.glob(self._dataset_path + "*" + file_format)
        self._num_examples = len(self._dataset_examples_paths)
        
        self._img_size = (img_height,img_width)
        self._channels = channels
        self._patch_size = patch_size
        self._fine_tune_flag = fine_tune_flag
        logger.info(f"Dataset created with {self._num_examples} examples!")

        # Data Augmentation Transformations
        if self._channels == 3:
            self.transformations_robust = transforms.Compose([
                transforms.Resize(self._img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.RGB_MEAN_IMAGENET,std=self.RGB_STD_IMAGENET)
            ])
            self.transformations_ft = transforms.Compose([
                transforms.Resize(self._img_size),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.RGB_MEAN_IMAGENET,std=self.RGB_STD_IMAGENET)
            ])
        elif self._channels == 1:
            self.transformations_robust = transforms.Compose([
                transforms.Resize(self._img_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.GRY_MEAN_IMAGENET,std=self.GRY_STD_IMAGENET)
            ])
            self.transformations_ft = transforms.Compose([
                transforms.Resize(self._img_size),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.GRY_MEAN_IMAGENET,std=self.GRY_STD_IMAGENET)
            ])
        
    def __len__(self) -> int:
        return self._num_examples
    
    def __getitem__(self, idx : int) -> dict:
        patch = self._generate_patch(idx)
        return patch
    
    def _generate_patch(self, idx : int) -> list:
        # Read Anchor Image
        img_anchor_original_path = self._dataset_examples_paths[idx]
        img_anchor_original = Image.open(img_anchor_original_path).convert("RGB")
        
        # Augmentations
        if self._fine_tune_flag:
            patch = self.transformations_ft(img_anchor_original)
        else:
            patch = self.transformations_robust(img_anchor_original)    
        return patch
