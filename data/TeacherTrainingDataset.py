"""
TEACHER TRAINING DATASET

Class defining the loading and preprocessing steps related 
to the data for training the Teacher Network.
"""

__name__ = "TEACHER_DATASET_NODE"
__author__ = "Thiago Deeke Viek"
__version__ = "1.1.3"
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

class TeacherTrainingDataset(Dataset):
    
    RGB_MEAN_IMAGENET = [0.485, 0.456, 0.406]
    RGB_STD_IMAGENET = [0.229, 0.224, 0.225]
    GRY_MEAN_IMAGENET = [0.445]
    GRY_STD_IMAGENET = [0.269]
    
    def __init__(self, dataset_path : str, patch_size : int, channels : int, file_fomat : str = ".JPEG", fine_tune_flag : bool = False) -> None:
        super(TeacherTrainingDataset,self).__init__()
        
        # Dataset Settings
        self._dataset_path = dataset_path
        self._dataset_examples_paths = glob.glob(dataset_path + "*" + file_fomat)
        self._num_examples = len(self._dataset_examples_paths)
        logger.info(f"Dataset created with {self._num_examples} examples!")

        self._patch_size = patch_size
        self._channels = channels
        self._fine_tune_flag = fine_tune_flag

        # Data Augmentation Transformations
        if self._channels == 3:
            self.transformations_robust = transforms.Compose([
                transforms.CenterCrop(self._patch_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.RGB_MEAN_IMAGENET,std=self.RGB_STD_IMAGENET)
            ])
            self.transformations_ft = transforms.Compose([
                transforms.Resize((2 * self._patch_size,2 * self._patch_size)),
                transforms.RandomCrop(self._patch_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.RGB_MEAN_IMAGENET,std=self.RGB_STD_IMAGENET)
            ]) 
        elif self._channels == 1:
            self.transformations_robust = transforms.Compose([
                transforms.Resize((2 * self._patch_size,2 * self._patch_size)),
                transforms.CenterCrop(self._patch_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.GRY_MEAN_IMAGENET,std=self.GRY_STD_IMAGENET)
            ])
            self.transformations_ft = transforms.Compose([
                transforms.Resize((2 * self._patch_size, 2 * self._patch_size)),
                transforms.RandomCrop(self._patch_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.GRY_MEAN_IMAGENET,std=self.GRY_STD_IMAGENET)
            ])
        
    def __len__(self) -> int:
        return self._num_examples
    
    def __getitem__(self, idx : int) -> dict:
        patch = self._generate_example(idx)
        return patch
    
    def _generate_example(self, idx : int) -> list:
        # Read Anchor Image
        img_anchor_original_path = self._dataset_examples_paths[idx]
        img_anchor_original = Image.open(img_anchor_original_path).convert("RGB")
        
        # Augmentations
        if self._fine_tune_flag:
            patch = self.transformations_ft(img_anchor_original)
        else:
            patch = self.transformations_robust(img_anchor_original)
        return patch
