"""
TESTING PIPELINE

Collection of functions to test the method.
"""

__name__ = "TESTING_NODE"
__author__ = "Thiago Deeke Viek"
__version__ = "1.0.1"
__maintainer__ = "Thiago Deeke Viek"
__status__ = "Development"

############################################################
##### IMPORTS
############################################################

# Utils Imports
import pdb
import cv2
import logging
from tqdm import tqdm

# Custom Modules
from models.DescriptorNet import DescriptorNet65, DescriptorNet33, DescriptorNet17
from data.TestingDataset import TestingDataset

# Torch Imports
import torch
from torch.utils.data import DataLoader

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
##### TESTING FUNCTIONS
############################################################

# Anomaly Functions
def l2_norm(tensor : torch.Tensor) -> torch.Tensor:
    squared_tensor = torch.square(tensor)
    squared_sum_tensor = torch.sum(squared_tensor,dim=1)
    return torch.sqrt(squared_sum_tensor)

def get_anomaly_score(target_master : torch.Tensor, output_decoder : torch.Tensor) -> torch.Tensor:
    tensor_diff = output_decoder - target_master
    tensor_diff_norm_squared = l2_norm(tensor_diff)**2
    return tensor_diff_norm_squared

# Test Functions
def generate_heatmaps(
    dataset_good_path : str, 
    dataset_defect_path : str,
    teacher_model_path : str,
    student_model_path : str,
    img_width : int,
    img_height : int,
    channels : int,
    patch_size : int,
    device : torch.device,
) -> None:
    # Load Teacher Network
    if patch_size == 65:
        teacher_network = DescriptorNet65(img_width,img_height,channels,output_descriptors=True)
    elif patch_size == 33:
        teacher_network = DescriptorNet33(img_width,img_height,channels,output_descriptors=True)
    elif patch_size == 17:
        teacher_network = DescriptorNet17(img_width,img_height,channels,output_descriptors=True)
    else:
        raise ValueError("Invalid Patch Size")   
    teacher_network.load_state_dict(torch.load(teacher_model_path,map_location=torch.device(device)))
    teacher_network.to(device)
    teacher_network.eval()

    # Load Student Network
    if patch_size == 65:
        student_network = DescriptorNet65(img_width,img_height,channels,output_descriptors=True)
    elif patch_size == 33:
        student_network = DescriptorNet33(img_width,img_height,channelsoutput_descriptors=True)
    elif patch_size == 17:
        student_network = DescriptorNet17(img_width,img_height,channels,output_descriptors=True)
    else:
        raise ValueError("Invalid Patch Size") 
    student_network.load_state_dict(torch.load(student_model_path,map_location=torch.device(device)))
    student_network.to(device)
    student_network.eval()

    # Load Good Dataset
    dataset_good = TestingDataset(dataset_good_path,img_width,img_height,channels,patch_size,".bmp",defect=False)
    dataloader_good = DataLoader(dataset_good,batch_size=1,shuffle=False,num_workers=0)
    
    # Load Defect Dataset
    dataset_defect = TestingDataset(dataset_defect_path,img_width,img_height,channels,patch_size,".bmp",defect=True)
    dataloader_defect = DataLoader(dataset_defect,batch_size=1,shuffle=False,num_workers=0)
    
    # Generate Heatmaps Good Examples
    with torch.no_grad():
        for n,batch in tqdm(enumerate(dataloader_good)):
            # Generate Descriptors
            img_good = batch[0].to(device)
            teacher_descriptors = teacher_network(img_good)
            student_descriptors = student_network(img_good)

            # Compute Distance Norms
            distance_norm = get_anomaly_score(teacher_descriptors,student_descriptors)

            # Generate Heatmaps
            heatmap = distance_norm.numpy().reshape(img_width,img_height)
            heatmap *= 0.25
            heatmap = heatmap.astype(int)
            cv2.imwrite(f"logs/test-heatmaps/good/{n}.png",heatmap)

    # Generate Heatmaps Defect Examples
    with torch.no_grad():
        for n,batch in tqdm(enumerate(dataloader_defect)):
            # Generate Descriptors
            img_defect = batch[0].to(device)
            teacher_descriptors = teacher_network(img_defect)
            student_descriptors = student_network(img_defect)

            # Compute Distance Norms
            distance_norm = get_anomaly_score(teacher_descriptors,student_descriptors)

            # Generate Heatmaps
            heatmap = distance_norm.numpy().reshape(img_width,img_height)
            heatmap *= 0.25
            heatmap = heatmap.astype(int)
            cv2.imwrite(f"logs/test-heatmaps/defect/{batch[1][0]}.png",heatmap)
