"""
TRAINING PIPELINES

Collection of functions to train the Teacher and the Student 
Networks.
"""

__name__ = "TRAINING_NODE"
__author__ = "Thiago Deeke Viek"
__version__ = "1.0.5"
__maintainer__ = "Thiago Deeke Viek"
__status__ = "Development"

############################################################
##### IMPORTS
############################################################

# Utils Imports
import pdb
import math
import random
import logging
from datetime import datetime
from tqdm import tqdm

# Custom Modules
from data.TeacherTrainingDataset import TeacherTrainingDataset
from data.StudentTrainingDataset import StudentTrainingDataset
from models.MasterNetwork import MasterNetwork
from models.DescriptorNet import DescriptorNet65, DescriptorNet33, DescriptorNet17

# Torch Imports
import torch
import torch.optim as optim
#from torch.cuda.amp import GradScaler
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
##### TEACHER NETWORK TRAINING PIPELINE
############################################################

# Loss Functions
def l1_norm(tensor : torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(tensor),dim=1)

def calculate_teacher_distillation_loss(target_master : torch.Tensor, output_decoder : torch.Tensor) -> torch.Tensor:
    tensor_diff = output_decoder - target_master
    tensor_diff_norm_squared = l1_norm(tensor_diff)**2
    loss_distillation = torch.mean(tensor_diff_norm_squared)
    return loss_distillation

# Teacher Training Pipeline
def train_teacher_network(
    softmax_temperature : float,
    device : torch.device,
    flag_load_teacher_model : bool,
    teacher_model_loading_path : str,
    learning_rate : float,
    weight_decay : float,
    dataset_train_path : str,
    dataset_validation_path : str,
    patch_size : int,
    batch_size : int,
    iterations : int,
    channels : int,
    teacher_model_saving_path : str,
    training_mode_fine_tuning : bool
) -> None:
    """
    Training Pipeline for the Teacher Network.

    Args:
        softmax_temperature : Temperature for the Softmax Activation.
        device : Device to use.
        flag_load_teacher_model : Flag to load a previously trained Teacher Network.
        teacher_model_loading_path : Path to the Teacher Network Model to load.
        learning_rate : Learning Rate for the Optimizer.
        weight_decay : Weight Decay for the Optimizer.
        dataset_train_path : Path to the Train Dataset.
        dataset_validation_path : Path to the Validation Dataset.
        patch_size : Size of the Patches to extract.
        batch_size : Batch Size for the DataLoader.
        iterations : Number of Iterations to train the Network.
        teacher_model_saving_path : Path to the Teacher Network Model to save.

    Returns:
        None
    """
    # Create Master Network
    logger.info("Creating Master Network...")
    master_network = MasterNetwork(channels, softmax_temperature)
    master_network.to(device,memory_format=torch.channels_last)
    master_network.eval()
    logger.info("Master Network Successfully Created.")
    
    # Create Teacher Network
    logger.info("Creating Teacher Network...")
    if patch_size == 65:
        teacher_network = DescriptorNet65(patch_size,patch_size,channels,output_descriptors=False,softmax_temperature=softmax_temperature)
    elif patch_size == 33:
        teacher_network = DescriptorNet33(patch_size,patch_size,channels,output_descriptors=False,softmax_temperature=softmax_temperature)
    elif patch_size == 17:
        teacher_network = DescriptorNet17(patch_size,patch_size,channels,output_descriptors=False,softmax_temperature=softmax_temperature)
    else:
        raise ValueError(f"[ERROR][TRAINING PIPELINE] Patch-Size not Valid: {patch_size}")

    if flag_load_teacher_model:
        logger.info("Loading Teacher Network Checkpoint...")
        teacher_network.load_state_dict(torch.load(teacher_model_loading_path,map_location=torch.device(device)))

    teacher_network.to(device,memory_format=torch.channels_last)
    teacher_network.train()
    logger.info("Teacher Network Successfully Created.")
    
    # Network Optimizer
    teacher_optimizer = optim.Adam(teacher_network.parameters(),lr=learning_rate,weight_decay=weight_decay)
        
    # Dataset
    dataset_train = TeacherTrainingDataset(dataset_train_path,patch_size,channels,".JPEG",training_mode_fine_tuning)
    dataset_val = TeacherTrainingDataset(dataset_validation_path,patch_size,channels,".JPEG",training_mode_fine_tuning)
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
    dataloader_val = DataLoader(dataset_val,batch_size=1,shuffle=False)
    
    # Training Loop
    ITERATIONS_PER_EPOCH_TRAIN = math.ceil(len(dataset_train) / batch_size)
    ITERATIONS_PER_EPOCH_VAL = len(dataset_val)

    #gradient_scaler = GradScaler()
    num_epochs = int(iterations // (len(dataset_train) / batch_size) + 1)
    min_total_loss = 1e10

    loss_vals = list()
    logger.info("Starting Student Training Loop...")
    for epoch in range(num_epochs):
        loss_train_epoch = 0.
        loss_val_epoch = 0.
        teacher_optimizer.zero_grad()
        
        # Training Iterations
        for i,batch in tqdm(enumerate(dataloader_train)):
            # Prepare Data
            batch = batch.to(device,memory_format=torch.channels_last)
            
            # Forward Pass Master Network
            with torch.no_grad():
                target_master = master_network(batch)
            
            # Forward Pass Teacher Network & Loss Computation               
            output_decoder_anchor = teacher_network(batch) 
            loss_train = calculate_teacher_distillation_loss(target_master,output_decoder_anchor)                
            loss_train_epoch += loss_train.item()
            
            # Back-Prop and Updates
            loss_train.backward()
            p = random.uniform(0,1)
            if p > 0.3 or (i+1) == len(dataloader_train):
                teacher_optimizer.step()
                teacher_optimizer.zero_grad()

        # Validation Iterations
        for j,batch in tqdm(enumerate(dataloader_val)):
            # Prepare Data
            batch = batch.to(device,memory_format=torch.channels_last)
            
            # Forward Pass Master Network
            with torch.no_grad():
                target_master = master_network(batch)
                output_decoder_anchor = teacher_network(batch)                
                loss_val = calculate_teacher_distillation_loss(target_master,output_decoder_anchor)                
                loss_val_epoch += loss_val.item()                
            
        # Logs
        loss_train_epoch = loss_train_epoch / ITERATIONS_PER_EPOCH_TRAIN
        loss_val_epoch = loss_val_epoch / ITERATIONS_PER_EPOCH_VAL
        loss_vals.append((loss_train_epoch,loss_val_epoch))
        logger.info(f"[INFO][TRAINING PIPELINE] EPOCH {epoch + 1}: Loss Train = {loss_train_epoch:.4f}, Loss Val = {loss_val_epoch:.4f}")
        
        # Save Best Models
        if loss_val_epoch < min_total_loss:
            torch.save(teacher_network.state_dict(), teacher_model_saving_path)
            min_total_loss = loss_val_epoch
    
    # Write Losses to File
    dt = datetime.now()
    timestamp = dt.strftime("%Y%m%d_%H%M%S")
    loss_logs_filename = f"TeacherTraining_LossLogs_{timestamp}.txt"
    filepath = "./logs/" + loss_logs_filename    
    with open(filepath,"w") as f:
        for loss in loss_vals:
            f.write(f"{loss}\n")
    
    logger.info("[INFO][TRAINING PIPELINE] Teacher Network Training Completed.")

############################################################
##### STUDENT NETWORK TRAINING PIPELINE
############################################################^

# Loss Functions
def l2_norm(tensor : torch.Tensor) -> torch.Tensor:
    squared_tensor = torch.square(tensor)
    squared_sum_tensor = torch.sum(squared_tensor,dim=1)
    return torch.sqrt(squared_sum_tensor)

def calculate_student_distillation_loss(target_master : torch.Tensor, output_decoder : torch.Tensor) -> torch.Tensor:
    tensor_diff = output_decoder - target_master
    tensor_diff_norm_squared = l2_norm(tensor_diff)**2
    loss_distillation = torch.mean(tensor_diff_norm_squared)
    return loss_distillation

# Student Training Pipeline
def train_student_network(
    img_width : int,
    img_height : int,
    channels : int,
    device : torch.device,
    teacher_model_loading_path : str,
    flag_load_student_network : bool,
    student_model_loading_path : str,
    learning_rate : float,
    weight_decay : float,
    dataset_train_path : str,
    dataset_val_path : str,
    patch_size : int,
    batch_size : int,
    iterations : int,
    optim_step_rate : int,
    student_model_saving_path : str,
    training_mode_fine_tuning : bool
) -> None:
    """
    Training Pipeline for the Student Network.

    Args:
        img_width : Width of the dataset images.
        img_height : Height of the dataset images.
        device : Device to use.
        teacher_model_loading_path : Path to the Teacher Network Model to load.
        flag_load_student_network : Flag to load a previously trained Student Network.
        student_model_loading_path : Path to the Student Network Model to load.
        learning_rate : Learning Rate for the Optimizer.
        weight_decay : Weight Decay for the Optimizer.
        dataset_train_path : Path to the Train Dataset.
        dataset_val_path : Path to the Validation Dataset.
        patch_size : Size of the Patches to extract.
        batch_size : Batch Size for the DataLoader.
        iterations : Number of Iterations to train the Network.
        student_model_saving_path : Path to the Student Network Model to save.

    Returns:
        None
    """    
    # Create Teacher Network
    logger.info("Loading Teacher Network...")
    if patch_size == 65:
        teacher_network = DescriptorNet65(img_width,img_height,channels,output_descriptors=True)
    elif patch_size == 33:
        teacher_network = DescriptorNet33(img_width,img_height,channels,output_descriptors=True)
    elif patch_size == 17:
        teacher_network = DescriptorNet17(img_width,img_height,channels,output_descriptors=True)
    else:
        raise ValueError(f"[ERROR][TRAINING PIPELINE] Patch-Size not Valid: {patch_size}")

    teacher_network.load_state_dict(torch.load(teacher_model_loading_path,map_location=torch.device(device)))
    teacher_network.to(device,memory_format=torch.channels_last)
    teacher_network.eval()
    logger.info("Teacher Network Successfully Loaded.")
    
    # Create Student Network
    logger.info("Creating Student Network...")
    if patch_size == 65:
        student_network = DescriptorNet65(img_width,img_height,channels,output_descriptors=True)
    elif patch_size == 33:
        student_network = DescriptorNet33(img_width,img_height,channels,output_descriptors=True)
    elif patch_size == 17:
        student_network = DescriptorNet17(img_width,img_height,channels,output_descriptors=True)
    else:
        raise ValueError(f"[ERROR][TRAINING PIPELINE] Patch-Size not Valid: {patch_size}")

    if flag_load_student_network:
        logger.info("Loading Student Network...")
        student_network.load_state_dict(torch.load(student_model_loading_path,map_location=torch.device(device)))
    student_network.to(device,memory_format=torch.channels_last)
    student_network.train()
    logger.info("Student Network Successfully Created.")
    
    # Network Optimizer
    student_optimizer = optim.Adam(student_network.parameters(),lr=learning_rate,weight_decay=weight_decay)
        
    # Dataset Training
    dataset_train = StudentTrainingDataset(dataset_train_path,img_width,img_height,channels,patch_size,".bmp",training_mode_fine_tuning)
    dataset_val = StudentTrainingDataset(dataset_val_path,img_width,img_height,channels,patch_size,".bmp",training_mode_fine_tuning)
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
    dataloader_val = DataLoader(dataset_val,batch_size=1)

    ITERATIONS_PER_EPOCH_TRAIN = math.ceil(len(dataset_train) / batch_size)
    ITERATIONS_PER_EPOCH_VAL = len(dataset_val)
    
    # Training Loop
    num_epochs = int(iterations // (len(dataset_train) / batch_size) + 1)
    min_total_loss = 1e10

    loss_vals = list()
    logger.info("Starting Student Training Loop...")
    for epoch in range(num_epochs):
        # Mean Loss Containers for Logging
        mean_loss_epoch_train = 0.
        mean_loss_epoch_val = 0.
        
        # Set Gradients to Zero
        student_optimizer.zero_grad()
        
        # Training Epoch
        for i,batch in tqdm(enumerate(dataloader_train)):
            # Prepare Data
            patch = batch.to(device,memory_format=torch.channels_last)
            
            # Forward Pass Teacher Network
            with torch.no_grad():
                teacher_descriptors = teacher_network(patch)
            
            # Forward Pass Student Network & Loss Computation 
            output_descriptors = student_network(patch)
            loss_train = calculate_student_distillation_loss(output_descriptors,teacher_descriptors)                
            mean_loss_epoch_train += loss_train.item()
            
            # Back-Prop and Updates
            loss_train.backward()
            if (i+1) % optim_step_rate == 0 or (i+1) == len(dataloader_train):
                student_optimizer.step() 
                student_optimizer.zero_grad()                

        # Validation Epoch
        for i,batch in tqdm(enumerate(dataloader_val)):
            patch = batch.to(device)
            with torch.no_grad():
                teacher_descriptors = teacher_network(patch)
                output_descriptors = student_network(patch)
                loss_val = calculate_student_distillation_loss(output_descriptors,teacher_descriptors)
                mean_loss_epoch_val += loss_val.item()

        # Logs
        mean_loss_epoch_train = mean_loss_epoch_train / ITERATIONS_PER_EPOCH_TRAIN
        mean_loss_epoch_val = mean_loss_epoch_val / ITERATIONS_PER_EPOCH_VAL
        loss_vals.append((mean_loss_epoch_train,mean_loss_epoch_val))
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}: Loss Train = {mean_loss_epoch_train:.4f}, Loss Val = {mean_loss_epoch_val:.4f}")
        
        # Save Best Models
        if mean_loss_epoch_val < min_total_loss:
            torch.save(student_network.state_dict(), student_model_saving_path)
            min_total_loss = mean_loss_epoch_val

    # Write Losses to File
    dt = datetime.now()
    timestamp = dt.strftime("%Y%m%d_%H%M%S")
    loss_logs_filename = f"StudentTraining_LossLogs_{timestamp}.txt"
    filepath = "./logs/" + loss_logs_filename    
    with open(filepath,"w") as f:
        for loss in loss_vals:
            f.write(f"{loss}\n")

    logger.info("Student Training Completed!")

############################################################
##### DEVELOPMENT TESTING
############################################################

def test() -> None:
    x_decoder = torch.rand(1,3,65,65)
    x_FDFE = torch.rand(1,3,256,256)

    net_decoder = DescriptorNet65(256,256)
    y_decoder = net_decoder(x_decoder)

    net_FDFE = DescriptorNet65(256,256,True)
    y_FDFE = net_FDFE(x_FDFE)

if __name__ == "__main__":
    test()
