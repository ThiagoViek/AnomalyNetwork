"""
MAIN FILE

Script calls execution of given commands.
"""

__author__ = "Thiago Deeke Viek"
__version__ = "1.0.1"
__maintainer__ = "Thiago Deeke Viek"
__status__ = "Development"

############################################################
##### IMPORTS
############################################################

# Utils Imports
import pdb
import yaml
import argparse
import logging

# Custom Modules
from train import train_teacher_network, train_student_network
from test import generate_heatmaps

# Torch Imports
import torch

############################################################
##### EVENTS LOGGER
############################################################

# Main Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", "%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Session Specs Logger
logger_session_specs = logging.getLogger("session_specs")
logger_session_specs.setLevel(logging.DEBUG)

console_handler_specs = logging.StreamHandler()
console_handler_specs.setLevel(logging.DEBUG)

formatter_specs = logging.Formatter("%(message)s")
console_handler_specs.setFormatter(formatter_specs)
logger_session_specs.addHandler(console_handler_specs)

############################################################
##### MAIN EXECUTION
############################################################

# Clear Console
def clear_console() -> None:
    """
    Clears the console.
    """
    import os
    os.system('cls')

# Main Execution
def main() -> None:
    """
    Codebase Main Funtion.
    """
    # Clear Terminal Console
    clear_console()

    # Parse User Arguments
    parser = argparse.ArgumentParser(description="[DEEP STOCHASTIC ANOMALY DETECTION] Execution of given commands")
    parser.add_argument("--configurations", type=str, help="filename of the config yaml file")
    args = parser.parse_args()

    # Retrieve Session Specs from Config File
    config_filepath = "config/" + args.configurations
    with open(config_filepath,'r') as f:
        config = yaml.safe_load(f)

    # Log Session Specs
    logger_session_specs.info("---------- DEEP STOCHASTIC ANOMALY DETECTION ----------")
    logger_session_specs.info(f"{__copyright__}, All rights reserved\n")
    logger_session_specs.info(f"Current Version: {__version__}")
    logger_session_specs.info(f"Current Status: {__status__}\n")

    logger_session_specs.debug(f"Session Configuration Specs:")
    for key, value in config.items():
        logger_session_specs.debug(f"\t{key} : {value}")
    
    # Fork Execution Pipeline
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger_session_specs.debug(f"\tdevice : {DEVICE}\n")
    
    # User Specs Confirmation
    logger_session_specs.info("Are the hyperparameters correct? (y/n)")
    user_input = input()

    if user_input == "y":
        # Training Session
        if config["session"] == "train":
            # Train Teacher Network
            if config["learning_target"] == "teacher":
                logger.info("\nStarting Teacher Network Training Session...")
                train_teacher_network(
                    softmax_temperature=config["softmax_temperature"],
                    device=DEVICE,
                    flag_load_teacher_model=config["flag_load_teacher_model"],
                    teacher_model_loading_path=config["teacher_model_loading_path"],
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    dataset_train_path=config["dataset_train_path"],
                    dataset_validation_path=config["dataset_val_path"],
                    patch_size=config["patch_size"],
                    batch_size=config["batch_size"],
                    iterations=config["iterations"],
                    channels=config["channels"],
                    teacher_model_saving_path=config["teacher_model_saving_path"],
                    training_mode_fine_tuning=config["learning_fineTuning"],
                )
            
            # Train Student Network
            elif config["learning_target"] == "student":
                logger.info("Starting Student Network Training Session...")
                train_student_network(
                    img_width=config["img_width"],
                    img_height=config["img_height"],
                    channels=config["channels"],
                    device=DEVICE,
                    teacher_model_loading_path=config["teacher_network_loading_path"],
                    flag_load_student_network=config["flag_load_student_network"],
                    student_model_loading_path=config["student_network_loading_path"],
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    dataset_train_path=config["dataset_train_path"],
                    dataset_val_path=config["dataset_val_path"],
                    patch_size=config["patch_size"],
                    batch_size=config["batch_size"],
                    iterations=config["iterations"],
                    optim_step_rate=config["optimizer_step_rate"],
                    student_model_saving_path=config["student_network_saving_path"],
                    training_mode_fine_tuning=config["training_mode_fileTuning"],
                )
            
            # Unkown Command
            else:
                raise ValueError("Learning Target not recognized")
        
        # Testing Session
        elif config["session"] == "test":
            logger.info("Starting Testing Session...")
            generate_heatmaps(
                dataset_good_path=config["dataset_good_path"],
                dataset_defect_path=config["dataset_defect_path"],
                teacher_model_path=config["teacher_network_loading_path"],
                student_model_path=config["student_network_loading_path"],
                img_width=config["img_width"],
                img_height=config["img_height"],
                channels=config["channels"],
                patch_size=config["patch_size"],
                device=DEVICE
            )
            
    elif user_input == "n":
        logger.info("Session Canceled by User.")
    else:
        raise ValueError("User Input not recognized")

if __name__ == "__main__":
    main()
