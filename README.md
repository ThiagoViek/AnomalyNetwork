# DEVELOPMENT REPOSITORY FOR DEEP STOCHASTIC ANOMALY NETWORK

**Author:** Thiago Deeke Viek  
**Date:** 19.09.2024 

This repository contains the files and resources for the development of the Deep Stochastic Anomaly Network for Product Anomaly Detection. The rationale is to train the network to "overfit" the normal data and behave unexpectedly when encountering outliers, so that heuristics for defect segmentation can be applied.

## Repository Structure

- `config/`: contains the experiment files with its specifications
- `data/`: contains the datasets used for development
- `logs/`: contains the logging metadata from each experiment run
- `model-checkpoints/`: contains the state-dictionaries of the models run in the experiments
- `models/`: contains the python scripts implementing the Neural Networks

## Running New Experiments

The file to run training or testing code is the main file (`main.py`). The configuration file of the respective run specifies whether the run is a train session or a test session. Input all this information as arguments in the command line when executing:

```
main.py --session train --dataset datasetX --configurations configXXX
```
