## PyTorch ResNet demo

Demo to train a ResNet model on a custom dataset. The code is written such that it can be easily used as a base for specific projects.

### Setup

Install PyTorch and TorchVision in an Anaconda environment then install the dependencies (using conda) from the `environment.yml` file.

For 1080Ti GPUs, use: `conda install pytorch torchvision cuda80 -c soumith`.
This addresses the massive slowdown in executing `model.cuda()`.

### Usage

Experiment settings such as learning rates are defined in `config.py`, each setting being a key in a Python dict.

The training metrics are updated in a log file saved under `logs/MODEL-folder/log.csv`.
