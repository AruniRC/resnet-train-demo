## PyTorch ResNet demo

Demo to train a ResNet model on a custom dataset. The code is written such that it can be easily used as a base for specific projects.

### Setup

* (Install Anaconda)[https://conda.io/docs/user-guide/install/linux.html] if not already installed in the system.
* Create an Anaconda environment: `conda create -n resnet-demo python=2.7` and activate it: `source activate resnet-demo`.
* Install PyTorch and TorchVision inside the Anaconda environment. First add a channel to conda: `conda config --add channels soumith`. Then install: `conda install pytorch torchvision cuda80 -c soumith`.
* Install the dependencies using conda: `conda install scipy Pillow tqdm scikit-learn scikit-image numpy matplotlib ipython pyyaml`.


### Usage

Experimental settings such as learning rates are defined in `config.py`, each setting being a numbered key in a Python dict.

The demo code is in the file `train_resnet_demo.py`. The command for running it on GPU:0 and using configuration:1 is `python train_resnet_demo.py -g 0 -c 1`. The code has detailed comments and serves as a walkthrough for training deep networks on custom datasets in PyTorch.

Each time the training script is run, a new output folder with a timestamp is created by default under `./logs` -- `logs/MODEL-CFG-TIMESTAMP/`. Under an experiment's log folder the settings for each experiment can be viewed in `config.yml`; metrics such as the training and validation losses are updated in `log.csv`. 


