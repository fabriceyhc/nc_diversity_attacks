# nc_diversity_attacks

### About
Corresponding code to the paper "Is Neuron Coverage a Meaningful Measure for Testing Deep Neural Networks" by Fabrice Harel-Canada *et al.*.

### Pre-requisites
The following steps should be sufficient to get these attacks up and running on most systems running Python 3.7.3+.

```
numpy==1.16.2
pandas==0.24.2
torchvision==0.6.0
torch==1.5.0
tqdm==4.31.1
matplotlib==3.0.3
scipy==1.2.1
```
Note: these are the most recent versions of each library used, lower versions may be acceptable as well. It is also *highly* recommended that you use GPUs to execute the evaluation scripts. 

```
pip install -r requirements.txt
```

### Data
MNIST and CIFAR10 data are downloaded automatically when running an evaluation script. The Driving data comes from the Udacity [self-driving-car](https://github.com/udacity/self-driving-car) challenge and is included in the `data` folder. 

### Attacks
There are several versions of the CW Atttack that we experimented with, but ultimately decided upon `cw_div4_attack` (Diversity v4). We also use the PGD whitebox attack (PGD v1).

| Version | Loss Function | Scaling Constant | Regularizer | Adversary Selection |
| - | - | - | - | - |
|  Baseline CW | CW |  True | L2/L-inf |  L2/L-inf |
|  Diversity v1 | CW |  True | L2/L-inf , Batch Divergence | Instance Divergence |
|  Diversity v2 | CW |  False | L2/L-inf , Batch Divergence | Instance Divergence |
|  Diversity v3 | Cross Entropy |  False | L2/L-inf, Batch Divergence | Instance Divergence |
|  Diversity v4 | CW |  True | L2/L-inf, Batch Divergence | L2/L-inf |
|  PGD v1       | PGD | NA | L-inf | NA |

### Execution
To run the evaluation scripts:
```
# CW
python _CW_div_mnist.py
python _CW_div_cifar10.py
python _CW_div_driving.py

# PGD
python _PGD_div_mnist.py
python _PGD_div_cifar10.py
python _PGD_div_driving.py
```

### Jupyter Notebook
The results are agregated and visualized in a jupyter notebook, which can be viewed directly in GitHub or perused locally:
```
# install
pip install jupyter

# start notebook in working directory
jupyter notebook
```