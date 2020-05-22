# nc_diversity_attacks

### About
Corresponding code to the paper "Is Neuron Coverage a Meaningful Measure for Testing Deep Neural Networks" by Fabrice Harel-Canada *et al.*.

### Pre-requisites
The following steps should be sufficient to get these attacks up and running on most systems.
```
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

#or 

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

```
# numpy
# pandas
# matplotlib
# tqdm
# scipy
# typing

pip install -r requirements.txt
```

### Attacks
There are several versions of the CW Atttack that we experimented with, but ultimately decided upon `cw_div4_attack` (Diversity v4). We also use the PGD whitebox attack (PGD v1).

| Version | Loss Function | Scaling Constant | Regularizer | Adversary Selection |
| - | - | - | - | - |
|  Baseline CW | CW |  True | L_p |  L_p |
|  Diversity v1 | CW |  True | L_p, Batch Divergence | Instance Divergence |
|  Diversity v2 | CW |  False | L_p, Batch Divergence | Instance Divergence |
|  Diversity v3 | Cross Entropy |  False | L_p, Batch Divergence | Instance Divergence |
|  Diversity v4 | CW |  True | L_p, Batch Divergence | L_p |
|  PGD v1       | PGD | NA | L-inf | NA |

### Execution
To run the MNIST/CIFAR/Driving evaluation scripts:
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