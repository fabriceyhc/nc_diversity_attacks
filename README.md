# diversity_attacks

### About
Corresponding code to the paper "Is Neuron Coverage a Meaningful Measure for Testing Deep Neural Networks" by Fabrice *et al.*.

### Pre-requisites
The following steps should be sufficient to get these attacks up and running on most systems.

```
# pytorch
# torchvision
# numpy
# pandas
# matplotlib
# tqdm
# scipy

pip install -r requirements.txt
```

### Execution

To run the MNIST/CIFAR evaluation scripts:
```
python _CW_div_mnist.py
python _CW_div_cifar10.py
python _PGD_div_mnist.py
python _PGD_div_cifar10.py
```

### Test Suites
Test suites created for this evaluation are located in `assets`. Note that the CIFAR10 suites exceed the 100MB limitation on Github and are thus available [here](https://drive.google.com/open?id=1kQ8GdSDWiI6NA7fNd1C_CA2_gz5vu6Yz).
