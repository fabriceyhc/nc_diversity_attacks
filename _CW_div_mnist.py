#!/usr/bin/env python
# coding: utf-8

# # Investigating CW Attack Variants Using Diversity Promoting Regularization

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# import traceback

import datetime
import glob
import os

import pickle

import pandas as pd

from models import *
from cw_div import *
from neuron_coverage import *
from inception_score import *
from fid_score import *

# check if CUDA is available
device = torch.device("cpu")
if torch.cuda.is_available():
    print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda")
else:
    print('CUDA is not available.  Training on CPU ...')

# # load the results from file
# with open('assets/results.pickle', 'rb') as handle:
#     results = pickle.load(handle)

n_epochs = 10
batch_size_train = 64
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5

random_seed = 1
torch.manual_seed(random_seed)

#  torchvision.transforms.Normalize(
#    (0.1307,), (0.3081,))

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
    batch_size=batch_size_train, shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=False, download=True,
                         transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                         ])),
    batch_size=batch_size_test, shuffle=False, pin_memory=True)

# # Train or Load Pretrained Model if available

retrain = False
track_low_high = False

model = ConvNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# check to see if we can just load a previous model
latest_model = None
m_type = model.__class__.__name__
prev_models = glob.glob('pretrained_models/mnist/*'+ m_type +'*.pth')
if prev_models:
    latest_model = max(prev_models, key=os.path.getctime)

if (retrain is False 
    and latest_model is not None 
    and m_type in latest_model):  
    print('loading model', latest_model)
    model.load_state_dict(torch.load(latest_model))  
else:
    if track_low_high:
        model.init_dict(model.lowhigh_dict, inputs, 'relu', {'low': 0, 'high': 0})
        try:
            for epoch in range(1, n_epochs + 1):
                model.hook_lowhigh_dict('relu')
                train(model, device, train_loader, optimizer, epoch)
                model.remove_hooks()
                test(model, device, test_loader)    
        finally:
            model.remove_hooks()   
    else:
        for epoch in range(1, n_epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            acc = test(model, device, test_loader)  
    torch.save(model.state_dict(), 'pretrained_models/mnist/model_' + m_type + '_' + str(datetime.datetime.now()).replace(':','.') + '_' + str(acc) + '.pth')

# ### Layer Dict
# Required for extracting outputs for diversity regularization. 

layer_dict = get_model_modules(model)

# # Attack Time
def main():

    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    results = []
    save_file_path = "assets/results_mnist2019.08.17.pickle"

    # attack params
    search_steps=4
    targeted=False
    c_range=(1e-3, 1e10)
    max_steps=1000
    abort_early=True
    optimizer_lr=5e-4
    init_rand=False
    log_frequency = 100

    mean = (0.1307,) # the mean used in inputs normalization
    std = (0.3081,) # the standard deviation used in inputs normalization
    box = (min((0 - m) / s for m, s in zip(mean, std)),
           max((1 - m) / s for m, s in zip(mean, std)))

    n=2
    attack_versions = [cw_div4_attack] # [cw_div1_attack, cw_div2_attack, cw_div3_attack, cw_div4_attack]
    target_layers = list(layer_dict)[1::n]
    reg_weights = [0, 1, 10, 100, 1000, 10000]
    confidences = [0]

    # neuron coverage params
    nc_threshold = 0. # all activations are scaled to (0,1) after relu

    # inception score (is) params
    is_cuda = True
    is_batch_size = 10
    is_resize = True
    is_splits = 10

    # fréchet inception distance score (fid) params
    real_path = "C:/temp_imgs/mnist/real/"
    fake_path = "C:/temp_imgs/mnist/fake/"
    fid_batch_size = 64
    fid_cuda = True

    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    for attack in attack_versions:
        for layer_idx in target_layers:
            module = layer_dict[layer_idx]
            for rw in reg_weights:
                for c in confidences:
                    
                    timestamp = str(datetime.datetime.now()).replace(':','.')
                    
                    print('timestamp', timestamp, 
                          'attack', attack.__name__, 
                          'layer: ', layer_idx, 
                          'regularization_weight: ', rw, 
                          'confidence: ', c)
                    
                    # adversarial attack 
                    adversaries = attack(model, module, rw, inputs, targets, device, targeted,
                                         c, c_range, search_steps, max_steps, abort_early, box,
                                         optimizer_lr, init_rand, log_frequency)
                   
                    # evaluate adversary effectiveness
                    pert_acc, orig_acc = eval_performance(model, inputs, adversaries, targets)
                    sample_1D_images(model, inputs, adversaries, targets)
                    
                    pert_acc = pert_acc.item() / 100.
                    orig_acc = orig_acc.item() / 100.
                    
                    # neuron coverage
                    covered_neurons, total_neurons, neuron_coverage_000 = eval_nc(model, adversaries, 0.00)
                    print('neuron_coverage_000:', neuron_coverage_000)
                    covered_neurons, total_neurons, neuron_coverage_020 = eval_nc(model, adversaries, 0.20)
                    print('neuron_coverage_020:', neuron_coverage_020)
                    covered_neurons, total_neurons, neuron_coverage_050 = eval_nc(model, adversaries, 0.50)
                    print('neuron_coverage_050:', neuron_coverage_050)
                    covered_neurons, total_neurons, neuron_coverage_075 = eval_nc(model, adversaries, 0.75)
                    print('neuron_coverage_075:', neuron_coverage_075)
                    
                    # inception score
                    preprocessed_advs = preprocess_1D_imgs(adversaries)
                    mean_is, std_is = inception_score(preprocessed_advs, is_cuda, is_batch_size, is_resize, is_splits)
                    print('inception_score:', mean_is)
                    
                    # fid score 
                    paths = [real_path, fake_path]
                    
                    # dimensionality = 64
                    target_num = 64
                    generate_imgs(inputs, real_path, target_num)
                    generate_imgs(adversaries, fake_path, target_num)
                    fid_score_64 = calculate_fid_given_paths(paths, fid_batch_size, fid_cuda, dims=64)
                    print('fid_score_64:', fid_score_64)
                    
                    # dimensionality = 2048
                    target_num = 2048
                    generate_imgs(inputs, real_path, target_num)
                    generate_imgs(adversaries, fake_path, target_num)
                    fid_score_2048 = calculate_fid_given_paths(paths, fid_batch_size, fid_cuda, dims=2048)
                    print('fid_score_2048:', fid_score_2048)
                    
                    out = {'timestamp': timestamp, 
                           'attack': attack.__name__, 
                           'layer': layer_idx, 
                           'regularization_weight': rw, 
                           'confidence': c, 
                           'adversaries': adversaries,
                           'pert_acc':pert_acc, 
                           'orig_acc': orig_acc,
                           'neuron_coverage_000': neuron_coverage_000,
                           'neuron_coverage_020': neuron_coverage_020,
                           'neuron_coverage_050': neuron_coverage_050,
                           'neuron_coverage_075': neuron_coverage_075,
                           'inception_score': mean_is,
                           'fid_score_64': fid_score_64,
                           'fid_score_2048': fid_score_2048}
                    
                    results.append(out)
                    
                    # save incremental outputs
                    pickle.dump(results, open(save_file_path, "wb"))

if __name__ == '__main__':
    try:
        main()
    except Exception as e: 
        print(e)
